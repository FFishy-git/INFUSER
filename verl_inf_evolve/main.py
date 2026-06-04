"""
Entry point for V3 Self-Evolution training.

Follows verl's ``main_ppo.py`` pattern: Hydra config → Ray init → TaskRunner.
Key difference: two worker groups (generator + solver), no critic worker.

Usage::

    python -m verl_inf_evolve.main

    # Override config values
    python -m verl_inf_evolve.main training.max_ans_loop=5

Reference: verl/trainer/main_ppo.py
"""

import logging
import os
import socket
from glob import glob

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl_inf_evolve.utils.config_resolvers import register_config_template_resolvers
from verl_inf_evolve.utils.env_utils import load_startup_env
from verl_inf_evolve.storage.hf_transfer import (
    configure_hf_transfer_limits,
    get_hf_transfer_settings,
)
from verl_inf_evolve.utils.seeding import seed_all
from verl_inf_evolve.utils.startup_cleanup import maybe_clear_local_dir_on_start

load_startup_env()


def _prefetch_model_snapshot(
    model_path: str,
    *,
    snapshot_max_workers: int | None = None,
) -> None:
    """Best-effort prefetch of model files before enabling HF offline mode."""
    if not isinstance(model_path, str):
        return

    def _has_weight_files(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        required_markers = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        if any(os.path.exists(os.path.join(path, marker)) for marker in required_markers):
            return True
        if glob(os.path.join(path, "model-*.safetensors")):
            return True
        if glob(os.path.join(path, "pytorch_model-*.bin")):
            return True
        return False

    def _has_local_model_files() -> bool:
        if os.path.isdir(model_path):
            return _has_weight_files(model_path)

        cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        snapshots_dir = os.path.join(
            cache_root,
            "hub",
            f"models--{model_path.replace('/', '--')}",
            "snapshots",
        )
        if not os.path.isdir(snapshots_dir):
            return False
        for snapshot_id in sorted(os.listdir(snapshots_dir), reverse=True):
            snapshot_dir = os.path.join(snapshots_dir, snapshot_id)
            if _has_weight_files(snapshot_dir):
                return True
        return False

    if _has_local_model_files():
        logging.info("Skipping prefetch for %s; local model files already exist", model_path)
        return

    if os.path.isdir(model_path):
        logging.warning(
            "Model path %s is local but contains no detected weight files; skipping HF prefetch",
            model_path,
        )
        return

    try:
        from huggingface_hub import snapshot_download

        snapshot_kwargs = {}
        if snapshot_max_workers is not None:
            snapshot_kwargs["max_workers"] = snapshot_max_workers
        snapshot_dir = snapshot_download(
            repo_id=model_path,
            allow_patterns=[
                "*.safetensors",
                "*.safetensors.index.json",
                "pytorch_model*.bin",
                "*.pt",
                "*.json",
                "*.py",
                "*.txt",
            ],
            resume_download=True,
            **snapshot_kwargs,
        )
        logging.info("Prefetched model snapshot for %s at %s", model_path, snapshot_dir)
    except Exception as exc:
        logging.warning("Model prefetch failed for %s: %s", model_path, exc)


@hydra.main(config_path="config", config_name="self_evolution", version_base=None)
def main(config):
    run_self_evolution(config)


def run_self_evolution(config) -> None:
    """Initialize Ray cluster and launch the self-evolution training process.

    Mirrors ``run_ppo()`` in verl/trainer/main_ppo.py but sets up two
    actor-rollout worker groups (generator and solver) instead of one.

    Args:
        config: OmegaConf config with keys: generator, solver, training,
                influence, spice, curriculum, wandb, trainer, ray_kwargs.
    """
    # Seed every RNG we control *before* ray.init so PYTHONHASHSEED /
    # CUBLAS_WORKSPACE_CONFIG land in the environment that Ray will
    # propagate to its workers. Workers additionally re-seed themselves
    # with their rank offset on startup.
    seed_all(int(config.training.get("seed", 42)))

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        # Ensure deterministic env vars are forwarded to every Ray worker
        # — some clusters do not inherit the driver's full environment.
        forwarded_env = dict(runtime_env.get("env_vars", {}) or {})
        for key in ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG"):
            if key in os.environ:
                forwarded_env.setdefault(key, os.environ[key])
        if forwarded_env:
            runtime_env = OmegaConf.merge(runtime_env, {"env_vars": forwarded_env})
        ray_init_kwargs = OmegaConf.create(
            {**ray_init_kwargs, "runtime_env": runtime_env}
        )
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    register_config_template_resolvers()

    # Resolve all interpolations (including Hydra's ${now:...}) while
    # Hydra's resolvers are still registered.  The Ray actor won't have
    # Hydra's context, so unresolved interpolations would crash there.
    OmegaConf.resolve(config)

    task_runner_class = ray.remote(num_cpus=1)(TaskRunner)
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """Ray remote class that sets up workers and runs training.

    Mirrors verl's ``TaskRunner`` (main_ppo.py:108) but registers two
    separate ActorRolloutRefWorker groups for generator and solver.
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_generator_worker(self, config):
        """Register the generator ActorRolloutRefWorker.

        Uses the ``config.generator`` section (same schema as verl's
        ``actor_rollout_ref``).
        """
        from verl.single_controller.ray import RayWorkerGroup
        from verl_inf_evolve.workers.self_evolution_worker import (
            SelfEvolutionActorRolloutRefWorker,
        )

        # Extends verl's ActorRolloutRefWorker (verl/workers/fsdp_workers.py:138).
        # Config schema: config.generator.{model, actor, rollout, ref}
        # Default values: verl_inf_evolve/config/self_evolution.yaml → generator section
        # verl reference configs: verl/trainer/config/{model/hf_model, actor/dp_actor,
        #   rollout/rollout, ref/dp_ref}.yaml
        self.role_worker_mapping["generator"] = ray.remote(
            SelfEvolutionActorRolloutRefWorker
        )
        if config.trainer.get("separate_pools", False):
            self.mapping["generator"] = config.trainer.get(
                "generator_pool", "generator_pool"
            )
        else:
            self.mapping["generator"] = "global_pool"
        return SelfEvolutionActorRolloutRefWorker, RayWorkerGroup

    def add_solver_worker(self, config):
        """Register the solver ActorRolloutRefWorker.

        Uses the ``config.solver`` section (same schema as verl's
        ``actor_rollout_ref``).
        """
        from verl.single_controller.ray import RayWorkerGroup
        from verl_inf_evolve.workers.self_evolution_worker import (
            SelfEvolutionActorRolloutRefWorker,
        )

        # Same base class as generator, but with separate config section.
        # Config schema: config.solver.{model, actor, rollout, ref}
        # Default values: verl_inf_evolve/config/self_evolution.yaml → solver section
        # verl reference configs: verl/trainer/config/{model/hf_model, actor/dp_actor,
        #   rollout/rollout, ref/dp_ref}.yaml
        self.role_worker_mapping["solver"] = ray.remote(
            SelfEvolutionActorRolloutRefWorker
        )
        if config.trainer.get("separate_pools", False):
            self.mapping["solver"] = config.trainer.get(
                "solver_pool", "solver_pool"
            )
        else:
            self.mapping["solver"] = "global_pool"
        return SelfEvolutionActorRolloutRefWorker, RayWorkerGroup

    def init_resource_pool_mgr(self, config):
        """Initialize the resource pool manager for GPU allocation.

        Supports shared (single pool) or separate (generator_pool + solver_pool)
        GPU configurations.

        Reference: main_ppo.py:199
        """
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        if config.trainer.get("separate_pools", False):
            # Separate pools: generator and solver each get dedicated GPUs.
            gen_gpus = config.trainer.get("generator_n_gpus_per_node", config.trainer.n_gpus_per_node)
            solver_gpus = config.trainer.get("solver_n_gpus_per_node", config.trainer.n_gpus_per_node)
            resource_pool_spec = {
                "generator_pool": [gen_gpus] * config.trainer.nnodes,
                "solver_pool": [solver_gpus] * config.trainer.nnodes,
            }
        else:
            # Shared pool: both roles colocated on the same GPUs.
            resource_pool_spec = {
                "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )
        return resource_pool_manager

    def run(self, config):
        """Execute the self-evolution training workflow.

        Steps:
          1. Register generator and solver workers
          2. Initialize resource pools
          3. Create tokenizers for both models
          4. Load datasets (dev + documents)
          5. Instantiate SelfEvolutionTrainer
          6. Call trainer.init_workers() and trainer.fit()

        Reference: main_ppo.py:250
        """
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        # Configure logging inside the Ray actor (Hydra only sets up the
        # main process).  Reads LOGLEVEL env var, defaulting to INFO.
        # verl/__init__.py calls logging.basicConfig(level=WARNING) on import,
        # which adds a StreamHandler to root.  A second basicConfig() is a
        # no-op once handlers exist, so we set the root level directly.
        log_level = os.environ.get("LOGLEVEL", "INFO").upper()
        logging.root.setLevel(getattr(logging, log_level, logging.INFO))

        # Clear stale local artifacts before creating the persistent training
        # log inside default_local_dir and before any worker startup begins.
        maybe_clear_local_dir_on_start(config)

        # Add persistent file logging so logs survive pod crashes.
        # Writes to {default_local_dir}/training.log (append mode).
        log_dir = config.training.get("default_local_dir", ".")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training.log")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.root.addHandler(file_handler)
        logging.info("Persistent log file: %s", os.path.abspath(log_file))

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        # Re-seed inside the Ray actor process (this is a fresh Python
        # interpreter distinct from the driver). Driver already set
        # PYTHONHASHSEED in env so Python picked it up at actor startup.
        seed_all(int(config.training.get("seed", 42)))
        from verl_inf_evolve.storage.remote_backend import redact_config_secrets

        pprint(redact_config_secrets(OmegaConf.to_container(config, resolve=True)))

        # Disable struct mode so verl can access config fields with defaults
        # without requiring every field to be listed in our YAML.
        OmegaConf.set_struct(config, False)

        from verl_inf_evolve.dry_run.mock_runtime import (
            is_mock_cpu_dry_run_enabled,
            run_mock_cpu_dry_run,
        )

        if is_mock_cpu_dry_run_enabled(config):
            logging.info(
                "Dry-run mode enabled with backend=%s; skipping worker registration, "
                "resource pools, tokenizer loading, and rollout startup",
                config.dry_run.get("backend", "mock_cpu"),
            )
            run_mock_cpu_dry_run(config)
            return

        # Register worker groups
        _, ray_worker_group_cls = self.add_generator_worker(config)
        self.add_solver_worker(config)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        # Load generator tokenizer
        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)

        gen_local_path = copy_to_local(
            config.generator.model.path,
            use_shm=config.generator.model.get("use_shm", False),
        )
        gen_tokenizer = hf_tokenizer(gen_local_path, trust_remote_code=trust_remote_code)
        gen_chat_template = config.generator.model.get("custom_chat_template", None)
        if gen_chat_template is not None:
            gen_tokenizer.chat_template = gen_chat_template

        solver_local_path = copy_to_local(
            config.solver.model.path,
            use_shm=config.solver.model.get("use_shm", False),
        )
        solver_tokenizer = hf_tokenizer(solver_local_path, trust_remote_code=trust_remote_code)
        solver_chat_template = config.solver.model.get("custom_chat_template", None)
        if solver_chat_template is not None:
            solver_tokenizer.chat_template = solver_chat_template

        # Force offline mode after tokenizer initialization. Prefetch model
        # snapshots first so from_pretrained() can resolve shards from cache.
        # Skip offline mode when remote_sync_path uses hf:// — the HF backend
        # needs network access to upload/download artifacts during training.
        hf_transfer_settings = get_hf_transfer_settings(config.get("remote", {}))
        configure_hf_transfer_limits(
            upload_limit_mbps=hf_transfer_settings["upload_limit_mbps"],
            download_limit_mbps=hf_transfer_settings["download_limit_mbps"],
        )
        for model_path in dict.fromkeys(
            [config.generator.model.path, config.solver.model.path]
        ):
            _prefetch_model_snapshot(
                model_path,
                snapshot_max_workers=hf_transfer_settings["snapshot_max_workers"],
            )
        remote_sync_path = config.training.get("remote_sync_path", None) or ""
        if remote_sync_path.startswith("hf://"):
            logging.info(
                "Skipping HF_HUB_OFFLINE=1 because remote_sync_path uses hf:// scheme"
            )
        else:
            os.environ["HF_HUB_OFFLINE"] = "1"
            logging.info("Set HF_HUB_OFFLINE=1 after tokenizer/model prefetch")

        # Create trainer
        from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer

        trainer = SelfEvolutionTrainer(
            config=config,
            gen_tokenizer=gen_tokenizer,
            solver_tokenizer=solver_tokenizer,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
        )

        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
