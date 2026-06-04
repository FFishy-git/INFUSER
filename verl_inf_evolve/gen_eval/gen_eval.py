"""Entry point for Generator Evaluation with Fixed Solver.

Follows the same ``main.py → TaskRunner → evaluator class`` pattern as
``verl_inf_evolve/main.py``, but creates a :class:`GeneratorEvaluator`
instead of a :class:`SelfEvolutionTrainer`.

Usage::

    python -m verl_inf_evolve.gen_eval.gen_eval gen_eval.remote_sync_path=s3://...
    python -m verl_inf_evolve.gen_eval.gen_eval gen_eval_experiment=<name>
    python -m verl_inf_evolve.gen_eval.gen_eval gen_eval.ans_loop_indices=[0,5,10]
"""

from __future__ import annotations

import logging
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl_inf_evolve.utils.config_resolvers import register_config_template_resolvers
from verl_inf_evolve.utils.env_utils import load_startup_env

load_startup_env()


@hydra.main(config_path="../config", config_name="gen_eval", version_base=None)
def main(config):  # type: ignore[no-untyped-def]
    run_gen_eval(config)


def run_gen_eval(config) -> None:  # type: ignore[no-untyped-def]
    """Initialize Ray cluster and launch the generator evaluation process.

    In ``gpt_replay`` mode, Ray workers and GPUs are not needed — the
    evaluator runs directly in the main process using API calls.
    """
    register_config_template_resolvers()

    # Resolve all interpolations while Hydra's resolvers are still registered.
    OmegaConf.resolve(config)

    eval_mode = config.gen_eval.get("mode", "regenerate")

    if eval_mode == "gpt_replay":
        _run_gpt_replay(config)
        return

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create(
            {**ray_init_kwargs, "runtime_env": runtime_env}
        )
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner_class = ray.remote(num_cpus=1)(TaskRunner)
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


def _run_gpt_replay(config) -> None:
    """Run gpt_replay mode directly without Ray workers.

    No GPU workers are needed — questions are answered via OpenAI/Gemini API.
    Only the generator tokenizer is loaded (lightweight, needed for question
    text decoding from gen_output.pt).
    """
    from pprint import pprint

    from verl.utils.fs import copy_to_local

    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.root.setLevel(getattr(logging, log_level, logging.INFO))

    print(f"gpt_replay hostname: {socket.gethostname()}, PID: {os.getpid()}")
    from verl_inf_evolve.storage.remote_backend import redact_config_secrets

    pprint(redact_config_secrets(OmegaConf.to_container(config, resolve=True)))

    # Disable struct mode so verl can access config fields with defaults
    OmegaConf.set_struct(config, False)

    # Load only the generator tokenizer (needed for derive_gen_questions).
    # Try copy_to_local first (works for S3/local paths), then fall back to
    # loading directly from the HF model ID (for gpt_replay on machines
    # without pre-downloaded model weights).
    from verl.utils import hf_tokenizer

    trust_remote_code = config.data.get("trust_remote_code", False)
    model_path = str(config.generator.model.path)
    gen_local_path = copy_to_local(
        model_path,
        use_shm=config.generator.model.get("use_shm", False),
    )
    try:
        gen_tokenizer = hf_tokenizer(gen_local_path, trust_remote_code=trust_remote_code)
    except (TypeError, OSError):
        # copy_to_local returned a path that doesn't exist locally —
        # load tokenizer directly from HuggingFace model ID.
        logging.getLogger(__name__).info(
            "Falling back to loading tokenizer directly from %s", model_path,
        )
        gen_tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)

    # Create evaluator — no Ray workers, no resource pool, no solver tokenizer
    from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator

    evaluator = GeneratorEvaluator(
        config=config,
        gen_tokenizer=gen_tokenizer,
        solver_tokenizer=None,  # Not needed in gpt_replay mode
        role_worker_mapping={},
        resource_pool_manager=None,  # type: ignore[arg-type]
        ray_worker_group_cls=None,  # type: ignore[arg-type]
    )

    evaluator.init_workers()
    evaluator.evaluate()


class TaskRunner:
    """Ray remote class that sets up workers and runs evaluation.

    Mirrors the ``TaskRunner`` in ``verl_inf_evolve/main.py`` but creates
    a :class:`GeneratorEvaluator` instead of a :class:`SelfEvolutionTrainer`.
    """

    def __init__(self) -> None:
        self.role_worker_mapping: dict = {}
        self.mapping: dict = {}

    def add_generator_worker(self, config):  # type: ignore[no-untyped-def]
        """Register the generator ActorRolloutRefWorker."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl_inf_evolve.workers.self_evolution_worker import (
            SelfEvolutionActorRolloutRefWorker,
        )

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

    def add_solver_worker(self, config):  # type: ignore[no-untyped-def]
        """Register the solver ActorRolloutRefWorker."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl_inf_evolve.workers.self_evolution_worker import (
            SelfEvolutionActorRolloutRefWorker,
        )

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

    def init_resource_pool_mgr(self, config):  # type: ignore[no-untyped-def]
        """Initialize the resource pool manager for GPU allocation.

        In replay mode all GPUs are allocated to a single solver pool
        regardless of the ``trainer.separate_pools`` setting.
        """
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        is_replay = config.gen_eval.get("mode", "regenerate") == "replay"

        if is_replay:
            # Replay mode: all GPUs go to the solver — no generator pool.
            resource_pool_spec = {
                "solver_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
        elif config.trainer.get("separate_pools", False):
            gen_gpus = config.trainer.get(
                "generator_n_gpus_per_node", config.trainer.n_gpus_per_node
            )
            solver_gpus = config.trainer.get(
                "solver_n_gpus_per_node", config.trainer.n_gpus_per_node
            )
            resource_pool_spec = {
                "generator_pool": [gen_gpus] * config.trainer.nnodes,
                "solver_pool": [solver_gpus] * config.trainer.nnodes,
            }
        else:
            resource_pool_spec = {
                "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }

        return ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )

    def run(self, config) -> None:  # type: ignore[no-untyped-def]
        """Execute the generator evaluation workflow.

        Steps:
          1. Register generator and solver workers
          2. Initialize resource pools
          3. Create tokenizers for both models
          4. Instantiate GeneratorEvaluator
          5. Call evaluator.init_workers() and evaluator.evaluate()

        In replay mode (``gen_eval.mode == 'replay'``), step 1 skips the
        generator worker registration and step 2 allocates all GPUs to
        the solver pool.
        """
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        # Configure logging inside the Ray actor
        log_level = os.environ.get("LOGLEVEL", "INFO").upper()
        logging.root.setLevel(getattr(logging, log_level, logging.INFO))

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        from verl_inf_evolve.storage.remote_backend import redact_config_secrets

        pprint(redact_config_secrets(OmegaConf.to_container(config, resolve=True)))

        # Disable struct mode so verl can access config fields with defaults
        OmegaConf.set_struct(config, False)

        is_replay = config.gen_eval.get("mode", "regenerate") == "replay"

        # Register worker groups — in replay mode, skip generator workers.
        if is_replay:
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup
            self.add_solver_worker(config)
            # Override mapping: all GPUs go to a dedicated solver_pool.
            self.mapping["solver"] = "solver_pool"
        else:
            _, ray_worker_group_cls = self.add_generator_worker(config)
            self.add_solver_worker(config)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        # Load tokenizers — gen_tokenizer is still loaded in replay mode
        # (lightweight, needed for question text decoding).
        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)

        gen_local_path = copy_to_local(
            config.generator.model.path,
            use_shm=config.generator.model.get("use_shm", False),
        )
        gen_tokenizer = hf_tokenizer(gen_local_path, trust_remote_code=trust_remote_code)

        solver_local_path = copy_to_local(
            config.solver.model.path,
            use_shm=config.solver.model.get("use_shm", False),
        )
        solver_tokenizer = hf_tokenizer(solver_local_path, trust_remote_code=trust_remote_code)

        # Create evaluator
        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator

        evaluator = GeneratorEvaluator(
            config=config,
            gen_tokenizer=gen_tokenizer,
            solver_tokenizer=solver_tokenizer,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
        )

        evaluator.init_workers()
        evaluator.evaluate()


if __name__ == "__main__":
    main()
