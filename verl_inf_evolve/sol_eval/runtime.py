"""Runtime wrappers for solver benchmark evaluation.

Supports two modes:
- create a fresh solver worker group + AgentLoopManager for standalone eval
- wrap an existing trainer runtime for in-training eval reuse
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from verl_inf_evolve.sol_eval.eval_core import generate_with_metadata

logger = logging.getLogger(__name__)


@dataclass
class SolverEvalRuntime:
    """Shared runtime for benchmark evaluation execution."""

    tokenizer: Any
    num_workers: int
    generate_batch_fn: Callable[[Any], Any]
    solver_wg: Any | None = None
    rollout_manager: Any | None = None
    _shutdown_callback: Callable[[], None] | None = field(
        default=None,
        repr=False,
    )

    @classmethod
    def from_existing_worker_group(
        cls,
        tokenizer: Any,
        num_workers: int,
        generate_batch_fn: Callable[[Any], Any],
        solver_wg: Any | None = None,
        rollout_manager: Any | None = None,
    ) -> "SolverEvalRuntime":
        """Wrap an existing runtime (e.g., from SelfEvolutionTrainer)."""
        return cls(
            tokenizer=tokenizer,
            num_workers=num_workers,
            generate_batch_fn=generate_batch_fn,
            solver_wg=solver_wg,
            rollout_manager=rollout_manager,
        )

    @classmethod
    def from_fresh_worker_group(cls, config: Any) -> "SolverEvalRuntime":
        """Initialize a fresh solver worker group and rollout manager."""
        import ray
        from omegaconf import OmegaConf
        from verl.experimental.agent_loop import AgentLoopManager
        from verl.single_controller.ray import RayWorkerGroup
        from verl.single_controller.ray.base import RayClassWithInitArgs
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl_inf_evolve.workers.self_evolution_worker import (
            SelfEvolutionActorRolloutRefWorker,
        )

        owns_ray = False
        if not ray.is_initialized():
            default_runtime_env = get_ppo_ray_runtime_env()
            ray.init(runtime_env=default_runtime_env)
            owns_ray = True

        solver_remote_cls = ray.remote(SelfEvolutionActorRolloutRefWorker)
        mapping = {"solver": "global_pool"}
        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )
        resource_pool_manager.create_resource_pool()

        solver_resource_pool = resource_pool_manager.get_resource_pool("solver")
        solver_cls = RayClassWithInitArgs(
            cls=solver_remote_cls,
            config=config.solver,
            role="actor_rollout",
        )
        solver_wg = RayWorkerGroup(
            resource_pool=solver_resource_pool,
            ray_cls_with_init=solver_cls,
        )
        solver_wg.init_model()

        trust_remote_code = config.data.get("trust_remote_code", False)
        solver_local_path = copy_to_local(
            config.solver.model.path,
            use_shm=config.solver.model.get("use_shm", False),
        )
        tokenizer = hf_tokenizer(
            solver_local_path,
            trust_remote_code=trust_remote_code,
        )
        custom_chat_template = config.solver.model.get("custom_chat_template", None)
        if custom_chat_template is not None:
            tokenizer.chat_template = custom_chat_template

        run_id = uuid.uuid4().hex[:8]
        solver_cfg_container = OmegaConf.to_container(config.solver, resolve=True)
        rollout_cfg = solver_cfg_container.setdefault("rollout", {})
        agent_cfg = rollout_cfg.setdefault("agent", {})
        agent_cfg["worker_name_prefix"] = f"eval_agent_loop_worker_{run_id}"
        rollout_cfg["server_name_prefix"] = f"eval_vllm_server_{run_id}"

        solver_alm_config = OmegaConf.create(
            {
                "actor_rollout_ref": solver_cfg_container,
                "reward_model": OmegaConf.to_container(config.reward_model, resolve=True),
                "trainer": OmegaConf.to_container(config.trainer, resolve=True),
                "data": OmegaConf.to_container(config.data, resolve=True),
            }
        )
        OmegaConf.set_struct(solver_alm_config, False)

        rollout_manager = AgentLoopManager(
            config=solver_alm_config,
            worker_group=solver_wg,
        )
        num_workers = int(config.trainer.n_gpus_per_node)

        def _generate_batch(batch: Any) -> Any:
            return generate_with_metadata(
                manager=rollout_manager,
                batch=batch,
                num_workers=num_workers,
            )

        def _shutdown() -> None:
            nonlocal rollout_manager, solver_wg
            try:
                if rollout_manager is not None:
                    del rollout_manager
                    rollout_manager = None
                if solver_wg is not None:
                    del solver_wg
                    solver_wg = None
                if owns_ray and ray.is_initialized():
                    ray.shutdown()
            except Exception:
                logger.exception("Failed to shut down solver eval runtime")

        return cls(
            tokenizer=tokenizer,
            num_workers=num_workers,
            generate_batch_fn=_generate_batch,
            solver_wg=solver_wg,
            rollout_manager=rollout_manager,
            _shutdown_callback=_shutdown,
        )

    def generate_batch(self, batch: Any) -> Any:
        """Run one rollout generation step for a prepared benchmark batch."""
        return self.generate_batch_fn(batch)

    def shutdown(self) -> None:
        """Release owned resources (if this runtime created them)."""
        if self._shutdown_callback is None:
            return
        self._shutdown_callback()
        self._shutdown_callback = None
