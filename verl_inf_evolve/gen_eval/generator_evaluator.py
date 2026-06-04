"""GeneratorEvaluator — evaluate generator checkpoints with a fixed solver.

Mirrors the :class:`SelfEvolutionTrainer` structure (Ray init, resource pool
creation, worker group registration) but runs only evaluation stages:

- Stage 2: Question generation (generator rollout)
- Stage 3: Solver answer rollout
- Stage 4: Influence scoring

Stages 1 (dev rollout logging), 5 (generator PPO), and 6 (solver PPO) are
NOT executed.

Since the solver is fixed, the dev rollout and dev gradient/momentum are
computed only once:
- Dev rollout runs once at startup (identical output for all checkpoints).
- Dev gradient + momentum init (Phase 1) runs on the first checkpoint only.
- Subsequent checkpoints skip Phase 1 and reuse the cached momentum buffer.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import re
from pathlib import Path
import subprocess
import time
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.core_algos import AdvantageEstimator

from verl_inf_evolve.data.batch_utils import messages_to_dataproto
from verl_inf_evolve.data.document_dataset import DocumentDataset
from verl_inf_evolve.utils.mcq_utils import (
    extract_answer_scores,
    group_scores_by_qid,
)
from verl_inf_evolve.sol_eval.benchmark_adapters import (
    build_messages_for_question,
    build_verifier_metadata,
    detect_benchmark_type,
)
from verl_inf_evolve.utils.prompts import (
    FREE_FORM_QUESTION_GENERATION_PROMPT,
    FREE_FORM_QUESTION_GENERATION_SYSTEM_PROMPT,
    MCQ_QUESTION_GENERATION_PROMPT,
    MCQ_QUESTION_GENERATION_SYSTEM_PROMPT,
)
from verl_inf_evolve.utils.data_utils import derive_gen_questions, scatter_for_dispatch
from verl_inf_evolve.utils.question_parser import parse_generated_questions
from verl_inf_evolve.utils.influence_utils import (
    add_similarity_metric_stats,
    build_similarity_rewards,
)
from verl_inf_evolve.utils.reward_utils import expand_scores_to_token_level
from verl_inf_evolve.gen_eval.checkpoint_manager import CheckpointPrefetcher, GenOutputPrefetcher
from verl_inf_evolve.utils.metric_utils import add_distribution_stats
from verl_inf_evolve.storage.r2_utils import parse_r2_path, r2_rclone_path
from verl_inf_evolve.trainer.rollout_metrics import (
    compute_answer_rollout_metrics,
    compute_question_rollout_metrics,
    compute_reward_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper classes for colocated worker groups (copied from main trainer)
# ---------------------------------------------------------------------------

class _PrefixedActorHandleView:
    """Expose unprefixed rollout methods for a prefixed colocated actor."""

    __slots__ = ("_actor_handle", "_role_prefix")

    def __init__(self, actor_handle: Any, role_prefix: str):
        object.__setattr__(self, "_actor_handle", actor_handle)
        object.__setattr__(self, "_role_prefix", role_prefix)

    def __getattr__(self, name: str):  # type: ignore[override]
        actor_handle = object.__getattribute__(self, "_actor_handle")
        try:
            return getattr(actor_handle, name)
        except AttributeError as exc:
            role_prefix = object.__getattribute__(self, "_role_prefix")
            prefixed_name = f"{role_prefix}_{name}"
            try:
                return getattr(actor_handle, prefixed_name)
            except AttributeError:
                raise exc

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (
            _PrefixedActorHandleView,
            (self._actor_handle, self._role_prefix),
        )


class _AgentLoopWorkerGroupView:
    """Minimal worker-group view used by AgentLoopManager."""

    def __init__(self, worker_group: RayWorkerGroup, role_prefix: str):
        self.world_size = worker_group.world_size
        self.workers = [
            _PrefixedActorHandleView(worker, role_prefix)
            for worker in worker_group.workers
        ]


# ---------------------------------------------------------------------------
# GeneratorEvaluator
# ---------------------------------------------------------------------------

class GeneratorEvaluator:
    """Evaluate generator checkpoints from a training trajectory.

    For each checkpoint, runs question generation → solver answer rollout →
    influence scoring, logging metrics to wandb.

    Args:
        config: Full OmegaConf config (gen_eval.yaml schema).
        gen_tokenizer: HuggingFace tokenizer for the generator model.
        solver_tokenizer: HuggingFace tokenizer for the solver model.
        role_worker_mapping: ``{"generator": RemoteCls, "solver": RemoteCls}``.
        resource_pool_manager: Manages GPU allocation for worker groups.
        ray_worker_group_cls: The class used to create worker groups.
    """

    def __init__(
        self,
        config: Any,
        gen_tokenizer: Any,
        solver_tokenizer: Any,
        role_worker_mapping: dict[str, Any],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
    ):
        # Validate gen_eval.mode before any workers are created.
        _VALID_EVAL_MODES = ("regenerate", "replay", "gpt_replay")
        eval_mode = config.gen_eval.get("mode", "regenerate")
        if eval_mode not in _VALID_EVAL_MODES:
            raise ValueError(
                f"Invalid gen_eval.mode={eval_mode!r}. "
                f"Accepted values: {_VALID_EVAL_MODES}"
            )

        self.config = config
        self.gen_tokenizer = gen_tokenizer
        self.solver_tokenizer = solver_tokenizer
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls

        # Populated by init_workers()
        self.generator_wg: Optional[RayWorkerGroup] = None
        self.solver_wg: Optional[RayWorkerGroup] = None
        self.gen_rollout_manager: Optional[Any] = None
        self.solver_rollout_manager: Optional[Any] = None
        self._shared_engine_mode: bool = False
        self.api_solver: Optional[Any] = None  # APISolverClient for gpt_replay mode

        # Unique ID for AgentLoop worker actor names
        self._agent_loop_run_id = uuid4().hex[:8]

        # True after Phase 1 of compute_influence_scores has run,
        # meaning dev-gradient momentum is live on the workers.
        self._momentum_live = False

        # Tracking logger (initialized by _init_tracking())
        self._tracker: Optional[Any] = None

    # ------------------------------------------------------------------
    # Tracking / wandb initialization
    # ------------------------------------------------------------------

    def _init_tracking(self) -> Any:
        """Create the metric tracking logger and configure eval x-axis.

        Mirrors ``SelfEvolutionTrainer._init_tracking()`` but uses
        ``eval/ans_loop`` as the single step metric for all prefixes.

        The run name includes the trajectory name and a timestamp, e.g.
        ``eval_FW-Alr_2e-6-vanilla_token-TIS_token_20260225_120000``.
        The group name is derived from the sync path so that eval
        runs for the same trajectory can be compared side-by-side.
        """
        from verl.utils.tracking import Tracking

        wandb_cfg = self.config.wandb

        # Derive run name from sync path + timestamp, including mode
        eval_mode = self.config.gen_eval.get("mode", "regenerate")
        _MODE_TAGS = {"replay": "replay", "gpt_replay": "gpt_replay", "regenerate": "regen"}
        mode_tag = _MODE_TAGS.get(eval_mode, "regen")

        run_name = wandb_cfg.get("run_name", None)
        if not run_name:
            trajectory_name = self._extract_trajectory_name()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"eval_{mode_tag}_{trajectory_name}_{timestamp}"

        # Derive group name from sync path
        group_name = wandb_cfg.get("group_name", None)
        if not group_name:
            group_name = self._extract_trajectory_name()

        if wandb_cfg.get("entity", None):
            os.environ.setdefault("WANDB_ENTITY", wandb_cfg.entity)
        if group_name:
            os.environ.setdefault("WANDB_RUN_GROUP", group_name)

        from verl_inf_evolve.storage.remote_backend import redact_config_secrets

        tracker = Tracking(
            project_name=wandb_cfg.project_name,
            experiment_name=run_name,
            default_backend=self.config.trainer.get("logger", ["console"]),
            config=redact_config_secrets(OmegaConf.to_container(self.config, resolve=True)),
        )

        # Configure eval/ans_loop as the x-axis for all metric prefixes
        if "wandb" in tracker.logger:
            import wandb

            wandb.summary["eval/mode"] = eval_mode
            wandb.summary["launch_command"] = " ".join(sys.argv)

            skypilot_task_id = os.environ.get("SKYPILOT_TASK_ID")
            if skypilot_task_id:
                wandb.summary["skypilot_task_id"] = skypilot_task_id

            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            if slurm_job_id:
                wandb.summary["slurm_job_id"] = slurm_job_id

            remote_sync_path = self.config.gen_eval.get("remote_sync_path", None)
            if remote_sync_path:
                wandb.summary["remote_sync_path"] = remote_sync_path

            wandb.define_metric("eval/ans_loop")
            wandb.define_metric("gen_question_rollout/*", step_metric="eval/ans_loop")
            wandb.define_metric("gen_answer_rollout/*", step_metric="eval/ans_loop")
            wandb.define_metric("gen_reward/*", step_metric="eval/ans_loop")
            wandb.define_metric("influence_sim/*", step_metric="eval/ans_loop")
            wandb.define_metric("influence_dev/*", step_metric="eval/ans_loop")
            wandb.define_metric("influence_gen/*", step_metric="eval/ans_loop")
            wandb.define_metric("influence_timing_s/*", step_metric="eval/ans_loop")
            wandb.define_metric("eval_timing_s/*", step_metric="eval/ans_loop")
            wandb.define_metric("dev_rollout/*", step_metric="eval/ans_loop")
            if eval_mode in ("replay", "gpt_replay"):
                wandb.define_metric("eval_replay/*", step_metric="eval/ans_loop")

        self._tracker = tracker
        return tracker

    def _extract_trajectory_name(self) -> str:
        """Extract a short trajectory name from the remote sync path.

        For ``s3://bucket/prefix/FW-Alr_2e-6-vanilla_token-TIS_token/``
        returns ``FW-Alr_2e-6-vanilla_token-TIS_token``.
        """
        remote_path = self.config.gen_eval.get("remote_sync_path", "") or ""
        # Strip trailing slashes and split
        parts = remote_path.rstrip("/").split("/")
        # Use the last non-empty segment
        for part in reversed(parts):
            if part and part not in ("", "s3:", "r2:"):
                return part
        return "unknown_trajectory"

    # ------------------------------------------------------------------
    # Resume / progress tracking (via wandb)
    # ------------------------------------------------------------------

    def _query_completed_indices_from_wandb(self) -> set[int]:
        """Query wandb for ans_loop indices that already have results.

        Searches all runs in the same project + group for logged
        ``gen_answer_rollout/accuracy_strict`` values, and returns the
        set of ``eval/ans_loop`` steps that are already complete.

        Returns an empty set if wandb is disabled or the query fails.
        """
        wandb_cfg = self.config.wandb
        if not wandb_cfg.get("enabled", True):
            return set()

        try:
            import wandb as wandb_module

            entity = wandb_cfg.get("entity", None) or os.environ.get("WANDB_ENTITY")
            project = wandb_cfg.get("project_name", None)
            group = wandb_cfg.get("group_name", None) or os.environ.get("WANDB_RUN_GROUP")

            if not entity or not project or not group:
                logger.info(
                    "Resume: cannot query wandb — missing entity/project/group."
                )
                return set()

            api = wandb_module.Api()
            runs = api.runs(
                f"{entity}/{project}",
                filters={"group": group},
            )

            completed: set[int] = set()
            for run in runs:
                # Scan history for rows that have the accuracy metric
                for row in run.scan_history(
                    keys=["eval/ans_loop", "gen_answer_rollout/accuracy_strict"],
                ):
                    ans_loop = row.get("eval/ans_loop")
                    acc = row.get("gen_answer_rollout/accuracy_strict")
                    if ans_loop is not None and acc is not None:
                        completed.add(int(ans_loop))

            if completed:
                logger.info(
                    "Resume: found %d completed indices in wandb group %s: %s",
                    len(completed), group, sorted(completed),
                )
            return completed

        except Exception as exc:
            logger.warning("Resume: failed to query wandb for completed indices: %s", exc)
            return set()

    # ------------------------------------------------------------------
    # HF namespace resolution
    # ------------------------------------------------------------------

    def _resolve_hf_remote_sync_path(self) -> None:
        """Resolve ``__namespace__`` placeholder in ``gen_eval.remote_sync_path``.

        Mirrors ``SelfEvolutionTrainer._resolve_hf_remote_sync_path()``.
        """
        remote_sync_path = self.config.gen_eval.get("remote_sync_path", None)
        if not remote_sync_path or not str(remote_sync_path).startswith("hf://"):
            return

        from verl_inf_evolve.storage.hf_remote_resolver import resolve_hf_remote_from_pool

        remote_cfg = self.config.get("remote", {})
        resolved = resolve_hf_remote_from_pool(str(remote_sync_path), remote_cfg)
        if resolved is None:
            return

        self.config.gen_eval.remote_sync_path = resolved.uri
        logger.info(
            "Resolved HF remote namespace placeholder: %s -> %s",
            remote_sync_path,
            resolved.uri,
        )
        if resolved.warning:
            logger.warning(resolved.warning)

    # ------------------------------------------------------------------
    # Worker initialization (mirrors SelfEvolutionTrainer.init_workers)
    # ------------------------------------------------------------------

    def init_workers(self) -> None:
        """Initialize Ray worker groups for generator and solver.

        Follows the same structure as ``SelfEvolutionTrainer.init_workers()``.

        In replay mode (``gen_eval.mode == 'replay'``), only solver workers are
        created.  Generator workers are skipped entirely — no
        ``RayWorkerGroup``, no ``AgentLoopManager``, no model loading.  All
        GPUs are allocated to the solver pool.  The generator tokenizer
        (``self.gen_tokenizer``) remains available for question text decoding.

        In gpt_replay mode, no Ray workers are created at all.  An
        ``APISolverClient`` is used instead for answering questions via API.
        """
        eval_mode = self.config.gen_eval.get("mode", "regenerate")

        if eval_mode == "gpt_replay":
            self._init_workers_gpt_replay()
            return

        self.resource_pool_manager.create_resource_pool()

        if eval_mode == "replay":
            self._init_workers_replay()
        else:
            self._init_workers_regenerate()

    def _init_workers_replay(self) -> None:
        """Solver-only worker initialization for replay mode."""
        # In replay mode, all GPUs go to the solver.  The resource pool
        # manager was configured with solver-only pools by TaskRunner.
        solver_resource_pool = self.resource_pool_manager.get_resource_pool("solver")
        solver_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping["solver"],
            config=self.config.solver,
            role="actor_rollout_ref",
        )

        self.solver_wg = self.ray_worker_group_cls(
            resource_pool=solver_resource_pool,
            ray_cls_with_init=solver_cls,
        )
        self.solver_wg.init_model()

        # Generator workers are not created in replay mode.
        self.generator_wg = None
        self.gen_rollout_manager = None
        self._shared_engine_mode = False

        # Solver AgentLoopManager (same as regenerate mode, non-shared path).
        from verl.experimental.agent_loop import AgentLoopManager

        solver_alm_config = self._build_agent_loop_config("solver")
        self.solver_rollout_manager = AgentLoopManager(
            config=solver_alm_config,
            worker_group=self.solver_wg,
        )

        logger.info(
            "Replay mode: solver workers initialized; "
            "generator workers skipped (gen_tokenizer still available)."
        )

    def _init_workers_gpt_replay(self) -> None:
        """API-only initialization for gpt_replay mode.

        No Ray workers or GPU resources are created.  An ``APISolverClient``
        handles MCQ answering via OpenAI/Gemini API calls.
        """
        from verl_inf_evolve.gen_eval.api_solver import APISolverClient

        api_solver_cfg = self.config.get("api_solver", None)
        if api_solver_cfg is None:
            raise ValueError(
                "gen_eval.mode='gpt_replay' requires an 'api_solver' config section. "
                "Add api_solver.provider, api_solver.model, etc. to your config."
            )

        self.api_solver = APISolverClient(api_solver_cfg)

        # No GPU workers in gpt_replay mode.
        self.generator_wg = None
        self.solver_wg = None
        self.gen_rollout_manager = None
        self.solver_rollout_manager = None
        self._shared_engine_mode = False

        logger.info(
            "gpt_replay mode: API solver initialized (provider=%s, model=%s); "
            "no Ray GPU workers created.",
            self.api_solver.provider,
            self.api_solver.model,
        )

    def _init_workers_regenerate(self) -> None:
        """Full generator + solver worker initialization for regenerate mode."""
        # --- Generator ---
        gen_resource_pool = self.resource_pool_manager.get_resource_pool("generator")
        gen_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping["generator"],
            config=self.config.generator,
            role="actor_rollout_ref",
        )

        # --- Solver ---
        solver_resource_pool = self.resource_pool_manager.get_resource_pool("solver")
        solver_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping["solver"],
            config=self.config.solver,
            role="actor_rollout_ref",
        )

        # --- Spawn worker groups ---
        shared_pool = gen_resource_pool is solver_resource_pool
        if not shared_pool:
            self.generator_wg = self.ray_worker_group_cls(
                resource_pool=gen_resource_pool,
                ray_cls_with_init=gen_cls,
            )
            self.solver_wg = self.ray_worker_group_cls(
                resource_pool=solver_resource_pool,
                ray_cls_with_init=solver_cls,
            )
        else:
            resource_pool_to_cls = {gen_resource_pool: {
                "generator": gen_cls,
                "solver": solver_cls,
            }}

            all_wg: dict[str, RayWorkerGroup] = {}
            for resource_pool, class_dict in resource_pool_to_cls.items():
                worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                wg_dict = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=worker_dict_cls,
                )
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
                all_wg.update(spawn_wg)

            self.generator_wg = all_wg["generator"]
            self.solver_wg = all_wg["solver"]

        self.generator_wg.init_model()
        self.solver_wg.init_model()

        # Create AgentLoopManagers for async rollout (verl 0.7.0).
        from verl.experimental.agent_loop import AgentLoopManager

        gen_alm_worker_group = self.generator_wg
        solver_alm_worker_group = self.solver_wg
        if shared_pool:
            gen_alm_worker_group = _AgentLoopWorkerGroupView(
                self.generator_wg, "generator"
            )
            solver_alm_worker_group = _AgentLoopWorkerGroupView(
                self.solver_wg, "solver"
            )

        self._shared_engine_mode = shared_pool
        if self._shared_engine_mode:
            self._validate_shared_engine_compatibility()
            solver_alm_config = self._build_agent_loop_config("solver")
            self.solver_rollout_manager = AgentLoopManager(
                config=solver_alm_config,
                worker_group=solver_alm_worker_group,
            )
            self.gen_rollout_manager = None
        else:
            gen_alm_config = self._build_agent_loop_config("generator")
            self.gen_rollout_manager = AgentLoopManager(
                config=gen_alm_config,
                worker_group=gen_alm_worker_group,
            )

            solver_alm_config = self._build_agent_loop_config("solver")
            self.solver_rollout_manager = AgentLoopManager(
                config=solver_alm_config,
                worker_group=solver_alm_worker_group,
            )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _build_agent_loop_config(self, model_role: str) -> DictConfig:
        """Build config compatible with AgentLoopManager from role-specific config."""
        role_config = self.config[model_role]
        role_cfg_container = OmegaConf.to_container(role_config, resolve=True)
        rollout_cfg = role_cfg_container.setdefault("rollout", {})
        agent_cfg = rollout_cfg.setdefault("agent", {})
        agent_cfg["worker_name_prefix"] = (
            f"{model_role}_agent_loop_worker_{self._agent_loop_run_id}"
        )
        rollout_cfg["server_name_prefix"] = (
            f"{model_role}_vllm_server_{self._agent_loop_run_id}"
        )

        cfg = OmegaConf.create({
            "actor_rollout_ref": role_cfg_container,
            "reward_model": OmegaConf.to_container(self.config.reward_model, resolve=True),
            "trainer": OmegaConf.to_container(self.config.trainer, resolve=True),
            "data": OmegaConf.to_container(self.config.data, resolve=True),
        })
        OmegaConf.set_struct(cfg, False)
        return cfg

    def _validate_shared_engine_compatibility(self) -> None:
        """Fail fast if generator/solver rollout settings are incompatible."""
        mismatches: list[str] = []

        def _check(field_name: str, gen_value: Any, solver_value: Any) -> None:
            if gen_value != solver_value:
                mismatches.append(
                    f"{field_name}: generator={gen_value!r}, solver={solver_value!r}"
                )

        gen_rollout = self.config.generator.rollout
        solver_rollout = self.config.solver.rollout

        _check("generator.model.path", self.config.generator.model.path, self.config.solver.model.path)
        _check("rollout.name", gen_rollout.name, solver_rollout.name)
        _check("rollout.tensor_model_parallel_size",
               gen_rollout.tensor_model_parallel_size, solver_rollout.tensor_model_parallel_size)
        _check("rollout.prompt_length", gen_rollout.prompt_length, solver_rollout.prompt_length)
        _check("rollout.response_length", gen_rollout.response_length, solver_rollout.response_length)

        if mismatches:
            mismatch_text = "\n".join(f"- {msg}" for msg in mismatches)
            raise ValueError(
                "Shared rollout engine mode requires compatible generator/solver rollout settings.\n"
                f"Mismatches:\n{mismatch_text}\n"
                "Set `trainer.separate_pools=true` or align the listed fields."
            )

    def _gen_meta_info(self) -> dict[str, Any]:
        """Meta info required by generate_sequences() for generator rollouts."""
        return {
            "eos_token_id": self.gen_tokenizer.eos_token_id,
            "pad_token_id": self.gen_tokenizer.pad_token_id or self.gen_tokenizer.eos_token_id,
        }

    def _solver_meta_info(self) -> dict[str, Any]:
        """Meta info required by generate_sequences() for solver rollouts."""
        return {
            "eos_token_id": self.solver_tokenizer.eos_token_id,
            "pad_token_id": self.solver_tokenizer.pad_token_id or self.solver_tokenizer.eos_token_id,
        }

    # ------------------------------------------------------------------
    # Ans loop index resolution
    # ------------------------------------------------------------------

    def _resolve_ans_loop_indices(self) -> list[int]:
        """Resolve ans_loop indices from config.

        If ``gen_eval.ans_loop_indices`` is set, uses that directly.
        Otherwise, uses ``gen_eval.every_n_ans_loop`` and auto-detects the
        available range by listing ``global_step_*`` directories in the
        remote trajectory via rclone.

        Returns:
            Sorted list of ans_loop indices to evaluate.
        """
        eval_cfg = self.config.gen_eval

        # Explicit list takes priority
        if eval_cfg.ans_loop_indices is not None:
            indices = list(eval_cfg.ans_loop_indices)
            logger.info("Using explicit ans_loop_indices: %s", indices)
            return sorted(indices)

        # Auto-detect from remote trajectory
        every_n = eval_cfg.every_n_ans_loop
        if every_n is None:
            raise ValueError(
                "Either gen_eval.ans_loop_indices or gen_eval.every_n_ans_loop must be set."
            )

        remote_path = eval_cfg.remote_sync_path
        if remote_path is None:
            raise ValueError(
                "gen_eval.remote_sync_path must be set when using every_n_ans_loop."
            )

        # Parse directory names — in replay/gpt_replay mode look for ans_*, otherwise global_step_*
        is_replay = self.config.gen_eval.get("mode", "regenerate") in ("replay", "gpt_replay")
        if is_replay:
            step_pattern = re.compile(r"ans_(\d+)")
        else:
            step_pattern = re.compile(r"global_step_(\d+)")

        all_steps: list[int] = []

        if str(remote_path).startswith("hf://"):
            # HF backend: use RemoteBackend.list_immediate_children()
            from verl_inf_evolve.storage.remote_backend import create_remote_backend

            logger.info("Listing HF remote directories at %s ...", remote_path)
            backend = create_remote_backend(str(remote_path))
            entries = backend.list_immediate_children("")
            for dirname in entries:
                m = step_pattern.match(dirname)
                if m:
                    all_steps.append(int(m.group(1)))
        else:
            # R2/S3 backend: use rclone
            bucket, prefix = parse_r2_path(remote_path)
            rclone_path = r2_rclone_path(bucket, prefix)

            logger.info("Listing remote directories at %s to detect global_step_* ...", rclone_path)
            try:
                result = subprocess.run(
                    ["rclone", "lsd", rclone_path],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"rclone lsd failed (rc={result.returncode}): {result.stderr}"
                    )
            except FileNotFoundError:
                raise RuntimeError("rclone binary not found; install rclone for remote directory listing")

            for line in result.stdout.strip().splitlines():
                # rclone lsd output format: "          -1 2024-01-01 00:00:00        -1 dirname"
                parts = line.strip().split()
                if parts:
                    dirname = parts[-1]
                    m = step_pattern.match(dirname)
                    if m:
                        all_steps.append(int(m.group(1)))

        if not all_steps:
            pattern_name = "ans_*" if is_replay else "global_step_*"
            raise RuntimeError(
                f"No {pattern_name} directories found at {rclone_path}. "
                "Check gen_eval.remote_sync_path."
            )

        all_steps.sort()
        max_step = all_steps[-1]
        indices = list(range(0, max_step + 1, every_n))
        # Ensure we only include steps that actually exist
        available = set(all_steps)
        indices = [i for i in indices if i in available]

        dir_type = "ans" if is_replay else "global_step"
        logger.info(
            "Auto-detected %d %s dirs (max=%d), every_n=%d → %d indices: %s",
            len(all_steps), dir_type, max_step, every_n, len(indices), indices,
        )
        return indices

    # ------------------------------------------------------------------
    # Data pipeline (reused from SelfEvolutionTrainer)
    # ------------------------------------------------------------------

    def prepare_dev_batch(self) -> DataProto:
        """Prepare a DataProto batch from the dev dataset for solver rollout.

        MCQ vs free-form is dispatched per-question via the shared benchmark
        adapter (same path the trainer uses), and is_mcq / benchmark_type /
        data_source / verifier_metadata are propagated into the non-tensor
        batch so the downstream verifier dispatch routes correctly.
        """
        messages_list = []
        question_ids = []
        ground_truths = []
        is_mcq_flags = []
        benchmark_types = []
        data_sources = []
        verifier_metadatas = []

        for q in self.dev_questions:
            messages, normalized_gt, question_is_mcq = build_messages_for_question(q)
            messages_list.append(messages)
            question_ids.append(q["question_id"])
            ground_truths.append(normalized_gt)
            is_mcq_flags.append(question_is_mcq)
            benchmark_types.append(detect_benchmark_type(q))
            data_sources.append(q.get("data_source", ""))
            verifier_metadatas.append(build_verifier_metadata(q))

        return messages_to_dataproto(
            messages_list=messages_list,
            non_tensor_metadata={
                "question_id": question_ids,
                "ground_truth": ground_truths,
                "is_mcq": is_mcq_flags,
                "benchmark_type": benchmark_types,
                "data_source": data_sources,
                "verifier_metadata": verifier_metadatas,
            },
            meta_info=self._solver_meta_info(),
        )

    def _init_question_generation_routes(self) -> None:
        """Compile source-pattern routes for document-conditioned generation.

        Mirrors ``SelfEvolutionTrainer._init_question_generation_routes``.
        Default route is MCQ generation; matching documents (regex against
        ``doc['source_pdf']``) can be redirected to the free-form generator
        prompt so the parser/solver/verifier dispatch downstream sees
        ``benchmark_type=qa_open`` / ``data_source=math``.
        """
        self._question_generation_routes: list[dict[str, Any]] = []
        training_cfg = self.config.get("training", {}) or {}
        routes = training_cfg.get("question_generation_routes", []) or []
        for idx, route in enumerate(routes):
            prompt_type = str(route.get("prompt_type", "mcq") or "mcq").lower()
            if prompt_type not in {"mcq", "free_form"}:
                raise ValueError(
                    "training.question_generation_routes[%d].prompt_type must be "
                    "'mcq' or 'free_form', got %r" % (idx, prompt_type)
                )
            patterns = route.get("source_patterns", []) or []
            if not patterns:
                logger.warning(
                    "question_generation_routes[%d] has no source_patterns; skipping",
                    idx,
                )
                continue
            self._question_generation_routes.append({
                "prompt_type": prompt_type,
                "source_regexes": [re.compile(p) for p in patterns],
            })
        if self._question_generation_routes:
            logger.info(
                "question_generation_routes: loaded %d source-pattern route(s)",
                len(self._question_generation_routes),
            )

    def _resolve_question_generation_prompt_type(self, doc: dict[str, Any]) -> str:
        """Return ``mcq`` or ``free_form`` for a document generation prompt."""
        source_pdf = str(doc.get("source_pdf", ""))
        for route in getattr(self, "_question_generation_routes", []) or []:
            if any(regex.search(source_pdf) for regex in route["source_regexes"]):
                return route["prompt_type"]
        return "mcq"

    def prepare_doc_batch(self) -> DataProto:
        """Prepare a DataProto batch of document prompts for generator rollout.

        Per-document prompt type (MCQ vs free_form) is resolved via
        ``_resolve_question_generation_prompt_type`` so Nemotron-CC-Math docs
        (or any source matching ``training.question_generation_routes``) are
        routed to the free-form generator prompt.
        """
        batch_doc_ids, reshuffled = self.doc_dataset.next_batch()
        if reshuffled:
            logger.info("Document dataset reshuffled (epoch %d)", self.doc_dataset.epoch)

        max_prompt_tokens = self.config.generator.rollout.prompt_length
        messages_list = []
        doc_ids = []
        prompt_types = []
        skipped = 0

        for doc_id in batch_doc_ids:
            doc = self.documents[doc_id]
            text = doc.get("content") or doc.get("text", "")
            prompt_type = self._resolve_question_generation_prompt_type(doc)
            if prompt_type == "free_form":
                messages = [
                    {"role": "system", "content": FREE_FORM_QUESTION_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": FREE_FORM_QUESTION_GENERATION_PROMPT.format(text=text)},
                ]
            else:
                messages = [
                    {"role": "system", "content": MCQ_QUESTION_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": MCQ_QUESTION_GENERATION_PROMPT.format(text=text)},
                ]

            token_len = len(self.gen_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
            ))
            if token_len > max_prompt_tokens:
                skipped += 1
                continue

            messages_list.append(messages)
            doc_ids.append(doc_id)
            prompt_types.append(prompt_type)

        if skipped:
            logger.warning(
                "Filtered %d/%d docs exceeding prompt_length=%d",
                skipped, len(batch_doc_ids), max_prompt_tokens,
            )

        if not messages_list:
            raise RuntimeError(
                f"All {len(batch_doc_ids)} documents in batch exceeded "
                f"prompt_length={max_prompt_tokens}."
            )

        if self._question_generation_routes:
            from collections import Counter
            counts = Counter(prompt_types)
            logger.info(
                "prepare_doc_batch: %d docs routed (%s)",
                len(prompt_types),
                ", ".join(f"{k}={v}" for k, v in counts.items()),
            )

        return messages_to_dataproto(
            messages_list=messages_list,
            non_tensor_metadata={
                "doc_id": doc_ids,
                "question_generation_prompt_type": prompt_types,
            },
            meta_info=self._gen_meta_info(),
        )

    def prepare_question_batch(self, questions: list[dict]) -> DataProto:
        """Prepare a DataProto batch of generated questions for solver rollout.

        MCQ vs free-form is dispatched per-question via the shared benchmark
        adapter; is_mcq / benchmark_type / data_source / verifier_metadata are
        propagated into the non-tensor batch for the downstream verifier.
        """
        if not questions:
            raise ValueError("prepare_question_batch received an empty question list.")

        messages_list = []
        question_ids = []
        ground_truths = []
        doc_ids = []
        is_mcq_flags = []
        benchmark_types = []
        data_sources = []
        verifier_metadatas = []
        question_texts = []

        for q in questions:
            messages, normalized_gt, question_is_mcq = build_messages_for_question(q)
            messages_list.append(messages)
            question_ids.append(q["question_id"])
            ground_truths.append(normalized_gt)
            doc_ids.append(q["doc_id"])
            is_mcq_flags.append(question_is_mcq)
            benchmark_types.append(detect_benchmark_type(q))
            data_sources.append(q.get("data_source", ""))
            verifier_metadatas.append(build_verifier_metadata(q))
            question_texts.append(q.get("question_text", "") or "")

        return messages_to_dataproto(
            messages_list=messages_list,
            non_tensor_metadata={
                "question_id": question_ids,
                "ground_truth": ground_truths,
                "doc_id": doc_ids,
                "is_mcq": is_mcq_flags,
                "benchmark_type": benchmark_types,
                "data_source": data_sources,
                "verifier_metadata": verifier_metadatas,
                # Propagate question_text so the per-Q influence-score JSONL
                # export can include it without re-loading gen_output.pt.
                "question_text": question_texts,
            },
            meta_info=self._solver_meta_info(),
        )

    # ------------------------------------------------------------------
    # Rollout helpers
    # ------------------------------------------------------------------

    def _generate_with_metadata(
        self, manager: Any, batch: DataProto, num_workers: int
    ) -> DataProto:
        """Call AgentLoopManager.generate_sequences() preserving non_tensor_batch fields."""
        saved_ntb = {
            k: v.copy() for k, v in batch.non_tensor_batch.items()
            if k not in ("raw_prompt", "agent_name")
        }

        batch_padded, pad_size = pad_dataproto_to_divisor(batch, num_workers)
        output_padded = manager.generate_sequences(batch_padded)
        output = unpad_dataproto(output_padded, pad_size)

        output.meta_info.pop("timing", None)

        for k, v in saved_ntb.items():
            output.non_tensor_batch[k] = v

        return output

    def _resolve_role_method(self, worker: Any, role: str, method: str) -> Any:
        """Resolve direct rollout method on colocated WorkerDict actors."""
        prefixed_name = f"{role}_{method}"
        try:
            return getattr(worker, prefixed_name)
        except AttributeError:
            return getattr(worker, method)

    def _direct_role_wake_up_all_workers(
        self, worker_group: RayWorkerGroup, role: str,
    ) -> None:
        """Directly wake up all workers for a specific role."""
        futures = []
        for worker in worker_group.workers:
            wake = self._resolve_role_method(worker, role, "wake_up")
            futures.append(wake.remote())
        results = ray.get(futures)
        if not all(bool(result) for result in results):
            raise RuntimeError(f"{role} wake_up did not succeed on all workers: {results}")

    def _direct_role_sleep_all_workers(
        self, worker_group: RayWorkerGroup, role: str,
    ) -> None:
        """Directly sleep all workers for a specific role."""
        futures = []
        for worker in worker_group.workers:
            sleep = self._resolve_role_method(worker, role, "sleep")
            futures.append(sleep.remote())
        results = ray.get(futures)
        if not all(bool(result) for result in results):
            raise RuntimeError(f"{role} sleep did not succeed on all workers: {results}")

    def _generate_with_shared_engine(self, batch: DataProto, num_workers: int) -> DataProto:
        """Run generator rollout through shared solver manager in shared-pool mode."""
        if not self._shared_engine_mode:
            raise RuntimeError("_generate_with_shared_engine called while shared-engine mode is disabled.")

        manager = self.solver_rollout_manager
        original_wake = manager.wake_up
        original_sleep = manager.sleep
        pending_error: Exception | None = None

        try:
            self._direct_role_wake_up_all_workers(self.generator_wg, "generator")
            manager.wake_up = lambda: None
            manager.sleep = lambda: None
            return self._generate_with_metadata(manager, batch, num_workers)
        except Exception as err:
            pending_error = err
            raise
        finally:
            manager.wake_up = original_wake
            manager.sleep = original_sleep
            try:
                self._direct_role_sleep_all_workers(self.generator_wg, "generator")
            except Exception:
                logger.exception("Failed to sleep generator workers during shared-engine cleanup.")
                if pending_error is None:
                    raise

    # ------------------------------------------------------------------
    # Influence scoring (reused from SelfEvolutionTrainer)
    # ------------------------------------------------------------------

    def compute_advantage(
        self, batch: DataProto, normalization_mode: Optional[str] = None
    ) -> DataProto:
        """Compute advantages for an influence scoring batch."""
        adv_estimator = AdvantageEstimator(
            self.config.algorithm.get("adv_estimator", "grpo")
        )
        if "token_level_scores" in batch.batch.keys() and "token_level_rewards" not in batch.batch.keys():
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        return compute_advantage(
            data=batch,
            adv_estimator=adv_estimator,
            gamma=self.config.algorithm.get("gamma", 1.0),
            lam=self.config.algorithm.get("lam", 1.0),
            norm_adv_by_std_in_grpo=self.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            ),
            normalization_mode=normalization_mode,
        )

    def _quantify_scores_before_advantage(
        self,
        scores: torch.Tensor,
        quantification_mode: Optional[str],
    ) -> torch.Tensor:
        """Optionally quantize sequence-level rewards before compute_advantage()."""
        if quantification_mode not in (
            None,
            "1bit",
            "2bit",
            "group_std_top_gamma",
            "group_std_fixed_threshold",
        ):
            raise ValueError(f"Invalid quantification_mode: {quantification_mode}")
        if quantification_mode in (
            None,
            "group_std_top_gamma",
            "group_std_fixed_threshold",
        ) or scores.numel() == 0:
            return scores
        if quantification_mode == "1bit":
            return torch.where(scores > 0, torch.ones_like(scores), torch.zeros_like(scores))
        # 2-bit
        abs_scores = scores.abs()
        threshold = abs_scores.median()
        return torch.where(
            scores >= threshold,
            torch.ones_like(scores),
            torch.where(
                scores > 0,
                torch.full_like(scores, 0.1),
                torch.where(
                    scores > -threshold,
                    torch.full_like(scores, -0.1),
                    torch.full_like(scores, -1.0),
                ),
            ),
        )

    def _prepare_influence_batch(
        self,
        rollout_output: DataProto,
        phase: str,
        reset_momentum: Optional[bool] = None,
        drop_zero_variance: bool = True,
    ) -> tuple[Optional[DataProto], list[str], list[int]]:
        """Prepare a DataProto for influence scoring from rollout output.

        Note: No score quantification is applied here because answer_scores
        are already binary (0.0 or 1.0).

        Args:
            phase: Either ``"dev_gradient"`` or ``"gen_similarity"``.
            reset_momentum: For ``"dev_gradient"`` phase only.
        """
        from collections import defaultdict

        responses = rollout_output.batch["responses"]
        batch_size = responses.shape[0]
        response_length = responses.size(1)
        question_ids = rollout_output.non_tensor_batch["question_id"]
        answer_scores = rollout_output.non_tensor_batch["answer_score"]

        row_scores = []
        question_order: list[str] = []
        seen: set[str] = set()

        for i in range(batch_size):
            qid = str(question_ids[i])
            if qid not in seen:
                question_order.append(qid)
                seen.add(qid)
            score = answer_scores[i]
            row_scores.append(float(score) if score is not None else 0.0)

        input_ids = rollout_output.batch["input_ids"]
        attention_mask = rollout_output.batch["attention_mask"]
        position_ids = rollout_output.batch["position_ids"]

        rewards_tensor = torch.tensor(row_scores, dtype=torch.float32)
        # NOTE: No quantification — answer_scores are already binary (0.0/1.0).

        # Zero-variance group filtering
        if drop_zero_variance:
            qid_scores: dict[str, list[float]] = defaultdict(list)
            for i in range(batch_size):
                qid_scores[str(question_ids[i])].append(rewards_tensor[i].item())

            zero_var_qids = set()
            for qid, slist in qid_scores.items():
                if len(slist) > 1 and len(set(slist)) == 1:
                    zero_var_qids.add(qid)

            if zero_var_qids:
                logger.info(
                    "_prepare_influence_batch: filtering %d/%d zero-variance question groups",
                    len(zero_var_qids), len(question_order),
                )

            selected_indices = [
                i for i in range(batch_size)
                if str(question_ids[i]) not in zero_var_qids
            ]
            question_order = [qid for qid in question_order if qid not in zero_var_qids]

            if not selected_indices:
                return None, [], []
        else:
            selected_indices = list(range(batch_size))

        indices_tensor = torch.tensor(selected_indices, dtype=torch.long)
        input_ids = input_ids[indices_tensor]
        attention_mask = attention_mask[indices_tensor]
        position_ids = position_ids[indices_tensor]
        responses = responses[indices_tensor]
        rewards_tensor = rewards_tensor[indices_tensor]
        filtered_batch_size = len(selected_indices)

        response_mask = attention_mask[:, -response_length:].float()

        token_level_scores = expand_scores_to_token_level(
            rewards_tensor, attention_mask, position_ids, response_length,
        )

        rollout_log_probs = None
        if "rollout_log_probs" in rollout_output.batch.keys():
            rollout_log_probs = rollout_output.batch["rollout_log_probs"][indices_tensor]

        tensors = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "token_level_scores": token_level_scores,
        }
        if rollout_log_probs is not None:
            tensors["rollout_log_probs"] = rollout_log_probs

        td = TensorDict(tensors, batch_size=filtered_batch_size)

        non_tensor_batch = {
            "uid": np.array(
                [str(question_ids[i]) for i in selected_indices], dtype=object
            ),
        }

        batch = DataProto(batch=td, non_tensor_batch=non_tensor_batch)

        # Pack meta_info
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        if rollout_corr_config is not None:
            rollout_corr_config = dict(rollout_corr_config)
        batch.meta_info.update({
            "temperature": self.config.solver.rollout.temperature,
            "rollout_corr_config": rollout_corr_config,
            # Mirror the trainer: propagate similarity_mode so the worker
            # uses preconditioned_cosine / preconditioned_dot when configured.
            # Without this the worker silently defaults to "cosine".
            "similarity_mode": self.config.influence.get(
                "similarity_mode", "cosine"
            ),
        })
        if phase == "dev_gradient":
            batch.meta_info.update({
                "momentum_beta": self.config.influence.get("momentum_beta", 0.9),
                "reset_momentum": reset_momentum if reset_momentum is not None else True,
            })
        elif phase == "gen_similarity":
            num_questions = len(question_order)
            rollout_n = filtered_batch_size // num_questions if num_questions > 0 else 1
            dp_size = self.solver_wg.world_size
            batch.meta_info["mini_batch_size"] = rollout_n // dp_size

        return batch, question_order, selected_indices

    def compute_influence_scores(
        self,
        dev_data: DataProto,
        gen_data: DataProto,
        skip_dev: bool,
        ans_loop: int = 0,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute gradient-based influence scores via distributed workers.

        Phase 1 (if not skip_dev): Forward+backward on dev data → dev gradient + momentum.
        Phase 2: Per-question cosine similarity against dev momentum.

        Returns:
            ``(rewards, influence_metrics)`` where rewards maps question_id to
            influence score (cosine similarity).
        """
        from verl.utils.metric import reduce_metrics
        from verl.utils.profiler.performance import simple_timer

        influence_metrics: dict[str, float] = {}
        timing: dict[str, float] = {}

        # Phase 1: Dev gradient + momentum
        if not skip_dev:
            use_momentum = self.config.influence.get("use_momentum", False)
            reset_momentum = not (use_momentum and ans_loop > 0)
            logger.info(
                "Phase 1: ans_loop=%d, use_momentum=%s, reset_momentum=%s",
                ans_loop, use_momentum, reset_momentum,
            )
            dev_batch, dev_question_order, _ = self._prepare_influence_batch(
                dev_data,
                phase="dev_gradient",
                reset_momentum=reset_momentum,
            )
            if dev_batch is None:
                logger.warning("Phase 1: all dev groups zero-variance, skipping dev gradient")
            else:
                dev_batch = self.compute_advantage(dev_batch)

                num_dev_questions = len(dev_question_order)
                dev_total = dev_batch.batch["input_ids"].shape[0]
                dev_rollout_n = dev_total // num_dev_questions
                dp_size = self.solver_wg.world_size
                dev_batch = scatter_for_dispatch(
                    dev_batch, num_dev_questions, dev_rollout_n, dp_size
                )
                if self.config.algorithm.get("use_global_batch_norm", True):
                    dev_batch.meta_info["avg_response_tokens"] = (
                        dev_batch.batch["response_mask"].float().sum().item()
                        / dev_batch.batch["response_mask"].shape[0]
                    )
                    dev_batch.meta_info["dp_size"] = dp_size

                with simple_timer("dev_gradient", timing):
                    dev_result = self.solver_wg.compute_dev_gradient(dev_batch)
                self._momentum_live = True
                dev_raw_metrics = dev_result.meta_info.get("metrics", {})
                dev_reduced = reduce_metrics(dev_raw_metrics) if dev_raw_metrics else {}
                for k, v in dev_reduced.items():
                    if k.startswith("influence_timing_s/"):
                        influence_metrics[f"influence_timing_s/dev_gradient/{k.removeprefix('influence_timing_s/')}"] = v
                    else:
                        influence_metrics[f"influence_dev/{k}"] = v

        # Phase 2: Per-question similarity
        all_question_ids = set(str(qid) for qid in gen_data.non_tensor_batch["question_id"])

        if not self._momentum_live:
            logger.warning(
                "Phase 2: momentum not available, assigning 0.0 to all %d questions",
                len(all_question_ids),
            )
            rewards = {qid: 0.0 for qid in all_question_ids}
            for k, v in timing.items():
                influence_metrics[f"influence_timing_s/{k}"] = v
            return rewards, influence_metrics

        gen_batch, question_order, _ = self._prepare_influence_batch(
            gen_data,
            phase="gen_similarity",
        )

        if gen_batch is None:
            logger.warning("Phase 2: all gen groups zero-variance, assigning 0.0 to all")
            rewards = {qid: 0.0 for qid in all_question_ids}
            for k, v in timing.items():
                influence_metrics[f"influence_timing_s/{k}"] = v
            return rewards, influence_metrics

        gen_batch = self.compute_advantage(gen_batch)

        num_questions = len(question_order)
        total_samples = gen_batch.batch["input_ids"].shape[0]
        rollout_n = total_samples // num_questions if num_questions > 0 else 1
        dp_size = self.solver_wg.world_size

        gen_batch = scatter_for_dispatch(gen_batch, num_questions, rollout_n, dp_size)
        if self.config.algorithm.get("use_global_batch_norm", True):
            gen_batch.meta_info["avg_response_tokens"] = (
                gen_batch.batch["response_mask"].float().sum().item()
                / gen_batch.batch["response_mask"].shape[0]
            )
            gen_batch.meta_info["dp_size"] = dp_size

        with simple_timer("similarity", timing):
            result = self.solver_wg.compute_similarity(gen_batch)

        gen_raw_metrics = result.meta_info.get("metrics", {})
        gen_reduced = reduce_metrics(gen_raw_metrics) if gen_raw_metrics else {}
        for k, v in gen_reduced.items():
            if k.startswith("influence_timing_s/"):
                influence_metrics[f"influence_timing_s/similarity/{k.removeprefix('influence_timing_s/')}"] = v
            else:
                influence_metrics[f"influence_gen/{k}"] = v

        similarity_metrics = result.meta_info["similarity_metrics"]
        add_similarity_metric_stats(influence_metrics, similarity_metrics)

        # Convert similarity metrics to per-question rewards
        rewards, scores, _score_mode = build_similarity_rewards(
            question_order,
            all_question_ids,
            similarity_metrics,
        )
        filtered_qids = all_question_ids - set(question_order)

        for k, v in timing.items():
            influence_metrics[f"influence_timing_s/{k}"] = v

        logger.info(
            "compute_influence_scores: %d questions (%d surviving, %d zero-variance), "
            "score range [%.4f, %.4f]",
            len(rewards), len(question_order), len(filtered_qids),
            min(scores) if scores else 0.0,
            max(scores) if scores else 0.0,
        )

        return rewards, influence_metrics

    def _maybe_export_per_q_influence(
        self,
        rewards: dict[str, float],
        gen_data: DataProto,
        ans_loop: int,
    ) -> None:
        """Write per-question influence scores to JSONL and optionally upload to HF.

        Reads ``gen_eval.influence_export``:
        - ``enabled`` (bool) — must be true to write anything (default false).
        - ``name`` (str, required) — used as the per-run subdirectory and as the
          filename stem; e.g. ``filter_nemotron_aime400`` produces
          ``filter_nemotron_aime400/per_question_influence_ans_{N}.jsonl``.
        - ``local_dir`` (str) — root for local writes (default
          ``.output/gen_eval/influence_scores``).
        - ``remote_path`` (str, optional) — base ``hf://datasets/<ns>/<repo>/<prefix>``
          URI; the ``__namespace__`` placeholder is resolved via the configured
          token pool. The final upload key is
          ``<remote_prefix>/<name>/per_question_influence_ans_{N}.jsonl``.
        - ``filename_template`` (str, optional) — defaults to
          ``per_question_influence_ans_{ans_loop}.jsonl``.

        Pulls per-question metadata (doc_id, is_mcq, benchmark_type,
        question_text, ground_truth, parsed_ok) from the gen_data
        non-tensor batch so the offline filter step doesn't have to
        re-download gen_output.pt.
        """
        export_cfg = self.config.gen_eval.get("influence_export", None) or {}
        if not export_cfg.get("enabled", False):
            return
        name = export_cfg.get("name")
        if not name:
            logger.warning(
                "gen_eval.influence_export.enabled=true but name is unset; skipping export."
            )
            return

        nt = gen_data.non_tensor_batch or {}

        def _col(key: str) -> list:
            v = nt.get(key)
            if v is None:
                return []
            return list(v)

        qids = _col("question_id")
        doc_ids = _col("doc_id")
        is_mcq_flags = _col("is_mcq")
        bench_types = _col("benchmark_type")
        data_sources = _col("data_source")
        q_texts = _col("question_text")
        gts = _col("ground_truth")
        parsed_oks = _col("parsed_ok")

        n_rows = len(qids)
        # Group by qid (each rollout repeats the question metadata; keep one
        # record per unique qid).
        seen: set[str] = set()
        records: list[dict[str, Any]] = []
        for i in range(n_rows):
            qid = str(qids[i])
            if qid in seen:
                continue
            seen.add(qid)
            row: dict[str, Any] = {
                "question_id": qid,
                "ans_loop": ans_loop,
                "influence_score": rewards.get(qid),
            }
            if i < len(doc_ids):
                row["doc_id"] = str(doc_ids[i]) if doc_ids[i] is not None else None
            if i < len(is_mcq_flags) and is_mcq_flags[i] is not None:
                try:
                    row["is_mcq"] = bool(is_mcq_flags[i])
                except Exception:
                    row["is_mcq"] = None
            if i < len(bench_types):
                row["benchmark_type"] = str(bench_types[i]) if bench_types[i] is not None else None
            if i < len(data_sources):
                row["data_source"] = str(data_sources[i]) if data_sources[i] is not None else None
            if i < len(q_texts):
                row["question_text"] = str(q_texts[i]) if q_texts[i] is not None else None
            if i < len(gts):
                row["ground_truth"] = str(gts[i]) if gts[i] is not None else None
            if i < len(parsed_oks):
                row["parsed_ok"] = bool(parsed_oks[i]) if parsed_oks[i] is not None else None
            records.append(row)

        # Local write
        local_root = Path(export_cfg.get("local_dir", ".output/gen_eval/influence_scores"))
        local_dir = local_root / name
        local_dir.mkdir(parents=True, exist_ok=True)
        fname_tmpl = export_cfg.get("filename_template", "per_question_influence_ans_{ans_loop}.jsonl")
        fname = fname_tmpl.format(ans_loop=ans_loop)
        local_path = local_dir / fname
        with open(local_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        logger.info(
            "Wrote %d per-Q influence records to %s",
            len(records), local_path,
        )

        # Optional HF upload
        remote_path = export_cfg.get("remote_path")
        if not remote_path:
            return
        try:
            from verl_inf_evolve.storage.hf_remote_resolver import (
                resolve_hf_remote_from_pool,
            )
            from verl_inf_evolve.storage.remote_backend import _parse_hf_dataset_uri
            from huggingface_hub import HfApi

            remote_cfg = self.config.get("remote", {}) or {}
            resolved = resolve_hf_remote_from_pool(remote_path, remote_cfg)
            if resolved is None:
                # No __namespace__ placeholder: parse directly and use the
                # default HF token (env HF_TOKEN).
                parsed = _parse_hf_dataset_uri(remote_path)
                repo_id = parsed["repo_id"]
                prefix = parsed["prefix"]
                token = os.environ.get(
                    str(remote_cfg.get("hf_token_env_var", "HF_TOKEN")),
                    None,
                )
            else:
                repo_id = f"{resolved.namespace}/{resolved.repo}"
                prefix = resolved.prefix
                token = resolved.token

            key = f"{prefix.rstrip('/')}/{name}/{fname}" if prefix else f"{name}/{fname}"
            api = HfApi(token=token)
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=key,
                repo_id=repo_id,
                repo_type="dataset",
            )
            logger.info(
                "Uploaded per-Q influence scores to hf://datasets/%s/%s",
                repo_id, key,
            )
        except Exception as exc:
            logger.warning(
                "Failed to upload per-Q influence scores to %s: %s",
                remote_path, exc,
            )

    # ------------------------------------------------------------------
    # Dataset initialization
    # ------------------------------------------------------------------

    def _init_datasets(self) -> None:
        """Load dev questions and documents for evaluation.

        In replay/gpt_replay mode, documents are not needed (questions come
        from gen_output.pt) so only dev questions are loaded. In gpt_replay
        mode, dev questions are also optional (no influence scoring).
        """
        eval_cfg = self.config.gen_eval
        eval_mode = eval_cfg.get("mode", "regenerate")

        # Dev questions — not needed in gpt_replay (no influence scoring)
        if eval_mode == "gpt_replay":
            self.dev_questions = []
            logger.info("gpt_replay mode: skipping dev question loading.")
        else:
            with open(eval_cfg.dev_dataset_path, "r") as f:
                self.dev_questions = json.load(f)
            logger.info("Loaded %d dev questions from %s",
                         len(self.dev_questions), eval_cfg.dev_dataset_path)

        # Documents — only needed in regenerate mode for prepare_doc_batch()
        is_replay = eval_mode in ("replay", "gpt_replay")
        if not is_replay:
            with open(eval_cfg.doc_path, "r") as f:
                documents_list = json.load(f)
            self.documents = {str(doc["doc_id"]): doc for doc in documents_list}

            all_doc_ids = [str(doc["doc_id"]) for doc in documents_list]
            self.doc_dataset = DocumentDataset(
                all_doc_ids,
                batch_size=len(all_doc_ids),  # Use all docs in each eval round
                shuffle=False,
            )
            logger.info("Loaded %d eval documents from %s",
                         len(self.documents), eval_cfg.doc_path)

            # Compile training.question_generation_routes for prepare_doc_batch.
            self._init_question_generation_routes()
        else:
            self._question_generation_routes = []

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def evaluate(self) -> None:
        """Run the generator evaluation loop.

        Before the loop:
        - Runs dev rollout once (solver is fixed → identical for all checkpoints).
        - Logs dev rollout metrics once as a sanity check.

        **Regenerate mode** (default) — for each checkpoint:
        1. Download/load generator checkpoint via CheckpointPrefetcher
        2. Load generator weights via ``load_hf_checkpoint()``
        3. Run Stage 2 (question generation):
           ``prepare_doc_batch() → generator rollout → derive_gen_questions()``
        4. Run Stage 3 (solver answer rollout)
        5. Run Stage 4 (influence scoring)
        6. Log metrics

        **Replay mode** — for each ans_loop index:
        1. Download pre-saved gen_output.pt via GenOutputPrefetcher
        2. Derive questions via ``derive_gen_questions()`` (no generation)
        3. Run Stage 3 (solver answer rollout) — identical to regenerate
        4. Run Stage 4 (influence scoring) — identical to regenerate
        5. Log metrics

        **gpt_replay mode** — for each ans_loop index:
        1. Download pre-saved gen_output.pt via GenOutputPrefetcher
        2. Derive questions via ``derive_gen_questions()`` (no generation)
        3. Solve questions via API (OpenAI/Gemini) — no GPU workers
        4. Skip influence scoring (no local model)
        5. Log metrics

        In all modes, solver weights are never modified during the loop.
        """
        eval_mode = self.config.gen_eval.get("mode", "regenerate")

        if eval_mode == "gpt_replay":
            return self._evaluate_gpt_replay()

        # Resolve __namespace__ placeholder before anything reads the path.
        self._resolve_hf_remote_sync_path()

        self._init_datasets()

        # Initialize tracking / wandb
        tracker = self._init_tracking()

        # Resolve which checkpoints to evaluate
        ans_loop_indices = self._resolve_ans_loop_indices()

        # Resume support: skip checkpoints already logged to wandb.
        completed = self._query_completed_indices_from_wandb()
        if completed:
            before = len(ans_loop_indices)
            ans_loop_indices = [i for i in ans_loop_indices if i not in completed]
            skipped = before - len(ans_loop_indices)
            if skipped:
                logger.info(
                    "Resuming: skipping %d already-completed checkpoints (%s)",
                    skipped, sorted(completed),
                )
        logger.info("Will evaluate %d checkpoints: %s", len(ans_loop_indices), ans_loop_indices)

        # Create prefetcher based on mode
        eval_cfg = self.config.gen_eval
        is_replay = eval_cfg.get("mode", "regenerate") == "replay"

        if is_replay:
            gen_output_prefetcher = GenOutputPrefetcher(
                remote_base_path=eval_cfg.remote_sync_path,
                local_cache_dir=eval_cfg.local_cache_dir,
                prefetch_count=eval_cfg.prefetch_count,
                cleanup_after_eval=True,
            )
            gen_output_prefetcher.set_indices(ans_loop_indices)
        else:
            prefetcher = CheckpointPrefetcher(
                remote_base_path=eval_cfg.remote_sync_path,
                local_cache_dir=eval_cfg.local_cache_dir,
                prefetch_count=eval_cfg.prefetch_count,
                cleanup_after_eval=True,
            )
            prefetcher.set_indices(ans_loop_indices)

        # Rollout config values
        solver_n = self.config.solver.rollout.n
        solver_num_workers = self.config.solver.rollout.agent.num_workers
        if not is_replay:
            gen_n = self.config.generator.rollout.n
            gen_num_workers = self.config.generator.rollout.agent.num_workers

        skip_influence = eval_cfg.get("skip_influence", False)
        dev_output = None

        if skip_influence:
            logger.info("skip_influence=True — skipping dev rollout and influence scoring.")
        else:
            # Run dev rollout once before the checkpoint loop.
            # The solver is fixed, so dev rollout output is identical for all
            # checkpoints. This dev_output is passed to compute_influence_scores()
            # for every checkpoint.
            logger.info("Running dev rollout (solver is fixed, computed once)...")
            t_dev = time.time()
            dev_batch = self.prepare_dev_batch()
            dev_batch = dev_batch.repeat(repeat_times=solver_n, interleave=True)
            dev_output = self._generate_with_metadata(
                self.solver_rollout_manager, dev_batch, solver_num_workers
            )
            extract_answer_scores(dev_output, self.solver_tokenizer)
            dev_time = time.time() - t_dev
            logger.info("Dev rollout complete in %.1fs.", dev_time)

            # Log dev rollout metrics once — these are constant across all
            # checkpoints since the solver is fixed.
            dev_scores_dict = group_scores_by_qid(dev_output)
            dev_rollout_metrics = compute_answer_rollout_metrics(
                scores=dev_scores_dict,
                response_mask=dev_output.batch["response_mask"],
                rollout_n=solver_n,
                prefix="dev_rollout",
            )
            dev_rollout_metrics["eval_timing_s/dev_rollout"] = dev_time
            logger.info(
                "Dev rollout metrics (constant for all checkpoints): %s",
                {k: f"{v:.4f}" if isinstance(v, float) else v
                 for k, v in dev_rollout_metrics.items()},
            )

            # Log dev rollout metrics once — they are constant since solver is fixed.
            # Use step=0 (or the first ans_loop index) as a reference point.
            dev_log_step = ans_loop_indices[0] if ans_loop_indices else 0
            dev_rollout_metrics["eval/ans_loop"] = dev_log_step
            tracker.log(data=dev_rollout_metrics, step=dev_log_step)

        try:
            for ckpt_num, ans_loop_idx in enumerate(ans_loop_indices):
                logger.info(
                    "=== Evaluating checkpoint %d/%d: ans_loop=%d ===",
                    ckpt_num + 1, len(ans_loop_indices), ans_loop_idx,
                )
                checkpoint_metrics: dict[str, float] = {}

                if is_replay:
                    # ----------------------------------------------------------
                    # Replay: Download gen_output.pt (no generation needed)
                    # ----------------------------------------------------------
                    t0 = time.time()
                    gen_output = gen_output_prefetcher.get_gen_output(ans_loop_idx)
                    checkpoint_metrics["eval_timing_s/gen_output_download"] = time.time() - t0

                    gen_questions = derive_gen_questions(gen_output)

                    if not gen_questions:
                        logger.warning(
                            "No valid questions in gen_output.pt at ans_loop=%d, skipping.",
                            ans_loop_idx,
                        )
                        continue

                    # Question rollout metrics from pre-saved gen_output
                    total_samples = gen_output.batch["responses"].shape[0]
                    question_metrics = compute_question_rollout_metrics(
                        gen_questions=gen_questions,
                        total_samples=total_samples,
                        response_mask=gen_output.batch["response_mask"],
                        num_documents=len(set(gen_output.non_tensor_batch["doc_id"])),
                        prefix="gen_question_rollout",
                        reject_reasons=gen_output.non_tensor_batch.get("reject_reason"),
                    )
                    checkpoint_metrics.update(question_metrics)

                    # Replay-specific metadata for wandb
                    num_valid = len(gen_questions)
                    checkpoint_metrics["eval_replay/total_samples"] = float(total_samples)
                    checkpoint_metrics["eval_replay/num_valid_questions"] = float(num_valid)
                    checkpoint_metrics["eval_replay/num_rejected_questions"] = float(
                        total_samples - num_valid
                    )
                else:
                    # ----------------------------------------------------------
                    # Regenerate: Download checkpoint, load weights, generate
                    # ----------------------------------------------------------
                    # Step 1: Download and load generator checkpoint.
                    # Special case: ans_loop_idx == -1 + allow_missing means
                    # "use the base model from generator.model.path that
                    # init_workers() already loaded into self.generator_wg".
                    allow_missing = bool(self.config.gen_eval.get(
                        "allow_missing_generator_checkpoint", False
                    ))
                    if ans_loop_idx == -1 and allow_missing:
                        logger.info(
                            "ans_loop=-1 + allow_missing_generator_checkpoint=true: "
                            "skipping checkpoint download/load; using base model "
                            "loaded from generator.model.path at init_workers()."
                        )
                        checkpoint_metrics["eval_timing_s/checkpoint_download"] = 0.0
                        checkpoint_metrics["eval_timing_s/checkpoint_load"] = 0.0
                    else:
                        t0 = time.time()
                        checkpoint_path = prefetcher.get_checkpoint(ans_loop_idx)
                        checkpoint_metrics["eval_timing_s/checkpoint_download"] = time.time() - t0

                        logger.info("Loading generator checkpoint from %s", checkpoint_path)
                        t0 = time.time()
                        self.generator_wg.load_hf_checkpoint(local_path=checkpoint_path)  # type: ignore[union-attr]
                        checkpoint_metrics["eval_timing_s/checkpoint_load"] = time.time() - t0
                        logger.info("Generator checkpoint loaded for ans_loop=%d", ans_loop_idx)

                    # Step 2: Stage 2 — Question generation
                    t0 = time.time()
                    doc_batch = self.prepare_doc_batch()
                    doc_batch = doc_batch.repeat(repeat_times=gen_n, interleave=True)

                    if self._shared_engine_mode:
                        gen_output = self._generate_with_shared_engine(
                            doc_batch, solver_num_workers
                        )
                    else:
                        gen_output = self._generate_with_metadata(
                            self.gen_rollout_manager, doc_batch, gen_num_workers
                        )

                    parse_generated_questions(gen_output, self.gen_tokenizer)
                    gen_questions = derive_gen_questions(gen_output)
                    checkpoint_metrics["eval_timing_s/question_generation"] = time.time() - t0

                    if not gen_questions:
                        logger.warning(
                            "No valid questions generated at ans_loop=%d, skipping.",
                            ans_loop_idx,
                        )
                        continue

                    # Question generation metrics
                    question_metrics = compute_question_rollout_metrics(
                        gen_questions=gen_questions,
                        total_samples=gen_output.batch["responses"].shape[0],
                        response_mask=gen_output.batch["response_mask"],
                        num_documents=len(set(gen_output.non_tensor_batch["doc_id"])),
                        prefix="gen_question_rollout",
                        reject_reasons=gen_output.non_tensor_batch.get("reject_reason"),
                    )
                    checkpoint_metrics.update(question_metrics)

                # ----------------------------------------------------------
                # Step 3: Stage 3 — Solver answer rollout
                # ----------------------------------------------------------
                t0 = time.time()
                gen_q_batch = self.prepare_question_batch(gen_questions)
                gen_q_batch = gen_q_batch.repeat(repeat_times=solver_n, interleave=True)
                gen_answer_output = self._generate_with_metadata(
                    self.solver_rollout_manager, gen_q_batch, solver_num_workers
                )
                extract_answer_scores(gen_answer_output, self.solver_tokenizer)
                checkpoint_metrics["eval_timing_s/answer_rollout"] = time.time() - t0

                # Answer rollout metrics
                gen_scores_dict = group_scores_by_qid(gen_answer_output)
                answer_metrics = compute_answer_rollout_metrics(
                    scores=gen_scores_dict,
                    response_mask=gen_answer_output.batch["response_mask"],
                    rollout_n=solver_n,
                    prefix="gen_answer_rollout",
                )
                checkpoint_metrics.update(answer_metrics)

                # ----------------------------------------------------------
                # Step 4: Stage 4 — Influence scoring (optional)
                # ----------------------------------------------------------
                if not skip_influence:
                    # On the first checkpoint (ckpt_num == 0), run Phase 1
                    # (dev gradient → momentum init) with skip_dev=False.
                    # For all subsequent checkpoints, skip Phase 1 since the
                    # solver is fixed — the momentum buffer is identical and
                    # load_hf_checkpoint() on the generator doesn't touch it.
                    skip_dev = self._momentum_live
                    t0 = time.time()
                    rewards, influence_metrics = self.compute_influence_scores(
                        dev_data=dev_output,
                        gen_data=gen_answer_output,
                        skip_dev=skip_dev,
                        ans_loop=ans_loop_idx,
                    )
                    checkpoint_metrics["eval_timing_s/influence_scoring"] = time.time() - t0

                    # Reward metrics
                    reward_metrics = compute_reward_metrics(
                        rewards=rewards, prefix="gen_reward"
                    )
                    checkpoint_metrics.update(reward_metrics)
                    checkpoint_metrics.update(influence_metrics)

                    # Optional: export per-question influence scores to a
                    # JSONL file (and optionally upload to HF) so downstream
                    # filter/aggregation scripts have access to them.
                    self._maybe_export_per_q_influence(
                        rewards=rewards,
                        gen_data=gen_answer_output,
                        ans_loop=ans_loop_idx,
                    )

                # ----------------------------------------------------------
                # Step 5: Log metrics
                # ----------------------------------------------------------
                checkpoint_metrics["eval/ans_loop"] = ans_loop_idx

                # Log summary to console
                key_metrics = {
                    "gen_question_rollout/num_valid_questions": checkpoint_metrics.get(
                        "gen_question_rollout/num_valid_questions", 0
                    ),
                    "gen_answer_rollout/accuracy_strict": checkpoint_metrics.get(
                        "gen_answer_rollout/accuracy_strict", 0
                    ),
                    "gen_reward/mean": checkpoint_metrics.get("gen_reward/mean", 0),
                }
                logger.info(
                    "ans_loop=%d summary: %s",
                    ans_loop_idx,
                    {k: f"{v:.4f}" if isinstance(v, float) else v
                     for k, v in key_metrics.items()},
                )
                if is_replay:
                    logger.info(
                        "ans_loop=%d timing: gen_output_download=%.1fs, "
                        "answer=%.1fs, influence=%.1fs",
                        ans_loop_idx,
                        checkpoint_metrics.get("eval_timing_s/gen_output_download", 0),
                        checkpoint_metrics.get("eval_timing_s/answer_rollout", 0),
                        checkpoint_metrics.get("eval_timing_s/influence_scoring", 0),
                    )
                else:
                    logger.info(
                        "ans_loop=%d timing: download=%.1fs, load=%.1fs, "
                        "gen=%.1fs, answer=%.1fs, influence=%.1fs",
                        ans_loop_idx,
                        checkpoint_metrics.get("eval_timing_s/checkpoint_download", 0),
                        checkpoint_metrics.get("eval_timing_s/checkpoint_load", 0),
                        checkpoint_metrics.get("eval_timing_s/question_generation", 0),
                        checkpoint_metrics.get("eval_timing_s/answer_rollout", 0),
                        checkpoint_metrics.get("eval_timing_s/influence_scoring", 0),
                    )

                # Log all checkpoint metrics to wandb / tracking backends
                tracker.log(data=checkpoint_metrics, step=ans_loop_idx)

        finally:
            if is_replay:
                gen_output_prefetcher.shutdown()
            else:
                prefetcher.shutdown()

        logger.info(
            "Generator evaluation complete for %d checkpoints.",
            len(ans_loop_indices),
        )

    def _evaluate_gpt_replay(self) -> None:
        """Run the gpt_replay evaluation loop.

        Similar to replay mode but uses API calls instead of vLLM rollouts:
        1. Downloads gen_output.pt from remote (training-time generator outputs)
        2. Derives generated questions via ``derive_gen_questions()``
        3. Calls ``APISolverClient.solve_questions()`` instead of vLLM rollout
        4. Computes accuracy metrics (no influence scoring)
        5. Logs to wandb

        In **batch mode**, uses a two-pass approach for parallelism:
        - Pass 1: Download all gen_outputs, derive questions, submit all
          batches to OpenAI (non-blocking).
        - Pass 2: Poll each batch, collect results, compute metrics, log.
        All batches run concurrently on OpenAI's side, so wall time is
        ``max(batch_times)`` instead of ``sum(batch_times)``.
        """
        from verl_inf_evolve.gen_eval.api_solver import (
            BatchHandle,
            compute_api_answer_metrics,
        )

        # Resolve __namespace__ placeholder before anything reads the path.
        self._resolve_hf_remote_sync_path()

        self._init_datasets()

        # Initialize tracking / wandb
        tracker = self._init_tracking()

        # Resolve which checkpoints to evaluate
        ans_loop_indices = self._resolve_ans_loop_indices()

        # Resume support: skip checkpoints already logged to wandb.
        completed = self._query_completed_indices_from_wandb()
        if completed:
            before = len(ans_loop_indices)
            ans_loop_indices = [i for i in ans_loop_indices if i not in completed]
            skipped = before - len(ans_loop_indices)
            if skipped:
                logger.info(
                    "Resuming: skipping %d already-completed checkpoints (%s)",
                    skipped, sorted(completed),
                )
        logger.info(
            "gpt_replay: will evaluate %d checkpoints: %s",
            len(ans_loop_indices), ans_loop_indices,
        )

        # Create gen_output prefetcher (same as replay mode)
        eval_cfg = self.config.gen_eval
        backend_kwargs = self._build_remote_backend_kwargs()
        gen_output_prefetcher = GenOutputPrefetcher(
            remote_base_path=eval_cfg.remote_sync_path,
            local_cache_dir=eval_cfg.local_cache_dir,
            prefetch_count=eval_cfg.prefetch_count,
            cleanup_after_eval=True,
            backend_kwargs=backend_kwargs,
        )
        gen_output_prefetcher.set_indices(ans_loop_indices)

        # API solver rollout_n
        api_n = self.api_solver.n
        is_batch = self.api_solver.mode == "batch"

        logger.info(
            "gpt_replay: no dev rollout or influence scoring (API-only mode, %s).",
            "batch" if is_batch else "sync",
        )

        try:
            if is_batch:
                self._evaluate_gpt_replay_batch(
                    ans_loop_indices, eval_cfg, gen_output_prefetcher,
                    api_n, tracker,
                )
            else:
                self._evaluate_gpt_replay_sync(
                    ans_loop_indices, gen_output_prefetcher, api_n, tracker,
                )
        finally:
            gen_output_prefetcher.shutdown()

        logger.info(
            "gpt_replay evaluation complete for %d checkpoints.",
            len(ans_loop_indices),
        )

    def _evaluate_gpt_replay_sync(
        self,
        ans_loop_indices: list[int],
        gen_output_prefetcher: GenOutputPrefetcher,
        api_n: int,
        tracker: Any,
    ) -> None:
        """Single-pass gpt_replay for sync mode."""
        from verl_inf_evolve.gen_eval.api_solver import compute_api_answer_metrics

        for ckpt_num, ans_loop_idx in enumerate(ans_loop_indices):
            logger.info(
                "=== [gpt_replay/sync] Evaluating checkpoint %d/%d: ans_loop=%d ===",
                ckpt_num + 1, len(ans_loop_indices), ans_loop_idx,
            )
            checkpoint_metrics = self._download_and_derive_questions(
                gen_output_prefetcher, ans_loop_idx,
            )
            if checkpoint_metrics is None:
                continue
            gen_questions = checkpoint_metrics.pop("_gen_questions")

            t0 = time.time()
            gen_scores_dict, api_results = self.api_solver.solve_questions(
                gen_questions
            )
            checkpoint_metrics["eval_timing_s/answer_rollout"] = time.time() - t0

            self._finalize_gpt_replay_metrics(
                checkpoint_metrics, gen_scores_dict, api_results,
                api_n, ans_loop_idx, tracker,
            )

    def _evaluate_gpt_replay_batch(
        self,
        ans_loop_indices: list[int],
        eval_cfg: Any,
        gen_output_prefetcher: GenOutputPrefetcher,
        api_n: int,
        tracker: Any,
    ) -> None:
        """Two-pass gpt_replay for batch mode.

        Pass 1: Download gen_output for each checkpoint, derive questions,
        submit batch to OpenAI (non-blocking). All batches start processing
        concurrently on OpenAI's side.

        Pass 2: For each checkpoint, poll its batch until complete, collect
        results, compute metrics, and log to wandb.
        """
        from verl_inf_evolve.gen_eval.api_solver import (
            BatchHandle,
            compute_api_answer_metrics,
        )

        # Pass 1: Submit all batches
        pending: list[tuple[int, dict[str, float], BatchHandle]] = []
        for ckpt_num, ans_loop_idx in enumerate(ans_loop_indices):
            logger.info(
                "=== [gpt_replay/batch] Pass 1 — submitting checkpoint %d/%d: "
                "ans_loop=%d ===",
                ckpt_num + 1, len(ans_loop_indices), ans_loop_idx,
            )
            checkpoint_metrics = self._download_and_derive_questions(
                gen_output_prefetcher, ans_loop_idx,
            )
            if checkpoint_metrics is None:
                continue
            gen_questions = checkpoint_metrics.pop("_gen_questions")

            batch_output_dir = Path(
                eval_cfg.get("local_cache_dir", ".cache/gen_eval")
            ) / "batch" / str(ans_loop_idx)
            handle = self.api_solver.submit_batch(gen_questions, batch_output_dir)
            if handle is None:
                logger.warning(
                    "No valid questions for batch at ans_loop=%d, skipping.",
                    ans_loop_idx,
                )
                continue
            pending.append((ans_loop_idx, checkpoint_metrics, handle))

        logger.info(
            "gpt_replay/batch: Pass 1 complete — %d batches submitted. "
            "Waiting for results...",
            len(pending),
        )

        # Pass 2: Collect results
        for collect_num, (ans_loop_idx, checkpoint_metrics, handle) in enumerate(pending):
            logger.info(
                "=== [gpt_replay/batch] Pass 2 — collecting checkpoint %d/%d: "
                "ans_loop=%d (batch %d requests) ===",
                collect_num + 1, len(pending), ans_loop_idx, handle.num_requests,
            )
            t0 = time.time()
            gen_scores_dict, api_results = self.api_solver.collect_batch(handle)
            checkpoint_metrics["eval_timing_s/answer_rollout"] = time.time() - t0

            self._finalize_gpt_replay_metrics(
                checkpoint_metrics, gen_scores_dict, api_results,
                api_n, ans_loop_idx, tracker,
            )

    def _download_and_derive_questions(
        self,
        gen_output_prefetcher: GenOutputPrefetcher,
        ans_loop_idx: int,
    ) -> Optional[dict[str, Any]]:
        """Download gen_output.pt, derive questions, compute question metrics.

        Returns a checkpoint_metrics dict with a special ``_gen_questions``
        key containing the derived question list. Returns ``None`` if no
        valid questions are found.
        """
        checkpoint_metrics: dict[str, Any] = {}

        t0 = time.time()
        gen_output = gen_output_prefetcher.get_gen_output(ans_loop_idx)
        checkpoint_metrics["eval_timing_s/gen_output_download"] = time.time() - t0

        gen_questions = derive_gen_questions(gen_output)

        if not gen_questions:
            logger.warning(
                "No valid questions in gen_output.pt at ans_loop=%d, skipping.",
                ans_loop_idx,
            )
            return None

        total_samples = gen_output.batch["responses"].shape[0]
        question_metrics = compute_question_rollout_metrics(
            gen_questions=gen_questions,
            total_samples=total_samples,
            response_mask=gen_output.batch["response_mask"],
            num_documents=len(set(gen_output.non_tensor_batch["doc_id"])),
            prefix="gen_question_rollout",
            reject_reasons=gen_output.non_tensor_batch.get("reject_reason"),
        )
        checkpoint_metrics.update(question_metrics)

        num_valid = len(gen_questions)
        checkpoint_metrics["eval_replay/total_samples"] = float(total_samples)
        checkpoint_metrics["eval_replay/num_valid_questions"] = float(num_valid)
        checkpoint_metrics["eval_replay/num_rejected_questions"] = float(
            total_samples - num_valid
        )

        # Stash questions in metrics dict for caller (popped before logging)
        checkpoint_metrics["_gen_questions"] = gen_questions
        return checkpoint_metrics

    def _finalize_gpt_replay_metrics(
        self,
        checkpoint_metrics: dict[str, Any],
        gen_scores_dict: dict,
        api_results: list,
        api_n: int,
        ans_loop_idx: int,
        tracker: Any,
    ) -> None:
        """Compute answer metrics and log to wandb for one checkpoint."""
        from verl_inf_evolve.gen_eval.api_solver import compute_api_answer_metrics

        answer_metrics = compute_api_answer_metrics(
            scores=gen_scores_dict,
            rollout_n=api_n,
            prefix="gen_answer_rollout",
        )
        checkpoint_metrics.update(answer_metrics)

        num_errors = sum(1 for r in api_results if r.error is not None)
        checkpoint_metrics["eval_replay/api_solver_errors"] = float(num_errors)
        checkpoint_metrics["eval_replay/api_solver_total_calls"] = float(
            len(api_results)
        )

        # Token usage and cost metrics
        if hasattr(self.api_solver, "get_usage_metrics"):
            checkpoint_metrics.update(self.api_solver.get_usage_metrics())
            self.api_solver.reset_usage()

        checkpoint_metrics["eval/ans_loop"] = ans_loop_idx

        key_metrics = {
            "gen_question_rollout/num_valid_questions": checkpoint_metrics.get(
                "gen_question_rollout/num_valid_questions", 0
            ),
            "gen_answer_rollout/accuracy_strict": checkpoint_metrics.get(
                "gen_answer_rollout/accuracy_strict", 0
            ),
        }
        logger.info(
            "ans_loop=%d summary: %s",
            ans_loop_idx,
            {k: f"{v:.4f}" if isinstance(v, float) else v
             for k, v in key_metrics.items()},
        )
        logger.info(
            "ans_loop=%d timing: gen_output_download=%.1fs, api_answer=%.1fs",
            ans_loop_idx,
            checkpoint_metrics.get("eval_timing_s/gen_output_download", 0),
            checkpoint_metrics.get("eval_timing_s/answer_rollout", 0),
        )

        tracker.log(data=checkpoint_metrics, step=ans_loop_idx)

    def _build_remote_backend_kwargs(self) -> dict[str, Any]:
        """Build kwargs for remote backend from config.remote section."""
        remote_cfg = self.config.get("remote", {})
        remote_sync_path = str(
            self.config.gen_eval.get("remote_sync_path", "") or ""
        )

        if remote_sync_path.startswith("hf://"):
            from verl_inf_evolve.storage.remote_backend import build_hf_backend_kwargs

            return build_hf_backend_kwargs(remote_cfg)

        return {}
