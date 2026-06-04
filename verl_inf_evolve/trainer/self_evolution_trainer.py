"""
SelfEvolutionTrainer — the main training loop for V3 self-evolution.

Follows the ``RayPPOTrainer`` pattern (verl/trainer/ppo/ray_trainer.py:222)
but orchestrates **two** worker groups (generator + solver) instead of one,
and implements a nested ans_loop × gen_loop structure matching the v2 pipeline
(inf_evolve/train.py:2114-2260).

All methods are implemented: data pipeline (prepare_dev_batch,
prepare_doc_batch, prepare_question_batch), question parsing
(parse_generated_questions in question_parser), answer scoring
(extract_answer_scores in mcq_utils), SPICE scoring, prepare_gen_update_batch,
aggregate_solver_training_data, compute_influence_scores, save_checkpoint,
_load_checkpoint, and curriculum learning (maybe_refresh_curriculum).
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import time
import random
from collections import defaultdict
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
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.utils.profiler.performance import simple_timer

from verl_inf_evolve.data.batch_utils import messages_to_dataproto
from verl_inf_evolve.data.document_dataset import DocumentDataset
from verl_inf_evolve.sol_eval.benchmark_adapters import (
    build_messages_for_question,
    build_verifier_metadata,
    detect_benchmark_type,
)
from verl_inf_evolve.utils.mcq_utils import (
    extract_answer_scores,
    extract_boxed_answer,
    group_scores_by_qid,
    is_correct,
)
from verl_inf_evolve.utils.prompts import (
    FREE_FORM_QUESTION_GENERATION_PROMPT,
    FREE_FORM_QUESTION_GENERATION_SYSTEM_PROMPT,
    MCQ_QUESTION_GENERATION_PROMPT,
    MCQ_QUESTION_GENERATION_SYSTEM_PROMPT,
    SEEDED_MCQ_QUESTION_GENERATION_PROMPT,
    SEEDED_MCQ_QUESTION_GENERATION_SYSTEM_PROMPT,
    format_seed_examples,
)
from verl_inf_evolve.utils.startup_cleanup import maybe_clear_local_dir_on_start
from verl_inf_evolve.utils.data_utils import (
    derive_gen_questions,
    scatter_for_dispatch,
    stratify_by_doc,
)
from verl_inf_evolve.utils.reward_utils import expand_scores_to_token_level
from verl_inf_evolve.utils.advantage_utils import scalarize_masked_sequence_values
from verl_inf_evolve.utils.generator_reward_utils import (
    build_stage4_reward_payload,
    extract_influence_rewards_for_solver_filter,
    filter_influence_rewards_by_group_std,
    normalize_reward_payload,
    resolve_generator_reward_combination_mode,
    resolve_generator_reward_components,
    resolve_generator_reward_structure,
    validate_generator_reward_components,
    validate_generator_reward_structure,
)
from verl_inf_evolve.utils.question_parser import parse_generated_questions
from verl_inf_evolve.utils.solver_filter_utils import (
    alpha_target_count,
    get_per_doc_keep_count,
    normalize_solver_filter_mode,
)
from verl_inf_evolve.utils.metric_utils import add_distribution_stats
from verl_inf_evolve.utils.influence_utils import (
    add_similarity_metric_stats,
    build_similarity_rewards,
)
from verl_inf_evolve.utils.ppo_diag import (
    collect_process_resource_diag,
    collect_ppo_actor_diag,
    collect_ppo_batch_diag,
    render_diag_map,
)
from verl_inf_evolve.utils.prompt_length_utils import should_filter_by_prompt_length
from verl_inf_evolve.variance_sensitivity.utils import slice_dataproto
from verl_inf_evolve.trainer.resume_state import ResumeState
from verl_inf_evolve.trainer.stage_context import StageContext
from verl_inf_evolve.storage.storage_resolver import StorageResolver
from verl_inf_evolve.storage.stage_upload_manager import StageUploadManager
from verl_inf_evolve.trainer.rollout_metrics import (
    compute_answer_rollout_metrics,
    compute_opener_advantage_metrics,
    compute_opener_rollout_metrics,
    compute_question_rollout_metrics,
    compute_ans_loop_summary_metrics,
)

logger = logging.getLogger(__name__)


def _log_host_memory(label: str) -> None:
    """Log host memory usage from /proc/meminfo and process RSS.

    Intended as a lightweight diagnostic for tracking memory at stage
    transitions to help diagnose OOM / exit-code-137 crashes.
    """
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        rss_gb = rusage.ru_maxrss / (1024 * 1024)  # macOS: bytes, Linux: kB
        # On Linux ru_maxrss is in kB
        import platform
        if platform.system() == "Linux":
            rss_gb = rusage.ru_maxrss / (1024 * 1024)
        else:
            rss_gb = rusage.ru_maxrss / (1024 * 1024 * 1024)
    except Exception:
        rss_gb = -1

    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
        total_gb = meminfo.get("MemTotal", 0) / (1024 * 1024)
        avail_gb = meminfo.get("MemAvailable", 0) / (1024 * 1024)
        used_gb = total_gb - avail_gb
        pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
        logger.info(
            "[mem-diag] %s: host=%.1fGB/%.1fGB (%.0f%%) avail=%.1fGB peak_rss=%.1fGB",
            label, used_gb, total_gb, pct, avail_gb, rss_gb,
        )
    except Exception:
        logger.info("[mem-diag] %s: peak_rss=%.1fGB (/proc/meminfo unavailable)", label, rss_gb)


class _PrefixedActorHandleView:
    """Expose unprefixed rollout methods for a prefixed colocated actor."""

    __slots__ = ("_actor_handle", "_role_prefix")

    def __init__(self, actor_handle: Any, role_prefix: str):
        object.__setattr__(self, "_actor_handle", actor_handle)
        object.__setattr__(self, "_role_prefix", role_prefix)

    def __getattr__(self, name: str):
        # WorkerDict actors expose methods as "<role>_<method>" (e.g.,
        # "generator_get_zeromq_address"). AgentLoop code expects unprefixed
        # names on raw actor handles.
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

    def __reduce__(self):
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


class SelfEvolutionTrainer:
    """Orchestrates generator-solver joint training via Ray worker groups.

    This class owns the training loop and dispatches rollout / update / scoring
    calls to distributed worker groups.  It does NOT inherit from
    ``RayPPOTrainer`` — the control flow is sufficiently different (nested
    loops, two models, influence scoring) that composition is preferred over
    inheritance.

    The constructor and ``init_workers()`` follow the same structure as
    ``RayPPOTrainer`` so that verl utilities (checkpoint engine, resource pool
    manager, worker groups) are reused without modification.

    Args:
        config: Full OmegaConf config.  Expected top-level keys:
            ``generator``, ``solver``, ``training``, ``influence``, ``spice``,
            ``curriculum``, ``wandb``, ``trainer``, ``algorithm``.
        gen_tokenizer: HuggingFace tokenizer for the generator model.
        solver_tokenizer: HuggingFace tokenizer for the solver model.
        role_worker_mapping: ``{"generator": RemoteCls, "solver": RemoteCls}``.
        resource_pool_manager: Manages GPU allocation for worker groups.
        ray_worker_group_cls: The class used to create worker groups
            (default ``RayWorkerGroup``).
    """

    def __init__(
        self,
        config,
        gen_tokenizer,
        solver_tokenizer,
        role_worker_mapping: dict[str, Any],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
    ):
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
        # Note: In verl 0.7.0, FSDP→vLLM weight sync happens automatically
        # inside generate_sequences() when switching to rollout mode.
        # No explicit CheckpointEngineManager needed.

        # Derived from config
        self.num_gen_per_ans = (
            config.training.max_gen_loop // config.training.max_ans_loop
        )

        # Curriculum learning state (populated by _init_datasets if enabled)
        self._data_pool: list[dict] | None = None
        self._curriculum_state: dict[str, Any] | None = None
        # Per-question accuracy history: {qid: [mean_accuracy_per_ans_loop]}
        self._dev_accuracy_history: dict[str, list[float]] = {}
        # Used to create unique AgentLoop worker actor names across managers.
        self._agent_loop_run_id = uuid4().hex[:8]

        # Metric logging step counter (one step = one gen_loop iteration)
        self.gen_steps = 0

        # Whether Stage 1 (dev rollout) is needed.  Can be skipped when
        # influence_rewards is not a reward component and curriculum is off,
        # since dev_output is only consumed by influence scoring and
        # curriculum accuracy tracking.
        self._skip_dev_rollout = bool(
            config.training.get("skip_dev_rollout", False)
        )
        self._dev_only = bool(config.training.get("dev_only", False))
        self._stop_on_checkpoint_upload_failure = bool(
            config.training.get("stop_on_checkpoint_upload_failure", False)
        )
        self._pregenerated_source = str(
            config.training.get("pregenerated_question_source", None) or ""
        ) or None
        if self._pregenerated_source:
            config.training.fix_generator = True
            # Lazily initialized backend for downloading pre-generated questions.
            self._pregenerated_backend = None
        self._last_saved_ans_loop: int | None = None
        self._last_remote_checkpointed_ans_loop: int | None = None
        self._failed_remote_checkpoint_ans_loops: set[int] = set()
        self._pending_ckpt_upload_id: str | None = None
        self._pending_ckpt_marker_upload_id: str | None = None
        self._pending_ckpt_ans_loop: int | None = None

        # True after Phase 1 of compute_influence_scores has run in this
        # process, meaning dev-gradient momentum is live on the workers.
        # Reset to False on process start (including resume after crash).
        self._momentum_live = False
        # Warn once for deprecated solver filtering knobs.
        self._solver_filter_legacy_warned = False

        # ── Preemption / heartbeat tracking ──
        # Mutable state used by the SIGTERM handler and heartbeat logger.
        self._training_progress = {
            "ans_loop": -1,
            "local_gen": -1,
            "stage": "init",
            "last_checkpoint_ans_loop": None,
            "last_checkpoint_time": None,
            "fit_start_time": None,
        }
        self._heartbeat_interval = 60  # seconds
        self._last_heartbeat_time = 0.0
        self._upload_manager: Optional["StageUploadManager"] = None

    # ------------------------------------------------------------------
    # Tracking / logging initialization
    # ------------------------------------------------------------------

    def _init_tracking(self) -> "Tracking":
        """Create the metric tracking logger and configure per-prefix x-axes."""
        from verl.utils.tracking import Tracking

        if self.config.wandb.get("entity", None):
            os.environ.setdefault("WANDB_ENTITY", self.config.wandb.entity)
        if self.config.wandb.get("group_name", None):
            os.environ.setdefault("WANDB_RUN_GROUP", self.config.wandb.group_name)

        from verl_inf_evolve.storage.remote_backend import redact_config_secrets

        tracker = Tracking(
            project_name=self.config.wandb.project_name,
            experiment_name=self.config.wandb.get("run_name", None)
                or self.config.trainer.get("experiment_name", "self-evolution"),
            default_backend=self.config.trainer.get("logger", ["console"]),
            config=redact_config_secrets(OmegaConf.to_container(self.config, resolve=True)),
        )

        # Configure per-prefix x-axes so ans-loop and gen-loop metrics
        # live on independent step timelines in wandb.
        if "wandb" in tracker.logger:
            import wandb

            wandb.define_metric("train/gen_step")
            wandb.define_metric("train/ans_loop")
            # Gen-loop metrics → x-axis is train/gen_step
            wandb.define_metric("gen_question_rollout/*", step_metric="train/gen_step")
            wandb.define_metric("gen_answer_rollout/*", step_metric="train/gen_step")
            # Generator actor update metrics → x-axis is train/gen_step
            wandb.define_metric("gen_actor/*", step_metric="train/gen_step")
            wandb.define_metric("gen_perf/*", step_metric="train/gen_step")
            wandb.define_metric("gen_rollout_corr/*", step_metric="train/gen_step")
            wandb.define_metric("gen_quant/*", step_metric="train/gen_step")
            # Generator batch-level data metrics (advantages, rewards, lengths)
            wandb.define_metric("gen_critic/*", step_metric="train/gen_step")
            wandb.define_metric("gen_response_length/*", step_metric="train/gen_step")
            wandb.define_metric("gen_response_length_non_aborted/*", step_metric="train/gen_step")
            wandb.define_metric("gen_prompt_length/*", step_metric="train/gen_step")
            wandb.define_metric("gen_response/*", step_metric="train/gen_step")
            # Ans-loop metrics → x-axis is train/ans_loop
            wandb.define_metric("dev_rollout/*", step_metric="train/ans_loop")
            wandb.define_metric("ans_loop/*", step_metric="train/ans_loop")
            # Solver actor update metrics → x-axis is train/ans_loop
            wandb.define_metric("solver_actor/*", step_metric="train/ans_loop")
            wandb.define_metric("solver_perf/*", step_metric="train/ans_loop")
            wandb.define_metric("solver_rollout_corr/*", step_metric="train/ans_loop")
            # Solver batch-level data metrics (advantages, rewards, lengths)
            wandb.define_metric("solver_critic/*", step_metric="train/ans_loop")
            wandb.define_metric("solver_response_length/*", step_metric="train/ans_loop")
            wandb.define_metric("solver_response_length_non_aborted/*", step_metric="train/ans_loop")
            wandb.define_metric("solver_prompt_length/*", step_metric="train/ans_loop")
            wandb.define_metric("solver_response/*", step_metric="train/ans_loop")
            # Timing metrics
            wandb.define_metric("gen_timing_s/*", step_metric="train/gen_step")
            wandb.define_metric("ans_timing_s/*", step_metric="train/ans_loop")
            # In-training benchmark evaluation metrics
            wandb.define_metric("benchmark_eval/*", step_metric="train/ans_loop")

            # Log key metadata as top-level summary fields for easy visibility
            import sys

            remote_sync_path = self.config.training.get("remote_sync_path", None)
            if remote_sync_path:
                wandb.run.summary["remote_sync_path"] = remote_sync_path

            skypilot_task_id = os.environ.get("SKYPILOT_TASK_ID")
            if skypilot_task_id:
                wandb.run.summary["skypilot_task_id"] = skypilot_task_id

            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            if slurm_job_id:
                wandb.run.summary["slurm_job_id"] = slurm_job_id

            launch_command = " ".join(sys.argv)
            if launch_command:
                wandb.run.summary["launch_command"] = launch_command

        return tracker

    def _init_benchmark_evaluator(self) -> None:
        """Create the in-training benchmark evaluator if enabled."""
        self._benchmark_evaluator = None
        if not self.config.get("benchmark_eval", {}).get("enabled", False):
            return

        from verl_inf_evolve.trainer.benchmark_eval import (
            InTrainingBenchmarkEvaluator,
            benchmark_output_exists,
            persist_benchmark_output,
        )

        self._benchmark_evaluator = InTrainingBenchmarkEvaluator(
            config=self.config.benchmark_eval,
            solver_rollout_manager=self.solver_rollout_manager,
            solver_tokenizer=self.solver_tokenizer,
            solver_num_workers=self.config.solver.rollout.agent.num_workers,
            max_prompt_tokens=self.config.solver.rollout.prompt_length,
            max_ans_loop=self.config.training.max_ans_loop,
            generate_fn=self._generate_with_metadata,
            on_benchmark_output_fn=lambda ans_loop, benchmark_name, benchmark_output: persist_benchmark_output(
                upload_manager=self._upload_manager,
                ans_loop=ans_loop,
                benchmark_name=benchmark_name,
                benchmark_output=benchmark_output,
            ),
            benchmark_output_exists_fn=lambda ans_loop, benchmark_name: benchmark_output_exists(
                upload_manager=self._upload_manager,
                ans_loop=ans_loop,
                benchmark_name=benchmark_name,
                resume_dir=os.path.join(self.config.training.default_local_dir, f"ans_{ans_loop}"),
            ),
        )

    # ------------------------------------------------------------------
    # Worker initialization
    # ------------------------------------------------------------------

    def init_workers(self):
        """Initialize Ray worker groups for generator and solver.

        Follows ``RayPPOTrainer.init_workers()`` (ray_trainer.py:740):

        1. Create resource pools
        2. Create ``RayClassWithInitArgs`` for each worker role
        3. If sharing GPUs, use ``create_colocated_worker_cls()``
        4. Spawn worker groups and call ``init_model()``
        5. Create ``CheckpointEngineManager`` for each model

        After this method returns, ``self.generator_wg`` and
        ``self.solver_wg`` are ready for ``generate_sequences()`` and
        ``update_actor()`` calls.
        """
        maybe_clear_local_dir_on_start(self.config)
        self.resource_pool_manager.create_resource_pool()

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
            # Separate pools: create non-colocated worker groups directly
            self.generator_wg = self.ray_worker_group_cls(
                resource_pool=gen_resource_pool,
                ray_cls_with_init=gen_cls,
            )
            self.solver_wg = self.ray_worker_group_cls(
                resource_pool=solver_resource_pool,
                ray_cls_with_init=solver_cls,
            )
        else:
            # Shared pool: colocate both roles on the same actors
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
        # init_model() must complete first — it initializes vLLMAsyncRollout's
        # ZMQ listeners that the HTTP servers connect to.
        from verl.experimental.agent_loop import AgentLoopManager

        gen_alm_worker_group = self.generator_wg
        solver_alm_worker_group = self.solver_wg
        if shared_pool:
            # In shared-pool colocated mode, worker actor methods are prefixed
            # (e.g., generator_get_zeromq_address). AgentLoopManager passes raw
            # actor handles into RolloutReplica, which expects unprefixed names.
            # Use a lightweight view that remaps method lookups.
            gen_alm_worker_group = _AgentLoopWorkerGroupView(
                self.generator_wg, "generator"
            )
            solver_alm_worker_group = _AgentLoopWorkerGroupView(
                self.solver_wg, "solver"
            )

        self._shared_engine_mode = shared_pool
        if self._shared_engine_mode:
            self._validate_shared_engine_compatibility()
            self._validate_rollout_sharing()
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
    # Dataset initialization
    # ------------------------------------------------------------------

    def _init_datasets(self):
        """Load dev questions, documents, and create the document dataset.

        Called once at the start of ``fit()`` before the training loop.
        Performs three steps:

        1. **Curriculum init** (optional): If ``config.curriculum.enabled``,
           calls ``_init_curriculum()`` which loads a data pool from disk,
           resumes or samples the initial dev set, and redirects
           ``config.training.dev_dataset_path`` to ``curriculum_dev.json``.
        2. **Dev questions**: Loads the dev question set from
           ``config.training.dev_dataset_path`` (JSON list). When curriculum
           is active, this path already points to the curriculum-managed file.
        3. **Document dataset**: Loads documents from
           ``config.training.documents_path`` into a dict keyed by ``doc_id``,
           then creates a :class:`DocumentDataset` iterator that yields
           shuffled batches of ``doc_batch_size`` document IDs.

        Sets instance attributes:
            self.dev_questions: list[dict] — dev question dicts.
            self.documents: dict[str, dict] — documents keyed by doc_id.
            self.doc_dataset: DocumentDataset — batched document ID iterator.
        """
        # --- Validate skip_dev_rollout safety ---
        if self._skip_dev_rollout:
            reward_components = resolve_generator_reward_components(self.config.training)
            if "influence_rewards" in reward_components:
                raise ValueError(
                    "training.skip_dev_rollout=true is incompatible with "
                    "influence_rewards in generator_reward_components. "
                    "Dev rollout is required for influence scoring."
                )
            if self.config.curriculum.enabled:
                raise ValueError(
                    "training.skip_dev_rollout=true is incompatible with "
                    "curriculum.enabled=true. Dev rollout is required for "
                    "curriculum accuracy tracking."
                )
            logger.info("skip_dev_rollout=true: Stage 1 (dev rollout) will be skipped")

        # --- Validate dev_only mode ---
        if self._dev_only:
            if self._skip_dev_rollout:
                raise ValueError(
                    "training.dev_only=true is incompatible with "
                    "training.skip_dev_rollout=true. Dev rollout IS the "
                    "training data source in dev_only mode."
                )
            if self.config.training.fix_answer_model:
                raise ValueError(
                    "training.dev_only=true is incompatible with "
                    "training.fix_answer_model=true. There is no model to train."
                )
            self.config.training.fix_generator = True
            logger.info(
                "dev_only=true: Stages 2-5 (generator loop) will be skipped. "
                "Solver trains directly on dev data. fix_generator forced to True."
            )

        # --- Curriculum learning initialization ---
        if self.config.curriculum.enabled:
            self._init_curriculum()

        # --- Dev questions ---
        with open(self.config.training.dev_dataset_path, "r") as f:
            self.dev_questions = json.load(f)
        logger.info("Loaded %d dev questions from %s",
                     len(self.dev_questions), self.config.training.dev_dataset_path)

        # --- Question source: documents or seeded ---
        source_mode = getattr(self.config.training, "question_source_mode", "document")

        if source_mode == "seeded_dev":
            self._init_seed_dataset()
        else:
            # --- Documents (default) ---
            with open(self.config.training.documents_path, "r") as f:
                documents_list = json.load(f)
            self.documents = {str(doc["doc_id"]): doc for doc in documents_list}

            all_doc_ids = [str(doc["doc_id"]) for doc in documents_list]
            self.doc_dataset = DocumentDataset(
                all_doc_ids,
                batch_size=self.config.training.doc_batch_size,
                shuffle=True,
                seed=self.config.training.seed,
                repeat_batch=getattr(self.config.training, "repeat_doc_batch", False),
            )
            logger.info("Loaded %d documents, doc_batch_size=%d, seed=%s, first_5_doc_ids=%s",
                         len(self.documents), self.config.training.doc_batch_size,
                         self.config.training.seed,
                         self.doc_dataset.doc_ids[:5])
            self.seed_questions = None
            self.seed_prompt_bundles = None

            # --- Freeform-shortcut partition ---
            self._init_freeform_shortcut(documents_list)
            self._init_question_generation_routes()

    def _init_freeform_shortcut(self, documents_list: list[dict]) -> None:
        """Build the doc_id → free-form question metadata map for shortcut routing.

        When ``training.freeform_shortcut.enabled`` is True, every doc whose
        ``source_pdf`` matches any pattern in ``source_patterns`` is parsed
        via ``content_pattern`` (must define named groups ``question`` and
        ``answer``). Successfully parsed docs are stored in
        ``self._shortcut_doc_meta`` keyed by ``doc_id``; downstream stages
        skip the generator for these docs and inject the parsed Q+A directly
        as free-form solver-rollout questions.
        """
        self._shortcut_doc_meta: dict[str, dict] = {}
        cfg = self.config.training.get("freeform_shortcut", None)
        if cfg is None or not bool(cfg.get("enabled", False)):
            return

        patterns = cfg.get("source_patterns", []) or []
        if not patterns:
            logger.warning(
                "freeform_shortcut.enabled=true but source_patterns is empty; "
                "no docs will be routed."
            )
            return

        src_regexes = [re.compile(p) for p in patterns]
        content_re = re.compile(cfg.content_pattern, re.DOTALL)
        data_source = str(cfg.get("data_source", "") or "")
        benchmark_type = str(cfg.get("benchmark_type", "qa_open"))
        solver_max_prompt = self.config.solver.rollout.prompt_length

        n_matched_src = 0
        n_parse_failed = 0
        n_too_long = 0
        for doc in documents_list:
            src = str(doc.get("source_pdf", ""))
            if not any(r.search(src) for r in src_regexes):
                continue
            n_matched_src += 1
            m = content_re.match(doc.get("content", ""))
            if not m:
                n_parse_failed += 1
                continue
            question_text = m.group("question").strip()
            ground_truth = m.group("answer").strip()

            # Apply solver prompt-length filter at init: shortcut docs go
            # straight to the solver rollout, so the relevant cap is
            # solver.rollout.prompt_length (NOT generator.rollout.prompt_length).
            # One-time check — content is fixed per doc.
            check_q = {
                "question_text": question_text,
                "choices": [],
                "ground_truth": ground_truth,
                "data_source": data_source,
                "benchmark_type": benchmark_type,
            }
            messages, _, _ = build_messages_for_question(check_q)
            if should_filter_by_prompt_length(
                tokenizer=self.solver_tokenizer,
                messages=messages,
                max_prompt_tokens=solver_max_prompt,
                logger=logger,
                sample_kind="shortcut_doc",
                sample_id=str(doc["doc_id"]),
            ):
                n_too_long += 1
                continue

            self._shortcut_doc_meta[str(doc["doc_id"])] = {
                "question_text": question_text,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "benchmark_type": benchmark_type,
            }

        logger.info(
            "freeform_shortcut: routing %d docs "
            "(matched_source=%d, parse_failed=%d, too_long=%d, "
            "solver_max_prompt=%d) with data_source=%r benchmark_type=%r",
            len(self._shortcut_doc_meta),
            n_matched_src,
            n_parse_failed,
            n_too_long,
            solver_max_prompt,
            data_source,
            benchmark_type,
        )

    def _build_shortcut_gen_questions(self, shortcut_doc_ids: list[str]) -> list[dict]:
        """Build free-form question dicts to inject after Stage 2.

        One question per shortcut doc — solver_n rollouts are added later
        by the existing ``gen_q_batch.repeat(solver_n, interleave=True)``
        at the Stage 3 boundary, matching the dev-rollout pattern.

        Each dict carries ``is_shortcut=True`` so downstream stages can
        identify shortcut rows and exclude them from influence scoring
        while still routing them through the solver-PPO update.
        """
        out: list[dict] = []
        for doc_id in shortcut_doc_ids:
            meta = self._shortcut_doc_meta.get(str(doc_id))
            if meta is None:
                logger.warning(
                    "_build_shortcut_gen_questions: missing meta for doc_id=%s; skipping",
                    doc_id,
                )
                continue
            out.append({
                "question_id": f"shortcut:{doc_id}",
                "question_text": meta["question_text"],
                "choices": [],
                "ground_truth": meta["ground_truth"],
                "doc_id": str(doc_id),
                "is_mcq": False,
                "benchmark_type": meta["benchmark_type"],
                "data_source": meta["data_source"],
                "is_shortcut": True,
                "parsed_ok": True,
            })
        return out

    def _init_question_generation_routes(self) -> None:
        """Compile source-pattern routes for document-conditioned generation.

        The default route remains MCQ generation.  Optional
        ``training.question_generation_routes`` entries can redirect matching
        documents (matched against ``doc['source_pdf']``) to the free-form
        generator prompt, which then flows through the existing mixed
        MCQ/free-form parser, solver prompt adapter, and verifier dispatch.
        """
        self._question_generation_routes: list[dict[str, Any]] = []
        routes = self.config.training.get("question_generation_routes", []) or []
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
        """Return ``mcq`` or ``free_form`` for a document generation prompt.

        An explicit per-doc ``prompt_type`` (set by a gradeability tagging pass)
        takes priority over ``source_pdf`` pattern routing; untagged docs fall
        back to the route table, then the ``mcq`` default.
        """
        explicit = str(doc.get("prompt_type", "")).strip().lower()
        if explicit in {"mcq", "free_form"}:
            return explicit
        source_pdf = str(doc.get("source_pdf", ""))
        for route in getattr(self, "_question_generation_routes", []) or []:
            if any(regex.search(source_pdf) for regex in route["source_regexes"]):
                return route["prompt_type"]
        return "mcq"

    def _init_seed_dataset(self) -> None:
        """Initialize the seed question dataset and precompute few-shot bundles.

        Called from ``_init_datasets()`` when ``question_source_mode == "seeded_dev"``.
        Loads seed questions, builds deterministic few-shot bundles keyed by
        anchor question ID, and creates a ``DocumentDataset`` iterator over
        anchor IDs (reusing the same batching/shuffle/resume infrastructure).

        Sets:
            self.seed_questions: dict mapping question_id -> question dict
            self.seed_prompt_bundles: dict mapping anchor_qid -> list[dict]
            self.doc_dataset: DocumentDataset over anchor question IDs
            self.documents: empty dict (not used in seeded mode)
        """
        seed_path = self.config.training.seed_dataset_path
        if not seed_path:
            raise ValueError(
                "question_source_mode='seeded_dev' requires training.seed_dataset_path "
                "to be set (path to a JSON list of seed MCQ questions)."
            )

        with open(seed_path, "r") as f:
            seed_list = json.load(f)

        k = getattr(self.config.training, "seed_examples_per_prompt", 4)

        self.seed_questions = {
            str(q["question_id"]): q for q in seed_list
        }
        all_qids = [str(q["question_id"]) for q in seed_list]

        # Build deterministic few-shot bundles: for each anchor, pick k-1 others
        # as demonstration examples, then include the anchor itself.
        rng = random.Random(self.config.training.seed)
        self.seed_prompt_bundles = {}
        for anchor_qid in all_qids:
            others = [qid for qid in all_qids if qid != anchor_qid]
            n_others = min(k - 1, len(others))
            chosen = rng.sample(others, k=n_others)
            bundle = [self.seed_questions[qid] for qid in chosen]
            # The anchor is used to seed the domain but is NOT shown as an
            # example — this reduces the chance the model copies it verbatim.
            self.seed_prompt_bundles[anchor_qid] = bundle

        # Reuse DocumentDataset for anchor IDs — gives us batching, shuffle,
        # epoch tracking, and resume state for free.
        self.doc_dataset = DocumentDataset(
            all_qids,
            batch_size=self.config.training.doc_batch_size,
            shuffle=True,
            seed=self.config.training.seed,
            repeat_batch=getattr(self.config.training, "repeat_doc_batch", False),
        )
        self.documents = {}  # not used in seeded mode

        logger.info(
            "Seeded mode: loaded %d seed questions from %s, "
            "bundle_size=%d (k-1 examples per anchor), doc_batch_size=%d",
            len(self.seed_questions), seed_path, k - 1,
            self.config.training.doc_batch_size,
        )

    def _init_curriculum(self) -> None:
        """Initialize curriculum learning state and data pool.

        Loads the data pool, checks for existing curriculum state (from a
        previous run), and if none exists, samples the initial dev set.

        Reference: V2 ``train.py:1238-1288`` (initialize_curriculum_dev_set)
                   V2 ``train.py:2063-2078`` (curriculum init in main loop)
        """
        local_dir = self.config.training.default_local_dir
        state_path = os.path.join(local_dir, "curriculum_state.json")
        curriculum_dev_path = os.path.join(local_dir, "curriculum_dev.json")

        # Load data pool
        pool_path = self.config.curriculum.data_pool_path
        with open(pool_path, "r") as f:
            self._data_pool = json.load(f)
        logger.info("Curriculum: loaded data pool with %d questions from %s",
                     len(self._data_pool), pool_path)

        # Check for existing curriculum state (resume case).
        # Try local files first; if missing, fall back to R2 download.
        if not (os.path.exists(state_path) and os.path.exists(curriculum_dev_path)):
            self._maybe_download_curriculum_from_r2(
                local_dir, state_path, curriculum_dev_path
            )

        if os.path.exists(state_path) and os.path.exists(curriculum_dev_path):
            with open(state_path, "r") as f:
                self._curriculum_state = json.load(f)
            self._curriculum_state["used_question_ids"] = set(
                self._curriculum_state.get("used_question_ids", [])
            )
            # Point dev_dataset_path to curriculum dev set
            self.config.training.dev_dataset_path = curriculum_dev_path
            # Restore accuracy history
            self._dev_accuracy_history = self._curriculum_state.get(
                "accuracy_history", {}
            )
            logger.info("Curriculum: resumed from state (last_refresh=%d)",
                         self._curriculum_state["last_refresh_ans_loop"])
        else:
            # Initialize fresh curriculum
            num_to_sample = min(
                self.config.curriculum.num_dev_questions, len(self._data_pool)
            )
            random.seed(self.config.training.seed)
            sampled_dev = random.sample(self._data_pool, num_to_sample)
            logger.info("Curriculum: fresh init with seed=%s, sampled %d/%d dev questions, first_5_qids=%s",
                         self.config.training.seed, num_to_sample, len(self._data_pool),
                         [q["question_id"] for q in sampled_dev[:5]])

            # Save curriculum dev set
            os.makedirs(local_dir, exist_ok=True)
            with open(curriculum_dev_path, "w") as f:
                json.dump(sampled_dev, f, indent=2)

            self._curriculum_state = {
                "used_question_ids": set(),
                "last_refresh_ans_loop": -1,
                "current_dev_question_ids": [
                    q["question_id"] for q in sampled_dev
                ],
            }
            self._save_curriculum_state()

            # Point dev_dataset_path to curriculum dev set
            self.config.training.dev_dataset_path = curriculum_dev_path
            logger.info("Curriculum: initialized with %d questions", len(sampled_dev))

    def record_dev_accuracy(
        self, dev_output: DataProto, ans_loop: int
    ) -> None:
        """Record per-question dev accuracy into ``_dev_accuracy_history``.

        Called after each Stage 1 dev rollout. For each question, computes
        mean accuracy across its ``rollout_n`` responses (treating ``None``
        extraction failures as ``0.0``) and appends to the history list.

        No-op when ``curriculum.enabled`` is ``False``.

        The accumulated history is consumed by ``maybe_refresh_curriculum()``
        to identify mastered questions and swap them for harder ones.

        Args:
            dev_output: DataProto with ``answer_score`` populated in
                ``non_tensor_batch`` by ``extract_answer_scores()``.
            ans_loop: Current answer loop index (unused, reserved for
                future logging).
        """
        if not self.config.curriculum.enabled:
            return

        dev_scores = group_scores_by_qid(dev_output)
        for qid, scores_list in dev_scores.items():
            # Treat None (extraction failure) as 0.0 — failed extraction means wrong answer
            resolved = [s if s is not None else 0.0 for s in scores_list]
            mean_acc = sum(resolved) / len(resolved)
            self._dev_accuracy_history.setdefault(qid, []).append(mean_acc)

    def _checkpoint_policy_for_ans_loop(self, ans_loop: int) -> tuple[bool, bool | None]:
        """Return ``(should_save_ckpt, is_hf_keep_step)`` for an answer loop."""
        save_interval = self.config.training.save_every_n_steps
        is_last_step = ans_loop == self.config.training.max_ans_loop - 1

        if self.config.training.always_save_for_resume:
            return True, (
                (save_interval > 0 and ans_loop % save_interval == 0) or is_last_step
            )

        return (ans_loop % save_interval == 0) or is_last_step, None

    def _should_upload_stage_context_for_ans_loop(self, ans_loop: int) -> bool:
        """Upload stage context only on the configured save window."""
        save_interval = self.config.training.save_every_n_steps
        return save_interval > 0 and ans_loop % save_interval in (0, 1)

    # ------------------------------------------------------------------
    # Stage context helpers
    # ------------------------------------------------------------------

    def stage_ctx(
        self,
        name: str,
        stage_id: int,
        resume: ResumeState,
        resume_dir: str,
        is_done,
        mark_done,
        ans_loop: int,
        defer_state_update: bool = False,
        timing_dict: dict[str, float] | None = None,
        timer_name: str | None = None,
    ) -> StageContext:
        """Create a ``StageContext`` for an ans-level stage."""
        return StageContext(
            name=name,
            stage_id=stage_id,
            resume=resume,
            resume_dir=resume_dir,
            upload_manager=self._upload_manager,
            is_done=is_done,
            mark_done=mark_done,
            ans_loop=ans_loop,
            defer_state_update=defer_state_update,
            should_upload_remote=self._should_upload_stage_context_for_ans_loop(ans_loop),
            timing_dict=timing_dict,
            timer_name=timer_name,
        )

    def gen_stage_ctx(
        self,
        name: str,
        stage_id: int,
        resume: ResumeState,
        resume_dir: str,
        local_gen: int,
        is_done,
        mark_done,
        ans_loop: int,
        defer_state_update: bool = False,
        timing_dict: dict[str, float] | None = None,
        timer_name: str | None = None,
    ) -> StageContext:
        """Create a ``StageContext`` for a gen-loop stage."""
        return StageContext(
            name=name,
            stage_id=stage_id,
            resume=resume,
            resume_dir=resume_dir,
            upload_manager=self._upload_manager,
            is_done=is_done,
            mark_done=mark_done,
            ans_loop=ans_loop,
            gen_loop_prefix=f"gen_{local_gen}",
            defer_state_update=defer_state_update,
            should_upload_remote=self._should_upload_stage_context_for_ans_loop(ans_loop),
            timing_dict=timing_dict,
            timer_name=timer_name,
        )

    # ------------------------------------------------------------------
    # Resume state helpers
    # ------------------------------------------------------------------

    def _load_or_create_resume_state(
        self,
        ans_loop: int,
        resume_dir: str,
        save_interval: int,
    ) -> "ResumeState":
        """Load, validate, and return a ResumeState for *ans_loop*.

        Stale or mismatched states are discarded automatically.  When no
        valid state is found a fresh one is created.
        """
        logger.info(
            "Loading resume state for ans_loop=%d from %s ...", ans_loop, resume_dir
        )
        resume_resolver = StorageResolver(
            local_base=resume_dir,
            upload_manager=self._upload_manager,
            remote_prefix=f"ans_{ans_loop}",
            resolve_order="local_first",
        )
        resume = ResumeState.load(resume_dir, resolver=resume_resolver)

        # ── Staleness checks ──
        if resume is not None and resume.ans_loop != ans_loop:
            logger.warning(
                "Stale resume state (ans_loop=%d, expected=%d) — discarding",
                resume.ans_loop, ans_loop,
            )
            ResumeState.clear(resume_dir)
            resume = None
        if resume is not None and resume.num_gen_per_ans != self.num_gen_per_ans:
            logger.warning(
                "Resume state num_gen_per_ans mismatch (%d vs %d) — discarding",
                resume.num_gen_per_ans, self.num_gen_per_ans,
            )
            ResumeState.clear(resume_dir)
            resume = None
        # When checkpoints are not saved every loop (save_every_n_steps > 1),
        # state.json for non-checkpointed loops becomes stale on resume:
        # the model weights come from the last checkpoint
        # (start_ans_loop - 1), not from the loop that produced the
        # state.json.  Discard them so all stages re-run with the correct
        # checkpoint weights.
        #
        # When save_every_n_steps == 1 (or continuous mode), every loop
        # has a checkpoint, so start_ans_loop - 1 is always the
        # immediately previous loop and no state is stale.
        if (
            resume is not None
            and save_interval > 1
            and ans_loop >= self._start_ans_loop
            and self._start_ans_loop > 0
        ):
            logger.warning(
                "Interval mode: discarding stale resume state for "
                "ans_loop=%d (checkpoint was at step %d)",
                ans_loop, self._start_ans_loop - 1,
            )
            ResumeState.clear(resume_dir)
            resume = None

        # ── Create or log ──
        if resume is None:
            resume = ResumeState(
                ans_loop=ans_loop, num_gen_per_ans=self.num_gen_per_ans
            )
            print(
                f"  No existing resume state — starting ans_loop={ans_loop} fresh",
                flush=True,
            )
        else:
            self._log_resume_summary(resume, ans_loop)

        return resume

    @staticmethod
    def _log_resume_summary(resume: "ResumeState", ans_loop: int) -> None:
        """Print a human-readable summary of a loaded resume state."""
        completed_gen_loops = [
            gl.local_gen for gl in resume.gen_loops if gl.stage_5_done
        ]
        partial_gen_loops = [
            gl.local_gen for gl in resume.gen_loops if not gl.stage_5_done
        ]
        stage_summary = []
        if resume.stage_0_done:
            stage_summary.append("curriculum_refresh(done)")
        if resume.stage_1_done:
            stage_summary.append("dev_rollout(done)")
        if resume.stage_6_done:
            stage_summary.append("solver_ppo_update(done)")
        stages_str = ", ".join(stage_summary) if stage_summary else "(none)"
        completed_str = str(completed_gen_loops) if completed_gen_loops else "(none)"
        print(
            f"========== RESUME STATE LOADED | ans_loop={ans_loop} ==========",
            flush=True,
        )
        print(f"  Ans-level stages done: {stages_str}", flush=True)
        print(f"  Gen loops completed: {completed_str}", flush=True)
        if partial_gen_loops:
            for gl in resume.gen_loops:
                if not gl.stage_5_done:
                    stages_done = []
                    if gl.stage_2_done:
                        stages_done.append("question_gen")
                    if gl.stage_3_done:
                        stages_done.append("gen_answer_rollout")
                    if gl.stage_4_done:
                        stages_done.append("scoring")
                    done_str = ", ".join(stages_done) if stages_done else "none"
                    print(
                        f"  Gen loop {gl.local_gen} (partial): stages done = [{done_str}]",
                        flush=True,
                    )
        if resume.doc_dataset_state is not None:
            print(
                f"  Doc dataset state: position={resume.doc_dataset_state['position']}, "
                f"epoch={resume.doc_dataset_state['epoch']}",
                flush=True,
            )
        print(
            "==================================================",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self):
        """Run the nested ans_loop × gen_loop training loop.

        Structure mirrors v2 (train.py:2114-2260) with verl-native calls:

        .. code-block:: text

            for ans_loop:
                maybe_refresh_curriculum()
                dev_output = solver_wg.generate_sequences(dev_batch)     # Stage 1
                for gen_loop:
                    gen_output = generator_wg.generate_sequences(doc_batch)   # Stage 2
                    gen_answer = solver_wg.generate_sequences(gen_q_batch)    # Stage 3
                    rewards = compute_rewards(...)                            # Stage 4
                    generator_wg.update_actor(gen_batch)                      # Stage 5
                solver_wg.update_actor(solver_batch)                         # Stage 6
        """
        # --- Resolve HF namespace placeholder / token pool before auth ---
        remote_sync_path = self.config.training.get("remote_sync_path", None)
        if remote_sync_path and str(remote_sync_path).startswith("hf://"):
            self._resolve_hf_remote_sync_path()
            remote_sync_path = self.config.training.get("remote_sync_path", None)
            self._bootstrap_hf_token()

        # Also resolve __namespace__ in pregenerated_question_source
        if self._pregenerated_source and str(self._pregenerated_source).startswith("hf://"):
            self._resolve_pregenerated_source()

        # --- Initialize upload manager (before datasets/checkpoint so remote
        #     fallback is available for curriculum resume and checkpoint
        #     downloads) ---
        max_pending_gb = self.config.training.get("max_pending_upload_gb", 100.0)
        disable_upload = self.config.training.get("disable_remote_upload", False)
        remote_cfg = self.config.get("remote", {})
        backend_kwargs: dict[str, Any] = {}
        if remote_sync_path and str(remote_sync_path).startswith("hf://"):
            from verl_inf_evolve.storage.remote_backend import build_hf_backend_kwargs

            backend_kwargs = build_hf_backend_kwargs(
                remote_cfg,
                auto_create_repo=not disable_upload,
            )
        self._upload_manager = StageUploadManager(
            remote_sync_path=remote_sync_path,
            max_pending_gb=max_pending_gb,
            disable_upload=disable_upload,
            backend_kwargs=backend_kwargs,
            checkpoint_failure_callback=self._handle_checkpoint_upload_failure,
        )
        self._upload_manager.start()

        # --- Write run_metadata.json to remote root for eval auto-detection ---
        self._write_run_metadata()

        # Load datasets
        self._init_datasets()

        # Load any existing checkpoint and sync weights to vLLM
        start_ans_loop = self._load_checkpoint()
        # Optional one-time role-specific initialization checkpoints.
        # This is skipped automatically when start_ans_loop > 0 so the
        # existing resume path remains unchanged.
        self._maybe_load_initial_role_checkpoints(start_ans_loop=start_ans_loop)

        # --- Restore resume state ---
        self.gen_steps = start_ans_loop * self.num_gen_per_ans
        # Track the most recently saved checkpoint for cleanup.
        # On resume, start_ans_loop = last_saved + 1, so the previous
        # checkpoint is at start_ans_loop - 1.  Fresh start → None.
        self._last_saved_ans_loop = (start_ans_loop - 1) if start_ans_loop > 0 else None
        self._last_remote_checkpointed_ans_loop = (
            start_ans_loop - 1 if start_ans_loop > 0 else None
        )
        self._failed_remote_checkpoint_ans_loops = set()
        self._pending_ckpt_upload_id = None
        self._pending_ckpt_marker_upload_id = None
        self._pending_ckpt_ans_loop = None
        self._start_ans_loop = start_ans_loop

        # --- Initialize metric tracking ---
        tracking_logger = self._init_tracking()

        # --- Initialize in-training benchmark evaluator ---
        self._init_benchmark_evaluator()

        # --- Training config ---
        save_interval = self.config.training.save_every_n_steps

        # ── Install preemption handler ──
        self._training_progress["fit_start_time"] = time.time()
        self._install_sigterm_handler()

        logger.info(
            "========== TRAINING START | ans_loop_range=[%d, %d) "
            "| num_gen_per_ans=%d "
            "| save_every_n_steps=%d "
            "| always_save_for_resume=%s ==========",
            start_ans_loop, self.config.training.max_ans_loop,
            self.num_gen_per_ans,
            save_interval,
            self.config.training.always_save_for_resume,
        )

        for ans_loop in range(start_ans_loop, self.config.training.max_ans_loop):
            self._update_progress(ans_loop=ans_loop, local_gen=-1, stage="ans_loop_start")
            logger.info("========== ANS LOOP START | ans_loop=%d ==========", ans_loop)

            # Each ans_loop gets its own directory (mirrors R2 layout)
            resume_dir = os.path.join(
                self.config.training.default_local_dir, f"ans_{ans_loop}"
            )

            resume = self._load_or_create_resume_state(
                ans_loop, resume_dir, save_interval,
            )

            ans_timing_raw: dict[str, float] = {}
            _ans_loop_start = time.perf_counter()

            # ================================================================
            # Stage 0: Curriculum refresh
            # ================================================================
            self._update_progress(stage="stage0_curriculum")
            self._maybe_log_heartbeat()
            with self.stage_ctx(
                name="curriculum_refresh", stage_id=0, resume=resume,
                resume_dir=resume_dir,
                is_done=lambda: resume.stage_0_done,
                mark_done=lambda: setattr(resume, 'stage_0_done', True),
                ans_loop=ans_loop,
                timing_dict=ans_timing_raw,
            ) as ctx0:
                if ctx0.should_run:
                    self.maybe_refresh_curriculum(ans_loop)

            _log_host_memory("stage1-start")
            self._update_progress(stage="stage1_dev_rollout")
            self._maybe_log_heartbeat()
            # ================================================================
            # Stage 1: Dev answer rollout
            # ================================================================
            solver_n = self.config.solver.rollout.n
            solver_num_workers = self.config.solver.rollout.agent.num_workers

            if self._skip_dev_rollout:
                # Mark stage as done for resume consistency, skip rollout entirely.
                # dev_output is only needed for influence scoring and curriculum
                # tracking, both of which are validated as off in _init_datasets.
                resume.stage_1_done = True
                dev_output = None
                logger.info("  [Stage 1] dev_rollout SKIPPED (skip_dev_rollout=true)")
                # Still flush any timing from earlier stages (e.g. curriculum_refresh)
                early_timing = {f"ans_timing_s/{k}": ans_timing_raw.pop(k) for k in list(ans_timing_raw)}
                if early_timing:
                    tracking_logger.log({"train/ans_loop": ans_loop, **early_timing}, step=self.gen_steps)
            else:
                with self.stage_ctx(
                    name="dev_rollout", stage_id=1, resume=resume,
                    resume_dir=resume_dir,
                    is_done=lambda: resume.stage_1_done,
                    mark_done=lambda: setattr(resume, 'stage_1_done', True),
                    ans_loop=ans_loop,
                    timing_dict=ans_timing_raw,
                ) as ctx1:
                    if ctx1.should_run:
                        dev_questions = self._select_dev_questions_for_ans_loop(ans_loop)
                        dev_batch = self.prepare_dev_batch(dev_questions=dev_questions)
                        dev_batch = dev_batch.repeat(repeat_times=solver_n, interleave=True) # interleave to ensure same question's rollouts are adjacent
                        dev_batch.meta_info["progress_label"] = f"stage1/dev_rollout ans_loop={ans_loop}"
                        logger.info("  [Stage 1] _generate_with_metadata (solver rollout, n=%d, samples=%d) ...", solver_n, len(dev_batch))
                        dev_output = self._generate_with_metadata(
                            self.solver_rollout_manager, dev_batch, solver_num_workers
                        )
                        logger.info("  [Stage 1] _generate_with_metadata done")
                        extract_answer_scores(dev_output, self.solver_tokenizer)
                        # NOTE: Per-sample correctness score — 1.0 if the extracted answer matches ground truth (after normalization), 0.0 if it does not match, or None if no answer could be extracted. None should be assigned 0.0 in the downstream processing
                        self.record_dev_accuracy(dev_output, ans_loop)
                        ctx1.save("dev_output", dev_output)

                # On fresh run: returns the in-memory DataProto saved by ctx1.save().
                # On resume (stage_1_done=True, should_run=False): lazily downloads
                # dev_output.pt via StorageResolver (local disk first, R2 fallback)
                # the first time result() is called — no download happens at __enter__.
                dev_output = ctx1.result("dev_output")

                # ── Logging Point 1: dev rollout metrics ──
                dev_scores_dict = group_scores_by_qid(dev_output)
                dev_rollout_metrics = compute_answer_rollout_metrics(
                    scores=dev_scores_dict,
                    response_mask=dev_output.batch["response_mask"],
                    rollout_n=solver_n,
                    prefix="dev_rollout",
                )
                dev_rollout_metrics.update(
                    compute_opener_rollout_metrics(
                        dev_output, self.solver_tokenizer, prefix="dev_rollout"
                    )
                )
                # Log timing for stages completed so far (curriculum_refresh, dev_rollout)
                # and pop them so Logging Point 6 doesn't re-log the same keys.
                early_timing = {f"ans_timing_s/{k}": ans_timing_raw.pop(k) for k in list(ans_timing_raw)}
                tracking_logger.log({"train/ans_loop": ans_loop, **dev_rollout_metrics, **early_timing}, step=self.gen_steps)


            # ── Replay generator updates from completed gen_loops ──
            if resume.gen_loops:
                self._replay_generator_updates(resume, resume_dir)

            # ── Restore document dataset state ──
            self._restore_doc_dataset_state(resume, ans_loop)

            gen_loop_results: list[dict] = []

            # Rebuild gen_loop_results from completed gen_loops
            gen_loop_results = self._rebuild_gen_loop_results(resume, resume_dir)

            if self._dev_only:
                # dev_only: skip entire generator loop (Stages 2-5).
                # Solver will train directly on dev_output in Stage 6.
                logger.info(
                    "  [dev_only] Skipping Stages 2-5 (generator loop) — "
                    "solver will train on dev data"
                )
                # Advance gen_steps so wandb step axis keeps advancing
                self.gen_steps += self.num_gen_per_ans

            for local_gen in range(0 if self._dev_only else self.num_gen_per_ans):
                gen_loop = ans_loop * self.num_gen_per_ans + local_gen
                gl = resume.ensure_gen_loop(local_gen)

                # If all stages for this gen_loop are done, skip it entirely
                if gl.stage_5_done:
                    logger.info(
                        "========== GEN LOOP SKIP | ans_loop=%d | local_gen=%d/%d "
                        "| gen_loop=%d (resumed) ==========",
                        ans_loop, local_gen + 1, self.num_gen_per_ans, gen_loop,
                    )
                    self.gen_steps = max(self.gen_steps, gen_loop + 1)
                    continue

                self._update_progress(local_gen=local_gen, stage="gen_loop_start")
                logger.info(
                    "========== GEN LOOP START | ans_loop=%d | local_gen=%d/%d "
                    "| gen_loop=%d ==========",
                    ans_loop, local_gen + 1, self.num_gen_per_ans, gen_loop,
                )

                gen_n = self.config.generator.rollout.n
                gen_num_workers = self.config.generator.rollout.agent.num_workers
                gen_timing_raw: dict[str, float] = {}
                _gen_loop_start = time.perf_counter()

                # ================================================================
                # Stage 2: Question rollout
                # ================================================================
                self._update_progress(stage="stage2_question_gen")
                self._maybe_log_heartbeat()
                if self._pregenerated_source:
                    # ── Pre-generated question mode: download instead of generate ──
                    with self.gen_stage_ctx(
                        name="question_rollout", stage_id=2, resume=resume,
                        resume_dir=resume_dir, local_gen=local_gen,
                        is_done=lambda: gl.stage_2_done,
                        mark_done=lambda: self._mark_stage_2_done(gl),
                        ans_loop=ans_loop,
                        timing_dict=gen_timing_raw,
                    ) as ctx2:
                        if ctx2.should_run:
                            remote_key = f"ans_{ans_loop}/gen_{local_gen}/gen_questions.json"
                            gen_questions = self._download_pregenerated_questions(remote_key)
                            if not gen_questions:
                                logger.warning(
                                    "  [Stage 2] No valid pre-generated questions for "
                                    "ans_loop=%d gen_loop=%d, skipping gen_loop",
                                    ans_loop, gen_loop,
                                )
                                gl.stage_2_done = True
                                gl.stage_3_done = True
                                gl.stage_4_done = True
                                gl.stage_5_done = True
                                self.gen_steps += 1
                                continue
                            logger.info(
                                "  [Stage 2] Loaded %d pre-generated questions from %s",
                                len(gen_questions), remote_key,
                            )
                            ctx2.save_json("gen_questions_pregenerated", gen_questions)

                    gen_output = None
                    gen_questions = ctx2.result_json("gen_questions_pregenerated")

                    # ── Logging Point 2 (pre-generated): basic stats only ──
                    tracking_logger.log({
                        "train/gen_step": self.gen_steps,
                        "train/ans_loop": ans_loop,
                        "train/gen_loop": gen_loop,
                        "gen_question_rollout/num_valid_questions": float(len(gen_questions)),
                    }, step=self.gen_steps)

                else:
                    with self.gen_stage_ctx(
                        name="question_rollout", stage_id=2, resume=resume,
                        resume_dir=resume_dir, local_gen=local_gen,
                        is_done=lambda: gl.stage_2_done,
                        mark_done=lambda: self._mark_stage_2_done(gl),
                        ans_loop=ans_loop,
                        timing_dict=gen_timing_raw,
                    ) as ctx2:
                        if ctx2.should_run:
                            doc_batch = self.prepare_doc_batch()
                            doc_batch = doc_batch.repeat(repeat_times=gen_n, interleave=True)
                            doc_batch.meta_info["progress_label"] = (
                                f"stage2/question_rollout ans_loop={ans_loop} gen_loop={gen_loop}"
                            )
                            gen_func = "_generate_with_shared_engine" if self._shared_engine_mode else "_generate_with_metadata"
                            logger.info("  [Stage 2] %s (question rollout, n=%d, samples=%d) ...", gen_func, gen_n, len(doc_batch))
                            if self._shared_engine_mode:
                                gen_output = self._generate_with_shared_engine(
                                    doc_batch, solver_num_workers
                                )
                            else:
                                gen_output = self._generate_with_metadata(
                                    self.gen_rollout_manager, doc_batch, gen_num_workers
                                )
                            logger.info("  [Stage 2] %s done", gen_func)
                            parse_generated_questions(gen_output, self.gen_tokenizer)

                            # Anti-copy guard: reject generated questions that
                            # match seed questions (seeded_dev mode only).
                            if self.seed_questions is not None:
                                self._reject_seed_copies(gen_output)

                            if not any(gen_output.non_tensor_batch["parsed_ok"]):
                                raise RuntimeError(
                                    f"No valid generated questions parsed at ans_loop={ans_loop}, gen_loop={gen_loop}. "
                                    "All generator outputs failed parsing; inspect generator prompts/outputs."
                                )

                            # Save doc_dataset state *after* consuming batch so
                            # resume restores to the correct position for the next gen_loop
                            resume.doc_dataset_state = self.doc_dataset.get_state()

                            # Persist the shortcut partition for resume so the
                            # Stage 2→3 injection below can rebuild gen_questions
                            # identically after a restart.
                            ctx2.save_json(
                                "shortcut_doc_ids",
                                list(getattr(self, "_pending_shortcut_doc_ids", []) or []),
                            )
                            ctx2.save("gen_output", gen_output)

                    gen_output = ctx2.result("gen_output")
                    gen_questions = derive_gen_questions(gen_output)
                    shortcut_doc_ids_for_gen = ctx2.result_json("shortcut_doc_ids") or []
                    if shortcut_doc_ids_for_gen:
                        gen_questions.extend(
                            self._build_shortcut_gen_questions(shortcut_doc_ids_for_gen)
                        )

                    # ── Logging Point 2: question generation metrics ──
                    question_metrics = compute_question_rollout_metrics(
                        gen_questions=gen_questions,
                        total_samples=gen_output.batch["responses"].shape[0],
                        response_mask=gen_output.batch["response_mask"],
                        num_documents=len(set(gen_output.non_tensor_batch["doc_id"])),
                        prefix="gen_question_rollout",
                        reject_reasons=gen_output.non_tensor_batch.get("reject_reason"),
                    )
                    b1_timing = {f"gen_timing_s/{k}": gen_timing_raw.pop(k) for k in list(gen_timing_raw)}
                    tracking_logger.log({
                        "train/gen_step": self.gen_steps,
                        "train/ans_loop": ans_loop,
                        "train/gen_loop": gen_loop,
                        **question_metrics,
                        **b1_timing,
                    }, step=self.gen_steps)

                self._update_progress(stage="stage3_gen_answer")
                self._maybe_log_heartbeat()
                # ================================================================
                # Stage 3: Gen answer rollout
                # ================================================================
                with self.gen_stage_ctx(
                    name="gen_answer_rollout", stage_id=3, resume=resume,
                    resume_dir=resume_dir, local_gen=local_gen,
                    is_done=lambda: gl.stage_3_done,
                    mark_done=lambda: setattr(gl, 'stage_3_done', True),
                    ans_loop=ans_loop,
                    timing_dict=gen_timing_raw,
                ) as ctx3:
                    if ctx3.should_run:
                        gen_q_batch = self.prepare_question_batch(gen_questions)
                        gen_q_batch = gen_q_batch.repeat(
                            repeat_times=solver_n, interleave=True
                        )
                        gen_q_batch.meta_info["progress_label"] = (
                            f"stage3/gen_answer_rollout ans_loop={ans_loop} gen_loop={gen_loop}"
                        )
                        logger.info("  [Stage 3] _generate_with_metadata (gen answer rollout, n=%d, samples=%d) ...", solver_n, len(gen_q_batch))
                        gen_answer_output = self._generate_with_metadata(
                            self.solver_rollout_manager, gen_q_batch, solver_num_workers
                        )
                        logger.info("  [Stage 3] _generate_with_metadata done")
                        extract_answer_scores(gen_answer_output, self.solver_tokenizer)
                        ctx3.save("gen_answer_output", gen_answer_output)

                gen_answer_output = ctx3.result("gen_answer_output")

                # ── Logging Point 3: gen answer rollout metrics ──
                gen_scores_dict = group_scores_by_qid(gen_answer_output)
                answer_metrics = compute_answer_rollout_metrics(
                    scores=gen_scores_dict,
                    response_mask=gen_answer_output.batch["response_mask"],
                    rollout_n=solver_n,
                    prefix="gen_answer_rollout",
                )
                answer_metrics.update(
                    compute_opener_rollout_metrics(
                        gen_answer_output,
                        self.solver_tokenizer,
                        prefix="gen_answer_rollout",
                    )
                )
                b2_timing = {f"gen_timing_s/{k}": gen_timing_raw.pop(k) for k in list(gen_timing_raw)}
                tracking_logger.log({
                    "train/gen_step": self.gen_steps,
                    "train/ans_loop": ans_loop,
                    "train/gen_loop": gen_loop,
                    **answer_metrics,
                    **b2_timing,
                }, step=self.gen_steps)

                # Free unreferenced tensors before memory-intensive Stage 4
                # (influence scoring does forward+backward while Stage 3 data is
                # still live — this is the peak host memory point).
                _log_host_memory("pre-stage4-gc")
                gc.collect()
                _log_host_memory("post-stage4-gc")

                self._update_progress(stage="stage4_scoring")
                self._maybe_log_heartbeat()
                # ================================================================
                # Stage 4: Scoring
                # ================================================================
                with self.gen_stage_ctx(
                    name="scoring", stage_id=4, resume=resume,
                    resume_dir=resume_dir, local_gen=local_gen,
                    is_done=lambda: gl.stage_4_done,
                    mark_done=lambda: setattr(gl, 'stage_4_done', True),
                    ans_loop=ans_loop,
                    timing_dict=gen_timing_raw,
                ) as ctx4:
                    if ctx4.should_run:
                        selected_reward_components = resolve_generator_reward_components(
                            self.config.training
                        )
                        reward_structure = resolve_generator_reward_structure(
                            self.config.training
                        )
                        # Use gen_answer_output (not gen_output) so that
                        # questions filtered by prompt length in
                        # prepare_question_batch are excluded — they'll
                        # receive gen_invalid_penalty in the generator
                        # update instead of a silent 0.0.
                        valid_question_ids = {
                            str(qid)
                            for qid in gen_answer_output.non_tensor_batch["question_id"]
                            if qid is not None
                        }
                        need_influence = "influence_rewards" in selected_reward_components
                        need_spice = "spice_rewards" in selected_reward_components

                        influence_rewards: dict[str, float] = {}
                        spice_rewards: dict[str, float] = {}
                        influence_metrics = {}

                        if need_influence:
                            skip_dev = local_gen > 0 and self._momentum_live
                            # Drop shortcut rows from the gen-side influence input.
                            # Shortcut questions are injected post-Stage-2 from
                            # already-vetted free-form sources; they have no
                            # generator gradient to score and should not bias
                            # solver-filter ranking via influence cosine.
                            gen_data_for_influence = gen_answer_output
                            n_shortcut_dropped = 0
                            is_shortcut_arr = gen_answer_output.non_tensor_batch.get(
                                "is_shortcut", None
                            )
                            if is_shortcut_arr is not None and any(
                                bool(x) for x in is_shortcut_arr
                            ):
                                non_shortcut_idx = [
                                    i for i, sc in enumerate(is_shortcut_arr)
                                    if not bool(sc)
                                ]
                                n_shortcut_dropped = (
                                    len(is_shortcut_arr) - len(non_shortcut_idx)
                                )
                                if not non_shortcut_idx:
                                    logger.info(
                                        "  [Stage 4] all gen rows are shortcut; "
                                        "skipping influence (no MCQ-side rows)"
                                    )
                                    influence_rewards, influence_metrics = {}, {}
                                    gen_data_for_influence = None
                                else:
                                    gen_data_for_influence = slice_dataproto(
                                        gen_answer_output, non_shortcut_idx
                                    )
                            if gen_data_for_influence is not None:
                                logger.info(
                                    "  [Stage 4] compute_influence_scores "
                                    "(skip_dev=%s, shortcut_dropped=%d) ...",
                                    skip_dev, n_shortcut_dropped,
                                )
                                influence_rewards, influence_metrics = self.compute_influence_scores(
                                    dev_data=dev_output,
                                    gen_data=gen_data_for_influence,
                                    skip_dev=skip_dev,
                                    ans_loop=ans_loop,
                                    gen_loop=gen_loop,
                                )
                                logger.info("  [Stage 4] compute_influence_scores done")

                        if need_spice:
                            logger.info("  [Stage 4] compute_spice_scores ...")
                            spice_rewards = self.compute_spice_scores(gen_answer_output)
                            logger.info("  [Stage 4] compute_spice_scores done")

                        rewards_payload = build_stage4_reward_payload(
                            valid_question_ids=valid_question_ids,
                            influence_rewards=influence_rewards,
                            spice_rewards=spice_rewards,
                            selected_components=selected_reward_components,
                            reward_structure=reward_structure,
                        )

                        gl.influence_metrics = influence_metrics
                        ctx4.save_json("rewards", rewards_payload)

                rewards_payload = ctx4.result_json("rewards")
                solver_filter_rewards = extract_influence_rewards_for_solver_filter(
                    rewards_payload
                )
                influence_metrics = getattr(gl, "influence_metrics", {})

                # ── Logging Point 4: scoring & influence metrics ──
                b3_timing = {f"gen_timing_s/{k}": gen_timing_raw.pop(k) for k in list(gen_timing_raw)}
                tracking_logger.log({
                    "train/gen_step": self.gen_steps,
                    "train/ans_loop": ans_loop,
                    "train/gen_loop": gen_loop,
                    **influence_metrics,
                    **b3_timing,
                }, step=self.gen_steps)

                self._update_progress(stage="stage5_gen_ppo")
                self._maybe_log_heartbeat()
                # ================================================================
                # Stage 5: Generator PPO update
                # ================================================================
                gen_rc_metrics = {}
                gen_quant_metrics = {}
                gen_adv_metrics = {}
                gen_actor_output = None
                with self.gen_stage_ctx(
                    name="gen_ppo_update", stage_id=5, resume=resume,
                    resume_dir=resume_dir, local_gen=local_gen,
                    is_done=lambda: gl.stage_5_done,
                    mark_done=lambda: setattr(gl, 'stage_5_done', True),
                    ans_loop=ans_loop,
                    defer_state_update=True,
                    timing_dict=gen_timing_raw,
                ) as ctx5:
                    if ctx5.should_run:
                        if not self.config.training.fix_generator:
                            gen_update_batch, gen_quant_metrics = self.prepare_gen_update_batch(
                                gen_output, rewards_payload
                            )
                            if gen_update_batch is not None:
                                gen_update_batch = self._truncate_batch_for_workers(
                                    gen_update_batch, self.generator_wg, "gen_update"
                                )
                                self._ppo_stage_diag(
                                    "stage5-pre-rollout-correction",
                                    role_name="generator",
                                    batch=gen_update_batch,
                                    ans_loop=ans_loop,
                                    gen_loop=gen_loop,
                                )
                                logger.info("  [Stage 5] rollout_correction ...")
                                gen_update_batch, gen_rc_metrics = self._maybe_apply_rollout_correction(
                                    batch=gen_update_batch,
                                    worker_group=self.generator_wg,
                                    role_name="generator",
                                )
                                logger.info("  [Stage 5] rollout_correction done")
                                self._ppo_stage_diag(
                                    "stage5-post-rollout-correction",
                                    role_name="generator",
                                    batch=gen_update_batch,
                                    ans_loop=ans_loop,
                                    gen_loop=gen_loop,
                                )
                                gen_update_batch, gen_adv_metrics = self.compute_generator_advantage(
                                    gen_update_batch,
                                )

                                # Scatter so each rank sees samples from
                                # every doc group (matches V2 scattered_chunk).
                                gen_uids = gen_update_batch.non_tensor_batch["uid"]
                                gen_num_groups = len(dict.fromkeys(gen_uids))
                                gen_total = len(gen_update_batch)
                                gen_samples_per_group = gen_total // gen_num_groups
                                gen_dp = self.generator_wg.world_size
                                gen_update_batch = scatter_for_dispatch(
                                    gen_update_batch, gen_num_groups,
                                    gen_samples_per_group, gen_dp,
                                )
                                if self.config.generator.actor.get("on_policy_minibatch", False):
                                    gen_update_batch.meta_info["ppo_mini_batch_size"] = (
                                        len(gen_update_batch) // gen_dp
                                    )
                                else:
                                    gen_update_batch.meta_info["ppo_mini_batch_size"] = (
                                        self.config.generator.actor.ppo_mini_batch_size
                                        * self.config.generator.rollout.n
                                        // self.generator_wg.world_size
                                    )
                                if self._use_global_batch_norm("generator"):
                                    gen_update_batch.meta_info["avg_response_tokens"] = (
                                        gen_update_batch.batch["response_mask"].float().sum().item()
                                        / gen_update_batch.batch["response_mask"].shape[0]
                                    )
                                    gen_update_batch.meta_info["dp_size"] = gen_dp
                                self._ppo_stage_diag(
                                    "stage5-pre-update-actor",
                                    role_name="generator",
                                    batch=gen_update_batch,
                                    ans_loop=ans_loop,
                                    gen_loop=gen_loop,
                                )
                                logger.info("  [Stage 5] update_actor (generator PPO) ...")
                                gen_actor_output = self.generator_wg.update_actor(gen_update_batch)
                                logger.info("  [Stage 5] update_actor done")
                                self._ppo_stage_diag(
                                    "stage5-post-update-actor",
                                    role_name="generator",
                                    batch=gen_update_batch,
                                    actor_output=gen_actor_output,
                                    ans_loop=ans_loop,
                                    gen_loop=gen_loop,
                                )
                            else:
                                logger.warning(
                                    "gen_loop=%d: skipping generator update (no valid batch)",
                                    gen_loop,
                                )
                        else:
                            logger.info(
                                "---------- Stage 5 skipped | ans_loop=%d "
                                "| gen_loop=%d | fix_generator=True ----------",
                                ans_loop, gen_loop,
                            )

                # Accumulate data for Stage 6
                gen_loop_results.append(
                    {
                        "gen_answer_output": gen_answer_output,
                        "rewards": solver_filter_rewards,
                        "gen_questions": gen_questions,
                    }
                )

                # ── Logging Point 5: gen PPO update metrics ──
                metrics = {
                    "train/gen_step": self.gen_steps,
                    "train/ans_loop": ans_loop,
                    "train/gen_loop": gen_loop,
                }
                metrics.update({f"gen_timing_s/{k}": v for k, v in gen_timing_raw.items()})

                # Generator rollout correction metrics (pre-training: old_log_probs vs rollout_log_probs)
                if gen_rc_metrics:
                    metrics.update({f"gen_{k}": v for k, v in gen_rc_metrics.items()})
                if gen_quant_metrics:
                    metrics.update(gen_quant_metrics)
                if gen_adv_metrics:
                    metrics.update(gen_adv_metrics)

                # Generator actor update metrics
                # WARNING: Same overwrite caveat as solver metrics — see comment above.
                # Actor-level rollout_corr/* metrics overwrite the trainer-level values.
                if gen_actor_output is not None:
                    from verl.utils.metric import reduce_metrics
                    gen_actor_metrics = reduce_metrics(gen_actor_output.meta_info["metrics"])
                    metrics.update({f"gen_{k}": v for k, v in gen_actor_metrics.items()})
                    gen_data_metrics = compute_data_metrics(gen_update_batch, use_critic=False)
                    metrics.update({f"gen_{k}": v for k, v in gen_data_metrics.items()})
                    metrics["train/gen_num_training_samples"] = float(len(gen_update_batch))

                tracking_logger.log(metrics, step=self.gen_steps)
                self.gen_steps += 1

                # Accumulate gen_loop totals into ans-level timing
                gen_timing_raw["gen_loop_total"] = time.perf_counter() - _gen_loop_start
                if "gen_loops_total" not in ans_timing_raw:
                    ans_timing_raw["gen_loops_total"] = 0.0
                ans_timing_raw["gen_loops_total"] += gen_timing_raw["gen_loop_total"]

                logger.info(
                    "========== GEN LOOP END | ans_loop=%d | gen_loop=%d ==========",
                    ans_loop, gen_loop,
                )

            _log_host_memory("pre-stage6")
            self._update_progress(local_gen=-1, stage="stage6_solver_ppo")
            self._maybe_log_heartbeat()
            # ================================================================
            # Stage 6: Solver PPO update (after all gen_loops)
            # ================================================================
            solver_rc_metrics = {}
            solver_opener_adv_metrics = {}
            solver_actor_output = None
            with self.stage_ctx(
                name="solver_ppo_update", stage_id=6, resume=resume,
                resume_dir=resume_dir,
                is_done=lambda: resume.stage_6_done,
                mark_done=lambda: setattr(resume, 'stage_6_done', True),
                ans_loop=ans_loop,
                defer_state_update=True,
                timing_dict=ans_timing_raw,
            ) as ctx6:
                if ctx6.should_run:
                    if not self.config.training.fix_answer_model:
                        if self._dev_only:
                            solver_batch = self._build_dev_solver_update_batch(dev_output)
                        else:
                            solver_batch = self.aggregate_solver_training_data(gen_loop_results)
                        if solver_batch is not None:
                            solver_batch = self._truncate_batch_for_workers(
                                solver_batch, self.solver_wg, "solver_update"
                            )
                            self._ppo_stage_diag(
                                "stage6-pre-rollout-correction",
                                role_name="solver",
                                batch=solver_batch,
                                ans_loop=ans_loop,
                            )
                            logger.info("  [Stage 6] rollout_correction ...")
                            solver_batch, solver_rc_metrics = self._maybe_apply_rollout_correction(
                                batch=solver_batch,
                                worker_group=self.solver_wg,
                                role_name="solver",
                            )
                            logger.info("  [Stage 6] rollout_correction done")
                            self._ppo_stage_diag(
                                "stage6-post-rollout-correction",
                                role_name="solver",
                                batch=solver_batch,
                                ans_loop=ans_loop,
                            )
                            solver_batch = self.compute_advantage(
                                solver_batch,
                                normalization_mode=self.config.algorithm.get(
                                    "solver_normalization_mode", "group_std"
                                ),
                            )
                            solver_opener_adv_metrics = compute_opener_advantage_metrics(
                                solver_batch, prefix="solver_ppo"
                            )

                            # Scatter so each rank sees samples from
                            # every question group (matches V2 scattered_chunk).
                            solver_uids = solver_batch.non_tensor_batch["uid"]
                            solver_num_groups = len(dict.fromkeys(solver_uids))
                            solver_total = len(solver_batch)
                            solver_samples_per_group = solver_total // solver_num_groups
                            solver_dp = self.solver_wg.world_size
                            # Stratify question order by doc_id so contiguous
                            # PPO minibatches don't get dominated by a single
                            # document's questions when #docs < ppo_mini_batch_size.
                            solver_batch = stratify_by_doc(
                                solver_batch, solver_num_groups,
                                solver_samples_per_group,
                            )
                            solver_batch = scatter_for_dispatch(
                                solver_batch, solver_num_groups,
                                solver_samples_per_group, solver_dp,
                            )
                            if self.config.solver.actor.get("on_policy_minibatch", False):
                                solver_batch.meta_info["ppo_mini_batch_size"] = (
                                    len(solver_batch) // solver_dp
                                )
                            else:
                                solver_batch.meta_info["ppo_mini_batch_size"] = (
                                    self.config.solver.actor.ppo_mini_batch_size
                                    * self.config.solver.rollout.n
                                    // self.solver_wg.world_size
                                )
                            if self._use_global_batch_norm("solver"):
                                solver_batch.meta_info["avg_response_tokens"] = (
                                    solver_batch.batch["response_mask"].float().sum().item()
                                    / solver_batch.batch["response_mask"].shape[0]
                                )
                                solver_batch.meta_info["dp_size"] = solver_dp
                            self._ppo_stage_diag(
                                "stage6-pre-update-actor",
                                role_name="solver",
                                batch=solver_batch,
                                ans_loop=ans_loop,
                            )
                            logger.info("  [Stage 6] update_actor (solver PPO) ...")
                            solver_actor_output = self.solver_wg.update_actor(solver_batch)
                            logger.info("  [Stage 6] update_actor done")
                            self._ppo_stage_diag(
                                "stage6-post-update-actor",
                                role_name="solver",
                                batch=solver_batch,
                                actor_output=solver_actor_output,
                                ans_loop=ans_loop,
                            )
                        else:
                            logger.warning(
                                "ans_loop=%d: skipping solver update (no valid batch)",
                                ans_loop,
                            )
                    else:
                        logger.info(
                            "---------- Stage 6 skipped | ans_loop=%d "
                            "| fix_answer_model=True ----------",
                            ans_loop,
                        )

            ans_timing_raw["ans_loop_total"] = time.perf_counter() - _ans_loop_start

            # ── Logging Point 6: ans_loop summary ──
            # Derive gen_scores for each result so compute_ans_loop_summary_metrics
            # keeps its generic dict-based interface (no DataProto dependency).
            if not self._dev_only:
                for r in gen_loop_results:
                    r["gen_scores"] = group_scores_by_qid(r["gen_answer_output"])
                ans_metrics = compute_ans_loop_summary_metrics(
                    gen_loop_results=gen_loop_results,
                    prefix="ans_loop",
                )
            else:
                ans_metrics = {}
            ans_metrics["train/ans_loop"] = ans_loop

            # Solver rollout correction metrics (pre-training: old_log_probs vs rollout_log_probs)
            if solver_rc_metrics:
                ans_metrics.update({f"solver_{k}": v for k, v in solver_rc_metrics.items()})
            if solver_opener_adv_metrics:
                ans_metrics.update(solver_opener_adv_metrics)

            # Solver actor update metrics
            # WARNING: The actor micro-batch loop in dp_actor.py also computes
            # rollout_corr/* metrics (using current log_prob vs rollout_log_prob).
            # These OVERWRITE the trainer-level rollout_corr/* metrics set above,
            # because both resolve to the same prefixed keys (e.g. solver_rollout_corr/chi2_seq).
            # The actor-level values include training drift from optimizer steps across
            # mini-batches, making chi2_seq exponentially larger than the pre-training value.
            # See the warning in compute_offpolicy_metrics() for details.
            if solver_actor_output is not None:
                from verl.utils.metric import reduce_metrics
                solver_actor_metrics = reduce_metrics(solver_actor_output.meta_info["metrics"])
                ans_metrics.update({f"solver_{k}": v for k, v in solver_actor_metrics.items()})
                # Batch-level advantage / reward / response-length stats
                solver_data_metrics = compute_data_metrics(solver_batch, use_critic=False)
                ans_metrics.update({f"solver_{k}": v for k, v in solver_data_metrics.items()})
                ans_metrics["train/solver_num_training_samples"] = float(len(solver_batch))
                # Solver formatting bonus stats (set by _build_solver_update_batch
                # / _build_dev_solver_update_batch when bonus is enabled).
                for _k, _v in solver_batch.meta_info.items():
                    if isinstance(_k, str) and _k.startswith("solver_formatting_bonus/"):
                        ans_metrics[_k] = float(_v)

            ans_metrics.update({f"ans_timing_s/{k}": v for k, v in ans_timing_raw.items()})

            tracking_logger.log(ans_metrics, step=self.gen_steps)

            # ── Checkpointing ──
            # Two modes controlled by always_save_for_resume:
            #
            # False (default, "interval" mode):
            #   Save checkpoint only at save_every_n_steps intervals and
            #   the final loop.  Previous FSDP shards deleted from R2,
            #   HF metadata always kept.  Lower R2 upload pressure.
            #
            # True ("continuous" mode):
            #   Save FSDP checkpoint every loop for exact resume.
            #   save_every_n_steps controls which HF checkpoints are
            #   *kept* on R2 (at 0, N, 2N, ... plus the final loop).
            #   FSDP shards on R2 are always rotated (only the latest
            #   kept).  Non-interval HF dirs are cleaned from R2.
            self._update_progress(stage="checkpointing")
            self._maybe_log_heartbeat()
            continuous_mode = self.config.training.always_save_for_resume
            save_interval = self.config.training.save_every_n_steps
            should_save_ckpt, is_hf_keep_step = self._checkpoint_policy_for_ans_loop(
                ans_loop
            )
            should_upload_stage_context = (
                self._should_upload_stage_context_for_ans_loop(ans_loop)
            )

            if continuous_mode:
                logger.info(
                    "---------- Checkpointing | ans_loop=%d"
                    " | hf_keep=%s ----------",
                    ans_loop, is_hf_keep_step,
                )
            elif should_save_ckpt:
                logger.info(
                    "---------- Checkpointing | ans_loop=%d ----------",
                    ans_loop,
                )

            if should_save_ckpt:
                # Wait for previous checkpoint upload + cleanup before saving
                # the next one.  Without this, slow uploads cause multiple
                # checkpoints to accumulate on local disk and fill it up.
                if (
                    self._pending_ckpt_upload_id is not None
                    or self._pending_ckpt_marker_upload_id is not None
                ):
                    logger.info(
                        "Waiting for previous checkpoint upload %s to finish "
                        "before saving ans_loop=%d ...",
                        self._pending_ckpt_upload_id, ans_loop,
                    )
                    self._finalize_pending_checkpoint_upload(
                        raise_on_failure=self._stop_on_checkpoint_upload_failure,
                    )

                self.save_checkpoint(ans_loop)
                self._training_progress["last_checkpoint_ans_loop"] = ans_loop
                self._training_progress["last_checkpoint_time"] = time.time()
            else:
                logger.info(
                    "---------- Skipping checkpoint | ans_loop=%d "
                    "(save_every_n_steps=%d) ----------",
                    ans_loop, save_interval,
                )

            # Upload checkpoint to R2 (non-blocking).
            # FIFO queue ordering guarantees that state.json (with stages 5/6
            # marked done) reaches R2 only AFTER the checkpoint directory
            # containing the corresponding weight updates has fully landed.
            #
            # The freshly saved global_step_{N}/ directory is treated as a
            # staging area when remote uploads are enabled: it is deleted after
            # the upload task finishes, regardless of success or failure.
            # remove_previous_ckpt still controls whether older local resume
            # state / checkpoint scaffolding is pruned once a newer checkpoint
            # has been recorded.
            upload_enabled = (
                self._upload_manager is not None
                and self._upload_manager.upload_enabled
            )
            cleanup_previous_local = (
                self.config.training.remove_previous_ckpt and upload_enabled
            )

            # Mark deferred PPO stages done now that checkpoint is saved locally.
            # For local resume, the checkpoint is on disk, so this is consistent.
            for gl in resume.gen_loops:
                gl.stage_5_done = True
            resume.stage_6_done = True
            resume.save(resume_dir)

            if upload_enabled and should_save_ckpt:
                local_dir = self.config.training.default_local_dir
                step_dir = os.path.join(local_dir, f"global_step_{ans_loop}")
                self._pending_ckpt_ans_loop = ans_loop

                # Upload checkpoint dir only if it exists (it won't when both
                # fix_generator and fix_answer_model are True — no model shards
                # were saved).  Uses rclone copy (additive), so uploading a
                # partial dir (e.g. solver only) won't delete existing files.
                upload_task_id = None
                if os.path.isdir(step_dir):
                    upload_task_id = self._upload_manager.submit_checkpoint_upload(
                        local_dir=step_dir,
                        remote_key=f"global_step_{ans_loop}",
                        cleanup_after=True,
                    )
                    self._pending_ckpt_upload_id = upload_task_id

                self._upload_manager.submit_file_upload(
                    local_path=os.path.join(resume_dir, "state.json"),
                    remote_key=f"ans_{ans_loop}/state.json",
                    depends_on=upload_task_id,
                )
                marker_task_id = self._upload_manager.submit_file_upload(
                    local_path=os.path.join(local_dir, "latest_checkpointed_iteration.txt"),
                    remote_key="latest_checkpointed_iteration.txt",
                    depends_on=upload_task_id,
                )
                self._pending_ckpt_marker_upload_id = marker_task_id

                # After marker upload, delete previous checkpoint from R2.
                #
                # Interval mode: delete FSDP shards, always keep HF metadata
                #   (original behavior).
                #
                # Continuous mode: FSDP shards ALWAYS deleted (only keep the
                #   latest for resume).  HF dirs kept on R2 only at
                #   save_every_n_steps intervals and the final loop, so that
                #   curated HF checkpoints remain for evaluation.
                #
                # The remote delete is keyed off the last checkpoint whose
                # marker upload succeeded, not the most recent local save.
                if (
                    cleanup_previous_local
                    and self._last_remote_checkpointed_ans_loop is not None
                ):
                    prev_saved = self._last_remote_checkpointed_ans_loop
                    prev_step_dir = os.path.join(local_dir, f"global_step_{prev_saved}")

                    if continuous_mode:
                        # Keep HF only at interval steps and last step.
                        prev_is_hf_keep = (
                            (save_interval > 0 and prev_saved % save_interval == 0)
                            or prev_saved == self.config.training.max_ans_loop - 1
                        )
                        exclude = ["**/huggingface/**"] if prev_is_hf_keep else None
                    else:
                        # Interval mode: always preserve HF metadata on R2.
                        exclude = ["**/huggingface/**"]

                    self._upload_manager.submit_remote_delete(
                        remote_key=f"global_step_{prev_saved}",
                        exclude_patterns=exclude,
                        cleanup_local_path=prev_step_dir if os.path.isdir(prev_step_dir) else None,
                        depends_on=marker_task_id,
                    )
                for failed_step in sorted(
                    step
                    for step in self._failed_remote_checkpoint_ans_loops
                    if (
                        step < ans_loop
                        and step != self._last_remote_checkpointed_ans_loop
                    )
                ):
                    self._upload_manager.submit_remote_delete(
                        remote_key=f"global_step_{failed_step}",
                        exclude_patterns=None,
                        depends_on=marker_task_id,
                    )
                    # Also clean up any stale local checkpoints between the
                    # previous saved step and the current one (crash-recovery
                    # leftovers that were never cleaned).
                    for old_step in range(0, ans_loop):
                        if old_step == prev_saved:
                            continue
                        old_dir = os.path.join(local_dir, f"global_step_{old_step}")
                        if os.path.isdir(old_dir):
                            logger.info("Cleaning up stale local checkpoint: %s", old_dir)
                            shutil.rmtree(old_dir, ignore_errors=True)

                self._last_saved_ans_loop = ans_loop
            elif upload_enabled and not should_save_ckpt and should_upload_stage_context:
                # Upload ans-level state only for the one-loop post-checkpoint
                # window where we still keep stage-context artifacts remotely.
                self._upload_manager.submit_file_upload(
                    local_path=os.path.join(resume_dir, "state.json"),
                    remote_key=f"ans_{ans_loop}/state.json",
                )

            if cleanup_previous_local and should_save_ckpt:
                ResumeState.clear(resume_dir)
                # Clean up old ans_* dirs no longer needed for crash recovery
                # (checkpoint saved — mirrors stale global_step_* cleanup above).
                base_dir = os.path.dirname(resume_dir)
                for old_loop in range(0, ans_loop):
                    old_ans_dir = os.path.join(base_dir, f"ans_{old_loop}")
                    if os.path.isdir(old_ans_dir):
                        logger.info("Cleaning up stale local ans dir: %s", old_ans_dir)
                        shutil.rmtree(old_ans_dir, ignore_errors=True)
            elif cleanup_previous_local and not should_save_ckpt:
                # Interval mode non-checkpoint step: outputs already uploaded,
                # dir not needed for resume (no checkpoint was saved at this step).
                ResumeState.clear(resume_dir)

            # ── In-training benchmark evaluation ──
            # Run at the end of each ans_loop, after Stage 6 (solver PPO
            # update) and checkpoint save/upload, so the eval metrics
            # correspond to the same model weights stored in
            # global_step_{ans_loop}.
            if self._benchmark_evaluator and self._benchmark_evaluator.should_evaluate(ans_loop):
                with simple_timer("benchmark_eval", ans_timing_raw):
                    eval_metrics = self._benchmark_evaluator.evaluate_all(ans_loop)
                    eval_metrics["train/ans_loop"] = ans_loop
                    tracking_logger.log(eval_metrics, step=self.gen_steps)

            logger.info(
                "========== ANS LOOP END | ans_loop=%d ==========",
                ans_loop,
            )

            # Free large DataProto tensors accumulated during gen_loops
            # to prevent CPU memory growth across ans_loops (~2.8 GB per gen_loop).
            del gen_loop_results
            gc.collect()

            # Note: Momentum is managed by the worker. When use_momentum=False,
            # the worker's momentum is reset at the start of each ans_loop via
            # the reset_momentum flag in compute_dev_gradient's meta_info.

        # Shut down upload manager
        if self._upload_manager:
            self._upload_manager.shutdown()
            if (
                self._pending_ckpt_upload_id is not None
                or self._pending_ckpt_marker_upload_id is not None
            ):
                self._finalize_pending_checkpoint_upload(
                    raise_on_failure=self._stop_on_checkpoint_upload_failure,
                    timeout=5.0,
                )
        logger.info("========== TRAINING END ==========")

    # ------------------------------------------------------------------
    # Data pipeline methods
    # ------------------------------------------------------------------

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

    def _build_agent_loop_config(self, model_role: str) -> DictConfig:
        """Build config compatible with AgentLoopManager from role-specific config.

        AgentLoopManager expects ``config.actor_rollout_ref.*``,
        ``config.reward_model.*``, ``config.trainer.*``, and ``config.data.*``.
        Our config stores per-role settings under ``config.generator`` /
        ``config.solver``, so we remap here.

        Args:
            model_role: ``"generator"`` or ``"solver"``.
        """
        role_config = self.config[model_role]
        role_cfg_container = OmegaConf.to_container(role_config, resolve=True)
        rollout_cfg = role_cfg_container.setdefault("rollout", {})
        agent_cfg = rollout_cfg.setdefault("agent", {})
        # Avoid Ray named-actor collisions when multiple AgentLoopManager
        # instances coexist (e.g., generator + solver).
        agent_cfg["worker_name_prefix"] = (
            f"{model_role}_agent_loop_worker_{self._agent_loop_run_id}"
        )
        # Also differentiate vLLM server actor names per role.
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

    @staticmethod
    def _normalize_hf_name_or_path(name_or_path: str | None) -> str | None:
        """Normalize a HuggingFace ``name_or_path`` so that the hub model ID
        and the local cache snapshot path compare equal.

        Examples::
            "OctoThinker/OctoThinker-8B-Hybrid-Base"
            "/root/.cache/huggingface/hub/models--OctoThinker--OctoThinker-8B-Hybrid-Base/snapshots/f464..."

        Both normalise to ``"OctoThinker/OctoThinker-8B-Hybrid-Base"``.
        """
        if name_or_path is None:
            return None
        import re
        # Match HF cache path: ...models--<org>--<model>/snapshots/...
        m = re.search(r"models--([^/]+?)--([^/]+)", name_or_path)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
        return name_or_path

    def _validate_shared_engine_compatibility(self) -> None:
        """Fail fast if generator/solver rollout settings are incompatible."""

        def _short_repr(value: Any) -> str:
            text = repr(value)
            return text if len(text) <= 120 else f"{text[:117]}..."

        mismatches: list[str] = []

        def _check(field_name: str, gen_value: Any, solver_value: Any) -> None:
            if gen_value != solver_value:
                mismatches.append(
                    f"{field_name}: generator={_short_repr(gen_value)}, solver={_short_repr(solver_value)}"
                )

        gen_rollout = self.config.generator.rollout
        solver_rollout = self.config.solver.rollout

        _check("generator.model.path", self.config.generator.model.path, self.config.solver.model.path)
        _check("rollout.name", gen_rollout.name, solver_rollout.name)
        _check("rollout.mode", gen_rollout.get("mode", None), solver_rollout.get("mode", None))
        _check(
            "rollout.tensor_model_parallel_size",
            gen_rollout.tensor_model_parallel_size,
            solver_rollout.tensor_model_parallel_size,
        )
        _check(
            "rollout.pipeline_model_parallel_size",
            gen_rollout.pipeline_model_parallel_size,
            solver_rollout.pipeline_model_parallel_size,
        )
        _check("rollout.data_parallel_size", gen_rollout.data_parallel_size, solver_rollout.data_parallel_size)
        _check("rollout.temperature", gen_rollout.get("temperature", None), solver_rollout.get("temperature", None))
        _check("rollout.top_k", gen_rollout.get("top_k", None), solver_rollout.get("top_k", None))
        _check("rollout.top_p", gen_rollout.get("top_p", None), solver_rollout.get("top_p", None))
        _check("rollout.prompt_length", gen_rollout.prompt_length, solver_rollout.prompt_length)
        _check("rollout.response_length", gen_rollout.response_length, solver_rollout.response_length)

        # Use the configured model paths instead of tokenizer.name_or_path.
        # The tokenizer attribute varies depending on whether the HF API or
        # local cache was used to load it (race with rate limits), so it is
        # unreliable for equality checks.
        _check(
            "model.path",
            self.config.generator.model.path,
            self.config.solver.model.path,
        )
        _check(
            "tokenizer.chat_template",
            getattr(self.gen_tokenizer, "chat_template", None),
            getattr(self.solver_tokenizer, "chat_template", None),
        )

        if mismatches:
            mismatch_text = "\n".join(f"- {msg}" for msg in mismatches)
            raise ValueError(
                "Shared rollout engine mode requires compatible generator/solver rollout settings when "
                "`trainer.separate_pools=false`.\n"
                f"Mismatches:\n{mismatch_text}\n"
                "Set `trainer.separate_pools=true` or align the listed fields."
            )

    def _validate_rollout_sharing(self) -> None:
        """Verify that generator and solver share the same rollout object in each process.

        After init_model(), both roles should reference the same VLLMAsyncRollout
        instance via _SHARED_VLLM_ROLLOUTS (keyed by pid, rank, signature).
        If they don't, the generator's rollout will lack an inference_engine and
        _generate_with_shared_engine() will crash.
        """
        gen_ids = self.generator_wg.get_rollout_object_id()
        solver_ids = self.solver_wg.get_rollout_object_id()

        mismatched = []
        for i, (gen_id, solver_id) in enumerate(zip(gen_ids, solver_ids)):
            if gen_id != solver_id:
                mismatched.append(i)

        if mismatched:
            logger.error(
                "Rollout sharing FAILED on workers %s: generator and solver have "
                "different rollout objects (gen_ids=%s, solver_ids=%s). "
                "Check 'Rollout signature' logs to find which field differs.",
                mismatched,
                [gen_ids[i] for i in mismatched],
                [solver_ids[i] for i in mismatched],
            )
            raise RuntimeError(
                f"Shared engine mode requires generator and solver to share the same "
                f"rollout object, but {len(mismatched)}/{len(gen_ids)} workers have "
                f"different rollout objects. Check 'Rollout signature' log messages "
                f"from the workers to identify which signature field differs."
            )
        logger.info(
            "Rollout sharing verified: all %d workers share the same rollout object",
            len(gen_ids),
        )

    def _resolve_role_method(self, worker: Any, role: str, method: str) -> Any:
        """Resolve direct rollout method on colocated WorkerDict actors."""
        prefixed_name = f"{role}_{method}"
        try:
            return getattr(worker, prefixed_name)
        except AttributeError:
            return getattr(worker, method)

    def _direct_role_wake_up_all_workers(
        self,
        worker_group: RayWorkerGroup,
        role: str,
    ) -> None:
        """Directly wake up all workers for a specific role and wait for sync."""
        futures = []
        for worker in worker_group.workers:
            wake = self._resolve_role_method(worker, role, "wake_up")
            futures.append(wake.remote())

        results = ray.get(futures)
        if not all(bool(result) for result in results):
            raise RuntimeError(f"{role} wake_up did not succeed on all workers: {results}")

    def _direct_role_sleep_all_workers(
        self,
        worker_group: RayWorkerGroup,
        role: str,
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
        if self.generator_wg is None or self.solver_rollout_manager is None:
            raise RuntimeError("Shared-engine generation requires initialized worker groups and solver manager.")

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

    def _generate_with_metadata(
        self, manager, batch: DataProto, num_workers: int
    ) -> DataProto:
        """Call AgentLoopManager.generate_sequences() preserving non_tensor_batch fields.

        The AgentLoopWorker's ``_postprocess()`` only preserves ``raw_prompt``
        in ``non_tensor_batch``.  Our custom fields (``question_id``,
        ``ground_truth``, ``doc_id``) would be lost, so we save them before the
        call and graft them back onto the output.

        Also pads the batch to be divisible by ``num_workers`` (required by
        ``AgentLoopManager`` which chunks evenly across workers).

        Args:
            manager: An ``AgentLoopManager`` instance.
            batch: Input ``DataProto`` with ``raw_prompt`` and metadata.
            num_workers: Number of agent loop workers for padding.

        Returns:
            Rollout ``DataProto`` with original (unpadded) batch size ``N``:

            output (DataProto)
            ├── batch (TensorDict)
            │   ├── prompts:         [N, prompt_length]
            │   ├── responses:       [N, response_length]
            │   ├── response_mask:   [N, response_length]
            │   ├── input_ids:       [N, prompt_length + response_length]
            │   ├── attention_mask:  [N, prompt_length + response_length]
            │   ├── position_ids:    [N, total_len] or [N, C, total_len]
            │   └── rollout_log_probs (optional, controlled by generator.rollout.calculate_log_probs): [N, response_length]
            │
            ├── non_tensor_batch (dict of numpy arrays, dtype=object)
            │   ├── raw_prompt (from AgentLoopWorker)
            │   ├── __num_turns__ (from AgentLoopWorker)
            │   └── restored custom metadata from input (e.g. question_id,
            │       ground_truth, doc_id)
            │
            └── meta_info (dict)
                └── ``timing`` removed by this wrapper.
        """
        saved_ntb = {
            k: v.copy() for k, v in batch.non_tensor_batch.items()
            if k not in ("raw_prompt", "agent_name")
        }

        batch_padded, pad_size = pad_dataproto_to_divisor(batch, num_workers)
        output_padded = manager.generate_sequences(batch_padded) # async call to vllm servers 
        output = unpad_dataproto(output_padded, pad_size)

        output.meta_info.pop("timing", None)

        for k, v in saved_ntb.items():
            output.non_tensor_batch[k] = v

        return output

    def _select_dev_questions_for_ans_loop(self, ans_loop: int) -> list[dict]:
        """Return the dev-question slice used for Stage 1 in ``ans_loop``."""
        sample_size = int(
            self.config.training.get("dev_rollout_subsample_size", 0) or 0
        )
        total = len(self.dev_questions)

        if sample_size <= 0:
            return self.dev_questions

        if sample_size >= total:
            if sample_size > total:
                logger.warning(
                    "training.dev_rollout_subsample_size=%d exceeds dev set size=%d; "
                    "using full dev set",
                    sample_size,
                    total,
                )
            return self.dev_questions

        sample_seed = self.config.training.get("dev_rollout_subsample_seed", None)
        base_seed = (
            self.config.training.seed if sample_seed is None else int(sample_seed)
        )
        loop_seed = base_seed + ans_loop
        rng = random.Random(loop_seed)
        sampled_indices = rng.sample(range(total), sample_size)
        sampled_indices.sort()
        sampled_questions = [self.dev_questions[i] for i in sampled_indices]

        logger.info(
            "Dev rollout subsample: ans_loop=%d selected=%d/%d seed=%d",
            ans_loop,
            len(sampled_questions),
            total,
            loop_seed,
        )
        return sampled_questions

    def prepare_dev_batch(self, dev_questions: Optional[list[dict]] = None) -> DataProto:
        """Prepare a ``DataProto`` batch from the dev dataset for solver rollout.

        Loads dev questions (full set or sampled subset) and formats them as
        answer prompts. MCQ vs open-ended/free-form QA is inferred from each
        question row using the shared benchmark adapter: rows with non-empty
        ``choices`` are MCQ, rows without choices are free-form unless
        ``benchmark_type`` explicitly overrides that.

        Returns:
            Pre-rollout ``DataProto`` for ``AgentLoopManager.generate_sequences()``:

            dev_batch (DataProto)
            ├── batch (TensorDict)
            │   └── dummy: [B, 1] uint8 (required placeholder tensor)
            │
            ├── non_tensor_batch (dict of numpy arrays, dtype=object)
            │   ├── raw_prompt:   (B,)  — list of chat messages per sample
            │   ├── question_id:     (B,) — e.g. "79dea257a5264e6a8d633efac8573de5"
            │   ├── ground_truth:    (B,) — normalized letter for MCQ, raw answer for free-form
            │   ├── is_mcq:          (B,) — scoring dispatch flag
            │   ├── benchmark_type:  (B,) — e.g. "qa_mcq" or "qa_open"
            │   ├── data_source:     (B,) — optional benchmark/verifier routing key
            │   └── verifier_metadata: (B,) — optional verifier metadata
            │
            └── meta_info (dict)
                ├── eos_token_id: int
                └── pad_token_id: int

            ``input_ids`` / ``attention_mask`` / ``position_ids`` are produced
            later inside the AgentLoop worker during ``generate_sequences()``.
        """
        if dev_questions is None:
            dev_questions = self.dev_questions
        if not dev_questions:
            raise ValueError("No dev questions available for Stage 1 rollout.")

        max_prompt_tokens = self.config.solver.rollout.prompt_length
        messages_list = []
        question_ids = []
        ground_truths = []
        is_mcq_flags = []
        benchmark_types = []
        data_sources = []
        verifier_metadatas = []
        skipped = 0

        for q in dev_questions:
            messages, normalized_gt, question_is_mcq = build_messages_for_question(q)

            if should_filter_by_prompt_length(
                tokenizer=self.solver_tokenizer,
                messages=messages,
                max_prompt_tokens=max_prompt_tokens,
                logger=logger,
                sample_kind="dev question",
                sample_id=q["question_id"],
            ):
                skipped += 1
                continue

            messages_list.append(messages)
            question_ids.append(q["question_id"])
            ground_truths.append(normalized_gt)
            is_mcq_flags.append(question_is_mcq)
            benchmark_types.append(detect_benchmark_type(q))
            data_sources.append(q.get("data_source", ""))
            verifier_metadatas.append(build_verifier_metadata(q))

        if skipped:
            logger.warning(
                "Filtered %d/%d dev questions exceeding prompt_length=%d",
                skipped, len(dev_questions), max_prompt_tokens,
            )

        if not messages_list:
            raise RuntimeError(
                f"All {len(dev_questions)} dev questions exceeded "
                f"prompt_length={max_prompt_tokens}. Consider increasing "
                f"solver.rollout.prompt_length or using shorter questions."
            )

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

    def _download_pregenerated_questions(self, remote_key: str) -> list[dict]:
        """Download pre-generated questions from the configured source path."""
        import json as _json
        if self._pregenerated_backend is None:
            from verl_inf_evolve.storage.remote_backend import (
                build_hf_backend_kwargs,
                create_remote_backend,
            )
            remote_cfg = self.config.get("remote", {})
            kwargs = build_hf_backend_kwargs(remote_cfg, auto_create_repo=False)
            self._pregenerated_backend = create_remote_backend(
                self._pregenerated_source, **kwargs
            )
        data = self._pregenerated_backend.download_bytes(remote_key)
        payload = _json.loads(data)
        return [q for q in payload["questions"] if q.get("parsed_ok", True)]

    def prepare_doc_batch(self) -> DataProto:
        """Prepare a ``DataProto`` batch of prompts for generator rollout.

        Dispatches based on ``question_source_mode``:
        - ``"document"``: samples documents and builds doc-based prompts.
        - ``"seeded_dev"``: samples seed anchor IDs and builds few-shot
          seeded prompts (no document text).

        In both modes, the returned ``DataProto`` has the same schema so that
        downstream stages (parse, answer rollout, scoring) work unchanged.
        Seeded mode uses ``doc_id = "seed:<anchor_qid>"`` as the source group
        identifier.

        Returns:
            Pre-rollout ``DataProto`` for ``AgentLoopManager.generate_sequences()``.
        """
        source_mode = getattr(self.config.training, "question_source_mode", "document")
        if source_mode == "seeded_dev":
            return self._prepare_seeded_batch()
        return self._prepare_document_batch()

    def _prepare_document_batch(self) -> DataProto:
        """Build a batch of document-based question-generation prompts.

        When freeform_shortcut routing is active, docs in ``self._shortcut_doc_meta``
        are partitioned out and stashed on ``self._pending_shortcut_doc_ids``;
        the generator only sees MCQ-eligible docs. The Stage 2→3 boundary
        re-injects the shortcut docs as free-form questions for solver rollout.
        """
        batch_doc_ids, reshuffled = self.doc_dataset.next_batch()
        if reshuffled:
            logger.info("Document dataset reshuffled (epoch %d)", self.doc_dataset.epoch)

        # Partition into MCQ vs shortcut. Shortcut bucket is preserved on
        # `self` so the Stage 2→3 boundary can inject them into gen_questions.
        shortcut_meta = getattr(self, "_shortcut_doc_meta", {}) or {}
        shortcut_doc_ids = [d for d in batch_doc_ids if d in shortcut_meta]
        mcq_doc_ids = [d for d in batch_doc_ids if d not in shortcut_meta]
        self._pending_shortcut_doc_ids = shortcut_doc_ids
        if shortcut_doc_ids:
            logger.info(
                "freeform_shortcut: partitioned batch -> mcq=%d shortcut=%d",
                len(mcq_doc_ids), len(shortcut_doc_ids),
            )
        if not mcq_doc_ids:
            raise RuntimeError(
                f"All {len(batch_doc_ids)} docs in batch are freeform-shortcut. "
                "Increase training.doc_batch_size or reduce shortcut share so at "
                "least one MCQ-eligible doc is present."
            )

        max_prompt_tokens = self.config.generator.rollout.prompt_length
        messages_list = []
        doc_ids = []
        prompt_types = []
        skipped = 0

        for doc_id in mcq_doc_ids:
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

            if should_filter_by_prompt_length(
                tokenizer=self.gen_tokenizer,
                messages=messages,
                max_prompt_tokens=max_prompt_tokens,
                logger=logger,
                sample_kind=f"{prompt_type} doc",
                sample_id=doc_id,
            ):
                skipped += 1
                continue

            messages_list.append(messages)
            doc_ids.append(doc_id)
            prompt_types.append(prompt_type)

        if skipped:
            logger.warning(
                "Filtered %d/%d docs exceeding prompt_length=%d",
                skipped, len(mcq_doc_ids), max_prompt_tokens,
            )

        if not messages_list:
            raise RuntimeError(
                f"All {len(mcq_doc_ids)} MCQ documents in batch exceeded "
                f"prompt_length={max_prompt_tokens}. Consider increasing "
                f"generator.rollout.prompt_length or using shorter documents."
            )

        return messages_to_dataproto(
            messages_list=messages_list,
            non_tensor_metadata={
                "doc_id": doc_ids,
                "question_generation_prompt_type": prompt_types,
            },
            meta_info=self._gen_meta_info(),
        )

    def _prepare_seeded_batch(self) -> DataProto:
        """Build a batch of seed-based question-generation prompts.

        Each anchor question ID maps to a precomputed bundle of k-1 seed
        examples.  The examples are formatted as human-readable text and
        inserted into the seeded MCQ prompt template.

        Uses ``doc_id = "seed:<anchor_qid>"`` so downstream stages (grouping,
        metrics, logging) work unchanged.
        """
        batch_anchor_ids, reshuffled = self.doc_dataset.next_batch()
        if reshuffled:
            logger.info("Seed dataset reshuffled (epoch %d)", self.doc_dataset.epoch)

        max_prompt_tokens = self.config.generator.rollout.prompt_length
        messages_list = []
        doc_ids = []
        skipped = 0

        for anchor_qid in batch_anchor_ids:
            bundle = self.seed_prompt_bundles[anchor_qid]
            examples_text = format_seed_examples(bundle)
            messages = [
                {"role": "system", "content": SEEDED_MCQ_QUESTION_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": SEEDED_MCQ_QUESTION_GENERATION_PROMPT.format(
                    seed_examples=examples_text,
                )},
            ]

            synthetic_doc_id = f"seed:{anchor_qid}"

            if should_filter_by_prompt_length(
                tokenizer=self.gen_tokenizer,
                messages=messages,
                max_prompt_tokens=max_prompt_tokens,
                logger=logger,
                sample_kind="seed",
                sample_id=synthetic_doc_id,
            ):
                skipped += 1
                continue

            messages_list.append(messages)
            doc_ids.append(synthetic_doc_id)

        if skipped:
            logger.warning(
                "Filtered %d/%d seed prompts exceeding prompt_length=%d",
                skipped, len(batch_anchor_ids), max_prompt_tokens,
            )

        if not messages_list:
            raise RuntimeError(
                f"All {len(batch_anchor_ids)} seed prompts in batch exceeded "
                f"prompt_length={max_prompt_tokens}. Consider increasing "
                f"generator.rollout.prompt_length or reducing seed_examples_per_prompt."
            )

        return messages_to_dataproto(
            messages_list=messages_list,
            non_tensor_metadata={"doc_id": doc_ids},
            meta_info=self._gen_meta_info(),
        )

    # ------------------------------------------------------------------
    # Anti-copy guard (seeded mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for seed-copy detection: lowercase, strip whitespace/punctuation."""
        import re
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _reject_seed_copies(self, gen_output: "DataProto") -> None:
        """Mark parsed questions as rejected if they match any seed question.

        Operates in-place on ``gen_output.non_tensor_batch``. Uses exact
        normalized-text matching. Questions that were already rejected
        (``parsed_ok == False``) are skipped.

        Args:
            gen_output: DataProto after ``parse_generated_questions()``.
        """
        if not self.seed_questions:
            return

        # Build lookup set of normalized seed question texts (cached on first call)
        if not hasattr(self, "_seed_text_set"):
            self._seed_text_set = {
                self._normalize_text(q["question_text"])
                for q in self.seed_questions.values()
            }

        parsed_ok = gen_output.non_tensor_batch["parsed_ok"]
        question_texts = gen_output.non_tensor_batch["question_text"]
        reject_reasons = gen_output.non_tensor_batch["reject_reason"]
        rejected_count = 0

        for i in range(len(parsed_ok)):
            if not parsed_ok[i]:
                continue
            normalized = self._normalize_text(str(question_texts[i]))
            if normalized in self._seed_text_set:
                parsed_ok[i] = False
                reject_reasons[i] = "seed_copy"
                rejected_count += 1

        if rejected_count:
            logger.warning(
                "Anti-copy guard: rejected %d questions matching seed text",
                rejected_count,
            )

    def prepare_question_batch(self, questions: list[dict]) -> DataProto:
        """Prepare a ``DataProto`` batch of generated questions for solver rollout.

        Mirrors ``prepare_dev_batch``: dispatches each question through
        ``build_messages_for_question()`` (the shared QA adapter) which routes
        MCQ vs free-form via ``detect_benchmark_type`` (presence of ``choices``
        or explicit ``benchmark_type``). For generator-parsed MCQs this resolves
        to ``build_mcq_messages`` with default args, identical to the prior
        direct-call behavior; for shortcut free-form questions it resolves to
        the free-form builder. Verifier metadata and ``data_source`` are
        propagated so ``extract_answer_scores`` dispatches to the right verifier
        downstream.

        Prompts whose token length exceeds ``solver.rollout.prompt_length``
        are filtered out to avoid tensor size mismatches downstream.

        Args:
            questions: Parsed questions from ``parse_generated_questions()`` plus
                optional free-form shortcut questions injected after Stage 2.
                MCQ rows have non-empty ``choices``; free-form rows set
                ``is_mcq=False`` (and ``is_shortcut=True`` if applicable).

        Returns:
            DataProto with tokenized question prompts and metadata.
            ``non_tensor_batch`` carries: ``question_id``, ``ground_truth``,
            ``doc_id``, ``is_mcq``, ``benchmark_type``, ``data_source``,
            ``verifier_metadata``, ``is_shortcut``.
        """
        if not questions:
            raise ValueError(
                "prepare_question_batch received an empty question list. "
                "This indicates Stage 2 parsing failed."
            )

        max_prompt_tokens = self.config.solver.rollout.prompt_length
        messages_list = []
        question_ids = []
        ground_truths = []
        doc_ids = []
        is_mcq_flags = []
        benchmark_types = []
        data_sources = []
        verifier_metadatas = []
        is_shortcut_flags = []
        skipped = 0

        for q in questions:
            messages, normalized_gt, q_is_mcq = build_messages_for_question(q)

            if should_filter_by_prompt_length(
                tokenizer=self.solver_tokenizer,
                messages=messages,
                max_prompt_tokens=max_prompt_tokens,
                logger=logger,
                sample_kind="question",
                sample_id=q["question_id"],
            ):
                skipped += 1
                continue

            messages_list.append(messages)
            question_ids.append(q["question_id"])
            ground_truths.append(normalized_gt)
            doc_ids.append(q["doc_id"])
            is_mcq_flags.append(q_is_mcq)
            benchmark_types.append(detect_benchmark_type(q))
            data_sources.append(q.get("data_source", ""))
            verifier_metadatas.append(build_verifier_metadata(q))
            is_shortcut_flags.append(bool(q.get("is_shortcut", False)))

        if skipped:
            logger.warning(
                "Filtered %d/%d questions exceeding prompt_length=%d",
                skipped, len(questions), max_prompt_tokens,
            )

        if not messages_list:
            raise RuntimeError(
                f"All {len(questions)} questions in batch exceeded "
                f"prompt_length={max_prompt_tokens}. Consider increasing "
                f"solver.rollout.prompt_length or generating shorter questions."
            )

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
                "is_shortcut": is_shortcut_flags,
            },
            meta_info=self._solver_meta_info(),
        )

    def compute_spice_scores(
        self, gen_answer_output: DataProto
    ) -> dict[str, float]:
        """Compute SPICE rewards from answer-score variance.

        SPICE reward formula (from v2 train.py:1657-1761):
          ``exp(-(var - target_variance)^2 / variance_scale)``

        Runs on the driver — no model involvement.

        Args:
            gen_answer_output: DataProto with ``answer_score`` populated in
                ``non_tensor_batch`` by ``extract_answer_scores()``.

        Returns:
            Dict mapping ``question_id`` to a scalar SPICE reward.
        """
        target_var = self.config.spice.target_variance
        var_scale = self.config.spice.variance_scale

        scores = group_scores_by_qid(gen_answer_output)
        rewards: dict[str, float] = {}

        for qid, score_list in scores.items():
            # Unparseable answers have answer_score=None; map to 0.0
            # (consistent with all other downstream consumers)
            resolved_scores = [s if s is not None else 0.0 for s in score_list]

            variance = float(np.var(resolved_scores))
            reward = math.exp(-((variance - target_var) ** 2) / var_scale)
            rewards[qid] = reward

        return rewards

    def compute_influence_scores(
        self,
        dev_data: DataProto,
        gen_data: DataProto,
        skip_dev: bool,
        ans_loop: int = 0,
        gen_loop: int = 0,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute gradient-based influence scores via distributed workers.

        Phase 1 (if not skip_dev): Forward+backward on dev data to compute
        dev gradient and update momentum EMA on the solver workers.

        Phase 2: Compute per-question cosine similarity between gen-question
        gradients and the dev gradient momentum.

        Dispatches to ``solver_wg.compute_dev_gradient()`` (Phase 1) and
        ``solver_wg.compute_similarity()`` (Phase 2).

        Note: Score quantification (``influence.quantification_mode``) is NOT
        applied here — it's applied in ``prepare_gen_update_batch()`` where the
        influence/SPICE scores are used for the generator update.

        Reference: v2 ``verl_joint_dev_similarity.py:786-1062``

        Args:
            dev_data: Dev rollout DataProto (from Stage 1), with
                ``answer_score`` populated in ``non_tensor_batch``.
            gen_data: Gen-question answer rollout DataProto (from Stage 3),
                with ``answer_score`` populated in ``non_tensor_batch``.
            skip_dev: If ``True``, skip Phase 1 (reuse worker's momentum).

        Returns:
            ``(rewards, influence_metrics)`` where rewards maps question_id to
            influence score (cosine similarity), and influence_metrics contains
            reduced per-phase training metrics (prefixed ``influence_dev/``
            and ``influence_gen/``).
        """
        from verl.utils.metric import reduce_metrics

        influence_metrics: dict[str, float] = {}
        timing: dict[str, float] = {}
        self._log_influence_probe(
            "overall",
            "compute_influence_scores:start",
            question_count=len(
                set(str(qid) for qid in gen_data.non_tensor_batch["question_id"])
            ),
            extra={
                "ans_loop": ans_loop,
                "skip_dev": skip_dev,
                "dev_samples": dev_data.batch["input_ids"].shape[0],
                "gen_samples": gen_data.batch["input_ids"].shape[0],
            },
        )

        # ── Phase 1: Dev gradient + momentum ──
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
                normalization_mode = self.config.algorithm.get(
                    "solver_normalization_mode", "group_std"
                )
                num_dev_questions = len(dev_question_order)
                dev_total = dev_batch.batch["input_ids"].shape[0]
                dev_rollout_n = (
                    dev_total // num_dev_questions if num_dev_questions > 0 else 0
                )
                dp_size = self.solver_wg.world_size
                self._log_influence_probe(
                    "dev_gradient",
                    "prepare_batch:done",
                    batch=dev_batch,
                    question_count=num_dev_questions,
                    rollout_n=dev_rollout_n,
                    dp_size=dp_size,
                    extra={
                        "reset_momentum": reset_momentum,
                        "normalization_mode": normalization_mode,
                    },
                )

                adv_start = time.perf_counter()
                self._log_influence_probe(
                    "dev_gradient",
                    "compute_advantage:start",
                    batch=dev_batch,
                    question_count=num_dev_questions,
                    rollout_n=dev_rollout_n,
                    dp_size=dp_size,
                    extra={"normalization_mode": normalization_mode},
                )
                dev_batch = self.compute_advantage(
                    dev_batch,
                    normalization_mode=normalization_mode,
                )
                self._log_influence_probe(
                    "dev_gradient",
                    "compute_advantage:done",
                    batch=dev_batch,
                    question_count=num_dev_questions,
                    rollout_n=dev_rollout_n,
                    dp_size=dp_size,
                    extra={
                        "elapsed_s": f"{time.perf_counter() - adv_start:.2f}",
                        "normalization_mode": normalization_mode,
                    },
                )

                # # ---- DEBUG: log V3 dev advantages ----
                # _uids = dev_batch.non_tensor_batch["uid"]
                # _advs = dev_batch.batch["advantages"]
                # _rmask = dev_batch.batch["response_mask"]
                # _scores = dev_batch.batch["token_level_scores"]
                # # Per-sample: extract scalar advantage (constant across tokens)
                # for _i in range(min(len(_uids), _advs.shape[0])):
                #     _valid = _rmask[_i].bool()
                #     _adv_val = _advs[_i][_valid][0].item() if _valid.any() else 0.0
                #     _score_val = _scores[_i].sum().item()
                #     logger.info(
                #         "DEBUG_V3_DEV_ADV uid=%s row=%d score=%.6f adv=%.6f",
                #         _uids[_i], _i, _score_val, _adv_val,
                #     )
                # # ---- END DEBUG ----

                # Scatter dev data so each rank sees answers from every
                # question (matches V2 scattered_chunk behaviour).
                scatter_start = time.perf_counter()
                self._log_influence_probe(
                    "dev_gradient",
                    "scatter_for_dispatch:start",
                    batch=dev_batch,
                    question_count=num_dev_questions,
                    rollout_n=dev_rollout_n,
                    dp_size=dp_size,
                )
                dev_batch = scatter_for_dispatch(
                    dev_batch, num_dev_questions, dev_rollout_n, dp_size
                )
                self._log_influence_probe(
                    "dev_gradient",
                    "scatter_for_dispatch:done",
                    batch=dev_batch,
                    question_count=num_dev_questions,
                    rollout_n=dev_rollout_n,
                    dp_size=dp_size,
                    extra={"elapsed_s": f"{time.perf_counter() - scatter_start:.2f}"},
                )
                if self.config.algorithm.get("use_global_batch_norm", True):
                    dev_batch.meta_info["avg_response_tokens"] = (
                        dev_batch.batch["response_mask"].float().sum().item() / dev_batch.batch["response_mask"].shape[0]
                    )
                    dev_batch.meta_info["dp_size"] = dp_size

                dev_batch.meta_info["progress_label"] = (
                    f"stage4/dev_gradient ans_loop={ans_loop}"
                )
                worker_start = time.perf_counter()
                self._log_influence_probe(
                    "dev_gradient",
                    "compute_dev_gradient:start",
                    batch=dev_batch,
                    question_count=num_dev_questions,
                    rollout_n=dev_rollout_n,
                    dp_size=dp_size,
                )
                with simple_timer("dev_gradient", timing):
                    dev_result = self.solver_wg.compute_dev_gradient(dev_batch)
                self._log_influence_probe(
                    "dev_gradient",
                    "compute_dev_gradient:done",
                    batch=dev_batch,
                    question_count=num_dev_questions,
                    rollout_n=dev_rollout_n,
                    dp_size=dp_size,
                    extra={"elapsed_s": f"{time.perf_counter() - worker_start:.2f}"},
                )
                self._momentum_live = True
                dev_raw_metrics = dev_result.meta_info.get("metrics", {})
                dev_reduced = reduce_metrics(dev_raw_metrics) if dev_raw_metrics else {}
                for k, v in dev_reduced.items():
                    if k.startswith("influence_timing_s/"):
                        influence_metrics[f"influence_timing_s/dev_gradient/{k.removeprefix('influence_timing_s/')}"] = v
                    else:
                        influence_metrics[f"influence_dev/{k}"] = v

        # ── Phase 2: Per-question similarity ──
        # Compute all valid question IDs before filtering so we can assign
        # 0.0 to zero-variance questions (distinct from gen_invalid_penalty).
        all_question_ids = set(str(qid) for qid in gen_data.non_tensor_batch["question_id"])

        # If momentum was never computed (Phase 1 skipped due to zero-variance
        # or skip_dev=True on first call after recovery), we cannot compute
        # similarity — return 0.0 for all questions.
        if not self._momentum_live:
            logger.warning(
                "Phase 2: momentum not available (Phase 1 produced no dev gradient), "
                "assigning 0.0 to all %d questions", len(all_question_ids),
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

        normalization_mode = self.config.algorithm.get(
            "solver_normalization_mode", "group_std"
        )
        num_questions = len(question_order)
        total_samples = gen_batch.batch["input_ids"].shape[0]
        rollout_n = total_samples // num_questions if num_questions > 0 else 0
        dp_size = self.solver_wg.world_size
        self._log_influence_probe(
            "gen_similarity",
            "prepare_batch:done",
            batch=gen_batch,
            question_count=num_questions,
            rollout_n=rollout_n,
            dp_size=dp_size,
            extra={
                "normalization_mode": normalization_mode,
                "similarity_mode": self.config.influence.get(
                    "similarity_mode", "cosine"
                ),
            },
        )

        adv_start = time.perf_counter()
        self._log_influence_probe(
            "gen_similarity",
            "compute_advantage:start",
            batch=gen_batch,
            question_count=num_questions,
            rollout_n=rollout_n,
            dp_size=dp_size,
            extra={"normalization_mode": normalization_mode},
        )
        gen_batch = self.compute_advantage(
            gen_batch,
            normalization_mode=normalization_mode,
        )
        self._log_influence_probe(
            "gen_similarity",
            "compute_advantage:done",
            batch=gen_batch,
            question_count=num_questions,
            rollout_n=rollout_n,
            dp_size=dp_size,
            extra={
                "elapsed_s": f"{time.perf_counter() - adv_start:.2f}",
                "normalization_mode": normalization_mode,
            },
        )

        # # ---- DEBUG: log V3 gen advantages ----
        # _uids = gen_batch.non_tensor_batch["uid"]
        # _advs = gen_batch.batch["advantages"]
        # _rmask = gen_batch.batch["response_mask"]
        # _scores = gen_batch.batch["token_level_scores"]
        # for _i in range(min(len(_uids), _advs.shape[0])):
        #     _valid = _rmask[_i].bool()
        #     _adv_val = _advs[_i][_valid][0].item() if _valid.any() else 0.0
        #     _score_val = _scores[_i].sum().item()
        #     logger.info(
        #         "DEBUG_V3_GEN_ADV uid=%s row=%d score=%.6f adv=%.6f",
        #         _uids[_i], _i, _score_val, _adv_val,
        #     )
        # # ---- END DEBUG ----

        # Scatter data so each rank's mini-batch corresponds to one question
        # (v2_key_points #6: scattered interleaved data chunking)
        scatter_start = time.perf_counter()
        self._log_influence_probe(
            "gen_similarity",
            "scatter_for_dispatch:start",
            batch=gen_batch,
            question_count=num_questions,
            rollout_n=rollout_n,
            dp_size=dp_size,
        )
        gen_batch = scatter_for_dispatch(
            gen_batch, num_questions, rollout_n, dp_size
        )
        self._log_influence_probe(
            "gen_similarity",
            "scatter_for_dispatch:done",
            batch=gen_batch,
            question_count=num_questions,
            rollout_n=rollout_n,
            dp_size=dp_size,
            extra={"elapsed_s": f"{time.perf_counter() - scatter_start:.2f}"},
        )
        if self.config.algorithm.get("use_global_batch_norm", True):
            gen_batch.meta_info["avg_response_tokens"] = (
                gen_batch.batch["response_mask"].float().sum().item() / gen_batch.batch["response_mask"].shape[0]
            )
            gen_batch.meta_info["dp_size"] = dp_size

        gen_batch.meta_info["similarity_mode"] = self.config.influence.get(
            "similarity_mode", "cosine"
        )
        gen_batch.meta_info["progress_label"] = (
            f"stage4/compute_similarity ans_loop={ans_loop} gen_loop={gen_loop}"
        )

        sim_start = time.perf_counter()
        self._log_influence_probe(
            "gen_similarity",
            "compute_similarity:start",
            batch=gen_batch,
            question_count=num_questions,
            rollout_n=rollout_n,
            dp_size=dp_size,
            extra={"similarity_mode": gen_batch.meta_info["similarity_mode"]},
        )
        with simple_timer("similarity", timing):
            result = self.solver_wg.compute_similarity(gen_batch)
        self._log_influence_probe(
            "gen_similarity",
            "compute_similarity:done",
            batch=gen_batch,
            question_count=num_questions,
            rollout_n=rollout_n,
            dp_size=dp_size,
            extra={
                "elapsed_s": f"{time.perf_counter() - sim_start:.2f}",
                "similarity_mode": gen_batch.meta_info["similarity_mode"],
            },
        )

        gen_raw_metrics = result.meta_info.get("metrics", {})
        gen_reduced = reduce_metrics(gen_raw_metrics) if gen_raw_metrics else {}
        for k, v in gen_reduced.items():
            if k.startswith("influence_timing_s/"):
                influence_metrics[f"influence_timing_s/similarity/{k.removeprefix('influence_timing_s/')}"] = v
            else:
                influence_metrics[f"influence_gen/{k}"] = v

        similarity_metrics = result.meta_info["similarity_metrics"]
        add_similarity_metric_stats(influence_metrics, similarity_metrics)

        # ── Convert similarity metrics to per-question rewards ──
        rewards, scores, _score_mode = build_similarity_rewards(
            question_order,
            all_question_ids,
            similarity_metrics,
        )

        # Zero-variance questions get 0.0 (valid but no signal),
        # distinct from absent questions which get gen_invalid_penalty downstream.
        filtered_qids = all_question_ids - set(question_order)

        # Log reward stats over ALL valid questions (surviving + zero-variance)
        if rewards:
            add_distribution_stats(influence_metrics, "influence_sim/reward", list(rewards.values()))
            influence_metrics["influence_sim/reward/num_valid_questions"] = len(rewards)
            influence_metrics["influence_sim/reward/frac_diverse_questions"] = len(question_order) / len(rewards) if rewards else 0.0

        # Store trainer-side timing in influence_metrics for logging
        for k, v in timing.items():
            influence_metrics[f"influence_timing_s/{k}"] = v

        logger.info(
            "compute_influence_scores: %d questions (%d surviving, %d zero-variance), "
            "score range [%.4f, %.4f], timing: %s",
            len(rewards),
            len(question_order),
            len(filtered_qids),
            min(scores) if scores else 0.0,
            max(scores) if scores else 0.0,
            {k: f"{v:.2f}s" for k, v in timing.items()},
        )

        return rewards, influence_metrics

    def _quantify_scores_before_advantage(
        self,
        scores: torch.Tensor,
        quantification_mode: Optional[str],
    ) -> torch.Tensor:
        """Optionally quantize sequence-level rewards before ``compute_advantage()``.

        Uses the same score quantification behavior as the legacy
        sample-advantage preprocessor:
        - ``None``: keep raw scores.
        - ``"1bit"``: positive -> 1.0, non-positive -> 0.0.
        - ``"2bit"``: map to {-1.0, -0.1, 0.1, 1.0} via median(|score|).
        - ``"group_std_top_gamma"`` / ``"group_std_fixed_threshold"``:
          no value quantization (group filtering happens in
          ``prepare_gen_update_batch()``).
        """
        if quantification_mode not in (
            None,
            "1bit",
            "2bit",
            "group_std_top_gamma",
            "group_std_fixed_threshold",
        ):
            raise ValueError(
                "Invalid influence.quantification_mode: "
                f"{quantification_mode}. Must be None, '1bit', '2bit', or "
                "'group_std_top_gamma', 'group_std_fixed_threshold'."
            )
        if quantification_mode in (
            None,
            "group_std_top_gamma",
            "group_std_fixed_threshold",
        ) or scores.numel() == 0:
            return scores

        if quantification_mode == "1bit":
            return torch.where(scores > 0, torch.ones_like(scores), torch.zeros_like(scores))

        # 2-bit normalization: 4 levels (-1, -0.1, +0.1, +1)
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

        Reads per-sample scores from ``rollout_output.non_tensor_batch["answer_score"]``
        (set by ``extract_answer_scores()``).

        Builds response_mask, token_level_scores (from per-row correctness),
        and uid (question_id) for GRPO grouping.  When ``drop_zero_variance``
        is True, question groups where all scores are identical are removed
        to save compute.

        Note: No score quantification is applied here because answer_scores
        are already binary (0.0 or 1.0). Quantification of influence/SPICE
        scores happens in ``prepare_gen_update_batch()`` instead.

        Also packs common meta_info fields (temperature, rollout_corr_config)
        from ``self.config``. For ``"dev_gradient"`` phase, additionally packs
        momentum_beta and reset_momentum. For ``"gen_similarity"`` phase,
        packs mini_batch_size.

        Args:
            rollout_output: DataProto from ``generate_sequences()`` with
                ``answer_score`` already populated in ``non_tensor_batch``.
            phase: Either ``"dev_gradient"`` (Phase 1) or ``"gen_similarity"``
                (Phase 2). Controls which meta_info fields are packed.
            reset_momentum: For ``"dev_gradient"`` phase, whether to reset
                the momentum buffer. Ignored for ``"gen_similarity"``.
            drop_zero_variance: If True (default), filter out question groups
                where all scores are identical.

        Returns:
            (batch, question_order, selected_indices) where batch is ready for
            ``compute_advantage()`` (or None if all groups were filtered),
            question_order lists the surviving unique question IDs, and
            selected_indices maps each row in the filtered batch back to its
            index in the original ``rollout_output``.
        """
        responses = rollout_output.batch["responses"]
        batch_size = responses.shape[0]
        response_length = responses.size(1)
        question_ids = rollout_output.non_tensor_batch["question_id"]
        answer_scores = rollout_output.non_tensor_batch["answer_score"]

        # Build per-row scores.
        # Unparseable answers have answer_score=None; these are mapped to 0.0
        # and kept in training so the model learns to avoid unparseable formats.
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

        # Build DataProto
        input_ids = rollout_output.batch["input_ids"]
        attention_mask = rollout_output.batch["attention_mask"]
        position_ids = rollout_output.batch["position_ids"]

        rewards_tensor = torch.tensor(row_scores, dtype=torch.float32)
        # NOTE: No quantification here — answer_scores are already binary
        # (0.0 or 1.0), so quantification is a no-op / meaningless.
        # Quantification is applied in prepare_gen_update_batch() instead,
        # where influence/SPICE scores (continuous values) actually need it.

        # --- Zero-variance group filtering ---
        if drop_zero_variance:
            # Group quantified scores by question_id
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
                    len(zero_var_qids),
                    len(question_order),
                )

            selected_indices = [
                i for i in range(batch_size)
                if str(question_ids[i]) not in zero_var_qids
            ]
            question_order = [qid for qid in question_order if qid not in zero_var_qids]

            if not selected_indices:
                logger.warning("_prepare_influence_batch: all groups zero-variance")
                return None, [], []
        else:
            selected_indices = list(range(batch_size))

        # --- Apply selected_indices to build filtered tensors ---
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

        # Pack meta_info from config
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        if rollout_corr_config is not None:
            rollout_corr_config = dict(rollout_corr_config)
        batch.meta_info.update({
            "temperature": self.config.solver.rollout.temperature,
            "rollout_corr_config": rollout_corr_config,
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

        if filtered_batch_size != batch_size:
            logger.warning(
                "_prepare_influence_batch: sample size changed from %d to %d "
                "(dropped %d samples from zero-variance groups)",
                batch_size,
                filtered_batch_size,
                batch_size - filtered_batch_size,
            )

        return batch, question_order, selected_indices

    def prepare_gen_update_batch(
        self,
        gen_output: DataProto,
        rewards: Any,
    ) -> tuple[DataProto | None, dict[str, float]]:
        """Prepare a ``DataProto`` batch for generator PPO update.

        Uses ``gen_output`` as the single source of truth: every row in
        ``gen_output`` participates in the update. Reward components are:
        - ``influence_rewards``: per valid question (filtered by group-std tau)
        - ``spice_rewards``: per valid question
        - ``invalid_rewards``: ``gen_invalid_penalty`` for invalid questions,
          0.0 otherwise
        The final raw reward is the configured weighted sum of components.

        The generator is trained on its **question generation** rollout,
        not the solver's answer rollout (``v2_key_points #14``).

        Args:
            gen_output: ``DataProto`` from ``generator_wg.generate_sequences()``,
                with ``question_id`` and ``doc_id`` in
                ``non_tensor_batch`` (set by ``parse_generated_questions()``).
            rewards: Stage-4 reward payload (new format) or legacy
                ``{question_id: scalar}`` dict.

        Returns:
            ``(batch, gen_quant_metrics)`` where ``batch`` is ready for
            advantage computation and ``update_actor()`` (or ``None`` if no
            valid questions remain after filtering), and
            ``gen_quant_metrics`` contains Stage-5 quantization/filter stats.
        """
        start_t = time.perf_counter()

        component_tensor_keys = {
            "influence_rewards": "token_level_scores_influence",
            "spice_rewards": "token_level_scores_spice",
            "invalid_rewards": "token_level_scores_invalid",
        }

        batch_size = gen_output.batch["input_ids"].shape[0]
        parsed_qids = gen_output.non_tensor_batch["question_id"]
        doc_ids = gen_output.non_tensor_batch["doc_id"]
        penalty = float(self.config.training.gen_invalid_penalty)
        reward_combination_mode = resolve_generator_reward_combination_mode(
            self.config.training
        )
        decoupled_adv_enabled = reward_combination_mode == "decoupled"
        quantification_mode = self.config.influence.get("quantification_mode", None)

        valid_question_ids = {
            str(qid) for qid in parsed_qids if qid is not None
        }
        reward_payload = normalize_reward_payload(
            rewards=rewards,
            valid_question_ids=valid_question_ids,
            training_cfg=self.config.training,
        )
        selected_components = reward_payload["selected_components"]
        component_weights = {
            str(component): float(weight)
            for component, weight in reward_payload.get("component_weights", {}).items()
        }
        known_valid_qids = set(reward_payload["known_valid_question_ids"])
        reward_components = reward_payload["reward_components"]
        influence_rewards = reward_components["influence_rewards"]
        spice_rewards = reward_components["spice_rewards"]
        reward_structure = reward_payload.get("reward_structure", None)
        if decoupled_adv_enabled:
            if reward_structure is None:
                reward_structure = resolve_generator_reward_structure(
                    self.config.training
                )
            else:
                reward_structure = validate_generator_reward_structure(
                    list(reward_structure)
                )
        row_group_rewards: list[list[float]] = (
            [[] for _ in reward_structure] if decoupled_adv_enabled else []
        )

        influence_rewards, influence_filter_metrics = (
            filter_influence_rewards_by_group_std(
                influence_rewards=influence_rewards,
                parsed_qids=parsed_qids,
                doc_ids=doc_ids,
                known_valid_qids=known_valid_qids,
                influence_cfg=self.config.influence,
            )
        )

        gen_quant_metrics: dict[str, float] = {
            "gen_quant/decoupled_adv_enabled": 1.0 if decoupled_adv_enabled else 0.0,
            "gen_quant/reward_combination_mode_id": (
                1.0 if decoupled_adv_enabled else 0.0
            ),
            "gen_quant/selected_component/influence": (
                1.0 if "influence_rewards" in selected_components else 0.0
            ),
            "gen_quant/selected_component/spice": (
                1.0 if "spice_rewards" in selected_components else 0.0
            ),
            "gen_quant/selected_component/invalid": (
                1.0 if "invalid_rewards" in selected_components else 0.0
            ),
        }
        gen_quant_metrics.update(influence_filter_metrics)

        n_none_qids = sum(1 for q in parsed_qids if q is None)
        logger.info(
            "prepare_gen_update_batch DEBUG: batch_size=%d, known_valid_qids=%d, "
            "qid=None rows=%d, selected_components=%s, penalty=%.6f",
            batch_size,
            len(known_valid_qids),
            n_none_qids,
            selected_components,
            penalty,
        )

        row_component_rewards: dict[str, list[float]] = {
            "influence_rewards": [],
            "spice_rewards": [],
            "invalid_rewards": [],
        }
        row_rewards: list[float] = []
        row_uids: list[str] = []
        row_is_known_valid: list[bool] = []

        n_penalty = 0
        n_penalty_none_qid = 0
        n_penalty_not_in_rewards = 0
        n_reward_zero = 0
        n_reward_nonzero = 0

        for i in range(batch_size):
            raw_qid = parsed_qids[i]
            qid = None if raw_qid is None else str(raw_qid)
            doc_id = str(doc_ids[i])
            is_known_valid = qid is not None and qid in known_valid_qids

            influence_r = float(influence_rewards.get(qid, 0.0)) if is_known_valid else 0.0
            spice_r = float(spice_rewards.get(qid, 0.0)) if is_known_valid else 0.0
            invalid_r = 0.0 if is_known_valid else penalty

            component_reward_map = {
                "influence_rewards": influence_r,
                "spice_rewards": spice_r,
                "invalid_rewards": invalid_r,
            }
            if not is_known_valid:
                n_penalty += 1
                if qid is None:
                    n_penalty_none_qid += 1
                else:
                    n_penalty_not_in_rewards += 1

            if decoupled_adv_enabled:
                combined_r = 0.0
                assert reward_structure is not None
                for group_idx, group in enumerate(reward_structure):
                    group_reward = 0.0
                    for term in group["terms"]:
                        group_reward += float(term["weight"]) * component_reward_map[
                            str(term["name"])
                        ]
                    row_group_rewards[group_idx].append(group_reward)
                    combined_r += group_reward
            elif is_known_valid:
                combined_r = 0.0
                if "influence_rewards" in selected_components:
                    combined_r += component_weights.get("influence_rewards", 1.0) * influence_r
                if "spice_rewards" in selected_components:
                    combined_r += component_weights.get("spice_rewards", 1.0) * spice_r
                if "invalid_rewards" in selected_components:
                    combined_r += component_weights.get("invalid_rewards", 1.0) * invalid_r
            else:
                combined_r = 0.0
                if "invalid_rewards" in selected_components:
                    combined_r += component_weights.get("invalid_rewards", 1.0) * invalid_r

            if combined_r == 0.0:
                n_reward_zero += 1
            else:
                n_reward_nonzero += 1

            row_component_rewards["influence_rewards"].append(influence_r)
            row_component_rewards["spice_rewards"].append(spice_r)
            row_component_rewards["invalid_rewards"].append(invalid_r)
            row_rewards.append(combined_r)
            row_uids.append(doc_id)
            row_is_known_valid.append(is_known_valid)

        gen_quant_metrics.update({
            "gen_quant/input/rows_total": float(batch_size),
            "gen_quant/input/groups_total": float(len(set(row_uids))),
            "gen_quant/input/rows_from_reward_nonzero": float(n_reward_nonzero),
            "gen_quant/input/rows_from_reward_zero": float(n_reward_zero),
            "gen_quant/input/rows_penalty_none_qid": float(n_penalty_none_qid),
            "gen_quant/input/rows_penalty_not_in_rewards": float(n_penalty_not_in_rewards),
        })
        add_distribution_stats(gen_quant_metrics, "gen_quant/reward_pre_filter", row_rewards)
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_pre_filter/influence",
            row_component_rewards["influence_rewards"],
        )
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_pre_filter/spice",
            row_component_rewards["spice_rewards"],
        )
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_pre_filter/invalid",
            row_component_rewards["invalid_rewards"],
        )
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_post_influence_mask",
            row_rewards,
        )
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_post_influence_mask/influence",
            row_component_rewards["influence_rewards"],
        )
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_post_influence_mask/spice",
            row_component_rewards["spice_rewards"],
        )
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_post_influence_mask/invalid",
            row_component_rewards["invalid_rewards"],
        )

        if n_penalty > 0:
            logger.info(
                "prepare_gen_update_batch: assigned gen_invalid_penalty=%.4f "
                "to %d/%d rows (invalid questions)",
                penalty,
                n_penalty,
                batch_size,
            )

        eps = 1e-12
        filtered_docs: set[str] = set()
        if decoupled_adv_enabled:
            assert reward_structure is not None
            doc_group_values: dict[str, list[list[float]]] = {
                doc_id: [[] for _ in reward_structure]
                for doc_id in set(row_uids)
            }
            for idx, uid in enumerate(row_uids):
                for group_idx in range(len(reward_structure)):
                    doc_group_values[uid][group_idx].append(row_group_rewards[group_idx][idx])

            for doc_id, group_values in doc_group_values.items():
                all_zero_var = True
                for values in group_values:
                    std_val = float(np.std(np.asarray(values, dtype=np.float32)))
                    if std_val > eps:
                        all_zero_var = False
                        break
                if all_zero_var:
                    filtered_docs.add(doc_id)
            surviving_docs = set(doc_group_values.keys()) - filtered_docs
            n_total_docs = len(doc_group_values)
        else:
            # Legacy sum_scores path: preserve the historical component-wise
            # zero-variance filtering behavior.
            doc_component_values: dict[str, dict[str, list[float]]] = {
                doc_id: {component: [] for component in selected_components}
                for doc_id in set(row_uids)
            }
            for uid, idx in zip(row_uids, range(batch_size)):
                for component in selected_components:
                    doc_component_values[uid][component].append(
                        row_component_rewards[component][idx]
                    )

            for doc_id, comp_vals in doc_component_values.items():
                all_zero_var = True
                for component in selected_components:
                    std_val = float(
                        np.std(np.asarray(comp_vals[component], dtype=np.float32))
                    )
                    if std_val > eps:
                        all_zero_var = False
                        break
                if all_zero_var:
                    filtered_docs.add(doc_id)
            surviving_docs = set(doc_component_values.keys()) - filtered_docs
            n_total_docs = len(doc_component_values)

        logger.info(
            "prepare_gen_update_batch DEBUG: docs total=%d, filtered=%d, surviving=%d",
            n_total_docs, len(filtered_docs), len(surviving_docs),
        )
        if filtered_docs:
            logger.info(
                "prepare_gen_update_batch: filtering %d doc groups",
                len(filtered_docs),
            )

        selected_indices: list[int] = []
        selected_rewards: list[float] = []
        selected_component_rewards: dict[str, list[float]] = {
            "influence_rewards": [],
            "spice_rewards": [],
            "invalid_rewards": [],
        }
        selected_group_rewards: list[list[float]] = (
            [[] for _ in reward_structure] if decoupled_adv_enabled else []
        )
        selected_uids: list[str] = []
        selected_is_known_valid: list[bool] = []

        for i in range(batch_size):
            if row_uids[i] in surviving_docs:
                selected_indices.append(i)
                selected_rewards.append(row_rewards[i])
                for component in selected_component_rewards:
                    selected_component_rewards[component].append(
                        row_component_rewards[component][i]
                    )
                if decoupled_adv_enabled:
                    for group_idx in range(len(selected_group_rewards)):
                        selected_group_rewards[group_idx].append(
                            row_group_rewards[group_idx][i]
                        )
                selected_uids.append(row_uids[i])
                selected_is_known_valid.append(row_is_known_valid[i])

        rows_kept = len(selected_indices)
        rows_dropped = batch_size - rows_kept
        groups_kept = len(surviving_docs)
        groups_dropped = n_total_docs - groups_kept
        gen_quant_metrics.update({
            "gen_quant/survival/groups_kept": float(groups_kept),
            "gen_quant/survival/groups_dropped": float(groups_dropped),
            "gen_quant/survival/groups_keep_rate": (
                float(groups_kept) / float(n_total_docs) if n_total_docs > 0 else 0.0
            ),
            "gen_quant/survival/rows_kept": float(rows_kept),
            "gen_quant/survival/rows_dropped": float(rows_dropped),
            "gen_quant/survival/rows_keep_rate": (
                float(rows_kept) / float(batch_size) if batch_size > 0 else 0.0
            ),
            "gen_quant/survival/all_dropped": 1.0 if rows_kept == 0 else 0.0,
        })
        add_distribution_stats(gen_quant_metrics, "gen_quant/reward_post_filter_pre_quant", selected_rewards)
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_post_filter_pre_quant/influence",
            selected_component_rewards["influence_rewards"],
        )
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_post_filter_pre_quant/spice",
            selected_component_rewards["spice_rewards"],
        )
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_post_filter_pre_quant/invalid",
            selected_component_rewards["invalid_rewards"],
        )

        if not selected_indices:
            logger.warning("prepare_gen_update_batch: all groups filtered")
            add_distribution_stats(gen_quant_metrics, "gen_quant/reward_post_quant", [])
            gen_quant_metrics["gen_quant/timing_s"] = time.perf_counter() - start_t
            return None, gen_quant_metrics

        indices = torch.tensor(selected_indices, dtype=torch.long)
        input_ids = gen_output.batch["input_ids"][indices]
        attention_mask = gen_output.batch["attention_mask"][indices]
        position_ids = gen_output.batch["position_ids"][indices]
        responses = gen_output.batch["responses"][indices]
        rollout_log_probs = None
        if "rollout_log_probs" in gen_output.batch.keys():
            rollout_log_probs = gen_output.batch["rollout_log_probs"][indices]

        out_batch_size = len(selected_indices)
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:].float()

        selected_component_tensors: dict[str, torch.Tensor] = {}
        for component in selected_components:
            selected_component_tensors[component] = torch.tensor(
                selected_component_rewards[component],
                dtype=torch.float32,
            )

        # ``selected_rewards`` already reflects the configured weighted raw
        # reward composition.
        rewards_tensor = torch.tensor(selected_rewards, dtype=torch.float32)
        component_tensors_for_logging = selected_component_tensors

        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_post_quant",
            rewards_tensor.tolist(),
        )
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_post_quant/influence",
            component_tensors_for_logging.get(
                "influence_rewards", torch.zeros_like(rewards_tensor)
            ).tolist(),
        )
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_post_quant/spice",
            component_tensors_for_logging.get(
                "spice_rewards", torch.zeros_like(rewards_tensor)
            ).tolist(),
        )
        add_distribution_stats(
            gen_quant_metrics,
            "gen_quant/reward_post_quant/invalid",
            component_tensors_for_logging.get(
                "invalid_rewards", torch.zeros_like(rewards_tensor)
            ).tolist(),
        )

        token_level_scores = expand_scores_to_token_level(
            rewards_tensor, attention_mask, position_ids, response_length,
        )

        old_log_probs = torch.zeros(out_batch_size, response_length, dtype=torch.float32)

        tensors = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "token_level_scores": token_level_scores,
        }
        if decoupled_adv_enabled:
            assert reward_structure is not None
            for group_idx, group in enumerate(reward_structure):
                group_reward = torch.tensor(
                    selected_group_rewards[group_idx],
                    dtype=torch.float32,
                )
                tensors[f"token_level_scores_group_{group_idx}"] = expand_scores_to_token_level(
                    group_reward,
                    attention_mask,
                    position_ids,
                    response_length,
                )
            for component in selected_components:
                token_key = component_tensor_keys[component]
                tensors[token_key] = expand_scores_to_token_level(
                    selected_component_tensors[component],
                    attention_mask,
                    position_ids,
                    response_length,
                )
        if rollout_log_probs is not None:
            tensors["rollout_log_probs"] = rollout_log_probs

        td = TensorDict(
            tensors,
            batch_size=out_batch_size,
        )
        non_tensor_batch = {
            "uid": np.array(selected_uids, dtype=object),
        }
        meta_info = {
            "temperature": self.config.generator.rollout.temperature,
            "global_token_num": torch.sum(attention_mask, dim=-1).tolist(),
        }
        if decoupled_adv_enabled:
            meta_info["reward_components_for_adv"] = selected_components
            meta_info["reward_structure_for_adv"] = reward_structure

        if out_batch_size != batch_size:
            logger.warning(
                "prepare_gen_update_batch: sample size changed from %d to %d "
                "(dropped %d samples from group filtering)",
                batch_size,
                out_batch_size,
                batch_size - out_batch_size,
            )

        gen_quant_metrics["gen_quant/timing_s"] = time.perf_counter() - start_t

        return DataProto(
            batch=td,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        ), gen_quant_metrics

    def aggregate_solver_training_data(
        self, gen_loop_results: list[dict]
    ) -> DataProto | None:
        """Aggregate solver training data from all gen_loops in an ans_loop.

        For each gen_loop, builds a solver update batch from the solver's
        answer rollout (``gen_answer_output``) using answer correctness as
        the training signal. Applies solver-side quality filtering by
        question variance and ``influence.solver_filter_mode``.

        Reference: v2 ``train.py:2232-2237`` (aggregation before ans update)

        Args:
            gen_loop_results: List of dicts, each with keys
                ``gen_answer_output``, ``rewards``, ``gen_questions``.

        Returns:
            DataProto ready for advantage computation and ``update_actor()``,
            or ``None`` if no valid data remains after filtering.
        """
        all_batches: list[DataProto] = []
        total_input_samples = 0

        for result in gen_loop_results:
            gen_answer_output = result["gen_answer_output"]
            rewards = result["rewards"]  # {qid: float} influence-only solver filter score
            gen_questions = result["gen_questions"]

            total_input_samples += gen_answer_output.batch["input_ids"].shape[0]

            batch = self._build_solver_update_batch(
                gen_answer_output, rewards, gen_questions,
            )
            if batch is not None:
                all_batches.append(batch)

        if not all_batches:
            logger.warning("aggregate_solver_training_data: no valid batches")
            return None

        # Strip per-batch global_token_num before concat (recomputed below)
        for batch in all_batches:
            batch.meta_info.pop("global_token_num", None)

        # Concatenate all gen_loop batches
        combined = DataProto.concat(all_batches)
        combined_size = len(combined)

        if combined_size != total_input_samples:
            logger.warning(
                "aggregate_solver_training_data: sample size changed from %d to %d "
                "(dropped %d samples from zero-variance/solver filtering)",
                total_input_samples,
                combined_size,
                total_input_samples - combined_size,
            )

        # Update global_token_num for the combined batch
        combined.meta_info["global_token_num"] = torch.sum(
            combined.batch["attention_mask"], dim=-1
        ).tolist()

        return combined

    def _resolve_solver_filter_config(self) -> tuple[str, float]:
        """Resolve solver filtering config with backward-compatible fallbacks."""
        mode_cfg = self.config.influence.get("solver_filter_mode", None)
        alpha_cfg = self.config.influence.get("solver_filter_alpha", None)

        legacy_alpha = self.config.influence.get("alpha", None)
        legacy_alpha_random = bool(self.config.influence.get("alpha_random", False))
        legacy_filter_positive = bool(self.config.influence.get("filter_positive", False))

        mode: str
        if mode_cfg is None:
            if legacy_alpha is not None and float(legacy_alpha) < 1.0:
                mode = "random_alpha" if legacy_alpha_random else "top_alpha"
                if alpha_cfg is None:
                    alpha_cfg = legacy_alpha
                if not self._solver_filter_legacy_warned:
                    logger.warning(
                        "influence.alpha/alpha_random are deprecated for solver filtering; "
                        "use influence.solver_filter_mode and influence.solver_filter_alpha."
                    )
                    self._solver_filter_legacy_warned = True
            else:
                mode = "none"
        else:
            mode = normalize_solver_filter_mode(mode_cfg)

        if legacy_filter_positive and not self._solver_filter_legacy_warned:
            logger.warning(
                "influence.filter_positive is deprecated and ignored. "
                "Use influence.solver_filter_mode-based selection instead."
            )
            self._solver_filter_legacy_warned = True

        alpha = 1.0 if alpha_cfg is None else float(alpha_cfg)
        if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError(
                "Invalid influence.solver_filter_alpha: "
                f"{alpha_cfg}. Must be a finite float in [0, 1]."
            )
        return mode, alpha

    def _resolve_solver_formatting_bonus(self) -> float:
        """Return per-row Alright bonus to add in solver PPO paths, or 0.0."""
        cfg = self.config.algorithm.get("solver_formatting_bonus", None)
        if cfg is None or not bool(cfg.get("enabled", False)):
            return 0.0
        return float(cfg.get("alright_bonus", 0.0))

    def _resolve_solver_mid_eos_penalty(self) -> float:
        """Return per-row mid-EOS penalty to add in solver PPO paths, or 0.0.

        Mid-EOS = (answer_score is None) AND (last unmasked token == eos_token_id).
        Penalty is typically negative.
        """
        cfg = self.config.algorithm.get("solver_mid_eos_penalty", None)
        if cfg is None or not bool(cfg.get("enabled", False)):
            return 0.0
        return float(cfg.get("penalty", 0.0))

    @staticmethod
    def _compute_mid_eos_mask(
        responses: torch.Tensor,
        response_mask: torch.Tensor,
        answer_scores: Any,
        eos_token_id: int | None,
    ) -> "np.ndarray":
        """Per-row boolean: True iff (answer_score is None) AND last unmasked token == eos.

        Returns a numpy array of shape ``[batch_size]``.  When ``answer_scores``
        is None or ``eos_token_id`` is None, returns all-False.
        """
        batch_size = responses.shape[0]
        if answer_scores is None or eos_token_id is None:
            return np.zeros(batch_size, dtype=bool)
        # last unmasked token index per row, clamped to >= 0 for empty rows
        last_idx = (response_mask.sum(dim=-1) - 1).clamp_min(0).long()
        last_tok = (
            responses.gather(1, last_idx.unsqueeze(1))
            .squeeze(1)
            .detach()
            .cpu()
            .numpy()
        )
        ends_with_eos = last_tok == int(eos_token_id)
        unparsed = np.array(
            [s is None for s in answer_scores], dtype=bool
        )
        return ends_with_eos & unparsed

    def _build_solver_update_batch(
        self,
        gen_answer_output: DataProto,
        rewards: dict[str, float],
        gen_questions: list[dict],
    ) -> DataProto | None:
        """Build a solver PPO update batch from one gen_loop's answer rollout.

        Reads per-sample scores from ``gen_answer_output.non_tensor_batch["answer_score"]``
        (set by ``extract_answer_scores()``).
        The per-question influence rewards are used only for solver-side filtering.

        Args:
            gen_answer_output: Solver rollout output from Stage 3, with
                ``answer_score`` already populated in ``non_tensor_batch``.
            rewards: Per-question influence rewards (for filtering).
            gen_questions: Parsed questions (for question_id metadata).

        Returns:
            DataProto for solver update, or ``None`` if empty after filtering.
        """
        responses = gen_answer_output.batch["responses"]
        batch_size = responses.shape[0]

        # --- Reconstruct question_id for each row in gen_answer_output ---
        # The gen_answer_output rows correspond to question_ids from
        # prepare_question_batch, which processes gen_questions in order.
        # With rollout_n > 1, each question has multiple response rows.
        #
        # gen_answer_output.non_tensor_batch["question_id"] should exist
        # if prepare_question_batch set it correctly. For rollout_n handling,
        # assume the rollout engine repeats rows (or we repeat beforehand).
        question_ids = gen_answer_output.non_tensor_batch.get("question_id", None)

        if question_ids is None:
            # Fallback: infer from gen_questions order
            # Each question in gen_questions maps to one prompt row.
            # With rollout_n > 1, generate_sequences produces rollout_n
            # responses per prompt. Assume interleaved ordering.
            rollout_n = batch_size // len(gen_questions) if gen_questions else 1
            question_ids = np.array(
                [q["question_id"] for q in gen_questions for _ in range(rollout_n)],
                dtype=object,
            )

        answer_scores = gen_answer_output.non_tensor_batch.get("answer_score", None)
        answer_doc_ids = gen_answer_output.non_tensor_batch.get("doc_id", None)
        answer_openers = gen_answer_output.non_tensor_batch.get("opener_class", None)
        is_shortcut_arr = gen_answer_output.non_tensor_batch.get("is_shortcut", None)
        qid_to_doc_from_questions = {
            str(q["question_id"]): str(q["doc_id"])
            for q in gen_questions
            if q.get("question_id") is not None and q.get("doc_id") is not None
        }
        formatting_bonus = self._resolve_solver_formatting_bonus()
        mid_eos_penalty = self._resolve_solver_mid_eos_penalty()
        mid_eos_mask = (
            self._compute_mid_eos_mask(
                responses=responses,
                response_mask=gen_answer_output.batch["response_mask"],
                answer_scores=answer_scores,
                eos_token_id=self.solver_tokenizer.eos_token_id,
            )
            if mid_eos_penalty != 0.0
            else np.zeros(batch_size, dtype=bool)
        )

        # --- Build per-row correctness scores ---
        # Unparseable answers have answer_score=None; these are mapped to 0.0
        # and kept (valid_mask=True) so the model learns to avoid unparseable
        # formats. Only when the entire answer_scores array is missing
        # (extract_answer_scores never ran) do we mark rows invalid.
        row_scores = []
        row_qids = []
        row_doc_ids = []
        row_openers = []
        row_is_shortcut: list[bool] = []
        valid_mask = []
        n_mid_eos = 0

        for i in range(batch_size):
            qid = str(question_ids[i])
            # Group by question_id for GRPO
            doc_id = None
            if answer_doc_ids is not None and answer_doc_ids[i] is not None:
                doc_id = str(answer_doc_ids[i])
            elif qid in qid_to_doc_from_questions:
                doc_id = qid_to_doc_from_questions[qid]

            opener = (
                str(answer_openers[i]) if answer_openers is not None else None
            )
            bonus = formatting_bonus if opener == "alright" else 0.0
            penalty = mid_eos_penalty if mid_eos_mask[i] else 0.0
            if mid_eos_mask[i]:
                n_mid_eos += 1
            row_is_shortcut.append(
                bool(is_shortcut_arr[i]) if is_shortcut_arr is not None else False
            )

            if answer_scores is None:
                row_scores.append(0.0 + bonus + penalty)
                row_qids.append(qid)
                row_doc_ids.append(doc_id)
                if answer_openers is not None:
                    row_openers.append(opener)
                valid_mask.append(False)
                continue

            score = answer_scores[i]
            row_scores.append((float(score) if score is not None else 0.0) + bonus + penalty)
            row_qids.append(qid)
            row_doc_ids.append(doc_id)
            if answer_openers is not None:
                row_openers.append(opener)
            valid_mask.append(True)

        if mid_eos_penalty != 0.0:
            logger.info(
                "_build_solver_update_batch: mid_eos rows=%d/%d (penalty=%.4f applied)",
                n_mid_eos, batch_size, mid_eos_penalty,
            )

        # --- Solver training-question filtering ---
        # 1) Always remove zero-variance question groups.
        # 2) Apply mode-based filtering over surviving questions.
        qid_scores: dict[str, list[float]] = defaultdict(list)
        qid_to_doc: dict[str, str] = {}
        for qid, doc_id, score, valid in zip(
            row_qids, row_doc_ids, row_scores, valid_mask
        ):
            if not valid:
                continue
            qid_scores[qid].append(score)
            if doc_id is None:
                # Fallback so per-doc modes still behave sensibly even when
                # doc_id metadata is unexpectedly missing.
                doc_id = f"__unknown_doc__:{qid}"
            if qid not in qid_to_doc:
                qid_to_doc[qid] = doc_id

        mode, alpha = self._resolve_solver_filter_config()

        qid_variance = {
            qid: float(np.var(np.asarray(scores, dtype=np.float32)))
            for qid, scores in qid_scores.items()
        }
        nonzero_var_qids = {
            qid for qid, variance in qid_variance.items() if variance > 0.0
        }

        # Shortcut qids bypass mode-based filtering: they have no influence
        # score (Stage 4 dropped them), and re-ranking them against MCQ qids
        # via 0.0-default sort would kick them out under top_alpha. Always-keep
        # modulo zero-variance.
        shortcut_qid_set: set[str] = {
            row_qids[i]
            for i in range(batch_size)
            if row_is_shortcut[i] and valid_mask[i]
        }
        mcq_nonzero_var = nonzero_var_qids - shortcut_qid_set
        shortcut_keep = shortcut_qid_set & nonzero_var_qids

        keep_qids: set[str]
        total_questions = len(qid_scores)
        mcq_total_questions = total_questions - len(shortcut_qid_set)
        if mode == "none":
            mcq_keep = set(mcq_nonzero_var)
        elif mode in {"top_alpha", "random_alpha"}:
            target_count = min(
                len(mcq_nonzero_var),
                alpha_target_count(mcq_total_questions, alpha),
            )
            qid_list = list(mcq_nonzero_var)
            if mode == "random_alpha":
                random.shuffle(qid_list)
            else:
                qid_list.sort(
                    key=lambda qid: (-float(rewards.get(qid, 0.0)), qid)
                )
            mcq_keep = set(qid_list[:target_count])
        else:
            # random_i / top_i: pick up to i non-zero-variance questions per doc.
            per_doc_keep_count = get_per_doc_keep_count(mode)
            if per_doc_keep_count is None:
                raise ValueError(
                    "Invalid per-document solver filter mode after normalization: "
                    f"{mode!r}."
                )
            doc_to_qids: dict[str, list[str]] = defaultdict(list)
            for qid in mcq_nonzero_var:
                doc_to_qids[qid_to_doc[qid]].append(qid)

            mcq_keep = set()
            for _, qids in doc_to_qids.items():
                n_keep = min(per_doc_keep_count, len(qids))
                if mode.startswith("random_") or mode == "random_one":
                    random.shuffle(qids)
                    mcq_keep.update(qids[:n_keep])
                else:
                    qids.sort(key=lambda qid: (-float(rewards.get(qid, 0.0)), qid))
                    mcq_keep.update(qids[:n_keep])

        keep_qids = mcq_keep | shortcut_keep

        logger.info(
            "_build_solver_update_batch: mode=%s alpha=%.4f "
            "total_q=%d (mcq=%d shortcut=%d) nonzero_var=%d "
            "kept=%d (mcq_kept=%d shortcut_kept=%d)",
            mode,
            alpha,
            total_questions,
            mcq_total_questions,
            len(shortcut_qid_set),
            len(nonzero_var_qids),
            len(keep_qids),
            len(mcq_keep),
            len(shortcut_keep),
        )

        # --- Select valid rows ---
        selected_indices = [
            i for i in range(batch_size)
            if valid_mask[i] and row_qids[i] in keep_qids
        ]

        if not selected_indices:
            logger.warning(
                "_build_solver_update_batch: sample size changed from %d to 0 "
                "(all samples dropped by validity/zero-variance/solver filtering)",
                batch_size,
            )
            return None

        if len(selected_indices) != batch_size:
            logger.warning(
                "_build_solver_update_batch: sample size changed from %d to %d "
                "(dropped %d samples from validity/zero-variance/solver filtering)",
                batch_size,
                len(selected_indices),
                batch_size - len(selected_indices),
            )

        indices = torch.tensor(selected_indices, dtype=torch.long)
        input_ids = gen_answer_output.batch["input_ids"][indices]
        attention_mask = gen_answer_output.batch["attention_mask"][indices]
        position_ids = gen_answer_output.batch["position_ids"][indices]
        sel_responses = gen_answer_output.batch["responses"][indices]
        rollout_log_probs = None
        if "rollout_log_probs" in gen_answer_output.batch.keys():
            rollout_log_probs = gen_answer_output.batch["rollout_log_probs"][indices]

        sel_batch_size = len(selected_indices)
        sel_response_length = sel_responses.size(1)

        # Compute response_mask
        response_mask = attention_mask[:, -sel_response_length:].float()

        # Token-level rewards: broadcast correctness score to all response tokens
        sel_scores = torch.tensor(
            [row_scores[i] for i in selected_indices], dtype=torch.float32
        )
        token_level_scores = expand_scores_to_token_level(
            sel_scores, attention_mask, position_ids, sel_response_length,
        )

        # old_log_probs: placeholder (may be replaced by rollout correction prep)
        old_log_probs = torch.zeros(
            sel_batch_size, sel_response_length, dtype=torch.float32
        )

        tensors = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": sel_responses,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "token_level_scores": token_level_scores,
        }
        if rollout_log_probs is not None:
            tensors["rollout_log_probs"] = rollout_log_probs

        td = TensorDict(
            tensors,
            batch_size=sel_batch_size,
        )

        non_tensor_batch = {
            "uid": np.array(
                [row_qids[i] for i in selected_indices], dtype=object
            ),
            "doc_id": np.array(
                [
                    row_doc_ids[i] if row_doc_ids[i] is not None else row_qids[i]
                    for i in selected_indices
                ],
                dtype=object,
            ),
        }
        if answer_openers is not None:
            non_tensor_batch["opener_class"] = np.array(
                [row_openers[i] for i in selected_indices], dtype=object
            )

        meta_info = {
            "temperature": self.config.solver.rollout.temperature,
            "global_token_num": torch.sum(attention_mask, dim=-1).tolist(),
        }
        if formatting_bonus > 0.0:
            n_alright_kept = sum(
                1 for i in selected_indices if row_openers and row_openers[i] == "alright"
            )
            meta_info["solver_formatting_bonus/value"] = formatting_bonus
            meta_info["solver_formatting_bonus/n_applied"] = float(n_alright_kept)
            meta_info["solver_formatting_bonus/total_mass"] = (
                n_alright_kept * formatting_bonus
            )
            meta_info["solver_formatting_bonus/share_applied"] = (
                n_alright_kept / len(selected_indices) if selected_indices else 0.0
            )

        return DataProto(
            batch=td,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    def _build_dev_solver_update_batch(
        self, dev_output: DataProto,
    ) -> DataProto | None:
        """Build a solver PPO update batch from dev rollout data (dev_only mode).

        Simplified version of ``_build_solver_update_batch`` for dev_only mode:
        uses ``question_id`` and ``answer_score`` from dev_output directly,
        applies zero-variance filtering (no influence-based filtering), and
        produces the same output schema for downstream PPO.

        Args:
            dev_output: Solver rollout output from Stage 1, with
                ``answer_score`` populated by ``extract_answer_scores()``.

        Returns:
            DataProto for solver update, or ``None`` if empty after filtering.
        """
        responses = dev_output.batch["responses"]
        batch_size = responses.shape[0]

        question_ids = dev_output.non_tensor_batch.get("question_id", None)
        answer_scores = dev_output.non_tensor_batch.get("answer_score", None)
        answer_openers = dev_output.non_tensor_batch.get("opener_class", None)

        if question_ids is None:
            logger.warning("_build_dev_solver_update_batch: no question_id in dev_output")
            return None

        formatting_bonus = self._resolve_solver_formatting_bonus()
        mid_eos_penalty = self._resolve_solver_mid_eos_penalty()
        mid_eos_mask = (
            self._compute_mid_eos_mask(
                responses=responses,
                response_mask=dev_output.batch["response_mask"],
                answer_scores=answer_scores,
                eos_token_id=self.solver_tokenizer.eos_token_id,
            )
            if mid_eos_penalty != 0.0
            else np.zeros(batch_size, dtype=bool)
        )

        # --- Build per-row correctness scores ---
        row_scores = []
        row_qids = []
        row_openers = []
        valid_mask = []
        n_mid_eos = 0

        for i in range(batch_size):
            qid = str(question_ids[i])
            opener = (
                str(answer_openers[i]) if answer_openers is not None else None
            )
            bonus = formatting_bonus if opener == "alright" else 0.0
            penalty = mid_eos_penalty if mid_eos_mask[i] else 0.0
            if mid_eos_mask[i]:
                n_mid_eos += 1
            if answer_scores is None:
                row_scores.append(0.0 + bonus + penalty)
                row_qids.append(qid)
                if answer_openers is not None:
                    row_openers.append(opener)
                valid_mask.append(False)
                continue
            score = answer_scores[i]
            row_scores.append((float(score) if score is not None else 0.0) + bonus + penalty)
            row_qids.append(qid)
            if answer_openers is not None:
                row_openers.append(opener)
            valid_mask.append(True)

        if mid_eos_penalty != 0.0:
            logger.info(
                "_build_dev_solver_update_batch: mid_eos rows=%d/%d (penalty=%.4f applied)",
                n_mid_eos, batch_size, mid_eos_penalty,
            )

        # --- Zero-variance filtering ---
        qid_scores: dict[str, list[float]] = defaultdict(list)
        for qid, score, valid in zip(row_qids, row_scores, valid_mask):
            if valid:
                qid_scores[qid].append(score)

        nonzero_var_qids = {
            qid
            for qid, scores in qid_scores.items()
            if float(np.var(np.asarray(scores, dtype=np.float32))) > 0.0
        }

        total_questions = len(qid_scores)
        logger.info(
            "_build_dev_solver_update_batch: total_questions=%d "
            "nonzero_var_questions=%d",
            total_questions,
            len(nonzero_var_qids),
        )

        # --- Select valid rows ---
        selected_indices = [
            i for i in range(batch_size)
            if valid_mask[i] and row_qids[i] in nonzero_var_qids
        ]

        if not selected_indices:
            logger.warning(
                "_build_dev_solver_update_batch: all %d samples dropped "
                "by zero-variance filtering",
                batch_size,
            )
            return None

        if len(selected_indices) != batch_size:
            logger.warning(
                "_build_dev_solver_update_batch: sample size %d -> %d "
                "(dropped %d by zero-variance filtering)",
                batch_size,
                len(selected_indices),
                batch_size - len(selected_indices),
            )

        indices = torch.tensor(selected_indices, dtype=torch.long)
        input_ids = dev_output.batch["input_ids"][indices]
        attention_mask = dev_output.batch["attention_mask"][indices]
        position_ids = dev_output.batch["position_ids"][indices]
        sel_responses = dev_output.batch["responses"][indices]
        rollout_log_probs = None
        if "rollout_log_probs" in dev_output.batch.keys():
            rollout_log_probs = dev_output.batch["rollout_log_probs"][indices]

        sel_batch_size = len(selected_indices)
        sel_response_length = sel_responses.size(1)

        response_mask = attention_mask[:, -sel_response_length:].float()

        sel_scores = torch.tensor(
            [row_scores[i] for i in selected_indices], dtype=torch.float32
        )
        token_level_scores = expand_scores_to_token_level(
            sel_scores, attention_mask, position_ids, sel_response_length,
        )

        old_log_probs = torch.zeros(
            sel_batch_size, sel_response_length, dtype=torch.float32
        )

        tensors = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": sel_responses,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "token_level_scores": token_level_scores,
        }
        if rollout_log_probs is not None:
            tensors["rollout_log_probs"] = rollout_log_probs

        td = TensorDict(
            tensors,
            batch_size=sel_batch_size,
        )

        non_tensor_batch = {
            "uid": np.array(
                [row_qids[i] for i in selected_indices], dtype=object
            ),
            # Dev questions have no source document; treat each question as its
            # own "doc" so downstream stratify_by_doc becomes a no-op (1-Q-per-doc).
            "doc_id": np.array(
                [row_qids[i] for i in selected_indices], dtype=object
            ),
        }
        if answer_openers is not None:
            non_tensor_batch["opener_class"] = np.array(
                [row_openers[i] for i in selected_indices], dtype=object
            )

        meta_info = {
            "temperature": self.config.solver.rollout.temperature,
            "global_token_num": torch.sum(attention_mask, dim=-1).tolist(),
        }
        if formatting_bonus > 0.0:
            n_alright_kept = sum(
                1 for i in selected_indices if row_openers and row_openers[i] == "alright"
            )
            meta_info["solver_formatting_bonus/value"] = formatting_bonus
            meta_info["solver_formatting_bonus/n_applied"] = float(n_alright_kept)
            meta_info["solver_formatting_bonus/total_mass"] = (
                n_alright_kept * formatting_bonus
            )
            meta_info["solver_formatting_bonus/share_applied"] = (
                n_alright_kept / len(selected_indices) if selected_indices else 0.0
            )

        return DataProto(
            batch=td,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    @staticmethod
    def _truncate_batch_for_workers(
        batch: DataProto, worker_group, label: str = ""
    ) -> DataProto:
        """Truncate batch so its size is divisible by the number of workers."""
        ws = worker_group.world_size
        n = len(batch)
        if n % ws == 0:
            return batch
        new_n = (n // ws) * ws
        if new_n == 0:
            return batch  # let downstream raise if needed
        print(
            f"Truncating {label} batch from {n} to {new_n} (world_size={ws})",
            flush=True,
        )
        return batch[:new_n]

    def _maybe_apply_rollout_correction(
        self,
        batch: DataProto,
        worker_group: RayWorkerGroup,
        role_name: str,
    ) -> tuple[DataProto, dict]:
        """Compute old_log_probs and optionally apply rollout correction.

        Following upstream verl (ray_trainer.py), old_log_probs are always
        recomputed via a forward pass unless bypass mode is active.  This
        ensures the PPO ratio is well-defined regardless of whether rollout
        correction / TIS is configured.

        Returns:
            (batch, rollout_corr_metrics) — metrics dict is empty when
            rollout correction is skipped or in bypass mode.
        """
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)

        # --- Bypass mode: set old_log_probs = rollout_log_probs, no forward pass ---
        if rollout_corr_config is not None:
            bypass_mode = rollout_corr_config.get("bypass_mode", False)
            if bypass_mode:
                from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                apply_bypass_mode(
                    batch=batch,
                    rollout_corr_config=rollout_corr_config,
                    policy_loss_config=self.config[role_name].actor.policy_loss,
                )
                return batch, {}

        # --- Always recompute old_log_probs from current policy weights ---
        # Matches upstream verl/trainer/ppo/ray_trainer.py:1455-1473:
        # the else branch of `if bypass_recomputing_logprobs` always calls
        # _compute_old_log_prob() regardless of whether rollout_correction
        # is configured, ensuring the PPO ratio uses real log probs.
        old_log_prob = worker_group.compute_log_prob(batch)
        if "entropys" in old_log_prob.batch.keys():
            old_log_prob.batch.pop("entropys")
        # Remove stale old_log_probs from batch before union to avoid conflict
        if "old_log_probs" in batch.batch.keys():
            batch.batch.pop("old_log_probs")
        batch = batch.union(old_log_prob)

        # --- Apply rollout correction if configured ---
        if rollout_corr_config is None:
            return batch, {}

        if "rollout_log_probs" not in batch.batch.keys():
            logger.warning(
                "%s update: rollout_correction is configured but rollout_log_probs "
                "is missing from batch; skipping rollout correction",
                role_name,
            )
            return batch, {}

        from verl.trainer.ppo.rollout_corr_helper import (
            compute_rollout_correction_and_add_to_batch,
        )

        batch, rollout_corr_metrics = compute_rollout_correction_and_add_to_batch(
            batch, rollout_corr_config
        )
        logger.info(
            "%s rollout correction: rollout_is=%s, threshold=%s",
            role_name,
            rollout_corr_config.get("rollout_is", None),
            rollout_corr_config.get("rollout_is_threshold", None),
        )
        if rollout_corr_metrics:
            logger.debug("%s rollout correction metrics: %s", role_name, rollout_corr_metrics)

        return batch, rollout_corr_metrics

    def _use_global_batch_norm(self, role: str) -> bool:
        """Resolve per-role global batch normalization with shared fallback."""
        key_by_role = {
            "generator": "generator_use_global_batch_norm",
            "solver": "solver_use_global_batch_norm",
        }
        if role not in key_by_role:
            raise ValueError(f"Unsupported role for global batch norm: {role}")

        role_value = self.config.algorithm.get(key_by_role[role], None)
        if role_value is not None:
            return bool(role_value)
        return bool(self.config.algorithm.get("use_global_batch_norm", True))

    def compute_advantage(
        self, batch: DataProto, normalization_mode: Optional[str] = None
    ) -> DataProto:
        """Compute advantages for a PPO update batch.

        Delegates to verl's ``compute_advantage()`` (ray_trainer.py:130)
        with the configured advantage estimator (typically GRPO for
        self-evolution).

        Args:
            batch: DataProto with ``token_level_scores``, ``response_mask``,
                   and optionally ``uid`` in ``non_tensor_batch``.
            normalization_mode: Explicit normalization mode override.
                ``"group_std"`` for standard GRPO, ``"hybrid_std"`` for
                dampened generator advantages, ``"batch_std"`` to divide by
                the std of centered advantages across the whole batch,
                ``"none"`` for Dr.GRPO style.
                When ``None``, falls back to ``norm_adv_by_std_in_grpo``.

        Returns:
            DataProto augmented with ``advantages`` and ``returns``.

        Note:
            Input ``token_level_rewards`` are EOS-only (from
            ``expand_scores_to_token_level``). GRPO internally sums across
            tokens to recover the scalar, normalizes within prompt groups,
            then broadcasts the result across all response tokens via
            ``response_mask``.
        """
        adv_estimator = AdvantageEstimator(
            self.config.algorithm.get("adv_estimator", "grpo")
        )
        # verl's compute_advantage expects "token_level_rewards" but our
        # _prepare_influence_batch creates "token_level_scores".  Add alias if needed.
        if "token_level_scores" in batch.batch.keys() and "token_level_rewards" not in batch.batch.keys():
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        return compute_advantage(
            data=batch,
            adv_estimator=adv_estimator,
            gamma=self.config.algorithm.get("gamma", 1.0),
            lam=self.config.algorithm.get("lam", 1.0),
            norm_adv_by_std_in_grpo=self.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True # the norm_adv_by_std_in_grpo acts as a fall back if normalization_mode is not explicitly set. In the current codebase, we only rely on normalization_mode
            ),
            normalization_mode=normalization_mode,
        )

    def _build_advantage_component_batch(
        self,
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        uids: np.ndarray,
    ) -> DataProto:
        """Build a minimal DataProto for component-wise GRPO advantage computation."""
        td = TensorDict(
            {
                "token_level_rewards": token_level_rewards,
                "response_mask": response_mask,
            },
            batch_size=token_level_rewards.shape[0],
        )
        non_tensor_batch = {"uid": np.array(uids, dtype=object)}
        return DataProto(batch=td, non_tensor_batch=non_tensor_batch, meta_info={})

    def compute_generator_advantage(
        self, batch: DataProto
    ) -> tuple[DataProto, dict[str, float]]:
        """Compute generator advantages under configured reward-combination mode.

        Normalization mode selection depends on the combination mode:

        **sum_scores**: uses ``algorithm.generator_normalization_mode``.

        **decoupled + generator_reward_components** (legacy):
          - ``influence_rewards`` → ``influence.normalization_mode``
          - ``spice_rewards`` → ``spice.normalization_mode``
          - others → ``algorithm.generator_normalization_mode``

        **decoupled + generator_reward_structure**:
          - Each group reads ``normalization_mode`` from the structure dict.
            Defaults: ``hybrid_std`` if group contains ``influence_rewards``,
            ``group_std`` otherwise.
          - Batch-wise normalization (Eq. 6) always applies after combining groups.

        In ``sum_scores`` mode, delegates directly to ``compute_advantage()``.

        In ``decoupled`` mode (GDPO, arxiv 2601.05242):
          1. Per-group GRPO normalization (Eq. 4): terms inside a reward
             group are first summed at the raw-reward level, then the group's
             token-level scores are independently normalized within prompt
             groups using the group's ``normalization_mode``.
          2. Sum weighted normalized advantages across groups (Eq. 5).
          3. Batch-wise normalization (Eq. 6): subtract batch mean and
             divide by batch std over all response tokens.

        Note:
            Input ``token_level_scores_*`` tensors have the scalar reward
            placed at the EOS token only (via ``expand_scores_to_token_level``).
            GRPO's ``sum(dim=-1)`` recovers the original scalar. The output
            ``advantages`` are then broadcast uniformly across all response
            tokens (masked by ``response_mask``).
        """
        # Read per-component normalization modes from config.
        global_norm = self.config.algorithm.get("generator_normalization_mode", "group_std")
        influence_norm = self.config.influence.get("normalization_mode", "hybrid_std")
        spice_norm = self.config.spice.get("normalization_mode", "none")

        metrics: dict[str, float] = {
            "gen_adv/decoupled_enabled": 0.0,
            "gen_adv/decoupled_active": 0.0,
        }
        reward_combination_mode = resolve_generator_reward_combination_mode(
            self.config.training
        )
        if reward_combination_mode != "decoupled":
            return self.compute_advantage(batch, normalization_mode=global_norm), metrics

        metrics["gen_adv/decoupled_enabled"] = 1.0
        if str(self.config.algorithm.get("adv_estimator", "grpo")).lower() != "grpo":
            logger.warning(
                "Generator decoupled advantage currently assumes GRPO; adv_estimator=%s. "
                "Falling back to standard compute_advantage().",
                self.config.algorithm.get("adv_estimator", "grpo"),
            )
            metrics["gen_adv/decoupled_fallback_non_grpo"] = 1.0
            return self.compute_advantage(batch, normalization_mode=global_norm), metrics

        component_to_tensor = {
            "influence_rewards": "token_level_scores_influence",
            "spice_rewards": "token_level_scores_spice",
            "invalid_rewards": "token_level_scores_invalid",
        }
        reward_structure = batch.meta_info.get("reward_structure_for_adv", None)
        if reward_structure is None:
            reward_structure = resolve_generator_reward_structure(
                self.config.training
            )
        else:
            reward_structure = validate_generator_reward_structure(
                list(reward_structure)
            )

        configured_components = batch.meta_info.get("reward_components_for_adv", None)
        if configured_components is None:
            configured_components = resolve_generator_reward_components(
                self.config.training
            )
        elif isinstance(configured_components, str):
            configured_components = [configured_components]
        else:
            configured_components = [str(x) for x in configured_components]
        configured_components = validate_generator_reward_components(
            list(configured_components)
        )
        component_term_weights = {
            str(term["name"]): float(term["weight"])
            for group in reward_structure
            for term in group["terms"]
        }

        missing_groups = []
        for group_idx, group in enumerate(reward_structure):
            group_key = f"token_level_scores_group_{group_idx}"
            if group_key in batch.batch.keys():
                continue
            missing_members = [
                str(term["name"])
                for term in group["terms"]
                if component_to_tensor[str(term["name"])] not in batch.batch.keys()
            ]
            if missing_members:
                missing_groups.append((group_idx, missing_members))
        if missing_groups:
            logger.warning(
                "Generator decoupled advantage is enabled but tensors are missing "
                "for reward groups %s; falling back to standard compute_advantage().",
                missing_groups,
            )
            metrics["gen_adv/decoupled_fallback_missing_components"] = 1.0
            return self.compute_advantage(batch, normalization_mode=global_norm), metrics

        response_mask = batch.batch["response_mask"]
        uids = batch.non_tensor_batch["uid"]

        def _format_group_label(group: dict[str, Any]) -> str:
            parts: list[str] = []
            for term in group["terms"]:
                short = str(term["name"]).removesuffix("_rewards")
                term_weight = float(term["weight"])
                if abs(term_weight - 1.0) < 1e-12:
                    parts.append(short)
                else:
                    parts.append(f"{term_weight:g}x_{short}")
            return "+".join(parts)

        group_batches: list[DataProto] = []
        group_weights: list[float] = []
        for group_idx, group in enumerate(reward_structure):
            group_key = f"token_level_scores_group_{group_idx}"
            if group_key in batch.batch.keys():
                group_token_rewards = batch.batch[group_key]
            else:
                first_term = group["terms"][0]
                group_token_rewards = (
                    float(first_term["weight"])
                    * batch.batch[component_to_tensor[str(first_term["name"])]]
                ).clone()
                for term in group["terms"][1:]:
                    group_token_rewards = (
                        group_token_rewards
                        + float(term["weight"])
                        * batch.batch[component_to_tensor[str(term["name"])]]
                    )

            group_batch = self._build_advantage_component_batch(
                token_level_rewards=group_token_rewards,
                response_mask=response_mask,
                uids=uids,
            )
            # Per-group normalization_mode: read from structure if provided,
            # otherwise default to hybrid_std for influence groups, group_std for others.
            if "normalization_mode" in group:
                group_norm_mode = str(group["normalization_mode"])
            else:
                group_has_influence = any(
                    str(term["name"]) == "influence_rewards" for term in group["terms"]
                )
                group_norm_mode = "hybrid_std" if group_has_influence else "group_std"
            group_batch = self.compute_advantage(
                group_batch,
                normalization_mode=group_norm_mode,
            )
            group_batches.append(group_batch)
            group_weights.append(float(group["group_weight"]))

        combined_advantages = group_weights[0] * group_batches[0].batch["advantages"].clone()
        combined_returns = group_weights[0] * group_batches[0].batch["returns"].clone()
        for group_idx in range(1, len(group_batches)):
            combined_advantages = (
                combined_advantages
                + group_weights[group_idx] * group_batches[group_idx].batch["advantages"]
            )
            combined_returns = (
                combined_returns
                + group_weights[group_idx] * group_batches[group_idx].batch["returns"]
            )

        # GDPO Eq. 6: batch-wise normalization of summed advantages.
        # Normalize at the sample level (one scalar per response), not
        # per-token.  GRPO broadcasts a uniform value across response
        # tokens, so we recover the per-sample scalar, normalize across
        # all samples, then broadcast back.
        sample_adv = (combined_advantages * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
        n_samples = sample_adv.shape[0]
        if n_samples > 1:
            batch_mean = sample_adv.mean()
            batch_std = sample_adv.std() + 1e-8
            sample_adv = (sample_adv - batch_mean) / batch_std
        combined_advantages = sample_adv.unsqueeze(-1) * response_mask

        # Keep the full mixed reward signal for data metrics, but use summed
        # group advantages for policy update.
        if "token_level_rewards" not in batch.batch.keys():
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        batch.batch["advantages"] = combined_advantages
        batch.batch["returns"] = combined_returns

        for group_idx, group in enumerate(reward_structure):
            group_scalar = scalarize_masked_sequence_values(
                group_batches[group_idx].batch["advantages"], response_mask
            )
            group_name = _format_group_label(group)
            metrics[f"gen_adv/decoupled_group/{group_name}_mean"] = float(
                group_scalar.mean().item()
            )
            metrics[f"gen_adv/decoupled_group/{group_name}_abs_mean"] = float(
                group_scalar.abs().mean().item()
            )
            metrics[f"gen_adv/decoupled_group/{group_name}_weight"] = float(
                group_weights[group_idx]
            )

        for component in configured_components:
            short = component.removesuffix("_rewards")
            tensor_key = component_to_tensor[component]
            if tensor_key not in batch.batch.keys():
                continue
            component_token_rewards = batch.batch[tensor_key]
            if component in component_term_weights:
                component_token_rewards = (
                    component_term_weights[component] * component_token_rewards
                )
            comp_batch = self._build_advantage_component_batch(
                token_level_rewards=component_token_rewards,
                response_mask=response_mask,
                uids=uids,
            )
            # Per-component normalization: influence, spice, and others each
            # have their own configurable normalization mode.
            if component == "influence_rewards":
                comp_norm_mode = influence_norm
            elif component == "spice_rewards":
                comp_norm_mode = spice_norm
            else:
                comp_norm_mode = global_norm
            comp_batch = self.compute_advantage(
                comp_batch,
                normalization_mode=comp_norm_mode,
            )
            scalar = scalarize_masked_sequence_values(
                comp_batch.batch["advantages"], response_mask
            )
            metrics[f"gen_adv/decoupled_component/{short}_mean"] = float(scalar.mean().item())
            metrics[f"gen_adv/decoupled_component/{short}_abs_mean"] = float(
                scalar.abs().mean().item()
            )
            metrics[f"gen_adv/decoupled_component/{short}_weight"] = float(
                component_term_weights.get(component, 1.0)
            )

        combined_scalar = scalarize_masked_sequence_values(
            combined_advantages, response_mask
        )
        metrics.update(
            {
                "gen_adv/decoupled_active": 1.0,
                "gen_adv/decoupled_component/count": float(len(configured_components)),
                "gen_adv/decoupled_group/count": float(len(reward_structure)),
                "gen_adv/decoupled_component/combined_mean": float(combined_scalar.mean().item()),
                "gen_adv/decoupled_component/combined_abs_mean": float(combined_scalar.abs().mean().item()),
            }
        )
        return batch, metrics

    def maybe_refresh_curriculum(self, ans_loop: int) -> None:
        """Refresh the dev set via curriculum learning if enabled.

        When curriculum learning is active, periodically removes questions
        that are too easy or too hard and samples replacements from the
        data pool.

        Algorithm (v2_key_points #18):
          1. Compute mean accuracy per question from recorded history.
          2. Sort by accuracy; remove bottom X% (too hard) + top X% (too easy).
          3. Sample replacements from the data pool (excluding used questions).
          4. Update ``self.dev_questions`` in place and save state to disk.

        Reference: v2 ``train.py:1291-1462`` (refresh_curriculum_dev_set)

        Args:
            ans_loop: Current answer loop index.
        """
        if not self.config.curriculum.enabled:
            return
        if ans_loop == 0:
            return
        if self._curriculum_state is None or self._data_pool is None:
            return

        last_refresh = self._curriculum_state["last_refresh_ans_loop"]
        loops_since_refresh = ans_loop - last_refresh - 1
        if loops_since_refresh < self.config.curriculum.refresh_every_n_ans_loops:
            return

        logger.info("Curriculum: refreshing dev set at ans_loop=%d", ans_loop)

        # Compute mean accuracy per current dev question
        dev_with_accuracy = []
        for q in self.dev_questions:
            qid = q["question_id"]
            history = self._dev_accuracy_history.get(qid, [])
            if history:
                mean_acc = sum(history) / len(history)
            else:
                mean_acc = 0.5  # Default for questions with no data
            dev_with_accuracy.append((q, mean_acc))

        dev_with_accuracy.sort(key=lambda x: x[1])

        # Determine how many to remove
        num_questions = len(dev_with_accuracy)
        num_remove_bottom = int(num_questions * self.config.curriculum.remove_bottom_ratio)
        num_remove_top = int(num_questions * self.config.curriculum.remove_top_ratio)

        if num_remove_bottom + num_remove_top == 0:
            logger.info("Curriculum: no questions to remove (ratios too small)")
            self._curriculum_state["last_refresh_ans_loop"] = ans_loop
            self._save_curriculum_state()
            return

        # Identify questions to remove
        bottom_questions = dev_with_accuracy[:num_remove_bottom]
        top_questions = dev_with_accuracy[-num_remove_top:] if num_remove_top > 0 else []
        remove_ids = {q["question_id"] for q, _ in bottom_questions + top_questions}

        logger.info(
            "Curriculum: removing %d bottom (acc < %.3f) + %d top (acc > %.3f) questions",
            num_remove_bottom,
            bottom_questions[-1][1] if bottom_questions else 0.0,
            num_remove_top,
            top_questions[0][1] if top_questions else 1.0,
        )

        # Keep questions not being removed
        remaining = [q for q, _ in dev_with_accuracy if q["question_id"] not in remove_ids]
        remaining_ids = {q["question_id"] for q in remaining}

        # Update used_question_ids
        self._curriculum_state["used_question_ids"].update(remove_ids)
        used_ids = self._curriculum_state["used_question_ids"]

        # Sample replacements from pool
        available = [
            q for q in self._data_pool
            if q["question_id"] not in used_ids
            and q["question_id"] not in remaining_ids
        ]

        num_to_sample = min(len(remove_ids), len(available))
        if num_to_sample > 0:
            random.seed(self.config.training.seed + ans_loop)
            new_questions = random.sample(available, num_to_sample)
            logger.info("Curriculum: sampled %d new questions (pool remaining: %d)",
                         num_to_sample, len(available) - num_to_sample)
        else:
            new_questions = []
            logger.warning("Curriculum: no questions available in pool to sample")

        # Update dev set
        updated_dev = remaining + new_questions
        self.dev_questions = updated_dev

        # Clean up accuracy history for removed questions
        for qid in remove_ids:
            self._dev_accuracy_history.pop(qid, None)

        # Update and save state
        self._curriculum_state["last_refresh_ans_loop"] = ans_loop
        self._curriculum_state["current_dev_question_ids"] = [
            q["question_id"] for q in updated_dev
        ]

        # Save updated dev set to disk
        local_dir = self.config.training.default_local_dir
        curriculum_dev_path = os.path.join(local_dir, "curriculum_dev.json")
        with open(curriculum_dev_path, "w") as f:
            json.dump(updated_dev, f, indent=2)

        self._save_curriculum_state()

        logger.info("Curriculum: updated dev set has %d questions", len(updated_dev))

    def _maybe_download_curriculum_from_r2(
        self,
        local_dir: str,
        state_path: str,
        curriculum_dev_path: str,
    ) -> None:
        """Try to download curriculum files from R2 when missing locally.

        No-op when the upload manager is not configured or has no remote.
        """
        if (
            not hasattr(self, "_upload_manager")
            or self._upload_manager is None
            or not self._upload_manager.remote_configured
        ):
            return

        resolver = StorageResolver(
            local_base=local_dir,
            upload_manager=self._upload_manager,
            remote_prefix="",
            resolve_order="local_first",
        )

        if not os.path.exists(state_path):
            logger.info(
                "Curriculum: curriculum_state.json not found locally — checking R2..."
            )
            resolved = resolver.resolve_file("curriculum_state.json")
            if resolved:
                logger.info(
                    "Curriculum: downloaded curriculum_state.json from R2 → %s",
                    resolved,
                )
            else:
                logger.info(
                    "Curriculum: curriculum_state.json not found on R2 either"
                )

        if not os.path.exists(curriculum_dev_path):
            logger.info(
                "Curriculum: curriculum_dev.json not found locally — checking R2..."
            )
            resolved = resolver.resolve_file("curriculum_dev.json")
            if resolved:
                logger.info(
                    "Curriculum: downloaded curriculum_dev.json from R2 → %s",
                    resolved,
                )
            else:
                logger.info(
                    "Curriculum: curriculum_dev.json not found on R2 either"
                )

    def _save_curriculum_state(self) -> None:
        """Save curriculum state to disk and upload to R2 for resume support."""
        if self._curriculum_state is None:
            return
        local_dir = self.config.training.default_local_dir
        state_path = os.path.join(local_dir, "curriculum_state.json")
        curriculum_dev_path = os.path.join(local_dir, "curriculum_dev.json")
        os.makedirs(local_dir, exist_ok=True)

        state_to_save = {
            "used_question_ids": list(self._curriculum_state["used_question_ids"]),
            "last_refresh_ans_loop": self._curriculum_state["last_refresh_ans_loop"],
            "current_dev_question_ids": self._curriculum_state.get(
                "current_dev_question_ids", []
            ),
            "accuracy_history": self._dev_accuracy_history,
        }
        with open(state_path, "w") as f:
            json.dump(state_to_save, f, indent=2)

        # Upload both curriculum files to R2 for crash recovery
        if (
            hasattr(self, "_upload_manager")
            and self._upload_manager is not None
            and self._upload_manager.upload_enabled
        ):
            self._upload_manager.submit_file_upload(
                local_path=state_path,
                remote_key="curriculum_state.json",
            )
            if os.path.exists(curriculum_dev_path):
                self._upload_manager.submit_file_upload(
                    local_path=curriculum_dev_path,
                    remote_key="curriculum_dev.json",
                )

    # ------------------------------------------------------------------
    # Stage context mark_done helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_stage_2_done(gl) -> None:
        """Mark Stage 2 done — sets the gen_loop flag."""
        gl.stage_2_done = True

    # ------------------------------------------------------------------
    # Stage-level resume helpers
    # ------------------------------------------------------------------

    def _restore_doc_dataset_state(
        self, resume: ResumeState, ans_loop: int
    ) -> None:
        """Restore ``doc_dataset`` position for the current ans_loop.

        Two cases:

        1. **Mid-ans_loop resume** — ``resume.doc_dataset_state`` exists
           (persisted after each Stage 2).  Use it directly since it accounts
           for partially completed gen_loops.

        2. **Cross-ans_loop resume** — ``doc_dataset_state`` is ``None``
           (cleared after successful checkpoint) and ``ans_loop > 0``.
           Compute the position deterministically: every prior gen_loop
           called ``next_batch()`` exactly once, so
           ``total_batches = ans_loop * num_gen_per_ans``.
        """
        if resume.doc_dataset_state is not None:
            self.doc_dataset.restore_state(resume.doc_dataset_state)
            logger.info(
                "Restored doc_dataset state: position=%d, epoch=%d",
                resume.doc_dataset_state["position"],
                resume.doc_dataset_state["epoch"],
            )
        elif ans_loop > 0:
            total_batches = ans_loop * self.num_gen_per_ans
            total_consumed = total_batches * self.doc_dataset.batch_size
            total_docs = len(self.doc_dataset)
            computed_state = {
                "position": total_consumed % total_docs,
                "epoch": total_consumed // total_docs,
            }
            self.doc_dataset.restore_state(computed_state)
            logger.info(
                "Computed doc_dataset state from ans_loop=%d: "
                "position=%d, epoch=%d (total_batches=%d)",
                ans_loop, computed_state["position"],
                computed_state["epoch"], total_batches,
            )

    def _replay_generator_updates(
        self, resume: ResumeState, resume_dir: str
    ) -> None:
        """Replay generator PPO updates for completed gen_loops on resume.

        When resuming mid-ans-loop, the model was loaded from the last
        ans_loop checkpoint.  Any generator PPO updates that completed
        before the crash must be replayed to bring weights to the correct
        state.

        Args:
            resume: Current ``ResumeState`` with ``gen_loops`` populated.
            resume_dir: Path to the resume state directory.
        """
        if self.config.training.fix_generator:
            return

        completed = [gl for gl in resume.gen_loops if gl.stage_5_done]
        if not completed:
            return

        print(
            f"========== RESUME: REPLAYING {len(completed)} GENERATOR PPO UPDATE(S) ==========",
            flush=True,
        )

        _replay_total_start = time.perf_counter()
        for idx, gl in enumerate(completed, 1):
            print(
                f"  Replay [{idx}/{len(completed)}]: Loading gen_output for local_gen={gl.local_gen} ...",
                flush=True,
            )
            _load_start = time.perf_counter()
            gen_output = self._load_gen_data_with_r2_fallback(
                resume, resume_dir, gl.local_gen, "gen_output", "dataproto"
            )
            print(
                f"  Replay [{idx}/{len(completed)}]: gen_output loaded ({time.perf_counter() - _load_start:.1f}s)",
                flush=True,
            )
            try:
                rewards = self._load_gen_data_with_r2_fallback(
                    resume, resume_dir, gl.local_gen, "rewards", "json"
                )
            except Exception:
                print(
                    f"  Replay [{idx}/{len(completed)}]: rewards file not found, using in-state rewards",
                    flush=True,
                )
                rewards = gl.rewards
            if rewards is None:
                print(
                    f"  Replay [{idx}/{len(completed)}]: Skipping local_gen={gl.local_gen} (no rewards available)",
                    flush=True,
                )
                continue

            gen_update_batch, _ = self.prepare_gen_update_batch(
                gen_output, rewards
            )
            if gen_update_batch is not None:
                gen_update_batch = self._truncate_batch_for_workers(
                    gen_update_batch, self.generator_wg, "gen_update_replay"
                )
                gen_update_batch, _ = self._maybe_apply_rollout_correction(
                    batch=gen_update_batch,
                    worker_group=self.generator_wg,
                    role_name="generator",
                )
                gen_update_batch, _ = self.compute_generator_advantage(
                    gen_update_batch,
                )
                self.generator_wg.update_actor(gen_update_batch)
                print(
                    f"  Replay [{idx}/{len(completed)}]: Generator PPO update applied for local_gen={gl.local_gen}",
                    flush=True,
                )
            else:
                print(
                    f"  Replay [{idx}/{len(completed)}]: No valid batch for local_gen={gl.local_gen} — skipped",
                    flush=True,
                )
        print(
            f"========== RESUME: REPLAY COMPLETE ({time.perf_counter() - _replay_total_start:.1f}s total) ==========",
            flush=True,
        )

    def _rebuild_gen_loop_results(
        self, resume: ResumeState, resume_dir: str
    ) -> list[dict]:
        """Rebuild ``gen_loop_results`` from completed gen_loops in resume state.

        For Stage 6 (solver PPO update), we need the accumulated results
        from all gen_loops.  On resume, completed gen_loops' data is
        loaded from disk.

        Args:
            resume: Current ``ResumeState``.
            resume_dir: Path to the resume state directory.

        Returns:
            List of dicts matching the ``gen_loop_results`` format.
        """
        results: list[dict] = []
        completed = [gl for gl in resume.gen_loops if gl.stage_5_done]
        if not completed:
            return results

        print(
            f"========== RESUME: REBUILDING {len(completed)} GEN LOOP RESULT(S) ==========",
            flush=True,
        )
        _rebuild_start = time.perf_counter()
        for idx, gl in enumerate(completed, 1):
            print(
                f"  Rebuild [{idx}/{len(completed)}]: Loading data for local_gen={gl.local_gen} ...",
                flush=True,
            )
            _load_start = time.perf_counter()
            gen_answer_output = self._load_gen_data_with_r2_fallback(
                resume, resume_dir, gl.local_gen, "gen_answer_output", "dataproto"
            )
            gen_output = self._load_gen_data_with_r2_fallback(
                resume, resume_dir, gl.local_gen, "gen_output", "dataproto"
            )
            gen_questions = derive_gen_questions(gen_output)
            try:
                rewards = self._load_gen_data_with_r2_fallback(
                    resume, resume_dir, gl.local_gen, "rewards", "json"
                )
            except Exception:
                print(
                    f"  Rebuild [{idx}/{len(completed)}]: rewards file not found, using in-state rewards",
                    flush=True,
                )
                rewards = gl.rewards
            solver_filter_rewards = extract_influence_rewards_for_solver_filter(
                rewards
            )
            results.append(
                {
                    "gen_answer_output": gen_answer_output,
                    "rewards": solver_filter_rewards,
                    "gen_questions": gen_questions,
                }
            )
            print(
                f"  Rebuild [{idx}/{len(completed)}]: local_gen={gl.local_gen} loaded "
                f"({time.perf_counter() - _load_start:.1f}s)",
                flush=True,
            )
        print(
            f"========== RESUME: REBUILT {len(results)} GEN LOOP RESULT(S) "
            f"({time.perf_counter() - _rebuild_start:.1f}s total) ==========",
            flush=True,
        )
        return results

    def _load_gen_data_with_r2_fallback(
        self,
        resume: ResumeState,
        resume_dir: str,
        local_gen: int,
        name: str,
        kind: str = "dataproto",
    ) -> Any:
        """Load gen-loop data, trying R2 first then falling back to local disk.

        Args:
            resume: Current ``ResumeState``.
            resume_dir: Path to the resume state directory.
            local_gen: Gen loop index.
            name: Output name (e.g. ``"gen_output"``, ``"gen_answer_output"``).
            kind: ``"dataproto"`` or ``"json"``.

        Returns:
            The loaded data object.
        """
        local_base = os.path.join(resume_dir, f"gen_{local_gen}")
        remote_prefix = f"ans_{resume.ans_loop}/gen_{local_gen}"
        resolver = StorageResolver(
            local_base=local_base,
            upload_manager=self._upload_manager,
            remote_prefix=remote_prefix,
            resolve_order="remote_first",
        )
        ext = ".json" if kind == "json" else ".pt"
        filename = f"{name}{ext}"
        logger.debug(
            "Resolving %s (kind=%s) | local=%s/%s | remote=%s/%s",
            name, kind, local_base, filename, remote_prefix, filename,
        )
        _resolve_start = time.perf_counter()
        if kind == "json":
            data = resolver.resolve_json(filename)
        else:
            data = resolver.resolve_dataproto(filename)
        if data is None:
            raise FileNotFoundError(
                f"Could not resolve {filename} from local ({local_base}) or R2 ({remote_prefix})"
            )
        logger.debug(
            "Resolved %s in %.1fs", filename, time.perf_counter() - _resolve_start,
        )
        return data

    @staticmethod
    def _checkpoint_diag(label: str) -> None:
        """Log CPU memory, GPU memory, and disk free space for checkpoint diagnostics."""
        parts = []
        try:
            import psutil

            # System-wide memory
            mem = psutil.virtual_memory()
            parts.append(
                f"sys_mem: used={mem.used / 1024**3:.1f} GB / total={mem.total / 1024**3:.1f} GB ({mem.percent:.0f}%)"
            )

            # Per-process RSS (more accurate for this process)
            proc = psutil.Process()
            rss = proc.memory_info().rss
            parts.append(f"proc_rss={rss / 1024**3:.1f} GB")

            # Disk
            disk = psutil.disk_usage("/")
            parts.append(f"disk: free={disk.free / 1024**3:.1f} GB / total={disk.total / 1024**3:.1f} GB")
        except ImportError:
            parts.append("psutil not available")
        except Exception as e:
            parts.append(f"psutil error: {e}")

        # GPU memory (controller process — typically small, but log anyway)
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    alloc = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    parts.append(f"gpu{i}: alloc={alloc:.1f} GB, reserved={reserved:.1f} GB")
        except Exception:
            pass

        logger.info("[ckpt-diag %s] %s", label, ", ".join(parts))

    def _upload_queue_status_str(self) -> str:
        """Return a compact upload-queue snapshot for cross-stage diagnostics."""
        if self._upload_manager is None:
            return "disabled"

        try:
            qs = self._upload_manager.queue_status()
        except Exception as exc:
            return f"unavailable({exc})"

        return (
            f"pending={qs.get('pending', '?')}, "
            f"bytes={qs.get('pending_bytes', 0) / 1024**3:.2f}GB, "
            f"completed={qs.get('completed', '?')}, "
            f"failed={qs.get('failed', '?')}"
        )

    @staticmethod
    def _describe_batch_shapes(batch: Optional[DataProto]) -> str:
        """Return lightweight tensor-shape diagnostics for a DataProto batch."""
        if batch is None:
            return "batch=None"

        try:
            parts: list[str] = []
            if "input_ids" in batch.batch.keys():
                parts.append(f"input_ids={tuple(batch.batch['input_ids'].shape)}")
            if "responses" in batch.batch.keys():
                parts.append(f"responses={tuple(batch.batch['responses'].shape)}")
            if "response_mask" in batch.batch.keys():
                parts.append(
                    f"response_mask={tuple(batch.batch['response_mask'].shape)}"
                )
            if "advantages" in batch.batch.keys():
                parts.append(f"advantages={tuple(batch.batch['advantages'].shape)}")
            if "uid" in batch.non_tensor_batch:
                parts.append(f"uids={len(batch.non_tensor_batch['uid'])}")
            if "mini_batch_size" in batch.meta_info:
                parts.append(
                    f"mini_batch_size={batch.meta_info['mini_batch_size']}"
                )
            return ", ".join(parts) if parts else "batch=present"
        except Exception as exc:
            return f"batch=unavailable({exc})"

    def _log_influence_probe(
        self,
        phase: str,
        step: str,
        *,
        batch: Optional[DataProto] = None,
        question_count: Optional[int] = None,
        rollout_n: Optional[int] = None,
        dp_size: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log substep boundaries inside influence scoring with queue context."""
        parts = [f"phase={phase}", f"step={step}"]
        if question_count is not None:
            parts.append(f"questions={question_count}")
        if rollout_n is not None:
            parts.append(f"rollout_n={rollout_n}")
        if dp_size is not None:
            parts.append(f"dp_size={dp_size}")
        if batch is not None:
            parts.append(self._describe_batch_shapes(batch))
        if extra:
            for key, value in extra.items():
                parts.append(f"{key}={value}")
        parts.append(f"uploads={self._upload_queue_status_str()}")
        logger.info("[influence-probe] %s", " | ".join(parts))

    # ------------------------------------------------------------------
    # Preemption handler & heartbeat
    # ------------------------------------------------------------------

    def _install_sigterm_handler(self):
        """Install a SIGTERM handler that logs a snapshot before exiting.

        On spot instance preemption, the cloud provider sends SIGTERM before
        hard-killing the process.  This handler logs the current training
        state so the preemption point is visible in the log archive.
        """
        prev_handler = signal.getsignal(signal.SIGTERM)

        def _on_sigterm(signum, frame):
            p = self._training_progress
            ckpt_age = "N/A"
            if p["last_checkpoint_time"] is not None:
                ckpt_age = f"{time.time() - p['last_checkpoint_time']:.0f}s ago"

            upload_status = "N/A"
            if self._upload_manager is not None:
                try:
                    qs = self._upload_manager.queue_status()
                    upload_status = (
                        f"pending={qs.get('pending', '?')}, "
                        f"bytes={qs.get('pending_bytes', 0) / 1024**3:.1f}GB"
                    )
                except Exception:
                    upload_status = "unavailable"

            logger.critical(
                "====== PREEMPTION SNAPSHOT (SIGTERM received) ======\n"
                "  ans_loop=%s  local_gen=%s  stage=%s\n"
                "  last_checkpoint: ans_loop=%s (%s)\n"
                "  upload_queue: %s\n"
                "  pending_ckpt_upload: %s (ans_loop=%s)\n"
                "  wall_time_since_fit_start: %.0fs\n"
                "====================================================",
                p["ans_loop"], p["local_gen"], p["stage"],
                p["last_checkpoint_ans_loop"], ckpt_age,
                upload_status,
                self._pending_ckpt_upload_id, self._pending_ckpt_ans_loop,
                time.time() - (p["fit_start_time"] or time.time()),
            )
            # Flush so the snapshot survives even if killed shortly after
            logging.shutdown()

            # Re-raise to previous handler (or default)
            if callable(prev_handler):
                prev_handler(signum, frame)
            else:
                raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _on_sigterm)

    def _update_progress(self, *, ans_loop=None, local_gen=None, stage=None):
        """Update the mutable training progress dict (used by SIGTERM handler)."""
        p = self._training_progress
        if ans_loop is not None:
            p["ans_loop"] = ans_loop
        if local_gen is not None:
            p["local_gen"] = local_gen
        if stage is not None:
            p["stage"] = stage

    def _maybe_log_heartbeat(self):
        """Log a one-line heartbeat if enough time has passed since the last one."""
        now = time.time()
        if now - self._last_heartbeat_time < self._heartbeat_interval:
            return
        self._last_heartbeat_time = now

        p = self._training_progress
        ckpt_age = "none"
        if p["last_checkpoint_time"] is not None:
            ckpt_age = f"{now - p['last_checkpoint_time']:.0f}s ago"

        upload_info = ""
        if self._upload_manager is not None:
            try:
                qs = self._upload_manager.queue_status()
                upload_info = f" | uploads: pending={qs.get('pending', '?')}"
            except Exception:
                pass

        wall = now - (p["fit_start_time"] or now)
        logger.info(
            "[heartbeat] ans=%s gen=%s stage=%s | last_ckpt=%s (%s) | wall=%.0fs%s",
            p["ans_loop"], p["local_gen"], p["stage"],
            p["last_checkpoint_ans_loop"], ckpt_age,
            wall, upload_info,
        )

    def _ppo_stage_diag(
        self,
        label: str,
        *,
        role_name: str,
        batch: Optional[DataProto] = None,
        actor_output: Optional[DataProto] = None,
        ans_loop: Optional[int] = None,
        gen_loop: Optional[int] = None,
    ) -> None:
        """Log compact trainer-side diagnostics around PPO stages."""
        diag: dict[str, Any] = {"role": role_name}
        if ans_loop is not None:
            diag["ans_loop"] = ans_loop
        if gen_loop is not None:
            diag["gen_loop"] = gen_loop

        diag.update(collect_ppo_batch_diag(batch))
        diag.update(collect_ppo_actor_diag(actor_output))
        diag.update(collect_process_resource_diag())

        logger.info("[ppo-diag %s] %s", label, render_diag_map(diag))

    def save_checkpoint(self, ans_loop: int) -> None:
        """Save training checkpoint (models + loop state).

        Saves FSDP sharded checkpoints for both generator and solver.
        ``latest_checkpointed_iteration.txt`` is the single source of truth
        for which checkpoint to load on restart.

        Checkpoint structure::

            {default_local_dir}/global_step_{ans_loop}/
            ├── generator/   (FSDP sharded model + optimizer)
            └── solver/      (FSDP sharded model + optimizer)

        Reference: verl ``RayPPOTrainer._save_checkpoint()`` (ray_trainer.py:920)

        Args:
            ans_loop: Current answer loop index (checkpoint identifier).
        """
        local_dir = self.config.training.default_local_dir
        hdfs_dir = self.config.training.default_hdfs_dir
        step_dir = os.path.join(local_dir, f"global_step_{ans_loop}")

        gen_step = ans_loop * self.num_gen_per_ans + self.num_gen_per_ans - 1

        logger.info(
            "Saving checkpoint at ans_loop=%d (gen_loop=%d) to %s",
            ans_loop, gen_step, step_dir,
        )

        ckpt_t0 = time.time()

        # --- Diagnostic: baseline before any save ---
        self._checkpoint_diag("pre-save")

        # Release unreferenced CPU tensors (e.g. from prior forward/backward)
        # before we allocate FSDP state-dict copies.
        gc.collect()
        self._checkpoint_diag("post-gc-pre-gen")

        # Save generator model via worker group dispatch
        # Skip when fix_generator=True — weights never change, so re-saving
        # identical shards every ans_loop is pure waste (IO + R2 upload).
        fix_gen = self.config.training.fix_generator
        gen_elapsed = 0.0
        if not fix_gen:
            gen_path = os.path.join(step_dir, "generator")
            gen_hdfs = os.path.join(hdfs_dir, f"global_step_{ans_loop}", "generator") if hdfs_dir else None
            gen_t0 = time.time()
            try:
                self.generator_wg.save_checkpoint(
                    local_path=gen_path,
                    hdfs_path=gen_hdfs,
                    global_step=gen_step,
                )
                gen_elapsed = time.time() - gen_t0
                logger.info(
                    "[ckpt] generator save completed in %.1fs at ans_loop=%d",
                    gen_elapsed, ans_loop,
                )
            except Exception:
                gen_elapsed = time.time() - gen_t0
                logger.exception(
                    "[ckpt] generator save FAILED after %.1fs at ans_loop=%d",
                    gen_elapsed, ans_loop,
                )
                self._checkpoint_diag("post-gen-FAIL")
                raise

            # Force GC between the two model saves: the generator's CPU copies
            # (state_dict + optimizer state) can linger and overlap with solver
            # allocation, causing OOM on CPU.
            gc.collect()
            self._checkpoint_diag("post-gen-pre-solver")

            # Log gen checkpoint size on disk
            try:
                gen_size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fnames in os.walk(gen_path)
                    for f in fnames
                )
                logger.info(
                    "[ckpt] generator checkpoint size on disk: %.1f GB (%s)",
                    gen_size / 1024**3, gen_path,
                )
            except Exception:
                pass
        else:
            logger.info(
                "[ckpt] generator save skipped (fix_generator=True) at ans_loop=%d",
                ans_loop,
            )

        # Save solver model via worker group dispatch
        # Skip when fix_answer_model=True — weights never change.
        fix_solver = self.config.training.fix_answer_model
        solver_elapsed = 0.0
        if not fix_solver:
            solver_path = os.path.join(step_dir, "solver")
            solver_hdfs = os.path.join(hdfs_dir, f"global_step_{ans_loop}", "solver") if hdfs_dir else None
            solver_t0 = time.time()
            try:
                self.solver_wg.save_checkpoint(
                    local_path=solver_path,
                    hdfs_path=solver_hdfs,
                    global_step=ans_loop,
                )
                solver_elapsed = time.time() - solver_t0
                logger.info(
                    "[ckpt] solver save completed in %.1fs at ans_loop=%d",
                    solver_elapsed, ans_loop,
                )
            except Exception:
                solver_elapsed = time.time() - solver_t0
                logger.exception(
                    "[ckpt] solver save FAILED after %.1fs at ans_loop=%d",
                    solver_elapsed, ans_loop,
                )
                self._checkpoint_diag("post-solver-FAIL")
                raise
        else:
            logger.info(
                "[ckpt] solver save skipped (fix_answer_model=True) at ans_loop=%d",
                ans_loop,
            )

        self._checkpoint_diag("post-solver")

        # Write latest checkpoint marker
        latest_path = os.path.join(local_dir, "latest_checkpointed_iteration.txt")
        with open(latest_path, "w") as f:
            f.write(str(ans_loop))

        # Save curriculum state if enabled
        self._save_curriculum_state()

        # NOTE: Local checkpoint directory cleanup is handled by the async
        # upload task in the caller.  The newly saved global_step_{N}/ staging
        # directory is removed after the upload task completes, and older local
        # resume scaffolding is pruned separately based on remove_previous_ckpt.

        total_elapsed = time.time() - ckpt_t0
        logger.info(
            "Checkpoint saved at ans_loop=%d (gen=%.1fs, solver=%.1fs, total=%.1fs)",
            ans_loop, gen_elapsed, solver_elapsed, total_elapsed,
        )

    def _checkpoint_shards_match_world_size(
        self,
        role_name: str,
        checkpoint_dir: str,
        expected_world_size: int,
    ) -> bool:
        """Validate model shard files against current worker world size."""
        model_files = [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith("model_world_size_") and "_rank_" in f and f.endswith(".pt")
        ]
        if not model_files:
            raise FileNotFoundError(
                f"{role_name} checkpoint directory exists but model shard files are missing: {checkpoint_dir}"
            )

        shards_by_world_size: dict[int, set[int]] = defaultdict(set)
        for filename in model_files:
            suffix = filename[len("model_world_size_") :]
            world_size_part, rank_part = suffix.split("_rank_", 1)
            rank_part = rank_part[: -len(".pt")]
            try:
                world_size = int(world_size_part)
                rank = int(rank_part)
            except ValueError:
                logger.warning(
                    "Skipping malformed checkpoint shard name: %s/%s",
                    checkpoint_dir,
                    filename,
                )
                continue
            shards_by_world_size[world_size].add(rank)

        if expected_world_size not in shards_by_world_size:
            available_world_sizes = ", ".join(
                str(ws) for ws in sorted(shards_by_world_size)
            ) or "<none>"
            raise RuntimeError(
                f"{role_name} checkpoint world-size mismatch at {checkpoint_dir}: "
                f"expected {expected_world_size}, found [{available_world_sizes}]"
            )

        available_ranks = shards_by_world_size[expected_world_size]
        missing_ranks = [
            rank for rank in range(expected_world_size) if rank not in available_ranks
        ]
        if missing_ranks:
            raise FileNotFoundError(
                f"{role_name} checkpoint shards incomplete at {checkpoint_dir} for "
                f"world_size={expected_world_size}; missing ranks={missing_ranks}"
            )

        return True

    def _load_model_checkpoint(
        self,
        role_name: str,
        wg,
        ckpt_path: str,
        world_size: int,
    ) -> None:
        """Load model checkpoint with FSDP-first, HF-fallback strategy.

        Attempts to load in order:
        1. FSDP shards (model + optimizer + extra) — exact resume
        2. HF safetensors (model weights only) — optimizer/scheduler reset

        Args:
            role_name: Human-readable name for logging (e.g. "Generator").
            wg: Worker group with ``load_checkpoint`` and ``load_hf_checkpoint``.
            ckpt_path: Directory containing checkpoint files.
            world_size: Expected FSDP world size for shard validation.

        Raises:
            FileNotFoundError: If neither FSDP shards nor HF safetensors are found.
        """
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"{role_name} checkpoint directory not found: {ckpt_path}")

        # --- Attempt 1: FSDP shards ---
        has_fsdp_shards = any(
            f.startswith("model_world_size_")
            for f in os.listdir(ckpt_path)
            if os.path.isfile(os.path.join(ckpt_path, f))
        )
        if has_fsdp_shards:
            try:
                self._checkpoint_shards_match_world_size(role_name, ckpt_path, world_size)
                print(
                    f"  Loading {role_name} from FSDP shards (world_size={world_size}) ...",
                    flush=True,
                )
                wg.load_checkpoint(local_path=ckpt_path)
                print(
                    f"  {role_name}: loaded from FSDP shards (model + optimizer + extra)",
                    flush=True,
                )
                return
            except Exception as e:
                print(
                    f"  {role_name}: FSDP shard load failed ({e}), trying HF fallback...",
                    flush=True,
                )

        # --- Attempt 2: HF safetensors ---
        hf_path = os.path.join(ckpt_path, "huggingface")
        hf_safetensors = os.path.join(hf_path, "model.safetensors")
        hf_index = os.path.join(hf_path, "model.safetensors.index.json")
        if os.path.exists(hf_safetensors) or os.path.exists(hf_index):
            try:
                print(
                    f"  Loading {role_name} from HF safetensors ...",
                    flush=True,
                )
                wg.load_hf_checkpoint(local_path=hf_path)
                print(
                    f"  {role_name}: loaded from HF safetensors "
                    f"(model weights only; optimizer state reset, LR scheduler reset)",
                    flush=True,
                )
                return
            except Exception as e:
                logger.error(
                    "%s: HF checkpoint load also failed: %s", role_name, e,
                )

        # --- Both failed ---
        raise FileNotFoundError(
            f"{role_name} checkpoint at {ckpt_path}: no usable FSDP shards found and "
            f"no HF safetensors at {hf_path}. Cannot resume."
        )

    def _bootstrap_hf_token(self) -> None:
        """Resolve HF token and make it available to huggingface_hub.

        Precedence:
        1. ``remote.hf_token`` config field (explicit token).
        2. Environment variable named by ``remote.hf_token_env_var``
           (default ``HF_TOKEN``).
        3. Existing HF login cache (``~/.cache/huggingface/token``).

        If a token is resolved from (1) or (2), it is set in the ``HF_TOKEN``
        environment variable so that ``huggingface_hub`` picks it up
        automatically.  The token value is never logged.
        """
        resolved_token = getattr(self, "_resolved_hf_token", None)
        if resolved_token:
            os.environ["HF_TOKEN"] = str(resolved_token)
            logger.info(
                "HF token resolved from namespace auto-discovery pool for %s",
                getattr(self, "_resolved_hf_namespace", "unknown"),
            )
            return

        remote_cfg = self.config.get("remote", {})

        # (1) Explicit token in config
        token = remote_cfg.get("hf_token", None)
        if token:
            os.environ["HF_TOKEN"] = str(token)
            logger.info("HF token resolved from remote.hf_token config")
            return

        # (2) Token from named env var
        env_var_name = remote_cfg.get("hf_token_env_var", "HF_TOKEN")
        token = os.environ.get(env_var_name)
        if token:
            # Ensure it's also in the canonical HF_TOKEN env var
            os.environ["HF_TOKEN"] = token
            logger.info("HF token resolved from env var %s", env_var_name)
            return

        # (3) Fall back to cached HF login — huggingface_hub will find it
        # automatically; nothing to do here.
        logger.info(
            "No explicit HF token configured; relying on cached HF login"
        )

    def _resolve_hf_remote_sync_path(self) -> None:
        """Resolve ``hf://datasets/<namespace>/...`` via the configured token pool."""
        remote_sync_path = self.config.training.get("remote_sync_path", None)
        if not remote_sync_path or not str(remote_sync_path).startswith("hf://"):
            return

        from verl_inf_evolve.storage.hf_remote_resolver import resolve_hf_remote_from_pool

        remote_cfg = self.config.get("remote", {})
        resolved = resolve_hf_remote_from_pool(str(remote_sync_path), remote_cfg)
        if resolved is None:
            return

        self.config.training.remote_sync_path = resolved.uri
        self._resolved_hf_token = resolved.token
        self._resolved_hf_namespace = resolved.namespace

        logger.info(
            "Resolved HF remote namespace placeholder: %s -> %s",
            remote_sync_path,
            resolved.uri,
        )
        if resolved.warning:
            logger.warning(resolved.warning)

    def _resolve_pregenerated_source(self) -> None:
        """Resolve ``__namespace__`` placeholder in ``pregenerated_question_source``."""
        src = self._pregenerated_source
        if not src or not str(src).startswith("hf://"):
            return

        from verl_inf_evolve.storage.hf_remote_resolver import resolve_hf_remote_from_pool

        remote_cfg = self.config.get("remote", {})
        resolved = resolve_hf_remote_from_pool(str(src), remote_cfg)
        if resolved is None:
            return

        self._pregenerated_source = resolved.uri
        # If remote_sync_path wasn't HF-based, bootstrap token from this result.
        if not hasattr(self, "_resolved_hf_token") or not self._resolved_hf_token:
            self._resolved_hf_token = resolved.token
            self._resolved_hf_namespace = resolved.namespace
            self._bootstrap_hf_token()

        logger.info(
            "Resolved pregenerated_question_source namespace: %s -> %s",
            src, resolved.uri,
        )

    def _write_run_metadata(self) -> None:
        """Write run_metadata.json to the remote artifact root.

        Contains model info and run metadata so eval can auto-detect model
        paths without relying on R2 path heuristics.  Called early in
        ``fit()`` after the upload manager is initialized.
        """
        import datetime

        remote_sync_path = self.config.training.get("remote_sync_path", None)
        if not remote_sync_path or not hasattr(self, "_upload_manager"):
            return

        # Determine backend type from URI scheme
        path_str = str(remote_sync_path)
        if path_str.startswith("hf://"):
            backend_type = "hf"
        elif path_str.startswith(("s3://", "r2://")):
            backend_type = "r2"
        elif path_str.startswith("gs://"):
            backend_type = "gs"
        else:
            backend_type = "unknown"

        # Custom chat template presence
        gen_chat_template = self.config.generator.model.get(
            "custom_chat_template", None
        )
        solver_chat_template = self.config.solver.model.get(
            "custom_chat_template", None
        )

        metadata = {
            "solver_model_id": self.config.solver.model.path,
            "generator_model_id": self.config.generator.model.path,
            "solver_custom_chat_template": solver_chat_template is not None,
            "generator_custom_chat_template": gen_chat_template is not None,
            "run_name": self.config.wandb.get("group_name", "unknown"),
            "wandb_entity": self.config.wandb.get("entity", None),
            "wandb_project": self.config.wandb.get("project_name", None),
            "wandb_group": self.config.wandb.get("group_name", None),
            "resolved_remote_sync_path": str(remote_sync_path),
            "backend_type": backend_type,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        data = json.dumps(metadata, indent=2).encode("utf-8")
        ok = self._upload_manager.upload_bytes_sync(data, "run_metadata.json")
        if ok:
            logger.info("Wrote run_metadata.json to remote root")
        else:
            logger.warning("Failed to write run_metadata.json to remote root")

    def _handle_checkpoint_upload_failure(self, task, error: Exception) -> None:
        logger.error(
            "Checkpoint upload failed for %s: %s",
            getattr(task, "remote_key", "<unknown>"),
            error,
        )
        if not self._stop_on_checkpoint_upload_failure:
            return

        logger.critical(
            "Stopping job immediately because training.stop_on_checkpoint_upload_failure=true "
            "and checkpoint upload failed for %s",
            getattr(task, "remote_key", "<unknown>"),
        )
        os.kill(os.getpid(), signal.SIGTERM)

    def _finalize_pending_checkpoint_upload(
        self,
        *,
        raise_on_failure: bool,
        timeout: float | None = None,
    ) -> None:
        if (
            self._pending_ckpt_upload_id is None
            and self._pending_ckpt_marker_upload_id is None
        ):
            return

        pending_upload_id = self._pending_ckpt_upload_id
        pending_marker_id = self._pending_ckpt_marker_upload_id
        pending_ans_loop = self._pending_ckpt_ans_loop

        upload_ok = True
        if pending_upload_id:
            upload_ok = self._upload_manager.wait_for_task(
                pending_upload_id,
                timeout=timeout,
            )
        marker_ok = True
        if pending_marker_id:
            marker_ok = self._upload_manager.wait_for_task(
                pending_marker_id,
                timeout=timeout,
            )

        if pending_ans_loop is not None:
            if upload_ok and marker_ok:
                self._last_remote_checkpointed_ans_loop = pending_ans_loop
                self._failed_remote_checkpoint_ans_loops.discard(pending_ans_loop)
            else:
                self._failed_remote_checkpoint_ans_loops.add(pending_ans_loop)

        self._pending_ckpt_upload_id = None
        self._pending_ckpt_marker_upload_id = None
        self._pending_ckpt_ans_loop = None

        if not upload_ok:
            msg = f"Checkpoint upload failed: {pending_upload_id}"
            logger.error(msg)
            if raise_on_failure:
                raise RuntimeError(msg)
            return

        if not marker_ok:
            logger.error(
                "Checkpoint marker upload failed for ans_loop=%s: %s",
                pending_ans_loop,
                pending_marker_id,
            )

    @staticmethod
    def _is_remote_checkpoint_source(path: str) -> bool:
        """Return True if *path* looks like a remote object-store URI."""
        return path.startswith(("s3://", "r2://", "gs://", "hf://"))

    @staticmethod
    def _is_hf_checkpoint_dir(path: str) -> bool:
        """Return True when *path* is a HuggingFace safetensors directory."""
        return os.path.isfile(os.path.join(path, "model.safetensors")) or os.path.isfile(
            os.path.join(path, "model.safetensors.index.json")
        )

    def _resolve_init_checkpoint_source(self, role_name: str, source_path: str) -> str:
        """Resolve init-checkpoint source to a local directory.

        Local paths are returned as-is.
        Remote paths (s3://, r2://, gs://, hf://) are downloaded via the
        backend abstraction to a deterministic local cache directory under
        ``default_local_dir``.
        """
        if not self._is_remote_checkpoint_source(source_path):
            return source_path

        from verl_inf_evolve.storage.remote_backend import (
            build_hf_backend_kwargs,
            create_remote_backend,
        )

        remote_cfg = self.config.get("remote", {})
        backend_kwargs: dict[str, Any] = {}
        if source_path.startswith("hf://"):
            backend_kwargs = build_hf_backend_kwargs(remote_cfg)
        backend = create_remote_backend(source_path, **backend_kwargs)

        # Build a deterministic local cache path from the URI.
        source_hash = hashlib.md5(source_path.encode("utf-8")).hexdigest()[:10]
        # Extract a human-readable base name from the URI path.
        uri_path = source_path.rstrip("/").rsplit("/", 1)[-1] or "checkpoint"
        local_dir = os.path.join(
            self.config.training.default_local_dir,
            ".init_checkpoints",
            f"{role_name.lower()}_{uri_path}_{source_hash}",
        )

        if os.path.isdir(local_dir) and os.listdir(local_dir):
            logger.info(
                "%s init checkpoint already cached locally: %s", role_name, local_dir
            )
            return local_dir

        shutil.rmtree(local_dir, ignore_errors=True)
        os.makedirs(local_dir, exist_ok=True)
        logger.info(
            "Downloading %s init checkpoint from %s -> %s",
            role_name,
            source_path,
            local_dir,
        )
        download_key = ""
        download_local_dir = local_dir
        try:
            children = backend.list_immediate_children("")
        except Exception:
            children = []
        if isinstance(children, list) and "huggingface" in children:
            download_key = "huggingface"
            download_local_dir = os.path.join(local_dir, "huggingface")
            os.makedirs(download_local_dir, exist_ok=True)
            logger.info(
                "Downloading %s init checkpoint HF subtree only: %s/huggingface -> %s",
                role_name,
                source_path.rstrip("/"),
                download_local_dir,
            )

        backend.download_dir(download_key, download_local_dir)
        return local_dir

    def _load_initial_role_checkpoint(
        self,
        role_name: str,
        wg,
        world_size: int,
        source_path: str,
    ) -> None:
        """Load one role's init checkpoint from local/remote source.

        Supports either:
        - FSDP/role checkpoint dir (uses ``_load_model_checkpoint``), or
        - direct HuggingFace dir (contains model.safetensors, loaded via
          ``load_hf_checkpoint``).
        """
        ckpt_path = self._resolve_init_checkpoint_source(role_name, source_path)
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(
                f"{role_name} init checkpoint directory not found: {ckpt_path}"
            )

        # Direct HF directory (model.safetensors or sharded safetensors index).
        if self._is_hf_checkpoint_dir(ckpt_path):
            print(
                f"  Loading {role_name} init checkpoint from HF dir: {ckpt_path}",
                flush=True,
            )
            wg.load_hf_checkpoint(local_path=ckpt_path)
            print(
                f"  {role_name}: init loaded from HF safetensors "
                f"(model weights only; optimizer state reset, LR scheduler reset)",
                flush=True,
            )
            return

        # FSDP-first, HF-subdir fallback.
        self._load_model_checkpoint(
            role_name=role_name,
            wg=wg,
            ckpt_path=ckpt_path,
            world_size=world_size,
        )

    def _load_role_checkpoint_with_override(
        self,
        role_name: str,
        wg,
        world_size: int,
        ckpt_override: str | None,
        resume_path: str,
        is_fixed: bool,
    ) -> float:
        """Load a role's checkpoint with 3-way priority.

        Priority: ``ckpt_path`` override > normal resume checkpoint > skip (fixed).

        Returns:
            Elapsed time in seconds for the load operation.
        """
        if ckpt_override:
            _start = time.perf_counter()
            try:
                self._load_initial_role_checkpoint(
                    role_name=role_name,
                    wg=wg,
                    world_size=world_size,
                    source_path=ckpt_override,
                )
                elapsed = time.perf_counter() - _start
                print(
                    f"  {role_name} loaded from ckpt_path override: {ckpt_override} ({elapsed:.1f}s)",
                    flush=True,
                )
                return elapsed
            except (FileNotFoundError, OSError, RuntimeError) as e:
                logger.warning(
                    "%s ckpt_path override failed, falling back to normal resume",
                    role_name,
                    exc_info=True,
                )
                if not is_fixed and os.path.isdir(resume_path):
                    _start = time.perf_counter()
                    self._load_model_checkpoint(
                        role_name=role_name,
                        wg=wg,
                        ckpt_path=resume_path,
                        world_size=world_size,
                    )
                    elapsed = time.perf_counter() - _start
                    print(
                        f"  {role_name} checkpoint loaded via fallback ({elapsed:.1f}s)",
                        flush=True,
                    )
                    return elapsed
                elif is_fixed:
                    print(
                        f"  {role_name} ckpt_path failed and model is fixed — using initial weights",
                        flush=True,
                    )
                    return 0.0
                else:
                    raise RuntimeError(
                        f"{role_name} ckpt_path override failed and normal resume "
                        f"checkpoint is unavailable at {resume_path}"
                    ) from e
        elif not is_fixed:
            _start = time.perf_counter()
            self._load_model_checkpoint(
                role_name=role_name,
                wg=wg,
                ckpt_path=resume_path,
                world_size=world_size,
            )
            elapsed = time.perf_counter() - _start
            print(
                f"  {role_name} checkpoint loaded ({elapsed:.1f}s)",
                flush=True,
            )
            return elapsed
        else:
            print(
                f"  Skipping {role_name} checkpoint load (fixed — using initial weights)",
                flush=True,
            )
            return 0.0

    def _maybe_load_initial_role_checkpoints(self, start_ans_loop: int) -> None:
        """Optionally load role-specific init checkpoints on fresh starts.

        This hook is intentionally run *after* ``_load_checkpoint()`` and only
        when ``start_ans_loop == 0`` so existing resume semantics are unchanged.
        """
        gen_source = self.config.training.get("init_generator_checkpoint_path", None)
        solver_source = self.config.training.get("init_solver_checkpoint_path", None)
        if not gen_source and not solver_source:
            return

        if start_ans_loop > 0:
            logger.info(
                "Init checkpoints configured but skipping because start_ans_loop=%d "
                "(resume path takes precedence).",
                start_ans_loop,
            )
            return

        print("========== INIT CHECKPOINT LOAD (FRESH START) ==========", flush=True)
        self._checkpoint_diag("pre-init-role-checkpoint-load")

        total_start = time.perf_counter()
        if gen_source:
            load_start = time.perf_counter()
            self._load_initial_role_checkpoint(
                role_name="Generator",
                wg=self.generator_wg,
                world_size=self.generator_wg.world_size,
                source_path=gen_source,
            )
            print(
                f"  Generator init checkpoint loaded ({time.perf_counter() - load_start:.1f}s)",
                flush=True,
            )

        if solver_source:
            load_start = time.perf_counter()
            self._load_initial_role_checkpoint(
                role_name="Solver",
                wg=self.solver_wg,
                world_size=self.solver_wg.world_size,
                source_path=solver_source,
            )
            print(
                f"  Solver init checkpoint loaded ({time.perf_counter() - load_start:.1f}s)",
                flush=True,
            )

        self._checkpoint_diag("post-init-role-checkpoint-load")
        print(
            f"========== INIT CHECKPOINT LOAD COMPLETE ({time.perf_counter() - total_start:.1f}s) ==========",
            flush=True,
        )

    def _load_checkpoint(self) -> int:
        """Load training checkpoint if one exists.

        Detects the latest checkpoint from ``latest_checkpointed_iteration.txt``
        and restores model weights and loop state.

        If an ``ans_{N}/`` directory exists for the *next* ans_loop
        (i.e., the run crashed mid-loop after the last successful checkpoint),
        the returned ``start_ans_loop`` points to that incomplete loop so
        that stage-level resume can pick up where it left off.

        Returns:
            start_ans_loop — the ans_loop index to start (or resume) from.

        Raises:
            FileNotFoundError: If checkpoint metadata, directories, or shards are missing.
            RuntimeError: If checkpoint world size does not match current worker world size.
        """
        local_dir = self.config.training.default_local_dir
        latest_path = os.path.join(local_dir, "latest_checkpointed_iteration.txt")

        # If resume_from_remote is enabled, try to resolve the marker file
        # from the remote backend when it's missing locally.  This enables
        # resume on a fresh machine where no local checkpoint marker exists.
        resume_from_remote = self.config.training.get("resume_from_remote", False)
        if not os.path.exists(latest_path) and resume_from_remote:
            print(
                "========== REMOTE RESUME: CHECKPOINT DISCOVERY ==========",
                flush=True,
            )
            print(
                f"  No local checkpoint marker found at {latest_path}",
                flush=True,
            )
            print(
                "  resume_from_remote=True — searching remote for checkpoint marker...",
                flush=True,
            )
            marker_resolver = StorageResolver(
                local_base=local_dir,
                upload_manager=self._upload_manager,
                remote_prefix="",
                resolve_order="local_first",
            )
            resolved = marker_resolver.resolve_file("latest_checkpointed_iteration.txt")
            if resolved:
                print(
                    f"  Downloaded latest_checkpointed_iteration.txt from remote → {resolved}",
                    flush=True,
                )
            else:
                print(
                    "  Could not find checkpoint marker on remote — will start fresh",
                    flush=True,
                )

        if not os.path.exists(latest_path):
            print(
                f"No checkpoint found at {latest_path} — starting fresh.",
                flush=True,
            )
            return 0

        with open(latest_path, "r") as f:
            last_ans_loop = int(f.read().strip())
        requested_step_dir = os.path.join(local_dir, f"global_step_{last_ans_loop}")

        print(
            f"========== CHECKPOINT RESUME | ans_loop={last_ans_loop} | dir={requested_step_dir} ==========",
            flush=True,
        )

        # If checkpoint was cleaned up locally after remote upload, download
        # it back.  Only check paths for models that are actually checkpointed
        # (fixed models are intentionally absent from the checkpoint directory).
        # When ckpt_path is set for a role, we load from there instead, so the
        # normal checkpoint directory is not required for that role either.
        fix_gen = self.config.training.fix_generator
        fix_solver = self.config.training.fix_answer_model
        gen_cfg = self.config.get("generator", {})
        solver_cfg = self.config.get("solver", {})
        gen_ckpt_override = gen_cfg.get("ckpt_path", None)
        solver_ckpt_override = solver_cfg.get("ckpt_path", None)
        need_gen_ckpt = not fix_gen and not gen_ckpt_override
        need_solver_ckpt = not fix_solver and not solver_ckpt_override
        candidate_steps = self._checkpoint_resume_candidates(
            requested_step=last_ans_loop,
            local_dir=local_dir,
        )
        step_dir: str | None = None
        gen_path: str | None = None
        solver_path: str | None = None
        selected_step: int | None = None
        attempted_steps: list[int] = []

        for candidate_step in candidate_steps:
            attempted_steps.append(candidate_step)
            candidate_step_dir = os.path.join(local_dir, f"global_step_{candidate_step}")
            candidate_gen_path = os.path.join(candidate_step_dir, "generator")
            candidate_solver_path = os.path.join(candidate_step_dir, "solver")

            local_missing = (
                (need_gen_ckpt and not os.path.isdir(candidate_gen_path))
                or (need_solver_ckpt and not os.path.isdir(candidate_solver_path))
            )
            if local_missing:
                print(
                    f"  REMOTE RESUME: Local checkpoint for global_step_{candidate_step} not found — downloading from remote...",
                    flush=True,
                )
                _dl_start = time.perf_counter()
                ckpt_resolver = StorageResolver(
                    local_base=candidate_step_dir,
                    upload_manager=self._upload_manager,
                    remote_prefix=f"global_step_{candidate_step}",
                    resolve_order="local_first",
                )

                download_failed = False
                if need_gen_ckpt and not os.path.isdir(candidate_gen_path):
                    print(
                        f"  REMOTE RESUME: Downloading generator checkpoint (global_step_{candidate_step}/generator) ...",
                        flush=True,
                    )
                    gen_result = ckpt_resolver.resolve_dir("generator")
                    if gen_result is None:
                        download_failed = True
                    else:
                        _gen_elapsed = time.perf_counter() - _dl_start
                        print(
                            f"  REMOTE RESUME: Generator checkpoint downloaded ({_gen_elapsed:.1f}s)",
                            flush=True,
                        )
                elif fix_gen:
                    print(
                        "  REMOTE RESUME: Skipping generator checkpoint download (fix_generator=True)",
                        flush=True,
                    )

                if not download_failed and need_solver_ckpt and not os.path.isdir(candidate_solver_path):
                    _dl_start2 = time.perf_counter()
                    print(
                        f"  REMOTE RESUME: Downloading solver checkpoint (global_step_{candidate_step}/solver) ...",
                        flush=True,
                    )
                    solver_result = ckpt_resolver.resolve_dir("solver")
                    if solver_result is None:
                        download_failed = True
                    else:
                        _solver_elapsed = time.perf_counter() - _dl_start2
                        print(
                            f"  REMOTE RESUME: Solver checkpoint downloaded ({_solver_elapsed:.1f}s)",
                            flush=True,
                        )
                elif fix_solver:
                    print(
                        "  REMOTE RESUME: Skipping solver checkpoint download (fix_answer_model=True)",
                        flush=True,
                    )

                if download_failed:
                    if candidate_step == last_ans_loop:
                        print(
                            f"  REMOTE RESUME: Checkpoint marker points to global_step_{candidate_step}, but the checkpoint is unavailable.",
                            flush=True,
                        )
                    else:
                        print(
                            f"  REMOTE RESUME: Fallback checkpoint global_step_{candidate_step} is also unavailable.",
                            flush=True,
                        )
                    continue

                print(
                    f"  REMOTE RESUME: All checkpoint files downloaded to {candidate_step_dir}",
                    flush=True,
                )
            else:
                print(
                    f"  Checkpoint found locally at {candidate_step_dir} — no remote download needed",
                    flush=True,
                )

            selected_step = candidate_step
            step_dir = candidate_step_dir
            gen_path = candidate_gen_path
            solver_path = candidate_solver_path
            break

        if selected_step is None or step_dir is None or gen_path is None or solver_path is None:
            raise RuntimeError(
                "REMOTE RESUME: Failed to resolve a usable checkpoint. "
                f"marker={last_ans_loop}, attempted={attempted_steps}"
            )

        if selected_step != last_ans_loop:
            print(
                f"  REMOTE RESUME: Falling back from marker global_step_{last_ans_loop} to available checkpoint global_step_{selected_step}.",
                flush=True,
            )
            logger.warning(
                "Checkpoint marker %d is stale; falling back to available checkpoint %d",
                last_ans_loop,
                selected_step,
            )
            with open(latest_path, "w") as f:
                f.write(str(selected_step))
            last_ans_loop = selected_step

        self._checkpoint_diag("post-download-pre-load")
        _load_total_start = time.perf_counter()

        gen_world_size = self.generator_wg.world_size
        solver_world_size = self.solver_wg.world_size

        _gen_load_elapsed = self._load_role_checkpoint_with_override(
            role_name="Generator",
            wg=self.generator_wg,
            world_size=gen_world_size,
            ckpt_override=gen_ckpt_override,
            resume_path=gen_path,
            is_fixed=fix_gen,
        )
        self._checkpoint_diag("post-gen-load-pre-solver")

        _solver_load_elapsed = self._load_role_checkpoint_with_override(
            role_name="Solver",
            wg=self.solver_wg,
            world_size=solver_world_size,
            ckpt_override=solver_ckpt_override,
            resume_path=solver_path,
            is_fixed=fix_solver,
        )

        _load_total_elapsed = time.perf_counter() - _load_total_start
        self._checkpoint_diag("post-solver-load")
        print(
            f"  Checkpoint loading complete (gen={_gen_load_elapsed:.1f}s, solver={_solver_load_elapsed:.1f}s, total={_load_total_elapsed:.1f}s)",
            flush=True,
        )

        # Clean up downloaded checkpoint from disk now that weights are in
        # memory.  Only when remove_previous_ckpt is set and remote storage is
        # configured (so the data can be re-downloaded if needed).
        remote_configured = (
            self._upload_manager is not None
            and self._upload_manager.remote_configured
        )
        if self.config.training.remove_previous_ckpt and remote_configured:
            shutil.rmtree(step_dir, ignore_errors=True)
            print(
                f"  Cleaned up local checkpoint dir after load: {step_dir}",
                flush=True,
            )
            self._checkpoint_diag("post-load-cleanup")

        start_ans_loop = last_ans_loop + 1
        print(
            f"========== CHECKPOINT RESUME COMPLETE | resuming from ans_loop={start_ans_loop} ==========",
            flush=True,
        )
        # Resume from next ans_loop (the completed one was last_ans_loop)
        return start_ans_loop

    @staticmethod
    def _discover_local_checkpoint_steps(local_dir: str) -> list[int]:
        step_re = re.compile(r"^global_step_(\d+)$")
        if not os.path.isdir(local_dir):
            return []

        steps: list[int] = []
        for child in os.listdir(local_dir):
            match = step_re.match(child)
            if not match:
                continue
            if os.path.isdir(os.path.join(local_dir, child)):
                steps.append(int(match.group(1)))
        return sorted(steps)

    def _discover_remote_checkpoint_steps(self) -> list[int]:
        if (
            self._upload_manager is None
            or not self._upload_manager.remote_configured
            or self._upload_manager.backend is None
        ):
            return []

        step_re = re.compile(r"^global_step_(\d+)/?$")
        steps: list[int] = []
        for child in self._upload_manager.backend.list_immediate_children(""):
            match = step_re.match(child)
            if match:
                steps.append(int(match.group(1)))
        return sorted(steps)

    def _checkpoint_resume_candidates(
        self,
        requested_step: int,
        local_dir: str,
    ) -> list[int]:
        candidates: list[int] = []
        seen: set[int] = set()
        ordered_steps = [
            requested_step,
            *reversed(self._discover_local_checkpoint_steps(local_dir)),
            *reversed(self._discover_remote_checkpoint_steps()),
        ]
        for step in ordered_steps:
            if step > requested_step or step in seen:
                continue
            candidates.append(step)
            seen.add(step)
        return candidates
