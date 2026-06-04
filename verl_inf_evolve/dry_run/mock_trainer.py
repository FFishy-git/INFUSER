"""Dedicated trainer subclass for CPU-only mock dry runs."""

from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any

from verl_inf_evolve.dry_run.mock_runtime import (
    MockCPUWorkerGroup,
    MockDryRunCrash,
    build_mock_dev_output,
    build_mock_gen_answer_output,
    build_mock_question_output,
    build_mock_stage4_reward_payload,
)
from verl_inf_evolve.trainer.resume_state import ResumeState
from verl_inf_evolve.trainer.rollout_metrics import (
    compute_answer_rollout_metrics,
    compute_ans_loop_summary_metrics,
    compute_question_rollout_metrics,
)
from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer
from verl_inf_evolve.utils.data_utils import derive_gen_questions
from verl_inf_evolve.utils.generator_reward_utils import (
    extract_influence_rewards_for_solver_filter,
)
from verl_inf_evolve.utils.mcq_utils import group_scores_by_qid
from verl_inf_evolve.utils.startup_cleanup import maybe_clear_local_dir_on_start
from verl_inf_evolve.storage.stage_upload_manager import StageUploadManager

logger = logging.getLogger(__name__)


class MockDryRunTrainer(SelfEvolutionTrainer):
    """CPU-only trainer that exercises upload/resume plumbing with fake data."""

    def _validate_dry_run_config(self) -> None:
        backend = str(self.config.dry_run.get("backend", "mock_cpu"))
        resume_loader = str(self.config.dry_run.get("resume_loader", "mock"))
        if backend != "mock_cpu":
            raise ValueError(
                f"Unsupported dry_run.backend={backend!r}; expected 'mock_cpu'."
            )
        if resume_loader != "mock":
            raise ValueError(
                f"Unsupported dry_run.resume_loader={resume_loader!r}; expected 'mock'."
            )

    def _maybe_raise_dry_run_crash(self, phase: str) -> None:
        """Inject a controlled crash after flushing pending uploads."""
        crash_after = str(self.config.dry_run.get("crash_after", "none"))
        if crash_after != phase:
            return

        if getattr(self, "_upload_manager", None) is not None:
            self._upload_manager.wait_all_pending()

        raise MockDryRunCrash(
            f"Dry-run controlled crash after {phase} for resume validation."
        )

    def init_workers(self):
        """Initialize local CPU-only mock worker groups."""
        self._validate_dry_run_config()
        maybe_clear_local_dir_on_start(self.config)
        self.generator_wg = MockCPUWorkerGroup("generator", self.config)
        self.solver_wg = MockCPUWorkerGroup("solver", self.config)
        self.generator_wg.init_model()
        self.solver_wg.init_model()
        self.gen_rollout_manager = None
        self.solver_rollout_manager = None
        self._shared_engine_mode = False
        logger.info(
            "Initialized dry-run mock worker groups on CPU only (world_size=%d)",
            self.generator_wg.world_size,
        )

    def fit(self):
        """Run the mock dry-run training loop with normal upload plumbing."""
        self._validate_dry_run_config()

        remote_sync_path = self.config.training.get("remote_sync_path", None)
        if remote_sync_path and str(remote_sync_path).startswith("hf://"):
            self._resolve_hf_remote_sync_path()
            remote_sync_path = self.config.training.get("remote_sync_path", None)
            self._bootstrap_hf_token()

        max_pending_gb = self.config.training.get("max_pending_upload_gb", 100.0)
        disable_upload = self.config.training.get("disable_remote_upload", False)
        self._upload_manager = StageUploadManager(
            remote_sync_path=remote_sync_path,
            max_pending_gb=max_pending_gb,
            disable_upload=disable_upload,
            checkpoint_failure_callback=self._handle_checkpoint_upload_failure,
        )
        self._upload_manager.start()
        self._write_run_metadata()

        try:
            self._fit_mock_loop()
        finally:
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

    def _fit_mock_loop(self) -> None:
        """Run a CPU-only synthetic training loop for upload/resume testing."""
        if self.config.curriculum.enabled:
            logger.warning("dry_run.enabled=true ignores curriculum refresh logic.")
        if self.config.get("benchmark_eval", {}).get("enabled", False):
            logger.warning(
                "dry_run.enabled=true skips in-training benchmark evaluation."
            )

        start_ans_loop = self._load_checkpoint()
        self.gen_steps = start_ans_loop * self.num_gen_per_ans
        self._last_saved_ans_loop = (start_ans_loop - 1) if start_ans_loop > 0 else None
        self._last_remote_checkpointed_ans_loop = (
            start_ans_loop - 1 if start_ans_loop > 0 else None
        )
        self._failed_remote_checkpoint_ans_loops = set()
        self._pending_ckpt_upload_id = None
        self._pending_ckpt_marker_upload_id = None
        self._pending_ckpt_ans_loop = None
        self._start_ans_loop = start_ans_loop

        tracking_logger = self._init_tracking()
        save_interval = self.config.training.save_every_n_steps

        logger.info(
            "========== DRY-RUN START | ans_loop_range=[%d, %d) | num_gen_per_ans=%d "
            "| save_every_n_steps=%d | crash_after=%s ==========",
            start_ans_loop,
            self.config.training.max_ans_loop,
            self.num_gen_per_ans,
            save_interval,
            self.config.dry_run.get("crash_after", "none"),
        )

        for ans_loop in range(start_ans_loop, self.config.training.max_ans_loop):
            logger.info(
                "========== DRY-RUN ANS LOOP START | ans_loop=%d ==========",
                ans_loop,
            )

            resume_dir = os.path.join(
                self.config.training.default_local_dir, f"ans_{ans_loop}"
            )
            resume = self._load_or_create_resume_state(
                ans_loop, resume_dir, save_interval
            )

            ans_timing_raw: dict[str, float] = {}
            ans_loop_start = time.perf_counter()

            with self.stage_ctx(
                name="curriculum_refresh",
                stage_id=0,
                resume=resume,
                resume_dir=resume_dir,
                is_done=lambda: resume.stage_0_done,
                mark_done=lambda: setattr(resume, "stage_0_done", True),
                ans_loop=ans_loop,
                timing_dict=ans_timing_raw,
            ) as ctx0:
                if ctx0.should_run:
                    logger.info(
                        "  [Stage 0] dry-run no-op (curriculum refresh skipped)"
                    )

            solver_n = self.config.solver.rollout.n
            with self.stage_ctx(
                name="dev_rollout",
                stage_id=1,
                resume=resume,
                resume_dir=resume_dir,
                is_done=lambda: resume.stage_1_done,
                mark_done=lambda: setattr(resume, "stage_1_done", True),
                ans_loop=ans_loop,
                timing_dict=ans_timing_raw,
            ) as ctx1:
                if ctx1.should_run:
                    dev_output = build_mock_dev_output(self.config, ans_loop)
                    self.record_dev_accuracy(dev_output, ans_loop)
                    ctx1.save("dev_output", dev_output)

            dev_output = ctx1.result("dev_output")
            dev_scores_dict = group_scores_by_qid(dev_output)
            dev_rollout_metrics = compute_answer_rollout_metrics(
                scores=dev_scores_dict,
                response_mask=dev_output.batch["response_mask"],
                rollout_n=solver_n,
                prefix="dev_rollout",
            )
            early_timing = {
                f"ans_timing_s/{key}": ans_timing_raw.pop(key)
                for key in list(ans_timing_raw)
            }
            tracking_logger.log(
                {"train/ans_loop": ans_loop, **dev_rollout_metrics, **early_timing},
                step=self.gen_steps,
            )

            gen_loop_results: list[dict[str, Any]] = []

            for local_gen in range(self.num_gen_per_ans):
                gen_loop = ans_loop * self.num_gen_per_ans + local_gen
                gl = resume.ensure_gen_loop(local_gen)

                logger.info(
                    "========== DRY-RUN GEN LOOP START | ans_loop=%d | local_gen=%d/%d "
                    "| gen_loop=%d ==========",
                    ans_loop,
                    local_gen + 1,
                    self.num_gen_per_ans,
                    gen_loop,
                )

                gen_timing_raw: dict[str, float] = {}
                gen_loop_start = time.perf_counter()

                with self.gen_stage_ctx(
                    name="question_rollout",
                    stage_id=2,
                    resume=resume,
                    resume_dir=resume_dir,
                    local_gen=local_gen,
                    is_done=lambda: gl.stage_2_done,
                    mark_done=lambda: self._mark_stage_2_done(gl),
                    ans_loop=ans_loop,
                    timing_dict=gen_timing_raw,
                ) as ctx2:
                    if ctx2.should_run:
                        gen_output = build_mock_question_output(
                            self.config, ans_loop, local_gen
                        )
                        ctx2.save("gen_output", gen_output)

                gen_output = ctx2.result("gen_output")
                gen_questions = derive_gen_questions(gen_output)
                question_metrics = compute_question_rollout_metrics(
                    gen_questions=gen_questions,
                    total_samples=gen_output.batch["responses"].shape[0],
                    response_mask=gen_output.batch["response_mask"],
                    num_documents=len(set(gen_output.non_tensor_batch["doc_id"])),
                    prefix="gen_question_rollout",
                    reject_reasons=gen_output.non_tensor_batch.get("reject_reason"),
                )
                question_timing = {
                    f"gen_timing_s/{key}": gen_timing_raw.pop(key)
                    for key in list(gen_timing_raw)
                }
                tracking_logger.log(
                    {
                        "train/gen_step": self.gen_steps,
                        "train/ans_loop": ans_loop,
                        "train/gen_loop": gen_loop,
                        **question_metrics,
                        **question_timing,
                    },
                    step=self.gen_steps,
                )

                with self.gen_stage_ctx(
                    name="gen_answer_rollout",
                    stage_id=3,
                    resume=resume,
                    resume_dir=resume_dir,
                    local_gen=local_gen,
                    is_done=lambda: gl.stage_3_done,
                    mark_done=lambda: setattr(gl, "stage_3_done", True),
                    ans_loop=ans_loop,
                    timing_dict=gen_timing_raw,
                ) as ctx3:
                    if ctx3.should_run:
                        gen_answer_output = build_mock_gen_answer_output(
                            self.config, gen_questions
                        )
                        ctx3.save("gen_answer_output", gen_answer_output)

                gen_answer_output = ctx3.result("gen_answer_output")
                gen_scores_dict = group_scores_by_qid(gen_answer_output)
                answer_metrics = compute_answer_rollout_metrics(
                    scores=gen_scores_dict,
                    response_mask=gen_answer_output.batch["response_mask"],
                    rollout_n=solver_n,
                    prefix="gen_answer_rollout",
                )
                answer_timing = {
                    f"gen_timing_s/{key}": gen_timing_raw.pop(key)
                    for key in list(gen_timing_raw)
                }
                tracking_logger.log(
                    {
                        "train/gen_step": self.gen_steps,
                        "train/ans_loop": ans_loop,
                        "train/gen_loop": gen_loop,
                        **answer_metrics,
                        **answer_timing,
                    },
                    step=self.gen_steps,
                )

                with self.gen_stage_ctx(
                    name="scoring",
                    stage_id=4,
                    resume=resume,
                    resume_dir=resume_dir,
                    local_gen=local_gen,
                    is_done=lambda: gl.stage_4_done,
                    mark_done=lambda: setattr(gl, "stage_4_done", True),
                    ans_loop=ans_loop,
                    timing_dict=gen_timing_raw,
                ) as ctx4:
                    if ctx4.should_run:
                        rewards_payload = build_mock_stage4_reward_payload(
                            self.config, gen_questions
                        )
                        gl.influence_metrics = {}
                        gl.rewards = rewards_payload
                        ctx4.save_json("rewards", rewards_payload)

                rewards_payload = ctx4.result_json("rewards")
                solver_filter_rewards = extract_influence_rewards_for_solver_filter(
                    rewards_payload
                )
                influence_metrics = getattr(gl, "influence_metrics", {})
                scoring_timing = {
                    f"gen_timing_s/{key}": gen_timing_raw.pop(key)
                    for key in list(gen_timing_raw)
                }
                tracking_logger.log(
                    {
                        "train/gen_step": self.gen_steps,
                        "train/ans_loop": ans_loop,
                        "train/gen_loop": gen_loop,
                        **influence_metrics,
                        **scoring_timing,
                    },
                    step=self.gen_steps,
                )

                self._maybe_raise_dry_run_crash("stage4")

                with self.gen_stage_ctx(
                    name="gen_ppo_update",
                    stage_id=5,
                    resume=resume,
                    resume_dir=resume_dir,
                    local_gen=local_gen,
                    is_done=lambda: gl.stage_5_done,
                    mark_done=lambda: setattr(gl, "stage_5_done", True),
                    ans_loop=ans_loop,
                    defer_state_update=True,
                    timing_dict=gen_timing_raw,
                ) as ctx5:
                    if ctx5.should_run:
                        logger.info(
                            "  [Stage 5] dry-run no-op (generator PPO skipped)"
                        )

                gen_loop_results.append(
                    {
                        "gen_answer_output": gen_answer_output,
                        "rewards": solver_filter_rewards,
                        "gen_questions": gen_questions,
                    }
                )

                tracking_logger.log(
                    {
                        "train/gen_step": self.gen_steps,
                        "train/ans_loop": ans_loop,
                        "train/gen_loop": gen_loop,
                        **{
                            f"gen_timing_s/{key}": value
                            for key, value in gen_timing_raw.items()
                        },
                    },
                    step=self.gen_steps,
                )
                self.gen_steps += 1

                gen_timing_raw["gen_loop_total"] = (
                    time.perf_counter() - gen_loop_start
                )
                ans_timing_raw["gen_loops_total"] = ans_timing_raw.get(
                    "gen_loops_total", 0.0
                ) + gen_timing_raw["gen_loop_total"]

                logger.info(
                    "========== DRY-RUN GEN LOOP END | ans_loop=%d | gen_loop=%d ==========",
                    ans_loop,
                    gen_loop,
                )

            with self.stage_ctx(
                name="solver_ppo_update",
                stage_id=6,
                resume=resume,
                resume_dir=resume_dir,
                is_done=lambda: resume.stage_6_done,
                mark_done=lambda: setattr(resume, "stage_6_done", True),
                ans_loop=ans_loop,
                defer_state_update=True,
                timing_dict=ans_timing_raw,
            ) as ctx6:
                if ctx6.should_run:
                    logger.info("  [Stage 6] dry-run no-op (solver PPO skipped)")

            ans_timing_raw["ans_loop_total"] = time.perf_counter() - ans_loop_start

            for result in gen_loop_results:
                result["gen_scores"] = group_scores_by_qid(
                    result["gen_answer_output"]
                )
            ans_metrics = compute_ans_loop_summary_metrics(
                gen_loop_results=gen_loop_results,
                prefix="ans_loop",
            )
            ans_metrics["train/ans_loop"] = ans_loop
            ans_metrics.update(
                {f"ans_timing_s/{key}": value for key, value in ans_timing_raw.items()}
            )
            tracking_logger.log(ans_metrics, step=self.gen_steps)

            self._maybe_raise_dry_run_crash("before_checkpoint")

            continuous_mode = self.config.training.always_save_for_resume
            should_save_ckpt, is_hf_keep_step = self._checkpoint_policy_for_ans_loop(
                ans_loop
            )
            should_upload_stage_context = (
                self._should_upload_stage_context_for_ans_loop(ans_loop)
            )
            if continuous_mode:
                logger.info(
                    "---------- Checkpointing | ans_loop=%d | hf_keep=%s ----------",
                    ans_loop,
                    is_hf_keep_step,
                )
            else:
                if should_save_ckpt:
                    logger.info(
                        "---------- Checkpointing | ans_loop=%d ----------",
                        ans_loop,
                    )
                else:
                    logger.info(
                        "---------- Skipping checkpoint | ans_loop=%d "
                        "(save_every_n_steps=%d) ----------",
                        ans_loop,
                        save_interval,
                    )

            if should_save_ckpt:
                if self._pending_ckpt_upload_id is not None:
                    logger.info(
                        "Waiting for previous checkpoint upload %s to finish "
                        "before saving ans_loop=%d ...",
                        self._pending_ckpt_upload_id,
                        ans_loop,
                    )
                    self._finalize_pending_checkpoint_upload(
                        raise_on_failure=self._stop_on_checkpoint_upload_failure,
                    )

                self.save_checkpoint(ans_loop)

            upload_enabled = (
                self._upload_manager is not None
                and self._upload_manager.upload_enabled
            )
            cleanup_previous_local = (
                self.config.training.remove_previous_ckpt and upload_enabled
            )

            for gl in resume.gen_loops:
                gl.stage_5_done = True
            resume.stage_6_done = True
            resume.save(resume_dir)

            if upload_enabled and should_save_ckpt:
                local_dir = self.config.training.default_local_dir
                step_dir = os.path.join(local_dir, f"global_step_{ans_loop}")
                self._pending_ckpt_ans_loop = ans_loop

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
                    local_path=os.path.join(
                        local_dir, "latest_checkpointed_iteration.txt"
                    ),
                    remote_key="latest_checkpointed_iteration.txt",
                    depends_on=upload_task_id,
                )
                self._pending_ckpt_marker_upload_id = marker_task_id

                if (
                    cleanup_previous_local
                    and self._last_remote_checkpointed_ans_loop is not None
                ):
                    prev_saved = self._last_remote_checkpointed_ans_loop
                    prev_step_dir = os.path.join(
                        local_dir, f"global_step_{prev_saved}"
                    )

                    if continuous_mode:
                        prev_is_hf_keep = (
                            (save_interval > 0 and prev_saved % save_interval == 0)
                            or prev_saved == self.config.training.max_ans_loop - 1
                        )
                        exclude = ["**/huggingface/**"] if prev_is_hf_keep else None
                    else:
                        exclude = ["**/huggingface/**"]

                    self._upload_manager.submit_remote_delete(
                        remote_key=f"global_step_{prev_saved}",
                        exclude_patterns=exclude,
                        cleanup_local_path=(
                            prev_step_dir if os.path.isdir(prev_step_dir) else None
                        ),
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
                    for old_step in range(0, ans_loop):
                        if old_step == prev_saved:
                            continue
                        old_dir = os.path.join(local_dir, f"global_step_{old_step}")
                        if os.path.isdir(old_dir):
                            logger.info(
                                "Cleaning up stale local checkpoint: %s", old_dir
                            )
                            shutil.rmtree(old_dir, ignore_errors=True)

                self._last_saved_ans_loop = ans_loop
            elif upload_enabled and not should_save_ckpt and should_upload_stage_context:
                self._upload_manager.submit_file_upload(
                    local_path=os.path.join(resume_dir, "state.json"),
                    remote_key=f"ans_{ans_loop}/state.json",
                )

            if cleanup_previous_local and should_save_ckpt:
                ResumeState.clear(resume_dir)

            logger.info(
                "========== DRY-RUN ANS LOOP END | ans_loop=%d ==========",
                ans_loop,
            )

        logger.info("========== DRY-RUN COMPLETE ==========")
