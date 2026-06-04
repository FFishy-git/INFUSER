"""WandB logging integration for the evaluation pipeline.

Logs per-checkpoint evaluation metrics and run summaries to Weights & Biases
for tracking accuracy curves across checkpoints and comparing runs.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EvalWandBLogger:
    """WandB logger for evaluation metrics.

    Logs per-evaluation metrics at step=checkpoint_number and computes
    best accuracy / checkpoint summaries at the end of a run.

    In **backfill mode**, logs with the same schema as in-training
    benchmark_eval (``benchmark_eval/{name}/*`` keyed on ``train/ans_loop``)
    and can resume an existing training wandb run.
    """

    def __init__(
        self,
        run_name: str,
        config_dict: dict,
        benchmarks: list[str],
        entity: Optional[str] = None,
        project: str = "self-evolution-v3-ans-eval",
        group: Optional[str] = None,
        backfill_mode: bool = False,
        wandb_run_id: Optional[str] = None,
    ):
        """Initialize the WandB logger.

        Args:
            run_name: Training run name (used as WandB run name).
            config_dict: Config dict logged as wandb.config (checkpoints, benchmarks, etc.).
            benchmarks: List of benchmark names (used as tags).
            entity: WandB entity (team or user). None uses the default.
            project: WandB project name.
            group: WandB group name for grouping related runs.
            backfill_mode: Use in-training benchmark_eval schema
                (``benchmark_eval/`` prefix, ``train/ans_loop`` step).
            wandb_run_id: Resume an existing wandb run (for appending to
                training runs in backfill mode).
        """
        self.run_name = run_name
        self.config_dict = config_dict
        self.benchmarks = benchmarks
        self.entity = entity
        self.project = project
        self.group = group
        self.backfill_mode = backfill_mode
        self.wandb_run_id = wandb_run_id
        self._run = None
        self._pending_logs: dict[int, dict[str, Any]] = {}

    def start(self) -> bool:
        """Initialize WandB run.

        Returns:
            True if WandB was initialized successfully, False otherwise.
        """
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed — skipping WandB logging")
            return False

        if not os.environ.get("WANDB_API_KEY"):
            logger.warning(
                "WANDB_API_KEY not set — skipping WandB logging. "
                "Set the environment variable or use --no-wandb to suppress this warning."
            )
            return False

        tags = [t for t in list(self.benchmarks) + [self.run_name] if len(t) <= 64]

        init_kwargs = dict(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            config=self.config_dict,
            tags=tags,
        )
        if self.wandb_run_id:
            init_kwargs["id"] = self.wandb_run_id
            init_kwargs["resume"] = "must"
        elif self.group:
            init_kwargs["group"] = self.group

        self._run = wandb.init(**init_kwargs)

        if self.backfill_mode:
            wandb.define_metric("train/ans_loop")
            wandb.define_metric("benchmark_eval/*", step_metric="train/ans_loop")
        run = self._run

        # Log job scheduler metadata for cancel-via-wandb support
        skypilot_task_id = os.environ.get("SKYPILOT_TASK_ID")
        if skypilot_task_id:
            wandb.config.update({"skypilot_task_id": skypilot_task_id}, allow_val_change=True)

        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id and run is not None:
            run.summary["slurm_job_id"] = slurm_job_id

        if run is not None:
            run.summary["launch_command"] = " ".join(sys.argv)

        logger.info("WandB run initialized: %s/%s", self.project, self.run_name)
        return True

    def log_evaluation(
        self,
        benchmark: str,
        ckpt_num: int,
        metrics_dict: dict,
        question_results: list[dict] | None = None,
        n_samples: int | None = None,
    ) -> None:
        """Log metrics for a single (checkpoint, benchmark) evaluation.

        In normal mode, logs immediately at step=ckpt_num:
            - {benchmark}/accuracy_strict, etc.

        In backfill mode, accumulates metrics per checkpoint and logs them
        together via :meth:`flush_checkpoint` using the in-training schema:
            - benchmark_eval/{benchmark}/accuracy_strict, etc.
            - benchmark_eval/{benchmark}/sub_bench/{sub}/accuracy_strict

        Args:
            benchmark: Benchmark name.
            ckpt_num: Checkpoint number (used as step).
            metrics_dict: Metrics dict from compute_eval_metrics().
            question_results: Per-question result dicts (needed for sub_bench
                breakdown in backfill mode). Ignored in normal mode.
            n_samples: Number of samples per question (for sub_bench metrics).
        """
        if self._run is None:
            return

        if self.backfill_mode:
            self._log_evaluation_backfill(
                benchmark, ckpt_num, metrics_dict, question_results, n_samples,
            )
            return

        from verl_inf_evolve.sol_eval.eval_core import (
            compute_sub_bench_metrics,
            flatten_eval_metrics,
        )

        import wandb

        log_data = {
            f"{benchmark}/accuracy_strict": metrics_dict.get("accuracy_strict", 0.0),
            f"{benchmark}/accuracy_lenient": metrics_dict.get("accuracy_lenient", 0.0),
            f"{benchmark}/total_questions": metrics_dict.get("total_questions", 0),
        }

        score_name = metrics_dict.get("score_name")
        if score_name:
            strict_key = f"{score_name}_score_strict"
            lenient_key = f"{score_name}_score_lenient"
            if strict_key in metrics_dict:
                log_data[f"{benchmark}/{strict_key}"] = metrics_dict[strict_key]
            if lenient_key in metrics_dict:
                log_data[f"{benchmark}/{lenient_key}"] = metrics_dict[lenient_key]

        # pass@k ladders from pass_at_k_{strict,lenient}; keep the historical
        # ``pass_at_1`` alias for dashboards that already read it.
        pass_at_k_strict = metrics_dict.get("pass_at_k_strict", {})
        if isinstance(pass_at_k_strict, dict):
            for k_str, value in sorted(
                pass_at_k_strict.items(),
                key=lambda item: (
                    0, int(item[0])
                ) if str(item[0]).isdigit() else (1, str(item[0])),
            ):
                log_data[f"{benchmark}/pass_at_{k_str}_strict"] = value
        pass_at_k_lenient = metrics_dict.get("pass_at_k_lenient", {})
        if isinstance(pass_at_k_lenient, dict):
            for k_str, value in sorted(
                pass_at_k_lenient.items(),
                key=lambda item: (
                    0, int(item[0])
                ) if str(item[0]).isdigit() else (1, str(item[0])),
            ):
                log_data[f"{benchmark}/pass_at_{k_str}_lenient"] = value
        log_data[f"{benchmark}/pass_at_1"] = (
            pass_at_k_strict.get("1", 0.0)
            if isinstance(pass_at_k_strict, dict)
            else 0.0
        )

        # response length tokens mean
        response_length_tokens = metrics_dict.get("response_length_tokens", {})
        if isinstance(response_length_tokens, dict) and "mean" in response_length_tokens:
            log_data[f"{benchmark}/response_length_tokens_mean"] = response_length_tokens["mean"]

        # Prefer the pre-computed sub_bench_metrics that evaluate_benchmark_questions
        # already baked into the metrics dict — this survives
        # result_detail=metrics_only (which drops question_results). Fall back to
        # recomputing from question_results only when the metrics dict doesn't
        # carry it (e.g. an older cached result).
        sub_bench = dict(metrics_dict.get("sub_bench_metrics") or {})
        if not sub_bench and question_results:
            sub_bench = compute_sub_bench_metrics(question_results, n_samples=n_samples)
        for sub_name, sub_metrics in sub_bench.items():
            sub_prefix = f"{benchmark}/sub_bench/{sub_name}"
            log_data.update(flatten_eval_metrics(sub_prefix, sub_metrics))

        wandb.log(log_data, step=ckpt_num)
        # Also mirror into run.summary so the slash-notation keys are
        # definitely retrievable via the Public API on resumed runs, where
        # wandb.log({...}, step=N) at an already-locked history schema can
        # silently drop keys that weren't in the original schema (observed
        # with sub_bench/math, sub_bench/physics on tr81ew81 / xnzcc7pb / etc.
        # which were originally logged before those keys existed).
        try:
            self._run.summary.update(log_data)
        except Exception as exc:  # never break the eval over a summary write
            logger.warning("wandb summary.update failed: %s", exc)
        logger.info(
            "WandB logged: checkpoint=%d, benchmark=%s, accuracy_strict=%.4f, sub_bench_keys=%d",
            ckpt_num,
            benchmark,
            log_data.get(f"{benchmark}/accuracy_strict", 0.0),
            sum(1 for k in log_data if "/sub_bench/" in k),
        )

    def _log_evaluation_backfill(
        self,
        benchmark: str,
        ckpt_num: int,
        metrics_dict: dict,
        question_results: list[dict] | None,
        n_samples: int | None,
    ) -> None:
        """Accumulate backfill-mode metrics for a checkpoint.

        Uses ``benchmark_eval/{name}/*`` prefix and computes sub_bench
        breakdown from question_results when available.
        """
        from verl_inf_evolve.sol_eval.eval_core import (
            compute_sub_bench_metrics,
            flatten_eval_metrics,
        )

        payload = self._pending_logs.setdefault(ckpt_num, {"train/ans_loop": ckpt_num})

        prefix = f"benchmark_eval/{benchmark}"
        payload.update(flatten_eval_metrics(prefix, metrics_dict))

        # Prefer the pre-computed sub_bench_metrics on the metrics dict (populated
        # by evaluate_benchmark_questions) so backfill also works under
        # result_detail=metrics_only. Recompute from question_results only as a
        # last resort.
        sub_bench = dict(metrics_dict.get("sub_bench_metrics") or {})
        if not sub_bench and question_results:
            sub_bench = compute_sub_bench_metrics(question_results, n_samples=n_samples)
        for sub_name, sub_metrics in sub_bench.items():
            sub_prefix = f"benchmark_eval/{benchmark}/sub_bench/{sub_name}"
            payload.update(flatten_eval_metrics(sub_prefix, sub_metrics))

        logger.info(
            "WandB backfill accumulated: checkpoint=%d, benchmark=%s, accuracy_strict=%.4f",
            ckpt_num,
            benchmark,
            metrics_dict.get("accuracy_strict", 0.0),
        )

    def flush_checkpoint(self, ckpt_num: int) -> None:
        """Log all accumulated backfill metrics for a checkpoint."""
        if self._run is None:
            return
        payload = self._pending_logs.pop(ckpt_num, None)
        if not payload:
            return

        import wandb

        wandb.log(payload, step=ckpt_num)
        logger.info(
            "WandB backfill flushed: checkpoint=%d, metrics=%d",
            ckpt_num,
            len(payload),
        )

    def log_summary(self, all_results: list[dict]) -> None:
        """Compute and log best accuracy + checkpoint per benchmark as run summary.

        Args:
            all_results: List of result dicts, each with:
                - 'metrics' dict containing 'accuracy_strict'
                - '_eval_metadata' dict with 'benchmark' and 'ckpt_num' keys
                  (added by evaluate_run before calling this method)
        """
        if self._run is None:
            return

        import wandb

        # Track best accuracy per benchmark
        best: dict[str, dict[str, Any]] = {}
        for result in all_results:
            meta = result.get("_eval_metadata", {})
            benchmark = meta.get("benchmark")
            ckpt_num = meta.get("ckpt_num")
            if not benchmark or ckpt_num is None:
                continue

            metrics = result.get("metrics", {})
            metric_name = metrics.get("primary_metric_name", "accuracy_strict")
            metric_value = metrics.get(
                "primary_metric_value",
                metrics.get("accuracy_strict", 0.0),
            )

            if benchmark not in best or metric_value > best[benchmark]["metric_value"]:
                best[benchmark] = {
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "checkpoint": ckpt_num,
                }

        for benchmark, info in best.items():
            wandb.run.summary[f"best_{benchmark}_{info['metric_name']}"] = info["metric_value"]
            wandb.run.summary[f"best_{benchmark}_primary_metric_name"] = info["metric_name"]
            wandb.run.summary[f"best_{benchmark}_primary_metric"] = info["metric_value"]
            wandb.run.summary[f"best_{benchmark}_checkpoint"] = info["checkpoint"]

        logger.info("WandB summary logged for %d benchmarks", len(best))

    def finish(self) -> None:
        """Finish the WandB run."""
        if self._run is None:
            return

        import wandb

        wandb.finish()
        self._run = None
        logger.info("WandB run finished")
