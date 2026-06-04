"""In-training benchmark evaluation.

Runs standard benchmarks (GPQA, AIME, etc.) inside the training loop,
reusing the solver's existing vLLM engine and FSDP weights.

Core benchmark evaluation logic is centralized in ``verl_inf_evolve.sol_eval``
so in-training eval and standalone ``sol_eval`` share one implementation.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def benchmark_output_name(benchmark_name: str) -> str:
    """Convert benchmark name/path into a stable filename component."""
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(benchmark_name)).strip("_")
    if not normalized:
        normalized = "benchmark"
    return normalized


def benchmark_output_filename(benchmark_name: str) -> str:
    """Return the output filename for one benchmark eval result."""
    return f"benchmark_eval_{benchmark_output_name(benchmark_name)}.pt"


def benchmark_output_remote_key(ans_loop: int, benchmark_name: str) -> str:
    """Return the R2 key for one benchmark eval result."""
    return f"ans_{ans_loop}/{benchmark_output_filename(benchmark_name)}"


def benchmark_output_exists(
    upload_manager: Any,
    ans_loop: int,
    benchmark_name: str,
    resume_dir: str | None = None,
) -> bool:
    """Check whether benchmark raw output already exists (local or R2)."""
    filename = benchmark_output_filename(benchmark_name)

    if resume_dir:
        local_path = os.path.join(resume_dir, filename)
        if os.path.isfile(local_path):
            return True

    if upload_manager and upload_manager.remote_configured:
        return upload_manager.remote_exists(
            benchmark_output_remote_key(ans_loop, benchmark_name)
        )
    return False


def persist_benchmark_output(
    upload_manager: Any,
    ans_loop: int,
    benchmark_name: str,
    benchmark_output: Any,
) -> None:
    """Persist in-training benchmark rollout output to R2 only."""
    if upload_manager and upload_manager.remote_configured:
        upload_manager.submit_memory_upload(
            name=f"benchmark_eval_{benchmark_name}",
            data=benchmark_output,
            kind="dataproto",
            remote_key=benchmark_output_remote_key(ans_loop, benchmark_name),
        )


class InTrainingBenchmarkEvaluator:
    """Run benchmark evaluation inside the training loop.

    Reuses the trainer's existing solver runtime, but delegates benchmark
    conversion/scoring/metrics to ``sol_eval.eval_core``.
    """

    def __init__(
        self,
        config,
        solver_rollout_manager,
        solver_tokenizer,
        solver_num_workers: int,
        max_prompt_tokens: int,
        max_ans_loop: int,
        generate_fn: Callable,
        on_benchmark_output_fn: Optional[Callable[[int, str, Any], None]] = None,
        benchmark_output_exists_fn: Optional[Callable[[int, str], bool]] = None,
    ):
        from verl_inf_evolve.sol_eval.eval_core import (
            load_benchmark_questions,
            validate_code_execution_policy,
        )
        from verl_inf_evolve.sol_eval.runtime import SolverEvalRuntime

        self._config = config
        self._manager = solver_rollout_manager
        self._tokenizer = solver_tokenizer
        self._num_workers = solver_num_workers
        self._max_prompt_tokens = max_prompt_tokens
        self._max_ans_loop = max_ans_loop
        self._generate_fn = generate_fn
        self._on_benchmark_output_fn = on_benchmark_output_fn
        self._benchmark_output_exists_fn = benchmark_output_exists_fn
        code_execution_cfg = config.get("code_execution", {})
        self._code_execution_enabled = bool(code_execution_cfg.get("enabled", False))
        self._allow_code_execution_in_training = bool(
            code_execution_cfg.get("allow_in_training", False)
        )

        self._runtime = SolverEvalRuntime.from_existing_worker_group(
            tokenizer=solver_tokenizer,
            num_workers=solver_num_workers,
            generate_batch_fn=lambda batch: self._generate_fn(
                self._manager,
                batch,
                self._num_workers,
            ),
            rollout_manager=solver_rollout_manager,
        )

        # Pre-load all benchmark JSONs.
        self._benchmarks: dict[str, list[dict]] = {}
        for name in config.benchmarks:
            try:
                questions, path = load_benchmark_questions(
                    name,
                    max_questions=config.get("max_questions", 0),
                )
                validate_code_execution_policy(
                    questions,
                    code_execution_enabled=self._code_execution_enabled,
                    execution_scope="in_training",
                    allow_code_execution_in_training=self._allow_code_execution_in_training,
                )
                self._benchmarks[name] = questions
                logger.info(
                    "Loaded benchmark %s: %d questions from %s",
                    name,
                    len(questions),
                    path,
                )
            except FileNotFoundError:
                logger.warning("Benchmark %s not found — skipping", name)
            except ValueError:
                logger.exception("Benchmark %s load failed — skipping", name)

        if not self._benchmarks:
            logger.warning(
                "No benchmarks loaded — in-training evaluation will be a no-op"
            )

    def should_evaluate(self, ans_loop: int) -> bool:
        """Check whether evaluation should run at this ans_loop."""
        if not self._benchmarks:
            return False

        cfg = self._config
        is_first = ans_loop == 0
        is_last = ans_loop == self._max_ans_loop - 1

        if cfg.eval_on_first_step and is_first:
            return True
        if cfg.eval_on_last_step and is_last:
            return True
        if cfg.eval_every_n_steps > 0 and ans_loop % cfg.eval_every_n_steps == 0:
            return True
        return False

    def evaluate_all(self, ans_loop: int) -> dict[str, Any]:
        """Evaluate all loaded benchmarks and return flat WandB metrics."""
        all_metrics: dict[str, Any] = {}

        for name, questions in self._benchmarks.items():
            if self._benchmark_output_exists_fn is not None:
                try:
                    if self._benchmark_output_exists_fn(ans_loop, name):
                        logger.info(
                            "Benchmark eval: skipping %s at ans_loop=%d (raw output already exists)",
                            name,
                            ans_loop,
                        )
                        continue
                except Exception:
                    logger.exception(
                        "Benchmark eval: output existence check failed for %s at ans_loop=%d; proceeding",
                        name,
                        ans_loop,
                    )
            try:
                metrics = self._evaluate_single_benchmark(
                    ans_loop=ans_loop,
                    name=name,
                    questions=questions,
                )
                all_metrics.update(metrics)
            except Exception:
                logger.exception(
                    "Benchmark eval failed for %s at ans_loop=%d",
                    name,
                    ans_loop,
                )

        return all_metrics

    def _evaluate_single_benchmark(
        self,
        ans_loop: int,
        name: str,
        questions: list[dict],
    ) -> dict[str, Any]:
        """Evaluate one benchmark and return prefixed metrics."""
        from verl_inf_evolve.sol_eval.eval_core import (
            compute_sub_bench_metrics,
            evaluate_benchmark_questions,
            flatten_eval_metrics,
        )

        logger.info("Benchmark eval: %s (%d questions)", name, len(questions))

        def _on_generation_output(output: Any) -> None:
            if self._on_benchmark_output_fn is None:
                return
            try:
                self._on_benchmark_output_fn(ans_loop, name, output)
            except Exception:
                logger.exception(
                    "Failed to persist benchmark raw output for %s at ans_loop=%d",
                    name,
                    ans_loop,
                )

        eval_result = evaluate_benchmark_questions(
            questions=questions,
            tokenizer=self._tokenizer,
            max_prompt_tokens=self._max_prompt_tokens,
            n_samples=self._config.n_samples,
            generate_batch_fn=self._runtime.generate_batch,
            use_public_eval_prompt=bool(
                self._config.get("use_public_eval_prompt", False)
            ),
            raise_if_all_filtered=False,
            on_generation_output_fn=_on_generation_output,
            code_execution_enabled=self._code_execution_enabled,
            execution_scope="in_training",
            allow_code_execution_in_training=self._allow_code_execution_in_training,
        )
        if eval_result is None:
            logger.warning("Benchmark %s: all questions filtered — skipping", name)
            return {}

        metrics = eval_result.metrics
        logger.info(
            "Benchmark %s: accuracy_strict=%.4f, accuracy_lenient=%.4f (%d questions)",
            name,
            metrics["accuracy_strict"],
            metrics["accuracy_lenient"],
            metrics["total_questions"],
        )

        prefix = f"benchmark_eval/{name}"
        flat = flatten_eval_metrics(prefix, metrics)

        # Break down combined benchmarks by data_source.
        sub_bench = compute_sub_bench_metrics(
            eval_result.question_results,
            n_samples=self._config.n_samples,
        )
        for sub_name, sub_metrics in sub_bench.items():
            sub_prefix = f"benchmark_eval/{name}/sub_bench/{sub_name}"
            flat.update(flatten_eval_metrics(sub_prefix, sub_metrics))
            logger.info(
                "Benchmark %s/%s: accuracy_strict=%.4f (%d questions)",
                name,
                sub_name,
                sub_metrics["accuracy_strict"],
                sub_metrics["total_questions"],
            )

        return flat
