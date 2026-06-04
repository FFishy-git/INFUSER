"""Evaluation runner for single and multi-checkpoint evaluation.

Uses the verl worker class to evaluate checkpoints on benchmarks, following
the same inference path as training. Outputs in the run_eval_direct.py format
(with 'questions' and 'metrics' keys) using sol_eval/result_format.py utilities.

Supports multi-checkpoint evaluation with weight reloading between checkpoints
to avoid reinitializing Ray and vLLM for each checkpoint.
"""

from __future__ import annotations

import glob as globmod
import json
import logging
import os
import tempfile
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _resolve_n_samples(eval_cfg: Any, benchmark: str) -> int:
    """Return per-benchmark n_samples, falling back to eval.n_samples."""
    overrides = eval_cfg.get("benchmark_n_samples", None) or {}
    return int(overrides.get(benchmark, eval_cfg.n_samples))


def _resolve_max_out_len(eval_cfg: Any, benchmark: str, default: int) -> int:
    """Return per-benchmark max output length, falling back to *default*."""
    overrides = eval_cfg.get("benchmark_max_out_len", None) or {}
    return int(overrides.get(benchmark, default))


def _solver_temperature(config: Any) -> float:
    """Return the rollout temperature used for generation and eval metadata."""
    return float(config.solver.rollout.temperature)


def _solver_top_p(config: Any) -> float:
    """Return the rollout top-p used for direct vLLM generation."""
    return float(config.solver.rollout.top_p)


def _solver_top_k(config: Any) -> int:
    """Return the rollout top-k used for direct vLLM generation."""
    return int(config.solver.rollout.top_k)


def _solver_prompt_length(config: Any) -> int:
    """Return the rollout prompt-length cap."""
    return int(config.solver.rollout.prompt_length)


def _solver_response_length(config: Any) -> int:
    """Return the rollout response-length cap."""
    return int(config.solver.rollout.response_length)


def _solver_effective_max_model_len(config: Any) -> int:
    """Return an effective total context budget derived from rollout lengths."""
    return _solver_prompt_length(config) + _solver_response_length(config)


class EvalRunner:
    """Evaluation runner using verl worker for single checkpoint evaluation.

    Follows the BenchmarkEvalRunner pattern from benchmark_eval.py but outputs
    in the run_eval_direct.py format using sol_eval/result_format.py utilities.
    """

    def __init__(self, config, runtime: Any | None = None):
        """Initialize EvalRunner with a solver evaluation runtime.

        Args:
            config: Pre-built, resolved OmegaConf config with solver, trainer,
                data, eval, reward_model sections (from sol_eval.yaml).
            runtime: Optional ``SolverEvalRuntime``. If omitted, a fresh solver
                worker group is initialized.
        """
        from verl_inf_evolve.sol_eval.runtime import SolverEvalRuntime

        self._config = config
        self.model_path = config.solver.model.path
        self.tp_size = config.eval.tp_size
        self.n_samples = config.eval.n_samples
        self.temperature = _solver_temperature(config)
        self.top_p = _solver_top_p(config)
        self.top_k = _solver_top_k(config)
        self.gpu_memory_utilization = config.eval.gpu_memory_utilization
        self.result_detail = config.eval.get("result_detail", "metrics_only")
        self.code_execution_enabled = bool(
            config.eval.get("code_execution", {}).get("enabled", False)
        )

        self.runtime = runtime or SolverEvalRuntime.from_fresh_worker_group(config)
        self.solver_wg = self.runtime.solver_wg
        if self.solver_wg is None:
            raise ValueError("EvalRunner requires runtime.solver_wg for checkpoint loading")
        self.rollout_manager = self.runtime.rollout_manager
        self.tokenizer = self.runtime.tokenizer
        self.n_gpus = self.runtime.num_workers

        logger.info(
            "EvalRunner initialized: model=%s, n_gpus=%d, tp_size=%d",
            self.model_path,
            self.n_gpus,
            self.tp_size,
        )

    def evaluate_single(
        self,
        checkpoint_path: str,
        benchmark_path: str,
        n_samples: int,
        temperature: float,
        max_model_len: int,
        on_generation_output_fn: Callable[[Any], None] | None = None,
    ) -> dict:
        """Run evaluation of a single checkpoint on a single benchmark.

        Args:
            checkpoint_path: Local path to the checkpoint directory.
            benchmark_path: Path to benchmark JSON file (list of question dicts).
            n_samples: Number of rollout samples per question.
            temperature: Sampling temperature (used for output metadata).
            max_model_len: Maximum model length (used for output metadata).

        Returns:
            Result dict with 'questions' and 'metrics' keys matching
            run_eval_direct.py output format.
        """
        from verl_inf_evolve.sol_eval.eval_core import (
            evaluate_benchmark_questions,
            load_benchmark_questions,
        )
        from verl_inf_evolve.sol_eval.result_format import format_result_json

        # ``checkpoint_path`` is used by the caller for weight loading; keep
        # it in the signature for compatibility even though evaluation itself
        # only needs the benchmark questions and active runtime.
        del checkpoint_path, temperature, max_model_len  # noqa: F841

        max_questions = int(self._config.eval.get("max_questions", 0) or 0)
        questions, resolved_benchmark_path = load_benchmark_questions(
            benchmark_path,
            max_questions=max_questions,
        )
        logger.info(
            "Loaded %d benchmark questions from %s",
            len(questions),
            resolved_benchmark_path,
        )
        eval_result = evaluate_benchmark_questions(
            questions=questions,
            tokenizer=self.tokenizer,
            max_prompt_tokens=self._config.solver.rollout.prompt_length,
            n_samples=n_samples,
            generate_batch_fn=self.runtime.generate_batch,
            # Pass the raw config value through. build_benchmark_messages
            # resolves it per-benchmark via resolve_use_public_eval_prompt,
            # supporting both the legacy bool form and the new dict form
            # ``{default: bool, benchmarks: {data_source: bool}}``.
            use_public_eval_prompt=self._config.eval.get(
                "use_public_eval_prompt", False
            ),
            raise_if_all_filtered=True,
            on_generation_output_fn=on_generation_output_fn,
            code_execution_enabled=self.code_execution_enabled,
            execution_scope="standalone",
            mcq_choice_shuffle_config=self._config.eval.get("mcq_choice_shuffle", None),
        )
        if eval_result is None:
            raise ValueError("All questions were filtered out (too long)")

        result_json = format_result_json(
            eval_result.question_results,
            eval_result.metrics,
            result_detail=self.result_detail,
        )

        logger.info(
            "Evaluation complete: %d questions, accuracy_strict=%.4f, accuracy_lenient=%.4f",
            eval_result.metrics["total_questions"],
            eval_result.metrics["accuracy_strict"],
            eval_result.metrics["accuracy_lenient"],
        )

        return result_json

    def save_result(
        self,
        result_json: dict,
        benchmark: str,
        run_name: str,
        ckpt_num: int,
        temperature: float,
        max_model_len: int,
        result_detail: str = "metrics_only",
        output_dir: str = ".",
        model_tag: str | None = None,
    ) -> str:
        """Save result JSON to local path using build_output_filename.

        Args:
            result_json: Result dict to save.
            benchmark: Benchmark name (e.g. 'supergpqa_2000').
            run_name: Training run name.
            ckpt_num: Checkpoint number.
            temperature: Sampling temperature.
            max_model_len: Max model length.
            output_dir: Directory to save the result file.
            model_tag: Optional model identifier to disambiguate cross-model runs.

        Returns:
            Path to the saved file.
        """
        from verl_inf_evolve.sol_eval.result_format import build_output_filename

        filename = build_output_filename(
            benchmark,
            run_name,
            ckpt_num,
            temperature,
            max_model_len,
            result_detail=result_detail,
            model_tag=model_tag,
        )
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result_json, f, indent=2)
        logger.info("Result saved to %s", output_path)
        return output_path

    def _upload_training_artifact(
        self,
        dataproto_output: Any,
        benchmark: str,
        ckpt_num: int,
        remote_sync_path: str,
        remote_cfg: Any,
    ) -> bool:
        """Serialize DataProto and upload as .pt to training remote_sync_path.

        Mimics ``persist_benchmark_output()`` from ``trainer/benchmark_eval.py``.
        """
        import io
        import pickle

        from verl_inf_evolve.sol_eval.r2_ops import _create_remote_backend
        from verl_inf_evolve.trainer.benchmark_eval import benchmark_output_remote_key

        remote_key = benchmark_output_remote_key(ckpt_num, benchmark)

        buf = io.BytesIO()
        pickle.dump(dataproto_output, buf, protocol=pickle.HIGHEST_PROTOCOL)
        serialized = buf.getvalue()

        try:
            backend, resolved_uri = _create_remote_backend(
                remote_sync_path, remote_cfg, auto_create_repo=True,
            )
            ok = backend.upload_bytes(serialized, remote_key)
            if ok:
                logger.info(
                    "Uploaded training artifact to %s/%s (%.1f MB)",
                    resolved_uri,
                    remote_key,
                    len(serialized) / 1e6,
                )
            else:
                logger.error(
                    "Failed to upload training artifact to %s/%s",
                    resolved_uri,
                    remote_key,
                )
            return ok
        except Exception:
            logger.exception(
                "Failed to upload training artifact for %s at ckpt %d",
                benchmark,
                ckpt_num,
            )
            return False

    def _download_checkpoint(
        self,
        remote_sync_path: str,
        ckpt_num: int,
        run_name: str,
    ) -> str:
        """Download a checkpoint from remote storage to local cache.

        Checkpoint path resolution:
        - V3: {remote_sync_path}/global_step_{N}/solver/

        Local cache: .cache/eval_checkpoints/{run_name}/global_step_{N}/

        Args:
            remote_sync_path: Remote experiment root.
            ckpt_num: Checkpoint number (global step).
            run_name: Training run name (for local cache dir).

        Returns:
            Local path to the downloaded checkpoint directory.

        Raises:
            RuntimeError: If download fails.
        """
        from verl_inf_evolve.sol_eval.r2_ops import download_checkpoint_from_r2

        local_cache_dir = os.path.join(
            ".cache", "eval_checkpoints", run_name, f"global_step_{ckpt_num}"
        )
        return download_checkpoint_from_r2(
            remote_sync_path=remote_sync_path,
            ckpt_num=ckpt_num,
            local_cache_dir=local_cache_dir,
            remote_cfg=self._config.get("remote", {}),
        )

    @staticmethod
    def _detect_checkpoint_format(checkpoint_dir: str) -> str:
        """Detect whether a checkpoint is FSDP sharded or HuggingFace format.

        For eval we only need model weights (no optimizer/scheduler), so we
        prefer HuggingFace safetensors when available.  This avoids failures
        when FSDP checkpoints are incomplete (e.g. missing optimizer shards)
        which would corrupt the distributed process group.

        Args:
            checkpoint_dir: Local path to the checkpoint's global_step_{N} directory.

        Returns:
            "huggingface" if HF safetensors found, "fsdp" otherwise.

        Raises:
            ValueError: If neither format is detected.
        """
        solver_dir = os.path.join(checkpoint_dir, "solver")

        # Prefer HuggingFace: only needs config.json + model weights
        hf_subdir = os.path.join(solver_dir, "huggingface")
        for hf_dir in (hf_subdir, solver_dir):
            if os.path.isfile(os.path.join(hf_dir, "config.json")):
                has_weights = (
                    os.path.isfile(os.path.join(hf_dir, "model.safetensors"))
                    or os.path.isfile(
                        os.path.join(hf_dir, "model.safetensors.index.json")
                    )
                )
                if has_weights:
                    return "huggingface"

        # Fall back to FSDP: model_world_size_*_rank_*.pt
        fsdp_pattern = os.path.join(solver_dir, "model_world_size_*_rank_*.pt")
        if globmod.glob(fsdp_pattern):
            return "fsdp"

        # If we found config.json but no weights, the checkpoint was likely cleaned up
        has_config = os.path.isfile(
            os.path.join(hf_subdir, "config.json")
        ) or os.path.isfile(os.path.join(solver_dir, "config.json"))

        if has_config:
            raise ValueError(
                f"Checkpoint at {checkpoint_dir} has config/tokenizer metadata "
                f"but no model weights (no FSDP shards or safetensors). "
                f"The model weights were likely cleaned up from R2. "
                f"Only the most recent checkpoint retains full weights."
            )

        raise ValueError(
            f"Cannot detect checkpoint format at {checkpoint_dir}: "
            f"no FSDP shards (model_world_size_*_rank_*.pt) or "
            f"HuggingFace config.json found in solver/"
        )

    def _load_checkpoint_weights(self, checkpoint_dir: str) -> None:
        """Load checkpoint weights into the solver worker group.

        For FSDP checkpoints: calls solver_wg.load_checkpoint(local_path=solver_dir)
        For HuggingFace checkpoints: calls solver_wg.load_hf_checkpoint which
            loads safetensors via FSDP's FULL_STATE_DICT mechanism.

        Args:
            checkpoint_dir: Local path to the checkpoint's global_step_{N} directory.

        Raises:
            RuntimeError: If weight loading fails.
        """
        fmt = self._detect_checkpoint_format(checkpoint_dir)
        solver_dir = os.path.join(checkpoint_dir, "solver")

        if fmt == "fsdp":
            logger.info("Loading FSDP checkpoint from %s", solver_dir)
            try:
                self.solver_wg.load_checkpoint(local_path=solver_dir)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load FSDP checkpoint from {solver_dir}: {e}"
                ) from e
        else:
            # HuggingFace format — prefer huggingface/ subdir if it exists
            hf_subdir = os.path.join(solver_dir, "huggingface")
            hf_path = hf_subdir if os.path.isdir(hf_subdir) else solver_dir
            logger.info("Loading HuggingFace checkpoint from %s", hf_path)
            try:
                self.solver_wg.load_hf_checkpoint(local_path=hf_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load HuggingFace checkpoint from {hf_path}: {e}"
                ) from e

        logger.info("Checkpoint weights loaded from %s (format=%s)", checkpoint_dir, fmt)

    @staticmethod
    def _resolve_benchmark_path(benchmark_name: str) -> str:
        """Backward-compatible wrapper around shared benchmark path resolution."""
        from verl_inf_evolve.sol_eval.eval_core import resolve_benchmark_path

        return resolve_benchmark_path(benchmark_name)

    def evaluate_run(self) -> list[dict]:
        """Evaluate multiple checkpoints on multiple benchmarks.

        Reads all parameters from self._config.eval.* and self._config.wandb.*.
        Iterates over all checkpoints × benchmarks, reusing the worker group
        and reloading weights between checkpoints. Supports skip-existing
        (checks R2 for valid results), R2 upload, checkpoint cleanup, and
        optional WandB logging.

        Returns:
            List of result dicts (one per checkpoint × benchmark evaluation).
        """
        from verl_inf_evolve.sol_eval.r2_ops import (
            check_r2_result_exists,
            cleanup_checkpoint,
            download_r2_result,
            upload_result_to_r2,
        )
        from verl_inf_evolve.sol_eval.result_format import build_output_filename

        # Read eval parameters from config
        eval_cfg = self._config.eval
        run_name = eval_cfg.run_name
        model_tag = eval_cfg.get("model_tag", None) or None
        remote_sync_path = eval_cfg.remote_sync_path
        checkpoints = list(eval_cfg.checkpoints)
        benchmarks = list(eval_cfg.benchmarks)
        temperature = _solver_temperature(self._config)
        effective_max_model_len = _solver_effective_max_model_len(self._config)
        response_length = _solver_response_length(self._config)
        code_execution_enabled = bool(
            eval_cfg.get("code_execution", {}).get("enabled", False)
        )
        result_detail = eval_cfg.get("result_detail", "metrics_only")
        remote_eval_base = eval_cfg.get("remote_eval_base", None) or eval_cfg.r2_eval_base
        force = eval_cfg.force
        no_r2_upload = eval_cfg.no_r2_upload
        do_cleanup = eval_cfg.cleanup_checkpoints
        no_wandb = eval_cfg.no_wandb
        upload_training_artifacts = eval_cfg.get("upload_training_artifacts", False)
        backfill_mode = eval_cfg.get("backfill_mode", False)
        wandb_run_id = eval_cfg.get("wandb_run_id", None)
        remote_cfg = self._config.get("remote", {})
        wandb_entity = self._config.wandb.entity
        wandb_project = self._config.wandb.project_name
        wandb_run_name = self._config.wandb.get("run_name", None) or run_name
        wandb_group_name = self._config.wandb.get("group_name", None)

        # Initialize WandB logger if enabled
        wandb_logger = None
        if not no_wandb:
            from verl_inf_evolve.sol_eval.wandb_logger import EvalWandBLogger

            wandb_logger = EvalWandBLogger(
                run_name=wandb_run_name,
                config_dict={
                    "checkpoints": checkpoints,
                    "benchmarks": benchmarks,
                    "temperature": temperature,
                    "max_model_len": effective_max_model_len,
                    "n_samples": {b: _resolve_n_samples(eval_cfg, b) for b in benchmarks},
                    "result_detail": result_detail,
                    "remote_sync_path": remote_sync_path,
                },
                benchmarks=benchmarks,
                entity=wandb_entity,
                project=wandb_project,
                group=wandb_group_name,
                backfill_mode=backfill_mode,
                wandb_run_id=str(wandb_run_id) if wandb_run_id else None,
            )
            if not wandb_logger.start():
                wandb_logger = None

        checkpoint_cache_dir = eval_cfg.get("checkpoint_cache_dir", None)

        results = []
        downloaded_checkpoints: set[int] = set()
        total_ckpts = len(checkpoints)
        total_benchmarks = len(benchmarks)

        for ckpt_idx, ckpt_num in enumerate(checkpoints, 1):
            # Pre-check which benchmarks can be skipped before downloading
            benchmarks_to_eval = []
            for benchmark in benchmarks:
                output_filename = build_output_filename(
                    benchmark=benchmark,
                    run_name=run_name,
                    ckpt_num=ckpt_num,
                    temperature=temperature,
                    max_model_len=effective_max_model_len,
                    result_detail=result_detail,
                    model_tag=model_tag,
                )

                if not force and check_r2_result_exists(
                    remote_eval_base,
                    output_filename,
                    remote_cfg=remote_cfg,
                ):
                    print(
                        f"Skipping {benchmark} checkpoint {ckpt_num} "
                        f"(valid result exists remotely)"
                    )
                    existing = download_r2_result(
                        remote_eval_base,
                        output_filename,
                        remote_cfg=remote_cfg,
                    )
                    if existing is not None:
                        existing["_eval_metadata"] = {
                            "benchmark": benchmark,
                            "ckpt_num": ckpt_num,
                        }
                        if wandb_logger is not None:
                            wandb_logger.log_evaluation(
                                benchmark, ckpt_num, existing.get("metrics", {}),
                                question_results=existing.get("questions"),
                                n_samples=_resolve_n_samples(eval_cfg, benchmark),
                            )
                        results.append(existing)
                else:
                    benchmarks_to_eval.append(benchmark)

            # Skip download + weight loading if all benchmarks already done
            if not benchmarks_to_eval:
                logger.info(
                    "Checkpoint %d: all %d benchmarks already evaluated, "
                    "skipping download",
                    ckpt_num,
                    len(benchmarks),
                )
                if wandb_logger is not None:
                    wandb_logger.flush_checkpoint(ckpt_num)
                continue

            logger.info(
                "Checkpoint %d: %d/%d benchmarks need evaluation",
                ckpt_num,
                len(benchmarks_to_eval),
                len(benchmarks),
            )

            # Resolve checkpoint directory
            if ckpt_num == -1:
                # Base model: skip download and weight loading — the model
                # is already initialized with base weights from solver.model.path
                logger.info(
                    "Checkpoint -1 (base model): using initial weights from %s",
                    self.model_path,
                )
                local_ckpt_dir = None
            elif checkpoint_cache_dir:
                # Use pre-downloaded checkpoint from cache dir
                local_ckpt_dir = os.path.join(
                    checkpoint_cache_dir, run_name, f"global_step_{ckpt_num}",
                )
                if not os.path.isdir(local_ckpt_dir):
                    logger.warning(
                        "Pre-cached checkpoint not found at %s, falling back to download",
                        local_ckpt_dir,
                    )
                    local_ckpt_dir = self._download_checkpoint(
                        remote_sync_path=remote_sync_path,
                        ckpt_num=ckpt_num,
                        run_name=run_name,
                    )
                else:
                    logger.info("Using pre-cached checkpoint at %s", local_ckpt_dir)
            elif ckpt_num not in downloaded_checkpoints:
                local_ckpt_dir = self._download_checkpoint(
                    remote_sync_path=remote_sync_path,
                    ckpt_num=ckpt_num,
                    run_name=run_name,
                )
                downloaded_checkpoints.add(ckpt_num)
            else:
                local_ckpt_dir = os.path.join(
                    ".cache", "eval_checkpoints",
                    run_name, f"global_step_{ckpt_num}",
                )

            # Load checkpoint weights (skip for base model)
            if local_ckpt_dir is not None:
                self._load_checkpoint_weights(local_ckpt_dir)

            for bench_idx, benchmark in enumerate(benchmarks_to_eval, 1):
                bench_n = _resolve_n_samples(eval_cfg, benchmark)
                print(
                    f"Evaluating checkpoint {ckpt_idx}/{total_ckpts} "
                    f"on {benchmark} ({bench_idx}/{len(benchmarks_to_eval)}) "
                    f"n={bench_n}"
                )

                output_filename = build_output_filename(
                    benchmark=benchmark,
                    run_name=run_name,
                    ckpt_num=ckpt_num,
                    temperature=temperature,
                    max_model_len=effective_max_model_len,
                    result_detail=result_detail,
                    model_tag=model_tag,
                )
                captured_output = [None]
                from verl_inf_evolve.sol_eval.external_benchmarks import (
                    is_external_benchmark,
                    run_external_benchmark,
                )

                if is_external_benchmark(benchmark):
                    model_ref = local_ckpt_dir if local_ckpt_dir is not None else str(self.model_path)
                    result = run_external_benchmark(
                        benchmark=benchmark,
                        model_path_or_ckpt_dir=model_ref,
                        n_samples=bench_n,
                        temperature=float(temperature),
                        top_p=float(self.top_p),
                        top_k=int(self.top_k),
                        max_generation_length=_resolve_max_out_len(
                            eval_cfg, benchmark, int(response_length)
                        ),
                        prompt_length=_solver_prompt_length(self._config),
                        tp_size=int(self.tp_size),
                        trust_remote_code=bool(
                            self._config.solver.model.get("trust_remote_code", True)
                        ),
                        result_detail=result_detail,
                        base_model_path=str(self.model_path),
                    )
                else:
                    benchmark_path = self._resolve_benchmark_path(benchmark)

                    def _capture_output(output: Any) -> None:
                        captured_output[0] = output

                    result = self.evaluate_single(
                        checkpoint_path=local_ckpt_dir,
                        benchmark_path=benchmark_path,
                        n_samples=bench_n,
                        temperature=temperature,
                        max_model_len=effective_max_model_len,
                        on_generation_output_fn=_capture_output if upload_training_artifacts else None,
                    )

                # Save result locally
                effective_run_dir = f"{model_tag}-{run_name}" if model_tag else run_name
                output_dir = os.path.join(
                    ".cache", "eval_results", effective_run_dir
                )
                local_result_path = self.save_result(
                    result_json=result,
                    benchmark=benchmark,
                    run_name=run_name,
                    ckpt_num=ckpt_num,
                    temperature=temperature,
                    max_model_len=effective_max_model_len,
                    result_detail=result_detail,
                    output_dir=output_dir,
                    model_tag=model_tag,
                )

                # Upload to remote storage unless disabled
                if not no_r2_upload:
                    upload_result_to_r2(
                        local_path=local_result_path,
                        r2_eval_base=remote_eval_base,
                        output_filename=output_filename,
                        remote_cfg=remote_cfg,
                    )

                # Upload raw DataProto .pt artifact to training remote_sync_path
                if upload_training_artifacts and captured_output[0] is not None:
                    if not remote_sync_path:
                        logger.warning(
                            "upload_training_artifacts=true but remote_sync_path "
                            "is not set; skipping .pt upload for %s at ckpt %d",
                            benchmark,
                            ckpt_num,
                        )
                    else:
                        self._upload_training_artifact(
                            dataproto_output=captured_output[0],
                            benchmark=benchmark,
                            ckpt_num=ckpt_num,
                            remote_sync_path=remote_sync_path,
                            remote_cfg=remote_cfg,
                        )
                    captured_output[0] = None

                # Attach eval metadata and log to WandB
                result["_eval_metadata"] = {
                    "benchmark": benchmark,
                    "ckpt_num": ckpt_num,
                }
                if wandb_logger is not None:
                    wandb_logger.log_evaluation(
                        benchmark, ckpt_num, result.get("metrics", {}),
                        question_results=result.get("questions"),
                        n_samples=bench_n,
                    )

                results.append(result)

            # Flush accumulated backfill metrics for this checkpoint
            if wandb_logger is not None:
                wandb_logger.flush_checkpoint(ckpt_num)

            # Cleanup local checkpoint after all benchmarks for this checkpoint
            # Skip cleanup if using a shared pre-downloaded cache or base model
            if do_cleanup and not checkpoint_cache_dir and local_ckpt_dir is not None:
                cleanup_checkpoint(local_ckpt_dir)

        # Log WandB summary and finish
        if wandb_logger is not None:
            wandb_logger.log_summary(results)
            wandb_logger.finish()

        logger.info(
            "evaluate_run complete: %d evaluations (%d checkpoints x %d benchmarks)",
            len(results),
            total_ckpts,
            total_benchmarks,
        )
        return results

    def shutdown(self):
        """Clean up Ray resources."""
        if hasattr(self, "runtime"):
            self.runtime.shutdown()
        logger.info("EvalRunner shutdown complete")


class VllmEvalRunner:
    """Lightweight evaluation runner using direct vLLM (no Ray/verl).

    Reuses the same checkpoint download, skip-existing, remote upload,
    and WandB logging as ``EvalRunner`` but replaces the inference backend
    with ``VllmEvalRuntime`` (DP across GPUs via ``mp.Process``).
    """

    def __init__(self, config):
        from verl_inf_evolve.sol_eval.vllm_runtime import VllmEvalRuntime

        self._config = config
        self.model_path = config.solver.model.path
        self.tp_size = config.eval.tp_size
        self.n_samples = config.eval.n_samples
        self.temperature = _solver_temperature(config)
        self.top_p = _solver_top_p(config)
        self.top_k = _solver_top_k(config)
        self.gpu_memory_utilization = config.eval.gpu_memory_utilization
        self.result_detail = config.eval.get("result_detail", "metrics_only")

        custom_chat_template = config.solver.model.get("custom_chat_template", None)
        n_gpus = config.trainer.n_gpus_per_node

        # Prefer eval.seed (if explicitly set), otherwise fall back to
        # solver.rollout.seed so sol_eval inherits the training-time
        # reproducibility fixture by default.
        _eval_seed = config.eval.get("seed", None)
        if _eval_seed is None:
            _eval_seed = config.solver.rollout.get("seed", None)
        self.runtime = VllmEvalRuntime.create(
            model_path=str(self.model_path),
            n_gpus=int(n_gpus),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            top_k=int(self.top_k),
            max_model_len=int(_solver_effective_max_model_len(config)),
            gpu_memory_utilization=float(self.gpu_memory_utilization),
            enforce_eager=bool(config.solver.rollout.get("enforce_eager", False)),
            custom_chat_template=str(custom_chat_template) if custom_chat_template else None,
            seed=None if _eval_seed is None else int(_eval_seed),
        )
        self.tokenizer = self.runtime.tokenizer

        logger.info(
            "VllmEvalRunner initialized: model=%s, n_gpus=%d, tp_size=%d",
            self.model_path, self.runtime.n_gpus, self.tp_size,
        )

    def evaluate_run(self) -> list[dict]:
        """Evaluate multiple checkpoints on multiple benchmarks via vLLM.

        Same orchestration logic as ``EvalRunner.evaluate_run()`` — checkpoint
        download, skip-existing, WandB logging, remote upload — but uses
        ``evaluate_benchmark_vllm`` for inference.
        """
        from verl_inf_evolve.sol_eval.eval_core import load_benchmark_questions
        from verl_inf_evolve.sol_eval.r2_ops import (
            check_r2_result_exists,
            cleanup_checkpoint,
            download_checkpoint_from_r2,
            download_r2_result,
            upload_result_to_r2,
        )
        from verl_inf_evolve.sol_eval.result_format import (
            build_output_filename,
            format_result_json,
        )
        from verl_inf_evolve.sol_eval.vllm_runtime import evaluate_benchmark_vllm

        eval_cfg = self._config.eval
        run_name = eval_cfg.run_name
        model_tag = eval_cfg.get("model_tag", None) or None
        remote_sync_path = eval_cfg.remote_sync_path
        checkpoints = list(eval_cfg.checkpoints)
        benchmarks = list(eval_cfg.benchmarks)
        temperature = _solver_temperature(self._config)
        effective_max_model_len = _solver_effective_max_model_len(self._config)
        code_execution_enabled = bool(
            eval_cfg.get("code_execution", {}).get("enabled", False)
        )
        result_detail = eval_cfg.get("result_detail", "metrics_only")
        remote_eval_base = eval_cfg.get("remote_eval_base", None) or eval_cfg.r2_eval_base
        force = eval_cfg.force
        no_r2_upload = eval_cfg.no_r2_upload
        do_cleanup = eval_cfg.cleanup_checkpoints
        no_wandb = eval_cfg.no_wandb
        remote_cfg = self._config.get("remote", {})
        wandb_entity = self._config.wandb.entity
        wandb_project = self._config.wandb.project_name
        wandb_run_name = self._config.wandb.get("run_name", None) or run_name
        wandb_group_name = self._config.wandb.get("group_name", None)
        backfill_mode = eval_cfg.get("backfill_mode", False)
        wandb_run_id = eval_cfg.get("wandb_run_id", None)
        checkpoint_cache_dir = eval_cfg.get("checkpoint_cache_dir", None)
        max_prompt_tokens = self._config.solver.rollout.prompt_length
        max_response_tokens = self._config.solver.rollout.response_length

        # WandB logger
        wandb_logger = None
        if not no_wandb:
            from verl_inf_evolve.sol_eval.wandb_logger import EvalWandBLogger

            wandb_logger = EvalWandBLogger(
                run_name=wandb_run_name,
                config_dict={
                    "checkpoints": checkpoints,
                    "benchmarks": benchmarks,
                    "temperature": temperature,
                    "max_model_len": effective_max_model_len,
                    "n_samples": {b: _resolve_n_samples(eval_cfg, b) for b in benchmarks},
                    "result_detail": result_detail,
                    "remote_sync_path": remote_sync_path,
                    "backend": "vllm",
                },
                benchmarks=benchmarks,
                entity=wandb_entity,
                project=wandb_project,
                group=wandb_group_name,
                backfill_mode=backfill_mode,
                wandb_run_id=str(wandb_run_id) if wandb_run_id else None,
            )
            if not wandb_logger.start():
                wandb_logger = None

        results = []
        total_ckpts = len(checkpoints)

        for ckpt_idx, ckpt_num in enumerate(checkpoints, 1):
            # Pre-check skip-existing for all benchmarks
            benchmarks_to_eval = []
            for benchmark in benchmarks:
                output_filename = build_output_filename(
                    benchmark=benchmark,
                    run_name=run_name,
                    ckpt_num=ckpt_num,
                    temperature=temperature,
                    max_model_len=effective_max_model_len,
                    result_detail=result_detail,
                    model_tag=model_tag,
                )
                if not force and check_r2_result_exists(
                    remote_eval_base, output_filename, remote_cfg=remote_cfg,
                ):
                    print(
                        f"Skipping {benchmark} checkpoint {ckpt_num} "
                        f"(valid result exists remotely)"
                    )
                    existing = download_r2_result(
                        remote_eval_base, output_filename, remote_cfg=remote_cfg,
                    )
                    if existing is not None:
                        existing["_eval_metadata"] = {
                            "benchmark": benchmark, "ckpt_num": ckpt_num,
                        }
                        if wandb_logger is not None:
                            wandb_logger.log_evaluation(
                                benchmark, ckpt_num, existing.get("metrics", {}),
                                question_results=existing.get("questions"),
                                n_samples=_resolve_n_samples(eval_cfg, benchmark),
                            )
                        results.append(existing)
                else:
                    benchmarks_to_eval.append(benchmark)

            if not benchmarks_to_eval:
                logger.info(
                    "Checkpoint %d: all benchmarks already evaluated, skipping",
                    ckpt_num,
                )
                if wandb_logger is not None:
                    wandb_logger.flush_checkpoint(ckpt_num)
                continue

            # Resolve checkpoint path
            if ckpt_num == -1:
                # Base model: use initial weights
                logger.info("Checkpoint -1 (base model): using %s", self.model_path)
                self.runtime.update_model(str(self.model_path))
            else:
                if checkpoint_cache_dir:
                    local_ckpt_dir = os.path.join(
                        checkpoint_cache_dir, run_name, f"global_step_{ckpt_num}",
                    )
                    if not os.path.isdir(local_ckpt_dir):
                        logger.warning(
                            "Pre-cached checkpoint not found at %s, downloading",
                            local_ckpt_dir,
                        )
                        local_ckpt_dir = download_checkpoint_from_r2(
                            remote_sync_path=remote_sync_path,
                            ckpt_num=ckpt_num,
                            local_cache_dir=os.path.join(
                                ".cache", "eval_checkpoints", run_name,
                                f"global_step_{ckpt_num}",
                            ),
                            remote_cfg=remote_cfg,
                        )
                else:
                    local_ckpt_dir = download_checkpoint_from_r2(
                        remote_sync_path=remote_sync_path,
                        ckpt_num=ckpt_num,
                        local_cache_dir=os.path.join(
                            ".cache", "eval_checkpoints", run_name,
                            f"global_step_{ckpt_num}",
                        ),
                        remote_cfg=remote_cfg,
                    )

                # Point vLLM at the HF checkpoint
                hf_path = self._resolve_hf_checkpoint_path(local_ckpt_dir)
                self.runtime.update_model(hf_path)

            for bench_idx, benchmark in enumerate(benchmarks_to_eval, 1):
                # Brief pause between benchmarks to allow CUDA contexts from
                # previous spawned vLLM workers to fully clean up, avoiding
                # transient "engine core initialization failed" errors.
                if bench_idx > 1:
                    import time
                    time.sleep(5)

                bench_n = _resolve_n_samples(eval_cfg, benchmark)
                print(
                    f"Evaluating checkpoint {ckpt_idx}/{total_ckpts} "
                    f"on {benchmark} ({bench_idx}/{len(benchmarks_to_eval)}) "
                    f"n={bench_n}"
                )

                # Isolate torch inductor cache per benchmark to prevent
                # pickle corruption when sequential vLLM instances reuse
                # stale compiled artifacts from /tmp/torchinductor_root/.
                _inductor_safe = benchmark.replace("/", "_").replace(".", "_")
                _inductor_cache = tempfile.mkdtemp(
                    prefix=f"torchinductor_{_inductor_safe}_"
                )
                _prev_inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = _inductor_cache

                # Also isolate vLLM's own compile cache (separate from
                # TORCHINDUCTOR_CACHE_DIR). vLLM writes to
                # $VLLM_CACHE_ROOT/torch_compile_cache/<hash_key>/... and the
                # hash_key only depends on model config + vLLM version — not
                # on the benchmark or checkpoint — so successive benchmark
                # subprocesses collide on the same cache dir and hit
                # "Bytes object is corrupted, checksum does not match" when
                # the previous process's cache write wasn't fully flushed.
                _vllm_cache = tempfile.mkdtemp(
                    prefix=f"vllm_cache_{_inductor_safe}_"
                )
                _prev_vllm_cache = os.environ.get("VLLM_CACHE_ROOT")
                os.environ["VLLM_CACHE_ROOT"] = _vllm_cache

                try:
                    output_filename = build_output_filename(
                        benchmark=benchmark,
                        run_name=run_name,
                        ckpt_num=ckpt_num,
                        temperature=temperature,
                        max_model_len=effective_max_model_len,
                        result_detail=result_detail,
                        model_tag=model_tag,
                    )
                    from verl_inf_evolve.sol_eval.external_benchmarks import (
                        is_external_benchmark,
                        run_external_benchmark,
                    )

                    if is_external_benchmark(benchmark):
                        model_ref = local_ckpt_dir if ckpt_num != -1 else str(self.model_path)
                        ext_max_q = int(eval_cfg.get("max_questions", 0) or 0) or None
                        result = run_external_benchmark(
                            benchmark=benchmark,
                            model_path_or_ckpt_dir=model_ref,
                            n_samples=bench_n,
                            temperature=float(temperature),
                            top_p=float(self.top_p),
                            top_k=int(self.top_k),
                            max_generation_length=_resolve_max_out_len(
                                eval_cfg, benchmark, int(max_response_tokens)
                            ),
                            prompt_length=_solver_prompt_length(self._config),
                            tp_size=int(self.tp_size),
                            trust_remote_code=bool(
                                self._config.solver.model.get("trust_remote_code", True)
                            ),
                            result_detail=result_detail,
                            max_questions=ext_max_q,
                            custom_chat_template=self.runtime.custom_chat_template,
                            base_model_path=str(self.model_path),
                        )
                    else:
                        max_questions = int(eval_cfg.get("max_questions", 0) or 0)
                        questions, _ = load_benchmark_questions(
                            EvalRunner._resolve_benchmark_path(benchmark),
                            max_questions=max_questions,
                        )

                        bench_max_resp = _resolve_max_out_len(
                            eval_cfg, benchmark, int(max_response_tokens)
                        )
                        eval_result = evaluate_benchmark_vllm(
                            questions=questions,
                            tokenizer=self.tokenizer,
                            max_prompt_tokens=max_prompt_tokens,
                            n_samples=bench_n,
                            vllm_runtime=self.runtime,
                            max_response_tokens=bench_max_resp,
                            # Pass the raw config value through; resolution
                            # happens per-benchmark inside build_benchmark_messages
                            # via resolve_use_public_eval_prompt.
                            use_public_eval_prompt=eval_cfg.get(
                                "use_public_eval_prompt", False
                            ),
                            code_execution_enabled=code_execution_enabled,
                            execution_scope="standalone",
                            model_path=self.model_path,
                            rollout_cache_dir=os.path.join(
                                ".cache", "rollout_cache", run_name,
                                f"ckpt_{ckpt_num}",
                            ),
                            mcq_choice_shuffle_config=eval_cfg.get(
                                "mcq_choice_shuffle", None
                            ),
                            benchmark_prompts=eval_cfg.get(
                                "benchmark_prompts", None
                            ),
                            assistant_prefix=eval_cfg.get(
                                "assistant_prefix", None
                            ),
                        )
                        if eval_result is None:
                            logger.warning(
                                "All questions filtered for %s at checkpoint %d",
                                benchmark, ckpt_num,
                            )
                            continue

                        result = format_result_json(
                            eval_result.question_results,
                            eval_result.metrics,
                            result_detail=result_detail,
                        )

                    # Save locally
                    effective_run_dir = f"{model_tag}-{run_name}" if model_tag else run_name
                    output_dir = os.path.join(".cache", "eval_results", effective_run_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    local_path = os.path.join(output_dir, output_filename)
                    with open(local_path, "w") as f:
                        json.dump(result, f, indent=2)
                    logger.info("Result saved to %s", local_path)

                    # Upload remotely
                    if not no_r2_upload:
                        upload_result_to_r2(
                            local_path=local_path,
                            r2_eval_base=remote_eval_base,
                            output_filename=output_filename,
                            remote_cfg=remote_cfg,
                        )

                    # WandB
                    result["_eval_metadata"] = {
                        "benchmark": benchmark, "ckpt_num": ckpt_num,
                    }
                    if wandb_logger is not None:
                        wandb_logger.log_evaluation(
                            benchmark, ckpt_num, result.get("metrics", {}),
                            question_results=result.get("questions"),
                            n_samples=bench_n,
                        )
                    results.append(result)
                finally:
                    # Restore original inductor cache env and clean up
                    if _prev_inductor is None:
                        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
                    else:
                        os.environ["TORCHINDUCTOR_CACHE_DIR"] = _prev_inductor
                    if _prev_vllm_cache is None:
                        os.environ.pop("VLLM_CACHE_ROOT", None)
                    else:
                        os.environ["VLLM_CACHE_ROOT"] = _prev_vllm_cache
                    import shutil as _shutil
                    _shutil.rmtree(_inductor_cache, ignore_errors=True)
                    _shutil.rmtree(_vllm_cache, ignore_errors=True)

            if wandb_logger is not None:
                wandb_logger.flush_checkpoint(ckpt_num)

            # Cleanup
            if (
                do_cleanup
                and not checkpoint_cache_dir
                and ckpt_num != -1
                and "local_ckpt_dir" in dir()
            ):
                cleanup_checkpoint(local_ckpt_dir)

        if wandb_logger is not None:
            wandb_logger.log_summary(results)
            wandb_logger.finish()

        logger.info(
            "evaluate_run complete: %d evaluations (%d checkpoints x %d benchmarks)",
            len(results), total_ckpts, len(benchmarks),
        )
        return results

    @staticmethod
    def _resolve_hf_checkpoint_path(checkpoint_dir: str) -> str:
        """Find the HF model path inside a downloaded checkpoint directory."""
        solver_dir = os.path.join(checkpoint_dir, "solver")
        hf_subdir = os.path.join(solver_dir, "huggingface")
        # Prefer huggingface/ subdir
        if os.path.isfile(os.path.join(hf_subdir, "config.json")):
            return hf_subdir
        if os.path.isfile(os.path.join(solver_dir, "config.json")):
            return solver_dir
        raise ValueError(
            f"No HuggingFace checkpoint found at {checkpoint_dir}. "
            f"The vLLM backend requires HF-format checkpoints "
            f"(config.json + safetensors). FSDP-only checkpoints are not "
            f"supported — use the Ray backend or convert to HF first."
        )

    def shutdown(self):
        """No-op — no persistent resources."""
        self.runtime.shutdown()
        logger.info("VllmEvalRunner shutdown complete")
