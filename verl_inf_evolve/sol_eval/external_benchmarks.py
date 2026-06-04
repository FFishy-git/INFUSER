"""External benchmark adapters for paper-faithful third-party runners.

These benchmarks bypass sol_eval's native question-by-question generation and
scoring path and instead launch the shared OpenCompass runner, then normalize
its outputs back into the standard sol_eval result schema.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from verl_inf_evolve.sol_eval.opencompass_runner import (
    detect_hf_type,
    find_latest_run_dir,
    launch_opencompass,
)
from verl_inf_evolve.sol_eval.result_format import compute_eval_metrics

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTERNAL_CACHE_ROOT = _REPO_ROOT / ".cache" / "sol_eval" / "external_benchmarks"

_BENCHMARK_ALIASES = {
    "humaneval": "humaneval_plus_external",
    "humaneval_full": "humaneval_plus_external",
    "humaneval_plus": "humaneval_plus_external",
    "livecodebench": "livecodebench_v5_external",
    "livecodebench_v5": "livecodebench_v5_external",
}

_CANONICAL_OC_BENCHMARK = {
    "humaneval_plus_external": "humaneval",
    "livecodebench_v5_external": "livecodebench",
}

_RESULT_FILE_CANDIDATES = {
    "humaneval_plus_external": ("humaneval_plus.json",),
    "livecodebench_v5_external": ("lcb_code_generation.json",),
}


def canonical_external_benchmark(benchmark: str) -> str | None:
    """Return the canonical external benchmark key when applicable."""
    return _BENCHMARK_ALIASES.get(str(benchmark).strip())


def is_external_benchmark(benchmark: str) -> bool:
    """Return whether a benchmark should use the external runner path."""
    return canonical_external_benchmark(benchmark) is not None


def resolve_external_model_path(model_path_or_ckpt_dir: str) -> str:
    """Resolve a local HF-format model path usable by external runners."""
    candidate = Path(str(model_path_or_ckpt_dir)).expanduser()
    if candidate.is_dir():
        if (candidate / "config.json").is_file():
            return str(candidate)
        solver_hf = candidate / "solver" / "huggingface"
        if (solver_hf / "config.json").is_file():
            return str(solver_hf)
        solver_root = candidate / "solver"
        if (solver_root / "config.json").is_file():
            return str(solver_root)

    model_id = str(model_path_or_ckpt_dir)
    if "/" in model_id and not model_id.startswith("/"):
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id)

    raise ValueError(
        "External benchmark evaluation requires an HF-format model directory "
        f"or a downloadable HF model ID, got {model_path_or_ckpt_dir!r}"
    )


def run_external_benchmark(
    *,
    benchmark: str,
    model_path_or_ckpt_dir: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_generation_length: int,
    prompt_length: int = 4096,
    tp_size: int,
    trust_remote_code: bool = True,
    result_detail: str = "metrics_only",
    max_questions: int | None = None,
    custom_chat_template: str | None = None,
    base_model_path: str | None = None,
) -> dict[str, Any]:
    """Run an external benchmark and normalize it to the sol_eval result shape."""
    del tp_size, trust_remote_code

    canonical = canonical_external_benchmark(benchmark)
    if canonical is None:
        raise ValueError(f"Benchmark {benchmark!r} is not configured for external execution")

    # Detect hf_type from the ORIGINAL base model path (e.g.
    # "allenai/Olmo-3-7B-Instruct-SFT") rather than the checkpoint directory
    # (e.g. ".cache/eval_checkpoints/.../solver") whose generic path has no
    # chat markers.  Fallback to model_path_or_ckpt_dir for backward compat.
    hf_type = detect_hf_type(base_model_path or model_path_or_ckpt_dir)
    model_path = resolve_external_model_path(model_path_or_ckpt_dir)
    output_root = _EXTERNAL_CACHE_ROOT / canonical / _safe_model_label(model_path)
    output_root.mkdir(parents=True, exist_ok=True)
    passthrough = ["--dump-eval-details"] if result_detail != "metrics_only" else []

    # Isolate torch inductor cache per OC subprocess to prevent corruption
    # when sequential benchmarks reuse stale compiled artifacts.
    inductor_cache = tempfile.mkdtemp(prefix=f"torchinductor_{canonical}_")
    prev_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache

    try:
        exit_code = launch_opencompass(
            hf_path=model_path,
            hf_type=hf_type,
            benchmarks=_CANONICAL_OC_BENCHMARK[canonical],
            num_runs=str(int(n_samples)),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            max_out_len=str(int(max_generation_length)),
            prompt_length=int(prompt_length),
            max_questions=max_questions,
            output_dir=str(output_root),
            passthrough=passthrough,
            no_upload=True,
            custom_chat_template=custom_chat_template,
        )
    finally:
        if prev_cache is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = prev_cache
        import shutil
        shutil.rmtree(inductor_cache, ignore_errors=True)

    if exit_code != 0:
        _dump_oc_worker_logs(output_root, canonical)
        raise RuntimeError(
            f"OpenCompass exited with code {exit_code} for benchmark={benchmark!r}"
        )

    run_dir_str = find_latest_run_dir(str(output_root))
    if run_dir_str is None:
        _dump_oc_worker_logs(output_root, canonical)
        raise FileNotFoundError(f"No OpenCompass run directory found under {output_root}")
    run_dir = Path(run_dir_str)
    try:
        result_path = _find_result_json(run_dir, canonical)
    except FileNotFoundError:
        _dump_oc_worker_logs(Path(run_dir_str), canonical)
        raise
    with result_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if canonical == "humaneval_plus_external":
        return _normalize_humaneval_result(raw, result_detail=result_detail)
    if canonical == "livecodebench_v5_external":
        return _normalize_livecodebench_result(raw, result_detail=result_detail)
    raise AssertionError(f"Unhandled external benchmark: {canonical}")


def _dump_oc_worker_logs(output_root: Path, canonical: str) -> None:
    """Print OC worker .out logs to stdout for relay capture on failure."""
    import glob as globmod

    log_patterns = [
        str(output_root / "**" / "logs" / "infer" / "**" / "*.out"),
        str(output_root / "**" / "logs" / "eval" / "**" / "*.out"),
    ]
    for pattern in log_patterns:
        for log_file in sorted(globmod.glob(pattern, recursive=True)):
            print(f"\n{'='*60}", flush=True)
            print(f"OC WORKER LOG: {log_file}", flush=True)
            print(f"{'='*60}", flush=True)
            try:
                with open(log_file, "r") as f:
                    content = f.read()
                # Print last 100 lines to avoid flooding
                lines = content.strip().split("\n")
                if len(lines) > 100:
                    print(f"... ({len(lines) - 100} lines truncated) ...", flush=True)
                    lines = lines[-100:]
                print("\n".join(lines), flush=True)
            except Exception as exc:
                print(f"  (failed to read: {exc})", flush=True)


def select_livecodebench_runner_model(model_path_or_ckpt_dir: str) -> str:
    """Compatibility wrapper for existing tests and callers."""
    model_type = detect_hf_type(str(model_path_or_ckpt_dir))
    return "GenericBase" if model_type == "base" else "GenericChat"


def _safe_model_label(model_path: str) -> str:
    return str(model_path).replace("/", "__").replace(":", "_")


def _find_latest_run_dir(output_root: Path) -> Path:
    summary_files = sorted(output_root.glob("**/summary/summary_*.csv"))
    if not summary_files:
        raise FileNotFoundError(
            f"No OpenCompass summary CSV found under {output_root}"
        )
    return summary_files[-1].parents[1]


def _find_result_json(run_dir: Path, canonical: str) -> Path:
    results_dir = run_dir / "results" / "eval-model"
    for filename in _RESULT_FILE_CANDIDATES[canonical]:
        candidate = results_dir / filename
        if candidate.is_file():
            return candidate
    all_json = sorted(results_dir.glob("*.json"))
    if len(all_json) == 1:
        return all_json[0]
    raise FileNotFoundError(
        f"Expected OpenCompass result JSON for {canonical} under {results_dir}"
    )


def _normalize_pct(value: Any) -> float:
    score = float(value or 0.0)
    return score / 100.0 if score > 1.0 else score


def _extract_metric(raw: dict[str, Any], prefixes: tuple[str, ...]) -> float:
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            continue
        lowered = key.lower()
        if any(lowered.startswith(prefix) for prefix in prefixes):
            return _normalize_pct(value)
    return 0.0


def _detail_entries(details: Any) -> list[dict[str, Any]]:
    if isinstance(details, dict):
        return [item for item in details.values() if isinstance(item, dict)]
    if isinstance(details, list):
        return [item for item in details if isinstance(item, dict)]
    return []


def _first_value(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _per_sample_scores(detail: dict, key: str) -> list[float]:
    """OC dumps is_correct/correct as bool when n=1, list[bool] when n>1.

    NB: in practice the OpenCompass HumanEval+ evaluator emits ``is_correct``
    as a scalar even when ``n_samples > 1`` (it aggregates to "any sample
    passed" rather than reporting per-sample correctness), so callers should
    treat a length-1 result with ``len(predictions) > 1`` as a signal to
    re-grade per-sample via :func:`_humaneval_regrade_per_sample`.
    """
    raw = detail.get(key)
    if isinstance(raw, list):
        return [1.0 if bool(v) else 0.0 for v in raw]
    if raw is None:
        return []
    return [1.0 if bool(raw) else 0.0]


def _per_sample_list(detail: dict, key: str) -> list:
    """Wrap scalar fields into a 1-element list so n=1 vs n>1 are uniform."""
    raw = detail.get(key)
    if isinstance(raw, list):
        return list(raw)
    if raw is None:
        return []
    return [raw]


_HUMANEVAL_REGRADE_WORKERS = int(os.environ.get("SOL_EVAL_HE_REGRADE_WORKERS", "16"))


def _humaneval_init_worker() -> None:
    """Process-pool initializer: pre-load HumanEval+ problems + groundtruth."""
    global _HE_PROBLEMS, _HE_EXPECTED  # noqa: PLW0603
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.evaluate import get_groundtruth

    _HE_PROBLEMS = get_human_eval_plus()
    hashcode = get_human_eval_plus_hash()
    _HE_EXPECTED = get_groundtruth(_HE_PROBLEMS, hashcode, [])


def _humaneval_regrade_one(args: tuple[int, str, str]) -> tuple[int, bool, bool]:
    """Worker: returns (flat_idx, base_pass, plus_pass)."""
    from evalplus.evaluate import check_correctness

    flat_idx, task_id, solution = args
    if task_id not in _HE_PROBLEMS:
        return flat_idx, False, False
    try:
        res = check_correctness(
            "humaneval",
            flat_idx,
            _HE_PROBLEMS[task_id],
            solution,
            _HE_EXPECTED[task_id],
            base_only=False,
            fast_check=False,
        )
        return flat_idx, res["base"][0] == "pass", res["plus"][0] == "pass"
    except Exception:  # noqa: BLE001 — sandbox failures shouldn't take the run down
        return flat_idx, False, False


def _humaneval_regrade_per_sample(
    work: list[tuple[int, str, str]],
) -> dict[int, tuple[bool, bool]]:
    """Re-grade ``work = [(flat_idx, task_id, solution), ...]`` in parallel.

    Returns a ``{flat_idx: (base_pass, plus_pass)}`` map. Empty dict if
    ``evalplus`` is not importable.
    """
    if not work:
        return {}
    try:
        import evalplus  # noqa: F401  (importability probe)
    except ImportError:
        logger.warning(
            "evalplus not installed — skipping per-sample HumanEval+ regrade. "
            "Saved answer_scores will collapse to 1 sample/question."
        )
        return {}

    out: dict[int, tuple[bool, bool]] = {}
    workers = min(_HUMANEVAL_REGRADE_WORKERS, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_humaneval_init_worker
    ) as ex:
        futures = [ex.submit(_humaneval_regrade_one, item) for item in work]
        for fut in as_completed(futures):
            flat_idx, base_ok, plus_ok = fut.result()
            out[flat_idx] = (base_ok, plus_ok)
    return out


def _oc_reported_pass_at_k(raw: dict, prefix: str = "") -> dict[str, float]:
    """Pull pass@K headline numbers OC already wrote into raw, keyed by str(K)."""
    out: dict[str, float] = {}
    for metric, value in raw.items():
        if not isinstance(value, (int, float)):
            continue
        lowered = metric.lower()
        if prefix and not lowered.startswith(prefix):
            continue
        if "pass@" not in lowered:
            continue
        suffix = lowered.split("pass@", 1)[1]
        k_str = suffix.split()[0]            # strip " (N runs average)"
        if k_str.isdigit():
            out[k_str] = _normalize_pct(value)
    return out


def _normalize_humaneval_result(
    raw: dict[str, Any],
    *,
    result_detail: str,
) -> dict[str, Any]:
    detail_entries = _detail_entries(raw.get("details", {}))

    question_results: list[dict[str, Any]] = []
    for detail in detail_entries:
        task_id = (
            _first_value(detail.get("reference"))
            or detail.get("example_abbr")
        )
        answer_scores = _per_sample_scores(detail, "is_correct")
        predictions = [str(p) for p in _per_sample_list(detail, "prediction")]
        base_results = _per_sample_list(detail, "base_result")
        plus_results = _per_sample_list(detail, "plus_result")

        q: dict[str, Any] = {
            "question_id": task_id,
            "answer_scores": answer_scores,
            "metadata": {
                "base_results": base_results,
                "plus_results": plus_results,
            },
        }
        if predictions:
            q["sampled_answers"] = predictions
        question_results.append(q)

    # OC's HumanEval+ evaluator collapses ``is_correct`` / ``base_result`` /
    # ``plus_result`` to scalars even when n > 1, so when ``predictions`` has
    # more entries than ``answer_scores`` we re-grade each saved completion
    # locally with ``evalplus.evaluate.check_correctness`` to recover the full
    # per-sample correctness vector. Without this the saved file's
    # ``pass_at_k`` ladder collapses to ``{"1": ...}``.
    regrade_work: list[tuple[int, str, str]] = []
    regrade_locator: list[tuple[int, int]] = []  # (q_idx, s_idx) per flat_idx
    for q_idx, q in enumerate(question_results):
        n_pred = len(q.get("sampled_answers", []))
        n_score = len(q.get("answer_scores", []))
        if n_pred > 1 and n_score < n_pred and q.get("question_id"):
            for s_idx, sol in enumerate(q["sampled_answers"]):
                regrade_locator.append((q_idx, s_idx))
                regrade_work.append((len(regrade_work), q["question_id"], sol))
    if regrade_work:
        logger.info(
            "Regrading %d HumanEval+ samples across %d questions via evalplus "
            "(OC collapsed is_correct to scalar)",
            len(regrade_work),
            sum(1 for q in question_results
                if len(q.get("sampled_answers", [])) > len(q.get("answer_scores", []))),
        )
        regrade_map = _humaneval_regrade_per_sample(regrade_work)
        if regrade_map:
            # Initialise per-question buffers sized to the prediction count.
            for q in question_results:
                n_pred = len(q.get("sampled_answers", []))
                if n_pred > 1 and len(q["answer_scores"]) < n_pred:
                    q["answer_scores"] = [0.0] * n_pred
                    q["metadata"]["base_results"] = ["fail"] * n_pred
                    q["metadata"]["plus_results"] = ["fail"] * n_pred
            # Fill the per-sample slots from the parallel regrade.
            for flat_idx, (base_ok, plus_ok) in regrade_map.items():
                q_idx, s_idx = regrade_locator[flat_idx]
                question_results[q_idx]["answer_scores"][s_idx] = (
                    1.0 if (base_ok and plus_ok) else 0.0
                )
                question_results[q_idx]["metadata"]["base_results"][s_idx] = (
                    "pass" if base_ok else "fail"
                )
                question_results[q_idx]["metadata"]["plus_results"][s_idx] = (
                    "pass" if plus_ok else "fail"
                )

    n_samples = max((len(q["answer_scores"]) for q in question_results), default=1)
    metrics = compute_eval_metrics(question_results, n_samples=n_samples)

    oc_pass = _oc_reported_pass_at_k(raw, prefix="humaneval_plus_")
    if oc_pass:
        metrics["oc_reported_pass_at_k"] = oc_pass
    metrics["external_runner"] = "opencompass"
    metrics["external_runner_dataset"] = "humaneval_plus"

    if result_detail == "metrics_only":
        question_results = []
    elif result_detail == "scores":
        for q in question_results:
            q.pop("sampled_answers", None)

    return {"questions": question_results, "metrics": metrics}


def _normalize_livecodebench_result(
    raw: dict[str, Any],
    *,
    result_detail: str,
) -> dict[str, Any]:
    detail_entries = _detail_entries(raw.get("details", []))

    question_results: list[dict[str, Any]] = []
    for idx, detail in enumerate(detail_entries):
        final_metadata_list = _per_sample_list(detail, "final_metadata")
        first_meta = final_metadata_list[0] if final_metadata_list else {}
        question_id = (
            detail.get("question_id")
            or (first_meta.get("question_id") if isinstance(first_meta, dict) else None)
            or detail.get("example_abbr")
            or f"lcb_{idx}"
        )
        answer_scores = _per_sample_scores(detail, "correct")
        predictions = [str(p) for p in _per_sample_list(detail, "prediction")]
        eval_results = _per_sample_list(detail, "eval_result")

        q: dict[str, Any] = {
            "question_id": question_id,
            "answer_scores": answer_scores,
            "metadata": {
                "eval_results": eval_results,
                "final_metadata": final_metadata_list,
            },
        }
        if predictions:
            q["sampled_answers"] = predictions
        question_results.append(q)

    n_samples = max((len(q["answer_scores"]) for q in question_results), default=1)
    metrics = compute_eval_metrics(question_results, n_samples=n_samples)

    oc_pass = _oc_reported_pass_at_k(raw)
    if oc_pass:
        metrics["oc_reported_pass_at_k"] = oc_pass
    metrics["external_runner"] = "opencompass"
    metrics["external_runner_dataset"] = "livecodebench"

    if result_detail == "metrics_only":
        question_results = []
    elif result_detail == "scores":
        for q in question_results:
            q.pop("sampled_answers", None)

    return {"questions": question_results, "metrics": metrics}
