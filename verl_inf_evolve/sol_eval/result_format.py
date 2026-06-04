"""
Result formatting utilities for the evaluation pipeline.

Produces output JSON matching the run_eval_direct.py format so that existing
analysis notebooks can parse results from both pipelines.

All functions are pure Python (no GPU, no Ray, no network).
"""

import os
import statistics
from math import comb
from typing import Optional


def _normalize_benchmark_name(benchmark: str) -> str:
    """Extract benchmark stem name from a path or name.

    Examples:
        'supergpqa_2000' -> 'supergpqa_2000'
        '.cache/data/preprocessed/benchmarks/self_evolution_train_eval.json'
            -> 'self_evolution_train_eval'
    """
    # Strip .json extension if present, then take the basename
    base = os.path.basename(benchmark)
    if base.endswith(".json"):
        base = base[:-5]
    return base


def build_output_filename(
    benchmark: str,
    run_name: str,
    ckpt_num: int,
    temperature: float,
    max_model_len: int,
    result_detail: str = "metrics_only",
    model_tag: Optional[str] = None,
) -> str:
    """Build the output filename for an evaluation result.

    When *model_tag* is set the effective run identifier becomes
    ``{model_tag}-{run_name}`` so that results from different base models
    (e.g. ``ol37is`` vs ``qw4bb``) with the same training config name
    do not collide in the flat remote-storage namespace.

    Patterns (with model_tag="ol37is"):
      metrics_only: {benchmark}_ol37is-{run_name}_ans{ckpt}_temp{T}_len{L}.json
      scores:       … _scores.json
      full:         … _full.json
    """
    bench_name = _normalize_benchmark_name(benchmark)
    effective_run = f"{model_tag}-{run_name}" if model_tag else run_name
    filename = (
        f"{bench_name}_{effective_run}_ans{ckpt_num}_"
        f"temp{temperature}_len{max_model_len}.json"
    )
    if result_detail == "metrics_only":
        return filename
    if result_detail == "scores":
        return filename.replace(".json", "_scores.json")
    if result_detail == "full":
        return filename.replace(".json", "_full.json")
    raise ValueError(
        f"Unsupported result_detail={result_detail!r}. "
        f"Expected 'metrics_only', 'scores', or 'full'."
    )


def compute_pass_at_k(n_total: int, n_correct: int, k: int) -> float:
    """Compute pass@k for a single question.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n_total: Total number of samples.
        n_correct: Number of correct samples.
        k: Number of samples to draw.

    Returns:
        pass@k score in [0.0, 1.0]. Returns 0.0 if n_total < k.
    """
    if n_total < k:
        return 0.0
    if n_correct == 0:
        return 0.0
    num_incorrect = n_total - n_correct
    if num_incorrect < k:
        return 1.0
    return 1.0 - comb(num_incorrect, k) / comb(n_total, k)


def _compute_aggregate_pass_at_k(
    question_results: list[dict],
    max_k: int,
    strict: bool,
) -> dict[str, float]:
    """Compute aggregate pass@k for powers of 2 up to max_k.

    Args:
        question_results: List of question dicts with 'answer_scores' field.
        max_k: Maximum k value (n_samples). Computes for k in [1, 2, 4, 8, ...] up to max_k.
        strict: If True, treat None as incorrect. If False, exclude None from computation.

    Returns:
        Dict mapping str(k) to average pass@k across questions.
    """
    # Determine k values: powers of 2 up to max_k
    k_values = []
    k = 1
    while k <= max_k:
        k_values.append(k)
        k *= 2

    results = {}
    for k_val in k_values:
        scores = []
        for q in question_results:
            answer_scores = q.get("answer_scores", [])
            if strict:
                n = len(answer_scores)
                c = sum(1 for s in answer_scores if s is not None and s > 0)
            else:
                valid = [s for s in answer_scores if s is not None]
                n = len(valid)
                c = sum(1 for s in valid if s > 0)
            score = compute_pass_at_k(n, c, k_val)
            scores.append(score)

        results[str(k_val)] = sum(scores) / len(scores) if scores else 0.0

    return results


def _compute_length_stats(lengths: list[int]) -> dict:
    """Compute mean, std, min, max for a list of lengths."""
    if not lengths:
        return {}
    return {
        "mean": statistics.mean(lengths),
        "std": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "min": min(lengths),
        "max": max(lengths),
    }


def _compute_code_metrics(question_results: list[dict]) -> dict:
    """Aggregate execution-aware metrics when sample_exec_results are present."""
    exec_results = [
        exec_result
        for q in question_results
        for exec_result in q.get("sample_exec_results", [])
        if isinstance(exec_result, dict)
    ]
    if not exec_results:
        return {}

    total_exec_samples = len(exec_results)
    status_counts: dict[str, int] = {}
    compile_ok_count = 0
    runtime_ok_count = 0
    passed_count = 0

    for result in exec_results:
        status = str(result.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if result.get("candidate_compile_ok") is True:
            compile_ok_count += 1
        if result.get("candidate_runtime_ok") is True:
            runtime_ok_count += 1
        if result.get("passed") is True:
            passed_count += 1

    return {
        "total_exec_samples": total_exec_samples,
        "passed_exec_samples": passed_count,
        "compile_ok_samples": compile_ok_count,
        "runtime_ok_samples": runtime_ok_count,
        "compile_ok_rate": compile_ok_count / total_exec_samples if total_exec_samples else 0.0,
        "runtime_ok_rate": runtime_ok_count / total_exec_samples if total_exec_samples else 0.0,
        "runtime_pass_rate": passed_count / total_exec_samples if total_exec_samples else 0.0,
        "status_counts": status_counts,
    }


def compute_eval_metrics(
    question_results: list[dict],
    n_samples: Optional[int] = None,
) -> dict:
    """Compute all evaluation metrics from question results.

    Args:
        question_results: List of question dicts, each with:
            - answer_scores: list of float|None (1.0 correct, 0.0 incorrect, None failed)
            - sampled_answers: list of str (for char length stats)
            - response_token_lengths: optional list of int (for token length stats)
        n_samples: Number of samples per question. Auto-detected if not provided.

    Returns:
        Metrics dict matching the run_eval_direct.py output schema.
    """
    if not question_results:
        return {
            "total_questions": 0,
            "total_answers": 0,
            "valid_answers": 0,
            "failed_extractions": 0,
            "correct_answers": 0,
            "accuracy_lenient": 0.0,
            "accuracy_strict": 0.0,
            "pass_at_k_lenient": {},
            "pass_at_k_strict": {},
            "correct_count_distribution_strict": {},
            "none_count_distribution": {},
            "response_length_chars": {},
            "response_length_tokens": {},
            "primary_metric_name": "accuracy_strict",
            "primary_metric_value": 0.0,
        }

    # Auto-detect n_samples from data
    if n_samples is None:
        n_samples = max(len(q.get("answer_scores", [])) for q in question_results)

    total_questions = len(question_results)
    total_answers = sum(len(q.get("answer_scores", [])) for q in question_results)
    valid_answers = sum(
        sum(1 for s in q.get("answer_scores", []) if s is not None)
        for q in question_results
    )
    correct_answers = sum(
        sum(1 for s in q.get("answer_scores", []) if s == 1.0)
        for q in question_results
    )

    # Accuracy
    accuracy_lenient = correct_answers / valid_answers if valid_answers > 0 else 0.0
    accuracy_strict = correct_answers / total_answers if total_answers > 0 else 0.0

    # Pass@k for powers of 2
    pass_at_k_lenient = _compute_aggregate_pass_at_k(
        question_results, max_k=n_samples, strict=False
    )
    pass_at_k_strict = _compute_aggregate_pass_at_k(
        question_results, max_k=n_samples, strict=True
    )

    # Correct count distribution (strict: None = incorrect)
    correct_counts: dict[int, int] = {}
    none_counts: dict[int, int] = {}
    for q in question_results:
        scores = q.get("answer_scores", [])
        num_correct = sum(1 for s in scores if s == 1.0)
        correct_counts[num_correct] = correct_counts.get(num_correct, 0) + 1
        num_none = sum(1 for s in scores if s is None)
        none_counts[num_none] = none_counts.get(num_none, 0) + 1

    correct_count_distribution_strict = {
        k: correct_counts.get(k, 0) / total_questions
        for k in range(n_samples + 1)
    }
    none_count_distribution = {
        k: none_counts.get(k, 0) / total_questions
        for k in range(n_samples + 1)
    }

    # Response length stats (characters)
    char_lengths = []
    for q in question_results:
        for answer in q.get("sampled_answers", []):
            text = answer if isinstance(answer, str) else str(answer) if answer else ""
            char_lengths.append(len(text))

    response_length_chars = _compute_length_stats(char_lengths)

    # Response length stats (tokens) - from pre-computed field
    token_lengths = []
    for q in question_results:
        for tl in q.get("response_token_lengths", []):
            if isinstance(tl, (int, float)):
                token_lengths.append(int(tl))

    response_length_tokens = _compute_length_stats(token_lengths)

    metrics = {
        "total_questions": total_questions,
        "total_answers": total_answers,
        "valid_answers": valid_answers,
        "failed_extractions": total_answers - valid_answers,
        "correct_answers": correct_answers,
        "accuracy_lenient": accuracy_lenient,
        "accuracy_strict": accuracy_strict,
        "pass_at_k_lenient": pass_at_k_lenient,
        "pass_at_k_strict": pass_at_k_strict,
        "correct_count_distribution_strict": correct_count_distribution_strict,
        "none_count_distribution": none_count_distribution,
        "response_length_chars": response_length_chars,
        "response_length_tokens": response_length_tokens,
        "primary_metric_name": "accuracy_strict",
        "primary_metric_value": accuracy_strict,
    }

    has_sample_scores = any("sample_scores" in q for q in question_results)
    if has_sample_scores:
        score_name = next(
            (q.get("sample_score_name") for q in question_results if q.get("sample_score_name")),
            None,
        )
        score_scale_max = next(
            (
                q.get("sample_score_scale_max")
                for q in question_results
                if q.get("sample_score_scale_max") is not None
            ),
            None,
        )
        total_scored_answers = sum(len(q.get("sample_scores", [])) for q in question_results)
        valid_sample_scores = [
            score
            for q in question_results
            for score in q.get("sample_scores", [])
            if score is not None
        ]
        strict_score_sum = sum(
            0.0 if score is None else float(score)
            for q in question_results
            for score in q.get("sample_scores", [])
        )
        score_mean_strict = (
            strict_score_sum / total_scored_answers if total_scored_answers > 0 else 0.0
        )
        score_mean_lenient = (
            sum(float(score) for score in valid_sample_scores) / len(valid_sample_scores)
            if valid_sample_scores
            else 0.0
        )

        metrics["score_mean_strict"] = score_mean_strict
        metrics["score_mean_lenient"] = score_mean_lenient
        if score_name:
            metrics["score_name"] = score_name
            metrics[f"{score_name}_score_strict"] = score_mean_strict
            metrics[f"{score_name}_score_lenient"] = score_mean_lenient
            metrics["primary_metric_name"] = f"{score_name}_score_strict"
            metrics["primary_metric_value"] = score_mean_strict
        if score_scale_max is not None:
            metrics["score_scale_max"] = float(score_scale_max)

    code_metrics = _compute_code_metrics(question_results)
    if code_metrics:
        metrics["code_metrics"] = code_metrics
        metrics["compile_ok_rate"] = code_metrics["compile_ok_rate"]
        metrics["runtime_ok_rate"] = code_metrics["runtime_ok_rate"]
        metrics["runtime_pass_rate"] = code_metrics["runtime_pass_rate"]

    return metrics


def format_result_json(
    question_results: list[dict],
    metrics: dict,
    result_detail: str = "metrics_only",
) -> dict:
    """Format the full result JSON with 'questions' and 'metrics' keys.

    Args:
        question_results: List of question result dicts.
        metrics: Metrics dict from compute_eval_metrics().
        result_detail: One of ``metrics_only``, ``scores``, or ``full``.
            - ``metrics_only``: aggregate metrics only, empty questions list.
            - ``scores``: per-question scoring data (answer_scores,
              extracted_answers, response_token_lengths, etc.) without
              full response trajectories (sampled_answers).
            - ``full``: complete per-question data including trajectories.

    Returns:
        Dict with 'questions' and 'metrics' top-level keys.
    """
    _SCORES_DROP_KEYS = {"sampled_answers"}

    if result_detail == "metrics_only":
        questions = []
    elif result_detail == "scores":
        questions = [
            {k: v for k, v in q.items() if k not in _SCORES_DROP_KEYS}
            for q in question_results
        ]
    elif result_detail == "full":
        questions = question_results
    else:
        raise ValueError(
            f"Unsupported result_detail={result_detail!r}. "
            f"Expected 'metrics_only', 'scores', or 'full'."
        )

    return {
        "questions": questions,
        "metrics": metrics,
    }


def is_result_complete(result_json: dict) -> bool:
    """Check if a result JSON represents a complete evaluation.

    Args:
        result_json: Result dict to validate.

    Returns:
        True if result has valid metrics with non-zero totals.
    """
    metrics = result_json.get("metrics")
    if not isinstance(metrics, dict):
        return False
    if metrics.get("total_questions", 0) <= 0:
        return False
    if metrics.get("total_answers", 0) <= 0:
        return False
    return True
