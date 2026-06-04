"""Pure utility functions for benchmark evaluation scoring and question filtering.

All functions in this module are pure — no GPU, Ray, or network dependencies.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

# Default target difficulty distribution for 8-sample rollouts.
# Keys are accuracy values (n_correct / 8), values are proportions.
DEFAULT_TARGET_DISTRIBUTION: Dict[float, float] = {
    0.0: 0.10,     # 0/8
    0.125: 0.10,   # 1/8
    0.25: 0.15,    # 2/8
    0.375: 0.15,   # 3/8
    0.5: 0.10,     # 4/8
    0.625: 0.10,   # 5/8
    0.75: 0.10,    # 6/8
    0.875: 0.10,   # 7/8
    1.0: 0.10,     # 8/8
}


def group_and_compute_accuracy(
    results_list: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-response scores into per-question accuracy.

    Args:
        results_list: List of dicts, each with at least ``"question_id"`` and
            ``"score"`` (float 0.0/1.0 or None). Mirrors the shape produced
            by iterating over DataProto non_tensor_batch after
            ``extract_answer_scores()``.

    Returns:
        Dict mapping question_id to ``{"n_correct": int, "n_total": int,
        "accuracy": float}``.
    """
    grouped: Dict[str, List[Optional[float]]] = {}
    for item in results_list:
        qid = str(item["question_id"])
        score = item.get("score")
        grouped.setdefault(qid, []).append(score)

    result: Dict[str, Dict[str, Any]] = {}
    for qid, scores in grouped.items():
        valid = [s for s in scores if s is not None]
        n_correct = int(sum(valid))
        n_total = len(valid)
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        result[qid] = {
            "n_correct": n_correct,
            "n_total": n_total,
            "accuracy": accuracy,
        }
    return result


def redistribute_shortfall(
    buckets: Dict[float, List[str]],
    shortfall: int,
    over_represented: List[Tuple[float, int]],
) -> Dict[float, int]:
    """Redistribute missing count proportionally among over-represented buckets.

    When some difficulty buckets have fewer questions than the target count,
    the surplus is redistributed proportionally among buckets that have more
    questions than their original target.

    Args:
        buckets: Mapping of accuracy value to list of available question IDs
            (only the over-represented buckets need to be populated).
        shortfall: Total number of extra questions to draw.
        over_represented: List of ``(accuracy, surplus)`` where *surplus* is
            how many extra questions are available beyond the original target
            in that bucket.

    Returns:
        Dict mapping accuracy value to the **additional** number of questions
        to sample from that bucket.
    """
    if shortfall <= 0 or not over_represented:
        return {}

    total_surplus = sum(s for _, s in over_represented)
    if total_surplus <= 0:
        return {}

    extra: Dict[float, int] = {}
    remaining = shortfall

    for i, (acc, surplus) in enumerate(over_represented):
        if i == len(over_represented) - 1:
            # Last bucket gets whatever is left to avoid rounding errors
            alloc = remaining
        else:
            alloc = round(shortfall * surplus / total_surplus)
        alloc = min(alloc, surplus, remaining)
        if alloc > 0:
            extra[acc] = alloc
            remaining -= alloc

    return extra


def filter_questions_by_difficulty(
    results: Dict[str, Dict[str, Any]],
    target_distribution: Optional[Dict[float, float]] = None,
    total: int = 150,
    seed: int = 42,
) -> List[str]:
    """Filter questions to match a target difficulty distribution.

    Groups questions into accuracy buckets and samples to achieve the desired
    proportion of each difficulty level.

    Args:
        results: Output of :func:`group_and_compute_accuracy` — dict mapping
            ``question_id`` to ``{"n_correct", "n_total", "accuracy"}``.
        target_distribution: Mapping of accuracy value to target proportion
            (must sum to ~1.0). Defaults to :data:`DEFAULT_TARGET_DISTRIBUTION`.
        total: Total number of questions to select.
        seed: Random seed for reproducibility.

    Returns:
        List of selected question IDs.
    """
    if target_distribution is None:
        target_distribution = DEFAULT_TARGET_DISTRIBUTION

    rng = random.Random(seed)

    # Group questions by accuracy bucket
    buckets: Dict[float, List[str]] = {acc: [] for acc in target_distribution}
    for qid, info in results.items():
        acc = info["accuracy"]
        if acc in buckets:
            buckets[acc].append(qid)

    # Compute target counts per bucket
    target_counts: Dict[float, int] = {}
    allocated = 0
    sorted_accs = sorted(target_distribution.keys())
    for i, acc in enumerate(sorted_accs):
        if i == len(sorted_accs) - 1:
            target_counts[acc] = total - allocated
        else:
            target_counts[acc] = round(total * target_distribution[acc])
            allocated += target_counts[acc]

    # Sample from each bucket
    selected: List[str] = []
    shortfall = 0
    over_represented: List[Tuple[float, int]] = []

    for acc in sorted_accs:
        available = buckets[acc]
        target = target_counts[acc]

        if len(available) >= target:
            sampled = rng.sample(available, target)
            selected.extend(sampled)
            surplus = len(available) - target
            if surplus > 0:
                over_represented.append((acc, surplus))
        else:
            # Take all available, track shortfall
            selected.extend(available)
            shortfall += target - len(available)

    # Redistribute shortfall among over-represented buckets
    if shortfall > 0 and over_represented:
        # Rebuild remaining available per bucket (exclude already-selected)
        selected_set = set(selected)
        remaining_buckets: Dict[float, List[str]] = {}
        for acc, _ in over_represented:
            remaining_buckets[acc] = [
                qid for qid in buckets[acc] if qid not in selected_set
            ]

        over_represented_remaining = [
            (acc, len(remaining_buckets[acc]))
            for acc, _ in over_represented
            if len(remaining_buckets[acc]) > 0
        ]

        extra_alloc = redistribute_shortfall(
            remaining_buckets, shortfall, over_represented_remaining
        )
        for acc, extra_count in extra_alloc.items():
            sampled = rng.sample(remaining_buckets[acc], extra_count)
            selected.extend(sampled)

    return selected


def format_dev_json(
    filtered_question_ids: List[str],
    source_results: List[Dict[str, Any]],
    data_source: str = "supergpqa",
) -> List[Dict[str, Any]]:
    """Format filtered questions into dev.json schema.

    Args:
        filtered_question_ids: List of question IDs to include (output of
            :func:`filter_questions_by_difficulty`).
        source_results: The ``"results"`` array from the benchmark evaluation
            output JSON. Each entry must have at least ``"question_id"``,
            ``"question_text"``, ``"choices"``, ``"ground_truth"``,
            ``"accuracy"``. ``"domain"`` is optional.
        data_source: Value for the ``data_source`` field.

    Returns:
        List of dicts matching dev.json schema: ``{question_id,
        question_text, choices, domain, difficulty, ground_truth,
        data_source}``.
    """
    id_set = set(filtered_question_ids)
    lookup = {r["question_id"]: r for r in source_results}

    dev_records: List[Dict[str, Any]] = []
    for qid in filtered_question_ids:
        r = lookup.get(qid)
        if r is None:
            continue
        accuracy = r.get("accuracy", 0.0)
        difficulty = _accuracy_to_difficulty(accuracy)
        dev_records.append({
            "question_id": r["question_id"],
            "question_text": r.get("question_text", ""),
            "choices": r.get("choices", []),
            "domain": r.get("domain", ""),
            "difficulty": difficulty,
            "ground_truth": r.get("ground_truth", ""),
            "data_source": data_source,
        })
    return dev_records


def _accuracy_to_difficulty(accuracy: float) -> str:
    """Map accuracy value to a human-readable difficulty label."""
    if accuracy <= 0.25:
        return "hard"
    elif accuracy <= 0.625:
        return "medium"
    else:
        return "easy"


__all__ = [
    "DEFAULT_TARGET_DISTRIBUTION",
    "filter_questions_by_difficulty",
    "format_dev_json",
    "group_and_compute_accuracy",
    "redistribute_shortfall",
]
