"""Helpers for consuming similarity metrics from influence scoring."""

from __future__ import annotations

from typing import Any

from verl_inf_evolve.utils.metric_utils import add_distribution_stats


def extract_similarity_scores(
    similarity_metrics: dict[str, Any],
) -> tuple[str, list[float]]:
    """Return the canonical reward-driving score list and its mode."""
    score_mode = str(similarity_metrics.get("score_mode", "cosine"))
    raw_scores = similarity_metrics.get("score")
    if raw_scores is None:
        if score_mode == "dot" and "dot" in similarity_metrics:
            raw_scores = similarity_metrics["dot"]
        elif score_mode == "cosine" and "cosine" in similarity_metrics:
            raw_scores = similarity_metrics["cosine"]
        elif (
            score_mode == "preconditioned_dot"
            and "preconditioned_dot" in similarity_metrics
        ):
            raw_scores = similarity_metrics["preconditioned_dot"]
        elif (
            score_mode == "preconditioned_cosine"
            and "preconditioned_cosine" in similarity_metrics
        ):
            raw_scores = similarity_metrics["preconditioned_cosine"]
        else:
            raise KeyError("similarity_metrics missing required 'score' field")
    return score_mode, [float(value) for value in raw_scores]


def build_similarity_rewards(
    question_order: list[str],
    all_question_ids: set[str],
    similarity_metrics: dict[str, Any],
) -> tuple[dict[str, float], list[float], str]:
    """Map one per-question score to each surviving question id."""
    score_mode, scores = extract_similarity_scores(similarity_metrics)
    rewards: dict[str, float] = {}
    for i, qid in enumerate(question_order):
        if i < len(scores):
            rewards[qid] = scores[i]

    for qid in all_question_ids - set(question_order):
        rewards[qid] = 0.0

    return rewards, scores, score_mode


def add_similarity_metric_stats(
    metrics: dict[str, float],
    similarity_metrics: dict[str, Any],
) -> str:
    """Log similarity metric distributions under the canonical prefixes."""
    score_mode, scores = extract_similarity_scores(similarity_metrics)
    add_distribution_stats(metrics, "influence_sim/score", scores, fill_empty=False)

    if "dot" in similarity_metrics:
        add_distribution_stats(
            metrics,
            "influence_sim/dot",
            similarity_metrics["dot"],
            fill_empty=False,
        )
    if "cosine" in similarity_metrics:
        add_distribution_stats(
            metrics,
            "influence_sim/cosine",
            similarity_metrics["cosine"],
            fill_empty=False,
        )
    if "grad_norm" in similarity_metrics:
        add_distribution_stats(
            metrics,
            "influence_sim/grad_norm",
            similarity_metrics["grad_norm"],
            fill_empty=False,
        )
    if "gamma_norm" in similarity_metrics:
        add_distribution_stats(
            metrics,
            "influence_sim/gamma_norm",
            similarity_metrics["gamma_norm"],
            fill_empty=False,
        )
    if "preconditioned_dot" in similarity_metrics:
        add_distribution_stats(
            metrics,
            "influence_sim/preconditioned_dot",
            similarity_metrics["preconditioned_dot"],
            fill_empty=False,
        )
    if "preconditioned_cosine" in similarity_metrics:
        add_distribution_stats(
            metrics,
            "influence_sim/preconditioned_cosine",
            similarity_metrics["preconditioned_cosine"],
            fill_empty=False,
        )

    ref_norm = similarity_metrics.get("ref_norm")
    if ref_norm is not None:
        metrics["influence_sim/ref_norm"] = float(ref_norm)
    return score_mode
