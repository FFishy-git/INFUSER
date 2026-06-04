"""Benchmark-native continuous scoring registry.

This is separate from the boolean verifier registry so existing correctness
checks and trainer reward paths can stay binary while benchmark-specific
metrics (for example PHYBench EED) are layered on top for evaluation/logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from verl_inf_evolve.utils.benchmarks.phybench_utils import normalize_phybench_expression


@dataclass(frozen=True)
class ScoreResult:
    """Continuous benchmark score for one sample."""

    score: float | None
    name: str
    scale_max: float = 1.0


ScoreFn = Callable[
    [Optional[str], str, Optional[dict]],
    Optional[ScoreResult],
]

_REGISTRY: dict[str, ScoreFn] = {}
_NEEDS_FULL_RESPONSE: set[str] = set()


def register(name: str, *, needs_full_response: bool = False):
    """Register a benchmark-native scorer under ``data_source``."""

    def decorator(fn: ScoreFn) -> ScoreFn:
        if name in _REGISTRY:
            raise ValueError(f"Scorer already registered for '{name}'")
        _REGISTRY[name] = fn
        if needs_full_response:
            _NEEDS_FULL_RESPONSE.add(name)
        return fn

    return decorator


def get_scorer(data_source: str) -> Optional[ScoreFn]:
    return _REGISTRY.get(data_source)


def get_needs_full_response(data_source: str) -> bool:
    return data_source in _NEEDS_FULL_RESPONSE


@register("phybench")
def score_phybench(
    predicted: Optional[str],
    ground_truth: str,
    metadata: Optional[dict] = None,  # noqa: ARG001 - reserved for future scorer hints
) -> Optional[ScoreResult]:
    """Compute PHYBench's official EED score on the native ``0..100`` scale."""
    from verl_inf_evolve.utils.benchmarks.phybench_eed import compute_eed_score

    del metadata

    cleaned_gt = normalize_phybench_expression(ground_truth)
    cleaned_pred = normalize_phybench_expression(predicted)
    if not cleaned_gt:
        return None
    if not cleaned_pred:
        return ScoreResult(score=0.0, name="eed", scale_max=100.0)

    try:
        score = compute_eed_score(cleaned_gt, cleaned_pred)
    except Exception:
        score = 0.0

    return ScoreResult(score=float(score), name="eed", scale_max=100.0)

