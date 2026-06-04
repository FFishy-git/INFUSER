"""Putnam verifier — math_verify with the same fallback as hmmt/aime/etc."""

from __future__ import annotations

from typing import Optional

from verl_inf_evolve.utils.benchmarks.verifiers import register
from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import verify_math_answer


@register("putnam")
def verify(
    predicted: Optional[str],
    ground_truth: str,
    metadata: Optional[dict] = None,
) -> Optional[bool]:
    return verify_math_answer(predicted, ground_truth)
