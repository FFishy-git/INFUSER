"""MATH-500 verifier — symbolic math comparison with _math_judger fallback."""

from __future__ import annotations

from typing import Optional

from verl_inf_evolve.utils.benchmarks.math500_prompting import extract_math500_answer
from verl_inf_evolve.utils.benchmarks.verifiers import register
from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import verify_math_answer


@register("math500", needs_full_response=True)
def verify(
    predicted: Optional[str],
    ground_truth: str,
    metadata: Optional[dict] = None,
) -> Optional[bool]:
    response_text = predicted
    extracted = extract_math500_answer(response_text)
    return verify_math_answer(extracted, ground_truth)


@register("math", needs_full_response=True)
def verify_math(
    predicted: Optional[str],
    ground_truth: str,
    metadata: Optional[dict] = None,
) -> Optional[bool]:
    return verify(predicted, ground_truth, metadata)
