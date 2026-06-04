"""PHYBench verifier — exact symbolic equivalence as auxiliary binary metric."""

from __future__ import annotations

from typing import Optional

from verl_inf_evolve.utils.benchmarks.phybench_utils import normalize_phybench_expression
from verl_inf_evolve.utils.benchmarks.verifiers import register
from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import verify_math_answer


@register("phybench")
def verify(
    predicted: Optional[str],
    ground_truth: str,
    metadata: Optional[dict] = None,
) -> Optional[bool]:
    del metadata

    cleaned_pred = normalize_phybench_expression(predicted)
    cleaned_gt = normalize_phybench_expression(ground_truth)
    if not cleaned_pred:
        return None
    return verify_math_answer(cleaned_pred, cleaned_gt)
