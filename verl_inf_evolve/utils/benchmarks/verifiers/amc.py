"""AMC verifier — same symbolic math comparison as MATH-500.

R-Zero's ``baselines/r-zero/evaluation/datasets_loader.py`` uses a single
``DatasetHandler`` base class (ANSWER_PATTERN_BOXED + ``math_verify.verify``)
for MATH, AMC, Minerva, and Olympiad.  We mirror that by delegating to the
MATH-500 verifier here.
"""

from __future__ import annotations

from typing import Optional

from verl_inf_evolve.utils.benchmarks.math500_prompting import extract_math500_answer
from verl_inf_evolve.utils.benchmarks.verifiers import register
from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import verify_math_answer


@register("amc", needs_full_response=True)
def verify(
    predicted: Optional[str],
    ground_truth: str,
    metadata: Optional[dict] = None,
) -> Optional[bool]:
    extracted = extract_math500_answer(predicted)
    return verify_math_answer(extracted, ground_truth)
