"""OlympiadBench verifier — handles LaTeX math, expressions, equations, intervals.

Uses the shared math_verify + _math_judger union strategy, with additional
preprocessing to strip units and pass answer_type metadata.
"""

from __future__ import annotations

from typing import Optional

from verl_inf_evolve.utils.benchmarks.verifiers import register
from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import verify_math_answer


@register("olympiadbench")
def verify(
    predicted: Optional[str],
    ground_truth: str,
    metadata: Optional[dict] = None,
) -> Optional[bool]:
    """Verify an OlympiadBench answer.

    ``metadata`` may contain:
    - ``answer_type``: one of "Numerical", "Expression", "Tuple",
      "Interval", "Equation"
    - ``unit``: unit string (stripped before comparison)
    """
    if predicted is None:
        return None

    answer_type = ""
    unit = ""
    if metadata:
        answer_type = metadata.get("answer_type", "")
        unit = metadata.get("unit", "")

    # Strip unit from predicted answer if present
    cleaned_predicted = predicted.strip()
    if unit and cleaned_predicted.endswith(unit):
        cleaned_predicted = cleaned_predicted[: -len(unit)].strip()

    return verify_math_answer(cleaned_predicted, ground_truth, answer_type)
