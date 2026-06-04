"""OlympiadBench prompt helpers aligned to the public benchmark evaluator.

The official OlympiadBench inference stack does not use generic free-form ICL.
For English text-only open-ended items, it builds a benchmark-specific prompt
conditioned on subject, answer type, multiplicity, and unit requirements.

Our preprocessed benchmark rows are an open-ended subset and do not retain the
full official schema (for example language and theorem-proof flags), so this
helper mirrors the English open-ended branch using the fields we preserve.
"""

from __future__ import annotations

from typing import Any


_ANSWER_TYPE_LABELS = {
    "Numerical": "a numerical value",
    "Expression": "an expression",
    "Equation": "an equation",
    "Interval": "an interval",
    "Tuple": "a tuple",
}


def _subject_name(question: dict[str, Any]) -> str:
    domain = str(question.get("domain", "") or "").strip()
    if domain.endswith("_Physics"):
        return "Physics"
    return "Math"


def _split_answer_types(answer_type: str) -> list[str]:
    parts = [part.strip() for part in str(answer_type or "").split(",")]
    normalized = [part for part in parts if part]
    return normalized or ["Expression"]


def _answer_type_phrase(answer_types: list[str]) -> str:
    labels = [_ANSWER_TYPE_LABELS.get(item, item.lower()) for item in answer_types]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _looks_like_multiple_answers(question: dict[str, Any]) -> bool:
    if "is_multiple_answer" in question:
        value = question.get("is_multiple_answer")
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() == "true"
    ground_truth = str(question.get("ground_truth", "") or "")
    return "; " in ground_truth


def build_olympiadbench_icl_messages(
    question: dict[str, Any],
) -> tuple[list[dict[str, str]], str]:
    """Build the public-eval-style OlympiadBench prompt for open-ended items."""

    question_text = str(question.get("question_text", "") or "").strip()
    answer_types = _split_answer_types(str(question.get("answer_type", "") or ""))
    answer_type_phrase = _answer_type_phrase(answer_types)
    subject = _subject_name(question)
    unit = str(question.get("unit", "") or "").strip()
    is_multiple = _looks_like_multiple_answers(question)

    lines = [
        f"The following is an open-ended problem from an International {subject} competition.",
    ]
    if is_multiple:
        lines.append(
            f"The problem may have multiple answers, and each answer should be {answer_type_phrase}."
        )
    else:
        lines.append(f"The answer should be {answer_type_phrase}.")
    lines.extend(
        [
            "Please calculate the answer according to the given requirements and the information provided.",
            "Please use LaTeX format to represent the variables and formulas used in your solution.",
        ]
    )
    if is_multiple:
        lines.append(
            "End your solution with \"So the final answer is \\boxed{answer1}, \\boxed{answer2}.\""
        )
    else:
        lines.append(
            "End your solution with \"So the final answer is \\boxed{answer}.\""
        )
    if unit:
        lines.append(
            "If the answer has a unit, write the unit outside the boxed answer."
        )
    lines.extend(["", question_text])

    prompt = "\n".join(lines)
    return [{"role": "user", "content": prompt}], str(question.get("ground_truth", ""))
