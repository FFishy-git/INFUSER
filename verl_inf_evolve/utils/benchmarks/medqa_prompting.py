"""MedQA prompt helpers aligned to public benchmark evaluation formatting.

The benchmark-aligned prompt shape intentionally mirrors the public
``lm-evaluation-harness`` MedQA task:
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/medqa/preprocess_medqa.py

"""

from __future__ import annotations

import re
import string
from typing import Any

from verl_inf_evolve.utils.mcq_utils import extract_boxed_answer, format_mcq_question

MEDQA_CHOICES = list(string.ascii_uppercase)


def _normalize_choices(question: dict[str, Any]) -> list[str]:
    choices = question.get("choices", [])
    if isinstance(choices, dict):
        normalized: list[str] = []
        for letter in MEDQA_CHOICES:
            if letter not in choices:
                break
            normalized.append(str(choices[letter]))
        return normalized
    return [str(choice) for choice in list(choices)]


def build_medqa_icl_messages(question: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    """Build the short public-style MedQA prompt used for benchmark-aligned mode.

    Although this repo routes the switch through ``use_public_eval_prompt``, the
    public MedQA format used here is effectively zero-shot:

    ``Question: ...``
    ``A. ...``
    ``B. ...``
    ``...``
    ``Answer:``
    """

    # Reuse the existing MCQ normalization/validation path so MedQA scoring
    # still compares against the canonical answer letter.
    _, normalized_gt = format_mcq_question(question)
    question_text = str(question.get("question_text", "")).strip()
    choices = _normalize_choices(question)

    prompt_lines = [f"Question: {question_text}"]
    for idx, choice in enumerate(choices):
        prompt_lines.append(f"{MEDQA_CHOICES[idx]}. {choice}")
    prompt_lines.append("Answer:")

    return [{"role": "user", "content": "\n".join(prompt_lines)}], normalized_gt

def extract_medqa_choice(response_text: str | None) -> str | None:
    """Extract an A-D style answer from public-format MedQA model output.

    This supports both the new public-eval prompt path and the existing boxed
    answer habit used by the training-aligned prompt.
    """

    if not response_text:
        return None

    boxed = extract_boxed_answer(response_text)
    if boxed is not None:
        return boxed

    text = str(response_text)
    patterns = [
        r"answer\s*(?:is|:)\s*\(?([A-D])\)?\b",
        r"final\s+answer\s*(?:is|:)\s*\(?([A-D])\)?\b",
        r"option\s*\(?([A-D])\)?\b",
        r"choice\s*\(?([A-D])\)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    matches = re.findall(r"\b([A-D])\b", text.upper())
    if matches:
        return matches[-1]

    return None
