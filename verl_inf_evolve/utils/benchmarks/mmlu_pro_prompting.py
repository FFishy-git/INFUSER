"""MMLU-Pro prompt helpers aligned to lm-evaluation-harness/reference eval.

The prompt construction here is intentionally kept close to:
https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu_pro
which itself documents alignment to the benchmark reference implementation:
https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

MMLU_PRO_CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
MMLU_PRO_HEADER = (
    'The following are multiple choice questions (with answers) about {$}. '
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice.\n\n\n\n"
)


def _normalize_options(options: list[str]) -> list[str]:
    return [str(opt).strip() for opt in options if str(opt) != "N/A"]


def format_mmlu_pro_cot_example(example: dict[str, Any], *, including_answer: bool) -> str:
    """Format one MMLU-Pro example using the public harness/reference style."""

    prompt = "Question:\n"
    prompt += str(example["question"]).strip() + "\n"
    prompt += "Options:\n"

    for i, opt in enumerate(_normalize_options(list(example["options"]))):
        if i >= len(MMLU_PRO_CHOICES):
            break
        prompt += f"{MMLU_PRO_CHOICES[i]}. {opt}\n"

    if including_answer:
        cot_content = str(example["cot_content"]).replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step.",
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."

    return prompt


@lru_cache(maxsize=1)
def _load_mmlu_pro_validation_pool() -> dict[str, list[dict[str, Any]]]:
    """Load validation few-shot docs grouped by category."""

    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ds:
        category = str(row.get("category", "")).strip()
        if not category:
            continue
        by_category[category].append(
            {
                "question": str(row.get("question", "")),
                "options": _normalize_options(list(row.get("options", []) or [])),
                "cot_content": str(row.get("cot_content", "")),
                "category": category,
            }
        )
    return dict(by_category)


def _normalized_subject(question: dict[str, Any]) -> str:
    return str(question.get("domain") or question.get("category") or "all").strip()


def build_mmlu_pro_icl_messages(
    question: dict[str, Any],
    *,
    num_fewshot: int = 5,
) -> tuple[list[dict[str, str]], str]:
    """Build a chat message matching the official MMLU-Pro CoT ICL prompt."""

    from verl_inf_evolve.utils.mcq_utils import format_mcq_question

    _, normalized_gt = format_mcq_question(question)

    prompt = MMLU_PRO_HEADER.replace("{$}", _normalized_subject(question))
    fewshot_pool = _load_mmlu_pro_validation_pool()
    subject = _normalized_subject(question)
    for example in fewshot_pool.get(subject, [])[:num_fewshot]:
        prompt += format_mmlu_pro_cot_example(example, including_answer=True)

    current_example = {
        "question": str(question.get("question_text", "")),
        "options": list(question.get("choices", []) or []),
        "cot_content": "",
    }
    prompt += format_mmlu_pro_cot_example(current_example, including_answer=False)
    return [{"role": "user", "content": prompt}], normalized_gt


def extract_mmlu_pro_choice(response_text: str | None) -> str | None:
    """Extract MMLU-Pro answer letter using public reference/harness heuristics."""

    if not response_text:
        return None

    text = str(response_text)
    match = re.search(r"answer is\s+\(?([A-J])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1).upper()

    matches = re.findall(r"\b([A-J])\b", text, re.DOTALL)
    if matches:
        return matches[-1].upper()

    return None
