"""HMMT prompt helpers aligned to the public MathArena competition configs.

When ``eval.use_public_eval_prompt`` is enabled for ``data_source=hmmt``,
we switch from the repo-generic free-form ICL prompt to the benchmark-owned
plain-text prompt used by the public MathArena HMMT competition configs:

``Put your final answer within \boxed{}.\n\n{problem}``
"""

from __future__ import annotations

from typing import Any


HMMT_INSTRUCTION = r"Put your final answer within \boxed{}."


def build_hmmt_icl_messages(
    question: dict[str, Any],
) -> tuple[list[dict[str, str]], str]:
    """Build the public-config-style HMMT prompt."""

    prompt = (
        HMMT_INSTRUCTION
        + "\n\n"
        + str(question.get("question_text", "")).strip()
    )
    return [{"role": "user", "content": prompt}], str(question.get("ground_truth", ""))
