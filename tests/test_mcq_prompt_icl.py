"""Tests for optional MCQ few-shot ICL prompt construction."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.utils.mcq_utils import build_mcq_messages
from verl_inf_evolve.utils.prompts import MCQ_ANSWER_GENERATION_FEW_SHOT_EXAMPLES


def _sample_mcq_question() -> dict:
    return {
        "question_text": "What is 2 + 3?",
        "choices": ["4", "5", "6", "7"],
        "ground_truth": "5",
    }


def test_build_mcq_messages_without_icl() -> None:
    messages, normalized_gt = build_mcq_messages(
        _sample_mcq_question(),
        use_few_shot_icl=False,
    )

    assert normalized_gt == "B"
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "demonstrated format" not in messages[0]["content"]
    assert "\\boxed{}" in messages[0]["content"]


def test_build_mcq_messages_with_icl_includes_examples_and_format_rules() -> None:
    messages, normalized_gt = build_mcq_messages(
        _sample_mcq_question(),
        use_few_shot_icl=True,
    )

    assert normalized_gt == "B"
    expected_len = 1 + 2 * len(MCQ_ANSWER_GENERATION_FEW_SHOT_EXAMPLES) + 1
    assert len(messages) == expected_len

    assert messages[0]["role"] == "system"
    assert "Think step by step" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert "Answer:" in messages[-1]["content"]
    assert "\\boxed{" not in messages[-1]["content"]

    for idx in range(1, len(messages) - 1, 2):
        assert messages[idx]["role"] == "user"
        assert messages[idx + 1]["role"] == "assistant"
        assert "Answer:" in messages[idx]["content"]
        assert "\\boxed{" in messages[idx + 1]["content"]

    # Final user prompt uses ICL_PROMPT (question + Answer: cue).
    assert "Question:" in messages[-1]["content"]
    assert messages[-1]["content"].strip().endswith("Let's think step by step.")


def test_build_mcq_messages_uses_custom_system_prompt() -> None:
    custom_system = "Custom MCQ system prompt."
    messages, _ = build_mcq_messages(
        _sample_mcq_question(),
        use_few_shot_icl=True,
        system_prompt=custom_system,
    )

    assert messages[0] == {"role": "system", "content": custom_system}
