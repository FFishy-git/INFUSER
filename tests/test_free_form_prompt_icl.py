"""Tests for optional free-form few-shot ICL prompt construction."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.utils.prompts import (
    FREE_FORM_ANSWER_GENERATION_FEW_SHOT_EXAMPLES,
    MCQ_ANSWER_GENERATION_FEW_SHOT_EXAMPLES,
    build_free_form_messages,
)


def test_build_free_form_messages_without_icl() -> None:
    messages = build_free_form_messages("What is 2 + 2?", use_few_shot_icl=False)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is 2 + 2?"
    assert "demonstrated format" not in messages[0]["content"]
    assert "\\boxed{}" in messages[0]["content"]


def test_build_free_form_messages_with_icl_includes_examples_and_format_rules() -> None:
    question = "Name the largest planet in our solar system."
    messages = build_free_form_messages(question, use_few_shot_icl=True)

    expected_len = 1 + 2 * len(FREE_FORM_ANSWER_GENERATION_FEW_SHOT_EXAMPLES) + 1
    assert len(messages) == expected_len

    # System message first, user question last
    assert messages[0]["role"] == "system"
    assert "Think step by step" in messages[0]["content"]
    assert "correct final answer" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert question in messages[-1]["content"]
    assert "Question:" in messages[-1]["content"]
    assert "Answer:" in messages[-1]["content"]
    assert "\\boxed{" not in messages[-1]["content"]
    assert "A)" not in messages[-1]["content"]
    assert "B)" not in messages[-1]["content"]
    assert "C)" not in messages[-1]["content"]
    assert "D)" not in messages[-1]["content"]
    assert messages[-1]["content"].strip().endswith("Let's think step by step.")

    # Few-shot pairs should alternate user/assistant
    for idx in range(1, len(messages) - 1, 2):
        assert messages[idx]["role"] == "user"
        assert messages[idx + 1]["role"] == "assistant"
        assert "Question:" in messages[idx]["content"]
        assert "Answer:" in messages[idx]["content"]
        assert "A)" not in messages[idx]["content"]
        assert "B)" not in messages[idx]["content"]
        assert "C)" not in messages[idx]["content"]
        assert "D)" not in messages[idx]["content"]
        assert "The answer is \\boxed{" in messages[idx + 1]["content"]
        assert "Final Answer:" not in messages[idx + 1]["content"]
        assert "\\boxed{" in messages[idx + 1]["content"]


def test_build_free_form_messages_uses_custom_system_prompt() -> None:
    custom_system = "Custom system prompt for testing."
    messages = build_free_form_messages(
        "Question text",
        use_few_shot_icl=True,
        system_prompt=custom_system,
    )
    assert messages[0] == {"role": "system", "content": custom_system}


def test_free_form_few_shot_examples_share_mcq_questions_and_box_answer_text() -> None:
    assert len(FREE_FORM_ANSWER_GENERATION_FEW_SHOT_EXAMPLES) == len(
        MCQ_ANSWER_GENERATION_FEW_SHOT_EXAMPLES
    )

    for (free_user, free_assistant), (mcq_user, mcq_assistant) in zip(
        FREE_FORM_ANSWER_GENERATION_FEW_SHOT_EXAMPLES,
        MCQ_ANSWER_GENERATION_FEW_SHOT_EXAMPLES,
    ):
        assert free_user.split("\n", 1)[0] == mcq_user.split("\n", 1)[0]
        assert "A)" not in free_user
        assert "B)" not in free_user
        assert "C)" not in free_user
        assert "D)" not in free_user
        assert "The answer is \\boxed{" in free_assistant
        assert "The answer is \\boxed{" in mcq_assistant

        mcq_boxed = mcq_assistant.split("\\boxed{", 1)[1].rstrip("}")
        free_boxed = free_assistant.split("\\boxed{", 1)[1].rstrip("}")

        assert len(mcq_boxed) == 1
        assert mcq_boxed.isalpha()
        assert free_boxed != mcq_boxed
        assert f") {free_boxed}" in mcq_user
