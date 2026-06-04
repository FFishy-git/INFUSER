from __future__ import annotations

from verl_inf_evolve.utils.benchmarks.medqa_prompting import (
    build_medqa_icl_messages,
    extract_medqa_choice,
)


def test_build_medqa_icl_messages_matches_public_shape():
    messages, normalized_gt = build_medqa_icl_messages(
        {
            "question_text": "Which vitamin deficiency causes scurvy?",
            "choices": ["Vitamin A", "Vitamin B12", "Vitamin C", "Vitamin D"],
            "ground_truth": "Vitamin C",
            "data_source": "medqa",
        }
    )

    assert normalized_gt == "C"
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    text = messages[0]["content"]
    assert text.startswith("Question: Which vitamin deficiency causes scurvy?")
    assert "A. Vitamin A" in text
    assert "D. Vitamin D" in text
    assert text.rstrip().endswith("Answer:")
def test_extract_medqa_choice_prefers_boxed_answer():
    assert extract_medqa_choice("Reasoning... \\boxed{C}") == "C"


def test_extract_medqa_choice_supports_answer_prefix():
    assert extract_medqa_choice("Reasoning\nAnswer: B") == "B"


def test_extract_medqa_choice_supports_final_answer_phrase():
    assert extract_medqa_choice("After reasoning, final answer is (D).") == "D"


def test_extract_medqa_choice_supports_last_letter_fallback():
    assert extract_medqa_choice("I considered A and C, but B") == "B"


def test_extract_medqa_choice_returns_none_when_missing():
    assert extract_medqa_choice("No valid choice here.") is None
