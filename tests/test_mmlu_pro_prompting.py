from __future__ import annotations

from verl_inf_evolve.utils.benchmarks.mmlu_pro_prompting import (
    build_mmlu_pro_icl_messages,
    extract_mmlu_pro_choice,
)


def test_build_mmlu_pro_icl_messages_matches_reference_shape(monkeypatch):
    def _fake_pool():
        return {
            "biology": [
                {
                    "question": "Few-shot question?",
                    "options": ["opt1", "opt2", "opt3", "opt4"],
                    "cot_content": "A: Let's think step by step. the answer is (B)",
                    "category": "biology",
                }
            ]
        }

    monkeypatch.setattr(
        "verl_inf_evolve.utils.benchmarks.mmlu_pro_prompting._load_mmlu_pro_validation_pool",
        _fake_pool,
    )

    messages, normalized_gt = build_mmlu_pro_icl_messages(
        {
            "question_text": "Current question?",
            "choices": ["x", "y", "z", "w"],
            "ground_truth": "z",
            "domain": "biology",
            "data_source": "mmlu_pro",
        }
    )

    assert normalized_gt == "C"
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    text = messages[0]["content"]
    assert "The following are multiple choice questions (with answers) about biology." in text
    assert "Question:\nFew-shot question?" in text
    assert "Answer: Let's think step by step. the answer is (B)" in text
    assert "Question:\nCurrent question?" in text
    assert text.rstrip().endswith("Answer: Let's think step by step.")


def test_extract_mmlu_pro_choice_prefers_reference_pattern():
    assert extract_mmlu_pro_choice("After reasoning, the answer is (D).") == "D"


def test_extract_mmlu_pro_choice_supports_answer_prefix_fallback():
    assert extract_mmlu_pro_choice("Reasoning\nAnswer: C") == "C"


def test_extract_mmlu_pro_choice_supports_last_letter_fallback():
    assert extract_mmlu_pro_choice("I considered A and B, final C") == "C"


def test_extract_mmlu_pro_choice_returns_none_when_missing():
    assert extract_mmlu_pro_choice("No valid choice here.") is None
