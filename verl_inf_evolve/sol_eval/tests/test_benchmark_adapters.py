from __future__ import annotations

from verl_inf_evolve.utils.mcq_utils import format_mcq_question
from verl_inf_evolve.sol_eval.benchmark_adapters import score_response_for_question


def test_score_response_for_question_accepts_reduced_normalized_mcq_payload():
    question = {
        "benchmark_type": "qa_mcq",
        "ground_truth": "B",
        "data_source": "",
        "verifier_metadata": None,
    }

    scored = score_response_for_question(question, r"Reasoning... \boxed{B}")

    assert scored.extracted_answer == "B"
    assert scored.answer_score == 1.0


def test_score_response_for_question_full_mcq_payload_still_normalizes_choice_text():
    question = {
        "benchmark_type": "qa_mcq",
        "question_text": "What is 2 + 3?",
        "choices": ["4", "5", "6", "7"],
        "ground_truth": "5",
        "data_source": "",
        "verifier_metadata": None,
    }

    scored = score_response_for_question(question, r"Reasoning... \boxed{B}")

    assert scored.extracted_answer == "B"
    assert scored.answer_score == 1.0


def test_score_response_for_question_full_mcq_payload_respects_shuffle_config():
    question = {
        "benchmark_type": "qa_mcq",
        "question_id": "gpqa-q1",
        "data_source": "gpqa_diamond",
        "question_text": "Pick the correct city.",
        "choices": ["London", "Paris", "Rome", "Berlin"],
        "ground_truth": "Paris",
        "verifier_metadata": None,
    }
    shuffle_cfg = {"enabled": True, "mode": "per_question_seeded", "seed": 123}
    _, shuffled_gt = format_mcq_question(
        question,
        shuffle_choices=True,
        shuffle_seed=123,
        shuffle_mode="per_question_seeded",
    )

    scored = score_response_for_question(
        question,
        rf"Reasoning... \boxed{{{shuffled_gt}}}",
        mcq_choice_shuffle_config=shuffle_cfg,
    )

    assert scored.extracted_answer == shuffled_gt
    assert scored.answer_score == 1.0


def test_score_response_for_question_math500_uses_symbolic_verifier_first(monkeypatch):
    from verl_inf_evolve.utils.benchmarks.verifiers import math500 as math500_verifier

    monkeypatch.setattr(math500_verifier, "verify_math_answer", lambda predicted, gt: True)

    question = {
        "benchmark_type": "qa_open",
        "ground_truth": "24",
        "data_source": "math500",
        "verifier_metadata": None,
    }

    scored = score_response_for_question(question, r"Reasoning... \boxed{24}")

    assert scored.extracted_answer == "24"
    assert scored.answer_score == 1.0


def test_score_response_for_question_math500_returns_incorrect_when_symbolic_check_fails(monkeypatch):
    from verl_inf_evolve.utils.benchmarks.verifiers import math500 as math500_verifier

    monkeypatch.setattr(math500_verifier, "verify_math_answer", lambda predicted, gt: False)

    question = {
        "benchmark_type": "qa_open",
        "ground_truth": "24",
        "data_source": "math500",
        "verifier_metadata": None,
    }

    scored = score_response_for_question(question, r"Reasoning... \boxed{24}")

    assert scored.extracted_answer == "24"
    assert scored.answer_score == 0.0
