from __future__ import annotations

from verl_inf_evolve.utils.benchmarks.math500_prompting import (
    build_math500_icl_messages,
    extract_math500_answer,
)


def test_build_math500_icl_messages_matches_minerva_style_shape():
    messages, normalized_gt = build_math500_icl_messages(
        {
            "question_text": "What is 6 * 7?",
            "ground_truth": "42",
            "data_source": "math500",
        }
    )

    assert normalized_gt == "42"
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    text = messages[0]["content"]
    assert text.count("Problem:\n") == 5
    assert text.count("\n\nSolution:") == 5
    assert "Final Answer: The final answer is" in text
    assert text.rstrip().endswith("Problem:\nWhat is 6 * 7?\n\nSolution:")


def test_extract_math500_answer_supports_minerva_final_answer_pattern():
    text = (
        "Reasoning...\n"
        "Final Answer: The final answer is $-\\frac{2}{3}$. "
        "I hope it is correct."
    )
    assert extract_math500_answer(text) == r"-\frac{2}{3}"


def test_extract_math500_answer_supports_answer_line_pattern():
    text = "Reasoning...\nANSWER: $24$"
    assert extract_math500_answer(text) == "24"


def test_extract_math500_answer_prefers_last_box_when_valid():
    text = r"Reasoning...\n\boxed{12}\nMore reasoning...\n\boxed{24}"
    assert extract_math500_answer(text) == "24"


def test_extract_math500_answer_falls_back_to_first_box_when_last_is_incomplete():
    text = r"Reasoning...\n\boxed{24}\nMore reasoning...\n\boxed{"
    assert extract_math500_answer(text) == "24"


def test_build_math_prompt_branch_uses_same_math500_prompting():
    from verl_inf_evolve.sol_eval.benchmark_adapters import build_messages_for_question

    messages, normalized_gt, is_mcq = build_messages_for_question(
        {
            "question_id": "math-icl",
            "question_text": "What is 6 * 7?",
            "ground_truth": "42",
            "data_source": "math",
            "choices": [],
        },
        use_public_eval_prompt=True,
    )

    assert not is_mcq
    assert normalized_gt == "42"
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].count("Problem:\n") == 5
