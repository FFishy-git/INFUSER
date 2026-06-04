from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.utils.benchmarks.phybench_prompting import (
    PHYBENCH_PUBLIC_ZERO_SHOT_PROMPT,
    build_phybench_icl_messages,
    extract_phybench_answer,
)


def test_build_phybench_icl_messages_matches_public_opencompass_shape():
    messages, normalized_gt = build_phybench_icl_messages(
        {
            "question_text": "Find the dispersion relation.",
            "ground_truth": r"\omega = ck",
            "data_source": "phybench",
        }
    )

    assert normalized_gt == r"\omega = ck"
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    text = messages[0]["content"]
    assert text.startswith(PHYBENCH_PUBLIC_ZERO_SHOT_PROMPT + "\n\nQuestion: ")
    assert text.rstrip().endswith("Answer:")


def test_extract_phybench_answer_prefers_boxed_content():
    text = "Reasoning...\n\\boxed{\\frac{n e^2}{m}}"
    assert extract_phybench_answer(text) == r"\frac{n e^2}{m}"


def test_extract_phybench_answer_falls_back_to_last_line():
    text = "Reasoning...\nFinal line expression"
    assert extract_phybench_answer(text) == "Final line expression"
