from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.utils.benchmarks.olympiadbench_prompting import (
    build_olympiadbench_icl_messages,
)


def test_build_olympiadbench_icl_messages_math_open_ended_prompt_shape():
    messages, normalized_gt = build_olympiadbench_icl_messages(
        {
            "question_text": "Find x if x + 2 = 5.",
            "ground_truth": "3",
            "domain": "OlympiadBench_Math",
            "data_source": "olympiadbench",
            "answer_type": "Numerical",
            "unit": "",
        }
    )

    assert normalized_gt == "3"
    assert messages == [{"role": "user", "content": messages[0]["content"]}]
    text = messages[0]["content"]
    assert "International Math competition" in text
    assert "The answer should be a numerical value." in text
    assert 'So the final answer is \\boxed{answer}.' in text
    assert "Question:" not in text
    assert "Answer: Let's think step by step." not in text
    assert text.rstrip().endswith("Find x if x + 2 = 5.")


def test_build_olympiadbench_icl_messages_physics_mentions_unit_rule():
    messages, normalized_gt = build_olympiadbench_icl_messages(
        {
            "question_text": "A body moves with speed v. Find its momentum.",
            "ground_truth": "mv",
            "domain": "OlympiadBench_Physics",
            "data_source": "olympiadbench",
            "answer_type": "Expression,Numerical",
            "unit": "kg m/s",
        }
    )

    assert normalized_gt == "mv"
    text = messages[0]["content"]
    assert "International Physics competition" in text
    assert "an expression or a numerical value" in text
    assert "unit outside the boxed answer" in text
