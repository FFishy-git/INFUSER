from __future__ import annotations

from verl_inf_evolve.utils.benchmarks.hmmt_prompting import (
    HMMT_INSTRUCTION,
    build_hmmt_icl_messages,
)


def test_build_hmmt_icl_messages_matches_public_config_shape():
    messages, normalized_gt = build_hmmt_icl_messages(
        {
            "question_text": "Compute 6 + 7.",
            "ground_truth": "13",
            "data_source": "hmmt",
        }
    )

    assert normalized_gt == "13"
    assert messages == [{"role": "user", "content": messages[0]["content"]}]
    text = messages[0]["content"]
    assert text.startswith(HMMT_INSTRUCTION + "\n\n")
    assert text.rstrip().endswith("Compute 6 + 7.")
    assert "Answer: Let's think step by step." not in text
