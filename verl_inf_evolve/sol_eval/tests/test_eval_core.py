from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

import pytest

from verl_inf_evolve.sol_eval.eval_core import (
    build_benchmark_messages,
    build_question_results,
    compute_sub_bench_metrics,
    flatten_eval_metrics,
    load_benchmark_questions,
    resolve_code_execution_enabled,
    validate_code_execution_policy,
)


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        total_len = sum(len(str(m.get("content", ""))) for m in messages)
        return [0] * total_len

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [0] * len(str(text))


def test_load_benchmark_questions_caps(tmp_path):
    benchmark_path = tmp_path / "demo.json"
    payload = [
        {"question_id": "q1"},
        {"question_id": "q2"},
        {"question_id": "q3"},
    ]
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")

    questions, resolved = load_benchmark_questions(
        str(benchmark_path),
        max_questions=2,
    )
    assert len(questions) == 2
    assert resolved == str(benchmark_path)


def test_load_benchmark_questions_rejects_non_list(tmp_path):
    benchmark_path = tmp_path / "invalid.json"
    benchmark_path.write_text(json.dumps({"question_id": "q1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a list"):
        load_benchmark_questions(str(benchmark_path))


def test_discover_eval_remote_sync_path_skips_base_model_smoke():
    from omegaconf import OmegaConf
    from verl_inf_evolve.sol_eval.sol_eval import _discover_eval_remote_sync_path

    config = OmegaConf.create(
        {
            "eval": {
                "remote_sync_path": None,
                "run_name": "base-model-smoke",
                "checkpoints": [-1],
                "model_path": "Qwen/Qwen3-8B-Base",
            },
            "solver": {"model": {"path": "Qwen/Qwen3-8B-Base"}},
        }
    )

    _discover_eval_remote_sync_path(config)

    assert config.eval.remote_sync_path is None


def test_discover_eval_remote_sync_path_skips_base_model_smoke_string_spec():
    from omegaconf import OmegaConf
    from verl_inf_evolve.sol_eval.sol_eval import _discover_eval_remote_sync_path

    config = OmegaConf.create(
        {
            "eval": {
                "remote_sync_path": None,
                "run_name": "base-model-smoke",
                "checkpoints": "[-1]",
                "model_path": "Qwen/Qwen3-8B-Base",
            },
            "solver": {"model": {"path": "Qwen/Qwen3-8B-Base"}},
        }
    )

    _discover_eval_remote_sync_path(config)

    assert config.eval.remote_sync_path is None


def test_build_benchmark_messages_filters_long_prompts():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "short",
            "question_text": "2+2?",
            "ground_truth": "4",
        },
        {
            "question_id": "long",
            "question_text": "x" * 500,
            "ground_truth": "0",
        },
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=250,
    )

    assert msg_data.question_ids == ["short"]
    assert len(msg_data.messages_list) == 1
    assert msg_data.ground_truths == ["4"]


def test_build_benchmark_messages_mcq_choice_shuffle_is_seeded_and_remaps_gt():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "q1",
            "question_text": "Pick the correct city.",
            "choices": ["London", "Paris", "Rome", "Berlin"],
            "ground_truth": "Paris",
        }
    ]

    shuffle_cfg = {"enabled": True, "mode": "per_question_seeded", "seed": 123}
    msg_data_1 = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        mcq_choice_shuffle_config=shuffle_cfg,
    )
    msg_data_2 = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        mcq_choice_shuffle_config=shuffle_cfg,
    )

    user_prompt = msg_data_1.messages_list[0][1]["content"]
    assert user_prompt == msg_data_2.messages_list[0][1]["content"]
    assert msg_data_1.ground_truths == msg_data_2.ground_truths

    choice_lines = [
        line for line in user_prompt.splitlines() if line[:2] in {"A)", "B)", "C)", "D)"}
    ]
    assert len(choice_lines) == 4
    expected_gt = next(line[0] for line in choice_lines if line.endswith("Paris"))
    assert msg_data_1.ground_truths == [expected_gt]


def test_build_benchmark_messages_mcq_choice_shuffle_disabled_preserves_order():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "q1",
            "question_text": "Pick the correct city.",
            "choices": ["London", "Paris", "Rome", "Berlin"],
            "ground_truth": "Paris",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        mcq_choice_shuffle_config={"enabled": False},
    )

    user_prompt = msg_data.messages_list[0][1]["content"]
    assert "A) London" in user_prompt
    assert "B) Paris" in user_prompt
    assert "C) Rome" in user_prompt
    assert "D) Berlin" in user_prompt
    assert msg_data.ground_truths == ["B"]


def test_build_benchmark_messages_gpqa_auto_shuffle_is_enabled_by_default():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "gpqa-q1",
            "data_source": "gpqa_diamond",
            "question_text": "Pick the correct city.",
            "choices": ["London", "Paris", "Rome", "Berlin"],
            "ground_truth": "Paris",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        mcq_choice_shuffle_config={"enabled": None, "mode": "per_question_seeded", "seed": 123},
    )

    user_prompt = msg_data.messages_list[0][1]["content"]
    assert "B) Paris" not in user_prompt
    choice_lines = [
        line for line in user_prompt.splitlines() if line[:2] in {"A)", "B)", "C)", "D)"}
    ]
    expected_gt = next(line[0] for line in choice_lines if line.endswith("Paris"))
    assert msg_data.ground_truths == [expected_gt]


def test_build_benchmark_messages_supergpqa_auto_shuffle_stays_disabled():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "sgpqa-q1",
            "data_source": "supergpqa",
            "question_text": "Pick the correct city.",
            "choices": ["London", "Paris", "Rome", "Berlin"],
            "ground_truth": "Paris",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        mcq_choice_shuffle_config={"enabled": None, "mode": "per_question_seeded", "seed": 123},
    )

    user_prompt = msg_data.messages_list[0][1]["content"]
    assert "A) London" in user_prompt
    assert "B) Paris" in user_prompt
    assert "C) Rome" in user_prompt
    assert "D) Berlin" in user_prompt
    assert msg_data.ground_truths == ["B"]


def test_flatten_eval_metrics():
    metrics = {
        "accuracy_strict": 0.5,
        "accuracy_lenient": 0.75,
        "total_questions": 20,
        "pass_at_k_strict": {"1": 0.5, "2": 0.6},
        "pass_at_k_lenient": {"1": 0.7, "2": 0.8},
        "response_length_tokens": {"mean": 128.0},
    }

    flat = flatten_eval_metrics("benchmark_eval/demo", metrics)

    assert flat["benchmark_eval/demo/accuracy_strict"] == 0.5
    assert flat["benchmark_eval/demo/accuracy_lenient"] == 0.75
    assert flat["benchmark_eval/demo/total_questions"] == 20
    assert flat["benchmark_eval/demo/pass_at_1_strict"] == 0.5
    assert flat["benchmark_eval/demo/pass_at_2_strict"] == 0.6
    assert flat["benchmark_eval/demo/pass_at_1_lenient"] == 0.7
    assert flat["benchmark_eval/demo/pass_at_2_lenient"] == 0.8
    assert flat["benchmark_eval/demo/response_length_tokens_mean"] == 128.0


def test_flatten_eval_metrics_includes_score_alias():
    metrics = {
        "accuracy_strict": 0.1,
        "accuracy_lenient": 0.2,
        "total_questions": 5,
        "score_name": "eed",
        "eed_score_strict": 47.0,
        "eed_score_lenient": 63.0,
        "pass_at_k_strict": {},
    }

    flat = flatten_eval_metrics("benchmark_eval/phybench", metrics)

    assert flat["benchmark_eval/phybench/eed_score_strict"] == 47.0
    assert flat["benchmark_eval/phybench/eed_score_lenient"] == 63.0


def test_build_benchmark_messages_mcq_icl_produces_multi_turn_messages():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "mcq-icl",
            "question_text": "What is 2 + 3?",
            "choices": ["4", "5", "6", "7"],
            "ground_truth": "5",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        use_public_eval_prompt=True,
    )

    assert len(msg_data.messages_list) == 1
    assert len(msg_data.messages_list[0]) > 2
    assert msg_data.is_mcq_flags == [True]
    assert msg_data.prompt_texts == [None]


def test_build_benchmark_messages_mmlu_pro_icl_bypasses_chat_template():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "mmlu-pro-icl",
            "question_text": "Current question?",
            "choices": ["x", "y", "z", "w"],
            "ground_truth": "z",
            "domain": "biology",
            "data_source": "mmlu_pro",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        use_public_eval_prompt=True,
    )

    assert len(msg_data.messages_list) == 1
    assert msg_data.prompt_texts == [msg_data.messages_list[0][0]["content"]]
    assert msg_data.messages_list[0][0]["role"] == "user"


def test_build_benchmark_messages_medqa_normalizes_ground_truth_to_letter():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "medqa-1",
            "question_text": "Which vitamin deficiency causes scurvy?",
            "choices": ["Vitamin A", "Vitamin B12", "Vitamin C", "Vitamin D"],
            "ground_truth": "Vitamin C",
            "data_source": "medqa",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
    )

    assert msg_data.question_ids == ["medqa-1"]
    assert msg_data.ground_truths == ["C"]
    assert msg_data.is_mcq_flags == [True]
    assert msg_data.data_sources == ["medqa"]
    assert "A) Vitamin A" in msg_data.messages_list[0][1]["content"]
    assert "C) Vitamin C" in msg_data.messages_list[0][1]["content"]


def test_build_benchmark_messages_medqa_icl_uses_public_prompt_shape():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "medqa-icl",
            "question_text": "Which vitamin deficiency causes scurvy?",
            "choices": ["Vitamin A", "Vitamin B12", "Vitamin C", "Vitamin D"],
            "ground_truth": "Vitamin C",
            "data_source": "medqa",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        use_public_eval_prompt=True,
    )

    assert len(msg_data.messages_list) == 1
    assert msg_data.prompt_texts == [msg_data.messages_list[0][0]["content"]]
    assert msg_data.messages_list[0][0]["role"] == "user"
    assert "Question: Which vitamin deficiency causes scurvy?" in msg_data.messages_list[0][0]["content"]
    assert "A. Vitamin A" in msg_data.messages_list[0][0]["content"]
    assert msg_data.messages_list[0][0]["content"].rstrip().endswith("Answer:")


def test_build_benchmark_messages_freeform_icl_produces_multi_turn_messages():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "free-icl",
            "question_text": "What is 17 + 25?",
            "ground_truth": "42",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        use_public_eval_prompt=True,
    )

    assert len(msg_data.messages_list) == 1
    assert len(msg_data.messages_list[0]) > 2
    assert msg_data.is_mcq_flags == [False]


def test_build_benchmark_messages_olympiadbench_icl_uses_benchmark_specific_prompt():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "olympiadbench-icl",
            "question_text": "Find the value of x.",
            "ground_truth": "3",
            "domain": "OlympiadBench_Math",
            "data_source": "olympiadbench",
            "answer_type": "Numerical",
            "unit": "",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        use_public_eval_prompt=True,
    )

    assert len(msg_data.messages_list) == 1
    assert msg_data.is_mcq_flags == [False]
    assert msg_data.prompt_texts == [msg_data.messages_list[0][0]["content"]]
    assert msg_data.messages_list[0] == [
        {"role": "user", "content": msg_data.messages_list[0][0]["content"]}
    ]
    text = msg_data.messages_list[0][0]["content"]
    assert "International Math competition" in text
    assert 'So the final answer is \\boxed{answer}.' in text
    assert "Answer: Let's think step by step." not in text


def test_build_benchmark_messages_math500_training_prompt_without_icl():
    """Without ICL flag, math500 uses the generic training prompt (system+user)."""
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "math500-train",
            "question_text": "What is 6 * 7?",
            "ground_truth": "42",
            "data_source": "math500",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        use_public_eval_prompt=False,
    )

    assert len(msg_data.messages_list) == 1
    # Generic path: system + user messages, prompt_text not set
    assert len(msg_data.messages_list[0]) == 2
    assert msg_data.messages_list[0][0]["role"] == "system"
    assert msg_data.messages_list[0][1]["role"] == "user"
    assert msg_data.prompt_texts == [None]


def test_build_benchmark_messages_math500_icl_oc_aligned_instruct():
    """With ICL flag on instruct model, math500 uses OC prompt with chat template."""
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "math500-icl-instruct",
            "question_text": "What is 6 * 7?",
            "ground_truth": "42",
            "data_source": "math500",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        use_public_eval_prompt=True,
        model_path="Qwen/Qwen3-8B-Instruct",
    )

    assert len(msg_data.messages_list) == 1
    text = msg_data.messages_list[0][0]["content"]
    assert "Please reason step by step" in text
    assert text.startswith("What is 6 * 7?")
    # Instruct: prompt_text NOT set → chat template will be applied
    assert msg_data.prompt_texts == [None]


def test_build_benchmark_messages_math500_icl_oc_aligned_base():
    """With ICL flag on base model, math500 uses OC prompt WITHOUT chat template."""
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "math500-icl-base",
            "question_text": "What is 6 * 7?",
            "ground_truth": "42",
            "data_source": "math500",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        use_public_eval_prompt=True,
        model_path="Qwen/Qwen3-8B-Base",
    )

    assert len(msg_data.messages_list) == 1
    text = msg_data.messages_list[0][0]["content"]
    assert "Please reason step by step" in text
    assert text.startswith("What is 6 * 7?")
    # Base model: prompt_text SET → bypasses chat template
    assert msg_data.prompt_texts == [text]


def test_build_benchmark_messages_hmmt_icl_uses_benchmark_specific_prompt():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "hmmt-icl",
            "question_text": "Compute 6 + 7.",
            "ground_truth": "13",
            "data_source": "hmmt",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        use_public_eval_prompt=True,
    )

    assert len(msg_data.messages_list) == 1
    assert msg_data.is_mcq_flags == [False]
    assert msg_data.prompt_texts == [msg_data.messages_list[0][0]["content"]]
    assert msg_data.messages_list[0] == [
        {"role": "user", "content": msg_data.messages_list[0][0]["content"]}
    ]
    text = msg_data.messages_list[0][0]["content"]
    assert text.startswith("Put your final answer within \\boxed{}.\n\n")
    assert text.rstrip().endswith("Compute 6 + 7.")
    assert "Answer: Let's think step by step." not in text


def test_build_benchmark_messages_phybench_icl_uses_public_prompt_shape():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "phybench-icl",
            "question_text": "Derive the expression for v.",
            "ground_truth": r"\[v\]",
            "data_source": "phybench",
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
        use_public_eval_prompt=True,
    )

    assert len(msg_data.messages_list) == 1
    assert msg_data.is_mcq_flags == [False]
    assert msg_data.prompt_texts == [msg_data.messages_list[0][0]["content"]]
    assert msg_data.messages_list[0] == [
        {"role": "user", "content": msg_data.messages_list[0][0]["content"]}
    ]
    text = msg_data.messages_list[0][0]["content"]
    assert text.startswith(
        "Solve the following physics problem and return only the final result"
    )
    assert "Question: Derive the expression for v." in text
    assert text.rstrip().endswith("Answer:")
    assert "Answer: Let's think step by step." not in text


def test_build_benchmark_messages_preserves_verifier_metadata():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "phy-1",
            "question_text": "Derive the expression.",
            "ground_truth": r"\[x+1\]",
            "data_source": "phybench",
            "verifier_metadata": {
                "phybench_id": 495,
                "tag": "OPTICS",
            },
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
    )

    assert msg_data.verifier_metadatas == [{"phybench_id": 495, "tag": "OPTICS"}]


def test_build_benchmark_messages_supports_code_functional_rows():
    tokenizer = _FakeTokenizer()
    questions = [
        {
            "question_id": "HumanEval/0",
            "benchmark_type": "code_functional",
            "question_text": "def f(x):\n    pass\n",
            "ground_truth": "",
            "data_source": "dummy_code_benchmark",
            "system_prompt": "Return only Python code.",
            "user_prompt": "Complete this function.\n\ndef f(x):\n    pass\n",
            "verifier_metadata": {"entry_point": "f"},
        }
    ]

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=10_000,
    )

    assert msg_data.question_ids == ["HumanEval/0"]
    assert msg_data.is_mcq_flags == [False]
    assert msg_data.benchmark_types == ["code_functional"]
    assert msg_data.messages_list[0][0]["content"] == "Return only Python code."
    assert "Complete this function." in msg_data.messages_list[0][1]["content"]


def test_resolve_code_execution_enabled_auto_enables_for_standalone_code_benchmarks():
    questions = [
        {
            "question_id": "HumanEval/0",
            "benchmark_type": "code_functional",
            "question_text": "def f(x):\n    pass\n",
        }
    ]

    assert resolve_code_execution_enabled(
        questions,
        code_execution_enabled=False,
        execution_scope="standalone",
    ) is True


def test_validate_code_execution_policy_allows_code_benchmarks_by_default_in_standalone():
    questions = [
        {
            "question_id": "HumanEval/0",
            "benchmark_type": "code_functional",
            "question_text": "def f(x):\n    pass\n",
        }
    ]

    validate_code_execution_policy(
        questions,
        code_execution_enabled=False,
        execution_scope="standalone",
    )


def test_evaluate_benchmark_questions_auto_enables_code_execution(monkeypatch):
    from verl_inf_evolve.sol_eval.eval_core import evaluate_benchmark_questions

    questions = [
        {
            "question_id": "HumanEval/0",
            "benchmark_type": "code_functional",
            "question_text": "def f(x):\n    pass\n",
            "ground_truth": "",
            "data_source": "dummy_code_benchmark",
            "verifier_metadata": {"entry_point": "f"},
        }
    ]
    captured: dict[str, bool] = {}
    fake_output = SimpleNamespace()

    monkeypatch.setattr(
        "verl_inf_evolve.sol_eval.eval_core.messages_to_benchmark_dataproto",
        lambda msg_data, tokenizer: object(),
    )
    monkeypatch.setattr(
        "verl_inf_evolve.sol_eval.eval_core.build_question_results",
        lambda output, questions, tokenizer: [{
            "question_id": "HumanEval/0",
            "question_text": questions[0]["question_text"],
            "choices": [],
            "ground_truth": "",
            "data_source": "dummy_code_benchmark",
            "sampled_answers": ["def f(x):\n    return x\n"],
            "extracted_answers": ["def f(x):\n    return x\n"],
            "answer_scores": [1.0],
            "response_token_lengths": [8],
        }],
    )

    def _fake_extract_answer_scores(output, tokenizer, allow_code_execution=False):
        del output, tokenizer
        captured["allow_code_execution"] = allow_code_execution

    monkeypatch.setattr(
        "verl_inf_evolve.utils.mcq_utils.extract_answer_scores",
        _fake_extract_answer_scores,
    )

    result = evaluate_benchmark_questions(
        questions=questions,
        tokenizer=_FakeTokenizer(),
        max_prompt_tokens=10_000,
        n_samples=1,
        generate_batch_fn=lambda batch: fake_output,
        code_execution_enabled=False,
        execution_scope="standalone",
    )

    assert captured["allow_code_execution"] is True
    assert result is not None


def test_validate_code_execution_policy_allows_code_benchmarks_when_enabled():
    questions = [
        {
            "question_id": "HumanEval/0",
            "benchmark_type": "code_functional",
            "question_text": "def f(x):\n    pass\n",
        }
    ]

    validate_code_execution_policy(
        questions,
        code_execution_enabled=True,
        execution_scope="standalone",
    )


def test_validate_code_execution_policy_blocks_code_benchmarks_in_training_by_default():
    questions = [
        {
            "question_id": "HumanEval/0",
            "benchmark_type": "code_functional",
            "question_text": "def f(x):\n    pass\n",
        }
    ]

    with pytest.raises(ValueError, match="disabled for in-training evaluation"):
        validate_code_execution_policy(
            questions,
            code_execution_enabled=True,
            execution_scope="in_training",
            allow_code_execution_in_training=False,
        )


def test_validate_code_execution_policy_allows_code_benchmarks_in_training_with_override():
    questions = [
        {
            "question_id": "HumanEval/0",
            "benchmark_type": "code_functional",
            "question_text": "def f(x):\n    pass\n",
        }
    ]

    validate_code_execution_policy(
        questions,
        code_execution_enabled=True,
        execution_scope="in_training",
        allow_code_execution_in_training=True,
    )


class _DecodeTokenizer(_FakeTokenizer):
    pad_token_id = 0
    eos_token_id = 0

    def decode(self, tokens, skip_special_tokens=True):  # noqa: ARG002
        return "decoded"


def test_build_question_results_includes_sample_scores():
    output = SimpleNamespace(
        batch={
            "responses": np.array([[1, 2, 0], [3, 4, 0]]),
            "response_mask": np.array([[1, 1, 0], [1, 1, 0]]),
        },
        non_tensor_batch={
            "question_id": np.array(["q1", "q1"], dtype=object),
            "ground_truth": np.array(["x+1", "x+1"], dtype=object),
            "extracted_answer": np.array(["x+1", "x+2"], dtype=object),
            "answer_score": np.array([1.0, 0.0], dtype=object),
            "primary_score": np.array([100.0, 26.0], dtype=object),
            "primary_score_name": np.array(["eed", "eed"], dtype=object),
            "primary_score_scale_max": np.array([100.0, 100.0], dtype=object),
        },
    )
    questions = [
        {
            "question_id": "q1",
            "question_text": "Derive the expression.",
            "choices": [],
            "ground_truth": "x+1",
            "data_source": "phybench",
            "domain": "mechanics",
        }
    ]

    results = build_question_results(output, questions, tokenizer=_DecodeTokenizer())

    assert len(results) == 1
    assert results[0]["domain"] == "mechanics"
    assert results[0]["sample_scores"] == [100.0, 26.0]
    assert results[0]["sample_score_name"] == "eed"
    assert results[0]["sample_score_scale_max"] == 100.0


def test_compute_sub_bench_metrics_splits_olympiadbench_by_domain():
    question_results = [
        {
            "question_id": "m1",
            "data_source": "olympiadbench",
            "domain": "OlympiadBench_Math",
            "answer_scores": [1.0, 0.0],
            "response_token_lengths": [10, 12],
        },
        {
            "question_id": "p1",
            "data_source": "olympiadbench",
            "domain": "OlympiadBench_Physics",
            "answer_scores": [0.0, 0.0],
            "response_token_lengths": [8, 9],
        },
    ]

    sub = compute_sub_bench_metrics(question_results, n_samples=2)

    assert set(sub.keys()) == {"math", "physics"}
    assert sub["math"]["accuracy_strict"] == 0.5
    assert sub["physics"]["accuracy_strict"] == 0.0
