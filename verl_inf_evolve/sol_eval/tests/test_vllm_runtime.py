from __future__ import annotations

import json
import sys

from verl_inf_evolve.utils.mcq_utils import format_mcq_question
from verl_inf_evolve.sol_eval.vllm_runtime import (
    _gpu_worker,
    prepare_vllm_requests,
    score_completions,
)


class _ChatTemplateTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        self.calls.append({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
        })
        return [11, 22, 33]

    def encode(self, text, add_special_tokens=False):
        self.calls.append({
            "text": text,
            "add_special_tokens": add_special_tokens,
        })
        return [44, 55]


def test_prepare_vllm_requests_uses_tokenized_chat_template():
    tokenizer = _ChatTemplateTokenizer()
    messages_list = [[{"role": "user", "content": "hello"}]]
    question_ids = ["q1"]

    prepared = prepare_vllm_requests(messages_list, question_ids, tokenizer)

    assert prepared == [{"question_id": "q1", "prompt_token_ids": [11, 22, 33]}]
    assert tokenizer.calls == [{
        "messages": messages_list[0],
        "add_generation_prompt": True,
        "tokenize": True,
    }]


def test_prepare_vllm_requests_uses_raw_prompt_text_when_provided():
    tokenizer = _ChatTemplateTokenizer()
    messages_list = [[{"role": "user", "content": "ignored"}]]
    question_ids = ["q1"]

    prepared = prepare_vllm_requests(
        messages_list,
        question_ids,
        tokenizer,
        prompt_texts=["plain prompt"],
    )

    assert prepared == [{"question_id": "q1", "prompt_token_ids": [44, 55]}]
    assert tokenizer.calls == [{
        "text": "plain prompt",
        "add_special_tokens": False,
    }]


def test_gpu_worker_passes_eos_stop_token_ids(tmp_path, monkeypatch):
    prompts_file = tmp_path / "prompts.json"
    output_file = tmp_path / "result.json"
    prompts_file.write_text(
        json.dumps([{"question_id": "q1", "prompt_token_ids": [1, 2, 3]}]),
        encoding="utf-8",
    )

    captured = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_params"] = kwargs

    class FakeTokensPrompt:
        def __init__(self, *, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    class FakeCompletion:
        def __init__(self):
            self.text = "answer"
            self.token_ids = [7, 8]
            self.finish_reason = "stop"

    class FakeOutput:
        def __init__(self):
            self.outputs = [FakeCompletion()]

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_init"] = kwargs

        def generate(self, prompts, params):  # noqa: ARG002
            captured["prompts"] = prompts
            return [FakeOutput()]

    class FakeVllmModule:
        LLM = FakeLLM
        SamplingParams = FakeSamplingParams
        TokensPrompt = FakeTokensPrompt

    monkeypatch.setitem(sys.modules, "vllm", FakeVllmModule)

    _gpu_worker(
        rank=0,
        model_path="Qwen/Qwen3-8B-Base",
        prompts_file=str(prompts_file),
        output_file=str(output_file),
        n_samples=1,
        temperature=0.7,
        top_p=0.9,
        top_k=17,
        max_tokens=128,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        enforce_eager=False,
        eos_token_id=151643,
    )

    assert captured["sampling_params"]["stop_token_ids"] == [151643]
    assert captured["sampling_params"]["top_p"] == 0.9
    assert captured["sampling_params"]["top_k"] == 17
    assert len(captured["prompts"]) == 1
    assert captured["prompts"][0].prompt_token_ids == [1, 2, 3]

    result = json.loads(output_file.read_text(encoding="utf-8"))
    assert result == [{
        "question_id": "q1",
        "completions": [{
            "text": "answer",
            "token_len": 2,
            "finish_reason": "stop",
        }],
    }]


def test_score_completions_normalizes_mcq_ground_truth_text_to_letter():
    questions = [{
        "question_id": "medqa-q1",
        "question_text": "Which option is correct?",
        "choices": ["Alpha", "Bravo", "Charlie", "Delta"],
        "ground_truth": "Bravo",
        "data_source": "medqa",
    }]
    raw_results = [{
        "question_id": "medqa-q1",
        "completions": [{
            "text": "Reasoning... \\boxed{B}",
            "token_len": 5,
            "finish_reason": "stop",
        }],
    }]

    results = score_completions(raw_results, questions)

    assert results[0]["extracted_answers"] == ["B"]
    assert results[0]["answer_scores"] == [1.0]


def test_score_completions_respects_mcq_shuffle_config_for_gpqa():
    questions = [{
        "question_id": "gpqa-q1",
        "question_text": "Pick the correct city.",
        "choices": ["London", "Paris", "Rome", "Berlin"],
        "ground_truth": "Paris",
        "data_source": "gpqa_diamond",
    }]
    shuffle_cfg = {"enabled": None, "mode": "per_question_seeded", "seed": 123}
    _, shuffled_gt = format_mcq_question(
        questions[0],
        shuffle_choices=True,
        shuffle_seed=123,
        shuffle_mode="per_question_seeded",
    )
    raw_results = [{
        "question_id": "gpqa-q1",
        "completions": [{
            "text": rf"Reasoning... \boxed{{{shuffled_gt}}}",
            "token_len": 5,
            "finish_reason": "stop",
        }],
    }]

    results = score_completions(
        raw_results,
        questions,
        mcq_choice_shuffle_config=shuffle_cfg,
    )

    assert results[0]["extracted_answers"] == [shuffled_gt]
    assert results[0]["answer_scores"] == [1.0]
