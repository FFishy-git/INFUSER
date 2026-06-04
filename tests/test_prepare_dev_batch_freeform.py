from __future__ import annotations

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        del add_generation_prompt
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if tokenize:
            return [0] * len(text)
        return text


def _make_trainer() -> SelfEvolutionTrainer:
    trainer = object.__new__(SelfEvolutionTrainer)
    trainer.solver_tokenizer = _FakeTokenizer()
    trainer.config = SimpleNamespace(
        solver=SimpleNamespace(
            rollout=SimpleNamespace(prompt_length=4096),
        ),
    )
    return trainer


def test_prepare_dev_batch_autodetects_mixed_mcq_and_freeform_questions():
    trainer = _make_trainer()

    batch = trainer.prepare_dev_batch(
        dev_questions=[
            {
                "question_id": "mcq-1",
                "question_text": "Pick the prime number.",
                "choices": ["4", "6", "7", "8"],
                "ground_truth": "7",
                "data_source": "supergpqa",
            },
            {
                "question_id": "open-1",
                "question_text": "What is 2 + 2?",
                "ground_truth": "4",
                "data_source": "math500",
            },
        ]
    )

    ntb = batch.non_tensor_batch

    assert ntb["question_id"].tolist() == ["mcq-1", "open-1"]
    assert ntb["is_mcq"].tolist() == [True, False]
    assert ntb["benchmark_type"].tolist() == ["qa_mcq", "qa_open"]
    assert ntb["data_source"].tolist() == ["supergpqa", "math500"]
    assert ntb["ground_truth"].tolist() == ["C", "4"]

    mcq_prompt = ntb["raw_prompt"][0]
    open_prompt = ntb["raw_prompt"][1]

    assert any("A) 4" in msg["content"] for msg in mcq_prompt)
    assert not any("A)" in msg["content"] for msg in open_prompt)
    assert open_prompt[-1]["content"] == "What is 2 + 2?"


def test_prepare_dev_batch_treats_empty_choices_as_freeform():
    trainer = _make_trainer()

    batch = trainer.prepare_dev_batch(
        dev_questions=[
            {
                "question_id": "open-empty-choices",
                "question_text": "Name the largest planet.",
                "choices": [],
                "ground_truth": "Jupiter",
            }
        ]
    )

    ntb = batch.non_tensor_batch

    assert ntb["is_mcq"].tolist() == [False]
    assert ntb["benchmark_type"].tolist() == ["qa_open"]
    assert ntb["ground_truth"].tolist() == ["Jupiter"]
