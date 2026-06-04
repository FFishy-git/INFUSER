from __future__ import annotations

import os
import sys

from omegaconf import OmegaConf

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer
from verl_inf_evolve.utils.prompts import (
    FREE_FORM_QUESTION_GENERATION_SYSTEM_PROMPT,
    MCQ_QUESTION_GENERATION_SYSTEM_PROMPT,
)


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
    trainer.gen_tokenizer = _FakeTokenizer()
    trainer.config = OmegaConf.create(
        {
            "generator": {
                "rollout": {"prompt_length": 20000},
            },
            "training": {
                "doc_batch_size": 2,
                "seed": 42,
                "question_generation_routes": [
                    {
                        "prompt_type": "free_form",
                        "source_patterns": ["Nemotron-CC-Math-v1"],
                    }
                ],
            },
        }
    )
    trainer.documents = {
        "orig": {
            "doc_id": "orig",
            "content": "Original science document.",
            "source_pdf": "datasets/websources/Astronomy_v1/example.pdf",
        },
        "nemotron": {
            "doc_id": "nemotron",
            "content": "Math document with $x^2$ and a theorem.",
            "source_pdf": "hf://datasets/nvidia/Nemotron-CC-Math-v1/4plus_MIND#abc",
        },
    }
    trainer._init_question_generation_routes()
    return trainer


def test_question_generation_routes_send_nemotron_docs_to_free_form_prompt():
    trainer = _make_trainer()

    assert trainer._resolve_question_generation_prompt_type(trainer.documents["orig"]) == "mcq"
    assert (
        trainer._resolve_question_generation_prompt_type(trainer.documents["nemotron"])
        == "free_form"
    )


def test_prepare_document_batch_mixes_mcq_and_free_form_prompts():
    trainer = _make_trainer()

    class _Dataset:
        def next_batch(self):
            return ["orig", "nemotron"], False

    trainer.doc_dataset = _Dataset()
    trainer._shortcut_doc_meta = {}

    batch = trainer._prepare_document_batch()
    ntb = batch.non_tensor_batch

    assert ntb["doc_id"].tolist() == ["orig", "nemotron"]
    assert ntb["question_generation_prompt_type"].tolist() == ["mcq", "free_form"]
    assert ntb["raw_prompt"][0][0]["content"] == MCQ_QUESTION_GENERATION_SYSTEM_PROMPT
    assert ntb["raw_prompt"][1][0]["content"] == FREE_FORM_QUESTION_GENERATION_SYSTEM_PROMPT
