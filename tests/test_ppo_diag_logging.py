import os
import sys

import numpy as np
import pytest
import torch
from tensordict import TensorDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("torchdata")

from verl import DataProto
from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer
from verl_inf_evolve.utils.ppo_diag import (
    collect_ppo_actor_diag,
    collect_ppo_batch_diag,
)


def _make_batch() -> DataProto:
    batch = TensorDict(
        {
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1], [1, 1, 0, 0]],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor(
                [[1, 1], [1, 0]],
                dtype=torch.long,
            ),
        },
        batch_size=2,
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={"uid": np.array(["q1", "q2"], dtype=object)},
        meta_info={
            "global_token_num": [4, 2],
            "ppo_mini_batch_size": 1,
            "dp_size": 8,
            "temperature": 0.7,
        },
    )


def _make_actor_output() -> DataProto:
    return DataProto(
        meta_info={
            "metrics": {
                "perf/mfu/actor": [0.4, 0.5],
                "perf/max_memory_allocated_gb": [10.0, 12.5],
                "perf/max_memory_reserved_gb": [11.0, 13.5],
                "perf/cpu_memory_used_gb": [64.0, 66.0],
                "actor/lr": [1.5e-6, 1.5e-6],
                "actor/num_minibatches": [2, 2],
                "actor/ppo_mini_batch_size": [1, 1],
                "actor/ppo_micro_batch_size_per_gpu": [4, 4],
                "actor/total_batch_size": [2, 2],
            }
        }
    )


def test_collect_ppo_batch_diag_summarizes_batch():
    diag = collect_ppo_batch_diag(_make_batch())

    assert diag["batch_size"] == 2
    assert diag["input_tokens"] == 6
    assert diag["response_tokens"] == 3
    assert diag["avg_response_tokens"] == pytest.approx(1.5)
    assert diag["max_response_tokens"] == 2
    assert diag["uid_groups"] == 2
    assert diag["global_tokens"] == 6
    assert diag["ppo_mini_batch_size"] == 1
    assert diag["dp_size"] == 8
    assert diag["temperature"] == pytest.approx(0.7)


def test_collect_ppo_actor_diag_reduces_worker_metrics():
    diag = collect_ppo_actor_diag(_make_actor_output())

    assert diag["worker_mfu"] == pytest.approx(0.45)
    assert diag["worker_max_alloc_gb"] == pytest.approx(12.5)
    assert diag["worker_max_reserved_gb"] == pytest.approx(13.5)
    assert diag["worker_cpu_mem_gb"] == pytest.approx(65.0)
    assert diag["worker_lr"] == pytest.approx(1.5e-6)
    assert diag["worker_num_minibatches"] == pytest.approx(2.0)
    assert diag["worker_ppo_mini_batch_size"] == pytest.approx(1.0)
    assert diag["worker_ppo_micro_batch_size_per_gpu"] == pytest.approx(4.0)
    assert diag["worker_total_batch_size"] == pytest.approx(2.0)


def test_ppo_stage_diag_logs_compact_summary(caplog):
    trainer = object.__new__(SelfEvolutionTrainer)

    with caplog.at_level("INFO"):
        trainer._ppo_stage_diag(
            "stage5-pre-update-actor",
            role_name="generator",
            batch=_make_batch(),
            actor_output=_make_actor_output(),
            ans_loop=3,
            gen_loop=7,
        )

    log_text = caplog.text
    assert "[ppo-diag stage5-pre-update-actor]" in log_text
    assert "role=generator" in log_text
    assert "ans_loop=3" in log_text
    assert "gen_loop=7" in log_text
    assert "batch_size=2" in log_text
    assert "global_tokens=6" in log_text
    assert "worker_max_alloc_gb=12.50" in log_text
    assert "worker_max_reserved_gb=13.50" in log_text
