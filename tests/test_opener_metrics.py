from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from verl_inf_evolve.trainer.rollout_metrics import (
    classify_opener,
    compute_opener_advantage_metrics,
    compute_opener_rollout_metrics,
)


class _FakeTokenizer:
    def __init__(self, decoded_by_first_token: dict[int, str]):
        self.decoded_by_first_token = decoded_by_first_token

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        if hasattr(token_ids, "detach"):
            token_ids = token_ids.detach().cpu().tolist()
        return self.decoded_by_first_token[int(token_ids[0])]


def test_classify_opener_matches_basin_rules():
    assert classify_opener("Alright, let's solve") == "alright"
    assert classify_opener("  alright!") == "alright"
    # All "To <verb>..." openers collapse to a single "to" bucket — they share
    # the same imperative-infinitive style and have no behavioral length signal
    # between them.
    assert classify_opener("To solve this problem") == "to"
    assert classify_opener("To this end") == "to"
    assert classify_opener("To determine the answer") == "to"
    assert classify_opener("To find the radius") == "to"
    # Anything that doesn't start with "Alright" or "To " falls into "others".
    assert classify_opener("Let's solve it") == "others"
    assert classify_opener("We are given") == "others"
    assert classify_opener("Step 1: analyze") == "others"


def test_compute_opener_rollout_metrics_adds_shares_lengths_and_unbiased_delta():
    output = SimpleNamespace(
        batch={
            "responses": torch.tensor(
                [
                    [1, 0, 0, 0],
                    [2, 0, 0, 0],
                    [3, 0, 0, 0],
                    [4, 0, 0, 0],
                ]
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1, 1, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 1],
                    [1, 0, 0, 0],
                ],
                dtype=torch.float32,
            ),
        },
        non_tensor_batch={
            "question_id": np.array(["q1", "q1", "q2", "q2"], dtype=object),
            "answer_score": np.array([1.0, 0.0, 0.0, 1.0], dtype=object),
        },
    )
    tokenizer = _FakeTokenizer(
        {
            1: "Alright, let's solve",
            2: "To solve this problem",
            3: "To determine the answer",
            4: "Let's solve",
        }
    )

    metrics = compute_opener_rollout_metrics(output, tokenizer, prefix="rollout")

    assert output.non_tensor_batch["opener_class"].tolist() == [
        "alright",
        "to",
        "to",
        "others",
    ]
    assert metrics["rollout/opener/share/alright"] == pytest.approx(0.25)
    assert metrics["rollout/opener/share/to"] == pytest.approx(0.5)
    assert metrics["rollout/opener/share/others"] == pytest.approx(0.25)
    assert metrics["rollout/opener/top_share"] == pytest.approx(0.5)
    assert metrics["rollout/opener/mean_response_len/alright"] == pytest.approx(3.0)
    assert metrics["rollout/opener/mean_response_len/others"] == pytest.approx(1.0)
    # to bucket = 2 rollouts (lengths 2 + 4) → mean 3.0.
    assert metrics["rollout/opener/mean_response_len/to"] == pytest.approx(3.0)
    # q1 delta = 1 - 0; q2 has no Alright rollout and is excluded.
    assert metrics["rollout/opener/delta_within/alright_minus_non"] == pytest.approx(1.0)
    assert metrics["rollout/opener/delta_within/n_mixed_questions"] == pytest.approx(1.0)
    assert metrics["rollout/opener/delta_within/mixed_question_frac"] == pytest.approx(0.5)
    # Per-class within-Q debiased acc: each Q where the class appeared
    # contributes its class-mean once.
    # q1: alright=1.0, to=0.0           q_mean=0.5
    # q2: to=0.0,    others=1.0         q_mean=0.5
    assert metrics["rollout/opener/within_q/acc/alright"] == pytest.approx(1.0)
    assert metrics["rollout/opener/within_q/acc/to"] == pytest.approx(0.0)
    assert metrics["rollout/opener/within_q/acc/others"] == pytest.approx(1.0)
    # acc_minus_qmean only counts Qs where another class was also present.
    assert metrics["rollout/opener/within_q/acc_minus_qmean/alright"] == pytest.approx(0.5)
    assert metrics["rollout/opener/within_q/acc_minus_qmean/to"] == pytest.approx(-0.5)
    assert metrics["rollout/opener/within_q/acc_minus_qmean/others"] == pytest.approx(0.5)
    assert metrics["rollout/opener/within_q/n_questions/alright"] == pytest.approx(1.0)
    assert metrics["rollout/opener/within_q/n_questions/to"] == pytest.approx(2.0)
    assert metrics["rollout/opener/within_q/n_questions/others"] == pytest.approx(1.0)
    assert metrics["rollout/opener/within_q/n_mixed_questions/alright"] == pytest.approx(1.0)
    assert metrics["rollout/opener/within_q/n_mixed_questions/to"] == pytest.approx(2.0)
    assert metrics["rollout/opener/within_q/n_mixed_questions/others"] == pytest.approx(1.0)
    # Q-level share: # of unique question_ids where the class produced ≥1 rollout,
    # divided by # of unique question_ids in the pass.
    # 2 unique Qs (q1, q2). alright in q1 only; to in q1 + q2; others in q2 only.
    assert metrics["rollout/opener/n_unique_questions"] == pytest.approx(2.0)
    assert metrics["rollout/opener/q_share/alright"] == pytest.approx(0.5)
    assert metrics["rollout/opener/q_share/to"] == pytest.approx(1.0)
    assert metrics["rollout/opener/q_share/others"] == pytest.approx(0.5)
    assert metrics["rollout/opener/q_count/alright"] == pytest.approx(1.0)
    assert metrics["rollout/opener/q_count/to"] == pytest.approx(2.0)
    assert metrics["rollout/opener/q_count/others"] == pytest.approx(1.0)


def test_compute_opener_advantage_metrics_uses_equal_question_weighting():
    batch = SimpleNamespace(
        batch={
            "advantages": torch.tensor(
                [
                    [1.0, 1.0],
                    [-1.0, -1.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                    [2.0, 2.0],
                    [4.0, 4.0],
                ]
            ),
            "response_mask": torch.ones(6, 2),
        },
        non_tensor_batch={
            "uid": np.array(["q1", "q1", "q1", "q2", "q2", "q2"], dtype=object),
            "opener_class": np.array(
                ["alright", "to", "others", "alright", "to", "others"],
                dtype=object,
            ),
        },
    )

    metrics = compute_opener_advantage_metrics(batch, prefix="solver_ppo")

    # q1: alright 1 - non mean( -1, 3 ) = 0
    # q2: alright 0 - non mean( 2, 4 ) = -3
    # equal-Q mean = -1.5
    assert metrics[
        "solver_ppo/opener/advantage_delta_within/alright_minus_non"
    ] == pytest.approx(-1.5)
    assert metrics[
        "solver_ppo/opener/advantage_delta_within/n_mixed_questions"
    ] == pytest.approx(2.0)
    assert metrics["solver_ppo/opener/advantage_mean/alright"] == pytest.approx(0.5)
    # Per-class within-Q debiased advantage:
    # q1 q_mean = mean(1, -1, 3)/3 = 1.0; alright=1, to=-1, others=3
    # q2 q_mean = mean(0,  2, 4)/3 = 2.0; alright=0, to= 2, others=4
    # within_q/advantage[alright] = mean(1, 0) = 0.5
    # within_q/advantage[to]      = mean(-1, 2) = 0.5
    # within_q/advantage[others]  = mean(3, 4) = 3.5
    assert metrics["solver_ppo/opener/within_q/advantage/alright"] == pytest.approx(0.5)
    assert metrics["solver_ppo/opener/within_q/advantage/to"] == pytest.approx(0.5)
    assert metrics["solver_ppo/opener/within_q/advantage/others"] == pytest.approx(3.5)
    # advantage_minus_qmean: q1 alright=1-1=0, q2 alright=0-2=-2 → mean -1
    # to: 1-1=... wait let me recompute. q1 to=-1, q1 q_mean=1 → -2. q2 to=2, q2 q_mean=2 → 0. mean -1.
    # others: q1 others=3-1=2; q2 others=4-2=2. mean 2.
    assert metrics["solver_ppo/opener/within_q/advantage_minus_qmean/alright"] == pytest.approx(-1.0)
    assert metrics["solver_ppo/opener/within_q/advantage_minus_qmean/to"] == pytest.approx(-1.0)
    assert metrics["solver_ppo/opener/within_q/advantage_minus_qmean/others"] == pytest.approx(2.0)
    assert metrics["solver_ppo/opener/within_q/n_questions/alright"] == pytest.approx(2.0)
    # Q-level share — both Qs have rollouts of every class in this synthetic batch.
    assert metrics["solver_ppo/opener/n_unique_questions"] == pytest.approx(2.0)
    assert metrics["solver_ppo/opener/q_share/alright"] == pytest.approx(1.0)
    assert metrics["solver_ppo/opener/q_share/to"] == pytest.approx(1.0)
    assert metrics["solver_ppo/opener/q_share/others"] == pytest.approx(1.0)
    assert metrics["solver_ppo/opener/q_count/alright"] == pytest.approx(2.0)
    assert metrics["solver_ppo/opener/q_count/to"] == pytest.approx(2.0)
    assert metrics["solver_ppo/opener/q_count/others"] == pytest.approx(2.0)


def _resolve_bonus(cfg_dict):
    """Replicates SelfEvolutionTrainer._resolve_solver_formatting_bonus.

    Tests the helper logic without importing the full trainer (which would
    pull in verl/Ray/torch.distributed).
    """
    config = OmegaConf.create({"algorithm": cfg_dict})
    cfg = config.algorithm.get("solver_formatting_bonus", None)
    if cfg is None or not bool(cfg.get("enabled", False)):
        return 0.0
    return float(cfg.get("alright_bonus", 0.0))


def test_resolve_solver_formatting_bonus_disabled_by_default():
    assert _resolve_bonus({}) == 0.0
    assert _resolve_bonus(
        {"solver_formatting_bonus": {"enabled": False, "alright_bonus": 0.5}}
    ) == 0.0


def test_resolve_solver_formatting_bonus_returns_value_when_enabled():
    bonus = _resolve_bonus(
        {"solver_formatting_bonus": {"enabled": True, "alright_bonus": 0.01}}
    )
    assert bonus == pytest.approx(0.01)
