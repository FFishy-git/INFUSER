import os
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

# Add project root to path for local imports (verl, verl_inf_evolve).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("torchdata")

from verl import DataProto
from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer
from verl_inf_evolve.utils.generator_reward_utils import (
    build_stage4_reward_payload,
    resolve_generator_reward_components,
    resolve_generator_reward_structure,
)


class _TrainerStub:
    def __init__(self, config):
        self.config = config

    _quantify_scores_before_advantage = SelfEvolutionTrainer._quantify_scores_before_advantage
    _build_advantage_component_batch = SelfEvolutionTrainer._build_advantage_component_batch
    compute_advantage = SelfEvolutionTrainer.compute_advantage


def _make_gen_output(question_ids: list[str], doc_ids: list[str]) -> DataProto:
    batch_size = len(question_ids)
    prompt_len = 4
    response_len = 3
    total_len = prompt_len + response_len

    input_ids = torch.arange(batch_size * total_len, dtype=torch.long).reshape(batch_size, total_len)
    attention_mask = torch.ones(batch_size, total_len, dtype=torch.long)
    position_ids = torch.arange(total_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    responses = input_ids[:, -response_len:]

    td = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
        },
        batch_size=batch_size,
    )
    return DataProto(
        batch=td,
        non_tensor_batch={
            "question_id": np.array(question_ids, dtype=object),
            "doc_id": np.array(doc_ids, dtype=object),
        },
    )


def _make_config(quant_mode, gamma=0.2):
    return OmegaConf.create(
        {
            "training": {
                "gen_invalid_penalty": 0.0,
                "generator_reward_components": ["influence_rewards", "invalid_rewards"],
                "generator_reward_combination_mode": "sum_scores",
            },
            "influence": {
                "quantification_mode": quant_mode,
                "group_std_gamma": gamma,
            },
            "generator": {"rollout": {"temperature": 0.7}},
        }
    )


def test_group_std_top_gamma_partial_survival():
    trainer = _TrainerStub(_make_config("group_std_top_gamma", gamma=0.5))
    gen_output = _make_gen_output(
        question_ids=["q1", "q2", "q3", "q4", "q5", "q6"],
        doc_ids=["d1", "d1", "d1", "d2", "d2", "d2"],
    )
    rewards = {"q1": 0.0, "q2": 1.0, "q3": 2.0, "q4": 0.0, "q5": 0.0, "q6": 0.0}

    batch, metrics = SelfEvolutionTrainer.prepare_gen_update_batch(trainer, gen_output, rewards)

    assert batch is not None
    assert len(batch) == 3
    assert metrics["gen_quant/mode_id"] == 3.0
    assert metrics["gen_quant/survival/groups_kept"] == 1.0
    assert metrics["gen_quant/survival/groups_dropped"] == 1.0
    assert metrics["gen_quant/survival/all_dropped"] == 0.0
    assert metrics["gen_quant/tau"] > 0.0


def test_group_std_top_gamma_tie_drops_all_strict_greater_than():
    trainer = _TrainerStub(_make_config("group_std_top_gamma", gamma=0.5))
    gen_output = _make_gen_output(
        question_ids=["q1", "q2", "q3", "q4"],
        doc_ids=["d1", "d1", "d2", "d2"],
    )
    rewards = {"q1": 0.0, "q2": 2.0, "q3": 3.0, "q4": 5.0}

    batch, metrics = SelfEvolutionTrainer.prepare_gen_update_batch(trainer, gen_output, rewards)

    assert batch is None
    assert abs(metrics["gen_quant/tau"] - 1.0) < 1e-6
    assert metrics["gen_quant/survival/groups_kept"] == 0.0
    assert metrics["gen_quant/survival/all_dropped"] == 1.0


def test_group_std_top_gamma_boundary_gamma_one():
    trainer = _TrainerStub(_make_config("group_std_top_gamma", gamma=1.0))
    gen_output = _make_gen_output(
        question_ids=["q1", "q2", "q3", "q4"],
        doc_ids=["d1", "d1", "d2", "d2"],
    )
    rewards = {"q1": 0.0, "q2": 2.0, "q3": 1.0, "q4": 1.0}

    batch, metrics = SelfEvolutionTrainer.prepare_gen_update_batch(trainer, gen_output, rewards)

    assert batch is not None
    assert len(batch) == 2
    assert metrics["gen_quant/tau"] == 0.0
    assert metrics["gen_quant/survival/groups_kept"] == 1.0
    assert metrics["gen_quant/survival/groups_dropped"] == 1.0


def test_1bit_bucket_metrics_and_zero_var_filter_coexist():
    trainer = _TrainerStub(_make_config("1bit"))
    gen_output = _make_gen_output(
        question_ids=["q1", "q2", "q3", "q4", "q5", "q6"],
        doc_ids=["d1", "d1", "d2", "d2", "d3", "d3"],
    )
    rewards = {"q1": -1.0, "q2": 2.0, "q3": 0.0, "q4": 3.0, "q5": 5.0, "q6": 5.0}

    batch, metrics = SelfEvolutionTrainer.prepare_gen_update_batch(trainer, gen_output, rewards)

    assert batch is not None
    assert len(batch) == 4
    assert metrics["gen_quant/mode_id"] == 1.0
    assert metrics["gen_quant/bucket/zero"] == 2.0
    assert metrics["gen_quant/bucket/one"] == 2.0
    assert metrics["gen_quant/survival/groups_kept"] == 2.0
    assert metrics["gen_quant/survival/groups_dropped"] == 1.0


def test_2bit_bucket_metrics():
    trainer = _TrainerStub(_make_config("2bit"))
    gen_output = _make_gen_output(
        question_ids=["q1", "q2", "q3", "q4"],
        doc_ids=["d1", "d1", "d2", "d2"],
    )
    rewards = {"q1": -2.0, "q2": -0.2, "q3": 0.2, "q4": 2.0}

    batch, metrics = SelfEvolutionTrainer.prepare_gen_update_batch(trainer, gen_output, rewards)

    assert batch is not None
    assert len(batch) == 4
    assert metrics["gen_quant/mode_id"] == 2.0
    total = (
        metrics["gen_quant/bucket/neg1"]
        + metrics["gen_quant/bucket/neg0p1"]
        + metrics["gen_quant/bucket/pos0p1"]
        + metrics["gen_quant/bucket/pos1"]
    )
    assert total == 4.0


def test_group_std_fixed_threshold_mode():
    cfg = _make_config("group_std_fixed_threshold")
    cfg.influence.group_std_tau = 0.4
    trainer = _TrainerStub(cfg)
    gen_output = _make_gen_output(
        question_ids=["q1", "q2", "q3", "q4", "q5", "q6"],
        doc_ids=["d1", "d1", "d1", "d2", "d2", "d2"],
    )
    rewards = {"q1": 0.0, "q2": 1.0, "q3": 2.0, "q4": 0.0, "q5": 0.0, "q6": 0.0}

    batch, metrics = SelfEvolutionTrainer.prepare_gen_update_batch(trainer, gen_output, rewards)

    assert batch is not None
    assert len(batch) == 3
    assert metrics["gen_quant/mode_id"] == 4.0
    assert metrics["gen_quant/tau"] == 0.4
    assert metrics["gen_quant/tau_quantile"] == -1.0
    assert metrics["gen_quant/survival/groups_kept"] == 1.0


def test_group_std_top_gamma_tau_max_clamp():
    # gamma=0.5 makes quantile tau_raw land between two std values, so
    # tau_max can clamp it.
    cfg = _make_config("group_std_top_gamma", gamma=0.5)
    cfg.influence.group_std_tau_max = 0.4
    trainer = _TrainerStub(cfg)
    gen_output = _make_gen_output(
        question_ids=["q1", "q2", "q3", "q4", "q5", "q6"],
        doc_ids=["d1", "d1", "d1", "d2", "d2", "d2"],
    )
    rewards = {"q1": 0.0, "q2": 1.0, "q3": 2.0, "q4": 0.0, "q5": 0.0, "q6": 0.0}

    batch, metrics = SelfEvolutionTrainer.prepare_gen_update_batch(trainer, gen_output, rewards)

    assert batch is not None
    assert metrics["gen_quant/mode_id"] == 3.0
    assert metrics["gen_quant/tau_raw"] > metrics["gen_quant/tau"]
    assert metrics["gen_quant/tau"] == 0.4
    assert metrics["gen_quant/tau_max"] == 0.4
    assert metrics["gen_quant/tau_was_clamped"] == 1.0


def test_invalid_penalty_zero_matches_legacy_group_std_filtering():
    """Regression: with invalid_penalty=0, refactor should match legacy behavior."""
    cfg = _make_config("group_std_top_gamma", gamma=0.5)
    cfg.training.gen_invalid_penalty = 0.0
    trainer = _TrainerStub(cfg)
    gen_output = _make_gen_output(
        question_ids=[
            "q1", "q2", "bad1",   # d1: [0.0, 2.0, invalid->0.0]
            "q3", "q4", "bad2",   # d2: [0.0, 0.0, invalid->0.0]
            "q5", "q6", "bad3",   # d3: [1.0, 1.4, invalid->0.0]
            "q7", "q8", "bad4",   # d4: [0.0, 0.3, invalid->0.0]
        ],
        doc_ids=[
            "d1", "d1", "d1",
            "d2", "d2", "d2",
            "d3", "d3", "d3",
            "d4", "d4", "d4",
        ],
    )
    rewards = {
        "q1": 0.0, "q2": 2.0,
        "q3": 0.0, "q4": 0.0,
        "q5": 1.0, "q6": 1.4,
        "q7": 0.0, "q8": 0.3,
    }

    def _legacy_reference():
        qids = gen_output.non_tensor_batch["question_id"]
        doc_ids = gen_output.non_tensor_batch["doc_id"]
        penalty = cfg.training.gen_invalid_penalty

        row_rewards = []
        row_uids = []
        for i in range(len(qids)):
            qid = qids[i]
            doc_id = str(doc_ids[i])
            if qid is not None and str(qid) in rewards:
                row_rewards.append(float(rewards[str(qid)]))
            else:
                row_rewards.append(float(penalty))
            row_uids.append(doc_id)

        doc_rewards: dict[str, list[float]] = {}
        for uid, r in zip(row_uids, row_rewards):
            doc_rewards.setdefault(uid, []).append(r)

        doc_stds = {uid: float(np.asarray(vals, dtype=np.float32).std()) for uid, vals in doc_rewards.items()}
        std_arr = np.asarray(list(doc_stds.values()), dtype=np.float32)
        tau_raw = float(np.quantile(std_arr, 1.0 - cfg.influence.group_std_gamma))
        tau = tau_raw
        surviving_docs = {uid for uid, s in doc_stds.items() if s > tau}

        selected_indices = [i for i, uid in enumerate(row_uids) if uid in surviving_docs]
        selected_rewards = [row_rewards[i] for i in selected_indices]
        return tau, surviving_docs, selected_rewards

    ref_tau, ref_surviving_docs, ref_selected_rewards = _legacy_reference()
    batch, metrics = SelfEvolutionTrainer.prepare_gen_update_batch(trainer, gen_output, rewards)

    assert batch is not None
    actual_surviving_docs = set(batch.non_tensor_batch["uid"].tolist())
    assert actual_surviving_docs == ref_surviving_docs
    assert metrics["gen_quant/tau"] == pytest.approx(ref_tau, abs=1e-6)

    # Since group_std_top_gamma does not do value quantization, the scalar
    # reward per selected row should match the legacy selected rewards.
    actual_selected_rewards = batch.batch["token_level_scores"].sum(dim=-1).cpu().tolist()
    assert actual_selected_rewards == pytest.approx(ref_selected_rewards, abs=1e-6)


def test_reward_structure_builds_weighted_group_tensors():
    cfg = OmegaConf.create(
        {
            "training": {
                "gen_invalid_penalty": -2.0,
                "generator_reward_structure": [
                    {
                        "group_weight": 1.0,
                        "terms": [
                            {"name": "influence_rewards", "weight": 2.0},
                        ],
                    },
                    {
                        "group_weight": 3.0,
                        "terms": [
                            {"name": "spice_rewards", "weight": 0.5},
                            {"name": "invalid_rewards", "weight": 1.5},
                        ],
                    },
                ],
            },
            "influence": {
                "quantification_mode": None,
                "group_std_gamma": 0.2,
            },
            "generator": {"rollout": {"temperature": 0.7}},
        }
    )
    trainer = _TrainerStub(cfg)
    gen_output = _make_gen_output(
        question_ids=["q1", "q2", None],
        doc_ids=["d1", "d1", "d1"],
    )
    reward_structure = resolve_generator_reward_structure(cfg.training)
    rewards_payload = build_stage4_reward_payload(
        valid_question_ids={"q1", "q2"},
        influence_rewards={"q1": 1.0, "q2": 3.0},
        spice_rewards={"q1": 10.0, "q2": 20.0},
        selected_components=resolve_generator_reward_components(cfg.training),
        reward_structure=reward_structure,
    )

    batch, metrics = SelfEvolutionTrainer.prepare_gen_update_batch(
        trainer,
        gen_output,
        rewards_payload,
    )

    assert batch is not None
    assert metrics["gen_quant/decoupled_adv_enabled"] == 1.0
    assert batch.meta_info["reward_structure_for_adv"] == reward_structure
    assert batch.batch["token_level_scores"].sum(dim=-1).cpu().tolist() == pytest.approx(
        [7.0, 16.0, -3.0],
        abs=1e-6,
    )
    assert batch.batch["token_level_scores_group_0"].sum(dim=-1).cpu().tolist() == pytest.approx(
        [2.0, 6.0, 0.0],
        abs=1e-6,
    )
    assert batch.batch["token_level_scores_group_1"].sum(dim=-1).cpu().tolist() == pytest.approx(
        [5.0, 10.0, -3.0],
        abs=1e-6,
    )


def test_reward_structure_drives_generator_advantage_end_to_end():
    cfg = OmegaConf.create(
        {
            "training": {
                "gen_invalid_penalty": -2.0,
                "generator_reward_structure": [
                    {
                        "group_weight": 1.0,
                        "terms": [
                            {"name": "influence_rewards", "weight": 2.0},
                        ],
                    },
                    {
                        "group_weight": 3.0,
                        "terms": [
                            {"name": "spice_rewards", "weight": 0.5},
                            {"name": "invalid_rewards", "weight": 1.5},
                        ],
                    },
                ],
            },
            "influence": {
                "quantification_mode": None,
                "group_std_gamma": 0.2,
                "normalization_mode": "group_std",
            },
            "generator": {"rollout": {"temperature": 0.7}},
            "algorithm": {
                "adv_estimator": "grpo",
                "gamma": 1.0,
                "lam": 1.0,
            },
        }
    )
    trainer = _TrainerStub(cfg)
    gen_output = _make_gen_output(
        question_ids=["q1", "q2", None],
        doc_ids=["d1", "d1", "d1"],
    )
    reward_structure = resolve_generator_reward_structure(cfg.training)
    rewards_payload = build_stage4_reward_payload(
        valid_question_ids={"q1", "q2"},
        influence_rewards={"q1": 1.0, "q2": 3.0},
        spice_rewards={"q1": 10.0, "q2": 20.0},
        selected_components=resolve_generator_reward_components(cfg.training),
        reward_structure=reward_structure,
    )

    batch, _ = SelfEvolutionTrainer.prepare_gen_update_batch(
        trainer,
        gen_output,
        rewards_payload,
    )

    assert batch is not None
    response_mask = batch.batch["response_mask"].clone()
    uids = batch.non_tensor_batch["uid"].copy()
    group0_rewards = batch.batch["token_level_scores_group_0"].clone()
    group1_rewards = batch.batch["token_level_scores_group_1"].clone()

    batch, metrics = SelfEvolutionTrainer.compute_generator_advantage(
        trainer,
        batch,
        normalization_mode="group_std",
    )

    group0_batch = trainer._build_advantage_component_batch(
        token_level_rewards=group0_rewards,
        response_mask=response_mask,
        uids=uids,
    )
    group0_batch = trainer.compute_advantage(
        group0_batch,
        normalization_mode="group_std",
    )
    group1_batch = trainer._build_advantage_component_batch(
        token_level_rewards=group1_rewards,
        response_mask=response_mask,
        uids=uids,
    )
    group1_batch = trainer.compute_advantage(
        group1_batch,
        normalization_mode="group_std",
    )

    expected_advantages = (
        group0_batch.batch["advantages"].clone()
        + 3.0 * group1_batch.batch["advantages"]
    )
    expected_returns = (
        group0_batch.batch["returns"].clone()
        + 3.0 * group1_batch.batch["returns"]
    )
    expected_sample_adv = (
        (expected_advantages * response_mask).sum(dim=-1)
        / response_mask.sum(dim=-1).clamp(min=1)
    )
    if expected_sample_adv.shape[0] > 1:
        expected_sample_adv = (
            expected_sample_adv - expected_sample_adv.mean()
        ) / (expected_sample_adv.std() + 1e-8)
    expected_advantages = expected_sample_adv.unsqueeze(-1) * response_mask

    assert torch.allclose(
        batch.batch["advantages"],
        expected_advantages,
        atol=1e-6,
    )
    assert torch.allclose(
        batch.batch["returns"],
        expected_returns,
        atol=1e-6,
    )
    assert metrics["gen_adv/decoupled_enabled"] == 1.0
    assert metrics["gen_adv/decoupled_active"] == 1.0
    assert metrics["gen_adv/decoupled_group/count"] == 2.0
    assert metrics["gen_adv/decoupled_component/count"] == 3.0
    assert metrics["gen_adv/decoupled_group/2x_influence_weight"] == 1.0
    assert metrics["gen_adv/decoupled_group/0.5x_spice+1.5x_invalid_weight"] == 3.0
    assert metrics["gen_adv/decoupled_component/influence_weight"] == 2.0
    assert metrics["gen_adv/decoupled_component/spice_weight"] == 0.5
    assert metrics["gen_adv/decoupled_component/invalid_weight"] == 1.5


def test_legacy_decoupled_groups_still_drive_generator_advantage():
    cfg = OmegaConf.create(
        {
            "training": {
                "gen_invalid_penalty": -2.0,
                "generator_reward_components": [
                    "influence_rewards",
                    "spice_rewards",
                    "invalid_rewards",
                ],
                "generator_reward_combination_mode": "decoupled",
                "generator_reward_groups": [
                    ["influence_rewards"],
                    ["spice_rewards", "invalid_rewards"],
                ],
                "generator_reward_group_weights": [1.0, 3.0],
            },
            "influence": {
                "quantification_mode": None,
                "group_std_gamma": 0.2,
                "normalization_mode": "group_std",
            },
            "generator": {"rollout": {"temperature": 0.7}},
            "algorithm": {
                "adv_estimator": "grpo",
                "gamma": 1.0,
                "lam": 1.0,
            },
        }
    )
    trainer = _TrainerStub(cfg)
    gen_output = _make_gen_output(
        question_ids=["q1", "q2", None],
        doc_ids=["d1", "d1", "d1"],
    )
    rewards_payload = build_stage4_reward_payload(
        valid_question_ids={"q1", "q2"},
        influence_rewards={"q1": 1.0, "q2": 3.0},
        spice_rewards={"q1": 10.0, "q2": 20.0},
        selected_components=resolve_generator_reward_components(cfg.training),
    )

    batch, _ = SelfEvolutionTrainer.prepare_gen_update_batch(
        trainer,
        gen_output,
        rewards_payload,
    )

    assert batch is not None
    assert batch.batch["token_level_scores_group_0"].sum(dim=-1).cpu().tolist() == pytest.approx(
        [1.0, 3.0, 0.0],
        abs=1e-6,
    )
    assert batch.batch["token_level_scores_group_1"].sum(dim=-1).cpu().tolist() == pytest.approx(
        [10.0, 20.0, -2.0],
        abs=1e-6,
    )

    batch, metrics = SelfEvolutionTrainer.compute_generator_advantage(
        trainer,
        batch,
        normalization_mode="group_std",
    )

    assert batch.batch["advantages"].shape == batch.batch["response_mask"].shape
    assert metrics["gen_adv/decoupled_enabled"] == 1.0
    assert metrics["gen_adv/decoupled_active"] == 1.0
    assert metrics["gen_adv/decoupled_group/count"] == 2.0
    assert metrics["gen_adv/decoupled_group/influence_weight"] == 1.0
    assert metrics["gen_adv/decoupled_group/spice+invalid_weight"] == 3.0
