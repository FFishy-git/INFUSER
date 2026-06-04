from __future__ import annotations

from unittest.mock import patch

import torch

from verl_inf_evolve.utils.influence_utils import (
    add_similarity_metric_stats,
    build_similarity_rewards,
)
from verl_inf_evolve.workers.similarity_optimizer import SimilarityComputingOptimizer


class _SingleParamModule(torch.nn.Module):
    def __init__(self, values: list[float]):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))


def test_similarity_optimizer_preconditioned_dot_uses_warm_state() -> None:
    module = _SingleParamModule([0.0, 0.0])
    grad = torch.tensor([2.0, -4.0], dtype=torch.float32)
    ref = torch.tensor([3.0, 5.0], dtype=torch.float32)
    module.weight.grad = grad.clone()

    beta2 = 0.9
    eps = 0.2
    step_prev = 6
    v_prev = torch.tensor([4.0, 9.0], dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=1e-3,
        betas=(0.8, beta2),
        eps=eps,
    )
    optimizer.state[module.weight]["exp_avg_sq"] = v_prev.clone()
    optimizer.state[module.weight]["step"] = torch.tensor(float(step_prev))

    sim_opt = SimilarityComputingOptimizer(
        fsdp_model=module,
        ref_gradients=[ref],
        actor_optimizer=optimizer,
        similarity_mode="preconditioned_dot",
    )
    sim_opt.step()
    metrics = sim_opt.get_concatenated_metrics()

    bias_correction2 = 1.0 - beta2 ** (step_prev + 1)
    expected_gamma = grad / (
        torch.sqrt((beta2 * v_prev + (1.0 - beta2) * grad.square()) / bias_correction2)
        + eps
    )
    expected_dot = torch.dot(ref, expected_gamma)
    expected_cosine = expected_dot / (ref.norm() * expected_gamma.norm())

    assert metrics["score_mode"] == "preconditioned_dot"
    assert "cosine" not in metrics
    torch.testing.assert_close(metrics["score"], expected_dot.unsqueeze(0))
    torch.testing.assert_close(metrics["grad_norm"], grad.norm().unsqueeze(0))
    torch.testing.assert_close(metrics["gamma_norm"], expected_gamma.norm().unsqueeze(0))
    torch.testing.assert_close(metrics["preconditioned_dot"], expected_dot.unsqueeze(0))
    torch.testing.assert_close(
        metrics["preconditioned_cosine"],
        expected_cosine.unsqueeze(0),
    )
    torch.testing.assert_close(metrics["ref_norm"], ref.norm())


def test_similarity_optimizer_preconditioned_dot_matches_first_step_adamw() -> None:
    module = _SingleParamModule([0.0, 0.0, 0.0])
    grad = torch.tensor([2.0, -3.0, 0.5], dtype=torch.float32)
    ref = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float32)
    module.weight.grad = grad.clone()

    eps = 0.1
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=eps,
    )

    sim_opt = SimilarityComputingOptimizer(
        fsdp_model=module,
        ref_gradients=[ref],
        actor_optimizer=optimizer,
        similarity_mode="preconditioned_dot",
    )
    sim_opt.step()
    metrics = sim_opt.get_concatenated_metrics()

    expected_gamma = grad / (grad.abs() + eps)
    expected_dot = torch.dot(ref, expected_gamma)
    expected_cosine = expected_dot / (ref.norm() * expected_gamma.norm())

    torch.testing.assert_close(metrics["score"], expected_dot.unsqueeze(0))
    torch.testing.assert_close(metrics["gamma_norm"], expected_gamma.norm().unsqueeze(0))
    torch.testing.assert_close(metrics["preconditioned_dot"], expected_dot.unsqueeze(0))
    torch.testing.assert_close(
        metrics["preconditioned_cosine"],
        expected_cosine.unsqueeze(0),
    )


def test_similarity_optimizer_cosine_mode_is_unchanged() -> None:
    module = _SingleParamModule([0.0, 0.0])
    grad = torch.tensor([3.0, 4.0], dtype=torch.float32)
    ref = torch.tensor([4.0, 3.0], dtype=torch.float32)
    module.weight.grad = grad.clone()

    sim_opt = SimilarityComputingOptimizer(
        fsdp_model=module,
        ref_gradients=[ref],
        similarity_mode="cosine",
    )
    sim_opt.step()
    metrics = sim_opt.get_concatenated_metrics()

    expected_dot = torch.dot(grad, ref)
    expected_ref_norm = ref.norm()
    expected_grad_norm = grad.norm()
    expected_cosine = expected_dot / (expected_grad_norm * expected_ref_norm)

    assert metrics["score_mode"] == "cosine"
    torch.testing.assert_close(metrics["score"], expected_cosine.unsqueeze(0))
    torch.testing.assert_close(metrics["cosine"], expected_cosine.unsqueeze(0))
    torch.testing.assert_close(metrics["dot"], expected_dot.unsqueeze(0))
    torch.testing.assert_close(metrics["grad_norm"], expected_grad_norm.unsqueeze(0))


def test_similarity_optimizer_dot_mode_keeps_cosine_available() -> None:
    module = _SingleParamModule([0.0, 0.0])
    grad = torch.tensor([1.0, 2.0], dtype=torch.float32)
    ref = torch.tensor([2.0, 1.0], dtype=torch.float32)
    module.weight.grad = grad.clone()

    sim_opt = SimilarityComputingOptimizer(
        fsdp_model=module,
        ref_gradients=[ref],
        similarity_mode="dot",
    )
    sim_opt.step()
    metrics = sim_opt.get_concatenated_metrics()

    expected_dot = torch.dot(grad, ref)
    expected_cosine = expected_dot / (grad.norm() * ref.norm())

    assert metrics["score_mode"] == "dot"
    torch.testing.assert_close(metrics["score"], expected_dot.unsqueeze(0))
    torch.testing.assert_close(metrics["dot"], expected_dot.unsqueeze(0))
    torch.testing.assert_close(metrics["cosine"], expected_cosine.unsqueeze(0))


def test_similarity_optimizer_preconditioned_cosine_keeps_dot_available() -> None:
    module = _SingleParamModule([0.0, 0.0])
    grad = torch.tensor([2.0, -1.0], dtype=torch.float32)
    ref = torch.tensor([1.0, 3.0], dtype=torch.float32)
    module.weight.grad = grad.clone()

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=1e-3,
        betas=(0.9, 0.8),
        eps=0.05,
    )

    sim_opt = SimilarityComputingOptimizer(
        fsdp_model=module,
        ref_gradients=[ref],
        actor_optimizer=optimizer,
        similarity_mode="preconditioned_cosine",
    )
    sim_opt.step()
    metrics = sim_opt.get_concatenated_metrics()

    expected_gamma = grad / (grad.abs() + 0.05)
    expected_dot = torch.dot(ref, expected_gamma)
    expected_cosine = expected_dot / (ref.norm() * expected_gamma.norm())

    assert metrics["score_mode"] == "preconditioned_cosine"
    torch.testing.assert_close(metrics["score"], expected_cosine.unsqueeze(0))
    torch.testing.assert_close(metrics["preconditioned_dot"], expected_dot.unsqueeze(0))
    torch.testing.assert_close(
        metrics["preconditioned_cosine"],
        expected_cosine.unsqueeze(0),
    )


def test_similarity_optimizer_cosine_mode_fuses_scalar_collectives() -> None:
    module = _SingleParamModule([0.0, 0.0])
    grad = torch.tensor([3.0, 4.0], dtype=torch.float32)
    ref = torch.tensor([4.0, 3.0], dtype=torch.float32)
    module.weight.grad = grad.clone()

    sim_opt = SimilarityComputingOptimizer(
        fsdp_model=module,
        ref_gradients=[ref],
        ref_norm=ref.norm(),
        similarity_mode="cosine",
    )

    reduce_calls: list[torch.Tensor] = []

    def fake_all_reduce(tensor: torch.Tensor, op=None, group=None) -> None:
        reduce_calls.append(tensor.clone())

    with (
        patch("verl_inf_evolve.workers.similarity_optimizer.dist.is_available", return_value=True),
        patch("verl_inf_evolve.workers.similarity_optimizer.dist.is_initialized", return_value=True),
        patch("verl_inf_evolve.workers.similarity_optimizer.dist.all_reduce", side_effect=fake_all_reduce),
    ):
        sim_opt.step()

    assert len(reduce_calls) == 1
    assert reduce_calls[0].shape == torch.Size([2])


def test_similarity_optimizer_preconditioned_cosine_fuses_scalar_collectives() -> None:
    module = _SingleParamModule([0.0, 0.0])
    grad = torch.tensor([2.0, -1.0], dtype=torch.float32)
    ref = torch.tensor([1.0, 3.0], dtype=torch.float32)
    module.weight.grad = grad.clone()

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=1e-3,
        betas=(0.9, 0.8),
        eps=0.05,
    )
    sim_opt = SimilarityComputingOptimizer(
        fsdp_model=module,
        ref_gradients=[ref],
        actor_optimizer=optimizer,
        ref_norm=ref.norm(),
        similarity_mode="preconditioned_cosine",
    )

    reduce_calls: list[torch.Tensor] = []

    def fake_all_reduce(tensor: torch.Tensor, op=None, group=None) -> None:
        reduce_calls.append(tensor.clone())

    with (
        patch("verl_inf_evolve.workers.similarity_optimizer.dist.is_available", return_value=True),
        patch("verl_inf_evolve.workers.similarity_optimizer.dist.is_initialized", return_value=True),
        patch("verl_inf_evolve.workers.similarity_optimizer.dist.all_reduce", side_effect=fake_all_reduce),
    ):
        sim_opt.step()

    assert len(reduce_calls) == 1
    assert reduce_calls[0].shape == torch.Size([3])


def test_similarity_reward_helpers_use_score_field() -> None:
    similarity_metrics = {
        "score_mode": "preconditioned_dot",
        "score": [1.25, -0.75],
        "grad_norm": [2.0, 4.0],
        "gamma_norm": [0.5, 1.5],
        "preconditioned_dot": [1.25, -0.75],
        "preconditioned_cosine": [0.25, -0.15],
        "ref_norm": 3.0,
    }

    rewards, scores, score_mode = build_similarity_rewards(
        question_order=["q1", "q2"],
        all_question_ids={"q1", "q2", "q3"},
        similarity_metrics=similarity_metrics,
    )

    assert score_mode == "preconditioned_dot"
    assert scores == [1.25, -0.75]
    assert rewards == {"q1": 1.25, "q2": -0.75, "q3": 0.0}

    logged_metrics: dict[str, float] = {}
    returned_mode = add_similarity_metric_stats(logged_metrics, similarity_metrics)
    assert returned_mode == "preconditioned_dot"
    assert logged_metrics["influence_sim/ref_norm"] == 3.0
    assert "influence_sim/score/mean" in logged_metrics
    assert "influence_sim/gamma_norm/mean" in logged_metrics
    assert "influence_sim/preconditioned_dot/mean" in logged_metrics
    assert "influence_sim/preconditioned_cosine/mean" in logged_metrics
    assert "influence_sim/cosine/mean" not in logged_metrics
