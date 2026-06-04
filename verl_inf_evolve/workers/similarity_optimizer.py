"""
SimilarityComputingOptimizer — gradient similarity computation via optimizer hook.

Ported from V2 ``verl_joint_dev_similarity.py:73-242``.

This is a custom "optimizer" that replaces ``optimizer.step()`` with similarity
computation against a reference gradient (momentum).  When verl's
``DataParallelPPOActor.update_policy()`` calls ``optimizer.step()``, this class
computes dot products, gradient norms, and cosine similarity instead of
updating weights.

Key adaptations for V3:
- Directly references the FSDP model and momentum (no disk I/O).
- Uses the process group established by verl's Ray worker.
- ``_compute_similarity_from_grads()`` is inlined into this class.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.distributed as dist


class SimilarityComputingOptimizer:
    """Optimizer that computes gradient similarity instead of stepping.

    Hooks into the gradient accumulation flow via verl's actor training loop:
    - ``zero_grad()`` is called once per mini-batch (before micro-batches).
    - ``step()`` is called once per mini-batch (after all micro-batches).
    - Each ``step()`` produces one set of similarity metrics.

    After all mini-batches, ``get_concatenated_metrics()`` stacks all metrics.

    Args:
        fsdp_model: The FSDP-wrapped model whose ``.parameters()`` provide
            gradients for similarity computation.
        ref_gradients: List of reference gradient tensors (one per FSDP
            parameter shard).  Typically the momentum buffer.
        actor_optimizer: Live AdamW optimizer used to read second-moment state
            for optimizer-aware similarity modes.
        ref_norm: Pre-computed global L2 norm of the reference gradients.
            If ``None``, will be computed on first ``step()`` call.
        process_group: Distributed process group for all-reduce.
            Defaults to the FSDP process group (``dist.group.WORLD``).
        similarity_mode: Scoring mode. Supported values are ``"dot"``,
            ``"cosine"``, ``"preconditioned_dot"``, and
            ``"preconditioned_cosine"``.
    """

    def __init__(
        self,
        fsdp_model: torch.nn.Module,
        ref_gradients: list[torch.Tensor | None],
        actor_optimizer: torch.optim.Optimizer | None = None,
        ref_norm: torch.Tensor | None = None,
        process_group: Any = None,
        similarity_mode: str = "cosine",
    ):
        self.fsdp_model = fsdp_model
        self.ref_gradients = ref_gradients
        self.actor_optimizer = actor_optimizer
        self.ref_norm = ref_norm
        self.process_group = process_group
        self.similarity_mode = similarity_mode

        self.similarity_metrics_list: list[dict[str, torch.Tensor]] = []
        self.current_minibatch_idx = 0
        self._param_group_lookup = self._build_param_group_lookup(actor_optimizer)

    # ------------------------------------------------------------------
    # Optimizer interface (called by DataParallelPPOActor)
    # ------------------------------------------------------------------

    def zero_grad(self):
        """Zero out gradients on all model parameters.

        Called once per mini-batch before micro-batch processing begins.
        """
        for p in self.fsdp_model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def step(self) -> float:
        """Compute similarity from accumulated gradients.

        Called once per mini-batch after all micro-batches have been
        processed and gradients are accumulated in ``p.grad``.

        Returns:
            0.0 as a dummy gradient norm for compatibility.
        """
        metrics = self._compute_similarity_from_grads()
        self.similarity_metrics_list.append(metrics)
        self.current_minibatch_idx += 1
        return 0.0

    def reset(self):
        """Reset accumulated metrics for a new computation round."""
        self.similarity_metrics_list = []
        self.current_minibatch_idx = 0

    def state_dict(self) -> dict:
        """Return empty state dict (no persistent state)."""
        return {}

    def load_state_dict(self, state_dict: dict):
        """No-op (similarity optimizer has no persistent state)."""
        pass

    # ------------------------------------------------------------------
    # Metrics retrieval
    # ------------------------------------------------------------------

    def get_concatenated_metrics(self) -> dict[str, torch.Tensor | int | str]:
        """Stack similarity metrics across all mini-batches.

        Returns:
            Dict with:
            - ``score``: ``[num_minibatches]`` reward-driving similarity values.
            - ``grad_norm``: ``[num_minibatches]`` gradient norms.
            - ``ref_norm``: Scalar reference norm (same for all).
            - ``score_mode``: Name of the active score mode.
            - optional family metrics such as ``dot``/``cosine`` or
              ``preconditioned_dot``/``preconditioned_cosine`` depending on
              the active mode.
            - ``num_minibatches``: Number of mini-batches processed.

        Raises:
            RuntimeError: If ``step()`` was never called.
        """
        if not self.similarity_metrics_list:
            raise RuntimeError("No similarity metrics computed. Was step() called?")

        first_metrics = self.similarity_metrics_list[0]
        output: dict[str, torch.Tensor | int | str] = {
            "score": torch.stack(
                [m["score"] for m in self.similarity_metrics_list], dim=0
            ),
            "grad_norm": torch.stack(
                [m["grad_norm"] for m in self.similarity_metrics_list], dim=0
            ),
            "ref_norm": first_metrics["ref_norm"],
            "num_minibatches": len(self.similarity_metrics_list),
            "score_mode": self.similarity_mode,
        }
        for key in (
            "dot",
            "cosine",
            "gamma_norm",
            "preconditioned_dot",
            "preconditioned_cosine",
        ):
            if key in first_metrics:
                output[key] = torch.stack(
                    [m[key] for m in self.similarity_metrics_list], dim=0
                )
        return output

    # ------------------------------------------------------------------
    # Core similarity computation
    # ------------------------------------------------------------------

    def _compute_similarity_from_grads(self) -> dict[str, torch.Tensor]:
        """Compute similarity between accumulated gradients and reference.

        Reads gradients from ``p.grad`` for all FSDP parameters, computes
        dot product and L2 norms locally, then all-reduces across ranks
        to get global metrics.

        Returns:
            Dict with scalar tensors for the selected score family.
        """
        if self.similarity_mode in {"dot", "cosine"}:
            return self._compute_raw_similarity_from_grads()
        if self.similarity_mode in {"preconditioned_dot", "preconditioned_cosine"}:
            return self._compute_preconditioned_similarity_from_grads()
        raise ValueError(f"Unsupported similarity_mode: {self.similarity_mode!r}")

    def _compute_raw_similarity_from_grads(self) -> dict[str, torch.Tensor]:
        params = list(self.fsdp_model.parameters())
        if self.ref_norm is None:
            self.ref_norm = self._compute_ref_norm()

        device = self._infer_device(params)
        local_dot = torch.zeros((), device=device, dtype=torch.float32)
        local_norm_sq = torch.zeros((), device=device, dtype=torch.float32)

        for p, ref_g in zip(params, self.ref_gradients):
            if p.grad is None or ref_g is None:
                continue

            grad = p.grad.float().view(-1)
            ref = ref_g.float().view(-1)

            local_dot += (grad * ref).sum()
            local_norm_sq += (grad * grad).sum()

        local_dot, local_norm_sq = self._maybe_all_reduce_scalars(
            local_dot,
            local_norm_sq,
        )

        grad_norm = local_norm_sq.sqrt().clamp_min(1e-12)
        cosine = local_dot / (grad_norm * self.ref_norm)
        score = local_dot if self.similarity_mode == "dot" else cosine

        return {
            "score": score,
            "dot": local_dot,
            "grad_norm": grad_norm,
            "ref_norm": self.ref_norm,
            "cosine": cosine,
        }

    def _compute_preconditioned_similarity_from_grads(self) -> dict[str, torch.Tensor]:
        if self.actor_optimizer is None:
            raise RuntimeError(
                f"similarity_mode={self.similarity_mode!r} requires actor_optimizer."
            )

        params = list(self.fsdp_model.parameters())
        if self.ref_norm is None:
            self.ref_norm = self._compute_ref_norm()

        device = self._infer_device(params)
        local_score = torch.zeros((), device=device, dtype=torch.float32)
        local_grad_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
        local_gamma_norm_sq = torch.zeros((), device=device, dtype=torch.float32)

        for p, ref_g in zip(params, self.ref_gradients):
            if p.grad is None or ref_g is None:
                continue

            grad = p.grad.float()
            ref = ref_g.float()
            gamma = self._compute_preconditioned_grad(p, grad)

            local_score += (gamma.view(-1) * ref.view(-1)).sum()
            local_grad_norm_sq += (grad.view(-1) ** 2).sum()
            local_gamma_norm_sq += (gamma.view(-1) ** 2).sum()

        local_score, local_grad_norm_sq, local_gamma_norm_sq = (
            self._maybe_all_reduce_scalars(
                local_score,
                local_grad_norm_sq,
                local_gamma_norm_sq,
            )
        )

        grad_norm = local_grad_norm_sq.sqrt().clamp_min(1e-12)
        gamma_norm = local_gamma_norm_sq.sqrt().clamp_min(1e-12)
        preconditioned_cosine = local_score / (self.ref_norm * gamma_norm)
        score = (
            local_score
            if self.similarity_mode == "preconditioned_dot"
            else preconditioned_cosine
        )

        return {
            "score": score,
            "grad_norm": grad_norm,
            "ref_norm": self.ref_norm,
            "gamma_norm": gamma_norm,
            "preconditioned_dot": local_score,
            "preconditioned_cosine": preconditioned_cosine,
        }

    def _compute_ref_norm(self) -> torch.Tensor:
        """Compute global L2 norm of reference gradients via all-reduce.

        Returns:
            Scalar tensor with the global reference gradient norm.
        """
        local_norm_sq = torch.zeros((), device=self._infer_device(), dtype=torch.float32)
        for g in self.ref_gradients:
            if g is not None:
                local_norm_sq += (g.float().view(-1) ** 2).sum()

        (local_norm_sq,) = self._maybe_all_reduce_scalars(local_norm_sq)
        return local_norm_sq.sqrt().clamp_min(1e-12)

    def _compute_preconditioned_grad(
        self, param: torch.nn.Parameter, grad: torch.Tensor
    ) -> torch.Tensor:
        group = self._param_group_lookup.get(param)
        if group is None:
            raise RuntimeError("Optimizer group lookup missing for model parameter.")

        beta2 = float(group["betas"][1])
        eps = float(group["eps"])
        state = self.actor_optimizer.state.get(param, {})

        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg_sq is None:
            exp_avg_sq = torch.zeros_like(grad, dtype=torch.float32)
        else:
            exp_avg_sq = exp_avg_sq.float()

        step_prev = state.get("step", 0)
        if torch.is_tensor(step_prev):
            step_prev_value = int(step_prev.detach().item())
        else:
            step_prev_value = int(step_prev)
        t = step_prev_value + 1
        bias_correction2 = 1.0 - (beta2**t)
        if bias_correction2 <= 0:
            raise RuntimeError(
                f"Invalid AdamW bias correction for beta2={beta2} and t={t}."
            )

        v_next = beta2 * exp_avg_sq + (1.0 - beta2) * grad.square()
        denom = (v_next / bias_correction2).sqrt() + eps
        return grad / denom

    @staticmethod
    def _build_param_group_lookup(
        actor_optimizer: torch.optim.Optimizer | None,
    ) -> dict[torch.nn.Parameter, dict[str, Any]]:
        if actor_optimizer is None:
            return {}

        lookup: dict[torch.nn.Parameter, dict[str, Any]] = {}
        for group in actor_optimizer.param_groups:
            for param in group["params"]:
                lookup[param] = group
        return lookup

    def _infer_device(
        self, params: Iterable[torch.nn.Parameter] | None = None
    ) -> torch.device:
        if params is not None:
            for param in params:
                return param.device
        for ref_grad in self.ref_gradients:
            if ref_grad is not None:
                return ref_grad.device
        return torch.device("cpu")

    def _maybe_all_reduce(self, tensor: torch.Tensor) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.process_group)

    def _maybe_all_reduce_scalars(
        self, *tensors: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """All-reduce scalar tensors with a single packed collective."""
        if not tensors:
            return ()
        if dist.is_available() and dist.is_initialized():
            packed = torch.stack([tensor.reshape(()) for tensor in tensors])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self.process_group)
            return tuple(packed[idx] for idx in range(len(tensors)))
        return tensors
