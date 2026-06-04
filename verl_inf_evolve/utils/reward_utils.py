"""Reward / score tensor utilities for training batch preparation."""

from __future__ import annotations

import torch


def expand_scores_to_token_level(
    scores: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """Place per-sample scalar reward at EOS token only.

    Mirrors verl's ``ActorRolloutRefWorker._expand_to_token_level``
    (fsdp_workers.py:1908-1923) so that ``sum(dim=-1)`` in
    ``compute_grpo_outcome_advantage`` recovers the original scalar
    without length-dependent scaling.
    """
    batch_size = scores.shape[0]
    pos = position_ids
    if pos.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
        pos = pos[:, 0, :]
    eos_mask_idx = torch.argmax(pos * attention_mask, dim=-1)  # (batch_size,)
    token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
    token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores
    token_level_scores = token_level_scores[:, -response_length:]
    return token_level_scores
