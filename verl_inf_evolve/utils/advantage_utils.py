"""Advantage-related utility helpers."""

from __future__ import annotations

import torch


def scalarize_masked_sequence_values(
    values: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Reduce token-level values to one scalar per response."""
    denom = response_mask.sum(dim=-1).clamp_min(1.0)
    return (values * response_mask).sum(dim=-1) / denom
