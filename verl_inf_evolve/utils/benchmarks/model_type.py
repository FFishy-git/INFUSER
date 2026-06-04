"""Model type detection: base vs instruct/chat.

This is shared with the OpenCompass launcher so prompt selection stays
consistent across ``sol_eval`` and the standalone OpenCompass entry point.
"""

from __future__ import annotations

from verl_inf_evolve.sol_eval.opencompass_runner import detect_hf_type


def detect_model_type(model_path: str) -> str:
    """Return ``"base"`` or ``"chat"`` for *model_path*.
    """
    return detect_hf_type(model_path)


def is_base_model(model_path: str) -> bool:
    """Convenience: ``True`` when *model_path* is detected as a base model."""
    return detect_model_type(model_path) == "base"
