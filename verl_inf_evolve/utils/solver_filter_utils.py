"""Helpers for solver training-data filtering modes."""

from __future__ import annotations

import math
import re
from typing import Any


def normalize_solver_filter_mode(mode_value: Any) -> str:
    """Normalize solver filter mode string and validate accepted values."""
    mode = str(mode_value).strip().lower().replace("-", "_").replace(" ", "_")
    mode = mode.replace("__", "_")
    aliases = {
        "randomone": "random_one",
        "topone": "top_one",
        "random_1": "random_one",
        "top_1": "top_one",
        "topalpha": "top_alpha",
        "randomalpha": "random_alpha",
    }
    mode = aliases.get(mode, mode)
    valid_modes = {"none", "random_one", "top_one", "top_alpha", "random_alpha"}
    dynamic_mode_re = re.compile(r"^(top|random)_(\d+)$")
    if mode not in valid_modes and dynamic_mode_re.match(mode) is None:
        raise ValueError(
            "Invalid influence.solver_filter_mode: "
            f"{mode_value!r}. Must be one of {sorted(valid_modes)} or "
            "'top_<i>'/'random_<i>' with i>=1."
        )
    match = dynamic_mode_re.match(mode)
    if match is not None and int(match.group(2)) < 1:
        raise ValueError(
            "Invalid influence.solver_filter_mode: "
            f"{mode_value!r}. In 'top_<i>'/'random_<i>', i must be >= 1."
        )
    return mode


def alpha_target_count(total_questions: int, alpha: float) -> int:
    """Convert alpha * total_questions threshold to an integer keep count."""
    if total_questions <= 0 or alpha <= 0.0:
        return 0
    return int(math.ceil(alpha * total_questions))


def get_per_doc_keep_count(mode: str) -> int | None:
    """Return per-document keep count for top/random modes, else None."""
    normalized_mode = normalize_solver_filter_mode(mode)
    if normalized_mode == "top_one" or normalized_mode == "random_one":
        return 1
    match = re.match(r"^(top|random)_(\d+)$", normalized_mode)
    if match is None:
        return None
    return int(match.group(2))
