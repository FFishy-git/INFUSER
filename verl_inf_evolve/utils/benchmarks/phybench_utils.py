"""Shared helpers for PHYBench preprocessing, verification, and EED scoring."""

from __future__ import annotations

import re


_OUTER_MATH_PATTERNS = [
    re.compile(r"^\s*\\\[(.*)\\\]\s*$", re.DOTALL),
    re.compile(r"^\s*\\\((.*)\\\)\s*$", re.DOTALL),
    re.compile(r"^\s*\$(.*)\$\s*$", re.DOTALL),
]


def normalize_phybench_expression(text: str | None) -> str:
    """Strip common display-math wrappers while preserving the inner LaTeX."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    changed = True
    while changed:
        changed = False
        for pattern in _OUTER_MATH_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                cleaned = match.group(1).strip()
                changed = True
                break

    return cleaned

