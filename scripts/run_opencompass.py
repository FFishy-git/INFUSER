#!/usr/bin/env python3
"""Thin CLI wrapper around the reusable OpenCompass launcher module."""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from verl_inf_evolve.sol_eval.opencompass_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
