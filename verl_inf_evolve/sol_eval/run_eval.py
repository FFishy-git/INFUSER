"""Deprecated CLI entry point — use the Hydra-based sol_eval.py instead.

Usage::

    python -m verl_inf_evolve.sol_eval.sol_eval sol_eval_experiment=debug-run
    python -m verl_inf_evolve.sol_eval.sol_eval eval.checkpoints=[0,5,10]
"""

from __future__ import annotations

import sys


def main() -> None:
    """Redirect to the Hydra-based entry point."""
    print(
        "run_eval.py is deprecated. Use the Hydra entry point instead:\n"
        "  python -m verl_inf_evolve.sol_eval.sol_eval sol_eval_experiment=debug-run",
        file=sys.stderr,
    )
    from verl_inf_evolve.sol_eval.sol_eval import main as sol_eval_main

    sol_eval_main()


if __name__ == "__main__":
    main()
