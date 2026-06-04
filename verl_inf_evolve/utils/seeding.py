"""Global seed fixture for reproducible RL training.

The training/sol_eval pipeline has several stochastic surfaces that
``training.seed`` alone does not cover:
  - Python ``hash()`` of str/bytes/frozenset is randomised per process
    (``PYTHONHASHSEED``); this affects ``set`` and ``dict`` iteration order.
  - ``random`` and ``numpy.random`` have their own RNGs.
  - ``torch`` has a CPU RNG and one CUDA RNG per device.
  - cuDNN and cuBLAS pick non-deterministic algorithms by default; cuBLAS
    additionally requires a workspace-size env var to be set before CUDA
    initialization when deterministic mode is enabled.

``seed_all(seed)`` below configures all of these. Call it:
  - once in the launcher/main process before ``ray.init()`` so the env
    vars propagate to every Ray worker;
  - once inside each Ray worker's entry point (``TaskRunner.run``, FSDP
    worker ``__init__``), passing ``rank`` so different ranks diverge
    deterministically.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional


logger = logging.getLogger(__name__)

_DETERMINISTIC_ENV_VARS = {
    # Required when torch.use_deterministic_algorithms(True) is combined
    # with CUDA >= 10.2 cuBLAS GEMM. Must be set before CUDA context init.
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def seed_all(seed: int, rank: int = 0, *, deterministic: bool = True) -> int:
    """Seed every RNG we know about and enable deterministic GPU kernels.

    Returns the effective seed (``seed + rank``) so the caller can log it.
    """
    effective_seed = int(seed) + int(rank)

    # PYTHONHASHSEED must be set in the environment BEFORE the Python
    # interpreter starts to actually randomise hash(); setting it here is a
    # no-op for the current process but is inherited by any subprocess or
    # Ray worker spawned after this call.
    os.environ["PYTHONHASHSEED"] = str(effective_seed)

    if deterministic:
        for k, v in _DETERMINISTIC_ENV_VARS.items():
            os.environ.setdefault(k, v)

    random.seed(effective_seed)
    try:
        import numpy as np

        np.random.seed(effective_seed % (2**32))
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # ``warn_only=True``: if an op has no deterministic kernel,
            # fall back to the non-deterministic one with a warning
            # instead of raising. Keeps training unblocked while surfacing
            # the offending op.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older torch has no ``warn_only`` kwarg.
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass

    logger.info(
        "seed_all: seed=%d rank=%d effective=%d deterministic=%s",
        seed, rank, effective_seed, deterministic,
    )
    return effective_seed


def derive_sampling_seed(
    base_seed: int,
    *components: int,
) -> int:
    """Derive a per-request vLLM sampling seed from a base seed and indices.

    Used to seed ``SamplingParams(seed=...)`` so that each rollout request
    is reproducible but distinct. Components can be any mix of
    ``(global_step, prompt_idx, sample_idx, ans_loop, role_id, ...)``.
    The output is stable across Python versions (does not depend on
    ``hash()``) and fits in 63 bits to satisfy vLLM's signed-int check.
    """
    # splitmix64-like mixing, avoids PYTHONHASHSEED dependence.
    x = int(base_seed) & 0xFFFFFFFFFFFFFFFF
    for c in components:
        x ^= int(c) & 0xFFFFFFFFFFFFFFFF
        x = (x * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 30)
        x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27)
        x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 31)
    # vLLM's SamplingParams.seed must be non-negative and fit in int64.
    return x & 0x7FFFFFFFFFFFFFFF
