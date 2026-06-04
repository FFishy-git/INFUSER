"""Metric logging utilities.

Provides ``add_distribution_stats``, a single helper for writing
mean / std / min / max (and optional extras) into a metrics dict.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def add_distribution_stats(
    metrics: dict[str, float],
    prefix: str,
    values: Union[list, "np.ndarray", "torch.Tensor"],
    *,
    extra_stats: Sequence[str] = (),
    fill_empty: bool = True,
) -> None:
    """Write distribution statistics into *metrics* under *prefix*.

    Always computes ``mean``, ``std``, ``min``, ``max``.  Additional
    statistics can be requested via *extra_stats*:

    - ``"median"`` — numpy median
    - ``"count"``  — number of elements
    - ``"p<N>"``   — percentile, e.g. ``"p90"``

    Args:
        metrics: Dict to mutate in-place.
        prefix: Key prefix; stats are written as ``f"{prefix}/{stat}"``.
        values: Data to summarise.  Accepts ``list``, ``np.ndarray``,
            or ``torch.Tensor``.
        extra_stats: Additional statistics beyond the base four.
        fill_empty: If *True* (default), write ``0.0`` for every stat
            when *values* is empty.  If *False*, skip silently.
    """
    # ── Normalise to 1-D numpy array ──
    arr: np.ndarray | None = None

    if _HAS_TORCH and isinstance(values, torch.Tensor):
        if values.numel() > 0:
            arr = values.detach().cpu().numpy().astype(np.float32, copy=False)
    elif isinstance(values, np.ndarray):
        if values.size > 0:
            arr = values.astype(np.float32, copy=False)
    else:
        # list / tuple / other iterable
        if values:
            arr = np.asarray(values, dtype=np.float32)

    base = ("mean", "std", "min", "max")
    all_stats = base + tuple(extra_stats)

    if arr is None or arr.size == 0:
        if fill_empty:
            for stat in all_stats:
                metrics[f"{prefix}/{stat}"] = 0.0
        return

    # ── Base statistics ──
    metrics[f"{prefix}/mean"] = float(arr.mean())
    metrics[f"{prefix}/std"] = float(arr.std())
    metrics[f"{prefix}/min"] = float(arr.min())
    metrics[f"{prefix}/max"] = float(arr.max())

    # ── Extra statistics ──
    for stat in extra_stats:
        if stat == "median":
            metrics[f"{prefix}/median"] = float(np.median(arr))
        elif stat == "count":
            metrics[f"{prefix}/count"] = float(arr.size)
        elif stat.startswith("p") and stat[1:].isdigit():
            pct = int(stat[1:])
            metrics[f"{prefix}/{stat}"] = float(np.percentile(arr, pct))
        else:
            raise ValueError(f"Unknown extra stat: {stat!r}")
