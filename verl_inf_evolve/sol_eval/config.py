"""Checkpoint specification parsing utility for the evaluation pipeline.

Provides parse_checkpoints() for converting various checkpoint spec formats
(list, slice, dash-range, mixed) into sorted, deduplicated integer lists.
Hydra + OmegaConf handles all other config loading.
"""

from numbers import Integral
import re
from typing import Union


def parse_checkpoints(spec: Union[int, list[int], str]) -> list[int]:
    """Parse a checkpoint specification into a sorted, deduplicated list of ints.

    Supported formats:
        - int:              5           → [5]
        - list[int]:        [0, 5, 10] → [0, 5, 10]
        - Slice notation:   "0:20:5"   → [0, 5, 10, 15]  (Python range semantics)
        - Two-part slice:   "0:20"     → [0, 5, 10, 15, 20] (only with implicit step)
        - Dash range:       "0-5"      → [0, 1, 2, 3, 4, 5]  (inclusive both ends)
        - Mixed:            "0-3,7,10-12" → [0, 1, 2, 3, 7, 10, 11, 12]
        - Single int str:   "5"        → [5]
        - Special:          -1          → [-1]  (base model, no checkpoint)

    Returns:
        Sorted, deduplicated list of checkpoint numbers.

    Raises:
        ValueError: If spec is empty, invalid format, or produces no checkpoints.
    """
    if isinstance(spec, Integral):
        return [int(spec)]

    # Convert OmegaConf ListConfig to plain list
    if not isinstance(spec, (list, str)):
        try:
            spec = list(spec)
        except TypeError:
            raise ValueError(f"Invalid checkpoint spec type: {type(spec)}")

    if isinstance(spec, list):
        if not spec:
            raise ValueError("Checkpoint list is empty")
        # If the list contains a single string (e.g. YAML parsed [0:50:5] as ['0:50:5']),
        # recurse to parse it as a spec string.
        if len(spec) == 1 and isinstance(spec[0], str):
            return parse_checkpoints(spec[0])
        return sorted(set(spec))

    if not isinstance(spec, str):
        raise ValueError(f"Invalid checkpoint spec type: {type(spec)}")

    spec = spec.strip()
    if not spec:
        raise ValueError("Checkpoint spec is empty string")

    # Hydra CLI overrides can arrive as bracketed list strings, e.g. "[-1]"
    # or "[0,5,10]". Normalize them into the plain comma-separated form that
    # the parser already understands.
    if spec.startswith("[") and spec.endswith("]"):
        inner = spec[1:-1].strip()
        if not inner:
            raise ValueError("Checkpoint spec is an empty bracketed list")
        return parse_checkpoints(inner)

    # Check for slice notation (colons) — must be handled before comma split
    # because slice notation doesn't mix with commas
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) == 2:
            start, stop = int(parts[0]), int(parts[1])
            result = list(range(start, stop + 1))
        elif len(parts) == 3:
            start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
            result = list(range(start, stop, step))
        else:
            raise ValueError(f"Invalid slice notation: {spec!r} (expected start:stop or start:stop:step)")
        if not result:
            raise ValueError(f"Slice notation {spec!r} produces no checkpoints")
        return sorted(set(result))

    # Comma-separated parts, each can be a dash-range or single int
    result: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        dash_match = re.match(r"^(\d+)-(\d+)$", part)
        if dash_match:
            start, end = int(dash_match.group(1)), int(dash_match.group(2))
            result.extend(range(start, end + 1))
        elif re.match(r"^-?\d+$", part):
            result.append(int(part))
        else:
            raise ValueError(f"Invalid checkpoint part: {part!r}")

    if not result:
        raise ValueError(f"Checkpoint spec {spec!r} produces no checkpoints")
    return sorted(set(result))
