"""Small backend-level remote helpers shared across evaluation pipelines."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def discover_checkpoints(
    backend: "RemoteBackend",
) -> list[int]:
    """Discover available checkpoint step numbers from a remote backend root."""
    children = backend.list_immediate_children("")
    step_re = re.compile(r"^global_step_(\d+)/?$")
    steps = []
    for child in children:
        match = step_re.match(child)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def load_run_metadata(
    backend: "RemoteBackend",
) -> Optional[dict]:
    """Download and parse ``run_metadata.json`` from a remote backend root."""
    try:
        data = backend.download_bytes("run_metadata.json")
        return json.loads(data)
    except FileNotFoundError:
        logger.debug("run_metadata.json not found on remote")
        return None
    except Exception as exc:
        logger.warning("Failed to load run_metadata.json: %s", exc)
        return None


def check_result_exists(
    backend: "RemoteBackend",
    output_filename: str,
) -> bool:
    """Check whether a remote result exists and is a valid completed eval JSON."""
    from verl_inf_evolve.sol_eval.result_format import is_result_complete

    try:
        if not backend.exists(output_filename):
            return False
    except Exception:
        return False

    try:
        data = backend.download_bytes(output_filename)
        result_json = json.loads(data)
        return is_result_complete(result_json)
    except Exception as exc:
        logger.warning(
            "Remote result exists but validation failed for %s: %s",
            output_filename,
            exc,
        )
        return False
