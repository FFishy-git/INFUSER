"""Helpers for one-shot startup cleanup of local artifact directories."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any


def maybe_clear_local_dir_on_start(config: Any) -> bool:
    """Delete ``training.default_local_dir`` once when startup cleanup is enabled.

    The cleanup flag is treated as one-shot by flipping it to ``False`` after
    the first check. This lets callers invoke the helper defensively from
    multiple startup phases without deleting the directory after logging or
    worker initialization has already begun.
    """

    training_cfg = config.training
    clear_local = training_cfg.get("clear_local_dir_on_start", True)
    local_dir = training_cfg.get("default_local_dir", None)

    # Make the cleanup one-shot so later startup phases cannot race with
    # logging or worker initialization by attempting a second delete.
    training_cfg.clear_local_dir_on_start = False

    if not clear_local or not local_dir or not os.path.isdir(local_dir):
        return False

    logging.getLogger(__name__).warning(
        "clear_local_dir_on_start: removing %s to avoid stale artifacts",
        local_dir,
    )
    shutil.rmtree(local_dir)
    return True
