"""Minimal .env loading helpers for verl_inf_evolve entrypoints."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_DOTENV_LOADED = False


def load_startup_env() -> list[str]:
    """Load .env-style files into the current process once.

    Search order:
    1. Paths from ``VERL_INF_EVOLVE_DOTENV_PATH`` (os.pathsep-separated).
    2. ``.env`` in the current working directory.

    Existing environment variables are never overwritten.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return []

    loaded: list[str] = []
    for path in _startup_dotenv_paths():
        if not path.is_file():
            continue
        _load_dotenv_file(path)
        loaded.append(str(path))

    sanitize_cuda_visible_device_env()

    _DOTENV_LOADED = True
    if loaded:
        logger.info("Loaded startup env from %s", ", ".join(loaded))
    return loaded


def sanitize_cuda_visible_device_env() -> dict[str, str]:
    """Drop conflicting ROCm visibility variables for NVIDIA/CUDA jobs.

    Some clusters export ``ROCR_VISIBLE_DEVICES`` globally even on NVIDIA
    nodes. Ray then sets ``CUDA_VISIBLE_DEVICES`` per actor, and verl treats
    the combined environment as ambiguous and aborts worker startup.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_visible or cuda_visible == "NoDevFiles":
        return {}
    if not _has_nvidia_gpu():
        return {}

    removed: dict[str, str] = {}
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        value = os.environ.get(key, "").strip()
        if not value:
            continue
        removed[key] = os.environ.pop(key)

    if removed:
        logger.warning(
            "Unset conflicting ROCm visible-device env vars for CUDA run: %s",
            ", ".join(f"{key}={value}" for key, value in removed.items()),
        )
    return removed


def _startup_dotenv_paths() -> list[Path]:
    configured = os.environ.get("VERL_INF_EVOLVE_DOTENV_PATH", "").strip()
    if configured:
        paths = [
            Path(part).expanduser()
            for part in configured.split(os.pathsep)
            if part.strip()
        ]
        return paths
    return [Path.cwd() / ".env"]


def _load_dotenv_file(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            parsed = _parse_dotenv_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    lexer = shlex.shlex(stripped, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    tokens = list(lexer)
    if not tokens:
        return None
    if tokens[0] == "export":
        tokens = tokens[1:]
    if len(tokens) != 1 or "=" not in tokens[0]:
        raise ValueError(f"Invalid .env line: {line.rstrip()}")

    key, value = tokens[0].split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid .env line: {line.rstrip()}")
    return key, value


def _has_nvidia_gpu() -> bool:
    if os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip():
        return True

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False

    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    return proc.returncode == 0 and bool(proc.stdout.strip())
