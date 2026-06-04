"""R2 (Cloudflare S3-compatible) utilities for the V3 self-evolution pipeline.

Provides:
- ``get_r2_client()`` — boto3 S3 client configured for R2.
- ``parse_r2_path()`` — parse ``s3://bucket/prefix`` into (bucket, prefix).
- ``rclone_copy()`` / ``rclone_sync()`` — subprocess wrappers for multi-file
  directory transfers.

Credentials are resolved in order:
  1. Environment variables (API_ENDPOINT, ACCESS_KEY_ID, SECRET_ACCESS_KEY)
  2. ``.cloudflare.conf`` file (cwd → repo root → home directory)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.config import Config as BotoConfig
    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False

# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

_CONF_FILENAME = ".cloudflare.conf"


def _load_cloudflare_conf() -> dict[str, str]:
    """Search for ``.cloudflare.conf`` and parse it into a dict.

    Search order:
      1. Current working directory
      2. This file's repo root  (``r2_utils.py`` → ``utils/`` → ``verl_inf_evolve/`` → repo root)
      3. Home directory
    """
    search_paths = [
        Path.cwd() / _CONF_FILENAME,
        Path(__file__).resolve().parent.parent.parent / _CONF_FILENAME,
        Path.home() / _CONF_FILENAME,
    ]
    conf_path: Path | None = None
    for p in search_paths:
        if p.exists():
            conf_path = p
            break

    if conf_path is None:
        return {}

    config_vars: dict[str, str] = {}
    with open(conf_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                key, value = line.split("=", 1)
                value = value.strip('"').strip("'")
                config_vars[key] = value
    return config_vars


def _load_credentials() -> tuple[str, str, str, str]:
    """Return ``(endpoint_url, access_key_id, secret_access_key, region)``.

    Priority: env vars → ``.cloudflare.conf``.
    """
    endpoint = os.getenv("API_ENDPOINT") or os.getenv("R2_ENDPOINT_URL", "")
    access_key = (
        os.getenv("ACCESS_KEY_ID")
        or os.getenv("R2_ACCESS_KEY_ID")
        or os.getenv("AWS_ACCESS_KEY_ID", "")
    )
    secret_key = (
        os.getenv("SECRET_ACCESS_KEY")
        or os.getenv("R2_SECRET_ACCESS_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY", "")
    )
    region = os.getenv("DEFAULT_REGION") or os.getenv("R2_REGION", "auto")

    if endpoint and access_key and secret_key:
        return endpoint, access_key, secret_key, region

    # Fall back to .cloudflare.conf
    conf = _load_cloudflare_conf()
    if conf:
        endpoint = endpoint or conf.get("API_ENDPOINT", "")
        access_key = access_key or conf.get("ACCESS_KEY_ID", "")
        secret_key = secret_key or conf.get("SECRET_ACCESS_KEY", "")
        region = region if region != "auto" else conf.get("DEFAULT_REGION", "auto")

    return endpoint, access_key, secret_key, region


# ---------------------------------------------------------------------------
# boto3 client
# ---------------------------------------------------------------------------

_cached_client = None


def get_r2_client(use_device_config_fallback: bool = False):
    """Return a boto3 S3 client configured for Cloudflare R2, or ``None``.

    The client is cached after first successful creation.
    """
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    if not _HAS_BOTO3:
        logger.warning("boto3 not installed; R2 uploads disabled")
        return None

    endpoint, access_key, secret_key, region = _load_credentials()

    if not (endpoint and access_key and secret_key):
        missing = []
        if not endpoint:
            missing.append("endpoint_url (API_ENDPOINT or R2_ENDPOINT_URL)")
        if not access_key:
            missing.append("access_key_id (ACCESS_KEY_ID or R2_ACCESS_KEY_ID)")
        if not secret_key:
            missing.append("secret_access_key (SECRET_ACCESS_KEY or R2_SECRET_ACCESS_KEY)")
        logger.error("R2 credentials incomplete, missing: %s", ", ".join(missing))
        return None

    boto_config = BotoConfig(
        connect_timeout=30,
        read_timeout=60,
        max_pool_connections=50,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=boto_config,
    )
    _cached_client = client
    return client


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


def parse_r2_path(path: str) -> tuple[str, str]:
    """Parse ``s3://bucket/prefix`` into ``(bucket, prefix)``.

    Also accepts ``gs://`` and ``r2://`` for compatibility.
    """
    for scheme in ("s3://", "gs://", "r2://"):
        if path.startswith(scheme):
            path = path[len(scheme):]
            break
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


# ---------------------------------------------------------------------------
# rclone helpers
# ---------------------------------------------------------------------------

_RCLONE_REMOTE = "r2"  # matches ``[r2]`` section in rclone.conf


def rclone_copy(
    src: str,
    dst: str,
    *,
    extra_flags: list[str] | None = None,
    timeout: int = 14400,
) -> bool:
    """Run ``rclone copy src dst`` with sensible defaults.

    Args:
        src: Local path or ``r2:bucket/prefix``.
        dst: Local path or ``r2:bucket/prefix``.
        extra_flags: Additional rclone CLI flags.
        timeout: Subprocess timeout in seconds (default 4h for large
            checkpoints that can be 100+ GB).

    Returns:
        ``True`` on success, ``False`` on failure (logged).
    """
    cmd = [
        "rclone", "copy",
        src, dst,
        "--transfers", "16",
        "--checkers", "8",
        "--s3-upload-concurrency", "4",
        "-v",
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.error("rclone copy failed (rc=%d): %s", result.returncode, result.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("rclone copy timed out after %ds: %s -> %s", timeout, src, dst)
        return False
    except FileNotFoundError:
        logger.error("rclone binary not found; install rclone to enable directory sync")
        return False


def rclone_sync(
    src: str,
    dst: str,
    *,
    extra_flags: list[str] | None = None,
    timeout: int = 3600,
) -> bool:
    """Run ``rclone sync src dst`` (makes *dst* match *src*)."""
    cmd = [
        "rclone", "sync",
        src, dst,
        "--transfers", "16",
        "--checkers", "8",
        "--s3-upload-concurrency", "4",
        "-v",
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.error("rclone sync failed (rc=%d): %s", result.returncode, result.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("rclone sync timed out after %ds: %s -> %s", timeout, src, dst)
        return False
    except FileNotFoundError:
        logger.error("rclone binary not found; install rclone to enable directory sync")
        return False


def rclone_delete(
    remote_path: str,
    *,
    exclude_patterns: list[str] | None = None,
    timeout: int = 600,
) -> bool:
    """Run ``rclone delete remote_path`` to remove files on the remote.

    Unlike ``rclone purge``, this only removes *files* matching the filters,
    leaving excluded files and directory structure intact.

    Args:
        remote_path: rclone remote path, e.g. ``r2:bucket/prefix/global_step_0``.
        exclude_patterns: Patterns to *exclude* from deletion (passed as
            ``--exclude <pattern>``).  For example ``["**/huggingface/**"]``
            preserves all files under ``huggingface/`` subdirectories.
        timeout: Subprocess timeout in seconds.

    Returns:
        ``True`` on success, ``False`` on failure (logged).
    """
    cmd = [
        "rclone", "delete",
        remote_path,
        "-v",
    ]
    if exclude_patterns:
        for pat in exclude_patterns:
            cmd.extend(["--exclude", pat])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.error("rclone delete failed (rc=%d): %s", result.returncode, result.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("rclone delete timed out after %ds: %s", timeout, remote_path)
        return False
    except FileNotFoundError:
        logger.error("rclone binary not found; install rclone to enable remote deletion")
        return False


def r2_rclone_path(bucket: str, prefix: str) -> str:
    """Build an rclone-style remote path: ``r2:bucket/prefix``."""
    return f"{_RCLONE_REMOTE}:{bucket}/{prefix}".rstrip("/")
