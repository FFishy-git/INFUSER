"""Backend-agnostic remote storage abstraction.

Provides:
- ``RemoteBackend`` — abstract base class for remote storage operations.
- ``R2RemoteBackend`` — Cloudflare R2 (S3-compatible) implementation.
- ``HFDatasetRemoteBackend`` — Hugging Face dataset repo implementation.
- ``create_remote_backend(uri, **kwargs)`` — factory that returns the correct
  backend based on URI scheme.

URI schemes:
- ``s3://bucket/prefix`` or ``r2://bucket/prefix`` → R2RemoteBackend
- ``hf://datasets/<namespace>/<repo>/<prefix>`` → HFDatasetRemoteBackend
"""

from __future__ import annotations

import abc
import copy
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class RemoteBackend(abc.ABC):
    """Abstract interface for remote storage operations."""

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether a remote object exists at *key* (relative to backend root)."""
        ...

    @abc.abstractmethod
    def download_bytes(self, key: str) -> bytes:
        """Download the object at *key* and return its contents as bytes.

        Raises ``FileNotFoundError`` if the object does not exist.
        """
        ...

    @abc.abstractmethod
    def download_file(self, key: str, local_path: str) -> bool:
        """Download the object at *key* to *local_path*.

        Returns ``True`` on success, ``False`` on failure.
        """
        ...

    @abc.abstractmethod
    def download_dir(self, key: str, local_path: str) -> bool:
        """Download all objects under *key* prefix to *local_path*.

        Returns ``True`` on success, ``False`` on failure.
        """
        ...

    @abc.abstractmethod
    def upload_bytes(self, data: bytes, key: str) -> bool:
        """Upload *data* to the remote at *key*.

        Returns ``True`` on success, ``False`` on failure.
        """
        ...

    @abc.abstractmethod
    def upload_file(self, local_path: str, key: str) -> bool:
        """Upload a local file to the remote at *key*.

        Returns ``True`` on success, ``False`` on failure.
        """
        ...

    @abc.abstractmethod
    def upload_dir(self, local_path: str, key: str) -> bool:
        """Upload a local directory tree to the remote under *key*.

        Returns ``True`` on success, ``False`` on failure.
        """
        ...

    @abc.abstractmethod
    def delete_prefix(self, key: str, *, exclude_patterns: list[str] | None = None) -> bool:
        """Delete all objects under *key* prefix.

        Args:
            key: Remote prefix to delete under.
            exclude_patterns: Glob patterns for files to *exclude* from deletion.

        Returns ``True`` on success, ``False`` on failure.
        """
        ...

    @abc.abstractmethod
    def list_immediate_children(self, key: str) -> list[str]:
        """List immediate child names (not full paths) under *key* prefix.

        Similar to ``ls`` (non-recursive). Returns directory-like prefixes
        and object names at the first level only.
        """
        ...

    @abc.abstractmethod
    def list_files_recursive(self, key: str) -> list[str]:
        """List all file paths recursively under *key* prefix.

        Returns paths relative to the given *key* prefix.
        """
        ...


# ---------------------------------------------------------------------------
# URI parsing and factory
# ---------------------------------------------------------------------------

# Pattern: hf://datasets/<namespace>/<repo>[/<prefix>]
_HF_DATASETS_RE = re.compile(
    r"^hf://datasets/(?P<namespace>[^/]+)/(?P<repo>[^/]+)(?:/(?P<prefix>.+?))?/?$"
)


def _cfg_get(cfg: Any | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_hf_token(remote_cfg: Any | None, *, explicit_token: str | None = None) -> str | None:
    if explicit_token:
        return explicit_token

    token = _cfg_get(remote_cfg, "hf_token", None)
    if token:
        return str(token)

    env_var_name = str(_cfg_get(remote_cfg, "hf_token_env_var", "HF_TOKEN"))
    env_token = os.environ.get(env_var_name)
    if env_token:
        return env_token

    if env_var_name != "HF_TOKEN":
        return os.environ.get("HF_TOKEN")
    return None


def build_hf_backend_kwargs(
    remote_cfg: Any | None,
    *,
    token: str | None = None,
    auto_create_repo: bool | None = None,
) -> dict[str, Any]:
    """Build constructor kwargs for ``HFDatasetRemoteBackend`` from config."""
    from verl_inf_evolve.storage.hf_transfer import get_hf_transfer_settings

    kwargs: dict[str, Any] = {
        "revision": str(_cfg_get(remote_cfg, "hf_revision", "main")),
    }

    resolved_token = _resolve_hf_token(remote_cfg, explicit_token=token)
    if resolved_token:
        kwargs["token"] = resolved_token

    kwargs.update(get_hf_transfer_settings(remote_cfg))

    if auto_create_repo is not None:
        kwargs["auto_create_repo"] = auto_create_repo

    return {key: value for key, value in kwargs.items() if value is not None}


def _parse_hf_dataset_uri(uri: str) -> dict[str, str]:
    """Parse ``hf://datasets/<namespace>/<repo>[/<prefix>]``.

    Returns dict with keys: ``repo_id``, ``prefix``.
    Raises ``ValueError`` on parse failure.
    """
    m = _HF_DATASETS_RE.match(uri)
    if not m:
        raise ValueError(
            f"Malformed HF dataset URI: {uri!r}. "
            f"Expected format: hf://datasets/<namespace>/<repo>[/<prefix>]"
        )
    namespace = m.group("namespace")
    repo = m.group("repo")
    prefix = m.group("prefix") or ""
    return {"repo_id": f"{namespace}/{repo}", "prefix": prefix}


def create_remote_backend(uri: str, **kwargs: Any) -> RemoteBackend:
    """Factory: parse *uri* and return the appropriate ``RemoteBackend``.

    Args:
        uri: Remote storage URI. Supported schemes:
            - ``s3://bucket/prefix`` or ``r2://bucket/prefix`` → R2RemoteBackend
            - ``hf://datasets/<namespace>/<repo>/<prefix>`` → HFDatasetRemoteBackend
        **kwargs: Extra keyword arguments passed to the backend constructor.

    Returns:
        A ``RemoteBackend`` instance.

    Raises:
        ValueError: If the URI is malformed, empty, or uses an unsupported scheme.
    """
    if not uri or not isinstance(uri, str):
        raise ValueError(f"Remote URI must be a non-empty string, got: {uri!r}")

    uri = uri.strip()
    if not uri:
        raise ValueError("Remote URI must be a non-empty string, got empty string")

    # --- S3 / R2 ---
    if uri.startswith("s3://") or uri.startswith("r2://"):
        from verl_inf_evolve.storage.r2_utils import parse_r2_path

        bucket, prefix = parse_r2_path(uri)
        if not bucket:
            raise ValueError(f"Malformed S3/R2 URI (no bucket): {uri!r}")
        # Lazy import to avoid circular deps
        from verl_inf_evolve.storage.r2_remote_backend import R2RemoteBackend

        return R2RemoteBackend(bucket=bucket, prefix=prefix, **kwargs)

    # --- Hugging Face ---
    if uri.startswith("hf://"):
        # Only dataset repos are supported
        if not uri.startswith("hf://datasets/"):
            raise ValueError(
                f"Only HF dataset repos are supported. Got: {uri!r}. "
                f"Use hf://datasets/<namespace>/<repo>[/<prefix>] instead."
            )
        parsed = _parse_hf_dataset_uri(uri)
        from verl_inf_evolve.storage.hf_remote_backend import HFDatasetRemoteBackend

        return HFDatasetRemoteBackend(
            repo_id=parsed["repo_id"],
            prefix=parsed["prefix"],
            **kwargs,
        )

    raise ValueError(
        f"Unsupported remote URI scheme: {uri!r}. "
        f"Supported schemes: s3://, r2://, hf://datasets/..."
    )


# ---------------------------------------------------------------------------
# Config redaction
# ---------------------------------------------------------------------------

# Keys that must be redacted before logging / serializing config dicts.
_SENSITIVE_KEYS = {"hf_token"}
_REDACTED = "***REDACTED***"


def redact_config_secrets(config: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of *config* with sensitive values replaced.

    Currently redacts ``remote.hf_token`` (and any other key in
    ``_SENSITIVE_KEYS``) wherever it appears in the config tree.
    """
    config = copy.deepcopy(config)
    _redact_recursive(config)
    return config


def _redact_recursive(obj: Any) -> None:
    """Walk *obj* in-place and replace sensitive leaf values."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _SENSITIVE_KEYS and value:
                obj[key] = _REDACTED
            else:
                _redact_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            _redact_recursive(item)
