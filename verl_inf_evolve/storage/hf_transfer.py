"""Helpers for constraining Hugging Face Hub transfer behavior.

The Hub client does not expose a direct bandwidth cap, so we install a custom
``requests.Session`` backend via ``huggingface_hub.configure_http_backend`` and
apply an approximate process-wide rate limit to upload and download streams.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests
from huggingface_hub import configure_http_backend

logger = logging.getLogger(__name__)

_DEFAULT_UPLOAD_CHUNK_SIZE = 64 * 1024
_CURRENT_LIMITS: tuple[float | None, float | None] | None = None
_CURRENT_LIMITS_LOCK = threading.Lock()


def _cfg_get(cfg: Any | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_positive_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if number <= 0:
        raise ValueError(f"Expected a positive float, got {value!r}")
    return number


def _normalize_positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    number = int(value)
    if number <= 0:
        raise ValueError(f"Expected a positive integer, got {value!r}")
    return number


def get_hf_transfer_settings(remote_cfg: Any | None) -> dict[str, float | int | None]:
    """Extract HF transfer-related settings from a config mapping."""
    return {
        "upload_limit_mbps": _normalize_positive_float(
            _cfg_get(remote_cfg, "hf_upload_limit_mbps", None)
        ),
        "download_limit_mbps": _normalize_positive_float(
            _cfg_get(remote_cfg, "hf_download_limit_mbps", None)
        ),
        "snapshot_max_workers": _normalize_positive_int(
            _cfg_get(remote_cfg, "hf_snapshot_max_workers", None)
        ),
    }


class _SharedRateLimiter:
    """Simple process-wide rate limiter shared across all HF sessions."""

    def __init__(self, rate_bps: float) -> None:
        self._rate_bps = float(rate_bps)
        self._lock = threading.Lock()
        self._next_available = time.monotonic()

    def throttle(self, num_bytes: int) -> None:
        if num_bytes <= 0:
            return

        wait_for = 0.0
        with self._lock:
            now = time.monotonic()
            slot_start = max(now, self._next_available)
            self._next_available = slot_start + (num_bytes / self._rate_bps)
            wait_for = max(0.0, slot_start - now)

        if wait_for > 0:
            time.sleep(wait_for)


class _RateLimitedUploadBody:
    """Wrap a readable request body and pace reads before they hit the wire."""

    def __init__(
        self,
        raw: Any,
        limiter: _SharedRateLimiter,
        chunk_size: int = _DEFAULT_UPLOAD_CHUNK_SIZE,
    ) -> None:
        self._raw = raw
        self._limiter = limiter
        self._chunk_size = chunk_size

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._chunk_size
        else:
            size = min(size, self._chunk_size)

        chunk = self._raw.read(size)
        self._limiter.throttle(len(chunk))
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class _RateLimitedDownloadStream:
    """Wrap a response stream and pace reads consumed by huggingface_hub."""

    def __init__(self, raw: Any, limiter: _SharedRateLimiter) -> None:
        self._raw = raw
        self._limiter = limiter

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        chunk = self._raw.read(*args, **kwargs)
        self._limiter.throttle(len(chunk))
        return chunk

    def readinto(self, buffer: Any) -> int:
        size = self._raw.readinto(buffer)
        self._limiter.throttle(size or 0)
        return size

    def stream(self, amt: int = 65536, decode_content: bool | None = None) -> Any:
        for chunk in self._raw.stream(amt, decode_content=decode_content):
            self._limiter.throttle(len(chunk))
            yield chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class _ThrottledSession(requests.Session):
    """Requests session that wraps HF upload/download streams."""

    def __init__(
        self,
        *,
        upload_limiter: _SharedRateLimiter | None,
        download_limiter: _SharedRateLimiter | None,
    ) -> None:
        super().__init__()
        self._upload_limiter = upload_limiter
        self._download_limiter = download_limiter

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        if (
            self._upload_limiter is not None
            and hasattr(request, "body")
            and hasattr(request.body, "read")
            and not isinstance(request.body, _RateLimitedUploadBody)
        ):
            request.body = _RateLimitedUploadBody(request.body, self._upload_limiter)

        response = super().send(request, **kwargs)

        if (
            self._download_limiter is not None
            and getattr(response, "raw", None) is not None
            and not isinstance(response.raw, _RateLimitedDownloadStream)
        ):
            response.raw = _RateLimitedDownloadStream(response.raw, self._download_limiter)

        return response


def configure_hf_transfer_limits(
    *,
    upload_limit_mbps: float | None = None,
    download_limit_mbps: float | None = None,
) -> None:
    """Install a custom HF HTTP backend with optional transfer limits."""
    upload_limit_mbps = _normalize_positive_float(upload_limit_mbps)
    download_limit_mbps = _normalize_positive_float(download_limit_mbps)
    limits = (upload_limit_mbps, download_limit_mbps)

    with _CURRENT_LIMITS_LOCK:
        global _CURRENT_LIMITS
        if _CURRENT_LIMITS == limits:
            return
        _CURRENT_LIMITS = limits

    upload_limiter = (
        _SharedRateLimiter(upload_limit_mbps * 1024 * 1024 / 8.0)
        if upload_limit_mbps is not None
        else None
    )
    download_limiter = (
        _SharedRateLimiter(download_limit_mbps * 1024 * 1024 / 8.0)
        if download_limit_mbps is not None
        else None
    )

    def _backend_factory() -> requests.Session:
        return _ThrottledSession(
            upload_limiter=upload_limiter,
            download_limiter=download_limiter,
        )

    configure_http_backend(backend_factory=_backend_factory)
    logger.info(
        "Configured HF transfer limits: upload=%s Mbps, download=%s Mbps",
        upload_limit_mbps if upload_limit_mbps is not None else "unlimited",
        download_limit_mbps if download_limit_mbps is not None else "unlimited",
    )
