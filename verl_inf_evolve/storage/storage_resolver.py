"""Unified local/remote file resolution layer.

Provides a single ``StorageResolver`` class that replaces ad-hoc
"try local, fallback to remote" patterns scattered across the codebase.

All remote I/O is delegated to :class:`StageUploadManager` — this module
imports neither ``boto3`` nor ``rclone`` nor ``huggingface_hub``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from verl_inf_evolve.storage.stage_upload_manager import StageUploadManager

logger = logging.getLogger(__name__)

ResolveOrder = Literal["local_first", "remote_first"]


class StorageResolver:
    """Resolve training artifacts from local disk and/or a remote backend.

    Args:
        local_base: Root local directory for this scope
            (e.g. ``{default_local_dir}/ans_{N}``).
        upload_manager: Existing :class:`StageUploadManager` instance,
            or ``None`` for local-only mode.
        remote_prefix: Remote key prefix (e.g. ``"ans_0"``).  Joined with
            ``relative_path`` to form the full remote key.
        resolve_order: ``"local_first"`` (default) or ``"remote_first"``.
    """

    def __init__(
        self,
        local_base: str,
        upload_manager: StageUploadManager | None = None,
        remote_prefix: str = "",
        resolve_order: ResolveOrder = "local_first",
    ):
        self._local_base = local_base
        self._upload_manager = upload_manager
        self._remote_prefix = remote_prefix
        self._resolve_order = resolve_order

    # ------------------------------------------------------------------
    # Public resolution methods
    # ------------------------------------------------------------------

    def resolve_json(
        self, relative_path: str, *, to_memory: bool = True
    ) -> dict | list | None:
        """Resolve a JSON file, returning parsed data or ``None``.

        Args:
            relative_path: Path relative to *local_base* / *remote_prefix*
                (e.g. ``"state.json"`` or ``"gen_output.json"``).
            to_memory: If ``True`` (default), deserialize and return the
                parsed object.  If ``False``, download to local disk and
                return the local file path (or ``None``).
        """
        if to_memory:
            return self._resolve_to_memory(relative_path, kind="json")
        return self.resolve_file(relative_path)

    def resolve_dataproto(
        self, relative_path: str, *, to_memory: bool = True
    ) -> Any | None:
        """Resolve a DataProto ``.pt`` file.

        Args:
            relative_path: e.g. ``"gen_output.pt"``.
            to_memory: If ``True``, deserialize and return the DataProto.
                If ``False``, download to local disk and return the path.
        """
        if to_memory:
            return self._resolve_to_memory(relative_path, kind="dataproto")
        return self.resolve_file(relative_path)

    def resolve_file(self, relative_path: str) -> str | None:
        """Resolve a file and return its local path (downloading if needed).

        Returns ``None`` if the file cannot be found anywhere.
        """
        local_path = os.path.join(self._local_base, relative_path)

        if self._resolve_order == "local_first":
            if os.path.isfile(local_path):
                logger.info("Resolved %s from local: %s", relative_path, local_path)
                return local_path
            if self._download_file(relative_path, local_path):
                logger.info(
                    "Resolved %s from remote (local miss) → %s", relative_path, local_path
                )
                return local_path
            logger.debug("Could not resolve %s (checked local + remote)", relative_path)
            return None
        else:
            # remote_first: try downloading even if local exists
            if self._download_file(relative_path, local_path):
                logger.info(
                    "Resolved %s from remote (remote_first) → %s", relative_path, local_path
                )
                return local_path
            if os.path.isfile(local_path):
                logger.info(
                    "Resolved %s from local (remote unavailable): %s", relative_path, local_path
                )
                return local_path
            logger.debug("Could not resolve %s (checked remote + local)", relative_path)
            return None

    def resolve_dir(
        self, relative_path: str, local_path: str | None = None
    ) -> str | None:
        """Resolve a directory and return its local path.

        Args:
            relative_path: Directory name relative to prefixes
                (e.g. ``"generator"``).
            local_path: Override for the local destination.  If ``None``,
                defaults to ``{local_base}/{relative_path}``.

        Returns:
            Local directory path, or ``None`` if unavailable.
        """
        if local_path is None:
            local_path = os.path.join(self._local_base, relative_path)

        if self._resolve_order == "local_first":
            if os.path.isdir(local_path):
                logger.info("Resolved dir %s from local: %s", relative_path, local_path)
                return local_path
            if self._download_dir(relative_path, local_path):
                logger.info(
                    "Resolved dir %s from remote (local miss) → %s", relative_path, local_path
                )
                return local_path
            logger.debug("Could not resolve dir %s (checked local + remote)", relative_path)
            return None
        else:
            if self._download_dir(relative_path, local_path):
                logger.info(
                    "Resolved dir %s from remote (remote_first) → %s", relative_path, local_path
                )
                return local_path
            if os.path.isdir(local_path):
                logger.info(
                    "Resolved dir %s from local (remote unavailable): %s",
                    relative_path, local_path,
                )
                return local_path
            logger.debug("Could not resolve dir %s (checked remote + local)", relative_path)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remote_key(self, relative_path: str) -> str:
        """Build full remote key from prefix + relative path."""
        if self._remote_prefix:
            return f"{self._remote_prefix}/{relative_path}"
        return relative_path

    def _has_remote(self) -> bool:
        return (
            self._upload_manager is not None
            and self._upload_manager.remote_configured
        )

    def _resolve_to_memory(self, relative_path: str, kind: str) -> Any | None:
        """Load into memory, respecting resolve_order."""
        local_path = os.path.join(self._local_base, relative_path)

        if self._resolve_order == "local_first":
            data = self._load_local(local_path, kind)
            if data is not None:
                logger.info(
                    "Resolved %s (%s) to memory from local: %s",
                    relative_path, kind, local_path,
                )
                return data
            data = self._download_to_memory(relative_path, kind)
            if data is not None:
                logger.info(
                    "Resolved %s (%s) to memory from remote (local miss)",
                    relative_path, kind,
                )
            return data
        else:
            data = self._download_to_memory(relative_path, kind)
            if data is not None:
                logger.info(
                    "Resolved %s (%s) to memory from remote (remote_first)",
                    relative_path, kind,
                )
                return data
            data = self._load_local(local_path, kind)
            if data is not None:
                logger.info(
                    "Resolved %s (%s) to memory from local (remote unavailable): %s",
                    relative_path, kind, local_path,
                )
            return data

    @staticmethod
    def _load_local(local_path: str, kind: str) -> Any | None:
        """Load from local disk.  Returns ``None`` on missing/error."""
        if not os.path.isfile(local_path):
            return None
        try:
            if kind == "json":
                with open(local_path, "r") as f:
                    return json.load(f)
            elif kind == "dataproto":
                from verl import DataProto

                return DataProto.load_from_disk(local_path)
            else:
                logger.warning("Unknown kind %r for local load", kind)
                return None
        except Exception:
            logger.warning(
                "Failed to load %s locally: %s", kind, local_path, exc_info=True
            )
            return None

    def _download_to_memory(self, relative_path: str, kind: str) -> Any | None:
        """Download from remote into memory.  Returns ``None`` on failure."""
        if not self._has_remote():
            return None
        remote_key = self._remote_key(relative_path)
        data = self._upload_manager.download_to_memory(remote_key, kind)
        if data is not None:
            logger.info("Downloaded %s from remote (in-memory): %s", kind, remote_key)
        return data

    def _download_file(self, relative_path: str, local_path: str) -> bool:
        """Download a single file from remote to local disk."""
        if not self._has_remote():
            return False
        remote_key = self._remote_key(relative_path)
        ok = self._upload_manager.download_file_to_local(remote_key, local_path)
        if ok:
            logger.info("Downloaded file from remote: %s → %s", remote_key, local_path)
        return ok

    def _download_dir(self, relative_path: str, local_path: str) -> bool:
        """Download a directory from remote to local disk."""
        if not self._has_remote():
            return False
        remote_key = self._remote_key(relative_path)
        ok = self._upload_manager.download_dir_to_local(remote_key, local_path)
        if ok:
            logger.info("Downloaded dir from remote: %s → %s", remote_key, local_path)
        return ok
