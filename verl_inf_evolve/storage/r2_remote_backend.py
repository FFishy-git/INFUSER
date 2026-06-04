"""R2 (Cloudflare S3-compatible) implementation of RemoteBackend.

This is a thin wrapper that delegates to existing ``r2_utils`` functions
(``get_r2_client``, ``rclone_copy``, ``rclone_delete``, ``r2_rclone_path``).
No changes to ``r2_utils.py`` internals.
"""

from __future__ import annotations

import logging
import os

from verl_inf_evolve.storage.remote_backend import RemoteBackend

logger = logging.getLogger(__name__)


class R2RemoteBackend(RemoteBackend):
    """RemoteBackend backed by Cloudflare R2 (S3-compatible).

    Args:
        bucket: S3/R2 bucket name.
        prefix: Key prefix within the bucket.
    """

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix

    def _full_key(self, key: str) -> str:
        """Combine the backend prefix with a relative *key*."""
        if self.prefix:
            return f"{self.prefix}/{key}".rstrip("/") if key else self.prefix
        return key

    def _rclone_remote_path(self, key: str) -> str:
        """Build an rclone remote path for the given key."""
        from verl_inf_evolve.storage.r2_utils import r2_rclone_path

        return r2_rclone_path(self.bucket, self._full_key(key))

    def _get_client(self):
        """Return the cached boto3 S3 client, or raise on failure."""
        from verl_inf_evolve.storage.r2_utils import get_r2_client

        client = get_r2_client()
        if client is None:
            raise RuntimeError("R2 client not available (missing credentials or boto3)")
        return client

    # ------------------------------------------------------------------
    # RemoteBackend implementation
    # ------------------------------------------------------------------

    def exists(self, key: str) -> bool:
        """Check whether a remote object exists using ``head_object``."""
        try:
            client = self._get_client()
            client.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except Exception:
            return False

    def download_bytes(self, key: str) -> bytes:
        """Download an object's contents using ``get_object``."""
        client = self._get_client()
        full_key = self._full_key(key)
        try:
            response = client.get_object(Bucket=self.bucket, Key=full_key)
            data = response["Body"].read()
            response["Body"].close()
            return data
        except Exception as e:
            raise FileNotFoundError(f"R2 object not found: s3://{self.bucket}/{full_key}") from e

    def download_file(self, key: str, local_path: str) -> bool:
        """Download a single file using ``rclone copy``."""
        from verl_inf_evolve.storage.r2_utils import rclone_copy

        remote = self._rclone_remote_path(key)
        # rclone copy copies the *contents* of src into dst directory,
        # so we target the parent directory of local_path.
        local_dir = os.path.dirname(local_path) or "."
        os.makedirs(local_dir, exist_ok=True)
        return rclone_copy(remote, local_dir)

    def download_dir(self, key: str, local_path: str) -> bool:
        """Download a directory tree using ``rclone copy``."""
        from verl_inf_evolve.storage.r2_utils import rclone_copy

        remote = self._rclone_remote_path(key)
        os.makedirs(local_path, exist_ok=True)
        return rclone_copy(remote, local_path)

    def upload_bytes(self, data: bytes, key: str) -> bool:
        """Upload bytes using ``put_object``."""
        try:
            client = self._get_client()
            client.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)
            return True
        except Exception as e:
            logger.error("R2 upload_bytes failed for key %s: %s", key, e)
            return False

    def upload_file(self, local_path: str, key: str) -> bool:
        """Upload a single file using ``rclone copy``."""
        from verl_inf_evolve.storage.r2_utils import rclone_copy

        remote_dir = self._rclone_remote_path(os.path.dirname(key)) if os.path.dirname(key) else self._rclone_remote_path("")
        local_dir = os.path.dirname(local_path) or "."
        # rclone copy copies all files from src dir into dst dir.
        # To upload a single file, we copy from the file's parent dir
        # with an --include filter.
        filename = os.path.basename(local_path)
        return rclone_copy(local_dir, remote_dir, extra_flags=["--include", filename])

    def upload_dir(self, local_path: str, key: str) -> bool:
        """Upload a local directory tree using ``rclone copy``."""
        from verl_inf_evolve.storage.r2_utils import rclone_copy

        remote = self._rclone_remote_path(key)
        return rclone_copy(local_path, remote)

    def delete_prefix(self, key: str, *, exclude_patterns: list[str] | None = None) -> bool:
        """Delete all objects under a prefix using ``rclone delete``."""
        from verl_inf_evolve.storage.r2_utils import rclone_delete

        remote = self._rclone_remote_path(key)
        return rclone_delete(remote, exclude_patterns=exclude_patterns)

    def list_immediate_children(self, key: str) -> list[str]:
        """List immediate children under a prefix using ``list_objects_v2`` with Delimiter."""
        client = self._get_client()
        full_prefix = self._full_key(key)
        # Ensure prefix ends with '/' for proper delimiter-based listing
        if full_prefix and not full_prefix.endswith("/"):
            full_prefix += "/"

        children: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.bucket,
            Prefix=full_prefix,
            Delimiter="/",
        )
        for page in pages:
            # Common prefixes = "subdirectories"
            for cp in page.get("CommonPrefixes", []):
                # Strip the full prefix and trailing slash to get just the name
                name = cp["Prefix"][len(full_prefix):].rstrip("/")
                if name:
                    children.append(name)
            # Objects at this level (files, not directories)
            for obj in page.get("Contents", []):
                name = obj["Key"][len(full_prefix):]
                if name and "/" not in name:
                    children.append(name)

        return children

    def list_files_recursive(self, key: str) -> list[str]:
        """List all files recursively under a prefix using boto3 paginator (no Delimiter)."""
        client = self._get_client()
        full_prefix = self._full_key(key)
        if full_prefix and not full_prefix.endswith("/"):
            full_prefix += "/"

        files: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(full_prefix):]
                if rel:
                    files.append(rel)
        return files
