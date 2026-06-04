"""Hugging Face dataset-repo implementation of RemoteBackend.

Uses ``huggingface_hub`` APIs to store and retrieve experiment artifacts
in a dataset repository.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
import tempfile
import time

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from verl_inf_evolve.storage.hf_transfer import configure_hf_transfer_limits
from verl_inf_evolve.storage.remote_backend import RemoteBackend

logger = logging.getLogger(__name__)


class HFDatasetRemoteBackend(RemoteBackend):
    """RemoteBackend backed by a Hugging Face dataset repository.

    Args:
        repo_id: HF dataset repo identifier, e.g. ``"org/my-dataset"``.
        prefix: Path prefix within the repository.
        revision: Git revision / branch in the repo (default ``"main"``).
        token: HF API token (optional; uses cached login if not provided).
        auto_create_repo: When ``True``, create the dataset repo as public if
            it does not already exist. Existing repo visibility is left
            unchanged.
    """

    def __init__(
        self,
        repo_id: str,
        prefix: str = "",
        revision: str = "main",
        token: str | None = None,
        auto_create_repo: bool = False,
        upload_limit_mbps: float | None = None,
        download_limit_mbps: float | None = None,
        snapshot_max_workers: int | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.prefix = prefix
        self.revision = revision
        self._token = token  # never log this
        self._upload_limit_mbps = upload_limit_mbps
        self._download_limit_mbps = download_limit_mbps
        self._snapshot_max_workers = snapshot_max_workers
        configure_hf_transfer_limits(
            upload_limit_mbps=upload_limit_mbps,
            download_limit_mbps=download_limit_mbps,
        )
        self._api = HfApi(token=self._token)
        self._auto_create_repo = auto_create_repo

        if self._auto_create_repo:
            self._ensure_repo_exists()

    def _ensure_repo_exists(self) -> None:
        """Create the dataset repo as public if it is missing."""
        try:
            self._api.create_repo(
                repo_id=self.repo_id,
                repo_type="dataset",
                private=False,
                exist_ok=True,
            )
            logger.info("Ensured public HF dataset repo exists: %s", self.repo_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to ensure public HF dataset repo exists: {self.repo_id}"
            ) from e

    def _full_path(self, key: str) -> str:
        """Combine the backend prefix with a relative *key*."""
        if self.prefix:
            return f"{self.prefix}/{key}".rstrip("/") if key else self.prefix
        return key

    # ------------------------------------------------------------------
    # RemoteBackend implementation
    # ------------------------------------------------------------------

    def exists(self, key: str) -> bool:
        """Check whether a remote file exists using ``get_paths_info``."""
        full_path = self._full_path(key)
        try:
            info = self._api.get_paths_info(
                repo_id=self.repo_id,
                paths=[full_path],
                repo_type="dataset",
                revision=self.revision,
            )
            return len(info) > 0
        except (RepositoryNotFoundError, EntryNotFoundError):
            return False
        except Exception as e:
            logger.error("HF exists check failed for %s: %s", full_path, e)
            return False

    def download_bytes(self, key: str) -> bytes:
        """Download a file to a temp location and return its contents as bytes."""
        full_path = self._full_path(key)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_file = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=full_path,
                    repo_type="dataset",
                    revision=self.revision,
                    token=self._token,
                    local_dir=tmpdir,
                )
                with open(local_file, "rb") as f:
                    return f.read()
        except EntryNotFoundError:
            raise FileNotFoundError(
                f"HF object not found: {self.repo_id}/{full_path}"
            )
        except Exception as e:
            raise FileNotFoundError(
                f"HF download failed: {self.repo_id}/{full_path}"
            ) from e

    def download_file(self, key: str, local_path: str, max_retries: int = 3) -> bool:
        """Download a single file using ``hf_hub_download``."""
        full_path = self._full_path(key)
        for attempt in range(1, max_retries + 1):
            try:
                local_dir = os.path.dirname(local_path) or "."
                os.makedirs(local_dir, exist_ok=True)

                with tempfile.TemporaryDirectory() as tmpdir:
                    downloaded = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=full_path,
                        repo_type="dataset",
                        revision=self.revision,
                        token=self._token,
                        local_dir=tmpdir,
                    )
                    shutil.copy2(downloaded, local_path)
                return True
            except (EntryNotFoundError, RepositoryNotFoundError):
                logger.error("HF download_file failed for %s: not found", full_path)
                return False
            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "HF download_file attempt %d/%d failed for %s: %s — retrying in %ds",
                        attempt, max_retries, full_path, e, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "HF download_file failed for %s after %d attempts: %s",
                        full_path, max_retries, e,
                    )
                    return False

    def download_dir(self, key: str, local_path: str, max_retries: int = 3) -> bool:
        """Download a directory subtree using ``snapshot_download`` with allow_patterns.

        Uses a staging directory on the same filesystem as *local_path* so that
        files can be **moved** (``shutil.move`` → ``os.rename``) rather than
        copied.  This avoids temporarily doubling ephemeral-storage usage and
        prevents leaked temp dirs under ``/tmp/`` when the process is killed.
        """
        full_path = self._full_path(key)
        # Scope to the subtree
        pattern = f"{full_path}/**" if full_path else None
        allow_patterns = [pattern] if pattern else None

        # Place the staging dir next to the target so os.rename works
        # (same filesystem).  The predictable suffix makes stale dirs easy to
        # identify and clean up after a crash.
        staging_dir = local_path.rstrip("/") + "._hf_staging"

        for attempt in range(1, max_retries + 1):
            try:
                os.makedirs(local_path, exist_ok=True)
                # Clean up any leftover staging dir from a previous crash
                shutil.rmtree(staging_dir, ignore_errors=True)
                os.makedirs(staging_dir, exist_ok=True)

                try:
                    snapshot_kwargs: dict[str, object] = {}
                    if self._snapshot_max_workers is not None:
                        snapshot_kwargs["max_workers"] = self._snapshot_max_workers
                    snapshot_dir = snapshot_download(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        revision=self.revision,
                        token=self._token,
                        allow_patterns=allow_patterns,
                        local_dir=staging_dir,
                        **snapshot_kwargs,
                    )
                    # Move only the subtree to the target path
                    src = os.path.join(snapshot_dir, full_path) if full_path else snapshot_dir
                    if os.path.isdir(src):
                        for item in os.listdir(src):
                            s = os.path.join(src, item)
                            d = os.path.join(local_path, item)
                            if os.path.exists(d):
                                if os.path.isdir(d):
                                    shutil.rmtree(d)
                                else:
                                    os.unlink(d)
                            shutil.move(s, d)
                    else:
                        logger.warning("HF download_dir: subtree %s not found in snapshot", full_path)
                        return False
                finally:
                    shutil.rmtree(staging_dir, ignore_errors=True)

                return True
            except (EntryNotFoundError, RepositoryNotFoundError):
                logger.error("HF download_dir failed for %s: not found", full_path)
                return False
            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "HF download_dir attempt %d/%d failed for %s: %s — retrying in %ds",
                        attempt, max_retries, full_path, e, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "HF download_dir failed for %s after %d attempts: %s",
                        full_path, max_retries, e,
                    )
                    return False

    def upload_bytes(self, data: bytes, key: str) -> bool:
        """Upload bytes using ``HfApi.upload_file``."""
        full_path = self._full_path(key)
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                self._api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo=full_path,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                )
            except Exception:
                # Retry with create_pr for repos that require PR-based writes.
                self._api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo=full_path,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    create_pr=True,
                )
            finally:
                os.unlink(tmp_path)
            return True
        except Exception as e:
            logger.error("HF upload_bytes failed for %s: %s", full_path, e)
            return False

    def upload_file(self, local_path: str, key: str) -> bool:
        """Upload a single local file using ``HfApi.upload_file``."""
        full_path = self._full_path(key)
        try:
            self._api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=full_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
            )
            return True
        except Exception:
            pass
        # Retry with create_pr for repos that require PR-based writes.
        try:
            self._api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=full_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                create_pr=True,
            )
            return True
        except Exception as e:
            logger.error("HF upload_file failed for %s: %s", full_path, e)
            return False

    def upload_dir(self, local_path: str, key: str) -> bool:
        """Upload a local directory tree using ``HfApi.upload_folder``."""
        full_path = self._full_path(key)
        try:
            self._api.upload_folder(
                folder_path=local_path,
                path_in_repo=full_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
            )
            return True
        except Exception:
            pass
        try:
            self._api.upload_folder(
                folder_path=local_path,
                path_in_repo=full_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                create_pr=True,
            )
            return True
        except Exception as e:
            logger.error("HF upload_dir failed for %s: %s", full_path, e)
            return False

    def delete_prefix(self, key: str, *, exclude_patterns: list[str] | None = None) -> bool:
        """Delete files under prefix, respecting exclude_patterns."""
        full_path = self._full_path(key)
        try:
            # List all files under the prefix
            all_items = list(self._api.list_repo_tree(
                repo_id=self.repo_id,
                path_in_repo=full_path,
                repo_type="dataset",
                revision=self.revision,
                recursive=True,
            ))
            # Filter to files only (items with 'size' attribute)
            file_paths = [item.path for item in all_items if hasattr(item, "size")]

            if exclude_patterns:
                # Filter out files matching any exclude pattern
                # Match on the relative path from the prefix
                def is_excluded(fpath: str) -> bool:
                    rel = fpath[len(full_path):].lstrip("/") if full_path and fpath.startswith(full_path) else fpath
                    return any(fnmatch.fnmatch(rel, pat) for pat in exclude_patterns)

                file_paths = [fp for fp in file_paths if not is_excluded(fp)]

            if not file_paths:
                return True

            # Delete in a single commit
            operations = []
            from huggingface_hub import CommitOperationDelete
            for fp in file_paths:
                operations.append(CommitOperationDelete(path_in_repo=fp))

            self._api.create_commit(
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                operations=operations,
                commit_message=f"Delete prefix: {full_path}",
            )
            return True
        except Exception as e:
            logger.error("HF delete_prefix failed for %s: %s", full_path, e)
            return False

    def list_immediate_children(self, key: str) -> list[str]:
        """List immediate child names under prefix using ``list_repo_tree``."""
        full_path = self._full_path(key)
        try:
            items = list(self._api.list_repo_tree(
                repo_id=self.repo_id,
                path_in_repo=full_path or None,
                repo_type="dataset",
                revision=self.revision,
                recursive=False,
            ))
            # Extract just the child name (last component)
            children: list[str] = []
            for item in items:
                name = item.path
                # Strip the prefix to get the relative name
                if full_path and name.startswith(full_path + "/"):
                    name = name[len(full_path) + 1:]
                elif full_path and name.startswith(full_path):
                    name = name[len(full_path):]
                # Only keep direct children (no nested paths)
                if "/" not in name and name:
                    children.append(name)
            return children
        except Exception as e:
            logger.error("HF list_immediate_children failed for %s: %s", full_path, e)
            return []

    def list_files_recursive(self, key: str) -> list[str]:
        """List all files recursively under prefix using ``list_repo_tree(recursive=True)``."""
        full_path = self._full_path(key)
        try:
            items = self._api.list_repo_tree(
                repo_id=self.repo_id,
                path_in_repo=full_path or None,
                repo_type="dataset",
                revision=self.revision,
                recursive=True,
            )
            files: list[str] = []
            for item in items:
                if not hasattr(item, "size"):  # skip directories
                    continue
                rel = item.path
                if full_path:
                    rel = rel[len(full_path) + 1:]
                if rel:
                    files.append(rel)
            return files
        except Exception as e:
            logger.error("HF list_files_recursive failed for %s: %s", full_path, e)
            return []
