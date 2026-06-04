"""Non-blocking upload manager for stage outputs.

Runs a background daemon thread that drains a queue of upload tasks.
Two upload paths:

1. **Memory path** (DataProto, JSON): ``pickle.dump`` / ``json.dumps`` →
   ``BytesIO`` → ``backend.upload_bytes()`` — zero disk I/O.
2. **Directory path** (momentum shards, checkpoints): ``backend.upload_dir()``
   for multi-file transfers.

All remote I/O is delegated to a ``RemoteBackend`` instance, which is created
automatically from the ``remote_sync_path`` URI via ``create_remote_backend()``.

Backpressure: tracks ``_pending_bytes`` (sum of serialized buffer sizes).
``submit_memory_upload()`` blocks when ``_pending_bytes > max_pending_bytes``.

Usage::

    mgr = StageUploadManager("s3://bucket/experiment")
    mgr.start()

    # Non-blocking: serialize in calling thread, upload in background
    task_id = mgr.submit_memory_upload("dev_output", data, "dataproto",
                                        "ans_0/dev_output.pt")

    # Async small-file upload (state.json — queued after outputs)
    mgr.submit_file_upload(local_path, "ans_0/state.json")

    # Download on resume
    data = mgr.download_to_memory("ans_0/dev_output.pt", "dataproto")

    mgr.shutdown()
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class UploadTask:
    """A single upload work item."""

    task_id: str
    remote_key: str
    # Memory upload: serialized bytes
    bytes_buf: bytes | None = None
    bytes_size: int = 0
    # Directory upload: local path
    local_path: str | None = None
    # If set, delete this local path after successful upload
    cleanup_local_path: str | None = None
    # If set, also delete cleanup_local_path when the task itself fails
    cleanup_on_failure: bool = False
    # Remote deletion task: exclude these patterns from deletion
    remote_delete_exclude: list[str] | None = None
    # Whether this task is a remote deletion (vs upload)
    is_delete: bool = False
    # If set, skip this task when the dependency task failed
    depends_on: tuple[str, ...] = ()
    # Logical task kind used for failure policy hooks
    task_kind: str = "generic"


class StageUploadManager:
    """Background upload manager for stage outputs to a remote backend.

    Args:
        remote_sync_path: Remote URI, e.g. ``s3://bucket/experiment_name`` or
            ``hf://datasets/org/repo/prefix``.
            If ``None``, uploads are disabled (no-op mode).
        max_pending_gb: Backpressure threshold — block ``submit_memory_upload``
            when pending serialized bytes exceed this.
    """

    def __init__(
        self,
        remote_sync_path: str | None,
        max_pending_gb: float = 100.0,
        disable_upload: bool = False,
        backend_kwargs: dict[str, Any] | None = None,
        checkpoint_failure_callback: Callable[[UploadTask, Exception], None] | None = None,
    ):
        self._remote_sync_path = remote_sync_path
        self._max_pending_bytes = int(max_pending_gb * 1024**3)
        self._upload_disabled = disable_upload

        # Create backend from URI
        self._backend: Optional["RemoteBackend"] = None
        if remote_sync_path:
            from verl_inf_evolve.storage.remote_backend import create_remote_backend

            backend_kwargs = dict(backend_kwargs or {})
            if str(remote_sync_path).startswith("hf://") and "auto_create_repo" not in backend_kwargs:
                backend_kwargs["auto_create_repo"] = not disable_upload
            self._backend = create_remote_backend(remote_sync_path, **backend_kwargs)

        # Thread + queue
        self._queue: list[UploadTask] = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._checkpoint_failure_callback = checkpoint_failure_callback

        # Backpressure tracking
        self._pending_bytes = 0

        # Task completion tracking
        self._results: dict[str, bool] = {}  # task_id -> success
        self._results_cond = threading.Condition()

    @property
    def backend(self) -> Optional["RemoteBackend"]:
        """The underlying RemoteBackend instance, or None if not configured."""
        return self._backend

    @property
    def remote_configured(self) -> bool:
        """True if a remote path was provided and backend is available."""
        return self._remote_sync_path is not None and self._backend is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def upload_enabled(self) -> bool:
        """True if remote is configured and uploads are not disabled."""
        return self.remote_configured and not self._upload_disabled

    def start(self) -> None:
        """Start the background upload thread."""
        if not self.remote_configured:
            logger.info("StageUploadManager: no remote_sync_path; uploads disabled")
            return
        if self._upload_disabled:
            logger.info("StageUploadManager: disable_upload=True; uploads disabled (downloads still active)")
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="stage-upload", daemon=True
        )
        self._thread.start()
        logger.info(
            "StageUploadManager started (remote=%s, backpressure=%.1f GB)",
            self._remote_sync_path,
            self._max_pending_bytes / 1024**3,
        )

    def shutdown(self, timeout: float = 3600.0) -> None:
        """Stop the background thread and wait for pending uploads."""
        if self._thread is None or not self._thread.is_alive():
            return

        self._stop.set()
        with self._cond:
            self._cond.notify_all()

        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("StageUploadManager: thread did not stop within %.0fs", timeout)

    def queue_status(self) -> dict:
        """Return a snapshot of the upload queue for diagnostic logging."""
        with self._cond:
            return {
                "pending": len(self._queue),
                "pending_bytes": self._pending_bytes,
                "completed": getattr(self, "_completed_count", 0),
                "failed": getattr(self, "_failed_count", 0),
            }

    # ------------------------------------------------------------------
    # Upload — non-blocking (queued)
    # ------------------------------------------------------------------

    def submit_memory_upload(
        self,
        name: str,
        data: Any,
        kind: str,
        remote_key: str,
        *,
        depends_on: str | list[str] | tuple[str, ...] | None = None,
        task_kind: str = "memory_upload",
    ) -> str:
        """Serialize *data* in the calling thread and queue bytes for upload.

        Serialization happens in the calling thread to avoid race conditions
        (the training loop may modify the DataProto before the upload thread
        gets to it).

        Args:
            name: Human-readable label (for logging).
            data: Object to serialize (DataProto or JSON-serializable).
            kind: ``"dataproto"`` or ``"json"``.
            remote_key: Remote object key relative to experiment prefix.

        Returns:
            Task ID for tracking.
        """
        if not self.upload_enabled:
            return ""

        buf = self._serialize(data, kind)
        buf_size = len(buf)

        # Backpressure: wait if too much data is pending
        with self._cond:
            while self._pending_bytes + buf_size > self._max_pending_bytes:
                logger.info(
                    "Backpressure: pending=%.1f GB, waiting (max=%.1f GB)",
                    self._pending_bytes / 1024**3,
                    self._max_pending_bytes / 1024**3,
                )
                self._cond.wait(timeout=5.0)
                if self._stop.is_set():
                    break

        task_id = uuid4().hex[:12]
        task = UploadTask(
            task_id=task_id,
            remote_key=remote_key,
            bytes_buf=buf,
            bytes_size=buf_size,
            depends_on=self._normalize_dependencies(depends_on),
            task_kind=task_kind,
        )

        with self._cond:
            self._pending_bytes += buf_size
            self._queue.append(task)
            pending_tasks = len(self._queue)
            pending_gb = self._pending_bytes / 1024**3
            self._cond.notify()

        logger.info(
            "Queued memory upload: %s (%.1f MB) → %s [%s] | kind=%s | "
            "queue: pending=%d bytes=%.2fGB | depends_on=%d",
            name,
            buf_size / 1024**2,
            remote_key,
            task_id,
            task_kind,
            pending_tasks,
            pending_gb,
            len(task.depends_on),
        )
        return task_id

    def submit_dir_upload(
        self,
        local_path: str,
        remote_key: str,
        *,
        cleanup_after: bool = False,
        cleanup_on_failure: bool = False,
        depends_on: str | list[str] | tuple[str, ...] | None = None,
        task_kind: str = "generic",
    ) -> str:
        """Queue a directory upload.

        Args:
            local_path: Local directory to upload.
            remote_key: Remote object key relative to experiment prefix.
            cleanup_after: If ``True``, delete *local_path* after a successful
                upload.  Useful for temp directories that should not persist.
        """
        if not self.upload_enabled:
            if cleanup_after:
                shutil.rmtree(local_path, ignore_errors=True)
            return ""

        # Estimate directory size for backpressure tracking
        dir_size = self._dir_size_bytes(local_path)

        # Backpressure: wait if too much data is pending upload
        with self._cond:
            while self._pending_bytes + dir_size > self._max_pending_bytes:
                logger.warning(
                    "Backpressure (dir): pending=%.1f GB + dir=%.1f GB > max=%.1f GB, waiting",
                    self._pending_bytes / 1024**3,
                    dir_size / 1024**3,
                    self._max_pending_bytes / 1024**3,
                )
                self._cond.wait(timeout=5.0)
                if self._stop.is_set():
                    break

        task_id = uuid4().hex[:12]
        task = UploadTask(
            task_id=task_id,
            remote_key=remote_key,
            local_path=local_path,
            bytes_size=dir_size,
            cleanup_local_path=local_path if cleanup_after else None,
            cleanup_on_failure=cleanup_on_failure,
            depends_on=self._normalize_dependencies(depends_on),
            task_kind=task_kind,
        )

        with self._cond:
            self._pending_bytes += dir_size
            self._queue.append(task)
            pending_tasks = len(self._queue)
            pending_gb = self._pending_bytes / 1024**3
            self._cond.notify()

        logger.warning(
            "Queued dir upload: %s (%.1f GB) → %s [%s] | kind=%s | "
            "queue: pending=%d bytes=%.2fGB | depends_on=%d",
            local_path,
            dir_size / 1024**3,
            remote_key,
            task_id,
            task_kind,
            pending_tasks,
            pending_gb,
            len(task.depends_on),
        )
        return task_id

    def submit_file_upload(
        self,
        local_path: str,
        remote_key: str,
        *,
        depends_on: str | list[str] | tuple[str, ...] | None = None,
        task_kind: str = "file_upload",
    ) -> str:
        """Read a local file in the calling thread and queue bytes for async upload.

        Useful for small files like ``state.json`` that should be uploaded
        **after** stage output uploads in the FIFO queue (crash-ordering safety).

        Args:
            local_path: Path to the local file.
            remote_key: Remote object key relative to experiment prefix.
            depends_on: If set, skip this upload when the referenced task failed.

        Returns:
            Task ID for tracking.
        """
        if not self.upload_enabled:
            return ""

        with open(local_path, "rb") as f:
            buf = f.read()

        task_id = uuid4().hex[:12]
        task = UploadTask(
            task_id=task_id,
            remote_key=remote_key,
            bytes_buf=buf,
            bytes_size=len(buf),
            depends_on=self._normalize_dependencies(depends_on),
            task_kind=task_kind,
        )

        with self._cond:
            self._pending_bytes += len(buf)
            self._queue.append(task)
            pending_tasks = len(self._queue)
            pending_gb = self._pending_bytes / 1024**3
            self._cond.notify()

        logger.info(
            "Queued file upload: %s (%.1f KB) → %s [%s] | kind=%s | "
            "queue: pending=%d bytes=%.2fGB | depends_on=%d",
            local_path,
            len(buf) / 1024,
            remote_key,
            task_id,
            task_kind,
            pending_tasks,
            pending_gb,
            len(task.depends_on),
        )
        return task_id

    def submit_checkpoint_upload(
        self, local_dir: str, remote_key: str, *, cleanup_after: bool = False,
    ) -> str:
        """Queue a checkpoint directory upload."""
        return self.submit_dir_upload(
            local_dir,
            remote_key,
            cleanup_after=cleanup_after,
            cleanup_on_failure=cleanup_after,
            task_kind="checkpoint_upload",
        )

    def submit_remote_delete(
        self, remote_key: str, *, exclude_patterns: list[str] | None = None,
        cleanup_local_path: str | None = None,
        depends_on: str | None = None,
    ) -> str:
        """Queue a remote deletion task.

        Deletes all files under *remote_key* on remote, **except** those matching
        *exclude_patterns*.  Runs in the same FIFO queue as uploads, so
        ordering is preserved (e.g. marker upload completes before deletion).

        Args:
            remote_key: Key (relative to experiment prefix) to delete from.
            exclude_patterns: Glob patterns for files to *exclude* from deletion.
            cleanup_local_path: If set, delete this local directory after the
                remote deletion task completes (FIFO-ordered after preceding
                uploads, so the newer checkpoint is safely on remote first).
            depends_on: If set, skip this task when the referenced task failed.
                Used to avoid deleting ``global_step_{N-1}`` when the upload
                of ``global_step_N`` did not succeed.

        Returns:
            Task ID for tracking.
        """
        if not self.upload_enabled:
            # Still clean up locally even if uploads are disabled
            if cleanup_local_path:
                shutil.rmtree(cleanup_local_path, ignore_errors=True)
                logger.info("Cleaned up local dir (uploads disabled): %s", cleanup_local_path)
            return ""

        task_id = uuid4().hex[:12]
        task = UploadTask(
            task_id=task_id,
            remote_key=remote_key,
            remote_delete_exclude=exclude_patterns,
            cleanup_local_path=cleanup_local_path,
            depends_on=self._normalize_dependencies(depends_on),
            is_delete=True,
            task_kind="remote_delete",
        )

        with self._cond:
            self._queue.append(task)
            pending_tasks = len(self._queue)
            self._cond.notify()

        logger.info(
            "Queued remote delete: %s (exclude=%s) [%s] | queue: pending=%d | "
            "depends_on=%d",
            remote_key,
            exclude_patterns,
            task_id,
            pending_tasks,
            len(task.depends_on),
        )
        return task_id

    # ------------------------------------------------------------------
    # Upload — synchronous (small files)
    # ------------------------------------------------------------------

    def upload_file_sync(self, local_path: str, remote_key: str) -> bool:
        """Upload a small local file synchronously (e.g. state.json)."""
        if not self.upload_enabled or self._backend is None:
            return False

        try:
            with open(local_path, "rb") as f:
                return self._backend.upload_bytes(f.read(), remote_key)
        except Exception as e:
            logger.error("Sync upload failed for %s: %s", remote_key, e)
            return False

    def upload_bytes_sync(self, data: bytes, remote_key: str) -> bool:
        """Upload raw bytes synchronously."""
        if not self.upload_enabled or self._backend is None:
            return False

        try:
            return self._backend.upload_bytes(data, remote_key)
        except Exception as e:
            logger.error("Sync upload failed for %s: %s", remote_key, e)
            return False

    # ------------------------------------------------------------------
    # Download — synchronous (for resume)
    # ------------------------------------------------------------------

    def download_to_memory(self, remote_key: str, kind: str) -> Any | None:
        """Download bytes from remote and deserialize in memory. No local disk.

        Args:
            remote_key: Remote object key relative to experiment prefix.
            kind: ``"dataproto"`` or ``"json"``.

        Returns:
            Deserialized object, or ``None`` on failure.
        """
        if not self.remote_configured or self._backend is None:
            return None

        try:
            raw = self._backend.download_bytes(remote_key)
        except FileNotFoundError:
            logger.warning("Download failed for %s: not found", remote_key)
            return None
        except Exception as e:
            logger.warning("Download failed for %s: %s", remote_key, e)
            return None

        return self._deserialize(raw, kind)

    def download_dir_to_local(self, remote_key: str, local_path: str) -> bool:
        """Download a directory from remote to a local path."""
        if not self.remote_configured or self._backend is None:
            return False

        os.makedirs(local_path, exist_ok=True)
        return self._backend.download_dir(remote_key, local_path)

    def download_file_to_local(self, remote_key: str, local_path: str) -> bool:
        """Download a single file from remote to a local path."""
        if not self.remote_configured or self._backend is None:
            return False

        try:
            return self._backend.download_file(remote_key, local_path)
        except Exception as e:
            logger.warning("Download file failed for %s: %s", remote_key, e)
            return False

    def remote_exists(self, remote_key: str) -> bool:
        """Check if a remote key exists."""
        if not self.remote_configured or self._backend is None:
            return False

        return self._backend.exists(remote_key)

    # ------------------------------------------------------------------
    # Task tracking
    # ------------------------------------------------------------------

    def wait_for_task(self, task_id: str, timeout: float | None = None) -> bool:
        """Wait for a specific upload task to complete.

        Returns:
            ``True`` if the task succeeded, ``False`` if it failed or timed out.
        """
        if not task_id:
            return True  # no-op mode

        deadline = time.monotonic() + timeout if timeout else None

        with self._results_cond:
            while task_id not in self._results:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                self._results_cond.wait(timeout=remaining or 5.0)

            return self._results[task_id]

    def wait_all_pending(self, timeout: float = 600.0) -> None:
        """Wait for all currently pending uploads to complete."""
        with self._cond:
            pending_ids = [t.task_id for t in self._queue]

        for tid in pending_ids:
            self.wait_for_task(tid, timeout=timeout)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _dir_size_bytes(path: str) -> int:
        """Estimate total size of a directory tree in bytes."""
        total = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for fname in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fname))
                except OSError:
                    pass
        return total

    @staticmethod
    def _serialize(data: Any, kind: str) -> bytes:
        """Serialize data to bytes. Called in the main thread."""
        buf = io.BytesIO()
        if kind == "dataproto":
            pickle.dump(data, buf, protocol=pickle.HIGHEST_PROTOCOL)
        elif kind == "json":
            buf.write(json.dumps(data).encode("utf-8"))
        else:
            raise ValueError(f"Unknown serialization kind: {kind}")
        return buf.getvalue()

    @staticmethod
    def _deserialize(raw: bytes, kind: str) -> Any:
        """Deserialize bytes back to an object."""
        buf = io.BytesIO(raw)
        if kind == "dataproto":
            return pickle.load(buf)
        elif kind == "json":
            return json.loads(raw)
        else:
            raise ValueError(f"Unknown deserialization kind: {kind}")

    def _complete(self, task_id: str, success: bool) -> None:
        """Record task completion and wake waiters."""
        if success:
            self._completed_count = getattr(self, "_completed_count", 0) + 1
        else:
            self._failed_count = getattr(self, "_failed_count", 0) + 1
        with self._results_cond:
            self._results[task_id] = success
            self._results_cond.notify_all()

    @staticmethod
    def _normalize_dependencies(
        depends_on: str | list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if depends_on is None:
            return ()
        if isinstance(depends_on, str):
            return (depends_on,)
        return tuple(dep for dep in depends_on if dep)

    def _dependency_failed(self, task: UploadTask) -> bool:
        if not task.depends_on:
            return False
        return any(not self._results.get(dep, True) for dep in task.depends_on)

    def _cleanup_local_path(self, task: UploadTask) -> None:
        if not task.cleanup_local_path:
            return
        try:
            shutil.rmtree(task.cleanup_local_path, ignore_errors=True)
            logger.info("Cleaned up local dir: %s", task.cleanup_local_path)
        except Exception as cleanup_err:
            logger.warning(
                "Failed to clean up %s: %s",
                task.cleanup_local_path, cleanup_err,
            )

    def _run_loop(self) -> None:
        """Background thread main loop."""
        logger.info("Upload thread started")

        while True:
            task: UploadTask | None = None

            with self._cond:
                while not self._queue and not self._stop.is_set():
                    self._cond.wait(timeout=1.0)

                if self._queue:
                    task = self._queue.pop(0)
                    queue_remaining = len(self._queue)
                    pending_before_gb = self._pending_bytes / 1024**3
                elif self._stop.is_set():
                    break

            if task is None:
                continue

            # Skip tasks whose dependency failed (e.g. don't delete
            # global_step_{N-1} if upload of global_step_N failed).
            if self._dependency_failed(task):
                logger.warning(
                    "Skipping task %s (%s): dependency %s failed",
                    task.task_id, task.remote_key, ",".join(task.depends_on),
                )
                self._complete(task.task_id, success=False)
                continue

            logger.info(
                "Upload task start: kind=%s | remote=%s [%s] | size=%.1f MB | "
                "queue: remaining=%d bytes=%.2fGB | depends_on=%s | delete=%s",
                task.task_kind,
                task.remote_key,
                task.task_id,
                task.bytes_size / 1024**2,
                queue_remaining,
                pending_before_gb,
                ",".join(task.depends_on) if task.depends_on else "-",
                task.is_delete,
            )
            task_start = time.perf_counter()
            task_success = False
            task_error: Exception | None = None
            try:
                if task.is_delete:
                    self._delete_remote(task)
                elif task.bytes_buf is not None:
                    self._upload_bytes(task)
                elif task.local_path is not None:
                    self._upload_dir(task)

                self._complete(task.task_id, success=True)
                task_success = True

                # Clean up local directory after successful upload
                if task.cleanup_local_path:
                    self._cleanup_local_path(task)
            except Exception as e:
                task_error = e
                self._complete(task.task_id, success=False)
                if task.cleanup_local_path and task.cleanup_on_failure:
                    self._cleanup_local_path(task)
                if (
                    task.task_kind == "checkpoint_upload"
                    and self._checkpoint_failure_callback is not None
                ):
                    try:
                        self._checkpoint_failure_callback(task, e)
                    except Exception:
                        logger.exception(
                            "Checkpoint failure callback raised for %s",
                            task.remote_key,
                        )
            finally:
                # Release backpressure
                with self._cond:
                    if task.bytes_size > 0:
                        self._pending_bytes -= task.bytes_size
                    pending_after = len(self._queue)
                    pending_after_gb = self._pending_bytes / 1024**3
                    self._cond.notify_all()

                elapsed = time.perf_counter() - task_start
                if task_success:
                    logger.info(
                        "Upload task done: kind=%s | remote=%s [%s] | elapsed=%.1fs | "
                        "queue: pending=%d bytes=%.2fGB",
                        task.task_kind,
                        task.remote_key,
                        task.task_id,
                        elapsed,
                        pending_after,
                        pending_after_gb,
                    )
                elif task_error is not None:
                    logger.error(
                        "Upload task failed: kind=%s | remote=%s [%s] | elapsed=%.1fs | "
                        "error=%s | queue: pending=%d bytes=%.2fGB",
                        task.task_kind,
                        task.remote_key,
                        task.task_id,
                        elapsed,
                        task_error,
                        pending_after,
                        pending_after_gb,
                    )

        logger.info("Upload thread stopped")

    def _upload_bytes(self, task: UploadTask) -> None:
        """Upload serialized bytes via backend."""
        if self._backend is None:
            raise RuntimeError("Remote backend not available")

        ok = self._backend.upload_bytes(task.bytes_buf, task.remote_key)
        if not ok:
            raise RuntimeError(f"upload_bytes failed for {task.remote_key}")
        # Free serialized bytes after upload
        task.bytes_buf = None
        logger.info("Uploaded %s (%.1f MB)", task.remote_key, task.bytes_size / 1024**2)

    def _upload_dir(self, task: UploadTask) -> None:
        """Upload a directory via backend."""
        if self._backend is None:
            raise RuntimeError("Remote backend not available")

        ok = self._backend.upload_dir(task.local_path, task.remote_key)
        if not ok:
            raise RuntimeError(f"upload_dir failed: {task.local_path} -> {task.remote_key}")
        logger.info("Uploaded dir %s → %s", task.local_path, task.remote_key)

    def _delete_remote(self, task: UploadTask) -> None:
        """Delete files on remote, respecting exclude patterns.

        Failure is logged as a warning but does NOT raise — deletion failure is
        a storage-waste issue, not a correctness issue.
        """
        if self._backend is None:
            logger.warning("Remote backend not configured; skipping remote delete")
            return

        ok = self._backend.delete_prefix(
            task.remote_key, exclude_patterns=task.remote_delete_exclude,
        )
        if ok:
            logger.info(
                "Deleted remote files: %s (exclude=%s)",
                task.remote_key, task.remote_delete_exclude,
            )
        else:
            logger.warning(
                "Remote delete failed (non-fatal): %s", task.remote_key,
            )
