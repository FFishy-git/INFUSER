"""Stage context manager for crash-recoverable training stages.

Wraps each training stage in a ``with`` block that:

- On **enter**: checks if the stage is already done (resume). If so, downloads
  outputs from R2 into memory and sets ``should_run = False``.
- On **exit** (success): marks the stage done, persists locally only when
  remote uploads are disabled, and otherwise keeps outputs in memory until
  the background uploader has accepted them. Remote ``state.json`` is queued
  only after the corresponding artifact uploads succeed.

Usage::

    with trainer.stage_ctx(
        name="dev_rollout", stage_id=1, resume=resume,
        is_done=lambda: resume.stage_1_done,
        mark_done=lambda: setattr(resume, 'stage_1_done', True),
        ans_loop=ans_loop,
    ) as ctx:
        if ctx.should_run:
            dev_output = ...
            ctx.save("dev_output", dev_output)

    dev_output = ctx.result("dev_output")

For gen-loop stages, use ``gen_stage_ctx()`` which adds gen-loop scoping
to the R2 key paths.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from typing import Any, Callable, Optional

from verl import DataProto

from verl_inf_evolve.trainer.resume_state import ResumeState
from verl_inf_evolve.storage.stage_upload_manager import StageUploadManager
from verl_inf_evolve.storage.storage_resolver import StorageResolver

logger = logging.getLogger(__name__)


class StageContext:
    """Context manager wrapping one training stage.

    Args:
        name: Human-readable stage name (e.g. ``"dev_rollout"``).
        stage_id: Numeric stage identifier for logging.
        resume: Current ``ResumeState`` instance.
        resume_dir: Local path to resume state directory.
        upload_manager: ``StageUploadManager`` (may be ``None`` if no remote).
        is_done: Callable returning ``True`` if this stage was already completed.
        mark_done: Callable to mark this stage as done in ``ResumeState``.
        ans_loop: Current answer loop index (for logging / R2 key paths).
        gen_loop_prefix: Optional subdirectory for gen-loop stages, e.g.
            ``"gen_0"``. Affects R2 key layout.
        defer_state_update: When ``True``, ``__exit__`` skips marking the stage
            done, saving ``state.json``, and uploading it to R2.  The caller is
            responsible for marking done and uploading state.json later (used
            for PPO stages 5/6 so state.json reaches R2 only after the
            checkpoint containing the weight updates).
        should_upload_remote: When ``False``, keep the normal local persistence
            behavior but skip remote uploads for this stage's outputs and
            ``state.json``.
        timing_dict: Optional dict to accumulate wall-clock timing into.
            When provided, ``__enter__`` records the start time and
            ``__exit__`` writes ``timing_dict[timer_name] += elapsed``.
        timer_name: Key used in *timing_dict*.  Defaults to *name*.
    """

    should_run: bool

    def __init__(
        self,
        name: str,
        stage_id: int,
        resume: ResumeState,
        resume_dir: str,
        upload_manager: StageUploadManager | None,
        is_done: Callable[[], bool],
        mark_done: Callable[[], None],
        ans_loop: int,
        gen_loop_prefix: str | None = None,
        defer_state_update: bool = False,
        should_upload_remote: bool = True,
        timing_dict: dict[str, float] | None = None,
        timer_name: str | None = None,
    ):
        self._name = name
        self._stage_id = stage_id
        self._resume = resume
        self._resume_dir = resume_dir
        self._upload_manager = upload_manager
        self._is_done = is_done
        self._mark_done = mark_done
        self._ans_loop = ans_loop
        self._gen_loop_prefix = gen_loop_prefix
        self._defer_state_update = defer_state_update
        self._should_upload_remote = should_upload_remote
        self._timing_dict = timing_dict
        self._timer_name = timer_name or name
        self._t_start: float | None = None

        # Output storage (in memory)
        self._outputs: dict[str, Any] = {}
        self._output_kinds: dict[str, str] = {}  # name -> "dataproto" | "json" | "directory"

        # Temp directories created during resume downloads (for cleanup)
        self._temp_dirs: list[str] = []

    # ------------------------------------------------------------------
    # Save methods (called inside the `with` block)
    # ------------------------------------------------------------------

    def save(self, name: str, data: DataProto) -> None:
        """Register a DataProto output. Held in memory only."""
        self._outputs[name] = data
        self._output_kinds[name] = "dataproto"

    def save_json(self, name: str, data: Any) -> None:
        """Register a JSON-serializable output."""
        self._outputs[name] = data
        self._output_kinds[name] = "json"

    def save_dir(self, name: str, local_path: str) -> None:
        """Register a directory output (e.g. momentum files written by workers).

        The directory is uploaded via rclone (parallel multi-file transfers).
        """
        self._outputs[name] = local_path  # store path string
        self._output_kinds[name] = "directory"

    # ------------------------------------------------------------------
    # Result methods (called after the `with` block)
    # ------------------------------------------------------------------

    def result(self, name: str) -> DataProto:
        """Get a DataProto output from memory (or download from R2 on resume).

        Works both for freshly computed outputs and for resumed stages.
        On resume, attempts to download from R2 if not already in memory.
        """
        if name not in self._outputs:
            # Try downloading from R2 (resume case)
            if self._ensure_downloaded(name, "dataproto"):
                return self._outputs[name]
            raise KeyError(
                f"Stage '{self._name}' has no output '{name}' and R2 download failed. "
                f"Available: {list(self._outputs.keys())}"
            )
        return self._outputs[name]

    def result_json(self, name: str) -> Any:
        """Get a JSON output from memory (or download from R2 on resume)."""
        if name not in self._outputs:
            if self._ensure_downloaded(name, "json"):
                return self._outputs[name]
            raise KeyError(
                f"Stage '{self._name}' has no JSON output '{name}' and R2 download failed. "
                f"Available: {list(self._outputs.keys())}"
            )
        return self._outputs[name]

    def result_dir(self, name: str) -> str:
        """Get a directory output path (or download from R2 on resume)."""
        if name not in self._outputs:
            if self._ensure_downloaded(name, "directory"):
                return self._outputs[name]
            raise KeyError(
                f"Stage '{self._name}' has no dir output '{name}' and R2 download failed. "
                f"Available: {list(self._outputs.keys())}"
            )
        return self._outputs[name]

    def has_result(self, name: str) -> bool:
        """Check if an output is available (in memory)."""
        return name in self._outputs

    # ------------------------------------------------------------------
    # R2 key helpers
    # ------------------------------------------------------------------

    def _remote_key(self, name: str) -> str:
        """Build the R2 object key for an output.

        Layout::

            ans_0/{name}.pt              # ans-level stage
            ans_0/gen_0/{name}.pt        # gen-loop stage
            ans_0/gen_0/{name}.json      # gen-loop JSON
            ans_0/momentum/              # directory
        """
        kind = self._output_kinds.get(name, "dataproto")
        if kind == "json":
            suffix = f"{name}.json"
        elif kind == "directory":
            suffix = f"{name}/"
        else:
            suffix = f"{name}.pt"

        prefix = f"ans_{self._ans_loop}"
        if self._gen_loop_prefix:
            return f"{prefix}/{self._gen_loop_prefix}/{suffix}"
        return f"{prefix}/{suffix}"

    def _local_path(self, name: str, kind: str) -> str:
        """Compute local file path for an output under ``resume_dir``.

        Layout mirrors the R2 key structure::

            {resume_dir}/{name}.pt              # ans-level DataProto
            {resume_dir}/gen_0/{name}.pt        # gen-loop DataProto
            {resume_dir}/{name}.json            # JSON output
        """
        if kind == "json":
            filename = f"{name}.json"
        elif kind == "directory":
            filename = name
        else:
            filename = f"{name}.pt"

        if self._gen_loop_prefix:
            return os.path.join(self._resume_dir, self._gen_loop_prefix, filename)
        return os.path.join(self._resume_dir, filename)

    def _remote_upload_enabled(self) -> bool:
        return bool(
            self._upload_manager is not None
            and self._upload_manager.upload_enabled
        )

    def _persist_stage_locally(self) -> bool:
        # When remote uploads are enabled, stage artifacts live in memory until
        # the async upload thread consumes them; we intentionally avoid writing
        # them into the local resume directory.
        return not self._remote_upload_enabled()

    # ------------------------------------------------------------------
    # Local disk persistence (when no remote is configured)
    # ------------------------------------------------------------------

    def _save_outputs_locally(self) -> None:
        """Persist all registered outputs to local disk under ``resume_dir``.

        Only used when remote uploads are disabled.  With remote uploads
        enabled, stage outputs stay in memory and are not copied into the
        local resume directory.
        """
        for name, data in self._outputs.items():
            kind = self._output_kinds[name]
            if kind == "directory":
                # Directory outputs are already on local disk; nothing to persist
                continue

            path = self._local_path(name, kind)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if kind == "dataproto":
                data.save_to_disk(path)
                logger.debug("Saved DataProto '%s' locally → %s", name, path)
            elif kind == "json":
                # Atomic write: tempfile + os.replace
                fd, tmp = tempfile.mkstemp(
                    dir=os.path.dirname(path), suffix=".tmp"
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(data, f)
                    os.replace(tmp, path)
                    logger.debug("Saved JSON '%s' locally → %s", name, path)
                except BaseException:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> StageContext:
        if self._timing_dict is not None:
            self._t_start = time.perf_counter()

        gen_str = f" | {self._gen_loop_prefix}" if self._gen_loop_prefix else ""
        if self._is_done():
            # Resume: try to download each output from R2 into memory
            if self._upload_manager and self._upload_manager.remote_configured:
                # We only download outputs that were registered via _declare_outputs
                # For stages with outputs, the caller must check should_run and
                # call result() — which uses _outputs populated here
                self.should_run = False
                print(
                    f"---------- Stage {self._stage_id} ({self._name}) | ans_loop={self._ans_loop}"
                    f"{gen_str} | SKIP (resumed, will lazy-download from R2) ----------",
                    flush=True,
                )
            else:
                # No remote: outputs must be available locally or re-run
                self.should_run = False
                print(
                    f"---------- Stage {self._stage_id} ({self._name}) | ans_loop={self._ans_loop}"
                    f"{gen_str} | SKIP (resumed, local only) ----------",
                    flush=True,
                )
        else:
            self.should_run = True
            print(
                f"---------- Stage {self._stage_id} ({self._name}) | ans_loop={self._ans_loop}"
                f"{gen_str} | RUNNING ----------",
                flush=True,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Record elapsed time regardless of success/failure/skip
        if self._timing_dict is not None and self._t_start is not None:
            elapsed = time.perf_counter() - self._t_start
            key = self._timer_name
            self._timing_dict[key] = self._timing_dict.get(key, 0) + elapsed

        if exc_type is not None:
            # Stage failed — do NOT mark done or upload
            return False

        if not self.should_run:
            # Resumed stage, nothing to persist
            return False

        # When defer_state_update is True (PPO stages 5/6), skip marking done,
        # saving state.json, and uploading it.  The checkpoint section of the
        # trainer will handle these after the checkpoint is saved locally and
        # queue state.json upload AFTER the checkpoint upload in the FIFO queue.
        if not self._defer_state_update:
            # 1. Mark stage done in ResumeState (in-memory flag only)
            self._mark_done()

        if self._persist_stage_locally():
            # 2. Save outputs locally FIRST — ensures output files exist on disk
            #    before state.json marks the stage as "done".
            self._save_outputs_locally()

            if not self._defer_state_update:
                # 3. Save state.json locally (atomic) — the "done" marker.
                #    Written AFTER outputs so a crash never leaves state.json saying
                #    "done" with missing output files.
                self._resume.save(self._resume_dir)

        if self._remote_upload_enabled():
            if not self._should_upload_remote:
                logger.debug(
                    "Stage %d (%s) | Skipping remote upload for ans_loop=%d",
                    self._stage_id,
                    self._name,
                    self._ans_loop,
                )
                return False

            # 4. Submit non-blocking uploads for each saved output FIRST
            output_task_ids: list[str] = []
            for name, data in self._outputs.items():
                kind = self._output_kinds[name]
                remote_key = self._remote_key(name)

                if kind == "directory":
                    task_id = self._upload_manager.submit_dir_upload(
                        local_path=data,  # path string
                        remote_key=remote_key,
                        cleanup_after=True,
                        cleanup_on_failure=True,
                        task_kind="artifact_upload",
                    )
                else:
                    task_id = self._upload_manager.submit_memory_upload(
                        name=name,
                        data=data,
                        kind=kind,
                        remote_key=remote_key,
                        task_kind="artifact_upload",
                    )
                if task_id:
                    output_task_ids.append(task_id)

            if not self._defer_state_update:
                # 5. Queue state.json upload LAST — and only after artifact
                #    uploads succeeded — so remote state never marks a stage
                #    done while its outputs are missing.
                self._upload_manager.submit_memory_upload(
                    name=f"{self._name}_state",
                    data=self._resume.to_dict(),
                    kind="json",
                    remote_key=f"ans_{self._ans_loop}/state.json",
                    depends_on=output_task_ids,
                    task_kind="state_upload",
                )

        return False

    # ------------------------------------------------------------------
    # Lazy download for resume
    # ------------------------------------------------------------------

    def _get_resolver(self) -> StorageResolver:
        """Build a :class:`StorageResolver` scoped to this stage's paths."""
        if self._gen_loop_prefix:
            local_base = os.path.join(self._resume_dir, self._gen_loop_prefix)
            remote_prefix = f"ans_{self._ans_loop}/{self._gen_loop_prefix}"
        else:
            local_base = self._resume_dir
            remote_prefix = f"ans_{self._ans_loop}"

        return StorageResolver(
            local_base=local_base,
            upload_manager=self._upload_manager,
            remote_prefix=remote_prefix,
            resolve_order="local_first",
        )

    def _ensure_downloaded(self, name: str, kind: str) -> bool:
        """Load a single output into memory, trying local disk first then R2.

        Returns ``True`` if the output is now available, ``False`` if both
        sources failed (the caller should re-run the stage).
        """
        if name in self._outputs:
            return True

        logger.info(
            "Stage %d (%s) | Downloading resumed output '%s' (kind=%s) ...",
            self._stage_id, self._name, name, kind,
        )
        resolver = self._get_resolver()

        if kind == "directory":
            local_path = tempfile.mkdtemp(prefix=f"{name}_")
            self._temp_dirs.append(local_path)
            result = resolver.resolve_dir(name, local_path=local_path)
            if result:
                self._outputs[name] = result
                self._output_kinds[name] = "directory"
                logger.info(
                    "Stage %d (%s) | Downloaded dir '%s' → %s",
                    self._stage_id, self._name, name, result,
                )
                return True
            logger.warning(
                "Stage %d (%s) | Failed to download dir '%s' from local or R2",
                self._stage_id, self._name, name,
            )
            return False
        else:
            ext = ".json" if kind == "json" else ".pt"
            if kind == "json":
                data = resolver.resolve_json(f"{name}{ext}")
            else:
                data = resolver.resolve_dataproto(f"{name}{ext}")
            if data is not None:
                self._outputs[name] = data
                self._output_kinds[name] = kind
                logger.info(
                    "Stage %d (%s) | Downloaded '%s%s' successfully",
                    self._stage_id, self._name, name, ext,
                )
                return True
            logger.warning(
                "Stage %d (%s) | Failed to download '%s%s' from local or R2",
                self._stage_id, self._name, name, ext,
            )
            return False

    def download_result(self, name: str, kind: str = "dataproto") -> Any | None:
        """Explicitly download an output from R2 (for resume scenarios).

        Unlike ``result()``, this always attempts an R2 download if the
        output isn't already in memory. Returns ``None`` on failure.
        """
        ok = self._ensure_downloaded(name, kind)
        if ok:
            return self._outputs[name]
        return None

    def cleanup_temp_dirs(self) -> None:
        """Delete temp directories created during resume downloads.

        Call this after the stage's outputs are no longer needed (e.g. after
        the data has been consumed by the training step).
        """
        for d in self._temp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
                logger.debug("Cleaned up temp dir: %s", d)
            except Exception:
                pass
        self._temp_dirs.clear()
