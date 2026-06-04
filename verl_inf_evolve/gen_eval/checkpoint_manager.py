"""Checkpoint and gen_output prefetch managers for generator evaluation.

Downloads generator checkpoints or pre-saved gen_output.pt files from a
remote backend in the background using a thread pool, so the next item is
ready by the time the current evaluation finishes.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import TYPE_CHECKING

from verl_inf_evolve.storage.remote_backend import RemoteBackend, create_remote_backend

if TYPE_CHECKING:
    from verl import DataProto

logger = logging.getLogger(__name__)


class CheckpointPrefetcher:
    """Download and cache generator checkpoints with background prefetching.

    Parameters
    ----------
    remote_base_path:
        Remote URI for the training trajectory, e.g.
        ``s3://bucket/experiment/trajectory`` or
        ``hf://datasets/org/repo/prefix``.  Checkpoints are expected at
        ``{remote_base_path}/global_step_{idx}/generator/huggingface/``.
    local_cache_dir:
        Local directory for downloaded checkpoints.
    prefetch_count:
        How many upcoming checkpoints to download ahead of time.
    cleanup_after_eval:
        If ``True``, delete previously evaluated checkpoint directories
        to save disk space.
    """

    def __init__(
        self,
        remote_base_path: str,
        local_cache_dir: str,
        prefetch_count: int = 1,
        cleanup_after_eval: bool = True,
        backend_kwargs: dict[str, object] | None = None,
    ) -> None:
        self.remote_base_path = remote_base_path.rstrip("/")
        self.local_cache_dir = local_cache_dir
        self.prefetch_count = prefetch_count
        self.cleanup_after_eval = cleanup_after_eval
        self.backend_kwargs = backend_kwargs or {}

        self._backend: RemoteBackend = create_remote_backend(
            self.remote_base_path,
            **self.backend_kwargs,
        )

        self._executor = ThreadPoolExecutor(max_workers=max(prefetch_count, 1))
        self._futures: dict[int, Future[str]] = {}
        self._lock = Lock()
        self._ans_loop_indices: list[int] = []
        self._last_evaluated_idx: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_indices(self, indices: list[int]) -> None:
        """Set the ordered list of ans_loop indices that will be evaluated.

        This is used to determine which checkpoints to prefetch next.
        """
        self._ans_loop_indices = list(indices)

    def get_checkpoint(self, ans_loop_idx: int) -> str:
        """Return the local path to a downloaded generator checkpoint.

        Blocks until the checkpoint is available.  On each call, triggers
        prefetch of the next ``prefetch_count`` checkpoints in the queue.

        Returns
        -------
        str
            Local path to ``{local_cache_dir}/global_step_{idx}/generator/huggingface/``.

        Raises
        ------
        RuntimeError
            If the download fails or the checkpoint is invalid.
        """
        # Start download for the requested checkpoint first
        future = self._ensure_download(ans_loop_idx)

        # Then trigger prefetch for upcoming checkpoints
        self._trigger_prefetch(ans_loop_idx)
        local_path = future.result()  # blocks until done

        # Clean up the previously evaluated checkpoint
        if self.cleanup_after_eval and self._last_evaluated_idx is not None:
            self._cleanup(self._last_evaluated_idx)
        self._last_evaluated_idx = ans_loop_idx

        return local_path

    def shutdown(self) -> None:
        """Shut down the background thread pool."""
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trigger_prefetch(self, current_idx: int) -> None:
        """Start background downloads for the next ``prefetch_count`` checkpoints."""
        if not self._ans_loop_indices:
            return

        try:
            pos = self._ans_loop_indices.index(current_idx)
        except ValueError:
            return

        for offset in range(1, self.prefetch_count + 1):
            next_pos = pos + offset
            if next_pos < len(self._ans_loop_indices):
                next_idx = self._ans_loop_indices[next_pos]
                self._ensure_download(next_idx)

    def _ensure_download(self, ans_loop_idx: int) -> Future[str]:
        """Return a future for the given checkpoint, starting download if needed."""
        with self._lock:
            if ans_loop_idx not in self._futures:
                self._futures[ans_loop_idx] = self._executor.submit(
                    self._download_checkpoint, ans_loop_idx
                )
            return self._futures[ans_loop_idx]

    def _download_checkpoint(self, ans_loop_idx: int) -> str:
        """Download a single checkpoint and validate it.

        Returns the local path to the ``generator/huggingface/`` directory.
        """
        local_step_dir = os.path.join(
            self.local_cache_dir,
            f"global_step_{ans_loop_idx}",
            "generator",
            "huggingface",
        )

        # Skip download if already present and valid
        if os.path.isdir(local_step_dir) and self._validate_checkpoint(local_step_dir):
            logger.info(
                "Checkpoint for ans_loop=%d already cached at %s",
                ans_loop_idx,
                local_step_dir,
            )
            return local_step_dir

        remote_key = f"global_step_{ans_loop_idx}/generator/huggingface"

        os.makedirs(local_step_dir, exist_ok=True)

        logger.info(
            "Downloading checkpoint ans_loop=%d: %s -> %s",
            ans_loop_idx,
            remote_key,
            local_step_dir,
        )
        t0 = time.time()
        success = self._backend.download_dir(remote_key, local_step_dir)
        elapsed = time.time() - t0

        if not success:
            raise RuntimeError(
                f"Failed to download checkpoint for ans_loop={ans_loop_idx} "
                f"from {remote_key}"
            )

        logger.info(
            "Downloaded checkpoint ans_loop=%d in %.1fs",
            ans_loop_idx,
            elapsed,
        )

        if not self._validate_checkpoint(local_step_dir):
            raise RuntimeError(
                f"Downloaded checkpoint for ans_loop={ans_loop_idx} is invalid: "
                f"missing config.json or .safetensors files in {local_step_dir}"
            )

        return local_step_dir

    @staticmethod
    def _validate_checkpoint(local_path: str) -> bool:
        """Check that a checkpoint directory contains the expected files.

        A valid HuggingFace checkpoint must have ``config.json`` and at least
        one ``.safetensors`` weight file.
        """
        if not os.path.isdir(local_path):
            return False

        has_config = os.path.isfile(os.path.join(local_path, "config.json"))
        has_weights = any(
            f.endswith(".safetensors") for f in os.listdir(local_path)
        )
        return has_config and has_weights

    def _cleanup(self, ans_loop_idx: int) -> None:
        """Remove a previously evaluated checkpoint from local disk."""
        step_dir = os.path.join(
            self.local_cache_dir, f"global_step_{ans_loop_idx}"
        )
        if os.path.isdir(step_dir):
            logger.info("Cleaning up checkpoint at %s", step_dir)
            shutil.rmtree(step_dir, ignore_errors=True)

        # Remove the future reference as well
        with self._lock:
            self._futures.pop(ans_loop_idx, None)


# Required non_tensor_batch keys that must be present in a valid gen_output.pt
_REQUIRED_GEN_OUTPUT_KEYS = frozenset(
    ["parsed_ok", "question_id", "question_text", "choices", "ground_truth", "doc_id"]
)


class GenOutputPrefetcher:
    """Download and cache gen_output.pt files with background prefetching.

    Used in replay mode to download pre-saved generation outputs from a
    training trajectory instead of regenerating questions from scratch.

    Parameters
    ----------
    remote_base_path:
        Remote URI for the training trajectory, e.g.
        ``s3://bucket/experiment/trajectory`` or
        ``hf://datasets/org/repo/prefix``.  Gen outputs are expected at
        ``{remote_base_path}/ans_{idx}/gen_0/gen_output.pt``.
    local_cache_dir:
        Local directory for downloaded gen_output.pt files.
    prefetch_count:
        How many upcoming gen_output.pt files to download ahead of time.
    cleanup_after_eval:
        If ``True``, delete previously loaded gen_output.pt files
        to save disk space.
    """

    def __init__(
        self,
        remote_base_path: str,
        local_cache_dir: str,
        prefetch_count: int = 1,
        cleanup_after_eval: bool = True,
        backend_kwargs: dict[str, object] | None = None,
    ) -> None:
        self.remote_base_path = remote_base_path.rstrip("/")
        self.local_cache_dir = local_cache_dir
        self.prefetch_count = prefetch_count
        self.cleanup_after_eval = cleanup_after_eval
        self.backend_kwargs = backend_kwargs or {}

        self._backend: RemoteBackend = create_remote_backend(
            self.remote_base_path,
            **self.backend_kwargs,
        )

        self._executor = ThreadPoolExecutor(max_workers=max(prefetch_count, 1))
        self._futures: dict[int, Future[str]] = {}
        self._lock = Lock()
        self._ans_loop_indices: list[int] = []
        self._last_evaluated_idx: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_indices(self, indices: list[int]) -> None:
        """Set the ordered list of ans_loop indices that will be evaluated."""
        self._ans_loop_indices = list(indices)

    def get_gen_output(self, ans_loop_idx: int) -> DataProto:
        """Return the deserialized DataProto from a downloaded gen_output.pt.

        Blocks until the file is available.  On each call, triggers
        prefetch of the next ``prefetch_count`` files in the queue.

        Returns
        -------
        DataProto
            The deserialized DataProto object with validated non_tensor_batch
            keys.

        Raises
        ------
        RuntimeError
            If the download fails, the file is missing, or validation fails.
        """
        # Start download for the requested file first
        future = self._ensure_download(ans_loop_idx)

        # Then trigger prefetch for upcoming files
        self._trigger_prefetch(ans_loop_idx)
        local_path = future.result()  # blocks until done

        # Deserialize the DataProto
        from verl import DataProto as _DataProto

        logger.info(
            "Loading gen_output.pt for ans_loop=%d from %s",
            ans_loop_idx,
            local_path,
        )
        gen_output = _DataProto.load_from_disk(local_path)

        # Validate required non_tensor_batch keys
        missing = _REQUIRED_GEN_OUTPUT_KEYS - set(gen_output.non_tensor_batch.keys())
        if missing:
            raise RuntimeError(
                f"gen_output.pt for ans_loop={ans_loop_idx} is missing required "
                f"non_tensor_batch keys: {sorted(missing)}"
            )

        # Clean up the previously evaluated gen_output
        if self.cleanup_after_eval and self._last_evaluated_idx is not None:
            self._cleanup(self._last_evaluated_idx)
        self._last_evaluated_idx = ans_loop_idx

        return gen_output

    def shutdown(self) -> None:
        """Shut down the background thread pool."""
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trigger_prefetch(self, current_idx: int) -> None:
        """Start background downloads for the next ``prefetch_count`` files."""
        if not self._ans_loop_indices:
            return

        try:
            pos = self._ans_loop_indices.index(current_idx)
        except ValueError:
            return

        for offset in range(1, self.prefetch_count + 1):
            next_pos = pos + offset
            if next_pos < len(self._ans_loop_indices):
                next_idx = self._ans_loop_indices[next_pos]
                self._ensure_download(next_idx)

    def _ensure_download(self, ans_loop_idx: int) -> Future[str]:
        """Return a future for the given gen_output, starting download if needed."""
        with self._lock:
            if ans_loop_idx not in self._futures:
                self._futures[ans_loop_idx] = self._executor.submit(
                    self._download_gen_output, ans_loop_idx
                )
            return self._futures[ans_loop_idx]

    def _download_gen_output(self, ans_loop_idx: int) -> str:
        """Download a single gen_output.pt and return its local path."""
        local_dir = os.path.join(
            self.local_cache_dir,
            f"ans_{ans_loop_idx}",
            "gen_0",
        )
        local_file = os.path.join(local_dir, "gen_output.pt")

        # Skip download if already present
        if os.path.isfile(local_file):
            file_size = os.path.getsize(local_file)
            logger.info(
                "gen_output.pt for ans_loop=%d already cached at %s (%.1f MB)",
                ans_loop_idx,
                local_file,
                file_size / (1024 * 1024),
            )
            return local_file

        remote_key = f"ans_{ans_loop_idx}/gen_0/gen_output.pt"

        os.makedirs(local_dir, exist_ok=True)

        logger.info(
            "Downloading gen_output.pt ans_loop=%d: %s -> %s",
            ans_loop_idx,
            remote_key,
            local_file,
        )
        t0 = time.time()
        success = self._backend.download_file(remote_key, local_file)
        elapsed = time.time() - t0

        if not success:
            raise RuntimeError(
                f"Failed to download gen_output.pt for ans_loop={ans_loop_idx} "
                f"from {remote_key}"
            )

        if not os.path.isfile(local_file):
            raise RuntimeError(
                f"gen_output.pt not found after download for ans_loop={ans_loop_idx}. "
                f"File does not exist at {local_file}. The remote key "
                f"{remote_key} may not exist."
            )

        file_size = os.path.getsize(local_file)
        logger.info(
            "Downloaded gen_output.pt ans_loop=%d in %.1fs (%.1f MB)",
            ans_loop_idx,
            elapsed,
            file_size / (1024 * 1024),
        )

        return local_file

    def _cleanup(self, ans_loop_idx: int) -> None:
        """Remove a previously loaded gen_output from local disk."""
        ans_dir = os.path.join(self.local_cache_dir, f"ans_{ans_loop_idx}")
        if os.path.isdir(ans_dir):
            logger.info("Cleaning up gen_output at %s", ans_dir)
            shutil.rmtree(ans_dir, ignore_errors=True)

        # Remove the future reference as well
        with self._lock:
            self._futures.pop(ans_loop_idx, None)
