"""Stage-level resume state for SelfEvolutionTrainer.

Tracks intra-ans-loop progress so that a crash mid-loop doesn't lose
all work.  DataProto files and JSON metadata are persisted under
``{default_local_dir}/ans_{N}/`` (one directory per ans_loop, mirroring
the R2 layout).

After a successful checkpoint at the end of an ans_loop, the local
directory is cleared.  R2 copies are preserved for progress tracking.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any

from verl import DataProto

logger = logging.getLogger(__name__)

_STATE_FILENAME = "state.json"


@dataclass
class GenLoopState:
    """Progress for one gen_loop iteration."""

    local_gen: int
    stage_2_done: bool = False  # question generation
    stage_3_done: bool = False  # gen answer rollout
    stage_4_done: bool = False  # scoring
    stage_5_done: bool = False  # generator PPO update
    rewards: dict[str, Any] | None = None  # Stage 4 output payload


@dataclass
class ResumeState:
    """Intra-ans-loop progress tracker.  Cleared after successful checkpoint."""

    ans_loop: int
    num_gen_per_ans: int  # for validation on reload
    stage_0_done: bool = False  # curriculum refresh
    stage_1_done: bool = False  # dev rollout
    stage_6_done: bool = False  # solver PPO update
    gen_loops: list[GenLoopState] = field(default_factory=list)
    doc_dataset_state: dict | None = None  # {position, epoch, total_docs, batch_size}

    # ------------------------------------------------------------------
    # Persistence — state.json
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable state payload."""
        return {
            "ans_loop": self.ans_loop,
            "num_gen_per_ans": self.num_gen_per_ans,
            "stage_0_done": self.stage_0_done,
            "stage_1_done": self.stage_1_done,
            "stage_6_done": self.stage_6_done,
            "doc_dataset_state": self.doc_dataset_state,
            "gen_loops": [asdict(gl) for gl in self.gen_loops],
        }

    def save(self, base_dir: str) -> None:
        """Atomically write ``state.json`` (temp file + rename)."""
        os.makedirs(base_dir, exist_ok=True)
        target = os.path.join(base_dir, _STATE_FILENAME)
        data = self.to_dict()
        fd, tmp = tempfile.mkstemp(dir=base_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, target)
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, base_dir: str, resolver=None) -> ResumeState | None:
        """Load from *base_dir*.  Returns ``None`` if no state exists.

        Args:
            base_dir: Directory containing ``state.json``.
            resolver: Optional :class:`StorageResolver`.  When provided,
                resolves ``state.json`` via local + R2 fallback instead of
                a plain local-only check.
        """
        if resolver is not None:
            logger.info(
                "ResumeState: resolving %s via StorageResolver (base_dir=%s)",
                _STATE_FILENAME, base_dir,
            )
            data = resolver.resolve_json(_STATE_FILENAME)
        else:
            path = os.path.join(base_dir, _STATE_FILENAME)
            if not os.path.exists(path):
                logger.info(
                    "ResumeState: no %s found at %s", _STATE_FILENAME, path
                )
                return None
            logger.info("ResumeState: loading from local %s", path)
            with open(path, "r") as f:
                data = json.load(f)
        if data is None:
            logger.info("ResumeState: %s could not be resolved", _STATE_FILENAME)
            return None
        state = cls(
            ans_loop=data["ans_loop"],
            num_gen_per_ans=data["num_gen_per_ans"],
            stage_0_done=data.get("stage_0_done", False),
            stage_1_done=data.get("stage_1_done", False),
            stage_6_done=data.get("stage_6_done", False),
            doc_dataset_state=data.get("doc_dataset_state"),
        )
        for gl_data in data.get("gen_loops", []):
            state.gen_loops.append(GenLoopState(**gl_data))
        logger.info(
            "ResumeState: loaded for ans_loop=%d | gen_loops=%d | stage_0=%s stage_1=%s stage_6=%s",
            state.ans_loop,
            len(state.gen_loops),
            state.stage_0_done,
            state.stage_1_done,
            state.stage_6_done,
        )
        return state

    @staticmethod
    def clear(base_dir: str) -> None:
        """Delete the entire resume-state directory."""
        if os.path.isdir(base_dir):
            shutil.rmtree(base_dir, ignore_errors=True)
            logger.info("Cleared resume state: %s", base_dir)

    # ------------------------------------------------------------------
    # DataProto helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_data(base_dir: str, name: str, data: DataProto) -> None:
        """Save a ``DataProto`` to ``{base_dir}/{name}.pt``."""
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"{name}.pt")
        data.save_to_disk(filepath)
        logger.info("Saved resume data: %s", filepath)

    @staticmethod
    def load_data(base_dir: str, name: str) -> DataProto:
        """Load a ``DataProto`` from ``{base_dir}/{name}.pt``."""
        filepath = os.path.join(base_dir, f"{name}.pt")
        return DataProto.load_from_disk(filepath)

    # ------------------------------------------------------------------
    # Gen-loop subdirectory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gen_dir(base_dir: str, local_gen: int) -> str:
        return os.path.join(base_dir, f"gen_{local_gen}")

    def save_gen_data(
        self, base_dir: str, local_gen: int, name: str, data: DataProto
    ) -> None:
        """Save a ``DataProto`` to ``{base_dir}/gen_{i}/{name}.pt``."""
        gdir = self._gen_dir(base_dir, local_gen)
        self.save_data(gdir, name, data)

    def load_gen_data(
        self, base_dir: str, local_gen: int, name: str
    ) -> DataProto:
        """Load a ``DataProto`` from ``{base_dir}/gen_{i}/{name}.pt``."""
        gdir = self._gen_dir(base_dir, local_gen)
        return self.load_data(gdir, name)

    # ------------------------------------------------------------------
    # Gen-loop tracking
    # ------------------------------------------------------------------

    def ensure_gen_loop(self, local_gen: int) -> GenLoopState:
        """Get or create the ``GenLoopState`` for *local_gen*."""
        for gl in self.gen_loops:
            if gl.local_gen == local_gen:
                return gl
        gl = GenLoopState(local_gen=local_gen)
        self.gen_loops.append(gl)
        return gl
