"""Document dataset with shuffled batching and epoch tracking.

Copied from ``inf_evolve/train_utils.py:452-546`` — pure Python class,
no adaptation needed for V3.
"""

from __future__ import annotations

import random
from typing import Optional


class DocumentDataset:
    """Dataset class for managing document IDs with shuffling and batching.

    Tracks current position and handles epoch boundaries properly:
    when documents are exhausted, collects remainder, reshuffles,
    then fills from the start of the new order.

    Example:
        >>> dataset = DocumentDataset(["0", "1", "2", "3", "4"], batch_size=2, seed=42)
        >>> dataset.next_batch()  # ["0", "1"], position=2
        >>> dataset.next_batch()  # ["2", "3"], position=4
        >>> dataset.next_batch()  # ["4"] + reshuffle + [first], position=1
    """

    def __init__(
        self,
        doc_ids: list[str],
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
        repeat_batch: bool = False,
    ):
        """Initialize the dataset.

        Args:
            doc_ids: List of all document IDs.
            batch_size: Number of documents per batch.
            shuffle: Whether to shuffle the document IDs.
            seed: Random seed for reproducibility. If None, uses system randomness.
            repeat_batch: When True, the first batch is cached and returned on
                every subsequent call to ``next_batch()``. The dataset position
                never advances, so checkpointing and resume work unchanged.
        """
        self.original_doc_ids = doc_ids.copy()
        self.batch_size = batch_size
        self.doc_ids = doc_ids.copy()
        self.position = 0
        self.epoch = 0
        self.seed = seed
        self.repeat_batch = repeat_batch
        self._cached_batch: Optional[tuple[list[str], bool]] = None

        self.rng = random.Random(seed)

        if shuffle:
            self.rng.shuffle(self.doc_ids)

    def __len__(self) -> int:
        """Return total number of documents."""
        return len(self.doc_ids)

    @property
    def num_batches_per_epoch(self) -> int:
        """Number of full batches needed to cover all documents once."""
        return (len(self.doc_ids) + self.batch_size - 1) // self.batch_size

    def reshuffle(self) -> None:
        """Reshuffle the document IDs and reset position."""
        self.rng.shuffle(self.doc_ids)
        self.position = 0
        self.epoch += 1

    def next_batch(self) -> tuple[list[str], bool]:
        """Get the next batch of document IDs.

        Handles epoch boundaries by collecting remainder from current order,
        reshuffling, then filling from the start of the new order.

        When ``repeat_batch`` is enabled, the first batch is cached and
        returned on every subsequent call without advancing the position.

        Returns:
            Tuple of (batch_doc_ids, reshuffled) where reshuffled indicates
            if a reshuffle occurred during this batch.
        """
        if self.repeat_batch and self._cached_batch is not None:
            return self._cached_batch

        n = len(self.doc_ids)
        remaining = n - self.position
        reshuffled = False

        if remaining >= self.batch_size:
            batch = self.doc_ids[self.position : self.position + self.batch_size]
            self.position += self.batch_size
        else:
            batch = self.doc_ids[self.position : n]
            self.reshuffle()
            reshuffled = True
            needed = self.batch_size - len(batch)
            batch.extend(self.doc_ids[0:needed])
            self.position = needed

        result = (batch, reshuffled)
        if self.repeat_batch:
            self._cached_batch = result
        return result

    def get_state(self) -> dict:
        """Get current state for logging/checkpointing."""
        return {
            "position": self.position,
            "epoch": self.epoch,
            "total_docs": len(self.doc_ids),
            "batch_size": self.batch_size,
        }

    def restore_state(self, state: dict) -> None:
        """Restore dataset position/epoch from a previously saved state.

        Replays the initial shuffle plus any epoch reshuffles so that the
        internal RNG is in the exact same state as when ``get_state()`` was
        called.  This ensures subsequent ``next_batch()`` calls produce
        the same document batches as the original run.

        Args:
            state: Dict with ``position`` and ``epoch`` keys (as returned
                by ``get_state()``).
        """
        target_epoch = state["epoch"]
        target_position = state["position"]

        # Reset RNG and replay shuffles from scratch
        self.rng = random.Random(self.seed)
        self.doc_ids = self.original_doc_ids.copy()
        self.rng.shuffle(self.doc_ids)  # initial shuffle (epoch 0)

        for _ in range(target_epoch):
            self.rng.shuffle(self.doc_ids)

        self.epoch = target_epoch
        self.position = target_position
