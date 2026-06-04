"""Unit tests for stratify_by_doc and its interaction with scatter_for_dispatch.

These tests verify the question-axis permutation that fixes per-document
correlation in solver PPO minibatches (gen-solver-joint-evolve-v6).
"""
from __future__ import annotations

import numpy as np
import torch

from verl.protocol import DataProto
from verl_inf_evolve.utils.data_utils import scatter_for_dispatch, stratify_by_doc


def _make_batch(doc_ids_per_question: list[str], samples_per_question: int) -> DataProto:
    """Build a synthetic DataProto in `[Q0_all, Q1_all, ...]` layout.

    sample_id encodes ``q_idx * 1000 + r`` so we can recover (question, rollout)
    after permutation. doc_id is replicated for each rollout of the question.
    """
    num_questions = len(doc_ids_per_question)
    sample_ids = torch.tensor(
        [q * 1000 + r for q in range(num_questions) for r in range(samples_per_question)],
        dtype=torch.int64,
    )
    doc_id_per_sample = np.array(
        [doc_ids_per_question[q] for q in range(num_questions) for _ in range(samples_per_question)],
        dtype=object,
    )
    return DataProto.from_dict(
        tensors={"sample_id": sample_ids},
        non_tensors={"doc_id": doc_id_per_sample},
    )


def _question_order(batch: DataProto, samples_per_question: int) -> list[int]:
    """Read back the question indices in their current order."""
    sids = batch.batch["sample_id"].tolist()
    # Take one sample per question block; sample_id // 1000 gives the original q_idx.
    return [sids[i] // 1000 for i in range(0, len(sids), samples_per_question)]


def test_stratify_unequal_buckets_matches_bresenham():
    """Canonical example: A=8, B=4, C=2 questions, samples_per_question=1.

    Expected order from Bresenham ideal-position spread:
        A B A C A B A A B A C A B A
    """
    doc_ids = (["A"] * 8) + (["B"] * 4) + (["C"] * 2)
    batch = _make_batch(doc_ids, samples_per_question=1)

    out = stratify_by_doc(batch, num_questions=14, samples_per_question=1)

    q_perm = _question_order(out, samples_per_question=1)
    assert q_perm == [0, 8, 1, 12, 2, 9, 3, 4, 10, 5, 13, 6, 11, 7]

    # Equivalently, the doc-id sequence should be A B A C A B A A B A C A B A
    doc_seq = [doc_ids[q] for q in q_perm]
    assert doc_seq == ["A", "B", "A", "C", "A", "B", "A", "A", "B", "A", "C", "A", "B", "A"]


def test_stratify_preserves_rollout_grouping():
    """Each question's `samples_per_question` rollouts must stay contiguous
    AND in their original within-question order after permutation."""
    doc_ids = (["A"] * 5) + (["B"] * 3) + (["C"] * 2)
    samples_per_q = 4
    batch = _make_batch(doc_ids, samples_per_question=samples_per_q)

    out = stratify_by_doc(batch, num_questions=10, samples_per_question=samples_per_q)
    sids = out.batch["sample_id"].tolist()

    assert len(sids) == 10 * samples_per_q
    for block_start in range(0, len(sids), samples_per_q):
        block = sids[block_start : block_start + samples_per_q]
        q_idx = block[0] // 1000
        # Same q_idx in every rollout of the block, rollout index 0..n-1 in order.
        assert block == [q_idx * 1000 + r for r in range(samples_per_q)], (
            f"block at {block_start} broke rollout grouping: {block}"
        )


def test_stratify_is_a_permutation():
    """No samples lost or duplicated."""
    doc_ids = (["A"] * 8) + (["B"] * 4) + (["C"] * 2)
    samples_per_q = 3
    batch = _make_batch(doc_ids, samples_per_question=samples_per_q)

    out = stratify_by_doc(batch, num_questions=14, samples_per_question=samples_per_q)
    sids_in = batch.batch["sample_id"].tolist()
    sids_out = out.batch["sample_id"].tolist()

    assert sorted(sids_in) == sorted(sids_out)
    assert len(set(sids_out)) == len(sids_out)


def test_stratify_doc_id_non_tensor_reordered_consistently():
    """The non_tensor `doc_id` field must be reordered the same way as the
    tensor batch — otherwise downstream code reading doc_id breaks."""
    doc_ids = (["A"] * 4) + (["B"] * 2) + (["C"] * 2)
    samples_per_q = 2
    batch = _make_batch(doc_ids, samples_per_question=samples_per_q)

    out = stratify_by_doc(batch, num_questions=8, samples_per_question=samples_per_q)
    sids = out.batch["sample_id"].tolist()
    doc_id_arr = out.non_tensor_batch["doc_id"]

    for i, sid in enumerate(sids):
        q_idx = sid // 1000
        assert doc_id_arr[i] == doc_ids[q_idx], (
            f"doc_id mismatch at row {i}: tensor q_idx={q_idx} (doc {doc_ids[q_idx]}) "
            f"vs non_tensor doc_id={doc_id_arr[i]}"
        )


def test_stratify_no_doc_starvation_in_window():
    """For the user's regime ("some docs with 8 questions, some with fewer"),
    every doc with bucket size ≥ 2 should appear in each contiguous half, and
    the largest doc shouldn't pile up at one end.

    Note: singleton buckets (k=1) all map to ideal position 0.5 and may clump
    in the middle — that's a known limitation of the Bresenham approach when
    multiple docs have only one question. Not exercised here because gen_n ≥ 2
    in practice; covered in `test_stratify_singleton_clumping_known_quirk`.
    """
    # 6 docs, sizes [8, 8, 4, 4, 2, 2] = 28 questions total
    doc_ids: list[str] = []
    for i, k in enumerate([8, 8, 4, 4, 2, 2]):
        doc_ids.extend([f"D{i}"] * k)
    num_q = len(doc_ids)  # 28
    batch = _make_batch(doc_ids, samples_per_question=1)

    out = stratify_by_doc(batch, num_questions=num_q, samples_per_question=1)
    q_perm = _question_order(out, samples_per_question=1)
    doc_seq = [doc_ids[q] for q in q_perm]

    # Property: in each contiguous half, every doc should appear at least once.
    half = num_q // 2
    for start in (0, half):
        window = doc_seq[start : start + half]
        for doc in set(doc_ids):
            assert doc in window, (
                f"Doc {doc} missing from window [{start}, {start + half}): {window}"
            )

    # Stronger: a doc with bucket size k should appear roughly k/N * window_size
    # times in any window. Largest doc D0 has share 8/28 ≈ 0.29; in a window of
    # 10, expect 2-4 occurrences (allow ±2 slack for rounding/tiebreak).
    for start in range(0, num_q - 10, 5):
        window = doc_seq[start : start + 10]
        d0_count = window.count("D0")
        assert 1 <= d0_count <= 5, (
            f"D0 over/under-represented in window [{start}, {start+10}): "
            f"got {d0_count}, expected 2-4"
        )


def test_stratify_singleton_clumping_known_quirk():
    """Documents the known limitation: when multiple docs have bucket size = 1,
    they all map to ideal position 0.5 and tiebreak by first-seen order, so
    they clump near the middle. Pin this behavior so a future "fix" doesn't
    silently change it without being noticed.
    """
    # 3 singletons followed by a large doc — singletons all want position 0.5.
    doc_ids = ["A", "B", "C"] + (["D"] * 6)
    batch = _make_batch(doc_ids, samples_per_question=1)
    out = stratify_by_doc(batch, num_questions=9, samples_per_question=1)
    q_perm = _question_order(out, samples_per_question=1)
    doc_seq = [doc_ids[q] for q in q_perm]

    # All three singletons land in the middle three slots, in first-seen order.
    middle = doc_seq[3:6]
    assert middle == ["A", "B", "C"], f"singleton clump pin: {middle}"


def test_stratify_then_scatter_per_rank_invariant():
    """End-to-end: after stratify -> scatter_for_dispatch, the contiguous
    per-rank slice must contain ALL questions in the SAME order on every rank
    (so FSDP grad-reduction sees aligned per-question groups)."""
    doc_ids = (["A"] * 4) + (["B"] * 4) + (["C"] * 4)  # 12 questions
    samples_per_q = 4  # solver_n
    dp_size = 2
    batch = _make_batch(doc_ids, samples_per_question=samples_per_q)

    stratified = stratify_by_doc(batch, num_questions=12, samples_per_question=samples_per_q)
    scattered = scatter_for_dispatch(
        stratified, num_questions=12, rollout_n=samples_per_q, dp_size=dp_size,
    )

    chunk_size = samples_per_q // dp_size  # 2 rollouts per question per rank
    per_rank_len = 12 * chunk_size  # 24
    assert len(scattered) == per_rank_len * dp_size

    sids = scattered.batch["sample_id"].tolist()

    # Per-rank question order — the q_idx of every chunk_size-th sample.
    rank_question_orders = []
    for rank in range(dp_size):
        rank_slice = sids[rank * per_rank_len : (rank + 1) * per_rank_len]
        rank_questions = [rank_slice[i] // 1000 for i in range(0, per_rank_len, chunk_size)]
        rank_question_orders.append(rank_questions)

    # All ranks must see the same question order — this is what makes
    # contiguous minibatch splits aligned across ranks for FSDP grad reduction.
    assert rank_question_orders[0] == rank_question_orders[1], (
        f"Per-rank question orders diverge: rank0={rank_question_orders[0]}, "
        f"rank1={rank_question_orders[1]}"
    )

    # Within each rank's chunk for a question, rollouts must be contiguous and
    # the union across ranks must cover all `samples_per_q` rollouts of that q.
    for rank in range(dp_size):
        rank_slice = sids[rank * per_rank_len : (rank + 1) * per_rank_len]
        for q_pos in range(0, per_rank_len, chunk_size):
            chunk = rank_slice[q_pos : q_pos + chunk_size]
            q_idx = chunk[0] // 1000
            rollout_indices = [s % 1000 for s in chunk]
            # All from the same question
            assert all(s // 1000 == q_idx for s in chunk), f"rank {rank} chunk mixes questions: {chunk}"
            # Rollout indices belong to a contiguous chunk_size-block of [0, samples_per_q)
            assert rollout_indices == sorted(rollout_indices)


def test_stratify_doc_id_key_param():
    """The `doc_id_key` param actually controls which non_tensor field is read."""
    # Stick a misleading "doc_id" field that would give a wrong stratification,
    # plus a "real_doc" field with the truth — request stratification on real_doc.
    doc_ids_real = (["A"] * 4) + (["B"] * 2) + (["C"] * 2)
    samples_per_q = 1
    batch = _make_batch(doc_ids_real, samples_per_question=samples_per_q)

    # Add a deliberately wrong "wrong_key" field — equal-size singleton "buckets"
    # so stratifying on it would just return identity order.
    batch.non_tensor_batch["wrong_key"] = np.array(
        [f"unique_{i}" for i in range(len(doc_ids_real))], dtype=object,
    )

    out_correct = stratify_by_doc(batch, 8, samples_per_q, doc_id_key="doc_id")
    out_wrong = stratify_by_doc(batch, 8, samples_per_q, doc_id_key="wrong_key")

    q_perm_correct = _question_order(out_correct, samples_per_q)
    q_perm_wrong = _question_order(out_wrong, samples_per_q)

    # Correct stratification interleaves docs; wrong key (all singletons) leaves
    # the original order intact (every "doc" has size 1, position = 0.5, sorted by
    # tiebreak = first-seen index = q_idx).
    assert q_perm_wrong == list(range(8)), f"singleton-bucket case should be identity: {q_perm_wrong}"
    assert q_perm_correct != list(range(8)), "real doc_id should reorder questions"
