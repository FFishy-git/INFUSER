"""Data manipulation utilities for DataProto batches."""

from __future__ import annotations

from collections import Counter

from verl.protocol import DataProto


def stratify_by_doc(
    data: DataProto,
    num_questions: int,
    samples_per_question: int,
    doc_id_key: str = "doc_id",
) -> DataProto:
    """Reorder a question-grouped batch so each doc's questions are spread evenly.

    The input is expected to be ordered by question groups, with all
    ``samples_per_question`` rollouts for each question contiguous:
    ``[Q0_r0, ..., Q0_r(n-1), Q1_r0, ..., Q1_r(n-1), ...]``.

    Questions are typically doc-grouped on entry (Q0..Q_{g-1} from doc0, then
    doc1, ...). This function permutes only the question axis so that each
    document's questions sit at evenly-spaced slots over the global question
    timeline (Bresenham-style "ideal position" spread). Rollouts within a
    question stay contiguous and intact, so a downstream
    :func:`scatter_for_dispatch` call is unaffected.

    With doc sizes ``k_i`` (questions per doc), question ``j`` of doc ``i``
    is assigned ideal position ``(j + 0.5) / k_i``. Sorting by position gives
    a sequence in which:

    * each doc's questions appear at evenly-spaced positions over the timeline,
    * within any contiguous window, doc representation is proportional to
      bucket size (no doc is starved or piled up at the tail), and
    * unequal bucket sizes are handled uniformly (no all-from-one-doc tail).

    The permutation is deterministic and computed from the batch's own
    ``non_tensor_batch[doc_id_key]`` — identical across DP ranks without any
    seed broadcast.

    Args:
        data: DataProto already ordered as ``[Q0_all, Q1_all, ..., QN_all]``.
        num_questions: Number of distinct questions (``N``).
        samples_per_question: Rollouts per question (``solver_n``).
        doc_id_key: Field in ``non_tensor_batch`` holding the document id
            for each sample.

    Returns:
        Reordered DataProto, still in ``[Q_all, Q_all, ...]`` layout but with
        questions permuted so docs are stratified.
    """
    assert len(data) == num_questions * samples_per_question, (
        f"len(data)={len(data)} != num_questions ({num_questions}) * "
        f"samples_per_question ({samples_per_question})"
    )

    doc_ids = data.non_tensor_batch[doc_id_key]
    # One doc_id per question (taken from the first sample of each block)
    question_doc_ids = [doc_ids[q * samples_per_question] for q in range(num_questions)]
    doc_size = Counter(question_doc_ids)

    within_doc_idx: dict[str, int] = {}
    doc_first_seen: dict[str, int] = {}
    positions: list[tuple[float, int, int]] = []  # (ideal_pos, doc_tiebreak, q_idx)
    for q_idx, doc_id in enumerate(question_doc_ids):
        j = within_doc_idx.get(doc_id, 0)
        within_doc_idx[doc_id] = j + 1
        if doc_id not in doc_first_seen:
            doc_first_seen[doc_id] = len(doc_first_seen)
        positions.append(((j + 0.5) / doc_size[doc_id], doc_first_seen[doc_id], q_idx))

    positions.sort()
    q_perm = [q for _, _, q in positions]

    sample_perm = [
        q * samples_per_question + r
        for q in q_perm
        for r in range(samples_per_question)
    ]
    return data.select_idxs(sample_perm)


def scatter_for_dispatch(
    data: DataProto,
    num_questions: int,
    rollout_n: int,
    dp_size: int,
) -> DataProto:
    """Reorder data for scattered dispatch across FSDP ranks.

    Ensures that after contiguous dispatch splitting, each rank's i-th
    mini-batch contains answers for the same question across all ranks.
    This is critical for FSDP gradient reduction to properly aggregate
    per-question gradients.

    Input ordering:  ``[Q1_all, Q2_all, ..., QN_all]``
    Output ordering: ``[Q1_r0, Q2_r0, ..., QN_r0, Q1_r1, ..., QN_r1, ...]``

    Reference: v2_key_points #6 (scattered interleaved data chunking).

    Args:
        data: DataProto ordered by question groups.
        num_questions: Number of distinct questions.
        rollout_n: Total answers per question (across all ranks).
        dp_size: Number of data-parallel ranks (FSDP world size).

    Returns:
        Reordered DataProto ready for contiguous dispatch splitting.
    """
    assert rollout_n % dp_size == 0, (
        f"rollout_n ({rollout_n}) must be divisible by dp_size ({dp_size})"
    )
    chunk_size = rollout_n // dp_size

    indices = []
    for rank in range(dp_size):
        for q in range(num_questions):
            for a in range(chunk_size):
                indices.append(q * rollout_n + rank * chunk_size + a)

    return data.select_idxs(indices)


def derive_gen_questions(gen_output: DataProto) -> list[dict]:
    """Reconstruct gen_questions list from gen_output's non_tensor_batch fields.

    This replaces the need to persist ``gen_questions.json`` separately —
    all required fields are already embedded in ``gen_output`` by
    ``parse_generated_questions()``.
    """
    nt = gen_output.non_tensor_batch
    parsed_ok = nt["parsed_ok"]
    questions = []
    for i in range(len(parsed_ok)):
        if not parsed_ok[i]:
            continue
        q = {
            "question_id": nt["question_id"][i],
            "question_text": nt["question_text"][i],
            "choices": nt["choices"][i],
            "ground_truth": nt["ground_truth"][i],
            "doc_id": nt["doc_id"][i],
        }
        for field in (
            "is_mcq",
            "benchmark_type",
            "data_source",
            "answer_type",
        ):
            if field in nt and nt[field][i] is not None:
                q[field] = nt[field][i]
        questions.append(q)
    return questions

def batch_num_tokens(
    data: DataProto,
    dp_size: int,
    ppo_micro_batch_size_per_gpu: int
) -> float:
    """Compute the per-micro-batch normalizer for the loss normalization in GRPO.

    Args:
        data (DataProto): The input data batch containing fields "response_mask"
        dp_size (int): The data parallel size (number of FSDP ranks)
        ppo_micro_batch_size_per_gpu (int): The size of the micro-batch per GPU used for gradient accumulation
    """
    loss_mask = data["response_mask"]  # Shape: [total_batch_size, seq_len]
    return loss_mask.float().sum() / loss_mask.shape[0] * (ppo_micro_batch_size_per_gpu * dp_size)
    
