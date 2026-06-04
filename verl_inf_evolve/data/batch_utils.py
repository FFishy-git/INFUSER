"""Shared tokenization and padding utilities for building DataProto batches.

Replaces V2's ``prepare_prompts_from_dataset()`` + ``DataProtoAdapter`` pattern.
Much simpler because verl's ``generate_sequences()`` handles response padding
internally — we only need to provide left-padded prompt tokens.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto


def messages_to_dataproto(
    messages_list: list[list[dict[str, str]]],
    non_tensor_metadata: dict[str, list[Any]] | None = None,
    meta_info: dict[str, Any] | None = None,
) -> DataProto:
    """Build a DataProto with raw_prompt for AgentLoopManager consumption.

    Unlike ``tokenize_and_pad_to_dataproto()``, this does NOT tokenize.
    The AgentLoopWorker handles tokenization internally via SingleTurnAgentLoop.

    Args:
        messages_list: List of chat message lists (same format as
            ``tokenize_and_pad_to_dataproto``).
        non_tensor_metadata: Optional per-sample metadata dict.
        meta_info: Optional batch-level metadata dict.

    Returns:
        DataProto with ``raw_prompt`` in ``non_tensor_batch`` and a dummy
        tensor batch (required by DataProto).
    """
    batch_size = len(messages_list)

    # DataProto requires a non-empty TensorDict; use a dummy tensor.
    td = TensorDict(
        {"dummy": torch.zeros(batch_size, 1, dtype=torch.uint8)},
        batch_size=batch_size,
    )

    ntb = {
        "raw_prompt": np.array(messages_list, dtype=object),
    }
    if non_tensor_metadata:
        for key, values in non_tensor_metadata.items():
            assert len(values) == batch_size, (
                f"non_tensor_metadata['{key}'] length {len(values)} != batch_size {batch_size}"
            )
            ntb[key] = np.array(values, dtype=object)

    return DataProto(
        batch=td,
        non_tensor_batch=ntb,
        meta_info=meta_info or {},
    )


def tokenize_and_pad_to_dataproto(
    messages_list: list[list[dict[str, str]]],
    tokenizer,
    non_tensor_metadata: dict[str, list[Any]] | None = None,
    meta_info: dict[str, Any] | None = None,
) -> DataProto:
    """Tokenize chat messages and left-pad into a DataProto for generate_sequences().

    This is the shared helper used by ``prepare_dev_batch()``,
    ``prepare_doc_batch()``, and ``prepare_question_batch()``.

    Args:
        messages_list: List of chat message lists. Each element is a list of
            dicts with ``role`` and ``content`` keys, e.g.::

                [
                    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
                    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
                ]

        tokenizer: HuggingFace tokenizer with ``apply_chat_template()`` support.
        non_tensor_metadata: Optional dict mapping field names to lists of
            per-sample metadata. Each list must have the same length as
            ``messages_list``. Values are stored in ``DataProto.non_tensor_batch``
            as numpy arrays with ``dtype=object``.
        meta_info: Optional dict of batch-level metadata stored in
            ``DataProto.meta_info``.

    Returns:
        DataProto with:
            - ``batch["input_ids"]``: left-padded token ids ``[B, max_len]``
            - ``batch["attention_mask"]``: ``1`` for real tokens, ``0`` for padding
            - ``batch["position_ids"]``: position indices (0-indexed, respecting padding)
            - ``non_tensor_batch``: per-sample metadata from ``non_tensor_metadata``
            - ``meta_info``: batch-level metadata
    """
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)

    # --- Tokenize each message list ---
    all_token_ids: list[list[int]] = []
    for messages in messages_list:
        # apply_chat_template returns the full prompt string
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        all_token_ids.append(token_ids)

    batch_size = len(all_token_ids)
    max_len = max(len(ids) for ids in all_token_ids)

    # --- Left-pad to max_len ---
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, ids in enumerate(all_token_ids):
        seq_len = len(ids)
        offset = max_len - seq_len  # left-pad offset
        input_ids[i, offset:] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, offset:] = 1

    # --- Position IDs (0-indexed, respecting left-padding) ---
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.clamp_(min=0)

    # --- Build DataProto ---
    td = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )

    # non_tensor_batch: convert lists to numpy object arrays
    ntb = {}
    if non_tensor_metadata:
        for key, values in non_tensor_metadata.items():
            assert len(values) == batch_size, (
                f"non_tensor_metadata['{key}'] length {len(values)} != batch_size {batch_size}"
            )
            ntb[key] = np.array(values, dtype=object)

    return DataProto(
        batch=td,
        non_tensor_batch=ntb,
        meta_info=meta_info or {},
    )
