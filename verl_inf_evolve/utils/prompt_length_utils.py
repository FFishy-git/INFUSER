"""Prompt-length utilities for chat-template based batches."""

from __future__ import annotations

from typing import Any
import logging


def should_filter_by_prompt_length(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    prompt_text: str | None = None,
    max_prompt_tokens: int,
    logger: logging.Logger,
    sample_kind: str,
    sample_id: Any,
    log_filtered_item: bool = True,
) -> bool:
    """Return True when a prompt exceeds ``max_prompt_tokens``.

    This helper centralizes the repeated chat-template token counting logic
    used before rollout batch construction.

    If tokenization fails (e.g. generated question content that the
    tokenizer backend cannot encode), the sample is filtered out and a
    warning is logged rather than crashing the training loop.
    """
    try:
        if prompt_text is not None:
            token_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        else:
            token_len = len(tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            ))
    except (TypeError, ValueError, Exception) as exc:
        op_name = "tokenizer.encode" if prompt_text is not None else "apply_chat_template"
        logger.warning(
            "Filtered %s %s: %s failed (%s: %s)",
            sample_kind,
            sample_id,
            op_name,
            type(exc).__name__,
            exc,
        )
        return True
    if token_len <= max_prompt_tokens:
        return False

    if log_filtered_item:
        logger.warning(
            "Filtered %s %s: prompt length %d exceeds prompt_length %d",
            sample_kind,
            sample_id,
            token_len,
            max_prompt_tokens,
        )
    return True
