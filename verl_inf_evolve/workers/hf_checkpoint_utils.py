from __future__ import annotations

import json
import logging
import os
from typing import Any


def augment_hf_state_dict_for_tied_embeddings(
    local_path: str,
    state_dict: dict[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Fill missing output embeddings for tied-embedding HF checkpoints.

    Some vanilla Hugging Face checkpoints omit ``lm_head.weight`` when
    ``tie_word_embeddings=true`` and the output weights are tied to
    ``model.embed_tokens.weight``. Our FSDP loader expects a full state dict,
    so materialize the missing alias before loading.
    """
    if "lm_head.weight" in state_dict:
        return state_dict

    embed_tokens = state_dict.get("model.embed_tokens.weight")
    if embed_tokens is None:
        return state_dict

    config_path = os.path.join(local_path, "config.json")
    if not os.path.exists(config_path):
        return state_dict

    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to inspect %s for tied embeddings: %s", config_path, exc)
        return state_dict

    if not config.get("tie_word_embeddings", False):
        return state_dict

    state_dict["lm_head.weight"] = embed_tokens
    if logger is not None:
        logger.info(
            "Synthesized lm_head.weight from model.embed_tokens.weight for tied-embedding checkpoint %s",
            local_path,
        )
    return state_dict
