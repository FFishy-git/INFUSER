import json

import torch

from verl_inf_evolve.workers.hf_checkpoint_utils import (
    augment_hf_state_dict_for_tied_embeddings,
)


def test_augment_hf_state_dict_adds_missing_lm_head_for_tied_embeddings(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    embed = torch.randn(4, 3)
    state_dict = {"model.embed_tokens.weight": embed}

    result = augment_hf_state_dict_for_tied_embeddings(str(tmp_path), state_dict)

    assert result["lm_head.weight"] is embed


def test_augment_hf_state_dict_leaves_existing_lm_head_untouched(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    embed = torch.randn(4, 3)
    lm_head = torch.randn(4, 3)
    state_dict = {
        "model.embed_tokens.weight": embed,
        "lm_head.weight": lm_head,
    }

    result = augment_hf_state_dict_for_tied_embeddings(str(tmp_path), state_dict)

    assert result["lm_head.weight"] is lm_head


def test_augment_hf_state_dict_skips_non_tied_embeddings(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"tie_word_embeddings": False}))
    embed = torch.randn(4, 3)
    state_dict = {"model.embed_tokens.weight": embed}

    result = augment_hf_state_dict_for_tied_embeddings(str(tmp_path), state_dict)

    assert "lm_head.weight" not in result
