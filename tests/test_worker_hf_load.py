from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch

from verl_inf_evolve.workers.self_evolution_worker import (
    SelfEvolutionActorRolloutRefWorker,
)


def _make_worker_stub() -> SelfEvolutionActorRolloutRefWorker:
    worker = object.__new__(SelfEvolutionActorRolloutRefWorker)
    worker._is_offload_param = False
    worker.actor_module_fsdp = object()
    return worker


def test_resolve_hf_safetensor_files_rejects_missing_shard(tmp_path) -> None:
    worker = _make_worker_stub()
    hf_dir = tmp_path / "huggingface"
    hf_dir.mkdir()
    (hf_dir / "model-00001-of-00002.safetensors").write_bytes(b"abc")
    (hf_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer1.weight": "model-00001-of-00002.safetensors",
                    "layer2.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing 1 shard file"):
        worker._resolve_hf_safetensor_files(str(hf_dir))


def test_load_hf_checkpoint_fails_before_fsdp_load_on_local_read_error(tmp_path) -> None:
    worker = _make_worker_stub()
    hf_dir = tmp_path / "huggingface"
    hf_dir.mkdir()
    (hf_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )

    with patch(
        "verl_inf_evolve.workers.self_evolution_worker.augment_hf_state_dict_for_tied_embeddings"
    ) as augment_mock, patch(
        "verl.utils.fsdp_utils.fsdp2_load_full_state_dict"
    ) as fsdp_load_mock:
        with pytest.raises(RuntimeError, match="failed to read HF checkpoint"):
            worker.load_hf_checkpoint(str(hf_dir))

    augment_mock.assert_not_called()
    fsdp_load_mock.assert_not_called()


def test_load_hf_checkpoint_fails_when_other_rank_reports_read_error(tmp_path) -> None:
    worker = _make_worker_stub()
    hf_dir = tmp_path / "huggingface"
    hf_dir.mkdir()
    (hf_dir / "model.safetensors").write_bytes(b"abc")

    def fake_all_gather_object(object_list, obj, group=None) -> None:
        object_list[0] = {"rank": "0", "error": None}
        object_list[1] = {"rank": "1", "error": "rank=1 failed to read HF checkpoint from /remote: corrupt shard"}

    with patch(
        "safetensors.torch.load_file",
        return_value={"model.embed_tokens.weight": torch.tensor([1.0])},
    ), patch(
        "verl_inf_evolve.workers.self_evolution_worker.augment_hf_state_dict_for_tied_embeddings",
        side_effect=lambda local_path, state_dict, logger=None: state_dict,
    ), patch(
        "verl.utils.fsdp_utils.fsdp2_load_full_state_dict"
    ) as fsdp_load_mock, patch(
        "verl_inf_evolve.workers.self_evolution_worker.dist.is_available", return_value=True
    ), patch(
        "verl_inf_evolve.workers.self_evolution_worker.dist.is_initialized", return_value=True
    ), patch(
        "verl_inf_evolve.workers.self_evolution_worker.dist.get_world_size", return_value=2
    ), patch(
        "verl_inf_evolve.workers.self_evolution_worker.dist.get_rank", return_value=0
    ), patch(
        "verl_inf_evolve.workers.self_evolution_worker.dist.all_gather_object",
        side_effect=fake_all_gather_object,
    ):
        with pytest.raises(RuntimeError, match="rank=1 failed to read HF checkpoint"):
            worker.load_hf_checkpoint(str(hf_dir))

    fsdp_load_mock.assert_not_called()
