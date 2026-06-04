"""Tests for US-008: Write run_metadata.json during training.

Covers:
- run_metadata.json is written to the remote root after backend init
- Contains required fields: solver/generator model IDs, chat template presence,
  run name, backend type, creation timestamp
- Does NOT contain tokens or secrets
- Backend type is correctly derived from URI scheme
- Handles missing upload manager gracefully
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf


def _make_trainer(
    remote_sync_path="s3://bucket/experiment",
    solver_model="Qwen/Qwen3-8B",
    generator_model="Qwen/Qwen3-8B",
    group_name="test-run",
    solver_custom_chat_template=None,
    generator_custom_chat_template=None,
    hf_token=None,
):
    """Create a minimal SelfEvolutionTrainer with mocked dependencies."""
    config = OmegaConf.create({
        "training": {
            "remote_sync_path": remote_sync_path,
            "max_gen_loop": 10,
            "max_ans_loop": 5,
        },
        "solver": {
            "model": {
                "path": solver_model,
                "custom_chat_template": solver_custom_chat_template,
            },
        },
        "generator": {
            "model": {
                "path": generator_model,
                "custom_chat_template": generator_custom_chat_template,
            },
        },
        "wandb": {
            "group_name": group_name,
        },
        "remote": {
            "hf_token": hf_token,
        },
    })

    # Patch __init__ to avoid full initialization
    with patch(
        "verl_inf_evolve.trainer.self_evolution_trainer.SelfEvolutionTrainer.__init__",
        return_value=None,
    ):
        from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer

        trainer = SelfEvolutionTrainer.__new__(SelfEvolutionTrainer)
        trainer.config = config

    return trainer


class TestWriteRunMetadata:
    """Tests for _write_run_metadata."""

    def test_writes_metadata_to_remote(self):
        trainer = _make_trainer()
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = True

        trainer._write_run_metadata()

        trainer._upload_manager.upload_bytes_sync.assert_called_once()
        args = trainer._upload_manager.upload_bytes_sync.call_args
        assert args[0][1] == "run_metadata.json"

        # Parse the written JSON
        metadata = json.loads(args[0][0])
        assert metadata["solver_model_id"] == "Qwen/Qwen3-8B"
        assert metadata["generator_model_id"] == "Qwen/Qwen3-8B"
        assert metadata["run_name"] == "test-run"
        assert metadata["wandb_group"] == "test-run"
        assert metadata["backend_type"] == "r2"
        assert metadata["resolved_remote_sync_path"] == "s3://bucket/experiment"
        assert "created_at" in metadata

    def test_contains_required_fields(self):
        trainer = _make_trainer()
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = True

        trainer._write_run_metadata()

        data = json.loads(
            trainer._upload_manager.upload_bytes_sync.call_args[0][0]
        )
        required_fields = [
            "solver_model_id",
            "generator_model_id",
            "solver_custom_chat_template",
            "generator_custom_chat_template",
            "run_name",
            "wandb_entity",
            "wandb_project",
            "wandb_group",
            "resolved_remote_sync_path",
            "backend_type",
            "created_at",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_chat_template_false_when_absent(self):
        trainer = _make_trainer()
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = True

        trainer._write_run_metadata()

        data = json.loads(
            trainer._upload_manager.upload_bytes_sync.call_args[0][0]
        )
        assert data["solver_custom_chat_template"] is False
        assert data["generator_custom_chat_template"] is False

    def test_chat_template_true_when_present(self):
        trainer = _make_trainer(
            solver_custom_chat_template="some jinja template",
            generator_custom_chat_template="another template",
        )
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = True

        trainer._write_run_metadata()

        data = json.loads(
            trainer._upload_manager.upload_bytes_sync.call_args[0][0]
        )
        assert data["solver_custom_chat_template"] is True
        assert data["generator_custom_chat_template"] is True

    def test_backend_type_r2_for_s3(self):
        trainer = _make_trainer(remote_sync_path="s3://bucket/prefix")
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = True

        trainer._write_run_metadata()

        data = json.loads(
            trainer._upload_manager.upload_bytes_sync.call_args[0][0]
        )
        assert data["backend_type"] == "r2"

    def test_backend_type_r2_for_r2_scheme(self):
        trainer = _make_trainer(remote_sync_path="r2://bucket/prefix")
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = True

        trainer._write_run_metadata()

        data = json.loads(
            trainer._upload_manager.upload_bytes_sync.call_args[0][0]
        )
        assert data["backend_type"] == "r2"

    def test_backend_type_hf(self):
        trainer = _make_trainer(
            remote_sync_path="hf://datasets/org/repo/prefix"
        )
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = True

        trainer._write_run_metadata()

        data = json.loads(
            trainer._upload_manager.upload_bytes_sync.call_args[0][0]
        )
        assert data["backend_type"] == "hf"

    def test_backend_type_gs(self):
        trainer = _make_trainer(remote_sync_path="gs://bucket/prefix")
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = True

        trainer._write_run_metadata()

        data = json.loads(
            trainer._upload_manager.upload_bytes_sync.call_args[0][0]
        )
        assert data["backend_type"] == "gs"

    def test_no_tokens_in_metadata(self):
        trainer = _make_trainer(hf_token="secret-token-value")
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = True

        trainer._write_run_metadata()

        raw = trainer._upload_manager.upload_bytes_sync.call_args[0][0]
        raw_str = raw.decode("utf-8")
        assert "secret-token-value" not in raw_str

    def test_no_op_when_remote_sync_path_is_none(self):
        trainer = _make_trainer(remote_sync_path=None)
        trainer._upload_manager = MagicMock()

        trainer._write_run_metadata()

        trainer._upload_manager.upload_bytes_sync.assert_not_called()

    def test_no_op_when_no_upload_manager(self):
        trainer = _make_trainer()
        # No _upload_manager attribute at all

        # Should not raise
        trainer._write_run_metadata()

    def test_custom_model_ids(self):
        trainer = _make_trainer(
            solver_model="meta-llama/Llama-3.1-8B",
            generator_model="Qwen/Qwen3-4B",
        )
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = True

        trainer._write_run_metadata()

        data = json.loads(
            trainer._upload_manager.upload_bytes_sync.call_args[0][0]
        )
        assert data["solver_model_id"] == "meta-llama/Llama-3.1-8B"
        assert data["generator_model_id"] == "Qwen/Qwen3-4B"

    def test_upload_failure_does_not_raise(self):
        trainer = _make_trainer()
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = False

        # Should not raise even if upload fails
        trainer._write_run_metadata()

    def test_created_at_is_iso_format(self):
        trainer = _make_trainer()
        trainer._upload_manager = MagicMock()
        trainer._upload_manager.upload_bytes_sync.return_value = True

        trainer._write_run_metadata()

        data = json.loads(
            trainer._upload_manager.upload_bytes_sync.call_args[0][0]
        )
        # Should be parseable as ISO 8601
        from datetime import datetime
        dt = datetime.fromisoformat(data["created_at"])
        assert dt is not None
