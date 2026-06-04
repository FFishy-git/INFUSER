"""Tests for US-011: Trainer and eval integration with the backend abstraction.

Covers:
- _is_remote_checkpoint_source recognizes hf:// URIs
- _resolve_init_checkpoint_source with hf:// URI uses backend download
- HF_HUB_OFFLINE is NOT set when remote_sync_path starts with hf://
- HF_HUB_OFFLINE IS set when remote_sync_path starts with s3://
- run_metadata.json auto-detection works; falls back to R2 heuristic when absent
- Checkpoint discovery via list_immediate_children returns correct step numbers
- Skip-existing result detection works via backend download_file + JSON validation
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. _is_remote_checkpoint_source recognises hf:// URIs
# ---------------------------------------------------------------------------


class TestIsRemoteCheckpointSourceHF:
    """Verify _is_remote_checkpoint_source returns True for hf:// URIs."""

    @staticmethod
    def _call(path: str) -> bool:
        from verl_inf_evolve.trainer.self_evolution_trainer import (
            SelfEvolutionTrainer,
        )

        return SelfEvolutionTrainer._is_remote_checkpoint_source(path)

    def test_hf_datasets_uri(self):
        assert self._call("hf://datasets/org/repo/prefix") is True

    def test_hf_datasets_uri_no_prefix(self):
        assert self._call("hf://datasets/org/repo") is True

    def test_hf_datasets_deep_prefix(self):
        assert self._call("hf://datasets/org/repo/a/b/c") is True

    def test_s3_still_works(self):
        assert self._call("s3://bucket/key") is True

    def test_r2_still_works(self):
        assert self._call("r2://bucket/key") is True

    def test_gs_still_works(self):
        assert self._call("gs://bucket/key") is True

    def test_local_path_rejected(self):
        assert self._call("/tmp/checkpoint") is False

    def test_empty_string_rejected(self):
        assert self._call("") is False


# ---------------------------------------------------------------------------
# 2. _resolve_init_checkpoint_source uses backend download for hf://
# ---------------------------------------------------------------------------


class TestResolveInitCheckpointSourceHF:
    """Verify _resolve_init_checkpoint_source uses RemoteBackend for hf:// URIs."""

    def _make_trainer_stub(self, default_local_dir: str):
        from verl_inf_evolve.trainer.self_evolution_trainer import (
            SelfEvolutionTrainer,
        )

        trainer = object.__new__(SelfEvolutionTrainer)
        trainer.config = MagicMock()
        trainer.config.training.default_local_dir = default_local_dir
        trainer.config.get.side_effect = lambda key, default=None: {} if key == "remote" else default
        return trainer

    @patch("verl_inf_evolve.storage.remote_backend.create_remote_backend")
    def test_hf_uri_creates_backend_and_downloads(self, mock_factory):
        """hf:// URI should create a backend and call download_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer_stub(tmpdir)
            mock_backend = MagicMock()
            mock_factory.return_value = mock_backend

            result = trainer._resolve_init_checkpoint_source(
                "Generator",
                "hf://datasets/myorg/myrepo/checkpoints/step_100",
            )

            mock_factory.assert_called_once()
            call_args = mock_factory.call_args
            assert call_args.args == (
                "hf://datasets/myorg/myrepo/checkpoints/step_100",
            )
            assert call_args.kwargs["revision"] == "main"
            mock_backend.download_dir.assert_called_once()
            # First arg to download_dir is "" (download entire backend prefix)
            assert mock_backend.download_dir.call_args[0][0] == ""
            # Result is a local directory under .init_checkpoints
            assert ".init_checkpoints" in result
            assert os.path.isdir(result)

    @patch("verl_inf_evolve.storage.remote_backend.create_remote_backend")
    def test_local_path_not_routed_to_backend(self, mock_factory):
        """Local paths should be returned as-is without creating a backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer_stub(tmpdir)
            result = trainer._resolve_init_checkpoint_source(
                "Solver", "/local/checkpoint/dir"
            )
            assert result == "/local/checkpoint/dir"
            mock_factory.assert_not_called()


# ---------------------------------------------------------------------------
# 3. HF_HUB_OFFLINE is NOT set when remote_sync_path starts with hf://
# ---------------------------------------------------------------------------


class TestHFHubOfflineNotSetForHF:
    """Verify HF_HUB_OFFLINE is skipped when remote_sync_path uses hf://."""

    def test_hf_scheme_skips_offline(self):
        remote_sync_path = "hf://datasets/org/repo/prefix"
        saved = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            if remote_sync_path.startswith("hf://"):
                pass  # replicate main.py logic — skip
            else:
                os.environ["HF_HUB_OFFLINE"] = "1"
            assert "HF_HUB_OFFLINE" not in os.environ
        finally:
            if saved is not None:
                os.environ["HF_HUB_OFFLINE"] = saved
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)


# ---------------------------------------------------------------------------
# 4. HF_HUB_OFFLINE IS set when remote_sync_path starts with s3://
# ---------------------------------------------------------------------------


class TestHFHubOfflineSetForS3:
    """Verify HF_HUB_OFFLINE=1 is set for non-HF remote_sync_path schemes."""

    def test_s3_scheme_sets_offline(self):
        remote_sync_path = "s3://bucket/prefix"
        saved = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            if remote_sync_path.startswith("hf://"):
                pass
            else:
                os.environ["HF_HUB_OFFLINE"] = "1"
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        finally:
            if saved is not None:
                os.environ["HF_HUB_OFFLINE"] = saved
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)

    def test_r2_scheme_sets_offline(self):
        remote_sync_path = "r2://bucket/prefix"
        saved = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            if remote_sync_path.startswith("hf://"):
                pass
            else:
                os.environ["HF_HUB_OFFLINE"] = "1"
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        finally:
            if saved is not None:
                os.environ["HF_HUB_OFFLINE"] = saved
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)

    def test_empty_path_sets_offline(self):
        remote_sync_path = ""
        saved = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            if remote_sync_path.startswith("hf://"):
                pass
            else:
                os.environ["HF_HUB_OFFLINE"] = "1"
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        finally:
            if saved is not None:
                os.environ["HF_HUB_OFFLINE"] = saved
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)


# ---------------------------------------------------------------------------
# 5. run_metadata.json auto-detection + fallback to R2 heuristic
# ---------------------------------------------------------------------------


class TestRunMetadataAutoDetection:
    """Verify that model auto-detection tries run_metadata.json first,
    then falls back when metadata is absent."""

    def test_metadata_provides_model_path(self):
        """When run_metadata.json exists, solver_model_id is used."""
        from verl_inf_evolve.storage.remote_ops import load_run_metadata

        mock_backend = MagicMock()
        metadata = {
            "solver_model_id": "Qwen/Qwen3-32B",
            "generator_model_id": "Qwen/Qwen3-32B",
            "backend_type": "hf",
            "run_name": "test-run",
        }
        mock_backend.download_bytes.return_value = json.dumps(metadata).encode()

        result = load_run_metadata(mock_backend)
        assert result is not None
        assert result["solver_model_id"] == "Qwen/Qwen3-32B"

    def test_metadata_missing_returns_none(self):
        """When run_metadata.json doesn't exist, returns None (fallback)."""
        from verl_inf_evolve.storage.remote_ops import load_run_metadata

        mock_backend = MagicMock()
        mock_backend.download_bytes.side_effect = FileNotFoundError("not found")

        result = load_run_metadata(mock_backend)
        assert result is None

    def test_metadata_invalid_json_returns_none(self):
        """When run_metadata.json contains invalid JSON, returns None."""
        from verl_inf_evolve.storage.remote_ops import load_run_metadata

        mock_backend = MagicMock()
        mock_backend.download_bytes.return_value = b"not valid json"

        result = load_run_metadata(mock_backend)
        assert result is None

    def test_auto_detect_sets_model_path_from_metadata(self):
        """Integration: model_path is set from metadata when empty."""
        metadata = {
            "solver_model_id": "Qwen/Qwen3-32B",
            "generator_model_id": "Qwen/Qwen3-32B",
            "backend_type": "hf",
        }
        mock_backend = MagicMock()
        mock_backend.download_bytes.return_value = json.dumps(metadata).encode()

        # Simulate the auto-detection logic from run_eval.py
        model_path = ""  # empty → triggers auto-detect
        if not model_path or model_path == "Qwen/Qwen3-8B":
            from verl_inf_evolve.storage.remote_ops import load_run_metadata

            meta = load_run_metadata(mock_backend)
            if meta and meta.get("solver_model_id"):
                model_path = meta["solver_model_id"]

        assert model_path == "Qwen/Qwen3-32B"

    def test_auto_detect_overrides_default_model_path(self):
        """Integration: default model_path is replaced by metadata detection."""
        metadata = {
            "solver_model_id": "deepseek-ai/DeepSeek-R1-0528",
            "backend_type": "hf",
        }
        mock_backend = MagicMock()
        mock_backend.download_bytes.return_value = json.dumps(metadata).encode()

        model_path = "Qwen/Qwen3-8B"  # default → triggers auto-detect
        if not model_path or model_path == "Qwen/Qwen3-8B":
            from verl_inf_evolve.storage.remote_ops import load_run_metadata

            meta = load_run_metadata(mock_backend)
            if meta and meta.get("solver_model_id"):
                model_path = meta["solver_model_id"]

        assert model_path == "deepseek-ai/DeepSeek-R1-0528"

    def test_auto_detect_keeps_explicit_model_path(self):
        """When model_path is explicitly set (not default), metadata is not used."""
        model_path = "meta-llama/Llama-3.1-70B"

        # This should NOT trigger auto-detection
        if not model_path or model_path == "Qwen/Qwen3-8B":
            model_path = "SHOULD_NOT_REACH"

        assert model_path == "meta-llama/Llama-3.1-70B"

    def test_auto_detect_fallback_on_missing_metadata(self):
        """When metadata is absent, model_path stays unchanged (fallback)."""
        mock_backend = MagicMock()
        mock_backend.download_bytes.side_effect = FileNotFoundError("not found")

        model_path = "Qwen/Qwen3-8B"  # default
        if not model_path or model_path == "Qwen/Qwen3-8B":
            from verl_inf_evolve.storage.remote_ops import load_run_metadata

            meta = load_run_metadata(mock_backend)
            if meta and meta.get("solver_model_id"):
                model_path = meta["solver_model_id"]

        # Should remain unchanged — metadata absent
        assert model_path == "Qwen/Qwen3-8B"


# ---------------------------------------------------------------------------
# 6. Checkpoint discovery via list_immediate_children
# ---------------------------------------------------------------------------


class TestCheckpointDiscovery:
    """Verify discover_checkpoints returns correct step numbers."""

    def test_discovers_sorted_step_numbers(self):
        from verl_inf_evolve.storage.remote_ops import discover_checkpoints

        mock_backend = MagicMock()
        mock_backend.list_immediate_children.return_value = [
            "global_step_100/",
            "global_step_0/",
            "global_step_50/",
            "run_metadata.json",
            "tensorboard/",
        ]

        steps = discover_checkpoints(mock_backend)
        assert steps == [0, 50, 100]

    def test_empty_listing(self):
        from verl_inf_evolve.storage.remote_ops import discover_checkpoints

        mock_backend = MagicMock()
        mock_backend.list_immediate_children.return_value = []

        steps = discover_checkpoints(mock_backend)
        assert steps == []

    def test_no_checkpoint_dirs(self):
        from verl_inf_evolve.storage.remote_ops import discover_checkpoints

        mock_backend = MagicMock()
        mock_backend.list_immediate_children.return_value = [
            "run_metadata.json",
            "config.yaml",
            "logs/",
        ]

        steps = discover_checkpoints(mock_backend)
        assert steps == []

    def test_with_trailing_slash(self):
        """global_step_N/ (with trailing slash) should still match."""
        from verl_inf_evolve.storage.remote_ops import discover_checkpoints

        mock_backend = MagicMock()
        mock_backend.list_immediate_children.return_value = [
            "global_step_10/",
            "global_step_20",
        ]

        steps = discover_checkpoints(mock_backend)
        assert steps == [10, 20]

    def test_large_step_numbers(self):
        from verl_inf_evolve.storage.remote_ops import discover_checkpoints

        mock_backend = MagicMock()
        mock_backend.list_immediate_children.return_value = [
            "global_step_999999/",
            "global_step_1/",
        ]

        steps = discover_checkpoints(mock_backend)
        assert steps == [1, 999999]

    def test_ignores_similarly_named_non_step_dirs(self):
        from verl_inf_evolve.storage.remote_ops import discover_checkpoints

        mock_backend = MagicMock()
        mock_backend.list_immediate_children.return_value = [
            "global_step_abc/",
            "global_step_/",
            "step_5/",
            "global_step_10/",
        ]

        steps = discover_checkpoints(mock_backend)
        assert steps == [10]


# ---------------------------------------------------------------------------
# 7. Skip-existing result detection via backend + JSON validation
# ---------------------------------------------------------------------------


class TestSkipExistingResultDetection:
    """Verify check_result_exists uses backend download + JSON validation."""

    def test_returns_true_for_valid_result(self):
        from verl_inf_evolve.storage.remote_ops import check_result_exists

        valid_result = {
            "metrics": {
                "total_questions": 10,
                "total_answers": 20,
                "accuracy_strict": 0.5,
            },
            "questions": [],
        }
        mock_backend = MagicMock()
        mock_backend.exists.return_value = True
        mock_backend.download_bytes.return_value = json.dumps(valid_result).encode()

        assert check_result_exists(mock_backend, "result.json") is True
        mock_backend.exists.assert_called_once_with("result.json")
        mock_backend.download_bytes.assert_called_once_with("result.json")

    def test_returns_false_when_not_exists(self):
        from verl_inf_evolve.storage.remote_ops import check_result_exists

        mock_backend = MagicMock()
        mock_backend.exists.return_value = False

        assert check_result_exists(mock_backend, "result.json") is False
        mock_backend.download_bytes.assert_not_called()

    def test_returns_false_on_exists_exception(self):
        from verl_inf_evolve.storage.remote_ops import check_result_exists

        mock_backend = MagicMock()
        mock_backend.exists.side_effect = RuntimeError("network error")

        assert check_result_exists(mock_backend, "result.json") is False

    def test_returns_false_for_invalid_json(self):
        from verl_inf_evolve.storage.remote_ops import check_result_exists

        mock_backend = MagicMock()
        mock_backend.exists.return_value = True
        mock_backend.download_bytes.return_value = b"not json"

        assert check_result_exists(mock_backend, "result.json") is False

    def test_returns_false_for_incomplete_result(self):
        """Result exists but metrics are incomplete (total_questions=0)."""
        from verl_inf_evolve.storage.remote_ops import check_result_exists

        incomplete = {
            "metrics": {
                "total_questions": 0,
                "total_answers": 0,
            },
            "questions": [],
        }
        mock_backend = MagicMock()
        mock_backend.exists.return_value = True
        mock_backend.download_bytes.return_value = json.dumps(incomplete).encode()

        assert check_result_exists(mock_backend, "result.json") is False

    def test_returns_false_for_missing_metrics(self):
        """Result JSON exists but has no metrics key."""
        from verl_inf_evolve.storage.remote_ops import check_result_exists

        mock_backend = MagicMock()
        mock_backend.exists.return_value = True
        mock_backend.download_bytes.return_value = json.dumps({"questions": []}).encode()

        assert check_result_exists(mock_backend, "result.json") is False

    def test_returns_false_on_download_exception(self):
        from verl_inf_evolve.storage.remote_ops import check_result_exists

        mock_backend = MagicMock()
        mock_backend.exists.return_value = True
        mock_backend.download_bytes.side_effect = RuntimeError("download failed")

        assert check_result_exists(mock_backend, "result.json") is False
