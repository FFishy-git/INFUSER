"""Tests for verl_inf_evolve.gen_eval.checkpoint_manager.

Tests for both CheckpointPrefetcher and GenOutputPrefetcher.
"""

from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from verl_inf_evolve.gen_eval.checkpoint_manager import (
    CheckpointPrefetcher,
    GenOutputPrefetcher,
    _REQUIRED_GEN_OUTPUT_KEYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All tests mock create_remote_backend so the prefetchers get a mock backend.
FACTORY_PATH = "verl_inf_evolve.gen_eval.checkpoint_manager.create_remote_backend"


def _create_valid_checkpoint(base_dir: str, ans_loop_idx: int) -> str:
    """Create a minimal valid HuggingFace checkpoint on disk."""
    ckpt_dir = os.path.join(
        base_dir,
        f"global_step_{ans_loop_idx}",
        "generator",
        "huggingface",
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    Path(os.path.join(ckpt_dir, "config.json")).write_text('{"model_type": "qwen2"}')
    Path(os.path.join(ckpt_dir, "model.safetensors")).write_bytes(b"\x00" * 16)
    return ckpt_dir


def _make_download_dir_success(dst_path_override: str | None = None):
    """Return a side_effect for backend.download_dir that creates valid checkpoint files."""
    def _side_effect(key: str, local_path: str) -> bool:
        target = dst_path_override or local_path
        os.makedirs(target, exist_ok=True)
        Path(os.path.join(target, "config.json")).write_text('{"model_type": "qwen2"}')
        Path(os.path.join(target, "model.safetensors")).write_bytes(b"\x00" * 16)
        return True
    return _side_effect


def _download_dir_fail(key: str, local_path: str) -> bool:
    return False


def _download_dir_incomplete(key: str, local_path: str) -> bool:
    """Download config but no weights."""
    os.makedirs(local_path, exist_ok=True)
    Path(os.path.join(local_path, "config.json")).write_text('{"model_type": "qwen2"}')
    return True


@pytest.fixture
def mock_backend():
    """Create a mock backend and patch the factory to return it."""
    backend = MagicMock()
    with patch(FACTORY_PATH, return_value=backend) as mock_factory:
        yield backend, mock_factory


# ---------------------------------------------------------------------------
# Tests: Construction
# ---------------------------------------------------------------------------


class TestCheckpointPrefetcherInit:
    def test_basic_construction(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        pf = CheckpointPrefetcher(
            remote_base_path="s3://my-bucket/experiment/trajectory",
            local_cache_dir=str(tmp_path),
            prefetch_count=2,
        )
        assert pf.remote_base_path == "s3://my-bucket/experiment/trajectory"
        assert pf.local_cache_dir == str(tmp_path)
        assert pf.prefetch_count == 2
        assert pf.cleanup_after_eval is True
        assert pf._backend is backend
        pf.shutdown()

    def test_strips_trailing_slash(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/path/",
            local_cache_dir=str(tmp_path),
        )
        assert pf.remote_base_path == "s3://bucket/path"
        pf.shutdown()

    def test_default_prefetch_count(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/path",
            local_cache_dir=str(tmp_path),
        )
        assert pf.prefetch_count == 1
        pf.shutdown()

    def test_cleanup_disabled(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/path",
            local_cache_dir=str(tmp_path),
            cleanup_after_eval=False,
        )
        assert pf.cleanup_after_eval is False
        pf.shutdown()

    def test_hf_uri_construction(self, tmp_path: Path, mock_backend) -> None:
        """HF URIs are accepted and routed through the factory."""
        backend, mock_factory = mock_backend
        pf = CheckpointPrefetcher(
            remote_base_path="hf://datasets/org/repo/prefix",
            local_cache_dir=str(tmp_path),
        )
        mock_factory.assert_called_once_with("hf://datasets/org/repo/prefix")
        assert pf._backend is backend
        pf.shutdown()


# ---------------------------------------------------------------------------
# Tests: get_checkpoint with mocked backend
# ---------------------------------------------------------------------------


class TestGetCheckpoint:
    def test_downloads_and_returns_path(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_dir.side_effect = _make_download_dir_success()
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        path = pf.get_checkpoint(5)
        expected = os.path.join(str(tmp_path), "global_step_5", "generator", "huggingface")
        assert path == expected
        assert os.path.isfile(os.path.join(path, "config.json"))
        assert os.path.isfile(os.path.join(path, "model.safetensors"))
        pf.shutdown()

    def test_uses_correct_remote_key(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_dir.side_effect = _make_download_dir_success()
        pf = CheckpointPrefetcher(
            remote_base_path="s3://my-bucket/experiment/traj",
            local_cache_dir=str(tmp_path),
        )
        pf.get_checkpoint(10)
        backend.download_dir.assert_called_once()
        key_arg = backend.download_dir.call_args[0][0]
        assert key_arg == "global_step_10/generator/huggingface"
        pf.shutdown()

    def test_raises_on_download_failure(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_dir.side_effect = _download_dir_fail
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        with pytest.raises(RuntimeError, match="Failed to download"):
            pf.get_checkpoint(3)
        pf.shutdown()

    def test_raises_on_invalid_checkpoint(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_dir.side_effect = _download_dir_incomplete
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        with pytest.raises(RuntimeError, match="invalid"):
            pf.get_checkpoint(3)
        pf.shutdown()

    def test_skips_download_if_cached(self, tmp_path: Path, mock_backend) -> None:
        """If a valid checkpoint already exists locally, no backend call is made."""
        backend, _ = mock_backend
        _create_valid_checkpoint(str(tmp_path), 7)
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        path = pf.get_checkpoint(7)
        backend.download_dir.assert_not_called()
        expected = os.path.join(str(tmp_path), "global_step_7", "generator", "huggingface")
        assert path == expected
        pf.shutdown()


# ---------------------------------------------------------------------------
# Tests: Prefetch triggers download of next N checkpoints
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_prefetch_triggers_next_checkpoint(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_dir.side_effect = _make_download_dir_success()
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            prefetch_count=1,
        )
        pf.set_indices([0, 5, 10, 15, 20])

        pf.get_checkpoint(0)
        time.sleep(0.5)

        # download_dir should have been called twice: once for idx=0, once for idx=5
        assert backend.download_dir.call_count == 2
        keys = {call[0][0] for call in backend.download_dir.call_args_list}
        assert any("global_step_0" in k for k in keys)
        assert any("global_step_5" in k for k in keys)
        pf.shutdown()

    def test_prefetch_multiple(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_dir.side_effect = _make_download_dir_success()
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            prefetch_count=2,
        )
        pf.set_indices([0, 5, 10, 15])

        pf.get_checkpoint(0)
        time.sleep(0.5)

        # Should have downloaded: 0, 5, 10 (current + 2 prefetch)
        assert backend.download_dir.call_count == 3
        pf.shutdown()

    def test_prefetch_at_end_of_list(self, tmp_path: Path, mock_backend) -> None:
        """When near the end of the list, prefetch only remaining indices."""
        backend, _ = mock_backend
        backend.download_dir.side_effect = _make_download_dir_success()
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            prefetch_count=3,
        )
        pf.set_indices([0, 5, 10])

        pf.get_checkpoint(5)
        time.sleep(0.5)

        # Should download 5 and 10 (only 1 more available, not 3)
        assert backend.download_dir.call_count == 2
        pf.shutdown()

    def test_no_duplicate_downloads(self, tmp_path: Path, mock_backend) -> None:
        """Requesting the same checkpoint twice should not trigger a second download."""
        backend, _ = mock_backend
        backend.download_dir.side_effect = _make_download_dir_success()
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            prefetch_count=0,
        )
        pf.get_checkpoint(5)
        pf.get_checkpoint(5)

        assert backend.download_dir.call_count == 1
        pf.shutdown()

    def test_no_prefetch_without_indices(self, tmp_path: Path, mock_backend) -> None:
        """Without set_indices, only the requested checkpoint is downloaded."""
        backend, _ = mock_backend
        backend.download_dir.side_effect = _make_download_dir_success()
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            prefetch_count=2,
        )
        pf.get_checkpoint(5)
        time.sleep(0.5)

        assert backend.download_dir.call_count == 1
        pf.shutdown()


# ---------------------------------------------------------------------------
# Tests: Checkpoint validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_checkpoint(self, tmp_path: Path) -> None:
        ckpt = _create_valid_checkpoint(str(tmp_path), 0)
        assert CheckpointPrefetcher._validate_checkpoint(ckpt) is True

    def test_missing_config(self, tmp_path: Path) -> None:
        ckpt_dir = os.path.join(str(tmp_path), "ckpt")
        os.makedirs(ckpt_dir)
        Path(os.path.join(ckpt_dir, "model.safetensors")).write_bytes(b"\x00")
        assert CheckpointPrefetcher._validate_checkpoint(ckpt_dir) is False

    def test_missing_safetensors(self, tmp_path: Path) -> None:
        ckpt_dir = os.path.join(str(tmp_path), "ckpt")
        os.makedirs(ckpt_dir)
        Path(os.path.join(ckpt_dir, "config.json")).write_text("{}")
        assert CheckpointPrefetcher._validate_checkpoint(ckpt_dir) is False

    def test_nonexistent_dir(self) -> None:
        assert CheckpointPrefetcher._validate_checkpoint("/nonexistent/path") is False

    def test_multiple_safetensors(self, tmp_path: Path) -> None:
        """Valid even with multiple safetensors shards."""
        ckpt_dir = os.path.join(str(tmp_path), "ckpt")
        os.makedirs(ckpt_dir)
        Path(os.path.join(ckpt_dir, "config.json")).write_text("{}")
        Path(os.path.join(ckpt_dir, "model-00001-of-00004.safetensors")).write_bytes(b"\x00")
        Path(os.path.join(ckpt_dir, "model-00002-of-00004.safetensors")).write_bytes(b"\x00")
        assert CheckpointPrefetcher._validate_checkpoint(ckpt_dir) is True


# ---------------------------------------------------------------------------
# Tests: Cleanup of old checkpoints
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_removes_previous_checkpoint(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_dir.side_effect = _make_download_dir_success()
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            cleanup_after_eval=True,
            prefetch_count=0,
        )
        pf.get_checkpoint(0)
        step0_dir = os.path.join(str(tmp_path), "global_step_0")
        assert os.path.isdir(step0_dir)

        # Getting checkpoint 5 should clean up checkpoint 0
        pf.get_checkpoint(5)
        assert not os.path.exists(step0_dir)
        pf.shutdown()

    def test_no_cleanup_when_disabled(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_dir.side_effect = _make_download_dir_success()
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            cleanup_after_eval=False,
            prefetch_count=0,
        )
        pf.get_checkpoint(0)
        pf.get_checkpoint(5)
        step0_dir = os.path.join(str(tmp_path), "global_step_0")
        assert os.path.isdir(step0_dir)
        pf.shutdown()

    def test_no_cleanup_on_first_call(self, tmp_path: Path, mock_backend) -> None:
        """First get_checkpoint should not try to clean up anything."""
        backend, _ = mock_backend
        backend.download_dir.side_effect = _make_download_dir_success()
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            cleanup_after_eval=True,
            prefetch_count=0,
        )
        pf.get_checkpoint(0)
        step0_dir = os.path.join(str(tmp_path), "global_step_0")
        assert os.path.isdir(step0_dir)
        pf.shutdown()


# ---------------------------------------------------------------------------
# Tests: set_indices
# ---------------------------------------------------------------------------


class TestSetIndices:
    def test_set_indices_stores_list(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        pf.set_indices([0, 5, 10, 15, 20])
        assert pf._ans_loop_indices == [0, 5, 10, 15, 20]
        pf.shutdown()

    def test_set_indices_copies_list(self, tmp_path: Path, mock_backend) -> None:
        """Modifying the original list should not affect the prefetcher."""
        backend, _ = mock_backend
        pf = CheckpointPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        indices = [0, 5, 10]
        pf.set_indices(indices)
        indices.append(99)
        assert 99 not in pf._ans_loop_indices
        pf.shutdown()


# ===========================================================================
# GenOutputPrefetcher tests
# ===========================================================================


class _MockDataProto:
    """Picklable mock for DataProto (must be module-level for pickle)."""

    def __init__(self, ntb: dict) -> None:
        self.batch = None
        self.non_tensor_batch = ntb
        self.meta_info = {}


def _make_mock_dataproto(
    batch_size: int = 4,
    include_keys: set[str] | None = None,
    exclude_keys: set[str] | None = None,
) -> _MockDataProto:
    """Create a mock DataProto-like object that can be pickled."""
    all_keys = set(_REQUIRED_GEN_OUTPUT_KEYS)
    if include_keys:
        all_keys = include_keys
    if exclude_keys:
        all_keys -= exclude_keys

    non_tensor_batch = {}
    for key in all_keys:
        if key == "parsed_ok":
            non_tensor_batch[key] = np.array(
                [True] * batch_size, dtype=object
            )
        elif key == "choices":
            non_tensor_batch[key] = np.array(
                [["A", "B", "C", "D"]] * batch_size, dtype=object
            )
        elif key == "ground_truth":
            non_tensor_batch[key] = np.array(
                ["A"] * batch_size, dtype=object
            )
        else:
            non_tensor_batch[key] = np.array(
                [f"{key}_{i}" for i in range(batch_size)], dtype=object
            )

    return _MockDataProto(non_tensor_batch)


def _save_mock_gen_output(base_dir: str, ans_loop_idx: int, **kwargs: object) -> str:
    """Create a mock gen_output.pt file on disk."""
    gen_dir = os.path.join(base_dir, f"ans_{ans_loop_idx}", "gen_0")
    os.makedirs(gen_dir, exist_ok=True)
    file_path = os.path.join(gen_dir, "gen_output.pt")
    mock_data = _make_mock_dataproto(**kwargs)
    with open(file_path, "wb") as f:
        pickle.dump(mock_data, f)
    return file_path


def _make_download_file_gen_output_success():
    """Return a side_effect for backend.download_file that creates a valid gen_output.pt."""
    def _side_effect(key: str, local_path: str) -> bool:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        mock_data = _make_mock_dataproto()
        with open(local_path, "wb") as f:
            pickle.dump(mock_data, f)
        return True
    return _side_effect


def _download_file_fail(key: str, local_path: str) -> bool:
    return False


def _download_file_gen_output_empty(key: str, local_path: str) -> bool:
    """Simulate download that succeeds but produces no file."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return True


def _make_download_file_gen_output_missing_keys():
    """Return a side_effect that creates a gen_output.pt with missing keys."""
    def _side_effect(key: str, local_path: str) -> bool:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        mock_data = _make_mock_dataproto(exclude_keys={"question_id", "choices"})
        with open(local_path, "wb") as f:
            pickle.dump(mock_data, f)
        return True
    return _side_effect


# ---------------------------------------------------------------------------
# Tests: GenOutputPrefetcher construction
# ---------------------------------------------------------------------------


class TestGenOutputPrefetcherInit:
    def test_basic_construction(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        pf = GenOutputPrefetcher(
            remote_base_path="s3://my-bucket/experiment/trajectory",
            local_cache_dir=str(tmp_path),
            prefetch_count=2,
        )
        assert pf.remote_base_path == "s3://my-bucket/experiment/trajectory"
        assert pf.local_cache_dir == str(tmp_path)
        assert pf.prefetch_count == 2
        assert pf.cleanup_after_eval is True
        assert pf._backend is backend
        pf.shutdown()

    def test_strips_trailing_slash(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/path/",
            local_cache_dir=str(tmp_path),
        )
        assert pf.remote_base_path == "s3://bucket/path"
        pf.shutdown()

    def test_default_prefetch_count(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/path",
            local_cache_dir=str(tmp_path),
        )
        assert pf.prefetch_count == 1
        pf.shutdown()

    def test_cleanup_disabled(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/path",
            local_cache_dir=str(tmp_path),
            cleanup_after_eval=False,
        )
        assert pf.cleanup_after_eval is False
        pf.shutdown()

    def test_hf_uri_construction(self, tmp_path: Path, mock_backend) -> None:
        """HF URIs are accepted and routed through the factory."""
        backend, mock_factory = mock_backend
        pf = GenOutputPrefetcher(
            remote_base_path="hf://datasets/org/repo/prefix",
            local_cache_dir=str(tmp_path),
        )
        mock_factory.assert_called_once_with("hf://datasets/org/repo/prefix")
        assert pf._backend is backend
        pf.shutdown()


# ---------------------------------------------------------------------------
# Tests: get_gen_output downloads and returns DataProto
# ---------------------------------------------------------------------------


class TestGetGenOutput:
    def test_downloads_and_returns_dataproto(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_file.side_effect = _make_download_file_gen_output_success()
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            result = pf.get_gen_output(5)
        assert hasattr(result, "non_tensor_batch")
        assert "parsed_ok" in result.non_tensor_batch
        assert "question_id" in result.non_tensor_batch
        assert "question_text" in result.non_tensor_batch
        assert "choices" in result.non_tensor_batch
        assert "ground_truth" in result.non_tensor_batch
        assert "doc_id" in result.non_tensor_batch
        pf.shutdown()

    def test_uses_correct_remote_key(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_file.side_effect = _make_download_file_gen_output_success()
        pf = GenOutputPrefetcher(
            remote_base_path="s3://my-bucket/experiment/traj",
            local_cache_dir=str(tmp_path),
        )
        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            pf.get_gen_output(10)
        backend.download_file.assert_called_once()
        key_arg = backend.download_file.call_args[0][0]
        assert key_arg == "ans_10/gen_0/gen_output.pt"
        pf.shutdown()

    def test_raises_on_download_failure(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_file.side_effect = _download_file_fail
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        with pytest.raises(RuntimeError, match="Failed to download gen_output.pt"):
            pf.get_gen_output(3)
        pf.shutdown()

    def test_raises_when_file_missing_after_download(self, tmp_path: Path, mock_backend) -> None:
        """Backend returns True but the file doesn't exist (edge case)."""
        backend, _ = mock_backend
        backend.download_file.side_effect = _download_file_gen_output_empty
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        with pytest.raises(RuntimeError, match="gen_output.pt not found after download"):
            pf.get_gen_output(3)
        pf.shutdown()

    def test_raises_on_missing_non_tensor_batch_keys(self, tmp_path: Path, mock_backend) -> None:
        """Downloaded gen_output.pt is missing required non_tensor_batch keys."""
        backend, _ = mock_backend
        backend.download_file.side_effect = _make_download_file_gen_output_missing_keys()
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            with pytest.raises(RuntimeError, match="missing required non_tensor_batch keys"):
                pf.get_gen_output(3)
        pf.shutdown()

    def test_skips_download_if_cached(self, tmp_path: Path, mock_backend) -> None:
        """If a valid gen_output.pt already exists locally, no backend call is made."""
        backend, _ = mock_backend
        _save_mock_gen_output(str(tmp_path), 7)
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            result = pf.get_gen_output(7)
        backend.download_file.assert_not_called()
        assert "parsed_ok" in result.non_tensor_batch
        pf.shutdown()


# ---------------------------------------------------------------------------
# Tests: GenOutputPrefetcher prefetch triggers
# ---------------------------------------------------------------------------


class TestGenOutputPrefetch:
    def test_prefetch_triggers_next_gen_output(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_file.side_effect = _make_download_file_gen_output_success()
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            prefetch_count=1,
        )
        pf.set_indices([0, 5, 10, 15, 20])

        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            pf.get_gen_output(0)

        time.sleep(0.5)

        # download_file should have been called twice: once for idx=0, once for idx=5
        assert backend.download_file.call_count == 2
        keys = {call[0][0] for call in backend.download_file.call_args_list}
        assert any("ans_0" in k for k in keys)
        assert any("ans_5" in k for k in keys)
        pf.shutdown()

    def test_prefetch_multiple(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_file.side_effect = _make_download_file_gen_output_success()
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            prefetch_count=2,
        )
        pf.set_indices([0, 5, 10, 15])

        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            pf.get_gen_output(0)
        time.sleep(0.5)

        # Should have downloaded: 0, 5, 10 (current + 2 prefetch)
        assert backend.download_file.call_count == 3
        pf.shutdown()

    def test_no_duplicate_downloads(self, tmp_path: Path, mock_backend) -> None:
        """Requesting the same gen_output twice should not trigger a second download."""
        backend, _ = mock_backend
        backend.download_file.side_effect = _make_download_file_gen_output_success()
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            prefetch_count=0,
        )
        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            pf.get_gen_output(5)
            pf.get_gen_output(5)

        assert backend.download_file.call_count == 1
        pf.shutdown()


# ---------------------------------------------------------------------------
# Tests: GenOutputPrefetcher cleanup
# ---------------------------------------------------------------------------


class TestGenOutputCleanup:
    def test_cleanup_removes_previous_gen_output(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_file.side_effect = _make_download_file_gen_output_success()
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            cleanup_after_eval=True,
            prefetch_count=0,
        )
        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            pf.get_gen_output(0)
        ans0_dir = os.path.join(str(tmp_path), "ans_0")
        assert os.path.isdir(ans0_dir)

        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            pf.get_gen_output(5)
        assert not os.path.exists(ans0_dir)
        pf.shutdown()

    def test_no_cleanup_when_disabled(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_file.side_effect = _make_download_file_gen_output_success()
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            cleanup_after_eval=False,
            prefetch_count=0,
        )
        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            pf.get_gen_output(0)
            pf.get_gen_output(5)
        ans0_dir = os.path.join(str(tmp_path), "ans_0")
        assert os.path.isdir(ans0_dir)
        pf.shutdown()

    def test_no_cleanup_on_first_call(self, tmp_path: Path, mock_backend) -> None:
        backend, _ = mock_backend
        backend.download_file.side_effect = _make_download_file_gen_output_success()
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
            cleanup_after_eval=True,
            prefetch_count=0,
        )
        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            pf.get_gen_output(0)
        ans0_dir = os.path.join(str(tmp_path), "ans_0")
        assert os.path.isdir(ans0_dir)
        pf.shutdown()


# ---------------------------------------------------------------------------
# Tests: GenOutputPrefetcher validation
# ---------------------------------------------------------------------------


class TestGenOutputValidation:
    def test_required_keys_constant(self) -> None:
        """Verify the expected required keys are defined."""
        assert _REQUIRED_GEN_OUTPUT_KEYS == {
            "parsed_ok", "question_id", "question_text",
            "choices", "ground_truth", "doc_id",
        }

    def test_valid_gen_output_passes_validation(self, tmp_path: Path, mock_backend) -> None:
        """A gen_output.pt with all required keys passes validation."""
        backend, _ = mock_backend
        _save_mock_gen_output(str(tmp_path), 0)
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            result = pf.get_gen_output(0)
        for key in _REQUIRED_GEN_OUTPUT_KEYS:
            assert key in result.non_tensor_batch
        pf.shutdown()

    def test_missing_single_key_raises(self, tmp_path: Path, mock_backend) -> None:
        """Missing even one required key should raise RuntimeError."""
        backend, _ = mock_backend
        _save_mock_gen_output(str(tmp_path), 0, exclude_keys={"doc_id"})
        pf = GenOutputPrefetcher(
            remote_base_path="s3://bucket/exp",
            local_cache_dir=str(tmp_path),
        )
        with patch("verl.DataProto.load_from_disk", side_effect=lambda path: pickle.load(open(path, "rb"))):
            with pytest.raises(RuntimeError, match="doc_id"):
                pf.get_gen_output(0)
        pf.shutdown()
