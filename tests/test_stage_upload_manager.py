"""Tests for StageUploadManager backend integration.

Verifies that StageUploadManager delegates all remote I/O to a RemoteBackend
instance, and that the public API (method signatures) is unchanged.
"""

from __future__ import annotations

import json
import pickle
from unittest.mock import MagicMock, patch

import pytest

from verl_inf_evolve.storage.stage_upload_manager import StageUploadManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(uri: str = "s3://bucket/prefix", **kwargs) -> StageUploadManager:
    """Create a StageUploadManager with a mocked backend."""
    with patch(
        "verl_inf_evolve.storage.remote_backend.create_remote_backend"
    ) as mock_factory:
        mock_backend = MagicMock()
        mock_factory.return_value = mock_backend
        mgr = StageUploadManager(uri, **kwargs)
    return mgr


# ---------------------------------------------------------------------------
# Construction & Properties
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_backend_from_uri(self):
        with patch(
            "verl_inf_evolve.storage.remote_backend.create_remote_backend"
        ) as mock_factory:
            mock_factory.return_value = MagicMock()
            mgr = StageUploadManager("s3://mybucket/myprefix")
            mock_factory.assert_called_once_with("s3://mybucket/myprefix")
            assert mgr.backend is not None

    def test_none_uri_no_backend(self):
        mgr = StageUploadManager(None)
        assert mgr.backend is None
        assert not mgr.remote_configured

    def test_remote_configured_true(self):
        mgr = _make_manager("s3://bucket/prefix")
        assert mgr.remote_configured is True

    def test_upload_enabled_true(self):
        mgr = _make_manager("s3://bucket/prefix")
        assert mgr.upload_enabled is True

    def test_upload_disabled_flag(self):
        mgr = _make_manager("s3://bucket/prefix", disable_upload=True)
        assert mgr.remote_configured is True
        assert mgr.upload_enabled is False

    def test_backend_property(self):
        mgr = _make_manager("s3://bucket/prefix")
        assert mgr.backend is mgr._backend

    def test_hf_uri_creates_backend(self):
        with patch(
            "verl_inf_evolve.storage.remote_backend.create_remote_backend"
        ) as mock_factory:
            mock_factory.return_value = MagicMock()
            mgr = StageUploadManager("hf://datasets/org/repo/prefix")
            mock_factory.assert_called_once_with(
                "hf://datasets/org/repo/prefix",
                auto_create_repo=True,
            )
            assert mgr.remote_configured is True

    def test_hf_uri_skips_auto_create_when_upload_disabled(self):
        with patch(
            "verl_inf_evolve.storage.remote_backend.create_remote_backend"
        ) as mock_factory:
            mock_factory.return_value = MagicMock()
            mgr = StageUploadManager(
                "hf://datasets/org/repo/prefix",
                disable_upload=True,
            )
            mock_factory.assert_called_once_with(
                "hf://datasets/org/repo/prefix",
                auto_create_repo=False,
            )
            assert mgr.remote_configured is True


# ---------------------------------------------------------------------------
# Synchronous downloads
# ---------------------------------------------------------------------------


class TestDownloads:
    def test_remote_exists_delegates(self):
        mgr = _make_manager()
        mgr._backend.exists.return_value = True
        assert mgr.remote_exists("some/key") is True
        mgr._backend.exists.assert_called_once_with("some/key")

    def test_remote_exists_false(self):
        mgr = _make_manager()
        mgr._backend.exists.return_value = False
        assert mgr.remote_exists("missing/key") is False

    def test_remote_exists_no_backend(self):
        mgr = StageUploadManager(None)
        assert mgr.remote_exists("any/key") is False

    def test_download_to_memory_json(self):
        mgr = _make_manager()
        payload = {"hello": "world"}
        mgr._backend.download_bytes.return_value = json.dumps(payload).encode()
        result = mgr.download_to_memory("data.json", "json")
        assert result == payload
        mgr._backend.download_bytes.assert_called_once_with("data.json")

    def test_download_to_memory_dataproto(self):
        mgr = _make_manager()
        obj = {"key": [1, 2, 3]}
        mgr._backend.download_bytes.return_value = pickle.dumps(obj)
        result = mgr.download_to_memory("data.pt", "dataproto")
        assert result == obj

    def test_download_to_memory_not_found(self):
        mgr = _make_manager()
        mgr._backend.download_bytes.side_effect = FileNotFoundError("nope")
        result = mgr.download_to_memory("missing.pt", "dataproto")
        assert result is None

    def test_download_to_memory_no_backend(self):
        mgr = StageUploadManager(None)
        assert mgr.download_to_memory("key", "json") is None

    def test_download_dir_to_local(self, tmp_path):
        mgr = _make_manager()
        mgr._backend.download_dir.return_value = True
        local = str(tmp_path / "output")
        assert mgr.download_dir_to_local("ckpt/step_100", local) is True
        mgr._backend.download_dir.assert_called_once_with("ckpt/step_100", local)

    def test_download_dir_no_backend(self, tmp_path):
        mgr = StageUploadManager(None)
        assert mgr.download_dir_to_local("key", str(tmp_path)) is False

    def test_download_file_to_local(self, tmp_path):
        mgr = _make_manager()
        mgr._backend.download_file.return_value = True
        local = str(tmp_path / "state.json")
        assert mgr.download_file_to_local("ans/state.json", local) is True
        mgr._backend.download_file.assert_called_once_with("ans/state.json", local)

    def test_download_file_no_backend(self, tmp_path):
        mgr = StageUploadManager(None)
        assert mgr.download_file_to_local("key", str(tmp_path / "f")) is False


# ---------------------------------------------------------------------------
# Synchronous uploads
# ---------------------------------------------------------------------------


class TestSyncUploads:
    def test_upload_bytes_sync(self):
        mgr = _make_manager()
        mgr._backend.upload_bytes.return_value = True
        assert mgr.upload_bytes_sync(b"hello", "test/key") is True
        mgr._backend.upload_bytes.assert_called_once_with(b"hello", "test/key")

    def test_upload_bytes_sync_disabled(self):
        mgr = _make_manager(disable_upload=True)
        assert mgr.upload_bytes_sync(b"hello", "test/key") is False

    def test_upload_file_sync(self, tmp_path):
        mgr = _make_manager()
        mgr._backend.upload_bytes.return_value = True
        fpath = tmp_path / "test.txt"
        fpath.write_bytes(b"content")
        assert mgr.upload_file_sync(str(fpath), "remote/test.txt") is True
        mgr._backend.upload_bytes.assert_called_once_with(b"content", "remote/test.txt")

    def test_upload_file_sync_no_backend(self, tmp_path):
        mgr = StageUploadManager(None)
        assert mgr.upload_file_sync(str(tmp_path / "f"), "key") is False


# ---------------------------------------------------------------------------
# Background upload thread — internal methods
# ---------------------------------------------------------------------------


class TestInternalUpload:
    """Test the internal _upload_bytes, _upload_dir, _delete_remote methods."""

    def test_upload_bytes_delegates(self):
        from verl_inf_evolve.storage.stage_upload_manager import UploadTask

        mgr = _make_manager()
        mgr._backend.upload_bytes.return_value = True
        task = UploadTask(
            task_id="t1", remote_key="data.pt", bytes_buf=b"payload", bytes_size=7
        )
        mgr._upload_bytes(task)
        mgr._backend.upload_bytes.assert_called_once_with(b"payload", "data.pt")
        assert task.bytes_buf is None  # freed after upload

    def test_upload_bytes_failure_raises(self):
        from verl_inf_evolve.storage.stage_upload_manager import UploadTask

        mgr = _make_manager()
        mgr._backend.upload_bytes.return_value = False
        task = UploadTask(
            task_id="t1", remote_key="data.pt", bytes_buf=b"payload", bytes_size=7
        )
        with pytest.raises(RuntimeError, match="upload_bytes failed"):
            mgr._upload_bytes(task)

    def test_upload_dir_delegates(self):
        from verl_inf_evolve.storage.stage_upload_manager import UploadTask

        mgr = _make_manager()
        mgr._backend.upload_dir.return_value = True
        task = UploadTask(
            task_id="t1", remote_key="ckpt/step_100", local_path="/tmp/ckpt"
        )
        mgr._upload_dir(task)
        mgr._backend.upload_dir.assert_called_once_with("/tmp/ckpt", "ckpt/step_100")

    def test_upload_dir_failure_raises(self):
        from verl_inf_evolve.storage.stage_upload_manager import UploadTask

        mgr = _make_manager()
        mgr._backend.upload_dir.return_value = False
        task = UploadTask(
            task_id="t1", remote_key="ckpt/step_100", local_path="/tmp/ckpt"
        )
        with pytest.raises(RuntimeError, match="upload_dir failed"):
            mgr._upload_dir(task)

    def test_delete_remote_delegates(self):
        from verl_inf_evolve.storage.stage_upload_manager import UploadTask

        mgr = _make_manager()
        mgr._backend.delete_prefix.return_value = True
        task = UploadTask(
            task_id="t1",
            remote_key="ckpt/step_50",
            is_delete=True,
            remote_delete_exclude=["**/huggingface/**"],
        )
        mgr._delete_remote(task)
        mgr._backend.delete_prefix.assert_called_once_with(
            "ckpt/step_50", exclude_patterns=["**/huggingface/**"]
        )


class TestTaskDependencies:
    def test_dependent_file_upload_is_skipped_when_dependency_fails(self, tmp_path):
        mgr = _make_manager()
        mgr._backend.upload_dir.return_value = False
        mgr._backend.upload_bytes.return_value = True

        checkpoint_dir = tmp_path / "global_step_20"
        checkpoint_dir.mkdir()
        marker_path = tmp_path / "latest_checkpointed_iteration.txt"
        marker_path.write_text("20", encoding="utf-8")

        mgr.start()
        try:
            ckpt_task = mgr.submit_checkpoint_upload(
                local_dir=str(checkpoint_dir),
                remote_key="global_step_20",
            )
            marker_task = mgr.submit_file_upload(
                local_path=str(marker_path),
                remote_key="latest_checkpointed_iteration.txt",
                depends_on=ckpt_task,
            )

            assert mgr.wait_for_task(ckpt_task, timeout=5.0) is False
            assert mgr.wait_for_task(marker_task, timeout=5.0) is False
        finally:
            mgr.shutdown(timeout=5.0)

        mgr._backend.upload_dir.assert_called_once_with(
            str(checkpoint_dir),
            "global_step_20",
        )
        mgr._backend.upload_bytes.assert_not_called()

    def test_checkpoint_upload_failure_cleans_local_dir_and_invokes_callback(self, tmp_path):
        callback = MagicMock()
        mgr = _make_manager(checkpoint_failure_callback=callback)
        mgr._backend.upload_dir.return_value = False

        checkpoint_dir = tmp_path / "global_step_7"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "shard.pt").write_bytes(b"x")

        mgr.start()
        try:
            task_id = mgr.submit_checkpoint_upload(
                local_dir=str(checkpoint_dir),
                remote_key="global_step_7",
                cleanup_after=True,
            )
            assert mgr.wait_for_task(task_id, timeout=5.0) is False
        finally:
            mgr.shutdown(timeout=5.0)

        assert not checkpoint_dir.exists()
        callback.assert_called_once()
        task, error = callback.call_args[0]
        assert task.remote_key == "global_step_7"
        assert isinstance(error, Exception)

    def test_delete_remote_no_excludes(self):
        from verl_inf_evolve.storage.stage_upload_manager import UploadTask

        mgr = _make_manager()
        mgr._backend.delete_prefix.return_value = True
        task = UploadTask(
            task_id="t1",
            remote_key="ckpt/step_50",
            is_delete=True,
            remote_delete_exclude=None,
        )
        mgr._delete_remote(task)
        mgr._backend.delete_prefix.assert_called_once_with(
            "ckpt/step_50", exclude_patterns=None
        )


# ---------------------------------------------------------------------------
# Public API signatures unchanged
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify that the public method signatures still exist."""

    def test_has_expected_methods(self):
        mgr = StageUploadManager(None)
        expected = [
            "start",
            "shutdown",
            "submit_memory_upload",
            "submit_dir_upload",
            "submit_file_upload",
            "submit_checkpoint_upload",
            "submit_remote_delete",
            "upload_file_sync",
            "upload_bytes_sync",
            "download_to_memory",
            "download_dir_to_local",
            "download_file_to_local",
            "remote_exists",
            "wait_for_task",
            "wait_all_pending",
            "remote_configured",
            "upload_enabled",
        ]
        for method_name in expected:
            assert hasattr(mgr, method_name), f"Missing: {method_name}"
