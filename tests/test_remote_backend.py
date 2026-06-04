"""Comprehensive tests for remote backend abstraction and factory (US-004).

Covers:
- Factory URI routing for s3://, r2://, hf://datasets/...
- Error handling for non-dataset HF URIs, malformed URIs, empty/None
- HFDatasetRemoteBackend with mocked huggingface_hub APIs
- R2RemoteBackend with mocked boto3/rclone
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest

from verl_inf_evolve.storage.remote_backend import (
    RemoteBackend,
    create_remote_backend,
)
from verl_inf_evolve.storage.r2_remote_backend import R2RemoteBackend
from verl_inf_evolve.storage.hf_remote_backend import HFDatasetRemoteBackend


# ===================================================================
# Factory URI routing
# ===================================================================


class TestFactoryRouting:
    """Verify factory returns the correct backend type with correct params."""

    def test_s3_returns_r2_backend(self):
        backend = create_remote_backend("s3://bucket/prefix")
        assert isinstance(backend, R2RemoteBackend)
        assert backend.bucket == "bucket"
        assert backend.prefix == "prefix"

    def test_r2_returns_r2_backend(self):
        backend = create_remote_backend("r2://bucket/prefix")
        assert isinstance(backend, R2RemoteBackend)
        assert backend.bucket == "bucket"
        assert backend.prefix == "prefix"

    def test_hf_datasets_returns_hf_backend(self):
        backend = create_remote_backend("hf://datasets/org/repo/some/prefix")
        assert isinstance(backend, HFDatasetRemoteBackend)
        assert backend.repo_id == "org/repo"
        assert backend.prefix == "some/prefix"

    def test_hf_datasets_no_prefix(self):
        backend = create_remote_backend("hf://datasets/org/repo")
        assert isinstance(backend, HFDatasetRemoteBackend)
        assert backend.repo_id == "org/repo"
        assert backend.prefix == ""


class TestFactoryErrors:
    """Verify factory raises on bad URIs."""

    def test_hf_models_raises(self):
        with pytest.raises(ValueError, match="Only HF dataset repos"):
            create_remote_backend("hf://models/org/repo")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            create_remote_backend("")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            create_remote_backend(None)  # type: ignore[arg-type]

    def test_malformed_uri_raises(self):
        with pytest.raises(ValueError):
            create_remote_backend("hf://datasets/")

    def test_malformed_hf_no_repo_raises(self):
        with pytest.raises(ValueError, match="Malformed HF dataset URI"):
            create_remote_backend("hf://datasets/onlyns")

    def test_unsupported_scheme_raises(self):
        with pytest.raises(ValueError, match="Unsupported remote URI scheme"):
            create_remote_backend("gs://bucket/path")

    def test_whitespace_raises(self):
        with pytest.raises(ValueError):
            create_remote_backend("   ")


# ===================================================================
# HFDatasetRemoteBackend — mocked
# ===================================================================


class TestHFDatasetRemoteBackend:
    """Tests with huggingface_hub APIs mocked."""

    @pytest.fixture()
    def backend(self):
        with (
            patch("verl_inf_evolve.storage.hf_remote_backend.HfApi") as MockApi,
            patch("verl_inf_evolve.storage.hf_remote_backend.configure_hf_transfer_limits"),
        ):
            mock_api = MockApi.return_value
            b = HFDatasetRemoteBackend(
                repo_id="org/repo", prefix="exp1", revision="main", token="tok"
            )
            b._api = mock_api
            yield b, mock_api

    # -- exists --

    def test_exists_true(self, backend):
        b, api = backend
        api.get_paths_info.return_value = [MagicMock()]
        assert b.exists("file.txt") is True
        api.get_paths_info.assert_called_once_with(
            repo_id="org/repo",
            paths=["exp1/file.txt"],
            repo_type="dataset",
            revision="main",
        )

    def test_exists_false(self, backend):
        b, api = backend
        api.get_paths_info.return_value = []
        assert b.exists("missing.txt") is False

    def test_auto_create_repo_creates_public_dataset_repo(self):
        with patch("verl_inf_evolve.storage.hf_remote_backend.HfApi") as MockApi:
            mock_api = MockApi.return_value
            backend = HFDatasetRemoteBackend(
                repo_id="org/repo",
                auto_create_repo=True,
                token="tok",
            )
            assert backend.repo_id == "org/repo"
            mock_api.create_repo.assert_called_once_with(
                repo_id="org/repo",
                repo_type="dataset",
                private=False,
                exist_ok=True,
            )

    def test_auto_create_repo_failure_raises(self):
        with patch("verl_inf_evolve.storage.hf_remote_backend.HfApi") as MockApi:
            mock_api = MockApi.return_value
            mock_api.create_repo.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError, match="Failed to ensure public HF dataset repo exists"):
                HFDatasetRemoteBackend(
                    repo_id="org/repo",
                    auto_create_repo=True,
                    token="tok",
                )

    # -- upload_bytes --

    def test_upload_bytes(self, backend):
        b, api = backend
        result = b.upload_bytes(b"hello", "data.bin")
        assert result is True
        api.upload_file.assert_called_once()
        call_kwargs = api.upload_file.call_args
        assert call_kwargs.kwargs["path_in_repo"] == "exp1/data.bin"
        assert call_kwargs.kwargs["repo_id"] == "org/repo"
        assert call_kwargs.kwargs["repo_type"] == "dataset"

    # -- upload_file --

    def test_upload_file(self, backend):
        b, api = backend
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"content")
            tmp_path = f.name
        try:
            result = b.upload_file(tmp_path, "sub/file.txt")
            assert result is True
            api.upload_file.assert_called_once()
            assert api.upload_file.call_args.kwargs["path_in_repo"] == "exp1/sub/file.txt"
        finally:
            os.unlink(tmp_path)

    # -- upload_dir --

    def test_upload_dir(self, backend):
        b, api = backend
        with tempfile.TemporaryDirectory() as tmpdir:
            result = b.upload_dir(tmpdir, "checkpoint")
            assert result is True
            api.upload_folder.assert_called_once()
            assert api.upload_folder.call_args.kwargs["path_in_repo"] == "exp1/checkpoint"

    # -- download_file --

    @patch("verl_inf_evolve.storage.hf_remote_backend.hf_hub_download")
    def test_download_file(self, mock_download, backend):
        b, api = backend
        # hf_hub_download returns the local path of the downloaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as src:
            src.write(b"file contents")
            src_path = src.name
        mock_download.return_value = src_path
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, "out.txt")
                result = b.download_file("model.bin", local_path)
                assert result is True
                assert os.path.exists(local_path)
                with open(local_path, "rb") as f:
                    assert f.read() == b"file contents"
        finally:
            if os.path.exists(src_path):
                os.unlink(src_path)

    # -- download_dir --

    @patch("verl_inf_evolve.storage.hf_remote_backend.snapshot_download")
    def test_download_dir(self, mock_snapshot, backend):
        b, api = backend

        def _fake_snapshot(*, local_dir, **kwargs):
            # Simulate snapshot_download writing into the staging dir
            sub = os.path.join(local_dir, "exp1", "subdir")
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(sub, "file.txt"), "w") as f:
                f.write("data")
            return local_dir

        mock_snapshot.side_effect = _fake_snapshot

        with tempfile.TemporaryDirectory() as local_dir:
            result = b.download_dir("subdir", local_dir)
            assert result is True
            # File should be moved into local_dir
            assert os.path.isfile(os.path.join(local_dir, "file.txt"))
            # Staging dir should be cleaned up
            staging_dir = local_dir.rstrip("/") + "._hf_staging"
            assert not os.path.exists(staging_dir)

    @patch("verl_inf_evolve.storage.hf_remote_backend.snapshot_download")
    def test_download_dir_passes_snapshot_max_workers(self, mock_snapshot):
        with (
            patch("verl_inf_evolve.storage.hf_remote_backend.HfApi") as MockApi,
            patch("verl_inf_evolve.storage.hf_remote_backend.configure_hf_transfer_limits"),
        ):
            backend = HFDatasetRemoteBackend(
                repo_id="org/repo",
                prefix="exp1",
                revision="main",
                token="tok",
                snapshot_max_workers=2,
            )
            backend._api = MockApi.return_value

            def _fake_snapshot(*, local_dir, **kwargs):
                sub = os.path.join(local_dir, "exp1", "subdir")
                os.makedirs(sub, exist_ok=True)
                with open(os.path.join(sub, "file.txt"), "w") as f:
                    f.write("data")
                return local_dir

            mock_snapshot.side_effect = _fake_snapshot

            with tempfile.TemporaryDirectory() as local_dir:
                assert backend.download_dir("subdir", local_dir) is True

            assert mock_snapshot.call_args.kwargs["max_workers"] == 2

    # -- download_bytes --

    @patch("verl_inf_evolve.storage.hf_remote_backend.hf_hub_download")
    def test_download_bytes(self, mock_download, backend):
        b, api = backend
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as src:
            src.write(b"binary data")
            src_path = src.name
        mock_download.return_value = src_path
        try:
            data = b.download_bytes("data.bin")
            assert data == b"binary data"
        finally:
            if os.path.exists(src_path):
                os.unlink(src_path)

    @patch("verl_inf_evolve.storage.hf_remote_backend.hf_hub_download")
    def test_download_bytes_not_found(self, mock_download, backend):
        b, api = backend
        from huggingface_hub.utils import EntryNotFoundError

        mock_download.side_effect = EntryNotFoundError("not found")
        with pytest.raises(FileNotFoundError):
            b.download_bytes("no_such.bin")

    # -- list_immediate_children --

    def test_list_immediate_children(self, backend):
        b, api = backend
        # Simulate tree items with rpath
        item1 = MagicMock()
        item1.path = "exp1/child_dir"
        item2 = MagicMock()
        item2.path = "exp1/file.txt"
        api.list_repo_tree.return_value = [item1, item2]

        children = b.list_immediate_children("")
        assert sorted(children) == ["child_dir", "file.txt"]
        api.list_repo_tree.assert_called_once_with(
            repo_id="org/repo",
            path_in_repo="exp1",
            repo_type="dataset",
            revision="main",
            recursive=False,
        )

    def test_list_immediate_children_with_key(self, backend):
        b, api = backend
        item = MagicMock()
        item.path = "exp1/checkpoints/step_100"
        api.list_repo_tree.return_value = [item]

        children = b.list_immediate_children("checkpoints")
        assert children == ["step_100"]

    # -- delete_prefix --

    def test_delete_prefix_no_excludes(self, backend):
        b, api = backend
        # Simulate files under prefix
        f1 = MagicMock()
        f1.path = "exp1/old/file1.pt"
        f1.size = 100
        f2 = MagicMock()
        f2.path = "exp1/old/file2.pt"
        f2.size = 200
        api.list_repo_tree.return_value = [f1, f2]

        result = b.delete_prefix("old")
        assert result is True
        api.create_commit.assert_called_once()
        ops = api.create_commit.call_args.kwargs["operations"]
        deleted_paths = {op.path_in_repo for op in ops}
        assert deleted_paths == {"exp1/old/file1.pt", "exp1/old/file2.pt"}

    def test_delete_prefix_with_excludes(self, backend):
        b, api = backend
        f1 = MagicMock()
        f1.path = "exp1/old/model.pt"
        f1.size = 100
        f2 = MagicMock()
        f2.path = "exp1/old/huggingface/model.safetensors"
        f2.size = 200
        api.list_repo_tree.return_value = [f1, f2]

        result = b.delete_prefix("old", exclude_patterns=["huggingface/*"])
        assert result is True
        api.create_commit.assert_called_once()
        ops = api.create_commit.call_args.kwargs["operations"]
        deleted_paths = {op.path_in_repo for op in ops}
        # huggingface/model.safetensors should be excluded
        assert deleted_paths == {"exp1/old/model.pt"}

    def test_delete_prefix_empty_list(self, backend):
        b, api = backend
        api.list_repo_tree.return_value = []
        result = b.delete_prefix("empty_dir")
        assert result is True
        api.create_commit.assert_not_called()


# ===================================================================
# R2RemoteBackend — mocked
# ===================================================================


class TestR2RemoteBackend:
    """Tests with boto3 and rclone mocked."""

    @pytest.fixture()
    def backend(self):
        b = R2RemoteBackend(bucket="my-bucket", prefix="exp")
        yield b

    # -- exists --

    @patch.object(R2RemoteBackend, "_get_client")
    def test_exists_true(self, mock_get_client, backend):
        client = MagicMock()
        mock_get_client.return_value = client
        client.head_object.return_value = {}
        assert backend.exists("file.txt") is True
        client.head_object.assert_called_once_with(Bucket="my-bucket", Key="exp/file.txt")

    @patch.object(R2RemoteBackend, "_get_client")
    def test_exists_false(self, mock_get_client, backend):
        client = MagicMock()
        mock_get_client.return_value = client
        client.head_object.side_effect = Exception("404")
        assert backend.exists("missing.txt") is False

    # -- download_bytes --

    @patch.object(R2RemoteBackend, "_get_client")
    def test_download_bytes(self, mock_get_client, backend):
        client = MagicMock()
        mock_get_client.return_value = client
        body = MagicMock()
        body.read.return_value = b"hello world"
        client.get_object.return_value = {"Body": body}
        data = backend.download_bytes("data.bin")
        assert data == b"hello world"
        client.get_object.assert_called_once_with(Bucket="my-bucket", Key="exp/data.bin")
        body.close.assert_called_once()

    @patch.object(R2RemoteBackend, "_get_client")
    def test_download_bytes_not_found(self, mock_get_client, backend):
        client = MagicMock()
        mock_get_client.return_value = client
        client.get_object.side_effect = Exception("NoSuchKey")
        with pytest.raises(FileNotFoundError):
            backend.download_bytes("no_such.bin")

    # -- list_immediate_children --

    @patch.object(R2RemoteBackend, "_get_client")
    def test_list_immediate_children(self, mock_get_client, backend):
        client = MagicMock()
        mock_get_client.return_value = client

        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "exp/global_step_10/"},
                    {"Prefix": "exp/global_step_20/"},
                ],
                "Contents": [
                    {"Key": "exp/metadata.json"},
                ],
            }
        ]
        children = backend.list_immediate_children("")
        assert sorted(children) == ["global_step_10", "global_step_20", "metadata.json"]

    # -- upload_bytes --

    @patch.object(R2RemoteBackend, "_get_client")
    def test_upload_bytes(self, mock_get_client, backend):
        client = MagicMock()
        mock_get_client.return_value = client
        result = backend.upload_bytes(b"data", "out.bin")
        assert result is True
        client.put_object.assert_called_once_with(
            Bucket="my-bucket", Key="exp/out.bin", Body=b"data"
        )

    @patch.object(R2RemoteBackend, "_get_client")
    def test_upload_bytes_failure(self, mock_get_client, backend):
        client = MagicMock()
        mock_get_client.return_value = client
        client.put_object.side_effect = Exception("upload failed")
        result = backend.upload_bytes(b"data", "out.bin")
        assert result is False

    # -- download_file --

    @patch("verl_inf_evolve.storage.r2_utils.rclone_copy")
    def test_download_file(self, mock_rclone, backend):
        mock_rclone.return_value = True
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "file.bin")
            result = backend.download_file("sub/file.bin", local_path)
            assert result is True
            mock_rclone.assert_called_once()

    # -- upload_dir --

    @patch("verl_inf_evolve.storage.r2_utils.rclone_copy")
    def test_upload_dir(self, mock_rclone, backend):
        mock_rclone.return_value = True
        result = backend.upload_dir("/local/dir", "checkpoint")
        assert result is True
        mock_rclone.assert_called_once()

    # -- delete_prefix --

    @patch("verl_inf_evolve.storage.r2_utils.rclone_delete")
    def test_delete_prefix(self, mock_rclone_del, backend):
        mock_rclone_del.return_value = True
        result = backend.delete_prefix("old_step")
        assert result is True
        mock_rclone_del.assert_called_once()

    @patch("verl_inf_evolve.storage.r2_utils.rclone_delete")
    def test_delete_prefix_with_excludes(self, mock_rclone_del, backend):
        mock_rclone_del.return_value = True
        result = backend.delete_prefix("old_step", exclude_patterns=["**/huggingface/**"])
        assert result is True
        mock_rclone_del.assert_called_once()
        assert mock_rclone_del.call_args.kwargs["exclude_patterns"] == ["**/huggingface/**"]


# ===================================================================
# Key construction helpers
# ===================================================================


class TestKeyConstruction:
    """Verify _full_key / _full_path combines prefix and key correctly."""

    def test_r2_full_key_with_prefix(self):
        b = R2RemoteBackend(bucket="b", prefix="pfx")
        assert b._full_key("sub/file.txt") == "pfx/sub/file.txt"

    def test_r2_full_key_empty_key(self):
        b = R2RemoteBackend(bucket="b", prefix="pfx")
        assert b._full_key("") == "pfx"

    def test_r2_full_key_no_prefix(self):
        b = R2RemoteBackend(bucket="b", prefix="")
        assert b._full_key("file.txt") == "file.txt"

    @patch("verl_inf_evolve.storage.hf_remote_backend.HfApi")
    def test_hf_full_path_with_prefix(self, _):
        b = HFDatasetRemoteBackend(repo_id="org/repo", prefix="pfx")
        assert b._full_path("sub/file.txt") == "pfx/sub/file.txt"

    @patch("verl_inf_evolve.storage.hf_remote_backend.HfApi")
    def test_hf_full_path_empty_key(self, _):
        b = HFDatasetRemoteBackend(repo_id="org/repo", prefix="pfx")
        assert b._full_path("") == "pfx"

    @patch("verl_inf_evolve.storage.hf_remote_backend.HfApi")
    def test_hf_full_path_no_prefix(self, _):
        b = HFDatasetRemoteBackend(repo_id="org/repo", prefix="")
        assert b._full_path("file.txt") == "file.txt"
