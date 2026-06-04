"""Tests for StorageResolver — backend-neutral local/remote file resolution."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from verl_inf_evolve.storage.storage_resolver import StorageResolver


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def tmp_dir(tmp_path):
    """Return a temporary directory path as string."""
    return str(tmp_path)


@pytest.fixture()
def mock_upload_manager():
    """Return a mocked StageUploadManager with remote_configured=True."""
    mgr = MagicMock()
    mgr.remote_configured = True
    return mgr


@pytest.fixture()
def resolver(tmp_dir, mock_upload_manager):
    """StorageResolver with local_first (default) and mocked upload manager."""
    return StorageResolver(
        local_base=tmp_dir,
        upload_manager=mock_upload_manager,
        remote_prefix="ans_0",
    )


@pytest.fixture()
def resolver_remote_first(tmp_dir, mock_upload_manager):
    """StorageResolver with remote_first ordering."""
    return StorageResolver(
        local_base=tmp_dir,
        upload_manager=mock_upload_manager,
        remote_prefix="ans_0",
        resolve_order="remote_first",
    )


@pytest.fixture()
def resolver_local_only(tmp_dir):
    """StorageResolver with no upload manager (local-only mode)."""
    return StorageResolver(local_base=tmp_dir)


# ------------------------------------------------------------------
# Remote key construction
# ------------------------------------------------------------------

class TestRemoteKey:
    def test_with_prefix(self, resolver):
        assert resolver._remote_key("state.json") == "ans_0/state.json"

    def test_without_prefix(self, tmp_dir, mock_upload_manager):
        r = StorageResolver(tmp_dir, mock_upload_manager, remote_prefix="")
        assert r._remote_key("state.json") == "state.json"

    def test_nested_path(self, resolver):
        assert resolver._remote_key("sub/dir/file.pt") == "ans_0/sub/dir/file.pt"


# ------------------------------------------------------------------
# has_remote
# ------------------------------------------------------------------

class TestHasRemote:
    def test_with_manager(self, resolver):
        assert resolver._has_remote() is True

    def test_without_manager(self, resolver_local_only):
        assert resolver_local_only._has_remote() is False

    def test_manager_not_configured(self, tmp_dir):
        mgr = MagicMock()
        mgr.remote_configured = False
        r = StorageResolver(tmp_dir, mgr)
        assert r._has_remote() is False


# ------------------------------------------------------------------
# resolve_file — local_first
# ------------------------------------------------------------------

class TestResolveFileLocalFirst:
    def test_local_exists(self, resolver, tmp_dir):
        """File exists locally → return local path, no remote call."""
        path = os.path.join(tmp_dir, "state.json")
        with open(path, "w") as f:
            f.write("{}")
        result = resolver.resolve_file("state.json")
        assert result == path
        resolver._upload_manager.download_file_to_local.assert_not_called()

    def test_local_miss_remote_hit(self, resolver, tmp_dir):
        """File not local → download from remote."""
        resolver._upload_manager.download_file_to_local.return_value = True
        result = resolver.resolve_file("state.json")
        assert result == os.path.join(tmp_dir, "state.json")
        resolver._upload_manager.download_file_to_local.assert_called_once_with(
            "ans_0/state.json", os.path.join(tmp_dir, "state.json")
        )

    def test_local_miss_remote_miss(self, resolver):
        """File not found anywhere → return None."""
        resolver._upload_manager.download_file_to_local.return_value = False
        result = resolver.resolve_file("missing.json")
        assert result is None

    def test_local_only_mode_file_exists(self, resolver_local_only, tmp_dir):
        """Local-only mode with existing file."""
        path = os.path.join(tmp_dir, "data.json")
        with open(path, "w") as f:
            f.write("{}")
        result = resolver_local_only.resolve_file("data.json")
        assert result == path

    def test_local_only_mode_file_missing(self, resolver_local_only):
        """Local-only mode, file missing → None."""
        result = resolver_local_only.resolve_file("missing.json")
        assert result is None


# ------------------------------------------------------------------
# resolve_file — remote_first
# ------------------------------------------------------------------

class TestResolveFileRemoteFirst:
    def test_remote_hit(self, resolver_remote_first, tmp_dir):
        """remote_first: download succeeds → return local path."""
        resolver_remote_first._upload_manager.download_file_to_local.return_value = True
        result = resolver_remote_first.resolve_file("state.json")
        assert result == os.path.join(tmp_dir, "state.json")

    def test_remote_miss_local_hit(self, resolver_remote_first, tmp_dir):
        """remote_first: remote fails but local exists → return local path."""
        resolver_remote_first._upload_manager.download_file_to_local.return_value = False
        path = os.path.join(tmp_dir, "state.json")
        with open(path, "w") as f:
            f.write("{}")
        result = resolver_remote_first.resolve_file("state.json")
        assert result == path

    def test_both_miss(self, resolver_remote_first):
        """remote_first: both miss → None."""
        resolver_remote_first._upload_manager.download_file_to_local.return_value = False
        result = resolver_remote_first.resolve_file("missing.json")
        assert result is None


# ------------------------------------------------------------------
# resolve_dir
# ------------------------------------------------------------------

class TestResolveDir:
    def test_local_first_dir_exists(self, resolver, tmp_dir):
        """Directory exists locally → return it, no remote call."""
        dir_path = os.path.join(tmp_dir, "generator")
        os.makedirs(dir_path)
        result = resolver.resolve_dir("generator")
        assert result == dir_path
        resolver._upload_manager.download_dir_to_local.assert_not_called()

    def test_local_first_remote_download(self, resolver, tmp_dir):
        """Directory not local → download from remote."""
        resolver._upload_manager.download_dir_to_local.return_value = True
        result = resolver.resolve_dir("generator")
        assert result == os.path.join(tmp_dir, "generator")
        resolver._upload_manager.download_dir_to_local.assert_called_once_with(
            "ans_0/generator", os.path.join(tmp_dir, "generator")
        )

    def test_local_first_both_miss(self, resolver):
        resolver._upload_manager.download_dir_to_local.return_value = False
        result = resolver.resolve_dir("generator")
        assert result is None

    def test_remote_first_dir(self, resolver_remote_first, tmp_dir):
        resolver_remote_first._upload_manager.download_dir_to_local.return_value = True
        result = resolver_remote_first.resolve_dir("generator")
        assert result == os.path.join(tmp_dir, "generator")

    def test_custom_local_path(self, resolver, tmp_dir):
        """resolve_dir with explicit local_path override."""
        custom = os.path.join(tmp_dir, "custom_dest")
        os.makedirs(custom)
        result = resolver.resolve_dir("generator", local_path=custom)
        assert result == custom


# ------------------------------------------------------------------
# resolve_json
# ------------------------------------------------------------------

class TestResolveJson:
    def test_to_memory_local(self, resolver, tmp_dir):
        """JSON file exists locally → parse and return."""
        path = os.path.join(tmp_dir, "state.json")
        data = {"step": 42, "status": "ok"}
        with open(path, "w") as f:
            json.dump(data, f)
        result = resolver.resolve_json("state.json")
        assert result == data

    def test_to_memory_remote(self, resolver):
        """JSON not local → download from remote into memory."""
        expected = {"step": 100}
        resolver._upload_manager.download_to_memory.return_value = expected
        result = resolver.resolve_json("state.json")
        assert result == expected
        resolver._upload_manager.download_to_memory.assert_called_once_with(
            "ans_0/state.json", "json"
        )

    def test_to_memory_none(self, resolver):
        """JSON not found → None."""
        resolver._upload_manager.download_to_memory.return_value = None
        result = resolver.resolve_json("missing.json")
        assert result is None

    def test_to_disk(self, resolver, tmp_dir):
        """to_memory=False → resolve_file behavior."""
        path = os.path.join(tmp_dir, "state.json")
        with open(path, "w") as f:
            f.write("{}")
        result = resolver.resolve_json("state.json", to_memory=False)
        assert result == path


# ------------------------------------------------------------------
# resolve_dataproto
# ------------------------------------------------------------------

class TestResolveDataproto:
    def test_to_memory_remote(self, resolver):
        """DataProto from remote → deserialized object."""
        fake_dp = MagicMock()
        resolver._upload_manager.download_to_memory.return_value = fake_dp
        result = resolver.resolve_dataproto("gen_output.pt")
        assert result is fake_dp
        resolver._upload_manager.download_to_memory.assert_called_once_with(
            "ans_0/gen_output.pt", "dataproto"
        )

    def test_to_memory_not_found(self, resolver):
        resolver._upload_manager.download_to_memory.return_value = None
        result = resolver.resolve_dataproto("gen_output.pt")
        assert result is None


# ------------------------------------------------------------------
# Works with both R2 and HF URIs (backend-agnostic)
# ------------------------------------------------------------------

class TestBackendAgnostic:
    """Verify StorageResolver works regardless of backend type.

    StorageResolver delegates to StageUploadManager, which delegates to
    RemoteBackend. These tests confirm the interface contract is maintained.
    """

    def test_works_with_any_backend(self, tmp_dir):
        """StorageResolver doesn't care about backend type — only uses
        StageUploadManager's public API."""
        mgr = MagicMock()
        mgr.remote_configured = True
        mgr.download_file_to_local.return_value = True

        resolver = StorageResolver(tmp_dir, mgr, remote_prefix="prefix")
        result = resolver.resolve_file("test.json")
        assert result is not None
        mgr.download_file_to_local.assert_called_once()

    def test_no_r2_specific_imports(self):
        """Verify storage_resolver.py does not import R2/boto3/rclone modules."""
        import verl_inf_evolve.storage.storage_resolver as mod
        source = open(mod.__file__).read()
        assert "import boto3" not in source
        assert "from botocore" not in source
        assert "import rclone" not in source
        assert "from verl_inf_evolve.storage.r2_utils" not in source
