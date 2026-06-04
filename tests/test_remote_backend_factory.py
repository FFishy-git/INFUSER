"""Tests for RemoteBackend ABC and create_remote_backend factory (US-001)."""

from __future__ import annotations

import pytest

from verl_inf_evolve.storage.remote_backend import (
    RemoteBackend,
    create_remote_backend,
    _parse_hf_dataset_uri,
)
from verl_inf_evolve.storage.r2_remote_backend import R2RemoteBackend
from verl_inf_evolve.storage.hf_remote_backend import HFDatasetRemoteBackend


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------

class TestRemoteBackendABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            RemoteBackend()  # type: ignore[abstract]

    def test_abc_has_required_methods(self):
        expected = {
            "exists",
            "download_bytes",
            "download_file",
            "download_dir",
            "upload_bytes",
            "upload_file",
            "upload_dir",
            "delete_prefix",
            "list_immediate_children",
            "list_files_recursive",
        }
        abstract_methods = set(RemoteBackend.__abstractmethods__)
        assert expected == abstract_methods


# ---------------------------------------------------------------------------
# Factory: S3/R2 URIs
# ---------------------------------------------------------------------------

class TestFactoryR2:
    def test_s3_uri(self):
        backend = create_remote_backend("s3://my-bucket/some/prefix")
        assert isinstance(backend, R2RemoteBackend)
        assert backend.bucket == "my-bucket"
        assert backend.prefix == "some/prefix"

    def test_r2_uri(self):
        backend = create_remote_backend("r2://my-bucket/other")
        assert isinstance(backend, R2RemoteBackend)
        assert backend.bucket == "my-bucket"
        assert backend.prefix == "other"

    def test_s3_uri_no_prefix(self):
        backend = create_remote_backend("s3://bucket-only")
        assert isinstance(backend, R2RemoteBackend)
        assert backend.bucket == "bucket-only"
        assert backend.prefix == ""


# ---------------------------------------------------------------------------
# Factory: HF dataset URIs
# ---------------------------------------------------------------------------

class TestFactoryHF:
    def test_hf_dataset_uri(self):
        backend = create_remote_backend("hf://datasets/myorg/myrepo/some/prefix")
        assert isinstance(backend, HFDatasetRemoteBackend)
        assert backend.repo_id == "myorg/myrepo"
        assert backend.prefix == "some/prefix"

    def test_hf_dataset_uri_no_prefix(self):
        backend = create_remote_backend("hf://datasets/myorg/myrepo")
        assert isinstance(backend, HFDatasetRemoteBackend)
        assert backend.repo_id == "myorg/myrepo"
        assert backend.prefix == ""

    def test_hf_dataset_uri_trailing_slash(self):
        backend = create_remote_backend("hf://datasets/myorg/myrepo/pfx/")
        assert isinstance(backend, HFDatasetRemoteBackend)
        assert backend.prefix == "pfx"

    def test_hf_dataset_with_kwargs(self):
        backend = create_remote_backend(
            "hf://datasets/myorg/myrepo/pfx",
            revision="dev",
            token="secret",
        )
        assert isinstance(backend, HFDatasetRemoteBackend)
        assert backend.revision == "dev"
        assert backend._token == "secret"

    def test_hf_dataset_with_transfer_kwargs(self):
        backend = create_remote_backend(
            "hf://datasets/myorg/myrepo/pfx",
            upload_limit_mbps=12.5,
            download_limit_mbps=34.0,
            snapshot_max_workers=1,
        )
        assert isinstance(backend, HFDatasetRemoteBackend)
        assert backend._upload_limit_mbps == 12.5
        assert backend._download_limit_mbps == 34.0
        assert backend._snapshot_max_workers == 1


# ---------------------------------------------------------------------------
# Factory: Error cases
# ---------------------------------------------------------------------------

class TestFactoryErrors:
    def test_hf_models_rejected(self):
        with pytest.raises(ValueError, match="Only HF dataset repos"):
            create_remote_backend("hf://models/org/repo")

    def test_hf_spaces_rejected(self):
        with pytest.raises(ValueError, match="Only HF dataset repos"):
            create_remote_backend("hf://spaces/org/repo")

    def test_empty_string(self):
        with pytest.raises(ValueError):
            create_remote_backend("")

    def test_whitespace_string(self):
        with pytest.raises(ValueError):
            create_remote_backend("   ")

    def test_none(self):
        with pytest.raises(ValueError):
            create_remote_backend(None)  # type: ignore[arg-type]

    def test_unsupported_scheme(self):
        with pytest.raises(ValueError, match="Unsupported remote URI scheme"):
            create_remote_backend("gs://bucket/path")

    def test_malformed_hf_uri(self):
        with pytest.raises(ValueError, match="Malformed HF dataset URI"):
            create_remote_backend("hf://datasets/")

    def test_malformed_hf_uri_no_repo(self):
        with pytest.raises(ValueError, match="Malformed HF dataset URI"):
            create_remote_backend("hf://datasets/onlyns")


# ---------------------------------------------------------------------------
# Internal URI parser
# ---------------------------------------------------------------------------

class TestHFURIParsing:
    def test_full_uri(self):
        result = _parse_hf_dataset_uri("hf://datasets/org/repo/a/b/c")
        assert result == {"repo_id": "org/repo", "prefix": "a/b/c"}

    def test_no_prefix(self):
        result = _parse_hf_dataset_uri("hf://datasets/org/repo")
        assert result == {"repo_id": "org/repo", "prefix": ""}

    def test_trailing_slash(self):
        result = _parse_hf_dataset_uri("hf://datasets/org/repo/prefix/")
        assert result == {"repo_id": "org/repo", "prefix": "prefix"}
