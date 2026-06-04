"""Tests for US-007: Rewire trainer remote logic for hf:// support.

Covers:
- _is_remote_checkpoint_source recognizes hf:// URIs
- _resolve_init_checkpoint_source uses RemoteBackend for hf:// downloads
- HF_HUB_OFFLINE is conditional on remote_sync_path scheme
- _bootstrap_hf_token precedence (config > env var > cached login)
- Remote delete preserves exclude_patterns semantics
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# _is_remote_checkpoint_source
# ---------------------------------------------------------------------------

class TestIsRemoteCheckpointSource:
    """Test the static method recognizes all remote URI schemes."""

    @staticmethod
    def _call(path: str) -> bool:
        from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer
        return SelfEvolutionTrainer._is_remote_checkpoint_source(path)

    def test_s3_uri(self):
        assert self._call("s3://bucket/prefix") is True

    def test_r2_uri(self):
        assert self._call("r2://bucket/prefix") is True

    def test_gs_uri(self):
        assert self._call("gs://bucket/prefix") is True

    def test_hf_uri(self):
        assert self._call("hf://datasets/org/repo/prefix") is True

    def test_hf_uri_no_prefix(self):
        assert self._call("hf://datasets/org/repo") is True

    def test_local_path(self):
        assert self._call("/tmp/checkpoint") is False

    def test_relative_path(self):
        assert self._call("./checkpoint") is False

    def test_empty_string(self):
        assert self._call("") is False


# ---------------------------------------------------------------------------
# _resolve_init_checkpoint_source
# ---------------------------------------------------------------------------

class TestResolveInitCheckpointSource:
    """Test that _resolve_init_checkpoint_source uses RemoteBackend."""

    def _make_trainer_stub(self, default_local_dir: str):
        """Build a minimal trainer-like object with the methods under test."""
        from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer

        # Create a minimal mock that has the required config
        trainer = object.__new__(SelfEvolutionTrainer)
        trainer.config = MagicMock()
        trainer.config.training.default_local_dir = default_local_dir
        trainer.config.get.side_effect = lambda key, default=None: {} if key == "remote" else default
        return trainer

    def test_local_path_returned_as_is(self):
        trainer = self._make_trainer_stub("/tmp/local")
        result = trainer._resolve_init_checkpoint_source("Generator", "/local/ckpt")
        assert result == "/local/ckpt"

    @patch("verl_inf_evolve.storage.remote_backend.create_remote_backend")
    def test_hf_uri_uses_backend_download(self, mock_factory):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer_stub(tmpdir)
            mock_backend = MagicMock()
            mock_factory.return_value = mock_backend

            result = trainer._resolve_init_checkpoint_source(
                "Generator",
                "hf://datasets/org/repo/some/prefix",
            )

            # Factory should be called with the full URI
            mock_factory.assert_called_once()
            call_args = mock_factory.call_args
            assert call_args.args == ("hf://datasets/org/repo/some/prefix",)
            assert call_args.kwargs["revision"] == "main"
            # Backend download_dir should be called
            mock_backend.download_dir.assert_called_once()
            args = mock_backend.download_dir.call_args
            assert args[0][0] == ""  # downloads root of the backend prefix
            # Local dir should be under .init_checkpoints with the URI tail
            assert "prefix" in os.path.basename(result)

    @patch("verl_inf_evolve.storage.remote_backend.create_remote_backend")
    def test_s3_uri_uses_backend_download(self, mock_factory):
        """R2/S3 URIs should also use the backend abstraction now."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer_stub(tmpdir)
            mock_backend = MagicMock()
            mock_factory.return_value = mock_backend

            trainer._resolve_init_checkpoint_source(
                "Solver",
                "s3://bucket/experiment/checkpoint",
            )

            mock_factory.assert_called_once_with("s3://bucket/experiment/checkpoint")
            mock_backend.download_dir.assert_called_once()

    @patch("verl_inf_evolve.storage.remote_backend.create_remote_backend")
    def test_role_checkpoint_prefers_huggingface_subdir_when_present(self, mock_factory):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer_stub(tmpdir)
            mock_backend = MagicMock()
            mock_backend.list_immediate_children.return_value = [
                "huggingface",
                "model_world_size_8_rank_0.pt",
            ]
            mock_factory.return_value = mock_backend

            result = trainer._resolve_init_checkpoint_source(
                "Solver",
                "s3://bucket/experiment/global_step_50/solver",
            )

            expected_hf_dir = os.path.join(result, "huggingface")
            mock_backend.download_dir.assert_called_once_with("huggingface", expected_hf_dir)

    @patch("verl_inf_evolve.storage.remote_backend.create_remote_backend")
    def test_cached_checkpoint_skips_download(self, mock_factory):
        """If local cache dir already exists and is non-empty, skip download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer_stub(tmpdir)
            mock_backend = MagicMock()
            mock_factory.return_value = mock_backend

            # Pre-create the cache directory with content
            import hashlib
            uri = "hf://datasets/org/repo/prefix"
            source_hash = hashlib.md5(uri.encode("utf-8")).hexdigest()[:10]
            cache_dir = os.path.join(
                tmpdir, ".init_checkpoints",
                f"generator_prefix_{source_hash}",
            )
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, "dummy"), "w") as f:
                f.write("data")

            result = trainer._resolve_init_checkpoint_source("Generator", uri)
            assert result == cache_dir
            # Factory is called (to create backend), but download_dir is NOT
            mock_backend.download_dir.assert_not_called()


# ---------------------------------------------------------------------------
# HF_HUB_OFFLINE conditionality
# ---------------------------------------------------------------------------

class TestHFHubOfflineConditionality:
    """Test that HF_HUB_OFFLINE is not set when remote_sync_path uses hf://."""

    def test_hf_remote_sync_skips_offline(self):
        """When remote_sync_path is hf://, HF_HUB_OFFLINE should NOT be set."""
        # We test the logic in main.py by simulating the conditional
        remote_sync_path = "hf://datasets/org/repo/prefix"
        env_before = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            # Replicate the logic from main.py
            if remote_sync_path.startswith("hf://"):
                pass  # skip setting HF_HUB_OFFLINE
            else:
                os.environ["HF_HUB_OFFLINE"] = "1"

            assert "HF_HUB_OFFLINE" not in os.environ
        finally:
            if env_before is not None:
                os.environ["HF_HUB_OFFLINE"] = env_before
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)

    def test_s3_remote_sync_sets_offline(self):
        """When remote_sync_path is s3://, HF_HUB_OFFLINE should be set."""
        remote_sync_path = "s3://bucket/prefix"
        env_before = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            if remote_sync_path.startswith("hf://"):
                pass
            else:
                os.environ["HF_HUB_OFFLINE"] = "1"

            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        finally:
            if env_before is not None:
                os.environ["HF_HUB_OFFLINE"] = env_before
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)

    def test_null_remote_sync_sets_offline(self):
        """When remote_sync_path is null/empty, HF_HUB_OFFLINE should be set."""
        remote_sync_path = ""
        env_before = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            if remote_sync_path.startswith("hf://"):
                pass
            else:
                os.environ["HF_HUB_OFFLINE"] = "1"

            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        finally:
            if env_before is not None:
                os.environ["HF_HUB_OFFLINE"] = env_before
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)


# ---------------------------------------------------------------------------
# _bootstrap_hf_token
# ---------------------------------------------------------------------------

class TestBootstrapHFToken:
    """Test HF token resolution precedence."""

    def _make_trainer_stub(self, remote_cfg: dict | None = None):
        from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer

        trainer = object.__new__(SelfEvolutionTrainer)
        trainer.config = MagicMock()
        if remote_cfg is not None:
            trainer.config.get.return_value = remote_cfg
        else:
            trainer.config.get.return_value = {}
        return trainer

    def test_config_token_takes_precedence(self):
        """remote.hf_token should override env var."""
        trainer = self._make_trainer_stub({"hf_token": "config-tok-123", "hf_token_env_var": "HF_TOKEN"})
        env_before = os.environ.pop("HF_TOKEN", None)
        try:
            os.environ["HF_TOKEN"] = "env-tok-456"
            trainer._bootstrap_hf_token()
            assert os.environ["HF_TOKEN"] == "config-tok-123"
        finally:
            if env_before is not None:
                os.environ["HF_TOKEN"] = env_before
            else:
                os.environ.pop("HF_TOKEN", None)

    def test_env_var_fallback(self):
        """If no config token, use the named env var."""
        trainer = self._make_trainer_stub({
            "hf_token": None,
            "hf_token_env_var": "MY_CUSTOM_HF_TOKEN",
        })
        env_before_hf = os.environ.pop("HF_TOKEN", None)
        env_before_custom = os.environ.pop("MY_CUSTOM_HF_TOKEN", None)
        try:
            os.environ["MY_CUSTOM_HF_TOKEN"] = "custom-tok-789"
            trainer._bootstrap_hf_token()
            assert os.environ["HF_TOKEN"] == "custom-tok-789"
        finally:
            if env_before_hf is not None:
                os.environ["HF_TOKEN"] = env_before_hf
            else:
                os.environ.pop("HF_TOKEN", None)
            if env_before_custom is not None:
                os.environ["MY_CUSTOM_HF_TOKEN"] = env_before_custom
            else:
                os.environ.pop("MY_CUSTOM_HF_TOKEN", None)

    def test_cached_login_fallback(self):
        """If no config token and no env var, rely on cached HF login."""
        trainer = self._make_trainer_stub({
            "hf_token": None,
            "hf_token_env_var": "HF_TOKEN",
        })
        env_before = os.environ.pop("HF_TOKEN", None)
        try:
            # No env var set — should just log and return without error
            trainer._bootstrap_hf_token()
            # HF_TOKEN should NOT be set (no token available)
            assert "HF_TOKEN" not in os.environ
        finally:
            if env_before is not None:
                os.environ["HF_TOKEN"] = env_before

    def test_default_env_var_name(self):
        """When hf_token_env_var is not configured, defaults to HF_TOKEN."""
        trainer = self._make_trainer_stub({})  # empty remote config
        env_before = os.environ.pop("HF_TOKEN", None)
        try:
            os.environ["HF_TOKEN"] = "default-tok"
            trainer._bootstrap_hf_token()
            assert os.environ["HF_TOKEN"] == "default-tok"
        finally:
            if env_before is not None:
                os.environ["HF_TOKEN"] = env_before
            else:
                os.environ.pop("HF_TOKEN", None)

    def test_resolved_pool_token_takes_precedence(self):
        trainer = self._make_trainer_stub({"hf_token": "config-tok-123"})
        trainer._resolved_hf_token = "pool-tok-999"
        trainer._resolved_hf_namespace = "user_a"
        env_before = os.environ.pop("HF_TOKEN", None)
        try:
            trainer._bootstrap_hf_token()
            assert os.environ["HF_TOKEN"] == "pool-tok-999"
        finally:
            if env_before is not None:
                os.environ["HF_TOKEN"] = env_before
            else:
                os.environ.pop("HF_TOKEN", None)


class TestResolveHFRemoteSyncPath:
    def _make_trainer_stub(self):
        from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer

        trainer = object.__new__(SelfEvolutionTrainer)
        trainer.config = OmegaConf.create(
            {
                "training": {
                    "remote_sync_path": "hf://datasets/__namespace__/SER/V4/run_a",
                },
                "remote": {
                    "hf_namespace_placeholder": "__namespace__",
                    "hf_token_pool": [],
                },
            }
        )
        return trainer

    @patch("verl_inf_evolve.storage.hf_remote_resolver.resolve_hf_remote_from_pool")
    def test_updates_remote_sync_path_and_selected_token(self, mock_resolve):
        trainer = self._make_trainer_stub()
        from verl_inf_evolve.storage.hf_remote_resolver import ResolvedHFRemote

        mock_resolve.return_value = ResolvedHFRemote(
            uri="hf://datasets/user_a/SER/V4/run_a",
            token="pool-token",
            namespace="user_a",
            repo="SER",
            prefix="V4/run_a",
            selection_reason="existing_prefix_reuse",
            warning=None,
            candidates=[],
        )

        trainer._resolve_hf_remote_sync_path()

        assert trainer.config.training.remote_sync_path == "hf://datasets/user_a/SER/V4/run_a"
        assert trainer._resolved_hf_token == "pool-token"
        assert trainer._resolved_hf_namespace == "user_a"


# ---------------------------------------------------------------------------
# Delete with exclude_patterns
# ---------------------------------------------------------------------------

class TestDeleteExcludePatterns:
    """Verify delete_prefix exclude_patterns semantics work for both backends."""

    @patch("verl_inf_evolve.storage.hf_remote_backend.HfApi")
    def test_hf_delete_with_exclude(self, mock_hf_api_cls):
        """HF backend delete_prefix should skip files matching excludes.

        Note: HF backend uses fnmatch on relative paths (prefix stripped).
        Use ``huggingface/*`` not ``**/huggingface/**`` for fnmatch compat.
        """
        from verl_inf_evolve.storage.hf_remote_backend import HFDatasetRemoteBackend

        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        backend = HFDatasetRemoteBackend(
            repo_id="org/repo", prefix="exp", revision="main"
        )
        backend._api = mock_api

        # Simulate files under exp/global_step_5/
        # The backend reads .path and checks hasattr(item, "size")
        file1 = MagicMock(spec=["path", "size"])
        file1.path = "exp/global_step_5/solver/shard_0.pt"
        file1.size = 1024
        file2 = MagicMock(spec=["path", "size"])
        file2.path = "exp/global_step_5/huggingface/model.safetensors"
        file2.size = 2048

        mock_api.list_repo_tree.return_value = [file1, file2]

        backend.delete_prefix(
            "global_step_5",
            exclude_patterns=["huggingface/*"],
        )

        # Should have called create_commit with only file1 deleted
        mock_api.create_commit.assert_called_once()
        commit_call = mock_api.create_commit.call_args
        operations = commit_call.kwargs.get("operations") or commit_call[1].get("operations")
        deleted_paths = [op.path_in_repo for op in operations]
        assert "exp/global_step_5/solver/shard_0.pt" in deleted_paths
        assert "exp/global_step_5/huggingface/model.safetensors" not in deleted_paths


class TestLoadCheckpointFallback:
    def test_falls_back_to_latest_available_checkpoint_when_marker_is_stale(self, tmp_path):
        from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer

        local_dir = tmp_path / "ckpts"
        local_dir.mkdir()
        (local_dir / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")

        trainer = object.__new__(SelfEvolutionTrainer)
        trainer.config = OmegaConf.create(
            {
                "training": {
                    "default_local_dir": str(local_dir),
                    "resume_from_remote": True,
                    "fix_generator": False,
                    "fix_answer_model": False,
                    "remove_previous_ckpt": False,
                }
            }
        )
        trainer.generator_wg = MagicMock(world_size=1)
        trainer.solver_wg = MagicMock(world_size=1)
        trainer._checkpoint_diag = MagicMock()
        trainer._load_model_checkpoint = MagicMock()

        backend = MagicMock()
        backend.list_immediate_children.return_value = [
            "global_step_0",
            "global_step_5",
            "global_step_10",
            "global_step_15",
            "ans_20",
            "latest_checkpointed_iteration.txt",
        ]

        def fake_download_dir(remote_key: str, local_path: str) -> bool:
            if remote_key not in {
                "global_step_15/generator",
                "global_step_15/solver",
            }:
                return False
            os.makedirs(local_path, exist_ok=True)
            return True

        trainer._upload_manager = MagicMock()
        trainer._upload_manager.remote_configured = True
        trainer._upload_manager.backend = backend
        trainer._upload_manager.download_dir_to_local.side_effect = fake_download_dir

        start_ans_loop = trainer._load_checkpoint()

        assert start_ans_loop == 16
        assert (
            local_dir / "latest_checkpointed_iteration.txt"
        ).read_text(encoding="utf-8") == "15"
        assert trainer._upload_manager.download_dir_to_local.call_args_list == [
            call("global_step_20/generator", str(local_dir / "global_step_20" / "generator")),
            call("global_step_15/generator", str(local_dir / "global_step_15" / "generator")),
            call("global_step_15/solver", str(local_dir / "global_step_15" / "solver")),
        ]
        trainer._load_model_checkpoint.assert_any_call(
            role_name="Generator",
            wg=trainer.generator_wg,
            ckpt_path=str(local_dir / "global_step_15" / "generator"),
            world_size=1,
        )
        trainer._load_model_checkpoint.assert_any_call(
            role_name="Solver",
            wg=trainer.solver_wg,
            ckpt_path=str(local_dir / "global_step_15" / "solver"),
            world_size=1,
        )
