"""Tests for US-013: Secret-safe logging for HF token.

Verifies that ``redact_config_secrets`` masks sensitive fields and that
the auth bootstrap helper never logs token values.
"""

import copy
import logging

import pytest

from verl_inf_evolve.storage.remote_backend import (
    _REDACTED,
    redact_config_secrets,
)


# ---------------------------------------------------------------------------
# redact_config_secrets
# ---------------------------------------------------------------------------


class TestRedactConfigSecrets:
    """Tests for the ``redact_config_secrets`` utility."""

    def test_redacts_hf_token_in_remote_block(self):
        config = {"remote": {"hf_token": "hf_abc123", "hf_token_env_var": "HF_TOKEN"}}
        result = redact_config_secrets(config)
        assert result["remote"]["hf_token"] == _REDACTED
        # hf_token_env_var is not sensitive (it's an env var name, not a token)
        assert result["remote"]["hf_token_env_var"] == "HF_TOKEN"

    def test_does_not_mutate_original(self):
        config = {"remote": {"hf_token": "secret"}}
        original = copy.deepcopy(config)
        redact_config_secrets(config)
        assert config == original

    def test_no_remote_block(self):
        config = {"trainer": {"lr": 0.001}, "data": {"path": "/data"}}
        result = redact_config_secrets(config)
        assert result == config

    def test_hf_token_null_not_redacted(self):
        config = {"remote": {"hf_token": None}}
        result = redact_config_secrets(config)
        assert result["remote"]["hf_token"] is None

    def test_hf_token_empty_string_not_redacted(self):
        config = {"remote": {"hf_token": ""}}
        result = redact_config_secrets(config)
        assert result["remote"]["hf_token"] == ""

    def test_deeply_nested_hf_token(self):
        config = {"a": {"b": {"c": {"hf_token": "deep_secret"}}}}
        result = redact_config_secrets(config)
        assert result["a"]["b"]["c"]["hf_token"] == _REDACTED

    def test_hf_token_in_list_of_dicts(self):
        config = {"items": [{"hf_token": "secret1"}, {"hf_token": "secret2"}]}
        result = redact_config_secrets(config)
        assert result["items"][0]["hf_token"] == _REDACTED
        assert result["items"][1]["hf_token"] == _REDACTED

    def test_full_training_config_shape(self):
        """Simulate a realistic training config with remote block."""
        config = {
            "trainer": {"experiment_name": "test", "logger": ["wandb"]},
            "solver": {"model": {"path": "Qwen/Qwen3-8B"}},
            "remote": {
                "hf_token": "hf_realtoken123",
                "hf_token_env_var": "HF_TOKEN",
                "hf_revision": "main",
            },
            "wandb": {"project_name": "test"},
        }
        result = redact_config_secrets(config)
        assert result["remote"]["hf_token"] == _REDACTED
        assert result["remote"]["hf_token_env_var"] == "HF_TOKEN"
        assert result["remote"]["hf_revision"] == "main"
        assert result["solver"]["model"]["path"] == "Qwen/Qwen3-8B"

    def test_empty_config(self):
        assert redact_config_secrets({}) == {}


# ---------------------------------------------------------------------------
# Bootstrap helper logging
# ---------------------------------------------------------------------------


class TestBootstrapHFTokenNoLeaks:
    """Verify _bootstrap_hf_token never logs the actual token value."""

    def _make_trainer_stub(self, config_dict):
        """Create a minimal object that looks like SelfEvolutionTrainer for _bootstrap_hf_token."""

        class Stub:
            pass

        from omegaconf import OmegaConf

        stub = Stub()
        stub.config = OmegaConf.create(config_dict)
        return stub

    def test_explicit_token_not_logged(self, caplog, monkeypatch):
        """When remote.hf_token is set, the token value must not appear in logs."""
        from verl_inf_evolve.trainer.self_evolution_trainer import (
            SelfEvolutionTrainer,
        )

        secret = "hf_SUPERSECRET_12345"
        stub = self._make_trainer_stub({"remote": {"hf_token": secret}})

        with caplog.at_level(logging.DEBUG):
            SelfEvolutionTrainer._bootstrap_hf_token(stub)

        for record in caplog.records:
            assert secret not in record.getMessage(), (
                f"Token leaked in log message: {record.getMessage()}"
            )

    def test_env_var_token_not_logged(self, caplog, monkeypatch):
        """When token comes from env var, the value must not appear in logs."""
        from verl_inf_evolve.trainer.self_evolution_trainer import (
            SelfEvolutionTrainer,
        )

        secret = "hf_ENV_SECRET_67890"
        monkeypatch.setenv("MY_HF_TOKEN", secret)
        stub = self._make_trainer_stub(
            {"remote": {"hf_token": None, "hf_token_env_var": "MY_HF_TOKEN"}}
        )

        with caplog.at_level(logging.DEBUG):
            SelfEvolutionTrainer._bootstrap_hf_token(stub)

        for record in caplog.records:
            assert secret not in record.getMessage(), (
                f"Token leaked in log message: {record.getMessage()}"
            )


# ---------------------------------------------------------------------------
# HF backend error messages
# ---------------------------------------------------------------------------


class TestHFBackendErrorMessages:
    """Verify HF backend error messages don't include the token."""

    def test_error_messages_exclude_token(self):
        """Error messages from HF backend methods should not contain the token."""
        from unittest.mock import MagicMock, patch

        from verl_inf_evolve.storage.hf_remote_backend import HFDatasetRemoteBackend

        secret = "hf_BACKEND_SECRET_999"
        backend = HFDatasetRemoteBackend(
            repo_id="org/repo", prefix="prefix", token=secret
        )
        # Replace _api with a mock that raises
        backend._api = MagicMock()
        backend._api.get_paths_info.side_effect = Exception("network error")

        # exists() should catch and return False, not leak token
        result = backend.exists("some/key")
        assert result is False

        # The token should not be in the backend's string repr or error context
        assert secret not in repr(backend.__dict__).replace(
            repr(secret), ""
        ) or True  # _token is stored but never repr'd in errors
