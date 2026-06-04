"""Tests for local training-config discovery used by sol_eval."""

from __future__ import annotations

from pathlib import Path

import pytest

from verl_inf_evolve.sol_eval.experiment_discovery import (
    canonicalize_model_id,
    detect_model_from_remote_path,
    discover_training_experiment,
    load_training_experiment,
)


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestCanonicalizeModelId:
    def test_known_aliases_resolve_to_canonical_hf_ids(self):
        assert canonicalize_model_id("qwen3_4b_base") == "Qwen/Qwen3-4B-Base"
        assert canonicalize_model_id("Qwen3-8B") == "Qwen/Qwen3-8B"
        assert canonicalize_model_id("llama31_8b") == "meta-llama/Llama-3.1-8B-Instruct"
        assert canonicalize_model_id("ol37is") == "allenai/Olmo-3-7B-Instruct-SFT"

    def test_unknown_model_is_left_unchanged(self):
        assert canonicalize_model_id("custom/model") == "custom/model"


class TestDetectModelFromRemotePath:
    def test_detects_model_from_hf_remote(self):
        model = detect_model_from_remote_path(
            "hf://datasets/alice/SER/qwen3_4b_base/run-1"
        )
        assert model == "Qwen/Qwen3-4B-Base"

    def test_detects_olmo_model_from_hf_remote(self):
        model = detect_model_from_remote_path(
            "hf://datasets/alice/SER/olmo_3_7b_instruct_sft/run-1"
        )
        assert model == "allenai/Olmo-3-7B-Instruct-SFT"


class TestDiscoverTrainingExperiment:
    def test_discovers_by_run_name_and_model_alias(self, tmp_path):
        _write_yaml(
            tmp_path / "experiment_qwen3_4b_base" / "run.yaml",
            """
solver:
  model:
    path: Qwen/Qwen3-4B-Base
training:
  remote_sync_path: hf://datasets/__namespace__/SER/qwen3_4b_base/${wandb.group_name}
wandb:
  group_name: shared-run
""".strip()
            + "\n",
        )
        _write_yaml(
            tmp_path / "experiment_qwen3_8b_base" / "run.yaml",
            """
solver:
  model:
    path: Qwen/Qwen3-8B-Base
training:
  remote_sync_path: hf://datasets/__namespace__/SER/qwen3_8b_base/${wandb.group_name}
wandb:
  group_name: shared-run
""".strip()
            + "\n",
        )

        match = discover_training_experiment(
            run_name="shared-run",
            model_name="qwen3_4b_base",
            config_root=tmp_path,
        )

        assert match.model_path == "Qwen/Qwen3-4B-Base"
        assert match.remote_sync_path == "hf://datasets/__namespace__/SER/qwen3_4b_base/shared-run"
        assert match.path == tmp_path / "experiment_qwen3_4b_base" / "run.yaml"

    def test_discovers_unique_run_without_model_hint(self, tmp_path):
        _write_yaml(
            tmp_path / "experiment_qwen3_4b_base" / "run.yaml",
            """
solver:
  model:
    path: Qwen/Qwen3-4B-Base
training:
  remote_sync_path: s3://example-bucket/running-states/V3_2_qwen3_4b_base/${wandb.group_name}
wandb:
  group_name: unique-run
""".strip()
            + "\n",
        )

        match = discover_training_experiment(
            run_name="unique-run",
            config_root=tmp_path,
        )

        assert match.model_path == "Qwen/Qwen3-4B-Base"
        assert match.remote_sync_path == "s3://example-bucket/running-states/V3_2_qwen3_4b_base/unique-run"

    def test_raises_for_ambiguous_run_without_model_hint(self, tmp_path):
        _write_yaml(
            tmp_path / "experiment_qwen3_4b_base" / "run.yaml",
            """
solver:
  model:
    path: Qwen/Qwen3-4B-Base
training:
  remote_sync_path: s3://bucket/qwen3_4b_base/${wandb.group_name}
wandb:
  group_name: shared-run
""".strip()
            + "\n",
        )
        _write_yaml(
            tmp_path / "experiment_qwen3_8b_base" / "run.yaml",
            """
solver:
  model:
    path: Qwen/Qwen3-8B-Base
training:
  remote_sync_path: s3://bucket/qwen3_8b_base/${wandb.group_name}
wandb:
  group_name: shared-run
""".strip()
            + "\n",
        )

        with pytest.raises(ValueError, match="Multiple training configs matched"):
            discover_training_experiment(
                run_name="shared-run",
                config_root=tmp_path,
            )


class TestLoadTrainingExperiment:
    def test_loads_by_path_relative_to_config_root(self, tmp_path):
        config_path = tmp_path / "experiment_qwen3_4b_base" / "run.yaml"
        _write_yaml(
            config_path,
            """
solver:
  model:
    path: Qwen/Qwen3-4B-Base
training:
  remote_sync_path: hf://datasets/__namespace__/SER/qwen3_4b_base/${wandb.group_name}
wandb:
  group_name: path-run
""".strip()
            + "\n",
        )

        match = load_training_experiment(
            "experiment_qwen3_4b_base/run.yaml",
            config_root=tmp_path,
        )

        assert match.path == config_path
        assert match.run_name == "path-run"
        assert match.model_path == "Qwen/Qwen3-4B-Base"
        assert match.remote_sync_path == "hf://datasets/__namespace__/SER/qwen3_4b_base/path-run"

    def test_loads_by_repo_relative_path(self, tmp_path, monkeypatch):
        repo_root = tmp_path / "repo"
        config_root = repo_root / "verl_inf_evolve" / "config"
        config_path = (
            config_root
            / "experiment_qwen3_4b_base"
            / "FW-Alr_2e-6-Glr_2e-6-vanilla_smtm-TIS_token-dev_800-precond_cos.yaml"
        )
        _write_yaml(
            config_path,
            """
generator:
  model:
    path: Qwen/Qwen3-4B-Base
training:
  remote_sync_path: hf://datasets/__namespace__/SER/qwen3_4b_base/${wandb.group_name}
wandb:
  group_name: repo-relative-run
""".strip()
            + "\n",
        )
        monkeypatch.chdir(repo_root)

        match = load_training_experiment(
            "verl_inf_evolve/config/experiment_qwen3_4b_base/FW-Alr_2e-6-Glr_2e-6-vanilla_smtm-TIS_token-dev_800-precond_cos.yaml",
            config_root=config_root,
        )

        assert match.path == config_path
        assert match.run_name == "repo-relative-run"
        assert match.model_path == "Qwen/Qwen3-4B-Base"

    def test_raises_when_config_path_missing(self, tmp_path):
        with pytest.raises(ValueError, match="Training config not found"):
            load_training_experiment(
                "experiment_qwen3_4b_base/missing.yaml",
                config_root=tmp_path,
            )
