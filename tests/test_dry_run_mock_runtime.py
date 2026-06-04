from __future__ import annotations

import json
import os
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.dry_run.mock_runtime import (
    MockCPUWorkerGroup,
    MockDryRunCrash,
    make_mock_tokenizer,
)
from verl_inf_evolve.dry_run.mock_trainer import MockDryRunTrainer


class _DummyTracking:
    def __init__(self):
        self.logged: list[tuple[dict, int | None]] = []

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))


def _make_config(tmp_path, **training_overrides):
    local_dir = str(tmp_path / "dry_run_ckpts")
    config = {
        "training": {
            "max_ans_loop": 1,
            "max_gen_loop": 1,
            "save_every_n_steps": 1,
            "always_save_for_resume": False,
            "default_local_dir": local_dir,
            "default_hdfs_dir": None,
            "clear_local_dir_on_start": False,
            "remote_sync_path": None,
            "resume_from_remote": False,
            "remove_previous_ckpt": False,
            "fix_generator": False,
            "fix_answer_model": False,
            "generator_reward_structure": [],
            "generator_reward_components": [
                "influence_rewards",
                "spice_rewards",
                "invalid_rewards",
            ],
            "generator_reward_combination_mode": "sum_scores",
            "gen_invalid_penalty": 0.0,
            "seed": 123,
        },
        "generator": {
            "model": {"path": "mock/generator"},
            "rollout": {"n": 2},
        },
        "solver": {
            "model": {"path": "mock/solver"},
            "rollout": {"n": 2},
        },
        "curriculum": {"enabled": False},
        "benchmark_eval": {"enabled": False},
        "wandb": {
            "entity": None,
            "group_name": "dry-run-tests",
            "project_name": "dry-run-tests",
            "run_name": None,
        },
        "dry_run": {
            "enabled": True,
            "backend": "mock_cpu",
            "resume_loader": "mock",
            "mock_world_size": 2,
            "crash_after": "none",
            "checkpoint": {
                "model_shard_mb": 1.0,
                "optim_shard_mb": 1.5,
                "hf_total_mb": 3.0,
                "hf_num_shards": 3,
                "extra_state_kb": 32,
            },
            "stage_outputs": {
                "dev_output_mb": 0.1,
                "gen_output_mb": 0.1,
                "gen_answer_output_mb": 0.1,
            },
        },
    }
    config["training"].update(training_overrides)
    return OmegaConf.create(config)


def _make_trainer(config):
    trainer = MockDryRunTrainer(
        config=config,
        gen_tokenizer=make_mock_tokenizer("dry-run-generator"),
        solver_tokenizer=make_mock_tokenizer("dry-run-solver"),
        role_worker_mapping={},
        resource_pool_manager=None,
        ray_worker_group_cls=None,
    )
    trainer._init_tracking = lambda: _DummyTracking()
    trainer.init_workers()
    return trainer


def test_mock_worker_checkpoint_tree_and_loaders(tmp_path):
    config = _make_config(tmp_path)
    config.dry_run.checkpoint.hf_shard_sizes_mb = [1.0, 2.0, 3.0]
    config.dry_run.checkpoint.hf_num_shards = 99
    config.dry_run.checkpoint.hf_total_mb = 999.0
    worker_group = MockCPUWorkerGroup("generator", config)
    checkpoint_dir = tmp_path / "global_step_0" / "generator"

    worker_group.save_checkpoint(local_path=str(checkpoint_dir), global_step=0)

    manifest_path = checkpoint_dir / "dry_run_manifest.json"
    assert manifest_path.is_file()

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert manifest["mode"] == "mock_cpu"
    assert manifest["world_size"] == 2
    assert manifest["sizes"]["hf_num_shards"] == 3
    assert manifest["sizes"]["hf_total_bytes"] == 6 * 1024**2
    assert os.path.getsize(
        checkpoint_dir / "model_world_size_2_rank_0.pt"
    ) == manifest["sizes"]["model_shard_bytes"]
    assert os.path.getsize(
        checkpoint_dir / "optim_world_size_2_rank_1.pt"
    ) == manifest["sizes"]["optim_shard_bytes"]
    assert os.path.getsize(
        checkpoint_dir / "huggingface" / "model-00001-of-00003.safetensors"
    ) == 1 * 1024**2
    assert os.path.getsize(
        checkpoint_dir / "huggingface" / "model-00002-of-00003.safetensors"
    ) == 2 * 1024**2
    assert os.path.getsize(
        checkpoint_dir / "huggingface" / "model-00003-of-00003.safetensors"
    ) == 3 * 1024**2

    worker_group.load_checkpoint(local_path=str(checkpoint_dir))
    worker_group.load_hf_checkpoint(local_path=str(checkpoint_dir / "huggingface"))


def test_dry_run_fit_writes_checkpoint_and_marker(tmp_path):
    config = _make_config(tmp_path)
    trainer = _make_trainer(config)

    trainer.fit()

    local_dir = tmp_path / "dry_run_ckpts"
    assert (local_dir / "latest_checkpointed_iteration.txt").read_text().strip() == "0"
    assert (local_dir / "global_step_0" / "generator" / "dry_run_manifest.json").is_file()
    assert (local_dir / "global_step_0" / "solver" / "dry_run_manifest.json").is_file()
    assert (local_dir / "ans_0" / "dev_output.pt").is_file()
    assert (local_dir / "ans_0" / "gen_0" / "gen_output.pt").is_file()
    assert (local_dir / "ans_0" / "gen_0" / "gen_answer_output.pt").is_file()
    assert (local_dir / "ans_0" / "gen_0" / "rewards.json").is_file()


def test_dry_run_startup_cleanup_is_one_shot(tmp_path):
    local_dir = tmp_path / "dry_run_ckpts"
    local_dir.mkdir(parents=True)
    stale_file = local_dir / "stale.txt"
    stale_file.write_text("stale")

    config = _make_config(tmp_path, clear_local_dir_on_start=True)
    trainer = _make_trainer(config)

    assert config.training.clear_local_dir_on_start is False
    assert not stale_file.exists()

    trainer.fit()

    assert (local_dir / "latest_checkpointed_iteration.txt").read_text().strip() == "0"


def test_dry_run_checkpoint_resume_and_stage4_local_resume(tmp_path):
    local_dir = tmp_path / "dry_run_ckpts"

    crash_config = _make_config(tmp_path)
    crash_config.dry_run.crash_after = "stage4"
    trainer = _make_trainer(crash_config)

    with pytest.raises(MockDryRunCrash):
        trainer.fit()

    dev_output_path = local_dir / "ans_0" / "dev_output.pt"
    rewards_path = local_dir / "ans_0" / "gen_0" / "rewards.json"
    state_path = local_dir / "ans_0" / "state.json"

    assert dev_output_path.is_file()
    assert rewards_path.is_file()
    assert state_path.is_file()

    dev_mtime_before = dev_output_path.stat().st_mtime_ns
    rewards_mtime_before = rewards_path.stat().st_mtime_ns
    with open(state_path, "r") as f:
        crash_state = json.load(f)
    assert crash_state["stage_1_done"] is True
    assert crash_state["gen_loops"][0]["stage_4_done"] is True
    assert crash_state["stage_6_done"] is False

    resume_config = _make_config(tmp_path)
    resume_trainer = _make_trainer(resume_config)
    resume_trainer.fit()

    assert (local_dir / "latest_checkpointed_iteration.txt").read_text().strip() == "0"
    assert dev_output_path.stat().st_mtime_ns == dev_mtime_before
    assert rewards_path.stat().st_mtime_ns == rewards_mtime_before

    checkpoint_resume_config = _make_config(
        tmp_path,
        max_ans_loop=2,
        max_gen_loop=2,
    )
    checkpoint_resume_trainer = _make_trainer(checkpoint_resume_config)
    checkpoint_resume_trainer.fit()

    assert (local_dir / "latest_checkpointed_iteration.txt").read_text().strip() == "1"
    assert (local_dir / "global_step_1" / "generator" / "dry_run_manifest.json").is_file()
    assert (local_dir / "global_step_1" / "solver" / "dry_run_manifest.json").is_file()


def test_dry_run_interval_mode_exercises_non_checkpoint_loop(tmp_path):
    config = _make_config(
        tmp_path,
        max_ans_loop=3,
        save_every_n_steps=2,
    )
    trainer = _make_trainer(config)

    trainer.fit()

    local_dir = tmp_path / "dry_run_ckpts"
    assert (local_dir / "latest_checkpointed_iteration.txt").read_text().strip() == "2"
    assert (local_dir / "global_step_0" / "generator" / "dry_run_manifest.json").is_file()
    assert (local_dir / "global_step_2" / "generator" / "dry_run_manifest.json").is_file()
