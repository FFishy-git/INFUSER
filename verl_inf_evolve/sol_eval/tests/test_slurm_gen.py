"""Tests for verl_inf_evolve/sol_eval/slurm_gen.py — SLURM job script generation."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl_inf_evolve.sol_eval.slurm_gen import (
    _build_eval_command,
    _load_template,
    generate_slurm_scripts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_run_config() -> SimpleNamespace:
    """A minimal config namespace for testing."""
    return SimpleNamespace(
        run_name="FW-Alr_2e-6",
        remote_sync_path="s3://example-bucket/experiments/run_001",
        model_path="Qwen/Qwen3-8B",
        checkpoints=[0, 5, 10],
        benchmarks=["supergpqa_2000", "aime"],
        temperature=0.7,
        response_length=32768,
        n_samples=8,
        tp_size=1,
        gpu_memory_utilization=0.5,
    )


@pytest.fixture
def single_run_config() -> SimpleNamespace:
    """Config namespace with single checkpoint and benchmark."""
    return SimpleNamespace(
        run_name="test-run",
        remote_sync_path="s3://bucket/run",
        model_path="Qwen/Qwen3-8B",
        checkpoints=[5],
        benchmarks=["aime"],
        temperature=0.3,
        response_length=16384,
        n_samples=4,
        tp_size=2,
        gpu_memory_utilization=0.5,
    )


# ---------------------------------------------------------------------------
# _load_template tests
# ---------------------------------------------------------------------------

class TestLoadTemplate:
    """Tests for _load_template()."""

    def test_loads_default_template(self):
        """Default template should load successfully."""
        content = _load_template("default")
        assert "{job_name}" in content
        assert "{gpus_per_node}" in content
        assert "{command}" in content

    def test_template_has_all_placeholders(self):
        """Template should contain all expected placeholders."""
        content = _load_template("default")
        placeholders = [
            "{job_name}",
            "{gpus_per_node}",
            "{timestamp}",
            "{base_path}",
            "{checkpoint_name}",
            "{benchmark}",
            "{temperature}",
            "{response_length}",
            "{command}",
            "{time_limit}",
            "{partition}",
        ]
        for ph in placeholders:
            assert ph in content, f"Missing placeholder: {ph}"

    def test_missing_template_raises_file_not_found(self):
        """Non-existent template should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="SLURM template not found"):
            _load_template("nonexistent_template_xyz")


# ---------------------------------------------------------------------------
# _build_eval_command tests
# ---------------------------------------------------------------------------

class TestBuildEvalCommand:
    """Tests for _build_eval_command()."""

    def test_basic_command(self):
        cmd = _build_eval_command(
            run_config=SimpleNamespace(
                run_name="FW-Alr_2e-6",
                remote_sync_path="hf://datasets/alice/SER/qwen3_4b_base/run1",
                model_path="Qwen/Qwen3-4B-Base",
                temperature=0.7,
                response_length=32768,
                n_samples=8,
                tp_size=2,
                gpu_memory_utilization=0.6,
            ),
            ckpt_num=5,
            benchmark="supergpqa_2000",
            gpus_per_node=4,
        )
        assert "python -m verl_inf_evolve.sol_eval.sol_eval" in cmd
        assert "eval.run_name=FW-Alr_2e-6" in cmd
        assert "eval.remote_sync_path=hf://datasets/alice/SER/qwen3_4b_base/run1" in cmd
        assert "eval.checkpoints=[5]" in cmd
        assert "eval.benchmarks=[supergpqa_2000]" in cmd
        assert "solver.model.path=Qwen/Qwen3-4B-Base" in cmd
        assert "eval.tp_size=2" in cmd
        assert "trainer.n_gpus_per_node=4" in cmd

    def test_hydra_override_format(self):
        cmd = _build_eval_command(
            run_config=SimpleNamespace(
                run_name="run1",
                remote_sync_path="s3://bucket/run",
                model_path="Qwen/Qwen3-8B",
                temperature=0.7,
                response_length=16384,
                n_samples=1,
                tp_size=1,
                gpu_memory_utilization=0.5,
            ),
            ckpt_num=0,
            benchmark="aime",
            gpus_per_node=1,
        )
        assert "eval.run_name=run1" in cmd
        assert "eval.checkpoints=[0]" in cmd
        assert "eval.benchmarks=[aime]" in cmd
        assert "eval.remote_sync_path=s3://bucket/run" in cmd
        assert "solver.model.path=Qwen/Qwen3-8B" in cmd

    def test_remote_sync_path_is_optional(self):
        cmd = _build_eval_command(
            run_config=SimpleNamespace(
                run_name="run1",
                model_path="qwen3_4b_base",
                temperature=0.7,
                response_length=16384,
                n_samples=1,
                tp_size=1,
                gpu_memory_utilization=0.5,
            ),
            ckpt_num=0,
            benchmark="aime",
            gpus_per_node=1,
        )
        assert "eval.run_name=run1" in cmd
        assert "eval.remote_sync_path=" not in cmd
        assert "eval.model_path=qwen3_4b_base" in cmd


# ---------------------------------------------------------------------------
# generate_slurm_scripts tests
# ---------------------------------------------------------------------------

class TestGenerateSlurmScripts:
    """Tests for generate_slurm_scripts()."""

    def test_generates_correct_number_of_scripts(self, sample_run_config):
        """Should generate one script per (checkpoint, benchmark) pair."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(
                sample_run_config,
                output_dir=tmpdir,
            )
            scripts = list(Path(out_dir).glob("*.sh"))
            # 3 checkpoints x 2 benchmarks = 6 scripts
            assert len(scripts) == 6

    def test_single_pair(self, single_run_config):
        """Single checkpoint/benchmark should produce one script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(
                single_run_config,
                output_dir=tmpdir,
            )
            scripts = list(Path(out_dir).glob("*.sh"))
            assert len(scripts) == 1

    def test_script_filenames(self, sample_run_config):
        """Script filenames should follow eval_{run_name}_ckpt{N}_{benchmark}.sh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(
                sample_run_config,
                output_dir=tmpdir,
            )
            filenames = sorted(f.name for f in Path(out_dir).glob("*.sh"))
            expected = sorted([
                "eval_FW-Alr_2e-6_ckpt0_supergpqa_2000.sh",
                "eval_FW-Alr_2e-6_ckpt0_aime.sh",
                "eval_FW-Alr_2e-6_ckpt5_supergpqa_2000.sh",
                "eval_FW-Alr_2e-6_ckpt5_aime.sh",
                "eval_FW-Alr_2e-6_ckpt10_supergpqa_2000.sh",
                "eval_FW-Alr_2e-6_ckpt10_aime.sh",
            ])
            assert filenames == expected

    def test_script_content_contains_sbatch_directives(self, single_run_config):
        """Generated script should contain SBATCH directives."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(
                single_run_config,
                output_dir=tmpdir,
            )
            script = next(Path(out_dir).glob("*.sh"))
            content = script.read_text()
            assert "#!/bin/bash" in content
            assert "#SBATCH --job-name=eval_test-run_ckpt5_aime" in content
            assert "#SBATCH --gres=gpu:2" in content  # tp_size=2

    def test_script_content_contains_eval_command(self, single_run_config):
        """Generated script should contain the Hydra eval command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(
                single_run_config,
                output_dir=tmpdir,
            )
            script = next(Path(out_dir).glob("*.sh"))
            content = script.read_text()
            assert "python -m verl_inf_evolve.sol_eval.sol_eval" in content
            assert "eval.run_name=test-run" in content
            assert "eval.remote_sync_path=s3://bucket/run" in content
            assert "eval.checkpoints=[5]" in content
            assert "eval.benchmarks=[aime]" in content
            assert "solver.model.path=Qwen/Qwen3-8B" in content
            assert "solver.rollout.response_length=16384" in content
            assert "trainer.n_gpus_per_node=2" in content

    def test_script_content_contains_metadata_comments(self, single_run_config):
        """Generated script should contain metadata comments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(
                single_run_config,
                output_dir=tmpdir,
            )
            script = next(Path(out_dir).glob("*.sh"))
            content = script.read_text()
            assert "# Model: Qwen/Qwen3-8B" in content
            assert "# Checkpoint: global_step_5" in content
            assert "# Benchmark: aime" in content
            assert "# Temperature: 0.3" in content
            assert "# Response Length: 16384" in content

    def test_remote_eval_base_is_included_when_present(self):
        run_config = SimpleNamespace(
            run_name="hf-run",
            remote_sync_path="hf://datasets/alice/SER/qwen3_4b_base/hf-run",
            remote_eval_base="hf://datasets/alice/eval-results/results",
            model_path="Qwen/Qwen3-4B-Base",
            checkpoints=[3],
            benchmarks=["aime"],
            temperature=0.7,
            response_length=16384,
            n_samples=2,
            tp_size=1,
            gpu_memory_utilization=0.5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(run_config, output_dir=tmpdir)
            script = next(Path(out_dir).glob("*.sh"))
            content = script.read_text()
            assert "eval.remote_eval_base=hf://datasets/alice/eval-results/results" in content

    def test_result_detail_is_included_when_present(self):
        run_config = SimpleNamespace(
            run_name="full-run",
            remote_sync_path="hf://datasets/alice/SER/qwen3_4b_base/full-run",
            model_path="Qwen/Qwen3-4B-Base",
            checkpoints=[3],
            benchmarks=["aime"],
            temperature=0.7,
            response_length=16384,
            n_samples=2,
            tp_size=1,
            gpu_memory_utilization=0.5,
            result_detail="full",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(run_config, output_dir=tmpdir)
            script = next(Path(out_dir).glob("*.sh"))
            content = script.read_text()
            assert "eval.result_detail=full" in content

    def test_scripts_are_executable(self, single_run_config):
        """Generated scripts should have executable permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(
                single_run_config,
                output_dir=tmpdir,
            )
            script = next(Path(out_dir).glob("*.sh"))
            assert os.access(script, os.X_OK)

    def test_custom_gpus_per_node(self, single_run_config):
        """gpus_per_node parameter should override tp_size default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(
                single_run_config,
                output_dir=tmpdir,
                gpus_per_node=4,
            )
            script = next(Path(out_dir).glob("*.sh"))
            content = script.read_text()
            assert "#SBATCH --gres=gpu:4" in content

    def test_default_output_dir_has_timestamp(self, single_run_config):
        """When output_dir is None, should create verl_inf_evolve/sol_eval/slurm_scripts_{timestamp}/."""
        import shutil

        out_dir = generate_slurm_scripts(
            single_run_config,
        )
        try:
            assert out_dir.startswith("verl_inf_evolve/sol_eval/slurm_scripts_")
            assert Path(out_dir).is_dir()
            scripts = list(Path(out_dir).glob("*.sh"))
            assert len(scripts) == 1
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_missing_template_raises_error(self, single_run_config):
        """Non-existent template name should raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                generate_slurm_scripts(
                    single_run_config,
                    template_name="nonexistent_xyz",
                    output_dir=tmpdir,
                )

    def test_custom_time_limit_and_partition(self, single_run_config):
        """Custom time_limit and partition should be substituted in template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(
                single_run_config,
                output_dir=tmpdir,
                time_limit="12:00:00",
                partition="general",
            )
            script = next(Path(out_dir).glob("*.sh"))
            content = script.read_text()
            assert "#SBATCH --time=12:00:00" in content
            assert "#SBATCH --partition=general" in content

    def test_no_unresolved_placeholders(self, sample_run_config):
        """Generated scripts should have no remaining {placeholder} strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_slurm_scripts(
                sample_run_config,
                output_dir=tmpdir,
            )
            for script in Path(out_dir).glob("*.sh"):
                content = script.read_text()
                # Check no Python format placeholders remain (skip bash ${ vars)
                import re
                # Match {word} but not ${word}
                remaining = re.findall(r"(?<!\$)\{[a-z_]+\}", content)
                assert remaining == [], f"Unresolved placeholders in {script.name}: {remaining}"


class TestReusableSlurmLauncher:
    """Smoke tests for the checked-in reusable SLURM launcher."""

    def test_launcher_has_valid_bash_syntax(self):
        launcher = Path("verl_inf_evolve/sol_eval/sol_eval_slurm.sh")
        result = subprocess.run(
            ["bash", "-n", str(launcher)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
