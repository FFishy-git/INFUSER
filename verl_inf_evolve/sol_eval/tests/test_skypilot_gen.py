"""Tests for verl_inf_evolve/sol_eval/skypilot_gen.py — SkyPilot task YAML generation."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl_inf_evolve.sol_eval.skypilot_gen import (
    _build_eval_command,
    _load_template,
    _slugify,
    generate_skypilot_tasks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_run_config() -> SimpleNamespace:
    """A config namespace with multiple checkpoints and benchmarks."""
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
# _slugify tests
# ---------------------------------------------------------------------------

class TestSlugify:
    """Tests for _slugify()."""

    def test_basic_name(self):
        assert _slugify("my-run") == "my-run"

    def test_underscores_replaced(self):
        assert _slugify("FW-Alr_2e-6") == "fw-alr-2e-6"

    def test_special_chars_replaced(self):
        assert _slugify("run@v3.1") == "run-v3-1"

    def test_multiple_consecutive_hyphens_collapsed(self):
        assert _slugify("a__b--c") == "a-b-c"

    def test_strips_leading_trailing_hyphens(self):
        assert _slugify("_test_") == "test"


# ---------------------------------------------------------------------------
# _load_template tests
# ---------------------------------------------------------------------------

class TestLoadTemplate:
    """Tests for _load_template()."""

    def test_loads_default_template(self):
        """Default skypilot_eval template should load successfully."""
        content = _load_template("skypilot_eval")
        assert "{run_name}" in content
        assert "{ckpt_num}" in content
        assert "{command}" in content

    def test_template_has_all_placeholders(self):
        """Template should contain all expected placeholders."""
        content = _load_template("skypilot_eval")
        placeholders = [
            "{timestamp}",
            "{run_name}",
            "{run_name_slug}",
            "{ckpt_num}",
            "{benchmarks_str}",
            "{accelerator}",
            "{n_gpus}",
            "{wandb_entity}",
            "{command}",
        ]
        for ph in placeholders:
            assert ph in content, f"Missing placeholder: {ph}"

    def test_template_has_skypilot_structure(self):
        """Template should have SkyPilot YAML structure."""
        content = _load_template("skypilot_eval")
        assert "resources:" in content
        assert "cloud: kubernetes" in content
        assert "setup:" in content
        assert "run:" in content
        assert "workdir:" in content
        assert "envs:" in content

    def test_missing_template_raises_file_not_found(self):
        """Non-existent template should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="SkyPilot template not found"):
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
            benchmarks=["supergpqa_2000", "aime"],
            n_gpus=4,
        )
        assert "python -m verl_inf_evolve.sol_eval.sol_eval" in cmd
        assert "eval.run_name=FW-Alr_2e-6" in cmd
        assert "eval.remote_sync_path=hf://datasets/alice/SER/qwen3_4b_base/run1" in cmd
        assert "eval.checkpoints=[5]" in cmd
        assert "eval.benchmarks=[supergpqa_2000,aime]" in cmd
        assert "solver.model.path=Qwen/Qwen3-4B-Base" in cmd
        assert "trainer.n_gpus_per_node=4" in cmd

    def test_single_benchmark(self):
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
            benchmarks=["aime"],
            n_gpus=1,
        )
        assert "eval.benchmarks=[aime]" in cmd
        assert "eval.run_name=run1" in cmd

    def test_multiple_benchmarks_comma_separated(self):
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
            ckpt_num=10,
            benchmarks=["a", "b", "c"],
            n_gpus=2,
        )
        assert "eval.benchmarks=[a,b,c]" in cmd

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
            ckpt_num=10,
            benchmarks=["aime"],
            n_gpus=2,
        )
        assert "eval.run_name=run1" in cmd
        assert "eval.remote_sync_path=" not in cmd
        assert "eval.model_path=qwen3_4b_base" in cmd


# ---------------------------------------------------------------------------
# generate_skypilot_tasks tests
# ---------------------------------------------------------------------------

class TestGenerateSkypilotTasks:
    """Tests for generate_skypilot_tasks()."""

    def test_generates_one_yaml_per_checkpoint(self, sample_run_config):
        """Should generate one YAML per checkpoint (not per checkpoint x benchmark)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                sample_run_config,
                output_dir=tmpdir,
            )
            yamls = list(Path(out_dir).glob("*.yaml"))
            # 3 checkpoints -> 3 YAMLs
            assert len(yamls) == 3

    def test_single_checkpoint(self, single_run_config):
        """Single checkpoint should produce one YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                single_run_config,
                output_dir=tmpdir,
            )
            yamls = list(Path(out_dir).glob("*.yaml"))
            assert len(yamls) == 1

    def test_yaml_filenames(self, sample_run_config):
        """YAML filenames should follow eval_{run_name}_ckpt{N}.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                sample_run_config,
                output_dir=tmpdir,
            )
            filenames = sorted(f.name for f in Path(out_dir).glob("*.yaml"))
            expected = sorted([
                "eval_FW-Alr_2e-6_ckpt0.yaml",
                "eval_FW-Alr_2e-6_ckpt5.yaml",
                "eval_FW-Alr_2e-6_ckpt10.yaml",
            ])
            assert filenames == expected

    def test_yaml_contains_skypilot_fields(self, single_run_config):
        """Generated YAML should contain SkyPilot resource fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                single_run_config,
                output_dir=tmpdir,
            )
            yaml_file = next(Path(out_dir).glob("*.yaml"))
            content = yaml_file.read_text()
            assert "resources:" in content
            assert "cloud: kubernetes" in content
            assert "setup:" in content
            assert "run:" in content

    def test_yaml_contains_eval_command(self, single_run_config):
        """Generated YAML should contain the Hydra eval command with all benchmarks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                single_run_config,
                output_dir=tmpdir,
            )
            yaml_file = next(Path(out_dir).glob("*.yaml"))
            content = yaml_file.read_text()
            assert "python -m verl_inf_evolve.sol_eval.sol_eval" in content
            assert "eval.run_name=test-run" in content
            assert "eval.remote_sync_path=s3://bucket/run" in content
            assert "eval.checkpoints=[5]" in content
            assert "eval.benchmarks=[aime]" in content
            assert "solver.model.path=Qwen/Qwen3-8B" in content
            assert "solver.rollout.response_length=16384" in content
            assert "trainer.n_gpus_per_node=2" in content

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
            out_dir = generate_skypilot_tasks(run_config, output_dir=tmpdir)
            yaml_file = next(Path(out_dir).glob("*.yaml"))
            content = yaml_file.read_text()
            assert "eval.result_detail=full" in content

    def test_yaml_contains_all_benchmarks(self, sample_run_config):
        """Each YAML should reference all benchmarks for that checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                sample_run_config,
                output_dir=tmpdir,
            )
            yaml_file = next(Path(out_dir).glob("*ckpt0.yaml"))
            content = yaml_file.read_text()
            assert "eval.benchmarks=[supergpqa_2000,aime]" in content

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
            out_dir = generate_skypilot_tasks(run_config, output_dir=tmpdir)
            yaml_file = next(Path(out_dir).glob("*.yaml"))
            content = yaml_file.read_text()
            assert "eval.remote_eval_base=hf://datasets/alice/eval-results/results" in content

    def test_yaml_contains_metadata_comments(self, single_run_config):
        """Generated YAML should contain metadata comments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                single_run_config,
                output_dir=tmpdir,
            )
            yaml_file = next(Path(out_dir).glob("*.yaml"))
            content = yaml_file.read_text()
            assert "# Run: test-run" in content
            assert "# Checkpoint: global_step_5" in content
            assert "# Benchmarks: aime" in content

    def test_yaml_contains_envs(self, single_run_config):
        """Generated YAML should contain environment variable definitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                single_run_config,
                output_dir=tmpdir,
            )
            yaml_file = next(Path(out_dir).glob("*.yaml"))
            content = yaml_file.read_text()
            assert "WANDB_API_KEY:" in content
            assert "WANDB_ENTITY:" in content

    def test_default_gpu_from_tp_size(self, single_run_config):
        """n_gpus should default to run_config.tp_size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                single_run_config,
                output_dir=tmpdir,
            )
            yaml_file = next(Path(out_dir).glob("*.yaml"))
            content = yaml_file.read_text()
            # tp_size=2, so should have H100:2
            assert "H100:2" in content

    def test_custom_n_gpus(self, single_run_config):
        """Custom n_gpus should override tp_size default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                single_run_config,
                output_dir=tmpdir,
                n_gpus=8,
            )
            yaml_file = next(Path(out_dir).glob("*.yaml"))
            content = yaml_file.read_text()
            assert "H100:8" in content

    def test_custom_accelerator(self, single_run_config):
        """Custom accelerator type should be used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                single_run_config,
                output_dir=tmpdir,
                accelerator="A100",
                n_gpus=4,
            )
            yaml_file = next(Path(out_dir).glob("*.yaml"))
            content = yaml_file.read_text()
            assert "A100:4" in content

    def test_default_output_dir_has_timestamp(self, single_run_config):
        """When output_dir is None, should create verl_inf_evolve/sol_eval/skypilot_tasks_{timestamp}/."""
        out_dir = generate_skypilot_tasks(single_run_config)
        try:
            assert out_dir.startswith("verl_inf_evolve/sol_eval/skypilot_tasks_")
            assert Path(out_dir).is_dir()
            yamls = list(Path(out_dir).glob("*.yaml"))
            assert len(yamls) == 1
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_missing_template_raises_error(self, single_run_config):
        """Non-existent template name should raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                generate_skypilot_tasks(
                    single_run_config,
                    template_name="nonexistent_xyz",
                    output_dir=tmpdir,
                )

    def test_no_unresolved_placeholders(self, sample_run_config):
        """Generated YAMLs should have no remaining {placeholder} strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                sample_run_config,
                output_dir=tmpdir,
            )
            for yaml_file in Path(out_dir).glob("*.yaml"):
                content = yaml_file.read_text()
                # Match {word} but not ${word} or ${{word}} (bash vars)
                remaining = re.findall(r"(?<!\$)(?<!\$\$)\{[a-z_]+\}", content)
                assert remaining == [], f"Unresolved placeholders in {yaml_file.name}: {remaining}"

    def test_job_name_uses_slug(self, sample_run_config):
        """SkyPilot job name should use slugified run name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                sample_run_config,
                output_dir=tmpdir,
            )
            yaml_file = next(Path(out_dir).glob("*ckpt0.yaml"))
            content = yaml_file.read_text()
            # FW-Alr_2e-6 -> fw-alr-2e-6
            assert "name: eval-fw-alr-2e-6-ckpt0" in content

    def test_wandb_entity(self, single_run_config):
        """Custom wandb_entity should appear in envs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                single_run_config,
                output_dir=tmpdir,
                wandb_entity="my-team",
            )
            yaml_file = next(Path(out_dir).glob("*.yaml"))
            content = yaml_file.read_text()
            assert 'WANDB_ENTITY: "my-team"' in content

    def test_rclone_config_in_setup(self, single_run_config):
        """Setup section should include rclone configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = generate_skypilot_tasks(
                single_run_config,
                output_dir=tmpdir,
            )
            yaml_file = next(Path(out_dir).glob("*.yaml"))
            content = yaml_file.read_text()
            assert "rclone" in content
            assert "r2" in content
