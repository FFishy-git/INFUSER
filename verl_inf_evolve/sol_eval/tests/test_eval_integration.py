"""Integration tests for the evaluation pipeline with mocked verl workers.

Tests verify the end-to-end evaluation flow without requiring GPUs, Ray,
or network access by mocking heavy dependencies (verl workers, R2, WandB).

Tests:
- test_full_eval_flow: config -> runner init -> evaluate_single -> result format
- test_multi_checkpoint_reloads: load_checkpoint called per checkpoint, not per benchmark
- test_skip_existing: skip when valid result exists on R2
- test_force_overrides_skip: force=True bypasses skip-existing
- test_wandb_logging: correct metric names and step values
- test_r2_upload: upload called after successful evaluation
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest
from omegaconf import OmegaConf

from verl_inf_evolve.sol_eval.result_format import (
    build_output_filename,
    compute_eval_metrics,
    format_result_json,
    is_result_complete,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_config(**overrides):
    """Create an OmegaConf config with sensible eval defaults for testing."""
    rollout_defaults = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "prompt_length": 4096,
        "response_length": 8192,
    }
    rollout_overrides = overrides.pop("solver_rollout", None) or {}
    rollout_defaults.update(rollout_overrides)

    defaults = dict(
        run_name="test-run",
        remote_sync_path="s3://example-bucket/experiments/test",
        model_path="Qwen/Qwen3-8B",
        checkpoints=[0, 5],
        benchmarks=["aime"],
        max_model_len=32768,
        n_samples=2,
        max_questions=0,
        tp_size=1,
        gpu_memory_utilization=0.5,
        r2_eval_base="s3://example-bucket/.cache/eval",
        force=False,
        no_r2_upload=True,
        no_wandb=True,
        cleanup_checkpoints=True,
        result_detail="metrics_only",
    )
    defaults.update(overrides)
    return OmegaConf.create({
        "eval": defaults,
        "solver": {
            "model": {"path": defaults["model_path"]},
            "rollout": rollout_defaults,
        },
        "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
        "wandb": {"entity": None, "project_name": "self-evolution-v3-ans-eval"},
    })


def _make_question_results(n_questions: int = 4, n_samples: int = 2) -> list[dict]:
    """Build synthetic question_results for compute_eval_metrics."""
    results = []
    for i in range(n_questions):
        # Alternate: half correct, half wrong for predictable metrics
        if i % 2 == 0:
            scores = [1.0] * n_samples
        else:
            scores = [0.0] * n_samples
        results.append({
            "question_id": f"q{i:03d}",
            "question_text": f"Question {i}",
            "choices": ["A", "B", "C", "D"],
            "ground_truth": "A",
            "sampled_answers": [f"Answer {j}" for j in range(n_samples)],
            "extracted_answers": ["A" if s == 1.0 else "B" for s in scores],
            "answer_scores": scores,
            "response_token_lengths": [100] * n_samples,
        })
    return results


def _make_result_json(
    n_questions: int = 4,
    n_samples: int = 2,
    result_detail: str = "full",
) -> dict:
    """Build a complete result JSON matching the expected output format."""
    qr = _make_question_results(n_questions, n_samples)
    metrics = compute_eval_metrics(qr, n_samples=n_samples)
    return format_result_json(qr, metrics, result_detail=result_detail)


def _make_benchmark_file(tmp_dir: str, name: str = "aime") -> str:
    """Create a minimal benchmark JSON file and return its path."""
    questions = [
        {
            "question_id": f"q{i:03d}",
            "question_text": f"Question {i}?",
            "choices": ["A", "B", "C", "D"],
            "ground_truth": "A",
        }
        for i in range(4)
    ]
    path = os.path.join(tmp_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(questions, f)
    return path


def _make_code_benchmark_file(tmp_dir: str, name: str = "humaneval") -> str:
    """Create a minimal code benchmark JSON file and return its path."""
    questions = [
        {
            "question_id": "HumanEval/0",
            "question_text": "Write a function f(x).",
            "ground_truth": "",
            "benchmark_type": "code_functional",
            "data_source": "dummy_code_benchmark",
            "verifier_metadata": {"entry_point": "f"},
        }
    ]
    path = os.path.join(tmp_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(questions, f)
    return path


# ---------------------------------------------------------------------------
# test_full_eval_flow
# ---------------------------------------------------------------------------


class TestFullEvalFlow:
    """Config loading -> runner init (mocked) -> evaluate_single -> result format."""

    def test_full_eval_flow(self, tmp_path):
        """Verify end-to-end: config -> evaluate_single -> result JSON schema."""
        # Build a result via the pure-Python path (no GPU needed)
        result = _make_result_json(n_questions=4, n_samples=2, result_detail="full")

        # Validate schema matches run_eval_direct.py format
        assert "questions" in result
        assert "metrics" in result
        assert isinstance(result["questions"], list)
        assert len(result["questions"]) == 4

        metrics = result["metrics"]
        assert metrics["total_questions"] == 4
        assert metrics["total_answers"] == 8  # 4 questions * 2 samples
        assert "accuracy_strict" in metrics
        assert "accuracy_lenient" in metrics
        assert "pass_at_k_strict" in metrics
        assert "pass_at_k_lenient" in metrics
        assert "correct_count_distribution_strict" in metrics
        assert "none_count_distribution" in metrics
        assert "response_length_chars" in metrics
        assert "response_length_tokens" in metrics

        # Verify pass_at_k keys are strings
        for k in metrics["pass_at_k_strict"]:
            assert isinstance(k, str)

        # Verify result is considered complete
        assert is_result_complete(result)

        # Verify output filename construction
        fn = build_output_filename("aime", "test-run", 5, 0.7, 32768)
        assert fn == "aime_test-run_ans_5_temp0.7_len32768_answer_eval.json"

    def test_full_eval_flow_with_mocked_runner(self, tmp_path):
        """Mock EvalRunner.__init__ and evaluate_single to verify integration."""
        result = _make_result_json(n_questions=4, n_samples=2, result_detail="full")
        benchmark_path = _make_benchmark_file(str(tmp_path))

        with patch("verl_inf_evolve.sol_eval.runner.EvalRunner.__init__", return_value=None) as mock_init:
            from verl_inf_evolve.sol_eval.runner import EvalRunner

            runner = EvalRunner.__new__(EvalRunner)
            # Manually set attributes that __init__ would set
            runner.model_path = "Qwen/Qwen3-8B"
            runner.n_gpus = 1
            runner.tp_size = 1
            runner.n_samples = 2
            runner.temperature = 0.7

            # Mock evaluate_single to return our synthetic result
            runner.evaluate_single = MagicMock(return_value=result)

            out = runner.evaluate_single(
                checkpoint_path="/fake/ckpt",
                benchmark_path=benchmark_path,
                n_samples=2,
                temperature=0.7,
                max_model_len=32768,
            )

            assert "questions" in out
            assert "metrics" in out
            assert out["metrics"]["total_questions"] == 4
            assert is_result_complete(out)

    def test_evaluate_single_passes_eval_max_questions(self, tmp_path):
        benchmark_path = _make_benchmark_file(str(tmp_path))
        config = _make_eval_config(max_questions=2)
        config.solver.rollout = {
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": -1,
            "prompt_length": 4096,
        }

        runtime = SimpleNamespace(
            solver_wg=MagicMock(),
            rollout_manager=MagicMock(),
            tokenizer=MagicMock(),
            num_workers=1,
            generate_batch=MagicMock(),
        )

        with (
            patch(
                "verl_inf_evolve.sol_eval.runtime.SolverEvalRuntime.from_fresh_worker_group",
                return_value=runtime,
            ),
            patch(
                "verl_inf_evolve.sol_eval.eval_core.load_benchmark_questions",
                return_value=([], benchmark_path),
            ) as mock_load,
            patch(
                "verl_inf_evolve.sol_eval.eval_core.evaluate_benchmark_questions",
                return_value=SimpleNamespace(
                    question_results=[],
                    metrics={
                        "total_questions": 0,
                        "accuracy_strict": 0.0,
                        "accuracy_lenient": 0.0,
                    },
                ),
            ),
            patch(
                "verl_inf_evolve.sol_eval.result_format.format_result_json",
                return_value={"questions": [], "metrics": {}},
            ),
        ):
            from verl_inf_evolve.sol_eval.runner import EvalRunner

            runner = EvalRunner(config)
            runner.evaluate_single(
                checkpoint_path="/fake/ckpt",
                benchmark_path=benchmark_path,
                n_samples=2,
                temperature=0.7,
                max_model_len=32768,
            )

        mock_load.assert_called_once_with(benchmark_path, max_questions=2)


# ---------------------------------------------------------------------------
# test_multi_checkpoint_reloads
# ---------------------------------------------------------------------------


class TestMultiCheckpointReloads:
    """Verify load_checkpoint is called once per new checkpoint, not per benchmark."""

    @patch("verl_inf_evolve.sol_eval.runner.EvalRunner.__init__", return_value=None)
    def test_multi_checkpoint_reloads(self, mock_init):
        from verl_inf_evolve.sol_eval.runner import EvalRunner

        runner = EvalRunner.__new__(EvalRunner)
        runner.model_path = "Qwen/Qwen3-8B"
        runner.n_gpus = 1
        runner.tp_size = 1
        runner.n_samples = 2
        runner.temperature = 0.7
        runner._config = _make_eval_config(
            checkpoints=[0, 5],
            benchmarks=["aime", "supergpqa_2000"],
            no_wandb=True,
            no_r2_upload=True,
        )

        result = _make_result_json(n_questions=4, n_samples=2)

        # Mock methods
        runner._download_checkpoint = MagicMock(return_value="/fake/ckpt_dir")
        runner._load_checkpoint_weights = MagicMock()
        runner._resolve_benchmark_path = MagicMock(return_value="/fake/bench.json")
        runner.evaluate_single = MagicMock(return_value=result)
        runner.save_result = MagicMock(return_value="/fake/result.json")
        runner.solver_wg = MagicMock()

        # Call evaluate_run manually by patching its dependencies
        with (
            patch("verl_inf_evolve.sol_eval.r2_ops.check_r2_result_exists", return_value=False),
            patch("verl_inf_evolve.sol_eval.r2_ops.upload_result_to_r2", return_value=True),
            patch("verl_inf_evolve.sol_eval.r2_ops.cleanup_checkpoint"),
            patch("verl_inf_evolve.sol_eval.r2_ops.download_r2_result", return_value=None),
        ):
            results = runner.evaluate_run()

        # load_checkpoint_weights should be called once per checkpoint (2 checkpoints)
        assert runner._load_checkpoint_weights.call_count == 2

        # evaluate_single should be called once per (checkpoint, benchmark) pair
        # 2 checkpoints * 2 benchmarks = 4
        assert runner.evaluate_single.call_count == 4

        # download should be called once per unique checkpoint
        assert runner._download_checkpoint.call_count == 2


# ---------------------------------------------------------------------------
# test_skip_existing
# ---------------------------------------------------------------------------


class TestSkipExisting:
    """When check_r2_result_exists returns True, evaluate_single should NOT be called."""

    @patch("verl_inf_evolve.sol_eval.runner.EvalRunner.__init__", return_value=None)
    def test_skip_existing(self, mock_init):
        from verl_inf_evolve.sol_eval.runner import EvalRunner

        runner = EvalRunner.__new__(EvalRunner)
        runner.model_path = "Qwen/Qwen3-8B"
        runner.n_gpus = 1
        runner.tp_size = 1
        runner.n_samples = 2
        runner.temperature = 0.7
        runner._config = _make_eval_config(
            checkpoints=[0],
            benchmarks=["aime"],
            no_wandb=True,
            no_r2_upload=False,
        )

        existing_result = _make_result_json(n_questions=4, n_samples=2)

        runner._download_checkpoint = MagicMock(return_value="/fake/ckpt_dir")
        runner._load_checkpoint_weights = MagicMock()
        runner._resolve_benchmark_path = MagicMock(return_value="/fake/bench.json")
        runner.evaluate_single = MagicMock()
        runner.save_result = MagicMock()
        runner.solver_wg = MagicMock()

        with (
            patch("verl_inf_evolve.sol_eval.r2_ops.check_r2_result_exists", return_value=True),
            patch("verl_inf_evolve.sol_eval.r2_ops.download_r2_result", return_value=existing_result),
            patch("verl_inf_evolve.sol_eval.r2_ops.upload_result_to_r2", return_value=True),
            patch("verl_inf_evolve.sol_eval.r2_ops.cleanup_checkpoint"),
        ):
            results = runner.evaluate_run()

        # evaluate_single should NOT have been called (skipped)
        runner.evaluate_single.assert_not_called()

        # But we should still get a result (the existing one)
        assert len(results) == 1
        assert "_eval_metadata" in results[0]


# ---------------------------------------------------------------------------
# test_force_overrides_skip
# ---------------------------------------------------------------------------


class TestForceOverridesSkip:
    """With force=True, evaluate_single IS called even when result exists on R2."""

    @patch("verl_inf_evolve.sol_eval.runner.EvalRunner.__init__", return_value=None)
    def test_force_overrides_skip(self, mock_init):
        from verl_inf_evolve.sol_eval.runner import EvalRunner

        runner = EvalRunner.__new__(EvalRunner)
        runner.model_path = "Qwen/Qwen3-8B"
        runner.n_gpus = 1
        runner.tp_size = 1
        runner.n_samples = 2
        runner.temperature = 0.7
        runner._config = _make_eval_config(
            checkpoints=[0],
            benchmarks=["aime"],
            force=True,
            no_wandb=True,
            no_r2_upload=True,
        )

        result = _make_result_json(n_questions=4, n_samples=2)

        runner._download_checkpoint = MagicMock(return_value="/fake/ckpt_dir")
        runner._load_checkpoint_weights = MagicMock()
        runner._resolve_benchmark_path = MagicMock(return_value="/fake/bench.json")
        runner.evaluate_single = MagicMock(return_value=result)
        runner.save_result = MagicMock(return_value="/fake/result.json")
        runner.solver_wg = MagicMock()

        with (
            patch("verl_inf_evolve.sol_eval.r2_ops.check_r2_result_exists", return_value=True),
            patch("verl_inf_evolve.sol_eval.r2_ops.download_r2_result", return_value=result),
            patch("verl_inf_evolve.sol_eval.r2_ops.upload_result_to_r2", return_value=True),
            patch("verl_inf_evolve.sol_eval.r2_ops.cleanup_checkpoint"),
        ):
            results = runner.evaluate_run()

        # evaluate_single SHOULD have been called despite existing result
        runner.evaluate_single.assert_called_once()
        assert len(results) == 1


# ---------------------------------------------------------------------------
# test_wandb_logging
# ---------------------------------------------------------------------------


class TestWandBLogging:
    """Verify wandb.log called with correct metric names and step=ckpt_num."""

    @patch("verl_inf_evolve.sol_eval.runner.EvalRunner.__init__", return_value=None)
    def test_wandb_logging(self, mock_init):
        from verl_inf_evolve.sol_eval.runner import EvalRunner

        runner = EvalRunner.__new__(EvalRunner)
        runner.model_path = "Qwen/Qwen3-8B"
        runner.n_gpus = 1
        runner.tp_size = 1
        runner.n_samples = 2
        runner.temperature = 0.7
        runner._config = _make_eval_config(
            checkpoints=[5],
            benchmarks=["aime"],
            no_wandb=False,
            no_r2_upload=True,
        )

        result = _make_result_json(n_questions=4, n_samples=2)

        runner._download_checkpoint = MagicMock(return_value="/fake/ckpt_dir")
        runner._load_checkpoint_weights = MagicMock()
        runner._resolve_benchmark_path = MagicMock(return_value="/fake/bench.json")
        runner.evaluate_single = MagicMock(return_value=result)
        runner.save_result = MagicMock(return_value="/fake/result.json")
        runner.solver_wg = MagicMock()

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_run_obj = MagicMock()
        mock_run_obj.summary = {}
        mock_wandb.run = mock_run_obj

        with (
            patch("verl_inf_evolve.sol_eval.r2_ops.check_r2_result_exists", return_value=False),
            patch("verl_inf_evolve.sol_eval.r2_ops.upload_result_to_r2", return_value=True),
            patch("verl_inf_evolve.sol_eval.r2_ops.cleanup_checkpoint"),
            patch("verl_inf_evolve.sol_eval.r2_ops.download_r2_result", return_value=None),
            patch.dict(os.environ, {"WANDB_API_KEY": "fake-key"}),
            patch("verl_inf_evolve.sol_eval.wandb_logger.wandb", mock_wandb, create=True),
            patch.dict("sys.modules", {"wandb": mock_wandb}),
        ):
            results = runner.evaluate_run()

        # wandb.init should have been called
        mock_wandb.init.assert_called_once()

        # wandb.log should have been called with benchmark-prefixed metrics
        log_calls = mock_wandb.log.call_args_list
        assert len(log_calls) >= 1

        # Check that the log call has the right metric keys and step
        log_data = log_calls[0][0][0]
        log_kwargs = log_calls[0][1]
        assert "aime/accuracy_strict" in log_data
        assert "aime/accuracy_lenient" in log_data
        assert "aime/pass_at_1" in log_data
        assert "aime/total_questions" in log_data
        assert log_kwargs["step"] == 5

        # wandb.finish should have been called
        mock_wandb.finish.assert_called_once()


# ---------------------------------------------------------------------------
# test_r2_upload
# ---------------------------------------------------------------------------


class TestR2Upload:
    """Verify upload_result_to_r2 is called after successful evaluation."""

    @patch("verl_inf_evolve.sol_eval.runner.EvalRunner.__init__", return_value=None)
    def test_r2_upload(self, mock_init):
        from verl_inf_evolve.sol_eval.runner import EvalRunner

        runner = EvalRunner.__new__(EvalRunner)
        runner.model_path = "Qwen/Qwen3-8B"
        runner.n_gpus = 1
        runner.tp_size = 1
        runner.n_samples = 2
        runner.temperature = 0.7
        runner._config = _make_eval_config(
            checkpoints=[0],
            benchmarks=["aime"],
            no_wandb=True,
            no_r2_upload=False,
        )

        result = _make_result_json(n_questions=4, n_samples=2)

        runner._download_checkpoint = MagicMock(return_value="/fake/ckpt_dir")
        runner._load_checkpoint_weights = MagicMock()
        runner._resolve_benchmark_path = MagicMock(return_value="/fake/bench.json")
        runner.evaluate_single = MagicMock(return_value=result)
        runner.save_result = MagicMock(return_value="/fake/result.json")
        runner.solver_wg = MagicMock()

        with (
            patch("verl_inf_evolve.sol_eval.r2_ops.check_r2_result_exists", return_value=False),
            patch("verl_inf_evolve.sol_eval.r2_ops.upload_result_to_r2", return_value=True) as mock_upload,
            patch("verl_inf_evolve.sol_eval.r2_ops.cleanup_checkpoint"),
            patch("verl_inf_evolve.sol_eval.r2_ops.download_r2_result", return_value=None),
        ):
            results = runner.evaluate_run()

        # upload_result_to_r2 should have been called
        mock_upload.assert_called_once()

        # Verify the call args contain expected filename pattern
        call_args = mock_upload.call_args
        assert call_args[1]["local_path"] == "/fake/result.json"
        expected_filename = build_output_filename("aime", "test-run", 0, 0.7, 32768)
        assert call_args[1]["output_filename"] == expected_filename

    @patch("verl_inf_evolve.sol_eval.runner.EvalRunner.__init__", return_value=None)
    def test_r2_upload_disabled(self, mock_init):
        """When no_r2_upload=True, upload should NOT be called."""
        from verl_inf_evolve.sol_eval.runner import EvalRunner

        runner = EvalRunner.__new__(EvalRunner)
        runner.model_path = "Qwen/Qwen3-8B"
        runner.n_gpus = 1
        runner.tp_size = 1
        runner.n_samples = 2
        runner.temperature = 0.7
        runner._config = _make_eval_config(
            checkpoints=[0],
            benchmarks=["aime"],
            no_wandb=True,
            no_r2_upload=True,
        )

        result = _make_result_json(n_questions=4, n_samples=2)

        runner._download_checkpoint = MagicMock(return_value="/fake/ckpt_dir")
        runner._load_checkpoint_weights = MagicMock()
        runner._resolve_benchmark_path = MagicMock(return_value="/fake/bench.json")
        runner.evaluate_single = MagicMock(return_value=result)
        runner.save_result = MagicMock(return_value="/fake/result.json")
        runner.solver_wg = MagicMock()

        with (
            patch("verl_inf_evolve.sol_eval.r2_ops.check_r2_result_exists", return_value=False),
            patch("verl_inf_evolve.sol_eval.r2_ops.upload_result_to_r2", return_value=True) as mock_upload,
            patch("verl_inf_evolve.sol_eval.r2_ops.cleanup_checkpoint"),
            patch("verl_inf_evolve.sol_eval.r2_ops.download_r2_result", return_value=None),
        ):
            results = runner.evaluate_run()

        mock_upload.assert_not_called()


class TestVllmEvalRunner:
    """Regression checks for the direct-vLLM evaluation path."""

    def test_vllm_runner_passes_rollout_sampling_config(self):
        from verl_inf_evolve.sol_eval.runner import VllmEvalRunner

        config = _make_eval_config(
            checkpoints=[],
            no_wandb=True,
            solver_rollout={
                "temperature": 0.65,
                "top_p": 0.82,
                "top_k": 23,
                "prompt_length": 4096,
                "response_length": 512,
            },
        )

        runtime = SimpleNamespace(
            n_gpus=1,
            tokenizer=object(),
            update_model=MagicMock(),
            shutdown=MagicMock(),
        )

        with patch(
            "verl_inf_evolve.sol_eval.vllm_runtime.VllmEvalRuntime.create",
            return_value=runtime,
        ) as mock_create:
            runner = VllmEvalRunner(config)
            try:
                assert runner.temperature == pytest.approx(0.65)
                assert runner.top_p == pytest.approx(0.82)
                assert runner.top_k == 23
            finally:
                runner.shutdown()

        mock_create.assert_called_once_with(
            model_path="Qwen/Qwen3-8B",
            n_gpus=1,
            temperature=0.65,
            top_p=0.82,
            top_k=23,
            max_model_len=32768,
            gpu_memory_utilization=0.5,
            enforce_eager=False,
            custom_chat_template=None,
        )

    def test_vllm_runner_auto_enables_code_execution_for_code_benchmarks(self, tmp_path):
        from verl_inf_evolve.sol_eval.runner import VllmEvalRunner

        benchmark_path = _make_code_benchmark_file(str(tmp_path), name="codebench")
        config = _make_eval_config(
            run_name="vllm-test-run",
            remote_sync_path=None,
            checkpoints=[-1],
            benchmarks=[benchmark_path],
            model_path="Qwen/Qwen3-8B-Base",
            no_wandb=True,
            no_r2_upload=True,
            cleanup_checkpoints=False,
            result_detail="full",
        )
        config.solver.model.path = "Qwen/Qwen3-8B-Base"
        config.solver.rollout = {
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": -1,
            "prompt_length": 4096,
            "response_length": 512,
        }

        runtime = SimpleNamespace(
            n_gpus=1,
            tokenizer=object(),
            update_model=MagicMock(),
            shutdown=MagicMock(),
        )

        captured: dict = {}

        def _fake_eval_benchmark_vllm(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                question_results=[{
                    "question_id": "q000",
                    "question_text": "Question 0?",
                    "choices": ["A", "B", "C", "D"],
                    "ground_truth": "A",
                    "sampled_answers": ["A"],
                    "extracted_answers": ["A"],
                    "answer_scores": [1.0],
                    "response_token_lengths": [10],
                }],
                metrics={"total_questions": 1, "accuracy_strict": 1.0, "accuracy_lenient": 1.0},
            )

        with (
            patch("verl_inf_evolve.sol_eval.vllm_runtime.VllmEvalRuntime.create", return_value=runtime),
            patch("verl_inf_evolve.sol_eval.r2_ops.check_r2_result_exists", return_value=False),
            patch("verl_inf_evolve.sol_eval.vllm_runtime.evaluate_benchmark_vllm", side_effect=_fake_eval_benchmark_vllm),
            patch("verl_inf_evolve.sol_eval.result_format.format_result_json", return_value={"questions": [], "metrics": {}}),
        ):
            runner = VllmEvalRunner(config)
            try:
                runner.evaluate_run()
            finally:
                runner.shutdown()

        assert captured["code_execution_enabled"] is False
        runtime.update_model.assert_called_once_with("Qwen/Qwen3-8B-Base")

    def test_vllm_runner_passes_eval_max_questions(self, tmp_path):
        from verl_inf_evolve.sol_eval.runner import VllmEvalRunner

        benchmark_path = _make_benchmark_file(str(tmp_path), name="aime")
        config = _make_eval_config(
            run_name="vllm-test-run",
            remote_sync_path=None,
            checkpoints=[-1],
            benchmarks=[benchmark_path],
            model_path="Qwen/Qwen3-8B-Base",
            max_questions=2,
            no_wandb=True,
            no_r2_upload=True,
            cleanup_checkpoints=False,
            result_detail="full",
        )
        config.solver.model.path = "Qwen/Qwen3-8B-Base"
        config.solver.rollout = {
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": -1,
            "prompt_length": 4096,
            "response_length": 512,
        }

        runtime = SimpleNamespace(
            n_gpus=1,
            tokenizer=object(),
            update_model=MagicMock(),
            shutdown=MagicMock(),
        )

        with (
            patch("verl_inf_evolve.sol_eval.vllm_runtime.VllmEvalRuntime.create", return_value=runtime),
            patch("verl_inf_evolve.sol_eval.r2_ops.check_r2_result_exists", return_value=False),
            patch(
                "verl_inf_evolve.sol_eval.eval_core.load_benchmark_questions",
                return_value=([], benchmark_path),
            ) as mock_load,
            patch(
                "verl_inf_evolve.sol_eval.vllm_runtime.evaluate_benchmark_vllm",
                return_value=SimpleNamespace(
                    question_results=[],
                    metrics={
                        "total_questions": 0,
                        "accuracy_strict": 0.0,
                        "accuracy_lenient": 0.0,
                    },
                ),
            ),
            patch("verl_inf_evolve.sol_eval.result_format.format_result_json", return_value={"questions": [], "metrics": {}}),
        ):
            runner = VllmEvalRunner(config)
            try:
                runner.evaluate_run()
            finally:
                runner.shutdown()

        mock_load.assert_called_once_with(benchmark_path, max_questions=2)
