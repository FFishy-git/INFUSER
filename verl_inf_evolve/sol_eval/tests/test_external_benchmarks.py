from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from verl_inf_evolve.sol_eval.external_benchmarks import (
    canonical_external_benchmark,
    is_external_benchmark,
    resolve_external_model_path,
    run_external_benchmark,
    select_livecodebench_runner_model,
)


def test_external_benchmark_aliases_cover_humaneval_and_lcb() -> None:
    assert canonical_external_benchmark("humaneval") == "humaneval_plus_external"
    assert canonical_external_benchmark("humaneval_full") == "humaneval_plus_external"
    assert canonical_external_benchmark("livecodebench") == "livecodebench_v5_external"
    assert canonical_external_benchmark("livecodebench_v5") == "livecodebench_v5_external"
    assert is_external_benchmark("humaneval_plus")
    assert not is_external_benchmark("aime2024")


def test_resolve_external_model_path_accepts_hf_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_external_model_path(str(model_dir)) == str(model_dir)


def test_resolve_external_model_path_accepts_checkpoint_dir(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "global_step_5" / "solver" / "huggingface"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_external_model_path(str(tmp_path / "global_step_5")) == str(ckpt_dir)


def test_select_livecodebench_runner_model_uses_model_type_detection() -> None:
    assert select_livecodebench_runner_model("Qwen/Qwen3-8B-Base") == "GenericBase"
    assert select_livecodebench_runner_model("Qwen/Qwen3-8B-Instruct") == "GenericChat"


def test_run_external_benchmark_normalizes_humaneval_opencompass_output(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    run_dir = (
        tmp_path
        / "humaneval_plus_external"
        / str(model_dir).replace("/", "__")
        / "20260406_000000"
    )
    summary_dir = run_dir / "summary"
    results_dir = run_dir / "results" / "eval-model"
    summary_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (summary_dir / "summary_20260406_000000.csv").write_text(
        "dataset,version,metric,mode,eval-model\n",
        encoding="utf-8",
    )
    (results_dir / "humaneval_plus.json").write_text(
        """
        {
          "humaneval_plus_pass@1": 37.5,
          "humaneval_plus_pass@10": 62.5,
          "details": {
            "0": {
              "reference": "HumanEval/0",
              "base_result": "success",
              "plus_result": "success",
              "is_correct": true
            }
          }
        }
        """,
        encoding="utf-8",
    )

    with patch(
        "verl_inf_evolve.sol_eval.external_benchmarks._EXTERNAL_CACHE_ROOT",
        tmp_path,
    ), patch(
        "verl_inf_evolve.sol_eval.external_benchmarks.launch_opencompass",
        return_value=0,
    ):
        result = run_external_benchmark(
            benchmark="humaneval",
            model_path_or_ckpt_dir=str(model_dir),
            n_samples=8,
            temperature=0.7,
            top_p=1.0,
            top_k=-1,
            max_generation_length=4096,
            tp_size=1,
            result_detail="scores",
        )

    # OC's headline pass@1/pass@10 are preserved under oc_reported_pass_at_k
    # (raw OC metrics, computed across all n_samples by OC itself).
    assert result["metrics"]["oc_reported_pass_at_k"]["1"] == 0.375
    assert result["metrics"]["oc_reported_pass_at_k"]["10"] == 0.625
    # The single dumped sample becomes a 1-element answer_scores list; with n=1,
    # accuracy_strict and our recomputed pass@1 both reflect that one sample.
    assert result["questions"][0]["question_id"] == "HumanEval/0"
    assert result["questions"][0]["answer_scores"] == [1.0]
    assert result["metrics"]["accuracy_strict"] == 1.0
    assert result["metrics"]["pass_at_k_strict"]["1"] == 1.0
    assert result["metrics"]["external_runner"] == "opencompass"


def test_run_external_benchmark_normalizes_livecodebench_opencompass_output(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    run_dir = (
        tmp_path
        / "livecodebench_v5_external"
        / str(model_dir).replace("/", "__")
        / "20260406_000000"
    )
    summary_dir = run_dir / "summary"
    results_dir = run_dir / "results" / "eval-model"
    summary_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (summary_dir / "summary_20260406_000000.csv").write_text(
        "dataset,version,metric,mode,eval-model\n",
        encoding="utf-8",
    )
    (results_dir / "lcb_code_generation.json").write_text(
        """
        {
          "pass@1": 44.0,
          "details": [
            {
              "correct": true,
              "final_metadata": {"question_id": "LCB/0"},
              "eval_result": "passed"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with patch(
        "verl_inf_evolve.sol_eval.external_benchmarks._EXTERNAL_CACHE_ROOT",
        tmp_path,
    ), patch(
        "verl_inf_evolve.sol_eval.external_benchmarks.launch_opencompass",
        return_value=0,
    ):
        result = run_external_benchmark(
            benchmark="livecodebench",
            model_path_or_ckpt_dir=str(model_dir),
            n_samples=4,
            temperature=0.7,
            top_p=1.0,
            top_k=-1,
            max_generation_length=4096,
            tp_size=1,
            result_detail="scores",
        )

    # OC's pass@1 is preserved under oc_reported_pass_at_k.
    assert result["metrics"]["oc_reported_pass_at_k"]["1"] == 0.44
    assert result["questions"][0]["question_id"] == "LCB/0"
    assert result["questions"][0]["answer_scores"] == [1.0]
    assert result["metrics"]["accuracy_strict"] == 1.0
    assert result["metrics"]["pass_at_k_strict"]["1"] == 1.0
    assert result["metrics"]["external_runner_dataset"] == "livecodebench"


def test_run_external_benchmark_preserves_all_n_samples_humaneval(
    tmp_path: Path,
) -> None:
    """Regression: with n>1 OC dumps list-valued is_correct/prediction; the
    normalizer must keep all N samples per question (was collapsing to sample 0
    via _first_value before)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    run_dir = (
        tmp_path
        / "humaneval_plus_external"
        / str(model_dir).replace("/", "__")
        / "20260429_000000"
    )
    summary_dir = run_dir / "summary"
    results_dir = run_dir / "results" / "eval-model"
    summary_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (summary_dir / "summary_20260429_000000.csv").write_text(
        "dataset,version,metric,mode,eval-model\n",
        encoding="utf-8",
    )
    # OC n=4 fixture: per-sample lists for is_correct / prediction / *_result.
    (results_dir / "humaneval_plus.json").write_text(
        """
        {
          "humaneval_plus_pass@1 (4 runs average)": 50.0,
          "humaneval_plus_pass@4 (4 runs average)": 100.0,
          "details": {
            "0": {
              "reference": "HumanEval/0",
              "is_correct": [true, false, true, false],
              "prediction": ["code_a", "code_b", "code_c", "code_d"],
              "base_result": ["pass", "fail", "pass", "fail"],
              "plus_result": ["pass", "fail", "pass", "fail"]
            }
          }
        }
        """,
        encoding="utf-8",
    )

    with patch(
        "verl_inf_evolve.sol_eval.external_benchmarks._EXTERNAL_CACHE_ROOT",
        tmp_path,
    ), patch(
        "verl_inf_evolve.sol_eval.external_benchmarks.launch_opencompass",
        return_value=0,
    ):
        result = run_external_benchmark(
            benchmark="humaneval",
            model_path_or_ckpt_dir=str(model_dir),
            n_samples=4,
            temperature=0.7,
            top_p=1.0,
            top_k=-1,
            max_generation_length=4096,
            tp_size=1,
            result_detail="full",
        )

    q = result["questions"][0]
    # All 4 samples preserved per question:
    assert q["answer_scores"] == [1.0, 0.0, 1.0, 0.0]
    assert q["sampled_answers"] == ["code_a", "code_b", "code_c", "code_d"]
    assert q["metadata"]["base_results"] == ["pass", "fail", "pass", "fail"]
    assert q["metadata"]["plus_results"] == ["pass", "fail", "pass", "fail"]
    # Recomputed pass@k for powers of 2 up to n=4:
    pak = result["metrics"]["pass_at_k_strict"]
    assert pak["1"] == 0.5      # 2/4 correct
    assert pak["2"] == 5.0 / 6  # 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert pak["4"] == 1.0      # at least one correct in all 4
    # OC's headline pass@k still surfaces under oc_reported_pass_at_k:
    assert result["metrics"]["oc_reported_pass_at_k"]["1"] == 0.5
    assert result["metrics"]["oc_reported_pass_at_k"]["4"] == 1.0


def test_run_external_benchmark_preserves_all_n_samples_lcb(
    tmp_path: Path,
) -> None:
    """Regression for LCB: per-sample list preservation + pass@k for k>1."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    run_dir = (
        tmp_path
        / "livecodebench_v5_external"
        / str(model_dir).replace("/", "__")
        / "20260429_000000"
    )
    summary_dir = run_dir / "summary"
    results_dir = run_dir / "results" / "eval-model"
    summary_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (summary_dir / "summary_20260429_000000.csv").write_text(
        "dataset,version,metric,mode,eval-model\n",
        encoding="utf-8",
    )
    (results_dir / "lcb_code_generation.json").write_text(
        """
        {
          "pass@1": 25.0,
          "pass@2": 50.0,
          "details": [
            {
              "correct": [true, false, false, false],
              "prediction": ["c1", "c2", "c3", "c4"],
              "final_metadata": [
                {"question_id": "LCB/0"},
                {"question_id": "LCB/0"},
                {"question_id": "LCB/0"},
                {"question_id": "LCB/0"}
              ],
              "eval_result": ["passed", "failed", "failed", "failed"]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with patch(
        "verl_inf_evolve.sol_eval.external_benchmarks._EXTERNAL_CACHE_ROOT",
        tmp_path,
    ), patch(
        "verl_inf_evolve.sol_eval.external_benchmarks.launch_opencompass",
        return_value=0,
    ):
        result = run_external_benchmark(
            benchmark="livecodebench",
            model_path_or_ckpt_dir=str(model_dir),
            n_samples=4,
            temperature=0.7,
            top_p=1.0,
            top_k=-1,
            max_generation_length=4096,
            tp_size=1,
            result_detail="full",
        )

    q = result["questions"][0]
    assert q["question_id"] == "LCB/0"
    assert q["answer_scores"] == [1.0, 0.0, 0.0, 0.0]
    assert q["sampled_answers"] == ["c1", "c2", "c3", "c4"]
    assert q["metadata"]["eval_results"] == ["passed", "failed", "failed", "failed"]
    pak = result["metrics"]["pass_at_k_strict"]
    assert pak["1"] == 0.25
    assert pak["2"] == 0.5      # 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
    assert pak["4"] == 1.0
    # OC reported only pass@1, pass@2 — both round-trip:
    assert result["metrics"]["oc_reported_pass_at_k"] == {"1": 0.25, "2": 0.5}
