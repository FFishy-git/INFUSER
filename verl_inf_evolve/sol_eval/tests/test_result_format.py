"""Unit tests for verl_inf_evolve/sol_eval/result_format.py — result formatting and metrics.

Tests cover:
- build_output_filename
- compute_pass_at_k
- compute_eval_metrics (accuracy, pass@k, distributions, lengths)
- format_result_json
- is_result_complete
"""

from __future__ import annotations

from math import comb

import pytest

from verl_inf_evolve.sol_eval.result_format import (
    build_output_filename,
    compute_eval_metrics,
    compute_pass_at_k,
    format_result_json,
    is_result_complete,
)


# ---------------------------------------------------------------------------
# build_output_filename tests
# ---------------------------------------------------------------------------

class TestBuildOutputFilename:
    def test_standard_filename_metrics_only(self):
        result = build_output_filename(
            benchmark="supergpqa_2000",
            run_name="FW-Alr_2e-6",
            ckpt_num=18,
            temperature=0.7,
            max_model_len=8192,
        )
        assert result == "supergpqa_2000_FW-Alr_2e-6_ans_18_temp0.7_len8192_answer_eval.json"

    def test_full_filename_has_suffix(self):
        result = build_output_filename(
            benchmark="supergpqa_2000",
            run_name="FW-Alr_2e-6",
            ckpt_num=18,
            temperature=0.7,
            max_model_len=8192,
            result_detail="full",
        )
        assert result == "supergpqa_2000_FW-Alr_2e-6_ans_18_temp0.7_len8192_full_answer_eval.json"

    def test_integer_temperature(self):
        result = build_output_filename("bench", "run", 0, 1.0, 32768)
        assert "temp1.0" in result

    def test_zero_checkpoint(self):
        result = build_output_filename("bench", "run", 0, 0.7, 32768)
        assert "_ans_0_" in result

    def test_benchmark_path_normalized(self):
        """Full benchmark paths should be normalized to stem name."""
        result = build_output_filename(
            benchmark=".cache/data/preprocessed/benchmarks/self_evolution_train_eval.json",
            run_name="my_run",
            ckpt_num=5,
            temperature=0.7,
            max_model_len=16384,
        )
        assert result == "self_evolution_train_eval_my_run_ans_5_temp0.7_len16384_answer_eval.json"
        assert "/" not in result


# ---------------------------------------------------------------------------
# compute_pass_at_k tests
# ---------------------------------------------------------------------------

class TestComputePassAtK:
    def test_zero_correct(self):
        """0 correct out of any n should give pass@k = 0."""
        assert compute_pass_at_k(8, 0, 1) == 0.0
        assert compute_pass_at_k(8, 0, 4) == 0.0

    def test_all_correct(self):
        """All correct should give pass@k = 1."""
        assert compute_pass_at_k(8, 8, 1) == 1.0
        assert compute_pass_at_k(8, 8, 4) == 1.0

    def test_n_less_than_k(self):
        """If n < k, pass@k should be 0."""
        assert compute_pass_at_k(2, 1, 4) == 0.0

    def test_partial_correct_formula(self):
        """Verify against manual computation: n=8, c=2, k=1."""
        # pass@1 = 1 - C(6,1)/C(8,1) = 1 - 6/8 = 0.25
        result = compute_pass_at_k(8, 2, 1)
        assert abs(result - 0.25) < 1e-10

    def test_partial_correct_k2(self):
        """n=8, c=2, k=2: pass@2 = 1 - C(6,2)/C(8,2) = 1 - 15/28."""
        expected = 1.0 - comb(6, 2) / comb(8, 2)
        result = compute_pass_at_k(8, 2, 2)
        assert abs(result - expected) < 1e-10

    def test_n_correct_equals_n_minus_k_plus_1(self):
        """Edge: when n-c < k, pass@k = 1.0."""
        # n=8, c=7, k=2 -> n-c=1 < k=2 -> 1.0
        assert compute_pass_at_k(8, 7, 2) == 1.0


# ---------------------------------------------------------------------------
# compute_eval_metrics tests
# ---------------------------------------------------------------------------

def _make_question(
    qid: str,
    scores: list[float | None],
    answers: list[str] | None = None,
    token_lengths: list[int] | None = None,
) -> dict:
    """Build a question result dict for testing."""
    q: dict = {
        "question_id": qid,
        "answer_scores": scores,
        "sampled_answers": answers or [f"answer_{i}" for i in range(len(scores))],
    }
    if token_lengths is not None:
        q["response_token_lengths"] = token_lengths
    return q


class TestComputeEvalMetrics:
    """Tests for compute_eval_metrics with known inputs."""

    @pytest.fixture()
    def ten_questions(self):
        """10 questions, each with 4 samples. Known score distribution."""
        questions = []
        for i in range(10):
            if i < 3:
                # 3 questions: all correct
                scores = [1.0, 1.0, 1.0, 1.0]
            elif i < 6:
                # 3 questions: half correct
                scores = [1.0, 1.0, 0.0, 0.0]
            elif i < 8:
                # 2 questions: all wrong
                scores = [0.0, 0.0, 0.0, 0.0]
            else:
                # 2 questions: some None
                scores = [1.0, None, 0.0, None]
            questions.append(_make_question(f"q{i}", scores))
        return questions

    def test_total_counts(self, ten_questions):
        metrics = compute_eval_metrics(ten_questions, n_samples=4)
        assert metrics["total_questions"] == 10
        assert metrics["total_answers"] == 40  # 10 * 4

    def test_valid_and_failed(self, ten_questions):
        metrics = compute_eval_metrics(ten_questions, n_samples=4)
        # 8 questions * 4 = 32 valid + 2 questions with 2 valid each = 36 valid
        assert metrics["valid_answers"] == 36
        assert metrics["failed_extractions"] == 4  # 2 questions * 2 Nones

    def test_correct_count(self, ten_questions):
        metrics = compute_eval_metrics(ten_questions, n_samples=4)
        # 3*4 + 3*2 + 2*0 + 2*1 = 12 + 6 + 0 + 2 = 20
        assert metrics["correct_answers"] == 20

    def test_accuracy_strict(self, ten_questions):
        metrics = compute_eval_metrics(ten_questions, n_samples=4)
        # strict = correct / total = 20 / 40 = 0.5
        assert metrics["accuracy_strict"] == 0.5

    def test_accuracy_lenient(self, ten_questions):
        metrics = compute_eval_metrics(ten_questions, n_samples=4)
        # lenient = correct / valid = 20 / 36
        expected = 20 / 36
        assert abs(metrics["accuracy_lenient"] - expected) < 1e-10

    def test_pass_at_k_keys(self, ten_questions):
        metrics = compute_eval_metrics(ten_questions, n_samples=4)
        # n_samples=4 -> k in [1, 2, 4]
        assert set(metrics["pass_at_k_strict"].keys()) == {"1", "2", "4"}
        assert set(metrics["pass_at_k_lenient"].keys()) == {"1", "2", "4"}

    def test_pass_at_k_strict_values(self, ten_questions):
        metrics = compute_eval_metrics(ten_questions, n_samples=4)
        # pass@1 strict: average over questions of pass@1(n=4, c=correct_for_q)
        # q0-2: c=4, pass@1=1.0 (3 qs)
        # q3-5: c=2, pass@1=1-C(2,1)/C(4,1)=1-2/4=0.5 (3 qs)
        # q6-7: c=0, pass@1=0.0 (2 qs)
        # q8-9: c=1, pass@1=1-C(3,1)/C(4,1)=1-3/4=0.25 (2 qs)
        expected_pass1 = (3 * 1.0 + 3 * 0.5 + 2 * 0.0 + 2 * 0.25) / 10
        assert abs(metrics["pass_at_k_strict"]["1"] - expected_pass1) < 1e-10

    def test_correct_count_distribution(self, ten_questions):
        metrics = compute_eval_metrics(ten_questions, n_samples=4)
        dist = metrics["correct_count_distribution_strict"]
        # Correct counts: 3 qs with 4, 3 qs with 2, 2 qs with 0, 2 qs with 1
        assert dist[4] == 3 / 10
        assert dist[2] == 3 / 10
        assert dist[0] == 2 / 10
        assert dist[1] == 2 / 10
        assert dist[3] == 0 / 10

    def test_none_count_distribution(self, ten_questions):
        metrics = compute_eval_metrics(ten_questions, n_samples=4)
        dist = metrics["none_count_distribution"]
        # 8 questions with 0 Nones, 2 questions with 2 Nones
        assert dist[0] == 8 / 10
        assert dist[2] == 2 / 10
        assert dist[1] == 0 / 10

    def test_response_length_chars(self, ten_questions):
        metrics = compute_eval_metrics(ten_questions, n_samples=4)
        # All answers are "answer_0", "answer_1", etc. (8 chars each)
        assert metrics["response_length_chars"]["min"] == 8
        assert metrics["response_length_chars"]["max"] == 8
        assert metrics["response_length_chars"]["mean"] == 8.0

    def test_response_length_tokens(self):
        """Token lengths should be computed when provided."""
        questions = [
            _make_question("q0", [1.0, 0.0], token_lengths=[10, 20]),
            _make_question("q1", [1.0, 1.0], token_lengths=[30, 40]),
        ]
        metrics = compute_eval_metrics(questions, n_samples=2)
        assert metrics["response_length_tokens"]["min"] == 10
        assert metrics["response_length_tokens"]["max"] == 40
        assert metrics["response_length_tokens"]["mean"] == 25.0

    def test_empty_questions(self):
        metrics = compute_eval_metrics([])
        assert metrics["total_questions"] == 0
        assert metrics["total_answers"] == 0
        assert metrics["accuracy_strict"] == 0.0
        assert metrics["pass_at_k_strict"] == {}

    def test_auto_detect_n_samples(self):
        """n_samples should be auto-detected from max answer_scores length."""
        questions = [
            _make_question("q0", [1.0, 0.0, 1.0]),
            _make_question("q1", [0.0, 1.0, 0.0]),
        ]
        metrics = compute_eval_metrics(questions)  # no n_samples
        # Should detect n_samples=3 -> k in [1, 2]
        assert "1" in metrics["pass_at_k_strict"]
        assert "2" in metrics["pass_at_k_strict"]
        assert "4" not in metrics["pass_at_k_strict"]

    def test_sample_score_metrics_use_primary_metric(self):
        questions = [
            {
                **_make_question("q0", [1.0, 0.0]),
                "sample_scores": [100.0, 50.0],
                "sample_score_name": "eed",
                "sample_score_scale_max": 100.0,
            },
            {
                **_make_question("q1", [0.0, 0.0]),
                "sample_scores": [0.0, None],
                "sample_score_name": "eed",
                "sample_score_scale_max": 100.0,
            },
        ]

        metrics = compute_eval_metrics(questions, n_samples=2)

        assert metrics["score_name"] == "eed"
        assert metrics["score_scale_max"] == 100.0
        assert metrics["eed_score_strict"] == 37.5
        assert metrics["eed_score_lenient"] == 50.0
        assert metrics["primary_metric_name"] == "eed_score_strict"
        assert metrics["primary_metric_value"] == 37.5

    def test_code_metrics_are_computed_from_sample_exec_results(self):
        questions = [
            {
                **_make_question("q0", [1.0, 0.0]),
                "sample_exec_results": [
                    {
                        "status": "passed",
                        "passed": True,
                        "candidate_compile_ok": True,
                        "candidate_runtime_ok": True,
                    },
                    {
                        "status": "compile_error",
                        "passed": False,
                        "candidate_compile_ok": False,
                        "candidate_runtime_ok": False,
                    },
                ],
            },
            {
                **_make_question("q1", [0.0, 0.0]),
                "sample_exec_results": [
                    {
                        "status": "wrong_answer",
                        "passed": False,
                        "candidate_compile_ok": True,
                        "candidate_runtime_ok": True,
                    },
                    {
                        "status": "runtime_error",
                        "passed": False,
                        "candidate_compile_ok": True,
                        "candidate_runtime_ok": False,
                    },
                ],
            },
        ]

        metrics = compute_eval_metrics(questions, n_samples=2)

        assert metrics["compile_ok_rate"] == 3 / 4
        assert metrics["runtime_ok_rate"] == 2 / 4
        assert metrics["runtime_pass_rate"] == 1 / 4
        assert metrics["code_metrics"]["status_counts"]["passed"] == 1
        assert metrics["code_metrics"]["status_counts"]["compile_error"] == 1
        assert metrics["code_metrics"]["status_counts"]["wrong_answer"] == 1
        assert metrics["code_metrics"]["status_counts"]["runtime_error"] == 1


# ---------------------------------------------------------------------------
# format_result_json tests
# ---------------------------------------------------------------------------

class TestFormatResultJson:
    def test_top_level_keys(self):
        questions = [_make_question("q0", [1.0])]
        metrics = compute_eval_metrics(questions)
        result = format_result_json(questions, metrics)
        assert set(result.keys()) == {"questions", "metrics"}

    def test_questions_omitted_by_default(self):
        questions = [
            _make_question("q0", [1.0, 0.0]),
            _make_question("q1", [0.0, 0.0]),
        ]
        metrics = compute_eval_metrics(questions)
        result = format_result_json(questions, metrics)
        assert result["questions"] == []

    def test_questions_preserved_in_full_mode(self):
        questions = [
            _make_question("q0", [1.0, 0.0]),
            _make_question("q1", [0.0, 0.0]),
        ]
        metrics = compute_eval_metrics(questions)
        result = format_result_json(questions, metrics, result_detail="full")
        assert len(result["questions"]) == 2
        assert result["questions"][0]["question_id"] == "q0"

    def test_metrics_included(self):
        questions = [_make_question("q0", [1.0])]
        metrics = compute_eval_metrics(questions)
        result = format_result_json(questions, metrics)
        assert result["metrics"]["total_questions"] == 1

    def test_scores_preserve_exec_results(self):
        questions = [
            {
                **_make_question("q0", [1.0]),
                "sample_exec_results": [
                    {
                        "status": "passed",
                        "passed": True,
                        "candidate_compile_ok": True,
                        "candidate_runtime_ok": True,
                    }
                ],
            }
        ]
        metrics = compute_eval_metrics(questions)
        result = format_result_json(questions, metrics, result_detail="scores")

        assert "sampled_answers" not in result["questions"][0]
        assert result["questions"][0]["sample_exec_results"][0]["status"] == "passed"


# ---------------------------------------------------------------------------
# is_result_complete tests
# ---------------------------------------------------------------------------

class TestIsResultComplete:
    def test_valid_result(self):
        questions = [_make_question("q0", [1.0, 0.0])]
        metrics = compute_eval_metrics(questions)
        result = format_result_json(questions, metrics)
        assert is_result_complete(result) is True

    def test_missing_metrics(self):
        assert is_result_complete({"questions": []}) is False

    def test_metrics_not_dict(self):
        assert is_result_complete({"metrics": "invalid"}) is False

    def test_zero_total_questions(self):
        result = {"metrics": {"total_questions": 0, "total_answers": 5}}
        assert is_result_complete(result) is False

    def test_zero_total_answers(self):
        result = {"metrics": {"total_questions": 5, "total_answers": 0}}
        assert is_result_complete(result) is False

    def test_empty_dict(self):
        assert is_result_complete({}) is False
