from __future__ import annotations

from verl_inf_evolve.utils.benchmarks.benchmark_scorers import score_phybench
from verl_inf_evolve.utils.benchmarks.phybench_eed import compute_eed_score
from verl_inf_evolve.utils.benchmarks.verifiers.phybench import verify


def test_compute_eed_score_exact_match_is_100():
    assert compute_eed_score("x+1", "x+1") == 100.0


def test_compute_eed_score_partial_match_is_between_0_and_100():
    score = compute_eed_score("x+1", "x+2")
    assert 0.0 < score < 100.0


def test_phybench_scorer_returns_eed_result():
    result = score_phybench(r"\[x+1\]", r"\[x+2\]")
    assert result is not None
    assert result.name == "eed"
    assert result.scale_max == 100.0
    assert 0.0 <= float(result.score) <= 100.0


def test_phybench_verifier_accepts_equivalent_expression():
    assert verify("x+1", r"\[x+1\]") is True
