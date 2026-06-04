"""Unit tests for benchmark evaluation utility functions.

Tests cover:
- group_and_compute_accuracy
- redistribute_shortfall
- filter_questions_by_difficulty
- format_dev_json
- output JSON schema validation
- edge cases (empty, single bucket, n=1)
"""

from __future__ import annotations

import pytest

from verl_inf_evolve.utils.benchmarks.benchmark_eval_utils import (
    DEFAULT_TARGET_DISTRIBUTION,
    filter_questions_by_difficulty,
    format_dev_json,
    group_and_compute_accuracy,
    redistribute_shortfall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results_list(qid_scores: dict[str, list[float]]) -> list[dict]:
    """Build a flat results_list from {qid: [score, ...]}."""
    out = []
    for qid, scores in qid_scores.items():
        for s in scores:
            out.append({"question_id": qid, "score": s})
    return out


def _make_accuracy_results(qid_accuracy: dict[str, float], n: int = 8) -> dict[str, dict]:
    """Build group_and_compute_accuracy-style output from {qid: accuracy}."""
    out = {}
    for qid, acc in qid_accuracy.items():
        n_correct = round(acc * n)
        out[qid] = {"n_correct": n_correct, "n_total": n, "accuracy": n_correct / n}
    return out


def _make_source_results(qids: list[str], accuracy_map: dict[str, float] | None = None) -> list[dict]:
    """Build source_results array for format_dev_json."""
    out = []
    for qid in qids:
        out.append({
            "question_id": qid,
            "question_text": f"Question {qid}",
            "choices": ["A", "B", "C", "D"],
            "ground_truth": "A",
            "domain": "science",
            "accuracy": accuracy_map.get(qid, 0.5) if accuracy_map else 0.5,
        })
    return out


# ---------------------------------------------------------------------------
# group_and_compute_accuracy tests
# ---------------------------------------------------------------------------

class TestGroupAndComputeAccuracy:
    def test_basic_accuracy(self):
        """5 correct out of 8 should give accuracy 0.625."""
        results = _make_results_list({
            "q1": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        })
        acc = group_and_compute_accuracy(results)
        assert acc["q1"]["n_correct"] == 5
        assert acc["q1"]["n_total"] == 8
        assert acc["q1"]["accuracy"] == 0.625

    def test_multiple_questions(self):
        """Multiple questions should each be aggregated independently."""
        results = _make_results_list({
            "q1": [1.0, 0.0],
            "q2": [1.0, 1.0, 1.0],
        })
        acc = group_and_compute_accuracy(results)
        assert acc["q1"]["accuracy"] == 0.5
        assert acc["q2"]["accuracy"] == 1.0
        assert acc["q2"]["n_correct"] == 3
        assert acc["q2"]["n_total"] == 3

    def test_all_correct(self):
        results = _make_results_list({"q1": [1.0] * 8})
        acc = group_and_compute_accuracy(results)
        assert acc["q1"]["accuracy"] == 1.0

    def test_all_wrong(self):
        results = _make_results_list({"q1": [0.0] * 8})
        acc = group_and_compute_accuracy(results)
        assert acc["q1"]["accuracy"] == 0.0
        assert acc["q1"]["n_correct"] == 0
        assert acc["q1"]["n_total"] == 8

    def test_none_scores_excluded(self):
        """None scores should be excluded from n_total."""
        results = [
            {"question_id": "q1", "score": 1.0},
            {"question_id": "q1", "score": None},
            {"question_id": "q1", "score": 0.0},
        ]
        acc = group_and_compute_accuracy(results)
        assert acc["q1"]["n_correct"] == 1
        assert acc["q1"]["n_total"] == 2
        assert acc["q1"]["accuracy"] == 0.5

    def test_empty_results(self):
        acc = group_and_compute_accuracy([])
        assert acc == {}

    def test_single_sample(self):
        """n=1 rollout: single score per question."""
        results = _make_results_list({"q1": [1.0], "q2": [0.0]})
        acc = group_and_compute_accuracy(results)
        assert acc["q1"]["accuracy"] == 1.0
        assert acc["q2"]["accuracy"] == 0.0
        assert acc["q1"]["n_total"] == 1

    def test_all_none_scores(self):
        """All None scores should give accuracy 0.0 and n_total 0."""
        results = [
            {"question_id": "q1", "score": None},
            {"question_id": "q1", "score": None},
        ]
        acc = group_and_compute_accuracy(results)
        assert acc["q1"]["n_total"] == 0
        assert acc["q1"]["accuracy"] == 0.0

    def test_question_id_coerced_to_string(self):
        """Integer question IDs should be coerced to strings."""
        results = [{"question_id": 42, "score": 1.0}]
        acc = group_and_compute_accuracy(results)
        assert "42" in acc


# ---------------------------------------------------------------------------
# output_json_schema tests
# ---------------------------------------------------------------------------

class TestOutputJsonSchema:
    """Validate that a sample output JSON matches the FR-3 schema."""

    def _make_sample_output(self) -> dict:
        """Build a sample output JSON as benchmark_eval.py would produce."""
        results_list = _make_results_list({
            "q1": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "q2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })
        acc = group_and_compute_accuracy(results_list)

        results = []
        for qid, info in acc.items():
            results.append({
                "question_id": qid,
                "question_text": f"What is {qid}?",
                "choices": ["A", "B", "C", "D"],
                "ground_truth": "A",
                "domain": "science",
                "n_correct": info["n_correct"],
                "n_total": info["n_total"],
                "accuracy": info["accuracy"],
                "responses": [
                    {"extracted_answer": "A", "correct": True},
                ],
            })

        overall_acc = sum(r["accuracy"] for r in results) / len(results)
        return {
            "metadata": {
                "checkpoint": "model-v1",
                "benchmark": "supergpqa",
                "n_samples": 8,
                "total_questions": len(results),
                "overall_accuracy": overall_acc,
                "timestamp": "2026-02-21T12:00:00Z",
            },
            "results": results,
        }

    def test_metadata_fields(self):
        output = self._make_sample_output()
        meta = output["metadata"]
        required_fields = {"checkpoint", "benchmark", "n_samples", "total_questions",
                           "overall_accuracy", "timestamp"}
        assert required_fields.issubset(meta.keys())

    def test_results_fields(self):
        output = self._make_sample_output()
        for r in output["results"]:
            required = {"question_id", "question_text", "choices", "ground_truth",
                        "domain", "n_correct", "n_total", "accuracy", "responses"}
            assert required.issubset(r.keys())

    def test_accuracy_consistency(self):
        """accuracy should equal n_correct / n_total."""
        output = self._make_sample_output()
        for r in output["results"]:
            expected = r["n_correct"] / r["n_total"] if r["n_total"] > 0 else 0.0
            assert r["accuracy"] == expected

    def test_types(self):
        output = self._make_sample_output()
        meta = output["metadata"]
        assert isinstance(meta["n_samples"], int)
        assert isinstance(meta["total_questions"], int)
        assert isinstance(meta["overall_accuracy"], float)
        for r in output["results"]:
            assert isinstance(r["n_correct"], int)
            assert isinstance(r["n_total"], int)
            assert isinstance(r["accuracy"], float)
            assert isinstance(r["responses"], list)


# ---------------------------------------------------------------------------
# redistribute_shortfall tests
# ---------------------------------------------------------------------------

class TestRedistributeShortfall:
    def test_proportional_redistribution(self):
        """Shortfall should be split proportionally to surplus."""
        buckets = {
            0.5: [f"q{i}" for i in range(20)],
            1.0: [f"r{i}" for i in range(10)],
        }
        over_represented = [(0.5, 20), (1.0, 10)]
        extra = redistribute_shortfall(buckets, 9, over_represented)
        # 20/(20+10)*9 = 6, 10/(20+10)*9 = 3
        assert extra[0.5] == 6
        assert extra[1.0] == 3
        assert sum(extra.values()) == 9

    def test_zero_shortfall(self):
        extra = redistribute_shortfall({}, 0, [(0.5, 10)])
        assert extra == {}

    def test_empty_over_represented(self):
        extra = redistribute_shortfall({}, 5, [])
        assert extra == {}

    def test_single_bucket(self):
        buckets = {0.5: [f"q{i}" for i in range(10)]}
        extra = redistribute_shortfall(buckets, 5, [(0.5, 10)])
        assert extra == {0.5: 5}

    def test_capped_by_surplus(self):
        """Extra allocation should not exceed available surplus."""
        buckets = {0.5: ["q1", "q2"]}
        extra = redistribute_shortfall(buckets, 10, [(0.5, 2)])
        assert extra[0.5] == 2


# ---------------------------------------------------------------------------
# filter_questions_by_difficulty tests
# ---------------------------------------------------------------------------

class TestFilterQuestionsByDifficulty:
    def _make_large_results(self, per_bucket: int = 30) -> dict[str, dict]:
        """Create results with per_bucket questions in each of the 9 accuracy buckets."""
        results = {}
        bucket_accs = sorted(DEFAULT_TARGET_DISTRIBUTION.keys())
        for i, acc in enumerate(bucket_accs):
            for j in range(per_bucket):
                qid = f"q_{i}_{j}"
                n_correct = round(acc * 8)
                results[qid] = {
                    "n_correct": n_correct,
                    "n_total": 8,
                    "accuracy": n_correct / 8,
                }
        return results

    def test_total_count(self):
        """Should select exactly 150 questions."""
        results = self._make_large_results(per_bucket=30)
        selected = filter_questions_by_difficulty(results, total=150)
        assert len(selected) == 150

    def test_bucket_proportions(self):
        """Each bucket should have approximately the target count."""
        results = self._make_large_results(per_bucket=30)
        selected = filter_questions_by_difficulty(results, total=150)

        # Count per bucket
        bucket_counts: dict[float, int] = {acc: 0 for acc in DEFAULT_TARGET_DISTRIBUTION}
        for qid in selected:
            acc = results[qid]["accuracy"]
            bucket_counts[acc] += 1

        # Verify proportions match targets
        for acc, proportion in DEFAULT_TARGET_DISTRIBUTION.items():
            expected = round(150 * proportion)
            # Allow +-1 for rounding (last bucket gets remainder)
            assert abs(bucket_counts[acc] - expected) <= 1, (
                f"Bucket {acc}: expected ~{expected}, got {bucket_counts[acc]}"
            )

    def test_deterministic_with_seed(self):
        """Same seed should produce same selection."""
        results = self._make_large_results(per_bucket=30)
        sel1 = filter_questions_by_difficulty(results, seed=42)
        sel2 = filter_questions_by_difficulty(results, seed=42)
        assert sel1 == sel2

    def test_different_seed_different_result(self):
        """Different seeds should produce different selections."""
        results = self._make_large_results(per_bucket=30)
        sel1 = filter_questions_by_difficulty(results, seed=42)
        sel2 = filter_questions_by_difficulty(results, seed=99)
        assert sel1 != sel2

    def test_shortfall_redistribution(self):
        """When a bucket has fewer questions than target, extras come from other buckets."""
        results = {}
        bucket_accs = sorted(DEFAULT_TARGET_DISTRIBUTION.keys())
        for i, acc in enumerate(bucket_accs):
            # Give the first bucket only 2 questions (target ~15), rest have plenty
            count = 2 if i == 0 else 30
            for j in range(count):
                qid = f"q_{i}_{j}"
                n_correct = round(acc * 8)
                results[qid] = {
                    "n_correct": n_correct,
                    "n_total": 8,
                    "accuracy": n_correct / 8,
                }

        selected = filter_questions_by_difficulty(results, total=150)
        # Total should still be 150 (shortfall redistributed)
        assert len(selected) == 150

    def test_all_same_accuracy(self):
        """All questions have same accuracy — only one bucket is populated."""
        results = {f"q{i}": {"n_correct": 4, "n_total": 8, "accuracy": 0.5} for i in range(200)}
        selected = filter_questions_by_difficulty(results, total=150)
        # Should get at most the target for 0.5 bucket, plus any redistribution
        # Total will be less than 150 since only one bucket has questions
        assert len(selected) <= 200
        # All selected should be from 0.5 bucket
        for qid in selected:
            assert results[qid]["accuracy"] == 0.5

    def test_only_one_question_per_bucket(self):
        """Only 1 question per bucket — should take all available (9 total)."""
        results = {}
        for i, acc in enumerate(sorted(DEFAULT_TARGET_DISTRIBUTION.keys())):
            n_correct = round(acc * 8)
            results[f"q{i}"] = {"n_correct": n_correct, "n_total": 8, "accuracy": n_correct / 8}

        selected = filter_questions_by_difficulty(results, total=150)
        # Can only select 9 questions total
        assert len(selected) == 9

    def test_empty_results(self):
        selected = filter_questions_by_difficulty({}, total=150)
        assert selected == []

    def test_custom_distribution(self):
        """Custom distribution should be respected."""
        custom = {0.0: 0.5, 1.0: 0.5}
        results = {}
        for i in range(50):
            results[f"hard_{i}"] = {"n_correct": 0, "n_total": 8, "accuracy": 0.0}
            results[f"easy_{i}"] = {"n_correct": 8, "n_total": 8, "accuracy": 1.0}

        selected = filter_questions_by_difficulty(results, target_distribution=custom, total=20)
        assert len(selected) == 20

        hard_count = sum(1 for qid in selected if qid.startswith("hard_"))
        easy_count = sum(1 for qid in selected if qid.startswith("easy_"))
        assert hard_count == 10
        assert easy_count == 10


# ---------------------------------------------------------------------------
# format_dev_json tests
# ---------------------------------------------------------------------------

class TestFormatDevJson:
    def test_output_schema(self):
        """Output records should have all 7 required dev.json fields."""
        source = _make_source_results(["q1", "q2"])
        dev = format_dev_json(["q1", "q2"], source)
        required = {"question_id", "question_text", "choices", "domain",
                     "difficulty", "ground_truth", "data_source"}
        for record in dev:
            assert required == set(record.keys())

    def test_filters_to_selected_ids(self):
        """Only selected question IDs should appear in output."""
        source = _make_source_results(["q1", "q2", "q3"])
        dev = format_dev_json(["q1", "q3"], source)
        ids = {r["question_id"] for r in dev}
        assert ids == {"q1", "q3"}

    def test_difficulty_mapping(self):
        """Accuracy should map to difficulty labels correctly."""
        accuracy_map = {"q1": 0.0, "q2": 0.25, "q3": 0.375, "q4": 0.625, "q5": 0.75, "q6": 1.0}
        source = _make_source_results(list(accuracy_map.keys()), accuracy_map)
        dev = format_dev_json(list(accuracy_map.keys()), source)
        diff = {r["question_id"]: r["difficulty"] for r in dev}
        assert diff["q1"] == "hard"     # 0.0 <= 0.25
        assert diff["q2"] == "hard"     # 0.25 <= 0.25
        assert diff["q3"] == "medium"   # 0.375 <= 0.625
        assert diff["q4"] == "medium"   # 0.625 <= 0.625
        assert diff["q5"] == "easy"     # 0.75 > 0.625
        assert diff["q6"] == "easy"     # 1.0 > 0.625

    def test_custom_data_source(self):
        source = _make_source_results(["q1"])
        dev = format_dev_json(["q1"], source, data_source="custom_bench")
        assert dev[0]["data_source"] == "custom_bench"

    def test_missing_qid_in_source_skipped(self):
        """If a selected ID isn't in source_results, it's silently skipped."""
        source = _make_source_results(["q1"])
        dev = format_dev_json(["q1", "q_missing"], source)
        assert len(dev) == 1
        assert dev[0]["question_id"] == "q1"

    def test_empty_inputs(self):
        dev = format_dev_json([], [])
        assert dev == []

    def test_preserves_order(self):
        """Output should preserve the order of filtered_question_ids."""
        source = _make_source_results(["q3", "q1", "q2"])
        dev = format_dev_json(["q2", "q3", "q1"], source)
        assert [r["question_id"] for r in dev] == ["q2", "q3", "q1"]

    def test_missing_optional_fields_default(self):
        """Missing optional fields in source should get default values."""
        source = [{"question_id": "q1", "accuracy": 0.5}]
        dev = format_dev_json(["q1"], source)
        assert dev[0]["question_text"] == ""
        assert dev[0]["choices"] == []
        assert dev[0]["domain"] == ""
        assert dev[0]["ground_truth"] == ""
