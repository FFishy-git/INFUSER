"""Tests for replay mode evaluation logic.

Covers:
- derive_gen_questions() correctly reconstructing question list from a synthetic DataProto
- compute_question_rollout_metrics() from gen_output.pt's non_tensor_batch without running
  any generation

No GPU required — all DataProto objects are mock/synthetic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from verl_inf_evolve.utils.data_utils import derive_gen_questions
from verl_inf_evolve.trainer.rollout_metrics import compute_question_rollout_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDataProto:
    """Minimal DataProto stand-in with non_tensor_batch and batch dicts."""

    def __init__(
        self,
        non_tensor_batch: dict[str, np.ndarray],
        batch: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.non_tensor_batch = non_tensor_batch
        self.batch = batch or {}
        self.meta_info = {}


def _make_gen_output(
    n_total: int = 8,
    n_parsed_ok: int = 6,
    resp_len: int = 64,
    include_reject_reason: bool = False,
) -> _FakeDataProto:
    """Build a synthetic gen_output with realistic non_tensor_batch fields.

    Args:
        n_total: Total number of samples (rows) in the gen_output.
        n_parsed_ok: How many of those samples have ``parsed_ok == True``.
        resp_len: Response sequence length for the mock response_mask.
        include_reject_reason: Whether to include ``reject_reason`` field.
    """
    parsed_ok = np.array(
        [True] * n_parsed_ok + [False] * (n_total - n_parsed_ok),
        dtype=object,
    )
    question_ids = np.array([f"q_{i}" for i in range(n_total)], dtype=object)
    question_texts = np.array(
        [f"What is {i}+{i}?" for i in range(n_total)], dtype=object
    )
    choices = np.array(
        [["A", "B", "C", "D"] for _ in range(n_total)], dtype=object
    )
    ground_truths = np.array(["A"] * n_total, dtype=object)
    doc_ids = np.array(
        [f"doc_{i % 3}" for i in range(n_total)], dtype=object
    )

    non_tensor_batch: dict[str, np.ndarray] = {
        "parsed_ok": parsed_ok,
        "question_id": question_ids,
        "question_text": question_texts,
        "choices": choices,
        "ground_truth": ground_truths,
        "doc_id": doc_ids,
    }

    if include_reject_reason:
        reasons = np.array(
            [""] * n_parsed_ok
            + ["failed_parse", "empty_question"][:n_total - n_parsed_ok]
            + [""] * max(0, n_total - n_parsed_ok - 2),
            dtype=object,
        )[:n_total]
        non_tensor_batch["reject_reason"] = reasons

    # response_mask: each token is "active" for the first 32 positions
    response_mask = torch.zeros(n_total, resp_len)
    response_mask[:, :32] = 1.0

    batch = {
        "responses": torch.zeros(n_total, resp_len, dtype=torch.long),
        "response_mask": response_mask,
    }

    return _FakeDataProto(non_tensor_batch=non_tensor_batch, batch=batch)


# ===========================================================================
# derive_gen_questions() tests
# ===========================================================================


class TestDeriveGenQuestions:
    """Verify derive_gen_questions() correctly reconstructs question list."""

    def test_returns_only_parsed_ok_questions(self):
        """Only samples where parsed_ok is True appear in the output."""
        gen_output = _make_gen_output(n_total=8, n_parsed_ok=5)
        questions = derive_gen_questions(gen_output)
        assert len(questions) == 5

    def test_all_parsed_ok(self):
        """When all samples are parsed_ok, all are returned."""
        gen_output = _make_gen_output(n_total=4, n_parsed_ok=4)
        questions = derive_gen_questions(gen_output)
        assert len(questions) == 4

    def test_none_parsed_ok(self):
        """When no samples are parsed_ok, empty list is returned."""
        gen_output = _make_gen_output(n_total=4, n_parsed_ok=0)
        questions = derive_gen_questions(gen_output)
        assert len(questions) == 0

    def test_output_has_required_keys(self):
        """Each question dict has all required keys."""
        gen_output = _make_gen_output(n_total=4, n_parsed_ok=3)
        questions = derive_gen_questions(gen_output)
        required_keys = {"question_id", "question_text", "choices", "ground_truth", "doc_id"}
        for q in questions:
            assert set(q.keys()) == required_keys

    def test_question_id_values_correct(self):
        """question_id values match the indices where parsed_ok is True."""
        gen_output = _make_gen_output(n_total=6, n_parsed_ok=4)
        questions = derive_gen_questions(gen_output)
        # parsed_ok is True for indices 0..3, so question_ids should be q_0..q_3
        ids = [q["question_id"] for q in questions]
        assert ids == ["q_0", "q_1", "q_2", "q_3"]

    def test_question_text_values_correct(self):
        """question_text values match the original gen_output data."""
        gen_output = _make_gen_output(n_total=4, n_parsed_ok=2)
        questions = derive_gen_questions(gen_output)
        assert questions[0]["question_text"] == "What is 0+0?"
        assert questions[1]["question_text"] == "What is 1+1?"

    def test_choices_preserved(self):
        """choices arrays are preserved correctly."""
        gen_output = _make_gen_output(n_total=4, n_parsed_ok=2)
        questions = derive_gen_questions(gen_output)
        for q in questions:
            assert list(q["choices"]) == ["A", "B", "C", "D"]

    def test_ground_truth_preserved(self):
        """ground_truth values are preserved correctly."""
        gen_output = _make_gen_output(n_total=4, n_parsed_ok=2)
        questions = derive_gen_questions(gen_output)
        for q in questions:
            assert q["ground_truth"] == "A"

    def test_doc_id_preserved(self):
        """doc_id values cycle correctly per the modular assignment."""
        gen_output = _make_gen_output(n_total=6, n_parsed_ok=6)
        questions = derive_gen_questions(gen_output)
        doc_ids = [q["doc_id"] for q in questions]
        assert doc_ids == ["doc_0", "doc_1", "doc_2", "doc_0", "doc_1", "doc_2"]

    def test_single_sample(self):
        """Works correctly with a single sample."""
        gen_output = _make_gen_output(n_total=1, n_parsed_ok=1)
        questions = derive_gen_questions(gen_output)
        assert len(questions) == 1
        assert questions[0]["question_id"] == "q_0"

    def test_variable_choices_count(self):
        """Works when different questions have different numbers of choices."""
        ntb = {
            "parsed_ok": np.array([True, True, True], dtype=object),
            "question_id": np.array(["q_0", "q_1", "q_2"], dtype=object),
            "question_text": np.array(["Q0?", "Q1?", "Q2?"], dtype=object),
            "choices": np.array([["A", "B"], ["A", "B", "C"], ["A", "B", "C", "D", "E"]], dtype=object),
            "ground_truth": np.array(["A", "B", "C"], dtype=object),
            "doc_id": np.array(["d0", "d1", "d2"], dtype=object),
        }
        gen_output = _FakeDataProto(non_tensor_batch=ntb)
        questions = derive_gen_questions(gen_output)
        assert len(questions) == 3
        assert list(questions[0]["choices"]) == ["A", "B"]
        assert list(questions[1]["choices"]) == ["A", "B", "C"]
        assert list(questions[2]["choices"]) == ["A", "B", "C", "D", "E"]


# ===========================================================================
# compute_question_rollout_metrics() from gen_output.pt non_tensor_batch
# ===========================================================================


class TestQuestionRolloutMetricsFromGenOutput:
    """Verify question rollout metrics can be computed from gen_output.pt fields
    without running any question generation."""

    def _run_metrics(
        self,
        n_total: int = 8,
        n_parsed_ok: int = 6,
        include_reject_reason: bool = False,
    ) -> dict[str, float]:
        """Build synthetic gen_output, derive questions, compute metrics."""
        gen_output = _make_gen_output(
            n_total=n_total,
            n_parsed_ok=n_parsed_ok,
            include_reject_reason=include_reject_reason,
        )
        gen_questions = derive_gen_questions(gen_output)
        total_samples = gen_output.batch["responses"].shape[0]
        response_mask = gen_output.batch["response_mask"]
        num_documents = len(set(gen_output.non_tensor_batch["doc_id"]))

        return compute_question_rollout_metrics(
            gen_questions=gen_questions,
            total_samples=total_samples,
            response_mask=response_mask,
            num_documents=num_documents,
            prefix="gen_question_rollout",
            reject_reasons=gen_output.non_tensor_batch.get("reject_reason"),
        )

    def test_returns_dict(self):
        """Metrics are returned as a flat dict."""
        metrics = self._run_metrics()
        assert isinstance(metrics, dict)

    def test_num_valid_questions(self):
        """num_valid_questions matches the number of parsed_ok samples."""
        metrics = self._run_metrics(n_total=8, n_parsed_ok=6)
        assert metrics["gen_question_rollout/num_valid_questions"] == 6.0

    def test_num_invalid_questions(self):
        """num_invalid_questions = total_samples - num_valid."""
        metrics = self._run_metrics(n_total=8, n_parsed_ok=6)
        assert metrics["gen_question_rollout/num_invalid_questions"] == 2.0

    def test_num_generated_questions(self):
        """num_generated_questions records total samples (before filtering)."""
        metrics = self._run_metrics(n_total=10, n_parsed_ok=7)
        assert metrics["gen_question_rollout/num_generated_questions"] == 10.0

    def test_frac_valid_questions(self):
        """frac_valid_questions = num_valid / total_samples."""
        metrics = self._run_metrics(n_total=8, n_parsed_ok=6)
        assert metrics["gen_question_rollout/frac_valid_questions"] == pytest.approx(0.75)

    def test_num_documents(self):
        """num_documents reflects unique doc_ids."""
        # doc_ids cycle mod 3, so 8 samples → 3 unique docs
        metrics = self._run_metrics(n_total=8, n_parsed_ok=6)
        assert metrics["gen_question_rollout/num_documents"] == 3.0

    def test_valid_questions_per_doc(self):
        """valid_questions_per_doc = num_valid / num_documents."""
        metrics = self._run_metrics(n_total=8, n_parsed_ok=6)
        assert metrics["gen_question_rollout/valid_questions_per_doc"] == pytest.approx(2.0)

    def test_num_choices_stats(self):
        """All questions have 4 choices → mean=4, std=0, min=4, max=4."""
        metrics = self._run_metrics(n_total=8, n_parsed_ok=6)
        assert metrics["gen_question_rollout/num_choices/mean"] == 4.0
        assert metrics["gen_question_rollout/num_choices/std"] == 0.0
        assert metrics["gen_question_rollout/num_choices/min"] == 4.0
        assert metrics["gen_question_rollout/num_choices/max"] == 4.0

    def test_question_char_len_stats(self):
        """Question text lengths produce valid statistics."""
        metrics = self._run_metrics(n_total=4, n_parsed_ok=4)
        assert metrics["gen_question_rollout/question_char_len/mean"] > 0
        assert metrics["gen_question_rollout/question_char_len/min"] > 0
        assert metrics["gen_question_rollout/question_char_len/max"] >= metrics["gen_question_rollout/question_char_len/min"]

    def test_response_length_tokens_stats(self):
        """Response token length stats computed from response_mask."""
        metrics = self._run_metrics(n_total=4, n_parsed_ok=4)
        # Each sample has 32 active tokens in our mock
        assert metrics["gen_question_rollout/response_length_tokens/mean"] == 32.0
        assert metrics["gen_question_rollout/response_length_tokens/min"] == 32.0
        assert metrics["gen_question_rollout/response_length_tokens/max"] == 32.0

    def test_same_prefix_as_regenerate_mode(self):
        """Metrics use 'gen_question_rollout' prefix — same as regenerate mode."""
        metrics = self._run_metrics()
        for key in metrics:
            assert key.startswith("gen_question_rollout/")

    def test_reject_reasons_breakdown(self):
        """When reject_reasons are provided, per-category counts are emitted."""
        metrics = self._run_metrics(n_total=8, n_parsed_ok=6, include_reject_reason=True)
        assert "gen_question_rollout/num_failed_parses" in metrics
        assert "gen_question_rollout/num_empty_questions" in metrics
        assert "gen_question_rollout/num_invalid_ground_truth" in metrics

    def test_no_reject_reasons_omits_breakdown(self):
        """Without reject_reasons, per-category counts are not in the output."""
        metrics = self._run_metrics(n_total=8, n_parsed_ok=6, include_reject_reason=False)
        assert "gen_question_rollout/num_failed_parses" not in metrics

    def test_zero_valid_questions(self):
        """Handles the edge case of zero valid questions gracefully."""
        metrics = self._run_metrics(n_total=4, n_parsed_ok=0)
        assert metrics["gen_question_rollout/num_valid_questions"] == 0.0
        assert metrics["gen_question_rollout/num_invalid_questions"] == 4.0
        assert metrics["gen_question_rollout/frac_valid_questions"] == 0.0
        assert metrics["gen_question_rollout/num_choices/mean"] == 0.0

    def test_all_valid_questions(self):
        """All samples parsed successfully."""
        metrics = self._run_metrics(n_total=4, n_parsed_ok=4)
        assert metrics["gen_question_rollout/num_valid_questions"] == 4.0
        assert metrics["gen_question_rollout/num_invalid_questions"] == 0.0
        assert metrics["gen_question_rollout/frac_valid_questions"] == 1.0

    def test_end_to_end_replay_pipeline(self):
        """Simulate the full replay pipeline: download → derive → metrics.

        This mirrors what evaluate() does in replay mode, but with synthetic data.
        """
        gen_output = _make_gen_output(n_total=16, n_parsed_ok=12, include_reject_reason=True)

        # Step 1: derive_gen_questions (equivalent to replay mode Stage 2 replacement)
        gen_questions = derive_gen_questions(gen_output)
        assert len(gen_questions) == 12

        # Step 2: compute_question_rollout_metrics (Stage 2 metrics in replay)
        total_samples = gen_output.batch["responses"].shape[0]
        metrics = compute_question_rollout_metrics(
            gen_questions=gen_questions,
            total_samples=total_samples,
            response_mask=gen_output.batch["response_mask"],
            num_documents=len(set(gen_output.non_tensor_batch["doc_id"])),
            prefix="gen_question_rollout",
            reject_reasons=gen_output.non_tensor_batch.get("reject_reason"),
        )

        # Verify key metrics
        assert metrics["gen_question_rollout/num_valid_questions"] == 12.0
        assert metrics["gen_question_rollout/num_generated_questions"] == 16.0
        assert metrics["gen_question_rollout/num_invalid_questions"] == 4.0
        assert metrics["gen_question_rollout/frac_valid_questions"] == pytest.approx(0.75)
        assert metrics["gen_question_rollout/response_length_tokens/mean"] == 32.0
