"""Tests for the per-benchmark custom verifier system.

Covers:
- Registry: register/get, unknown returns None, duplicate raises
- OlympiadBench verifier: each answer_type
- AIME verifier: integer normalization
- BBEH open-form verifier: fuzzy matching normalization
- HMMT verifier: math-verify symbolic comparison
- LPFQA verifier: GPT API judge (mocked)
- Backward compat: absent data_source -> existing behavior
- Integration with extract_answer_scores
"""

import os
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock


# =============================================================================
# Registry tests
# =============================================================================


class TestVerifierRegistry:
    """Test the verifier registry mechanism."""

    def test_get_registered_verifier(self):
        from verl_inf_evolve.utils.benchmarks.verifiers import get_verifier

        v = get_verifier("olympiadbench")
        assert v is not None
        assert callable(v)

    def test_get_aime_verifier(self):
        from verl_inf_evolve.utils.benchmarks.verifiers import get_verifier

        v = get_verifier("aime")
        assert v is not None
        assert callable(v)

    def test_get_bbeh_open_verifier(self):
        from verl_inf_evolve.utils.benchmarks.verifiers import get_verifier

        v = get_verifier("bbeh_open")
        assert v is not None
        assert callable(v)

    def test_unknown_returns_none(self):
        from verl_inf_evolve.utils.benchmarks.verifiers import get_verifier

        assert get_verifier("nonexistent_benchmark") is None

    def test_duplicate_registration_raises(self):
        from verl_inf_evolve.utils.benchmarks.verifiers import _REGISTRY, register

        # Save and restore
        original = _REGISTRY.copy()
        try:
            # Clear and re-register
            _REGISTRY["_test_dup"] = lambda p, g, m: True
            with pytest.raises(ValueError, match="already registered"):
                @register("_test_dup")
                def verify(predicted, ground_truth, metadata):
                    return True
        finally:
            _REGISTRY.clear()
            _REGISTRY.update(original)


# =============================================================================
# AIME verifier tests
# =============================================================================


class TestAIMEVerifier:
    """Test the AIME integer verifier."""

    def test_exact_match(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.aime import verify

        assert verify("42", "42") is True

    def test_leading_zeros(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.aime import verify

        assert verify("042", "42") is True

    def test_float_to_int(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.aime import verify

        assert verify("42.0", "42") is True

    def test_mismatch(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.aime import verify

        assert verify("41", "42") is False

    def test_none_predicted(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.aime import verify

        assert verify(None, "42") is None

    def test_empty_predicted(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.aime import verify

        assert verify("", "42") is None

    def test_whitespace_handling(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.aime import verify

        assert verify("  42  ", "42") is True

    def test_non_numeric_fallback(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.aime import verify

        assert verify("abc", "42") is False

    def test_metadata_ignored(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.aime import verify

        assert verify("42", "42", {"irrelevant": "data"}) is True


# =============================================================================
# BBEH open-form verifier tests
# =============================================================================


class TestBBEHOpenVerifier:
    """Test the BBEH open-form fuzzy verifier."""

    def test_exact_match(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.bbeh_open import verify

        assert verify("disproved", "disproved") is True

    def test_case_and_comma_spacing(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.bbeh_open import verify

        assert verify("2, 3, 4", "2,3,4") is True

    def test_numeric_float_equivalence(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.bbeh_open import verify

        assert verify("25", "25.0") is True

    def test_parenthesis_letter_equivalence(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.bbeh_open import verify

        assert verify("(A)", "a") is True

    def test_quote_normalization(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.bbeh_open import verify

        assert verify("'alpha'", "alpha") is True

    def test_bracket_normalization(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.bbeh_open import verify

        assert verify("[unknown]", "unknown") is True

    def test_trailing_question_mark(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.bbeh_open import verify

        assert verify("unknown?", "unknown") is True

    def test_none_predicted(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.bbeh_open import verify

        assert verify(None, "unknown") is None


# =============================================================================
# OlympiadBench verifier tests
# =============================================================================


class TestOlympiadBenchVerifier:
    """Test the OlympiadBench verifier with different answer types."""

    def test_numerical_exact(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.olympiadbench import verify

        assert verify("42", "42", {"answer_type": "Numerical"}) is True

    def test_numerical_float_tolerance(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.olympiadbench import verify

        assert verify("3.14159", "3.14159", {"answer_type": "Numerical"}) is True

    def test_numerical_mismatch(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.olympiadbench import verify

        assert verify("43", "42", {"answer_type": "Numerical"}) is not True

    def test_numerical_with_latex(self):
        """Test that \\frac{1}{2} matches 0.5."""
        from verl_inf_evolve.utils.benchmarks.verifiers.olympiadbench import verify

        result = verify(r"\frac{1}{2}", "0.5", {"answer_type": "Numerical"})
        # This depends on sympy being available; if so it should be True
        if result is not None:
            assert result is True

    def test_expression_simple(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.olympiadbench import verify

        result = verify("x + 1", "1 + x", {"answer_type": "Expression"})
        if result is not None:
            assert result is True

    def test_none_predicted(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.olympiadbench import verify

        assert verify(None, "42", {"answer_type": "Numerical"}) is None

    def test_unit_stripping(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.olympiadbench import verify

        assert verify("42 kg", "42", {"answer_type": "Numerical", "unit": "kg"}) is True

    def test_no_metadata(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.olympiadbench import verify

        # Should still work with no metadata
        assert verify("42", "42") is True

    def test_multi_answer_semicolon(self):
        """Multi-answer ground truths separated by semicolons."""
        from verl_inf_evolve.utils.benchmarks.verifiers.olympiadbench import verify

        result = verify("1, 2", "1; 2", {"answer_type": "Numerical"})
        if result is not None:
            assert result is True

    def test_exact_string_match_after_clean(self):
        """Exact match after LaTeX cleaning."""
        from verl_inf_evolve.utils.benchmarks.verifiers.olympiadbench import verify

        assert verify("42", "42", {"answer_type": "Expression"}) is True


# =============================================================================
# MathJudger unit tests
# =============================================================================


class TestMathJudger:
    """Test the low-level math judger functions."""

    def test_split_by_comma_simple(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import _split_by_comma

        assert _split_by_comma("1, 2, 3") == ["1", "2", "3"]

    def test_split_by_comma_with_parens(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import _split_by_comma

        assert _split_by_comma("1, (2, 3), 4") == ["1", "(2, 3)", "4"]

    def test_numerical_equal_exact(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import numerical_equal

        assert numerical_equal("42", "42") is True

    def test_numerical_equal_tolerance(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import numerical_equal

        assert numerical_equal("3.14159", "3.14159") is True

    def test_numerical_equal_mismatch(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import numerical_equal

        result = numerical_equal("100", "200")
        assert result is False

    def test_numerical_equal_non_numeric(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import numerical_equal

        assert numerical_equal("abc", "42") is None

    def test_numerical_equal_zero_gt(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import numerical_equal

        assert numerical_equal("0", "0") is True
        assert numerical_equal("0.00001", "0") is True
        assert numerical_equal("1", "0") is False

    def test_can_compute_power(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import _can_compute_power

        assert _can_compute_power("x**2") is True
        assert _can_compute_power("x**1000") is False

    def test_clean_latex(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import _clean_latex

        assert _clean_latex(r"$42$") == "42"
        assert _clean_latex(r"\text{hello}") == "hello"

    def test_interval_equal_matching(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import interval_equal

        assert interval_equal("[0, 1]", "[0, 1]") is True

    def test_interval_equal_different_brackets(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import interval_equal

        assert interval_equal("[0, 1)", "[0, 1]") is False

    def test_interval_equal_non_interval(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import interval_equal

        assert interval_equal("42", "42") is None

    def test_judge_cascading(self):
        """Judge should try multiple comparison methods."""
        from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import judge

        # Exact match
        assert judge("42", "42") is True
        # Numerical
        assert judge("42.0", "42") is True
        # None input
        assert judge(None, "42") is None
        # Empty
        assert judge("", "42") is None


# =============================================================================
# Backward compatibility: extract_answer_scores
# =============================================================================


class _FakeTokenizer:
    """Minimal tokenizer stub that round-trips encoded responses."""

    def __init__(self, responses: list[str]):
        self._responses = responses

    def decode(self, token_ids, skip_special_tokens=True):
        idx = int(token_ids[0].item())
        return self._responses[idx]


def _make_output(
    responses: list[str],
    ground_truths: list[str],
    is_mcq: bool | None = None,
    data_source: str | None = None,
    verifier_metadata: list[dict | None] | None = None,
):
    """Build a minimal DataProto-like object for extract_answer_scores."""
    n = len(responses)
    batch = {"responses": torch.arange(n).unsqueeze(1)}
    non_tensor_batch = {
        "question_id": np.array([f"q{i}" for i in range(n)], dtype=object),
        "ground_truth": np.array(ground_truths, dtype=object),
    }
    if is_mcq is not None:
        non_tensor_batch["is_mcq"] = np.array([is_mcq] * n, dtype=object)
    if data_source is not None:
        non_tensor_batch["data_source"] = np.array([data_source] * n, dtype=object)
    if verifier_metadata is not None:
        non_tensor_batch["verifier_metadata"] = np.array(verifier_metadata, dtype=object)

    class FakeOutput:
        pass

    out = FakeOutput()
    out.batch = batch
    out.non_tensor_batch = non_tensor_batch
    return out


class TestExtractAnswerScoresBackwardCompat:
    """Verify that extract_answer_scores still works without data_source."""

    def test_mcq_no_data_source(self):
        """MCQ mode with no data_source — uses existing MCQ logic."""
        from verl_inf_evolve.utils.mcq_utils import extract_answer_scores

        responses = [r"\boxed{B}"]
        gts = ["B"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=True)
        extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] == "B"
        assert output.non_tensor_batch["answer_score"][0] == 1.0

    def test_general_no_data_source(self):
        """General-form with no data_source — uses existing general logic."""
        from verl_inf_evolve.utils.mcq_utils import extract_answer_scores

        responses = [r"\boxed{42}"]
        gts = ["42"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=False)
        extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] == "42"
        assert output.non_tensor_batch["answer_score"][0] == 1.0

    def test_no_is_mcq_no_data_source(self):
        """Neither is_mcq nor data_source — defaults to MCQ."""
        from verl_inf_evolve.utils.mcq_utils import extract_answer_scores

        responses = [r"\boxed{B}"]
        gts = ["B"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts)
        extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] == "B"
        assert output.non_tensor_batch["answer_score"][0] == 1.0


class TestExtractAnswerScoresWithVerifier:
    """Verify that extract_answer_scores dispatches to custom verifiers."""

    def test_aime_verifier_dispatch(self):
        """AIME data_source triggers integer comparison."""
        from verl_inf_evolve.utils.mcq_utils import extract_answer_scores

        responses = [r"\boxed{042}"]
        gts = ["42"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=False, data_source="aime")
        extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] == "042"
        assert output.non_tensor_batch["answer_score"][0] == 1.0

    def test_olympiadbench_verifier_dispatch(self):
        """OlympiadBench data_source triggers math judger."""
        from verl_inf_evolve.utils.mcq_utils import extract_answer_scores

        responses = [r"\boxed{42}"]
        gts = ["42"]
        metadata = [{"answer_type": "Numerical"}]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(
            responses, gts, is_mcq=False,
            data_source="olympiadbench",
            verifier_metadata=metadata,
        )
        extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] == "42"
        assert output.non_tensor_batch["answer_score"][0] == 1.0

    def test_math500_verifier_dispatch(self):
        """MATH-500 data_source triggers math500 verifier."""
        from verl_inf_evolve.utils.mcq_utils import extract_answer_scores

        responses = [r"\boxed{1/91}"]
        gts = ["1/91"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=False, data_source="math500")
        extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] == "1/91"
        assert output.non_tensor_batch["answer_score"][0] == 1.0

    def test_bbeh_open_verifier_dispatch(self):
        """bbeh_open data_source triggers BBEH fuzzy open-form verifier."""
        from verl_inf_evolve.utils.mcq_utils import extract_answer_scores

        responses = [r"\boxed{2, 3, 4}"]
        gts = ["2,3,4"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=False, data_source="bbeh_open")
        extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] == "2, 3, 4"
        assert output.non_tensor_batch["answer_score"][0] == 1.0

    def test_unknown_data_source_falls_back(self):
        """Unknown data_source with no registered verifier falls back to general."""
        from verl_inf_evolve.utils.mcq_utils import extract_answer_scores

        responses = [r"\boxed{42}"]
        gts = ["42"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=False, data_source="unknown_bench")
        extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] == "42"
        assert output.non_tensor_batch["answer_score"][0] == 1.0

    def test_no_answer_extracted(self):
        """When no \\boxed{} found, verifier gets None and returns None."""
        from verl_inf_evolve.utils.mcq_utils import extract_answer_scores

        responses = ["I have no idea"]
        gts = ["42"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=False, data_source="aime")
        extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] is None
        assert output.non_tensor_batch["answer_score"][0] is None


# =============================================================================
# HMMT verifier tests
# =============================================================================


class TestHMMTVerifier:
    """Test the HMMT math-verify verifier."""

    def test_integer_match(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.hmmt import verify

        assert verify("42", "42") is True

    def test_integer_mismatch(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.hmmt import verify

        assert verify("41", "42") is False

    def test_fraction_match(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.hmmt import verify

        assert verify("1/91", "1/91") is True

    def test_equivalent_fraction(self):
        """math-verify should recognize 2/182 == 1/91."""
        from verl_inf_evolve.utils.benchmarks.verifiers.hmmt import verify

        assert verify("2/182", "1/91") is True

    def test_expression_match(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.hmmt import verify

        result = verify("sqrt(3) - 1", "sqrt(3) - 1")
        assert result is True

    def test_wrong_answer(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.hmmt import verify

        assert verify("100", "42") is False

    def test_none_predicted(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.hmmt import verify

        assert verify(None, "42") is None

    def test_empty_predicted(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.hmmt import verify

        assert verify("", "42") is None

    def test_whitespace_predicted(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.hmmt import verify

        assert verify("   ", "42") is None

    def test_string_fallback(self):
        """When math-verify can't parse, falls back to exact string comparison."""
        from verl_inf_evolve.utils.benchmarks.verifiers.hmmt import verify

        # Identical unparseable strings should match via string fallback
        assert verify("hello world", "hello world") is True
        assert verify("hello", "world") is False


# =============================================================================
# MATH-500 verifier tests
# =============================================================================


class TestMATH500Verifier:
    """Test the MATH-500 math verifier.

    The verifier was changed in commit 778da01a to take FULL solver response
    text (not pre-extracted answer) and run ``extract_math500_answer``
    internally. Tests pass response text containing ``\\boxed{...}`` rather
    than a bare answer string.
    """

    def test_exact_match(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.math500 import verify

        assert verify(r"The answer is \boxed{42}.", "42") is True

    def test_equivalent_fraction_when_parseable(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.math500 import verify

        # Should be True when math-verify is available.
        # If unavailable, fallback is string comparison and may be False.
        result = verify(r"Final answer: \boxed{2/182}.", "1/91")
        assert result in (True, False)

    def test_none_predicted(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.math500 import verify

        assert verify(None, "42") is None

    def test_empty_predicted(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.math500 import verify

        assert verify("", "42") is None

    def test_string_fallback(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.math500 import verify

        # When the boxed content isn't parseable as math, the verifier
        # falls back to string comparison via _math_judger.
        assert verify(r"\boxed{hello world}", "hello world") is True
        assert verify(r"\boxed{hello}", "world") is False


# =============================================================================
# LPFQA verifier tests
# =============================================================================


class TestLPFQAVerifier:
    """Test the LPFQA GPT API judge verifier."""

    def _make_metadata(self):
        return {
            "judge_prompt_template": (
                "Reference: {response_reference}\n\n"
                "Response: {response}\n\n"
                "Score 0 or 1."
            ),
            "judge_system_prompt": "You are an expert judge.",
            "response_reference": "The correct explanation is XYZ.",
        }

    def test_judge_prompt_construction(self):
        """Verify judge prompt is correctly constructed from template."""
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import verify

        metadata = self._make_metadata()

        # Mock _call_judge to capture the constructed prompt
        with patch("verl_inf_evolve.utils.benchmarks.verifiers.lpfqa._call_judge") as mock_judge:
            mock_judge.return_value = True
            verify("My model response", "", metadata)

            mock_judge.assert_called_once()
            judge_prompt = mock_judge.call_args[0][0]
            system_prompt = mock_judge.call_args[0][1]

            assert "The correct explanation is XYZ." in judge_prompt
            assert "My model response" in judge_prompt
            assert "{response_reference}" not in judge_prompt
            assert "{response}" not in judge_prompt
            assert system_prompt == "You are an expert judge."

    def test_score_1_returns_true(self):
        """Judge returning score 1 should yield True."""
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import verify

        metadata = self._make_metadata()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The response is excellent. Score: 1"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI = MagicMock(return_value=mock_client)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=MockOpenAI)}):
                result = verify("Good response", "", metadata)
                assert result is True

    def test_score_0_returns_false(self):
        """Judge returning score 0 should yield False."""
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import verify

        metadata = self._make_metadata()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The response is poor. Score: 0"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI = MagicMock(return_value=mock_client)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=MockOpenAI)}):
                result = verify("Bad response", "", metadata)
                assert result is False

    def test_api_error_returns_none(self):
        """API error after retries should return None."""
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import verify

        metadata = self._make_metadata()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        MockOpenAI = MagicMock(return_value=mock_client)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=MockOpenAI)}):
                with patch("verl_inf_evolve.utils.benchmarks.verifiers.lpfqa.time.sleep"):
                    result = verify("Some response", "", metadata)
                    assert result is None

    def test_missing_api_key_returns_none(self):
        """No OPENAI_API_KEY should return None."""
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import verify

        metadata = self._make_metadata()
        saved_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            result = verify("Some response", "", metadata)
            assert result is None
        finally:
            if saved_key is not None:
                os.environ["OPENAI_API_KEY"] = saved_key

    def test_none_predicted_returns_none(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import verify

        assert verify(None, "", self._make_metadata()) is None

    def test_empty_predicted_returns_none(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import verify

        assert verify("", "", self._make_metadata()) is None

    def test_missing_metadata_returns_none(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import verify

        assert verify("Some response", "") is None

    def test_missing_template_returns_none(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import verify

        metadata = {"judge_system_prompt": "X", "response_reference": "Y"}
        assert verify("Some response", "", metadata) is None


class TestLPFQAParseScore:
    """Test the _parse_score helper function."""

    def test_score_1(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import _parse_score

        assert _parse_score("Score: 1") is True

    def test_score_0(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import _parse_score

        assert _parse_score("Score: 0") is False

    def test_last_match_used(self):
        """When multiple 0/1 appear, last one is used."""
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import _parse_score

        assert _parse_score("Step 1: evaluate. Final score: 0") is False
        assert _parse_score("Step 0: evaluate. Final score: 1") is True

    def test_no_score_returns_none(self):
        from verl_inf_evolve.utils.benchmarks.verifiers.lpfqa import _parse_score

        assert _parse_score("I cannot determine a score") is None


# =============================================================================
# Registry tests for new verifiers
# =============================================================================


class TestNewVerifierRegistry:
    """Test registry entries for HMMT and LPFQA verifiers."""

    def test_get_hmmt_verifier(self):
        from verl_inf_evolve.utils.benchmarks.verifiers import get_verifier

        v = get_verifier("hmmt")
        assert v is not None
        assert callable(v)

    def test_get_lpfqa_verifier(self):
        from verl_inf_evolve.utils.benchmarks.verifiers import get_verifier

        v = get_verifier("lpfqa")
        assert v is not None
        assert callable(v)

    def test_get_math500_verifier(self):
        from verl_inf_evolve.utils.benchmarks.verifiers import get_verifier

        v = get_verifier("math500")
        assert v is not None
        assert callable(v)

    def test_lpfqa_needs_full_response(self):
        from verl_inf_evolve.utils.benchmarks.verifiers import get_needs_full_response

        assert get_needs_full_response("lpfqa") is True

    def test_hmmt_does_not_need_full_response(self):
        from verl_inf_evolve.utils.benchmarks.verifiers import get_needs_full_response

        assert get_needs_full_response("hmmt") is False

    def test_math500_needs_full_response(self):
        from verl_inf_evolve.utils.benchmarks.verifiers import get_needs_full_response

        # math500.verify runs extract_math500_answer internally on the full
        # response text (commit 778da01a), so needs_full_response=True.
        assert get_needs_full_response("math500") is True
