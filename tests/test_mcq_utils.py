"""Tests for MCQ answer extraction and evaluation utilities.

Covers both MCQ (single-letter) and general-form (free-text) extraction,
with focus on LaTeX wrapper stripping and nested-brace handling.
"""

import numpy as np
import pytest
import torch

from verl_inf_evolve.utils.mcq_utils import (
    _extract_last_boxed_content,
    _strip_latex_wrappers,
    extract_boxed_answer,
    extract_boxed_answer_general,
    extract_answer_scores as verl_extract_answer_scores,
    is_correct,
    is_correct_general,
)
from verl_inf_evolve.utils import mcq_utils as verl_mcq_utils


# =============================================================================
# _extract_last_boxed_content — brace-counting extraction
# =============================================================================


class TestExtractLastBoxedContent:
    """Tests for the low-level brace-counting boxed extractor."""

    def test_simple(self):
        assert _extract_last_boxed_content(r"\boxed{42}") == "42"

    def test_nested_text(self):
        assert _extract_last_boxed_content(r"\boxed{\text{disproved}}") == r"\text{disproved}"

    def test_nested_braces(self):
        assert _extract_last_boxed_content(r"\boxed{x^{2} + 1}") == "x^{2} + 1"

    def test_deeply_nested(self):
        assert _extract_last_boxed_content(r"\boxed{\textbf{\text{Yes}}}") == r"\textbf{\text{Yes}}"

    def test_last_match_wins(self):
        text = r"First \boxed{A}, then \boxed{B}"
        assert _extract_last_boxed_content(text) == "B"

    def test_last_match_nested(self):
        text = r"First \boxed{\text{wrong}}, then \boxed{\text{correct}}"
        assert _extract_last_boxed_content(text) == r"\text{correct}"

    def test_double_backslash(self):
        assert _extract_last_boxed_content("\\\\boxed{hello}") == "hello"

    def test_space_after_boxed(self):
        assert _extract_last_boxed_content(r"\boxed {spaced}") == "spaced"

    def test_empty_boxed(self):
        assert _extract_last_boxed_content(r"\boxed{}") is None

    def test_no_boxed(self):
        assert _extract_last_boxed_content("no boxed here") is None

    def test_none_input(self):
        assert _extract_last_boxed_content(None) is None

    def test_empty_input(self):
        assert _extract_last_boxed_content("") is None

    def test_unmatched_brace(self):
        assert _extract_last_boxed_content(r"\boxed{unclosed") is None


# =============================================================================
# _strip_latex_wrappers
# =============================================================================


class TestStripLatexWrappers:

    def test_text(self):
        assert _strip_latex_wrappers(r"\text{disproved}") == "disproved"

    def test_textbf(self):
        assert _strip_latex_wrappers(r"\textbf{Yes}") == "Yes"

    def test_textit(self):
        assert _strip_latex_wrappers(r"\textit{No}") == "No"

    def test_mathrm(self):
        assert _strip_latex_wrappers(r"\mathrm{proved}") == "proved"

    def test_operatorname(self):
        assert _strip_latex_wrappers(r"\operatorname{unknown}") == "unknown"

    def test_nested_wrappers(self):
        assert _strip_latex_wrappers(r"\textbf{\text{Yes}}") == "Yes"

    def test_no_wrapper(self):
        assert _strip_latex_wrappers("42") == "42"

    def test_plain_text(self):
        assert _strip_latex_wrappers("disproved") == "disproved"

    def test_partial_latex_not_stripped(self):
        # Only strip if the entire string is a wrapper
        assert _strip_latex_wrappers(r"answer is \text{x}") == r"answer is \text{x}"

    def test_whitespace_preserved_inside(self):
        assert _strip_latex_wrappers(r"\text{hello world}") == "hello world"


# =============================================================================
# extract_boxed_answer — MCQ (single letter)
# =============================================================================


class TestExtractBoxedAnswerMCQ:
    """MCQ extraction: should return a single uppercase letter."""

    def test_simple_letter(self):
        assert extract_boxed_answer(r"The answer is \boxed{B}") == "B"

    def test_lowercase_normalized(self):
        assert extract_boxed_answer(r"\boxed{c}") == "C"

    def test_text_wrapped_letter(self):
        """This was the bug: \boxed{\text{B}} should extract B."""
        assert extract_boxed_answer(r"\boxed{\text{B}}") == "B"

    def test_textbf_wrapped_letter(self):
        assert extract_boxed_answer(r"\boxed{\textbf{A}}") == "A"

    def test_last_boxed_wins(self):
        assert extract_boxed_answer(r"\boxed{A} wait, \boxed{D}") == "D"

    def test_no_boxed(self):
        assert extract_boxed_answer("The answer is B") is None

    def test_empty(self):
        assert extract_boxed_answer("") is None

    def test_none(self):
        assert extract_boxed_answer(None) is None

    def test_numeric_answer(self):
        # MCQ extractor: non-letter content returned as-is for numeric answers
        assert extract_boxed_answer(r"\boxed{42}") == "42"

    def test_dollar_boxed(self):
        """Model output with $$ delimiters."""
        assert extract_boxed_answer(r"$$\boxed{\text{C}}$$") == "C"

    def test_multiline_response(self):
        text = """Let me think about this...
First, I considered A.
But actually the answer is:
$$
\\boxed{\\text{B}}
$$"""
        assert extract_boxed_answer(text) == "B"


# =============================================================================
# extract_boxed_answer_general — free-form answers
# =============================================================================


class TestExtractBoxedAnswerGeneral:
    """General-form extraction: should return the full answer text."""

    def test_simple_number(self):
        assert extract_boxed_answer_general(r"\boxed{42}") == "42"

    def test_simple_word(self):
        assert extract_boxed_answer_general(r"\boxed{hydrogen}") == "hydrogen"

    def test_text_wrapped_disproved(self):
        """The core bug: \text{disproved} was not being stripped."""
        assert extract_boxed_answer_general(r"\boxed{\text{disproved}}") == "disproved"

    def test_text_wrapped_proved(self):
        assert extract_boxed_answer_general(r"\boxed{\text{proved}}") == "proved"

    def test_text_wrapped_unknown(self):
        assert extract_boxed_answer_general(r"\boxed{\text{unknown}}") == "unknown"

    def test_text_wrapped_yes(self):
        assert extract_boxed_answer_general(r"\boxed{\text{Yes}}") == "Yes"

    def test_text_wrapped_no(self):
        assert extract_boxed_answer_general(r"\boxed{\text{No}}") == "No"

    def test_text_wrapped_ambiguous(self):
        assert extract_boxed_answer_general(r"\boxed{\text{Ambiguous}}") == "Ambiguous"

    def test_textbf_wrapped(self):
        assert extract_boxed_answer_general(r"\boxed{\textbf{unanswerable}}") == "unanswerable"

    def test_mathrm_wrapped(self):
        assert extract_boxed_answer_general(r"\boxed{\mathrm{proved}}") == "proved"

    def test_nested_wrappers(self):
        assert extract_boxed_answer_general(r"\boxed{\textbf{\text{Yes}}}") == "Yes"

    def test_nested_braces_math(self):
        assert extract_boxed_answer_general(r"\boxed{x^{2} + 1}") == "x^{2} + 1"

    def test_multiword_answer(self):
        assert extract_boxed_answer_general(r"\boxed{orange ball}") == "orange ball"

    def test_comma_separated(self):
        assert extract_boxed_answer_general(r"\boxed{90, 2}") == "90, 2"

    def test_tuple_answer(self):
        assert extract_boxed_answer_general(r"\boxed{0,1,1}") == "0,1,1"

    def test_yes_no_tuple(self):
        assert extract_boxed_answer_general(r"\boxed{\text{no, unknown, no}}") == "no, unknown, no"

    def test_letter_tuple(self):
        assert extract_boxed_answer_general(r"\boxed{\text{B, B, B}}") == "B, B, B"

    def test_name(self):
        assert extract_boxed_answer_general(r"\boxed{\text{Lola}}") == "Lola"

    def test_last_boxed_wins(self):
        text = r"First \boxed{wrong}, then correction: \boxed{\text{disproved}}"
        assert extract_boxed_answer_general(text) == "disproved"

    def test_no_boxed(self):
        assert extract_boxed_answer_general("The answer is disproved") is None

    def test_empty(self):
        assert extract_boxed_answer_general("") is None

    def test_none(self):
        assert extract_boxed_answer_general(None) is None

    def test_dollar_boxed(self):
        assert extract_boxed_answer_general(r"$$\boxed{\text{proved}}$$") == "proved"

    def test_realistic_model_output(self):
        """Simulate a real model chain-of-thought ending."""
        text = """Based on my analysis, the finch does not shout at the mermaid.

### Final Answer

$$
\\boxed{\\text{unknown}}
$$"""
        assert extract_boxed_answer_general(text) == "unknown"

    def test_plain_number_no_latex(self):
        assert extract_boxed_answer_general(r"\boxed{7}") == "7"

    def test_n_and_m_format(self):
        assert extract_boxed_answer_general(r"\boxed{\text{1 and 8}}") == "1 and 8"


# =============================================================================
# is_correct / is_correct_general — scoring
# =============================================================================


class TestIsCorrectMCQ:

    def test_match(self):
        assert is_correct("A", "A") is True

    def test_case_insensitive(self):
        assert is_correct("a", "A") is True

    def test_mismatch(self):
        assert is_correct("B", "A") is False

    def test_none_predicted(self):
        assert is_correct(None, "A") is False


class TestIsCorrectGeneral:

    def test_exact_match(self):
        assert is_correct_general("disproved", "disproved") is True

    def test_case_insensitive(self):
        assert is_correct_general("Disproved", "disproved") is True

    def test_numeric_normalization(self):
        assert is_correct_general("42.0", "42") is True

    def test_mismatch(self):
        assert is_correct_general("proved", "disproved") is False

    def test_none_predicted(self):
        assert is_correct_general(None, "disproved") is False

    def test_whitespace(self):
        assert is_correct_general("  answer  ", "answer") is True


# =============================================================================
# End-to-end: extraction + scoring
# =============================================================================


class TestEndToEnd:
    """Verify that extraction followed by scoring gives correct results."""

    @pytest.mark.parametrize("boxed_text,ground_truth,expected_correct", [
        (r"\boxed{\text{disproved}}", "disproved", True),
        (r"\boxed{\text{proved}}", "proved", True),
        (r"\boxed{\text{unknown}}", "unknown", True),
        (r"\boxed{\text{Yes}}", "Yes", True),
        (r"\boxed{\text{No}}", "No", True),
        (r"\boxed{\text{Ambiguous}}", "Ambiguous", True),
        (r"\boxed{\text{unanswerable}}", "unanswerable", True),
        (r"\boxed{42}", "42", True),
        (r"\boxed{\text{no, unknown, no}}", "no, unknown, no", True),
        (r"\boxed{\text{0,1,1}}", "0,1,1", True),
        (r"\boxed{\text{B, B, B}}", "B, B, B", True),
        (r"\boxed{\text{90, 2}}", "90, 2", True),
        (r"\boxed{\text{orange ball}}", "orange ball", True),
        (r"\boxed{\text{Lola}}", "Lola", True),
        # Wrong answers should still score False
        (r"\boxed{\text{proved}}", "disproved", False),
        (r"\boxed{\text{Yes}}", "No", False),
    ])
    def test_general_form_e2e(self, boxed_text, ground_truth, expected_correct):
        extracted = extract_boxed_answer_general(boxed_text)
        assert extracted is not None, f"Failed to extract from: {boxed_text}"
        result = is_correct_general(extracted, ground_truth)
        assert result == expected_correct, (
            f"Expected {expected_correct} for extracted={extracted!r} vs gt={ground_truth!r}"
        )

    @pytest.mark.parametrize("boxed_text,ground_truth,expected_correct", [
        (r"\boxed{B}", "B", True),
        (r"\boxed{\text{B}}", "B", True),
        (r"\boxed{\textbf{C}}", "C", True),
        (r"\boxed{a}", "A", True),
        (r"\boxed{\text{D}}", "A", False),
    ])
    def test_mcq_e2e(self, boxed_text, ground_truth, expected_correct):
        extracted = extract_boxed_answer(boxed_text)
        assert extracted is not None, f"Failed to extract from: {boxed_text}"
        result = is_correct(extracted, ground_truth)
        assert result == expected_correct, (
            f"Expected {expected_correct} for extracted={extracted!r} vs gt={ground_truth!r}"
        )


# =============================================================================
# extract_answer_scores dispatch (verl_inf_evolve)
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
    benchmark_type: str | None = None,
):
    """Build a minimal DataProto-like object for extract_answer_scores."""
    n = len(responses)
    # Encode each response as a single token equal to its index
    batch = {"responses": torch.arange(n).unsqueeze(1)}
    non_tensor_batch = {
        "question_id": np.array([f"q{i}" for i in range(n)], dtype=object),
        "ground_truth": np.array(ground_truths, dtype=object),
    }
    if is_mcq is not None:
        non_tensor_batch["is_mcq"] = np.array([is_mcq] * n, dtype=object)
    if benchmark_type is not None:
        non_tensor_batch["benchmark_type"] = np.array([benchmark_type] * n, dtype=object)

    class FakeOutput:
        pass

    out = FakeOutput()
    out.batch = batch
    out.non_tensor_batch = non_tensor_batch
    return out


class TestExtractAnswerScoresDispatch:
    """Verify extract_answer_scores dispatches MCQ vs general-form correctly."""

    def test_mcq_dispatch(self):
        """MCQ mode: extracts single letter, matches against letter ground truth."""
        responses = [
            r"The answer is \boxed{\text{B}}",
            r"I think \boxed{C}",
        ]
        gts = ["B", "A"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=True)
        verl_extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] == "B"
        assert output.non_tensor_batch["answer_score"][0] == 1.0
        assert output.non_tensor_batch["extracted_answer"][1] == "C"
        assert output.non_tensor_batch["answer_score"][1] == 0.0

    def test_general_dispatch(self):
        """General-form mode: extracts full text, matches against free-text ground truth."""
        responses = [
            r"$$\boxed{\text{disproved}}$$",
            r"\boxed{\text{Yes}}",
            r"\boxed{42}",
        ]
        gts = ["disproved", "No", "42"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=False)
        verl_extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] == "disproved"
        assert output.non_tensor_batch["answer_score"][0] == 1.0
        assert output.non_tensor_batch["extracted_answer"][1] == "Yes"
        assert output.non_tensor_batch["answer_score"][1] == 0.0
        assert output.non_tensor_batch["extracted_answer"][2] == "42"
        assert output.non_tensor_batch["answer_score"][2] == 1.0

    def test_general_no_extraction(self):
        """General-form: no \\boxed{} → None extracted, None score."""
        responses = ["I have no idea"]
        gts = ["disproved"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=False)
        verl_extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] is None
        assert output.non_tensor_batch["answer_score"][0] is None

    def test_backward_compat_no_is_mcq(self):
        """When is_mcq is absent, falls back to MCQ behavior."""
        responses = [r"\boxed{B}"]
        gts = ["B"]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=None)  # no is_mcq field
        verl_extract_answer_scores(output, tokenizer)

        assert output.non_tensor_batch["extracted_answer"][0] == "B"
        assert output.non_tensor_batch["answer_score"][0] == 1.0

    def test_general_latex_wrapped_categories(self):
        """General-form: all BBEH answer categories that were broken before."""
        responses = [
            r"\boxed{\text{proved}}",
            r"\boxed{\text{unknown}}",
            r"\boxed{\text{Ambiguous}}",
            r"\boxed{\text{unanswerable}}",
            r"\boxed{\text{no, unknown, no}}",
            r"\boxed{\text{0,1,1}}",
            r"\boxed{\text{90, 2}}",
            r"\boxed{\text{orange ball}}",
            r"\boxed{\text{Lola}}",
        ]
        gts = [
            "proved", "unknown", "Ambiguous", "unanswerable",
            "no, unknown, no", "0,1,1", "90, 2", "orange ball", "Lola",
        ]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(responses, gts, is_mcq=False)
        verl_extract_answer_scores(output, tokenizer)

        for i in range(len(responses)):
            assert output.non_tensor_batch["answer_score"][i] == 1.0, (
                f"Failed at index {i}: extracted={output.non_tensor_batch['extracted_answer'][i]!r} "
                f"vs gt={gts[i]!r}"
            )

    def test_code_benchmark_blocked_by_default(self):
        responses = ["```python\nreturn x + 1\n```"]
        gts = [""]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(
            responses,
            gts,
            is_mcq=False,
            benchmark_type="code_functional",
        )

        with pytest.raises(ValueError, match="Code benchmark execution is disabled"):
            verl_extract_answer_scores(output, tokenizer)

    def test_code_benchmark_allowed_when_enabled(self):
        responses = ["```python\nreturn x + 1\n```"]
        gts = [""]
        tokenizer = _FakeTokenizer(responses)
        output = _make_output(
            responses,
            gts,
            is_mcq=False,
            benchmark_type="code_functional",
        )

        verl_extract_answer_scores(output, tokenizer, allow_code_execution=True)

        assert output.non_tensor_batch["extracted_answer"][0].strip() == "return x + 1"

    def test_pool_worker_score_hard_times_out_wedged_row(self, monkeypatch):
        """A wedged row must return one timeout tuple instead of blocking."""
        import multiprocessing as mp
        import time

        if "fork" not in mp.get_all_start_methods():
            pytest.skip("requires fork so the child inherits the monkeypatch")

        from verl_inf_evolve.sol_eval import benchmark_adapters

        def _hang(*args, **kwargs):
            time.sleep(60)

        monkeypatch.setattr(benchmark_adapters, "score_response_for_question", _hang)

        payload = (
            7,
            {
                "benchmark_type": "qa_open",
                "ground_truth": "1",
                "data_source": "",
                "verifier_metadata": None,
            },
            r"\boxed{1}",
            False,
            1.0,
        )
        started = time.monotonic()

        idx, scored, status = verl_mcq_utils._pool_worker_score(payload)

        assert time.monotonic() - started < 5.0
        assert idx == 7
        assert scored is None
        assert status == "timeout"
