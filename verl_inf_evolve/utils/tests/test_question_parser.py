"""Unit tests for generated-question parsing.

These tests cover the lightweight JSON/parser layer only. They do not require
vLLM, Ray, or GPU execution.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np


class _DecodeByFirstToken:
    def __init__(self, texts: list[str]):
        self._texts = texts

    def decode(self, response, skip_special_tokens: bool = True):
        del skip_special_tokens
        return self._texts[int(response[0])]


class GeneratedQuestionParserTest(unittest.TestCase):
    def test_parse_free_form_response_without_choices(self):
        from verl_inf_evolve.utils.question_parser import (
            parse_generated_question_response,
        )

        parsed = parse_generated_question_response(
            json.dumps(
                {
                    "question_text": "If x^2=9 and x>0, find x.",
                    "ground_truth": "3",
                    "answer_type": "numerical",
                    "benchmark_type": "qa_open",
                    "data_source": "math",
                }
            )
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertFalse(parsed["is_mcq"])
        self.assertNotIn("choices", parsed)
        self.assertEqual(parsed["benchmark_type"], "qa_open")
        self.assertEqual(parsed["data_source"], "math")
        self.assertEqual(parsed["answer_type"], "numerical")

    def test_parse_generated_questions_mixed_mcq_and_free_form(self):
        from verl_inf_evolve.utils.data_utils import derive_gen_questions
        from verl_inf_evolve.utils.question_parser import parse_generated_questions

        mcq = json.dumps(
            {
                "question_text": "What is 2+3?",
                "choices": ["4", "5", "6", "7"],
                "ground_truth": "5",
            }
        )
        free_form = json.dumps(
            {
                "question_text": "If x^2=9 and x>0, find x.",
                "ground_truth": "3",
                "answer_type": "numerical",
                "benchmark_type": "qa_open",
                "data_source": "math",
            }
        )
        malformed = json.dumps(
            {
                "question_text": "Missing answer",
                "benchmark_type": "qa_open",
            }
        )
        output = SimpleNamespace(
            batch={"responses": np.array([[0], [1], [2]])},
            non_tensor_batch={"doc_id": np.array(["d0", "d1", "d2"], dtype=object)},
        )

        parse_generated_questions(
            output,
            _DecodeByFirstToken([mcq, free_form, malformed]),
        )

        self.assertEqual(output.non_tensor_batch["parsed_ok"].tolist(), [True, True, False])
        questions = derive_gen_questions(output)
        self.assertEqual(len(questions), 2)
        self.assertTrue(questions[0]["is_mcq"])
        self.assertFalse(questions[1]["is_mcq"])
        self.assertEqual(questions[1]["choices"], [])
        self.assertEqual(questions[1]["benchmark_type"], "qa_open")
        self.assertEqual(questions[1]["data_source"], "math")
        self.assertEqual(questions[1]["answer_type"], "numerical")

    def test_free_form_ground_truth_list_is_rejected(self):
        from verl_inf_evolve.utils.question_parser import (
            parse_generated_question_response,
        )

        parsed = parse_generated_question_response(
            json.dumps(
                {
                    "question_text": "If x^2=9 and x>0, find x.",
                    "ground_truth": ["3"],
                    "benchmark_type": "qa_open",
                    "data_source": "math",
                }
            )
        )

        self.assertIsNone(parsed)

    def test_training_parser_rejects_unparseable_free_form_ground_truth(self):
        from verl_inf_evolve.utils.data_utils import derive_gen_questions
        from verl_inf_evolve.utils.question_parser import (
            parse_generated_question_response,
            parse_generated_questions,
        )

        prose_gt = json.dumps(
            {
                "question_text": "What are the triangle side lengths?",
                "ground_truth": "The sides are 15, 36, and 39 inches.",
                "benchmark_type": "qa_open",
                "data_source": "math",
            }
        )
        valid_gt = json.dumps(
            {
                "question_text": "If x^2=9 and x>0, find x.",
                "ground_truth": "3",
                "benchmark_type": "qa_open",
                "data_source": "math",
            }
        )
        ambiguous_factorial_gt = json.dumps(
            {
                "question_text": "Express the square of n factorial.",
                "ground_truth": "n!^2",
                "benchmark_type": "qa_open",
                "data_source": "math",
            }
        )

        # Offline candidate parsing stays permissive so callers can inspect
        # the parsed record and its separate gt_self_parseable flag.
        self.assertIsNotNone(parse_generated_question_response(prose_gt))

        output = SimpleNamespace(
            batch={"responses": np.array([[0], [1], [2]])},
            non_tensor_batch={"doc_id": np.array(["d0", "d1", "d2"], dtype=object)},
        )

        parse_generated_questions(
            output,
            _DecodeByFirstToken([prose_gt, valid_gt, ambiguous_factorial_gt]),
        )

        self.assertEqual(
            output.non_tensor_batch["parsed_ok"].tolist(),
            [False, True, False],
        )
        self.assertEqual(
            output.non_tensor_batch["reject_reason"].tolist(),
            ["invalid_ground_truth", None, "invalid_ground_truth"],
        )
        self.assertEqual(
            output.non_tensor_batch["gt_self_parseable"].tolist(),
            [False, True, False],
        )
        questions = derive_gen_questions(output)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["ground_truth"], "3")


class JsonEscapeRecoveryTest(unittest.TestCase):
    """Regression tests for the LaTeX-mangled escape recovery in
    ``fix_json_escapes`` / ``extract_thinking_and_json``.

    Without the fix, ``\\frac``/``\\beta``/etc. emitted with a single
    backslash by the model are silently mapped to a control byte
    (``\\x0c``/``\\x08``/``\\x09``/``\\x0a``/``\\x0d``) by ``json.loads``,
    corrupting math ground truths. The fix preserves the literal LaTeX
    command when ``\\b/\\f/\\n/\\r/\\t`` is followed by an ASCII letter.
    """

    def _parse_gt(self, gt_literal: str) -> str:
        from verl_inf_evolve.utils.question_parser import (
            parse_generated_question_response,
        )

        # Build a JSON object literal by hand so we control the exact
        # backslash bytes (json.dumps would auto-escape them away).
        response = (
            '{"question_text": "Q?", "ground_truth": "'
            + gt_literal
            + '", "answer_type": "expression", "benchmark_type": "qa_open", '
            '"data_source": "math"}'
        )
        parsed = parse_generated_question_response(response)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        return parsed["ground_truth"]

    def test_single_backslash_frac_preserved(self):
        # Model emits "\frac{1}{2}" (single backslash). Without the fix this
        # parses to "\x0crac{1}{2}".
        gt = self._parse_gt(r"\frac{1}{2}")
        self.assertEqual(gt, r"\frac{1}{2}")
        self.assertNotIn("\x0c", gt)

    def test_single_backslash_beta_preserved(self):
        gt = self._parse_gt(r"\beta")
        self.assertEqual(gt, r"\beta")
        self.assertNotIn("\x08", gt)

    def test_single_backslash_theta_to_tau_preserved(self):
        for cmd in ("theta", "to", "tau", "times", "text"):
            gt = self._parse_gt("\\" + cmd)
            self.assertEqual(gt, "\\" + cmd, f"{cmd!r} mangled")
            self.assertNotIn("\x09", gt)

    def test_single_backslash_nabla_preserved(self):
        gt = self._parse_gt(r"\nabla")
        self.assertEqual(gt, r"\nabla")
        self.assertNotIn("\x0a", gt)

    def test_single_backslash_rightarrow_preserved(self):
        gt = self._parse_gt(r"\rightarrow")
        self.assertEqual(gt, r"\rightarrow")
        self.assertNotIn("\x0d", gt)

    def test_double_backslash_already_correct_passes_through(self):
        # ``\\frac`` (double backslash) is the JSON-correct way to encode a
        # literal backslash + ``frac``; behaviour must be unchanged.
        gt = self._parse_gt(r"\\frac{1}{2}")
        self.assertEqual(gt, r"\frac{1}{2}")

    def test_existing_invalid_escape_recovery_still_works(self):
        # ``\sum`` raises ``Invalid \escape`` under raw json.loads; fix
        # path doubles the backslash. This was the original recovery case.
        gt = self._parse_gt(r"\sum_{k=1}^n k")
        self.assertEqual(gt, r"\sum_{k=1}^n k")

    def test_mixed_latex_in_ground_truth(self):
        # Single-backslash combined: ``\frac{\beta}{\theta}``
        gt = self._parse_gt(r"\frac{\beta}{\theta}")
        self.assertEqual(gt, r"\frac{\beta}{\theta}")

    def test_control_chars_alone_are_not_treated_as_latex(self):
        # ``\b`` (etc.) NOT followed by a letter should still be a normal
        # JSON escape — only the backslash-letter LaTeX form is fixed up.
        # The pattern ``\\b\\b`` represents two consecutive ``\b`` escapes
        # (not a LaTeX command), so it should still parse to two backspaces.
        from verl_inf_evolve.utils.question_parser import fix_json_escapes

        # Direct check of the helper to keep this case unambiguous.
        text = '{"x": "\\b\\b"}'  # two \b escapes, no following letter
        out = json.loads(fix_json_escapes(text))
        self.assertEqual(out["x"], "\b\b")


class MathVerifierCanonicalizationTest(unittest.TestCase):
    """Tests for ``canonicalize_for_math_verify`` and its effect on
    ``is_math_verifier_parseable`` — the column gen_data_refine emits to
    let the stage-3 judge filter ground truths math_verify can interpret.
    """

    def test_percent_canonicalized(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import (
            canonicalize_for_math_verify,
            is_math_verifier_parseable,
        )

        for raw in ("127.32%", r"127.32\%", "50%", r"100\% + 5\%"):
            self.assertTrue(
                is_math_verifier_parseable(raw),
                f"{raw!r} should be parseable after %-canonicalization",
            )
            self.assertNotIn("%", canonicalize_for_math_verify(raw))

    def test_factorial_then_exponent_left_alone(self):
        # ``X!^k`` is ambiguous in raw form (LaTeX would render the ``^`` as
        # binding to the postfix factorial, which the parser doesn't accept).
        # We deliberately do NOT canonicalize it — the judge should see
        # ``gt_self_parseable=False`` and decide.
        from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import (
            canonicalize_for_math_verify,
            is_math_verifier_parseable,
        )

        for raw in ("n!^2", r"n!^2 \geq n^n", "5!^2 = 14400", "(n+1)!^2"):
            self.assertEqual(
                canonicalize_for_math_verify(raw), raw,
                f"{raw!r} should be left untouched",
            )
            self.assertFalse(
                is_math_verifier_parseable(raw),
                f"{raw!r} is ambiguous; must not pass gt_self_parseable",
            )

    def test_parenthesized_factorial_with_exponent_passes(self):
        # The fully-parenthesized canonical form is unambiguous and parseable.
        from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import (
            is_math_verifier_parseable,
        )

        self.assertTrue(is_math_verifier_parseable("(n!)^2"))
        self.assertTrue(is_math_verifier_parseable(r"(n!)^2 \geq n^n"))

    def test_already_canonical_unchanged(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import (
            canonicalize_for_math_verify,
        )

        for raw in (r"\frac{1}{2}", "10^{-8}", "n!", r"\sum_{k=1}^n k"):
            self.assertEqual(canonicalize_for_math_verify(raw), raw)

    def test_prose_still_rejected(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import (
            is_math_verifier_parseable,
        )

        for raw in (
            "<exact concise mathematical answer>",
            "The sides of the triangle are 15, 36, and 39 inches.",
        ):
            self.assertFalse(
                is_math_verifier_parseable(raw),
                f"{raw!r} should not be parseable",
            )

    def test_empty_inputs(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import (
            canonicalize_for_math_verify,
            is_math_verifier_parseable,
        )

        self.assertEqual(canonicalize_for_math_verify(""), "")
        self.assertFalse(is_math_verifier_parseable(""))
        self.assertFalse(is_math_verifier_parseable(None))

    def test_batch_parseability_preserves_order(self):
        from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import (
            is_math_verifier_parseable_many,
        )

        self.assertEqual(
            is_math_verifier_parseable_many(
                ["3", "The answer is three.", "x=1 or x=2", "n!^2"],
                workers=2,
                chunksize=1,
            ),
            [True, False, True, False],
        )


if __name__ == "__main__":
    unittest.main()
