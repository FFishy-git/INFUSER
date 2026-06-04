"""Question JSON parsing helpers for generator outputs.

This is a local copy of the subset currently used by
`verl_inf_evolve.trainer.self_evolution_trainer`.
"""

import json
import random
import re
from typing import Any, Callable, Optional, Tuple


# Cap the brute-force fallback in ``extract_thinking_and_json`` so a
# malformed long response with hundreds of ``{`` chars cannot stall parsing
# for tens of seconds. These limits do not affect the fast path (balanced
# top-level ``{...}`` spans) — they only bound the slow recovery for
# unbalanced/garbage inputs that are almost certainly unparseable anyway.
_FALLBACK_MAX_CHARS = 16384
_FALLBACK_MAX_BRACES = 8


def fix_json_escapes(text: str) -> str:
    """Fix invalid / silently-mangled JSON escape sequences (LaTeX backslashes).

    Two cases are handled:

    1. Backslash + ``b/f/n/r/t`` followed by an ASCII letter
       (``\\frac``, ``\\beta``, ``\\theta``, ``\\nabla``, ``\\rightarrow``).
       These ARE valid JSON escapes for backspace/formfeed/tab/newline/CR, so
       ``json.loads`` accepts them silently and produces a control byte
       followed by the rest of the LaTeX command (e.g. ``\\frac{{1}}{{2}}``
       parses to ``"\\x0crac{{1}}{{2}}"``). Math verifiers then cannot
       symbolically match. We treat ``\\X[a-zA-Z]`` as unambiguously LaTeX
       and double-escape so the literal backslash survives.

    2. Backslash + any other character that isn't a valid JSON escape
       (``\\sum``, ``\\int``, ``\\sqrt``, ``\\cdot``, ``\\(``, ``\\[``).
       Plain ``json.loads`` raises ``Invalid \\escape``; we double the
       backslash so the literal LaTeX command survives parsing.

    Existing correctly-escaped ``\\\\`` sequences pass through unchanged.
    """
    text = text.replace("\\\\", "\x00DOUBLE_BACKSLASH\x00")

    # Case 1: silent-mangle escapes. Must run before the generic pass below,
    # because that pass whitelists b/f/n/r/t as valid one-letter escapes.
    text = re.sub(r"\\([bfnrt])(?=[a-zA-Z])", r"\\\\\1", text)

    valid_escapes = ['"', "/", "b", "f", "n", "r", "t", "\\"]

    def fix_escape(match: re.Match[str]) -> str:
        escape_seq = match.group(0)
        next_char = escape_seq[1] if len(escape_seq) > 1 else ""
        if next_char in valid_escapes or (next_char == "u" and len(escape_seq) >= 6):
            return escape_seq
        return "\\\\" + next_char

    text = re.sub(r"\\(.)", fix_escape, text)
    text = text.replace("\x00DOUBLE_BACKSLASH\x00", "\\\\")
    return text


def _find_top_level_brace_spans(text: str) -> list[Tuple[int, int]]:
    """Linear scan for top-level (depth-1) ``{...}`` spans in ``text``.

    Tracks string-literal context so braces inside ``"..."`` don't affect
    nesting. Spans are returned as ``(start, end_exclusive)`` pairs in the
    order they close. Used to bound the candidate search in
    ``extract_thinking_and_json`` from O(B × N) — where B is the number of
    ``{`` and N the text length — down to O(N).
    """
    spans: list[Tuple[int, int]] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append((start, i + 1))
                    start = -1
    return spans


def _try_parse_json_candidate(candidate: str) -> Optional[dict[str, Any]]:
    """Try the LaTeX-fixed candidate first so JSON-valid escape sequences
    inside LaTeX commands (e.g. ``\\frac``, ``\\beta``, ``\\theta``) aren't
    silently mapped to control bytes by ``json.loads``. Fall back to the
    raw candidate if the fix produces an unparseable string.
    """
    try:
        return json.loads(fix_json_escapes(candidate))
    except json.JSONDecodeError:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None


def extract_thinking_and_json(
    response_text: str,
    validation_func: Optional[Callable[[dict[str, Any]], bool]] = None,
) -> Tuple[Optional[str], Optional[dict[str, Any]], Optional[str]]:
    """Extract think-tag content plus the last valid JSON object.

    Performance note: we first walk the cleaned text once to find balanced
    top-level ``{...}`` spans (O(N)) and try those as candidates. This is
    the fast path for well-formed model output. If none of them parse, we
    fall back to a bounded brute-force search over remaining ``{`` positions
    so we still recover JSON nested inside an unbalanced wrapper. Without
    the fast path, responses with hundreds of ``{`` chars (LaTeX inside
    long answers) make ``parse_generated_question_response`` O(B × N) per
    call, which dominated post-gen processing time on long rollouts.
    """
    if not response_text:
        return None, None, "empty_response"

    thinking_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    thinking_text = thinking_match.group(1).strip() if thinking_match else None

    cleaned_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"```json\s*", "", cleaned_text)
    cleaned_text = re.sub(r"```\s*", "", cleaned_text)
    cleaned_text = cleaned_text.strip()

    if not cleaned_text:
        return thinking_text, None, "no_json_content"
    if "{" not in cleaned_text:
        return thinking_text, None, "no_json_object"

    all_valid_jsons: list[dict[str, Any]] = []

    # --- Fast path: balanced top-level {...} spans ---
    for start_pos, end_pos in _find_top_level_brace_spans(cleaned_text):
        candidate = cleaned_text[start_pos:end_pos]
        result = _try_parse_json_candidate(candidate)
        if result is None:
            continue
        all_valid_jsons.append(
            {
                "json": result,
                "start": start_pos,
                "end": end_pos,
                "validated": validation_func(result) if validation_func else True,
            }
        )

    # --- Fallback: bounded brute-force for unbalanced wrappers ---
    # Only triggered when the fast path found nothing parseable. The
    # original brute-force was O(B × N) where B = number of ``{`` and
    # N = response length, which becomes catastrophic on malformed-JSON
    # responses with hundreds of braces (LaTeX inside long answers).
    # Cap both axes: try at most ``_FALLBACK_MAX_BRACES`` start positions
    # and skip the fallback entirely once the cleaned text exceeds
    # ``_FALLBACK_MAX_CHARS`` — those responses are almost certainly
    # malformed garbage anyway.
    if not all_valid_jsons and len(cleaned_text) <= _FALLBACK_MAX_CHARS:
        brace_positions = [
            i for i, char in enumerate(cleaned_text) if char == "{"
        ]
        for start_pos in brace_positions[: _FALLBACK_MAX_BRACES]:
            for end_pos in range(len(cleaned_text), start_pos, -1):
                candidate = cleaned_text[start_pos:end_pos].rstrip()
                if not candidate.endswith("}"):
                    continue
                result = _try_parse_json_candidate(candidate)
                if result is None:
                    continue
                all_valid_jsons.append(
                    {
                        "json": result,
                        "start": start_pos,
                        "end": end_pos,
                        "validated": validation_func(result)
                        if validation_func
                        else True,
                    }
                )
                break

    if not all_valid_jsons:
        if cleaned_text.count("{") > cleaned_text.count("}"):
            return thinking_text, None, "truncated_json"
        return thinking_text, None, "invalid_json"

    if validation_func is not None:
        validated_jsons = [item for item in all_valid_jsons if item["validated"]]
        if validated_jsons:
            last_valid = max(validated_jsons, key=lambda item: item["start"])
            return thinking_text, last_valid["json"], None
        last_json = max(all_valid_jsons, key=lambda item: item["start"])
        return thinking_text, last_json["json"], "missing_required_fields"

    last_json = max(all_valid_jsons, key=lambda item: item["start"])
    return thinking_text, last_json["json"], None


def _normalize_string_field(value: Any, *, allow_list: bool = False) -> Optional[str]:
    """Normalize model-emitted scalar-ish fields into strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if allow_list and isinstance(value, list) and all(
        isinstance(item, (str, int, float)) for item in value
    ):
        return " ".join(str(item) for item in value)
    if isinstance(value, dict) and len(value) == 1:
        val = next(iter(value.values()))
        if isinstance(val, (str, int, float)):
            return str(val)
    return None


def _normalize_choices(value: Any) -> Optional[list[str]]:
    """Normalize MCQ choices into a plain string list."""
    if value is None:
        return None
    if isinstance(value, dict):
        normalized = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if letter in value:
                item = value[letter]
            elif letter.lower() in value:
                item = value[letter.lower()]
            else:
                break
            item_text = _normalize_string_field(item)
            if item_text is None:
                return None
            normalized.append(item_text)
        return normalized
    if not isinstance(value, list):
        return None

    normalized = []
    for item in value:
        item_text = _normalize_string_field(item)
        if item_text is None:
            return None
        normalized.append(item_text)
    return normalized


def _looks_like_free_form(result: dict[str, Any]) -> bool:
    benchmark_type = str(result.get("benchmark_type", "") or "").strip().lower()
    choices = result.get("choices", None)
    return benchmark_type == "qa_open" or choices == []


def validate_generated_question_structure(result: dict[str, Any]) -> bool:
    """Check whether a generated question has an MCQ or free-form schema."""
    if not isinstance(result, dict):
        return False
    if "question_text" not in result or "ground_truth" not in result:
        return False

    choices = _normalize_choices(result.get("choices", None))
    if choices is not None and len(choices) > 0:
        return 4 <= len(choices) <= 8

    return _looks_like_free_form(result)


def parse_generated_question_response(response_text: str) -> Optional[dict[str, Any]]:
    """Parse model output to a normalized generated-question dict."""
    _, result, failure_reason = extract_thinking_and_json(
        response_text, validation_func=validate_generated_question_structure
    )
    if result is None or failure_reason is not None:
        return None

    question_text = _normalize_string_field(result.get("question_text"), allow_list=True)
    if question_text is None:
        return None
    result["question_text"] = question_text

    ground_truth = _normalize_string_field(result.get("ground_truth"))
    if ground_truth is None:
        return None
    result["ground_truth"] = ground_truth

    choices = _normalize_choices(result.get("choices", None))
    if choices is not None and len(choices) > 0:
        if not (4 <= len(choices) <= 8):
            return None
        result["choices"] = choices
        result["is_mcq"] = True
        result.setdefault("benchmark_type", "qa_mcq")

        if result["ground_truth"] and result["ground_truth"] not in result["choices"]:
            for choice in result["choices"]:
                if result["ground_truth"].strip().lower() == choice.strip().lower():
                    result["ground_truth"] = choice
                    break
        return result

    if not _looks_like_free_form(result):
        return None
    result.pop("choices", None)
    result["is_mcq"] = False
    result["benchmark_type"] = str(result.get("benchmark_type", "qa_open") or "qa_open")
    result["data_source"] = str(result.get("data_source", "math") or "math")
    if result.get("answer_type") is not None:
        result["answer_type"] = str(result["answer_type"])
    return result


def parse_question_response(response_text: str) -> Optional[dict[str, Any]]:
    """Backward-compatible alias for parsing one generated question response."""
    return parse_generated_question_response(response_text)


def parse_generated_questions(output: "DataProto", tokenizer: Any) -> None:
    """Parse generated MCQ or free-form questions from generator rollout output.

    Decodes response tokens, parses JSON with
    ``parse_generated_question_response()``, validates the generated-question
    schema, and constructs question IDs following the V2 convention
    ``gen_{doc_id}_{sample_idx}_{hash}``.

    MCQ rows must have 4-8 choices and ``ground_truth`` must match one of the
    choices. Free-form rows must carry ``benchmark_type="qa_open"`` or omit
    choices, plus a non-empty machine-verifiable ``ground_truth``. Free-form
    rows are stored with ``choices=[]`` and ``is_mcq=False`` for downstream
    routing.
    """
    import hashlib
    import logging

    import numpy as np

    logger = logging.getLogger(__name__)

    responses = output.batch["responses"]  # [batch_size, response_len]
    doc_ids = output.non_tensor_batch["doc_id"]
    batch_size = responses.shape[0]

    # Per-sample arrays for storing back into non_tensor_batch
    question_ids = [None] * batch_size
    question_texts = [None] * batch_size
    choices_list = [None] * batch_size
    ground_truths = [None] * batch_size
    is_mcq_flags = [None] * batch_size
    benchmark_types = [None] * batch_size
    data_sources = [None] * batch_size
    answer_types = [None] * batch_size
    gt_self_parseable = [None] * batch_size
    parsed_ok = [False] * batch_size
    reject_reasons = [None] * batch_size
    pending_free_form: list[tuple[int, dict[str, Any], str, str]] = []

    def accept_parsed(
        i: int,
        parsed: dict[str, Any],
        *,
        choices: list[str],
        ground_truth: str,
        is_mcq: bool,
        benchmark_type: str,
        gt_parseable: Optional[bool],
    ) -> None:
        text_hash = hashlib.md5(
            parsed["question_text"].encode("utf-8", errors="replace")
        ).hexdigest()[:8]
        question_id = f"gen_{str(doc_ids[i])}_{i}_{text_hash}"

        question_ids[i] = question_id
        question_texts[i] = parsed["question_text"]
        choices_list[i] = choices
        ground_truths[i] = ground_truth
        is_mcq_flags[i] = is_mcq
        benchmark_types[i] = benchmark_type
        data_sources[i] = str(parsed.get("data_source", "") or "")
        answer_types[i] = (
            str(parsed["answer_type"])
            if parsed.get("answer_type") is not None
            else None
        )
        gt_self_parseable[i] = gt_parseable
        parsed_ok[i] = True

    for i in range(batch_size):
        response_text = tokenizer.decode(
            responses[i], skip_special_tokens=True
        )

        parsed = parse_generated_question_response(response_text)
        if parsed is None:
            reject_reasons[i] = "failed_parse"
            continue

        # Check 2: empty question_text
        question_text = parsed.get("question_text", "")
        if not isinstance(question_text, str):
            reject_reasons[i] = "failed_parse"
            continue
        if not question_text.strip():
            reject_reasons[i] = "empty_question"
            continue

        is_mcq = bool(parsed.get("is_mcq", bool(parsed.get("choices"))))
        ground_truth = str(parsed.get("ground_truth", "") or "")
        if not ground_truth.strip():
            reject_reasons[i] = "invalid_ground_truth"
            continue

        if is_mcq:
            # Check 3: MCQ ground_truth not in choices.
            if ground_truth not in parsed.get("choices", []):
                reject_reasons[i] = "invalid_ground_truth"
                continue
            # Shuffle choices to avoid position bias (V2 line 434-435).
            choices = parsed["choices"].copy()
            random.shuffle(choices)
            benchmark_type = str(parsed.get("benchmark_type", "qa_mcq") or "qa_mcq")
            accept_parsed(
                i,
                parsed,
                choices=choices,
                ground_truth=ground_truth,
                is_mcq=True,
                benchmark_type=benchmark_type,
                gt_parseable=None,
            )
        else:
            benchmark_type = str(parsed.get("benchmark_type", "qa_open") or "qa_open")
            pending_free_form.append((i, parsed, ground_truth, benchmark_type))

    if pending_free_form:
        from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import (
            is_math_verifier_parseable_many,
        )

        parseable_flags = is_math_verifier_parseable_many(
            [ground_truth for _, _, ground_truth, _ in pending_free_form]
        )
        for (i, parsed, ground_truth, benchmark_type), gt_parseable in zip(
            pending_free_form, parseable_flags
        ):
            if not gt_parseable:
                gt_self_parseable[i] = False
                reject_reasons[i] = "invalid_ground_truth"
                continue
            accept_parsed(
                i,
                parsed,
                choices=[],
                ground_truth=ground_truth,
                is_mcq=False,
                benchmark_type=benchmark_type,
                gt_parseable=True,
            )

    # Log per-category rejection counts
    reject_counts = {}
    for r in reject_reasons:
        if r is not None:
            reject_counts[r] = reject_counts.get(r, 0) + 1
    if reject_counts:
        parts = [f"{reason}={count}" for reason, count in sorted(reject_counts.items())]
        logger.warning(
            "parse_generated_questions: %d/%d rejected (%s)",
            sum(reject_counts.values()), batch_size, ", ".join(parts),
        )

    # Store per-sample parsed data into non_tensor_batch for downstream consumers
    output.non_tensor_batch["question_id"] = np.array(question_ids, dtype=object)
    output.non_tensor_batch["question_text"] = np.array(question_texts, dtype=object)
    output.non_tensor_batch["choices"] = np.array(choices_list, dtype=object)
    output.non_tensor_batch["ground_truth"] = np.array(ground_truths, dtype=object)
    output.non_tensor_batch["is_mcq"] = np.array(is_mcq_flags, dtype=object)
    output.non_tensor_batch["benchmark_type"] = np.array(benchmark_types, dtype=object)
    output.non_tensor_batch["data_source"] = np.array(data_sources, dtype=object)
    output.non_tensor_batch["answer_type"] = np.array(answer_types, dtype=object)
    output.non_tensor_batch["gt_self_parseable"] = np.array(
        gt_self_parseable, dtype=object
    )
    output.non_tensor_batch["parsed_ok"] = np.array(parsed_ok, dtype=object)
    output.non_tensor_batch["reject_reason"] = np.array(reject_reasons, dtype=object)


def parse_mcq_questions(output: "DataProto", tokenizer: Any) -> None:
    """Backward-compatible alias for ``parse_generated_questions``."""
    parse_generated_questions(output, tokenizer)


__all__ = [
    "parse_generated_questions",
    "parse_generated_question_response",
    "parse_mcq_questions",
    "parse_question_response",
]
