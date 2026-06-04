"""Optional LLM judge for MATH500 / Minerva / similar math benchmark scoring.

Two judging modes are supported:

1. **direct** (default) — General-Reasoner's protocol, which the R-Zero ICLR
   2026 paper uses for its Table 1 numbers. The judge is invoked on **every**
   sample regardless of rule grader outcome, and its verdict OVERWRITES the
   rule score. Default model is ``gpt-4o`` with the GR ``EQUALITY_TEMPLATE``
   prompt (8 worked few-shot examples, "give benefit of the doubt to units").
   This reproduces R-Zero paper Table 1 Minerva exactly: applying it to our
   own 8B-Base sampled rollouts gives 0.4972 vs paper 0.500. Source:
   ``baselines/general-reasoner/evaluation/simple-evals/{minerva_eval_qwen,
   common}.py``.

2. **cascade** — OpenCompass ``CascadeEvaluator`` style upgrade-on-rule-fail.
   The judge is only invoked when the rule grader returns 0; on a True
   verdict we flip the rule score 0→1 in place. Original sol_eval default
   (gpt-5-mini + OC prompt). Use this when you want the rule grader to
   anchor the score and only catch unambiguous false negatives.

Opt-in via ``eval.math500_llm_judge.enabled: true`` in ``sol_eval.yaml``.
Scoped per-benchmark via ``applies_to`` (default: ``("math500", "math")``).

Design notes:
- Module-level singleton state so the feature can be configured once at
  sol_eval startup and read from the two post-scoring integration points
  (``score_completions`` in vllm_runtime and ``evaluate_benchmark_questions``
  in eval_core) without plumbing config through the rule-scoring seam.
- Execution model: rule scoring remains inline. After each benchmark's
  ``question_results`` list is built, ``apply_to_question_results`` sweeps
  it for eligible samples (rule-failed only in cascade mode, all samples in
  direct mode), dispatches them to a ``ThreadPoolExecutor`` running up to
  ``max_concurrent`` concurrent ``judge()`` calls, and writes verdicts back
  in place.
- Fail-soft: any network/SDK error in cascade mode preserves the rule score;
  in direct mode the rule score also stands as the fallback when the judge
  call errors (so no silent regressions).
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"enabled": False, "calls": 0, "upgrades": 0}
_DEFAULT_MAX_CONCURRENT = 16

# Prompt adapted verbatim from OpenCompass's math_500_cascade_eval_gen_6ff468
# GRADER_TEMPLATE. Matches the OC CascadeEvaluator flow our other OC-based
# configs rely on, so scores are comparable.
_GRADER_TEMPLATE = """Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
5. If the prediction is given with \\boxed{}, please ignore the \\boxed{} and only judge whether the candidate's answer is consistent with the standard answer.

Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
Just return the letters "A" or "B", with no text around it.

Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


<Original Question Begin>:
{problem}
<Original Question End>


<Gold Target Begin>:
{solution}
<Gold Target End>


<Predicted Answer Begin>:
{prediction}
<Predicted End>


Judging the correctness of candidates' answers:""".strip()

_SYSTEM_PROMPT = (
    "You are a helpful assistant who evaluates the correctness and quality of models' outputs."
)

# Verbatim from baselines/general-reasoner/evaluation/simple-evals/common.py:79
# Used by minerva_eval_qwen.py:52 (and the other math evals) as a direct
# per-sample equality checker. The 8 few-shot examples explicitly tell the
# judge to give benefit of the doubt on units and trivial simplifications.
# Reproducing R-Zero paper Table 1 Minerva to within rollout noise on both
# 4B and 8B (2026-04-15 diagnostic).
_GR_EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

# Benchmarks for which the judge is currently eligible. MATH500 first;
# extend deliberately so we don't silently inflate other benchmarks' scores.
_DEFAULT_APPLIES_TO: tuple[str, ...] = ("math500", "math")

# Mode constants.
MODE_CASCADE = "cascade"
MODE_DIRECT = "direct"

# Prompt-style constants.
PROMPT_OC_CASCADE = "oc_cascade"
PROMPT_GR_EQUALITY = "gr_equality"


def configure(
    *,
    enabled: bool,
    model: str = "gpt-4o",
    api_base: Optional[str] = None,
    temperature: Optional[float] = 0.5,
    max_retries: int = 2,
    request_timeout: float = 30.0,
    applies_to: Optional[tuple[str, ...]] = None,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    mode: str = MODE_DIRECT,
    prompt_style: str = PROMPT_GR_EQUALITY,
) -> None:
    """Configure the LLM judge singleton. Call once at sol_eval startup.

    Default config reproduces R-Zero paper Table 1 / SPICE convention:
    ``mode=direct`` + ``prompt_style=gr_equality`` + ``model=gpt-4o`` +
    ``temperature=0.5``. Verified 2026-04-15 against the published numbers
    on Qwen3-{4B,8B}-Base Minerva (within rollout noise on both sizes).

    ``temperature=None`` → omit the parameter entirely (use the model's
    default). Required for gpt-5-series models, which reject any explicit
    temperature except 1.0.

    ``mode``: ``"direct"`` calls the judge on every eligible sample and
    overwrites the rule score with the verdict (matches GR / R-Zero paper).
    ``"cascade"`` only calls the judge on rule-failed samples and upgrades
    0→1 on a True verdict (matches the original sol_eval / OC behaviour).

    ``prompt_style``: ``"gr_equality"`` uses General-Reasoner's
    ``EQUALITY_TEMPLATE`` (8 few-shot examples, "give benefit of the doubt
    to units"); ``"oc_cascade"`` uses OpenCompass's ``CascadeEvaluator``
    grader template (~250 words, no examples, A/B output).

    ``max_concurrent`` controls the ThreadPoolExecutor worker count used
    by :func:`judge_batch` / :func:`apply_to_question_results` for the
    end-of-benchmark pass. Defaults to 16.
    """
    if mode not in (MODE_CASCADE, MODE_DIRECT):
        raise ValueError(f"unknown mode={mode!r}; expected 'cascade' or 'direct'")
    if prompt_style not in (PROMPT_OC_CASCADE, PROMPT_GR_EQUALITY):
        raise ValueError(
            f"unknown prompt_style={prompt_style!r}; expected 'oc_cascade' or 'gr_equality'"
        )
    with _STATE_LOCK:
        _STATE.clear()
        _STATE.update(
            enabled=bool(enabled),
            model=str(model),
            api_base=api_base,
            temperature=(None if temperature is None else float(temperature)),
            max_retries=int(max_retries),
            request_timeout=float(request_timeout),
            applies_to=tuple(applies_to) if applies_to else _DEFAULT_APPLIES_TO,
            max_concurrent=max(1, int(max_concurrent)),
            mode=str(mode),
            prompt_style=str(prompt_style),
            calls=0,
            upgrades=0,
            downgrades=0,
            api_errors=0,
            parse_errors=0,
        )
    if enabled:
        logger.info(
            "LLM math judge ENABLED: mode=%s prompt_style=%s model=%s "
            "applies_to=%s api_base=%s max_concurrent=%d",
            mode,
            prompt_style,
            model,
            _STATE["applies_to"],
            api_base or "default (OpenAI)",
            _STATE["max_concurrent"],
        )


def is_enabled_for(data_source: str) -> bool:
    """True iff the judge is on AND this data source is in ``applies_to``."""
    if not _STATE.get("enabled"):
        return False
    return str(data_source) in _STATE.get("applies_to", ())


def get_stats() -> dict[str, int]:
    """Return a snapshot of judge call counters."""
    return {
        "calls": int(_STATE.get("calls", 0)),
        "upgrades": int(_STATE.get("upgrades", 0)),
        "downgrades": int(_STATE.get("downgrades", 0)),
        "api_errors": int(_STATE.get("api_errors", 0)),
        "parse_errors": int(_STATE.get("parse_errors", 0)),
    }


def _classify_verdict_oc(text: str) -> Optional[bool]:
    """Parse the OC CascadeEvaluator output (CORRECT/INCORRECT or A/B)."""
    if not text:
        return None
    s = text.strip().upper()
    first_token = s.split()[0] if s.split() else ""
    has_correct = "CORRECT" in s
    has_incorrect = "INCORRECT" in s
    if first_token == "A" and not has_incorrect:
        return True
    if first_token == "B" and not has_correct:
        return False
    if has_incorrect:
        return False
    if has_correct:
        return True
    return None


def _classify_verdict_gr(text: str) -> Optional[bool]:
    """Parse General-Reasoner's check_equality output ('Yes' / 'No').

    Mirrors common.py:161 ``response.lower().strip() == "yes"`` — strict
    yes-only acceptance, anything else (including empty / parse failure)
    is treated as False. To preserve fail-soft semantics on API errors we
    still return None at the call site rather than False, but a clean
    'no' / unrecognized answer maps to False here.
    """
    if not text:
        return None
    s = text.strip().lower()
    if s == "yes":
        return True
    if s == "no":
        return False
    # GR's check_equality returns False for anything that isn't exactly "yes",
    # but for our diagnostic purposes we distinguish ambiguous outputs so the
    # caller can fall back to the rule score in direct mode rather than
    # silently flipping a rule-correct sample to wrong.
    return None


def judge(
    *,
    question_text: str,
    ground_truth: str,
    prediction: str,
) -> Optional[bool]:
    """Ask the configured LLM whether ``prediction`` matches ``ground_truth``.

    Returns ``True``/``False`` on a clear verdict, ``None`` on any error so
    the caller can preserve the rule-based score without surprises.
    """
    if not _STATE.get("enabled"):
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("LLM math judge enabled but OPENAI_API_KEY not set; skipping")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai SDK not available; LLM judge disabled at runtime")
        return None

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    api_base = _STATE.get("api_base")
    if api_base:
        client_kwargs["base_url"] = api_base
    client = OpenAI(**client_kwargs)

    prompt_style = _STATE.get("prompt_style", PROMPT_GR_EQUALITY)
    if prompt_style == PROMPT_GR_EQUALITY:
        # GR's check_equality dispatches a single user message containing the
        # full prompt; no system message. Use ``%`` formatting per common.py:159
        # so we don't have to escape literal braces in the few-shot examples.
        prompt = _GR_EQUALITY_TEMPLATE % {
            "expression1": ground_truth or "",
            "expression2": prediction or "",
        }
        messages = [{"role": "user", "content": prompt}]
    else:
        # OC CascadeEvaluator: system + user with the question / gold / prediction
        # interpolated. Use .replace() because the prompt contains literal
        # ``\boxed{}`` that str.format would misinterpret as a positional slot.
        prompt = (
            _GRADER_TEMPLATE
            .replace("{problem}", question_text or "")
            .replace("{solution}", ground_truth or "")
            .replace("{prediction}", prediction or "")
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    max_retries = int(_STATE.get("max_retries", 2))
    timeout = float(_STATE.get("request_timeout", 30.0))
    temperature = _STATE.get("temperature")

    # gpt-5-series models reject an explicit ``temperature`` other than 1.0.
    # Omit the parameter entirely when temperature is None so we use the
    # model's default. For older models (gpt-4o, etc.) pass it through.
    create_kwargs: dict[str, Any] = {"model": _STATE["model"], "messages": messages}
    if temperature is not None:
        create_kwargs["temperature"] = float(temperature)

    text: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.with_options(timeout=timeout).chat.completions.create(
                **create_kwargs,
            )
            text = resp.choices[0].message.content or ""
            break
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "LLM math judge attempt %d/%d failed: %r",
                attempt + 1,
                max_retries + 1,
                e,
            )
            if attempt == max_retries:
                with _STATE_LOCK:
                    _STATE["api_errors"] = int(_STATE.get("api_errors", 0)) + 1
                return None

    if text is None:
        return None

    verdict = (
        _classify_verdict_gr(text)
        if prompt_style == PROMPT_GR_EQUALITY
        else _classify_verdict_oc(text)
    )
    with _STATE_LOCK:
        _STATE["calls"] = int(_STATE.get("calls", 0)) + 1
        if verdict is None:
            _STATE["parse_errors"] = int(_STATE.get("parse_errors", 0)) + 1
            logger.warning("LLM math judge returned unparseable verdict: %r", text[:200])
        # Note: upgrade/downgrade counters are bumped by apply_to_question_results
        # at the call site, since we need to know the prior rule score to
        # distinguish an upgrade (rule=0 → judge=1) from a same-state verdict.
    return verdict


def judge_batch(
    items: list[dict[str, str]],
    *,
    max_workers: Optional[int] = None,
) -> list[Optional[bool]]:
    """Judge a batch of ``(question_text, ground_truth, prediction)`` triples
    concurrently via a ``ThreadPoolExecutor``.

    Each item must be a dict with keys ``question_text``, ``ground_truth``,
    ``prediction``. Returns a list of ``Optional[bool]`` verdicts aligned with
    the input order (``None`` on any judge failure — caller should preserve
    the pre-existing rule-based score).
    """
    if not items:
        return []
    if not _STATE.get("enabled"):
        return [None] * len(items)
    workers = int(max_workers or _STATE.get("max_concurrent", _DEFAULT_MAX_CONCURRENT))
    workers = max(1, min(workers, len(items)))

    results: list[Optional[bool]] = [None] * len(items)

    def _one(idx: int) -> tuple[int, Optional[bool]]:
        item = items[idx]
        verdict = judge(
            question_text=str(item.get("question_text", "")),
            ground_truth=str(item.get("ground_truth", "")),
            prediction=str(item.get("prediction", "")),
        )
        return idx, verdict

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, i) for i in range(len(items))]
        for fut in as_completed(futures):
            try:
                idx, verdict = fut.result()
            except Exception as e:  # noqa: BLE001
                logger.warning("judge_batch worker raised: %r", e)
                continue
            results[idx] = verdict
    return results


def apply_to_question_results(
    question_results: list[dict[str, Any]],
    *,
    max_workers: Optional[int] = None,
) -> int:
    """Apply the configured judge to ``question_results`` in place.

    Walks ``question_results`` (as produced by either ``build_question_results``
    or ``score_completions``) and, for each sample whose ``data_source`` is in
    the configured ``applies_to`` list, dispatches a judge call:

    - **cascade mode**: only rule-failed samples (``answer_score == 0.0``) are
      sent to the judge. A ``True`` verdict upgrades the score 0→1; any other
      verdict leaves the rule score unchanged.

    - **direct mode** (default, matches General-Reasoner / R-Zero paper): every
      sample with a non-``None`` rule score is sent to the judge. The verdict
      OVERWRITES the rule score (True → 1, False → 0). On API/parse errors
      the rule score is preserved as a fail-soft fallback. Samples with
      ``answer_score == None`` (extraction failure) remain ``None`` in
      cascade mode and are scored as 0 in direct mode (matches GR's
      ``check_equality(None) → False`` behaviour).

    Returns the number of scores ``upgrades`` (0 → 1 transitions). Downgrades
    in direct mode (rule-correct → judge-incorrect) are tracked separately
    via ``get_stats()['downgrades']``.
    """
    if not question_results or not _STATE.get("enabled"):
        return 0
    applies_to = set(_STATE.get("applies_to", ()))
    if not applies_to:
        return 0
    mode = _STATE.get("mode", MODE_DIRECT)

    # Each element keys back to (question_index, sample_index) so we can
    # write the verdict back in place after the concurrent sweep.
    targets: list[tuple[int, int]] = []
    items: list[dict[str, str]] = []
    rule_scores: list[Optional[float]] = []  # original rule score per target

    for qi, qr in enumerate(question_results):
        data_source = str(qr.get("data_source", "") or "")
        if data_source not in applies_to:
            continue
        scores = qr.get("answer_scores", []) or []
        sampled = qr.get("sampled_answers", []) or []
        extracted = qr.get("extracted_answers", []) or []
        gt = str(qr.get("ground_truth", "") or "")
        qtext = str(qr.get("question_text", "") or "")
        for si, score in enumerate(scores):
            if score is None:
                if mode == MODE_DIRECT:
                    # GR's check_equality returns 0 for unparseable / missing
                    # extraction; mirror that by setting the score to 0.0
                    # without spending a judge call.
                    question_results[qi]["answer_scores"][si] = 0.0
                continue
            if mode == MODE_CASCADE:
                try:
                    if float(score) != 0.0:
                        continue
                except (TypeError, ValueError):
                    continue
            # Prefer the extracted answer; fall back to the raw response text.
            prediction: str = ""
            if si < len(extracted) and extracted[si] is not None:
                prediction = str(extracted[si])
            elif si < len(sampled) and sampled[si] is not None:
                prediction = str(sampled[si])
            targets.append((qi, si))
            items.append(
                {"question_text": qtext, "ground_truth": gt, "prediction": prediction}
            )
            try:
                rule_scores.append(float(score))
            except (TypeError, ValueError):
                rule_scores.append(None)

    if not items:
        return 0

    logger.info(
        "LLM math judge: mode=%s dispatching %d samples across %d questions "
        "(max_workers=%d)",
        mode,
        len(items),
        len({qi for qi, _ in targets}),
        int(max_workers or _STATE.get("max_concurrent", _DEFAULT_MAX_CONCURRENT)),
    )
    verdicts = judge_batch(items, max_workers=max_workers)

    upgrades = 0
    downgrades = 0
    for (qi, si), verdict, rule_score in zip(targets, verdicts, rule_scores):
        if mode == MODE_CASCADE:
            if verdict is True:
                question_results[qi]["answer_scores"][si] = 1.0
                upgrades += 1
            # cascade: False / None leaves the rule score (0.0) alone
            continue

        # direct mode
        if verdict is True:
            question_results[qi]["answer_scores"][si] = 1.0
            if rule_score == 0.0:
                upgrades += 1
        elif verdict is False:
            question_results[qi]["answer_scores"][si] = 0.0
            if rule_score == 1.0:
                downgrades += 1
        else:
            # API/parse error → keep the rule score as fail-soft fallback.
            pass

    if downgrades:
        with _STATE_LOCK:
            _STATE["downgrades"] = int(_STATE.get("downgrades", 0)) + downgrades
    if upgrades:
        with _STATE_LOCK:
            _STATE["upgrades"] = int(_STATE.get("upgrades", 0)) + upgrades
    logger.info(
        "LLM math judge: %d upgrades (rule=0→judge=1), %d downgrades "
        "(rule=1→judge=0) over %d judged samples",
        upgrades,
        downgrades,
        len(items),
    )
    return upgrades
