"""MCQ helpers used by the verl_inf_evolve trainer.

This is a local copy of the subset currently used by
`verl_inf_evolve.trainer.self_evolution_trainer`.
"""

import logging
import os
import random
import re
import string
import time
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)

from verl_inf_evolve.utils.prompts import (
    MCQ_ANSWER_GENERATION_FEW_SHOT_EXAMPLES,
    MCQ_ANSWER_GENERATION_ICL_PROMPT,
    MCQ_ANSWER_GENERATION_ICL_SYSTEM_PROMPT,
    MCQ_ANSWER_GENERATION_PROMPT,
    MCQ_ANSWER_GENERATION_SYSTEM_PROMPT,
)

CHOICE_LETTERS = list(string.ascii_uppercase)


def _normalize_mcq_ground_truth(choices: List[str], ground_truth: Any) -> int:
    """Return the correct choice index in the original choice order."""
    valid_letters = CHOICE_LETTERS[: len(choices)]

    if isinstance(ground_truth, int):
        if ground_truth < 0 or ground_truth >= len(choices):
            raise ValueError(
                f"ground_truth index {ground_truth} out of range for {len(choices)} choices"
            )
        return ground_truth

    if not isinstance(ground_truth, str):
        raise ValueError(f"ground_truth must be a string or int, got {type(ground_truth)}")

    ground_truth = ground_truth.strip()
    if ground_truth.upper() in valid_letters:
        return valid_letters.index(ground_truth.upper())

    ground_truth_normalized = ground_truth.upper()
    for idx, choice in enumerate(choices):
        if choice.upper().strip() == ground_truth_normalized:
            return idx

    raise ValueError(
        f"ground_truth '{ground_truth}' does not match any choice: {choices}"
    )


def _build_mcq_choice_permutation(
    num_choices: int,
    *,
    enabled: bool = False,
    mode: str = "per_question_seeded",
    seed: Optional[int] = None,
    question_id: Any = None,
    question_text: str = "",
) -> List[int]:
    """Return a choice-order permutation for one MCQ row."""
    permutation = list(range(num_choices))
    if not enabled or num_choices < 2:
        return permutation

    if mode == "run_random":
        random.shuffle(permutation)
        return permutation

    if mode != "per_question_seeded":
        raise ValueError(
            f"Unsupported mcq choice shuffle mode '{mode}'. "
            "Expected 'per_question_seeded' or 'run_random'."
        )

    question_key = question_id if question_id is not None else question_text
    rng = random.Random()
    rng.seed(f"{seed}:{question_key}")
    rng.shuffle(permutation)
    return permutation


def format_mcq_question(
    question_block: Dict[str, Any],
    *,
    shuffle_choices: bool = False,
    shuffle_seed: Optional[int] = None,
    shuffle_mode: str = "per_question_seeded",
) -> Tuple[str, str]:
    """Format question text and normalize ground truth to a choice letter.

    Example::

        >>> block = {
        ...     "question_text": "What is 2 + 3?",
        ...     "choices": ["4", "5", "6", "7"],
        ...     "ground_truth": "5",
        ... }
        >>> text, gt = format_mcq_question(block)
        >>> print(text)
        Question: What is 2 + 3?
        A) 4
        B) 5
        C) 6
        D) 7
        >>> gt
        'B'
    """
    question_text = question_block.get("question_text", "")
    choices = question_block.get("choices", [])
    ground_truth = question_block.get("ground_truth", "")

    if not question_text:
        raise ValueError("question_text is required in question_block")
    if not choices:
        raise ValueError("choices is required in question_block")

    if isinstance(choices, dict):
        choices_list = []
        for letter in CHOICE_LETTERS:
            if letter in choices:
                choices_list.append(choices[letter])
            else:
                break
        choices = choices_list

    if not isinstance(choices, list):
        raise ValueError(f"choices must be a list or dict, got {type(choices)}")
    if len(choices) < 2:
        raise ValueError(f"at least 2 choices required, got {len(choices)}")

    original_gt_index = _normalize_mcq_ground_truth(choices, ground_truth)
    permutation = _build_mcq_choice_permutation(
        len(choices),
        enabled=shuffle_choices,
        mode=shuffle_mode,
        seed=shuffle_seed,
        question_id=question_block.get("question_id"),
        question_text=question_text,
    )
    choices = [choices[idx] for idx in permutation]
    ground_truth = CHOICE_LETTERS[permutation.index(original_gt_index)]

    lines = [f"Question: {question_text}"]
    for idx, choice in enumerate(choices):
        lines.append(f"{CHOICE_LETTERS[idx]}) {choice}")
    return "\n".join(lines), ground_truth


def build_mcq_messages(
    question_block: Dict[str, Any],
    use_few_shot_icl: bool = False,
    system_prompt: Optional[str] = None,
    *,
    shuffle_choices: bool = False,
    shuffle_seed: Optional[int] = None,
    shuffle_mode: str = "per_question_seeded",
) -> Tuple[List[Dict[str, str]], str]:
    """Build chat messages for MCQ answer generation."""
    formatted_question, normalized_gt = format_mcq_question(
        question_block,
        shuffle_choices=shuffle_choices,
        shuffle_seed=shuffle_seed,
        shuffle_mode=shuffle_mode,
    )

    if system_prompt is not None:
        effective_system_prompt = system_prompt
    elif use_few_shot_icl:
        effective_system_prompt = MCQ_ANSWER_GENERATION_ICL_SYSTEM_PROMPT
    else:
        effective_system_prompt = MCQ_ANSWER_GENERATION_SYSTEM_PROMPT

    if not use_few_shot_icl:
        messages = [
            {"role": "system", "content": effective_system_prompt},
            {
                "role": "user",
                "content": MCQ_ANSWER_GENERATION_PROMPT.format(question=formatted_question),
            },
        ]
        return messages, normalized_gt

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": effective_system_prompt},
    ]
    for example_user, example_assistant in MCQ_ANSWER_GENERATION_FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": example_assistant})
    messages.append(
        {
            "role": "user",
            "content": MCQ_ANSWER_GENERATION_ICL_PROMPT.format(
                question=formatted_question
            ),
        }
    )
    return messages, normalized_gt


def _extract_boxed_content(text: str, *, first: bool = False) -> Optional[str]:
    """Extract the content from the first/last \\boxed{...} in text.

    Handles nested braces and returns ``None`` if the selected box is missing
    or unbalanced.
    """
    if not text:
        return None

    positions = []
    for m in re.finditer(r'\\boxed\s*\{', text, re.IGNORECASE):
        positions.append(m.end())

    if not positions:
        return None

    start = positions[0] if first else positions[-1]
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth != 0:
        return None

    content = text[start:i - 1].strip()
    return content if content else None


def _extract_last_boxed_content(text: str) -> Optional[str]:
    """Extract the content from the last \\boxed{...} in text, handling nested braces."""
    return _extract_boxed_content(text, first=False)


def _extract_first_boxed_content(text: str) -> Optional[str]:
    """Extract the content from the first \\boxed{...} in text, handling nested braces."""
    return _extract_boxed_content(text, first=True)


def _strip_latex_wrappers(text: str) -> str:
    """Strip common LaTeX wrapper commands (\\text{}, \\textbf{}, etc.)."""
    pattern = r'^\\(?:text|textbf|textit|textrm|mathrm|mathbf|mathit|operatorname)\{(.+)\}$'
    prev = None
    result = text.strip()
    while result != prev:
        prev = result
        m = re.match(pattern, result, re.DOTALL)
        if m:
            result = m.group(1).strip()
    return result


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the final \\boxed{...} answer as a single letter."""
    content = _extract_last_boxed_content(text)
    if content is None:
        return None

    content = _strip_latex_wrappers(content)

    answer = content.upper().strip()
    if answer and answer[0].isalpha():
        return answer[0]
    # Return raw content for non-letter answers (e.g., numeric answers for AIME)
    return content.strip() if content.strip() else None


def extract_boxed_answer_general(text: str) -> Optional[str]:
    """Extract the full content from the last \\boxed{...} for general-form questions.

    Unlike ``extract_boxed_answer`` (MCQ, single letter), this returns the
    complete answer text with LaTeX wrappers stripped.
    """
    content = _extract_last_boxed_content(text)
    if content is None:
        return None
    return _strip_latex_wrappers(content)


def extract_first_boxed_answer_general(text: str) -> Optional[str]:
    """Extract the full content from the first \\boxed{...} for general-form questions."""
    content = _extract_first_boxed_content(text)
    if content is None:
        return None
    return _strip_latex_wrappers(content)


def is_correct(predicted: Optional[str], ground_truth: str) -> bool:
    """Compare predicted and ground-truth answers after normalization."""

    def normalize_answer(answer: str) -> str:
        if not answer:
            return ""
        return answer.upper().strip()

    if predicted is None:
        return False

    return normalize_answer(predicted) == normalize_answer(ground_truth)


def is_correct_general(predicted: Optional[str], ground_truth: str) -> bool:
    """Check if a predicted answer matches the ground truth for general-form questions.

    Uses case-insensitive comparison with numeric normalization
    (e.g. ``"42.0"`` matches ``"42"``).
    """

    def normalize_answer_general(answer: str) -> str:
        if not answer:
            return ""
        normalized = answer.strip().lower()
        try:
            num = float(normalized)
            if num == int(num):
                return str(int(num))
            else:
                return str(num).rstrip("0").rstrip(".")
        except ValueError:
            return normalized

    if predicted is None:
        return False

    return normalize_answer_general(predicted) == normalize_answer_general(ground_truth)


def _pool_worker_init(verify_timeout_inside_worker_s: float = 15.0) -> None:
    """Pool initializer: keep verify_math_answer's per-call ``fork+SIGKILL``
    timeout active inside each worker (the OS-delivered kill is the only thing
    that can stop sympy/mpmath C-bound code; SIGALRM inside the worker isn't
    delivered until the next bytecode boundary, which never comes for deep
    polynomial GCD or numerical-integration paths) and pre-import the verifier
    stack so the first task doesn't pay cold-start.

    Each worker is its own process, so per-call ``multiprocessing.fork()``
    runs uncontended — the ``_flush_std_streams`` global lock that serialized
    forks across the 32 ThreadPoolExecutor workers in the legacy
    single-process design is per-process here.

    Set ``verify_timeout_inside_worker_s=0`` to skip the verifier's inner
    fork entirely. The row-level child timeout in ``_pool_worker_score`` still
    bounds the full row.
    """
    # multiprocessing.Pool spawns workers with daemon=True by default, and
    # daemonic processes are forbidden from creating children
    # ("AssertionError: daemonic processes are not allowed to have children"
    # in Process.start). That breaks our fork-based per-call timeout, so flip
    # the config flag *inside* the worker to allow it. The worker is still
    # effectively daemonic (the main process's pool teardown will reap it);
    # we're just disabling the safety assertion. Children created by
    # _run_with_timeout are short-lived (≤15 s) and always exit before the
    # worker, so no orphan leak.
    import multiprocessing as _mp_local
    _mp_local.current_process()._config["daemon"] = False  # type: ignore[attr-defined]
    os.environ["VERL_MATH_VERIFY_TIMEOUT_S"] = str(float(verify_timeout_inside_worker_s))
    # Pre-import the verifier stack so cold-start latency is paid once at
    # worker spawn time, not on the first task.
    try:
        from verl_inf_evolve.sol_eval.benchmark_adapters import (  # noqa: F401
            score_response_for_question,
        )
        from verl_inf_evolve.utils.benchmarks.verifiers import (  # noqa: F401
            get_verifier,
        )
        from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import (  # noqa: F401
            verify_math_answer,
        )
        # Touch sympy + math_verify so their lazy imports happen now.
        import math_verify  # noqa: F401
        import sympy  # noqa: F401
    except Exception:
        # Keep going — workers without these imports can still service MCQ /
        # general-form rows; they only fail when a row needs a math verifier.
        pass


def _score_response_child(conn, question, response_text, allow_code_execution):
    """Run one row's scorer in a killable child process."""
    try:
        from verl_inf_evolve.sol_eval.benchmark_adapters import (
            score_response_for_question,
        )

        scored = score_response_for_question(
            question, response_text, allow_code_execution=allow_code_execution
        )
        conn.send(("ok", scored))
    except BaseException as exc:  # noqa: BLE001
        try:
            conn.send((f"error:{type(exc).__name__}", None))
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _pool_worker_score(payload):
    """Score one (question, response_text) pair inside a pool worker.

    The pool worker owns the row-level contract: it must return exactly one
    ``(idx, scored, status)`` tuple even if verifier code wedges.  The actual
    scoring call runs in a short-lived child process so timeout enforcement is
    an OS kill, not cooperative SIGALRM delivery inside SymPy/mpmath code.

    Returns ``(idx, ScoredResponse | None, status_str)`` where status is one
    of ``ok | timeout | error:<ExcName>``.
    """
    import multiprocessing as _mp

    idx, question, response_text, allow_code_execution, per_task_timeout_s = payload
    timeout_s = max(1.0, float(per_task_timeout_s))

    try:
        ctx = _mp.get_context("fork")
    except ValueError:
        ctx = _mp.get_context()

    parent_conn, child_conn = ctx.Pipe(duplex=False)
    child = ctx.Process(
        target=_score_response_child,
        args=(child_conn, question, response_text, allow_code_execution),
    )

    try:
        child.start()
        child_conn.close()
        child.join(timeout=timeout_s)

        if child.is_alive():
            try:
                child.kill()
            except AttributeError:
                child.terminate()
            child.join(timeout=1.0)
            return idx, None, "timeout"

        try:
            if parent_conn.poll(1.0):
                status, scored = parent_conn.recv()
            else:
                status, scored = "error:missing-result", None
        except (EOFError, OSError):
            status, scored = "error:missing-result", None

        return idx, scored, status
    finally:
        try:
            parent_conn.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            child_conn.close()
        except Exception:  # noqa: BLE001
            pass
        if child.is_alive():
            try:
                child.kill()
            except AttributeError:
                child.terminate()
            child.join(timeout=1.0)


def _build_score_payloads(
    *,
    batch_size: int,
    responses,
    ground_truths,
    is_mcq_flags,
    benchmark_types,
    data_sources,
    verifier_metas,
    tokenizer: Any,
    allow_code_execution: bool,
    per_task_timeout_s: float,
):
    """Decode all responses upfront in MainThread, build pickleable payloads."""
    payloads = []
    for i in range(batch_size):
        gt = str(ground_truths[i])
        mcq = bool(is_mcq_flags[i]) if is_mcq_flags is not None else True
        benchmark_type = (
            str(benchmark_types[i])
            if benchmark_types is not None and benchmark_types[i] is not None
            else ("qa_mcq" if mcq else "qa_open")
        )
        response_text = tokenizer.decode(responses[i], skip_special_tokens=True)
        data_source = str(data_sources[i]) if data_sources is not None else None
        meta = verifier_metas[i] if verifier_metas is not None else None
        question = {
            "benchmark_type": benchmark_type,
            "ground_truth": gt,
            "data_source": data_source or "",
            "verifier_metadata": meta,
        }
        payloads.append(
            (i, question, response_text, allow_code_execution, per_task_timeout_s)
        )
    return payloads


def extract_answer_scores(
    output: "DataProto",
    tokenizer: Any,
    *,
    allow_code_execution: bool = False,
    num_workers: Optional[int] = None,
) -> None:
    """Extract answer correctness scores from solver rollout output.

    Decodes solver responses, extracts answers from ``\\boxed{}``, and
    compares against normalized ground truth.

    Dispatches verification in this priority order:

    1. **Custom verifier** — looked up by ``data_source`` field via the
       verifier registry (``verl_inf_evolve.utils.benchmarks.verifiers``).  When a
       custom verifier is registered for the data source, it handles the
       comparison (e.g. sympy-based math for OlympiadBench).
    2. **MCQ** — single-letter comparison when ``is_mcq`` is ``True``.
    3. **General-form** — free-text comparison with numeric normalization.

    Falls back to MCQ when ``"is_mcq"`` is absent (backward compatibility).
    When ``data_source`` is absent, no custom verifier is triggered.

    Modifies ``output`` **in-place**, adding the following fields to
    ``output.non_tensor_batch``:

    - ``"extracted_answer"`` (``np.ndarray[object]``): The predicted answer
      string extracted from each response's ``\\boxed{}`` block, or ``None``
      if no ``\\boxed{}`` was found.
    - ``"answer_score"`` (``np.ndarray[object]``): Per-sample correctness
      score — ``1.0`` if the extracted answer matches ground truth
      (after normalization), ``0.0`` if it does not match, or ``None``
      if no answer could be extracted.
    - ``"primary_score"`` (optional ``np.ndarray[object]``): Benchmark-native
      continuous per-sample score (for example PHYBench EED on a ``0..100``
      scale). Populated only when a custom scorer is registered.

    Args:
        output: ``DataProto`` returned by ``solver_wg.generate_sequences()``.
            Must have ``batch["responses"]`` and ``non_tensor_batch`` with
            ``"question_id"`` and ``"ground_truth"``.
        tokenizer: Tokenizer used to decode solver response tokens.
        num_workers: Optional thread-pool size for parallel scoring. The
            slow path is the math verifier, which spawns a subprocess per
            sample with a timeout join (CPU-light but blocking). With many
            free-form math samples a single thread can take days; with
            32–64 threads each spawning its own verifier subprocess, we
            cut to minutes. If ``None`` (default), reads
            ``EXTRACT_ANSWER_SCORES_NUM_WORKERS`` env var (default 32).
            Set to ``1`` to force the legacy single-threaded path.
    """
    import numpy as np

    from verl_inf_evolve.sol_eval.benchmark_adapters import score_response_for_question

    responses = output.batch["responses"]  # [batch_size, response_len]
    ground_truths = output.non_tensor_batch["ground_truth"]
    is_mcq_flags = output.non_tensor_batch.get("is_mcq", None)
    benchmark_types = output.non_tensor_batch.get("benchmark_type", None)
    data_sources = output.non_tensor_batch.get("data_source", None)
    verifier_metas = output.non_tensor_batch.get("verifier_metadata", None)

    batch_size = responses.shape[0]

    if num_workers is None:
        num_workers = int(os.environ.get("EXTRACT_ANSWER_SCORES_NUM_WORKERS", "32"))

    def _score_one(i: int):
        gt = str(ground_truths[i])
        mcq = bool(is_mcq_flags[i]) if is_mcq_flags is not None else True
        benchmark_type = (
            str(benchmark_types[i])
            if benchmark_types is not None and benchmark_types[i] is not None
            else ("qa_mcq" if mcq else "qa_open")
        )
        response_text = tokenizer.decode(
            responses[i], skip_special_tokens=True
        )
        data_source = str(data_sources[i]) if data_sources is not None else None
        meta = verifier_metas[i] if verifier_metas is not None else None
        question = {
            "benchmark_type": benchmark_type,
            "ground_truth": gt,
            "data_source": data_source or "",
            "verifier_metadata": meta,
        }
        return score_response_for_question(
            question,
            response_text,
            allow_code_execution=allow_code_execution,
        )

    extracted_answers: list = [None] * batch_size
    per_sample_scores: list = [None] * batch_size
    primary_scores: list = [None] * batch_size
    primary_score_names: list = [None] * batch_size
    primary_score_scale_maxes: list = [None] * batch_size
    exec_results: list = [None] * batch_size
    any_primary_scores = False
    any_exec_results = False

    # Per-row wall-clock timeout enforced by _pool_worker_score's killable row
    # child. The pool worker itself stays alive and reports one result tuple.
    per_task_timeout_s = float(
        os.environ.get("EXTRACT_ANSWER_SCORES_PER_TASK_TIMEOUT_S", "30")
    )
    backend = os.environ.get("EXTRACT_ANSWER_SCORES_BACKEND", "process").lower()
    progress_every_n = max(1, batch_size // 20)  # log every 5%
    progress_every_s = float(os.environ.get("EXTRACT_ANSWER_SCORES_PROGRESS_EVERY_S", "30"))

    def _log_progress(label: str, completed: int, t0: float, n_timeouts: int, n_errors: int) -> None:
        elapsed = max(time.monotonic() - t0, 1e-6)
        rate = completed / elapsed
        remaining = max(0, batch_size - completed)
        eta = remaining / rate if rate > 1e-9 else float("inf")
        LOG.info(
            "[score] %s %d/%d (%.1f%%) rate=%.1f/s eta=%.0fs timeouts=%d errors=%d",
            label,
            completed,
            batch_size,
            completed * 100.0 / batch_size,
            rate,
            eta if eta != float("inf") else -1.0,
            n_timeouts,
            n_errors,
        )

    if num_workers <= 1 or batch_size <= 1:
        t0 = time.monotonic()
        last_log_t = t0
        n_timeouts_inner = 0
        n_errors_inner = 0
        # Drain any pre-existing counter so the summary reflects only this call.
        try:
            from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import (
                reset_timeout_count,
            )
            reset_timeout_count()
        except Exception:  # noqa: BLE001
            reset_timeout_count = None  # type: ignore[assignment]
        for i in range(batch_size):
            scored = _score_one(i)
            extracted_answers[i] = scored.extracted_answer
            per_sample_scores[i] = scored.answer_score
            primary_scores[i] = scored.primary_score
            primary_score_names[i] = scored.primary_score_name
            primary_score_scale_maxes[i] = scored.primary_score_scale_max
            exec_results[i] = scored.exec_result
            if scored.primary_score is not None:
                any_primary_scores = True
            if scored.exec_result is not None:
                any_exec_results = True
            now = time.monotonic()
            if (i + 1) % progress_every_n == 0 or (now - last_log_t) >= progress_every_s:
                _log_progress("serial", i + 1, t0, n_timeouts_inner, n_errors_inner)
                last_log_t = now
        if reset_timeout_count is not None:
            n_timeouts_inner = reset_timeout_count()
        _log_progress("serial-DONE", batch_size, t0, n_timeouts_inner, n_errors_inner)
    elif backend == "thread":
        # Legacy ThreadPoolExecutor path — kept as a fallback because each
        # call internally forks (verify_math_answer's _run_with_timeout) which
        # is contended on _flush_std_streams.  The "process" backend is
        # dramatically faster on free-form math at 32-way parallelism.
        from concurrent.futures import ThreadPoolExecutor

        try:
            from verl_inf_evolve.utils.benchmarks.verifiers._math_verifier import (
                reset_timeout_count,
            )
            reset_timeout_count()
        except Exception:  # noqa: BLE001
            reset_timeout_count = None  # type: ignore[assignment]

        t0 = time.monotonic()
        last_log_t = t0
        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for i, scored in zip(range(batch_size), pool.map(_score_one, range(batch_size))):
                extracted_answers[i] = scored.extracted_answer
                per_sample_scores[i] = scored.answer_score
                primary_scores[i] = scored.primary_score
                primary_score_names[i] = scored.primary_score_name
                primary_score_scale_maxes[i] = scored.primary_score_scale_max
                exec_results[i] = scored.exec_result
                if scored.primary_score is not None:
                    any_primary_scores = True
                if scored.exec_result is not None:
                    any_exec_results = True
                completed += 1
                now = time.monotonic()
                if completed % progress_every_n == 0 or (now - last_log_t) >= progress_every_s:
                    n_to_now = (
                        reset_timeout_count() if reset_timeout_count else 0
                    )
                    _log_progress("thread", completed, t0, n_to_now, 0)
                    last_log_t = now
        n_timeouts_total = (
            reset_timeout_count() if reset_timeout_count else 0
        )
        _log_progress("thread-DONE", batch_size, t0, n_timeouts_total, 0)
    else:
        # "process" backend (default): persistent multiprocessing.Pool of
        # spawn-context workers.  Each task pays only a queue-IPC round-trip;
        # the heavy sympy/math_verify imports happen ONCE at worker startup.
        # Per-task timeout via a killable child inside the worker
        # (see _pool_worker_score).
        # Avoids the per-call fork+_flush_std_streams serialization that made
        # the ThreadPool path bottleneck at ~0.4 verifications/sec.
        import multiprocessing as _mp

        payloads = _build_score_payloads(
            batch_size=batch_size,
            responses=responses,
            ground_truths=ground_truths,
            is_mcq_flags=is_mcq_flags,
            benchmark_types=benchmark_types,
            data_sources=data_sources,
            verifier_metas=verifier_metas,
            tokenizer=tokenizer,
            allow_code_execution=allow_code_execution,
            per_task_timeout_s=per_task_timeout_s,
        )

        chunksize = int(
            os.environ.get("EXTRACT_ANSWER_SCORES_CHUNKSIZE", "4")
        )
        chunksize = max(1, chunksize)
        maxtasksperchild = int(
            os.environ.get("EXTRACT_ANSWER_SCORES_MAXTASKSPERCHILD", "2000")
        )
        maxtasksperchild = max(50, maxtasksperchild)

        ctx = _mp.get_context("spawn")
        t0 = time.monotonic()
        last_log_t = t0
        completed = 0
        n_timeouts = 0
        n_errors = 0
        n_aborted = 0

        # Inner fork timeout: each worker runs verify_math_answer with a
        # ``multiprocessing.Process`` + ``p.kill()`` watchdog. SIGKILL is the
        # only thing that reliably stops sympy/mpmath C-bound code (SIGALRM is
        # only delivered at Python bytecode boundaries; deep polynomial GCD
        # and mpmath integration paths never check signals). Per-process fork
        # is uncontended because each worker has its own stdio flush lock.
        worker_inner_timeout_s = float(
            os.environ.get("EXTRACT_ANSWER_SCORES_WORKER_INNER_TIMEOUT_S", "15")
        )
        worker_inner_timeout_s = max(1.0, worker_inner_timeout_s)
        # Outer straggler watchdog (belt-and-suspenders): if ≥99% complete and
        # no new result in this many seconds, terminate the pool and mark the
        # remaining indices as None.  Backstop for the rare case where even
        # SIGKILL ⊕ fork() takes longer than expected.
        straggler_abort_after_s = float(
            os.environ.get("EXTRACT_ANSWER_SCORES_STRAGGLER_ABORT_S", "60")
        )
        straggler_abort_threshold = float(
            os.environ.get("EXTRACT_ANSWER_SCORES_STRAGGLER_ABORT_THRESHOLD", "0.99")
        )

        LOG.info(
            "[score] starting process-pool: workers=%d batch_size=%d chunksize=%d "
            "per_task_timeout_s=%.1f worker_inner_timeout_s=%.1f maxtasksperchild=%d "
            "straggler_abort_after_s=%.0f@>=%.2f%%",
            num_workers,
            batch_size,
            chunksize,
            per_task_timeout_s,
            worker_inner_timeout_s,
            maxtasksperchild,
            straggler_abort_after_s,
            straggler_abort_threshold * 100,
        )

        try:
            with ctx.Pool(
                processes=num_workers,
                initializer=_pool_worker_init,
                initargs=(worker_inner_timeout_s,),
                maxtasksperchild=maxtasksperchild,
            ) as pool:
                # Watchdog thread: terminates the pool if we cross the
                # straggler threshold and stop receiving results.
                import threading
                last_result_t = [time.monotonic()]
                aborted_flag = [False]
                stop_watchdog = threading.Event()

                def _straggler_watchdog():
                    while not stop_watchdog.is_set():
                        if stop_watchdog.wait(5.0):
                            return
                        progress_frac = completed / max(1, batch_size)
                        idle_s = time.monotonic() - last_result_t[0]
                        if (
                            progress_frac >= straggler_abort_threshold
                            and idle_s >= straggler_abort_after_s
                        ):
                            LOG.warning(
                                "[score] aborting %d stragglers (idle=%.0fs at %d/%d, "
                                "%.2f%%) — terminating pool",
                                batch_size - completed,
                                idle_s,
                                completed,
                                batch_size,
                                100 * progress_frac,
                            )
                            aborted_flag[0] = True
                            try:
                                pool.terminate()
                            except Exception:  # noqa: BLE001
                                pass
                            return

                watchdog = threading.Thread(target=_straggler_watchdog, daemon=True)
                watchdog.start()

                try:
                    for idx, scored, status in pool.imap_unordered(
                        _pool_worker_score, payloads, chunksize=chunksize
                    ):
                        if scored is not None:
                            extracted_answers[idx] = scored.extracted_answer
                            per_sample_scores[idx] = scored.answer_score
                            primary_scores[idx] = scored.primary_score
                            primary_score_names[idx] = scored.primary_score_name
                            primary_score_scale_maxes[idx] = scored.primary_score_scale_max
                            exec_results[idx] = scored.exec_result
                            if scored.primary_score is not None:
                                any_primary_scores = True
                            if scored.exec_result is not None:
                                any_exec_results = True
                        else:
                            if status == "timeout":
                                n_timeouts += 1
                            else:
                                n_errors += 1
                        completed += 1
                        last_result_t[0] = time.monotonic()
                        now = last_result_t[0]
                        if completed % progress_every_n == 0 or (now - last_log_t) >= progress_every_s:
                            _log_progress("process", completed, t0, n_timeouts, n_errors)
                            last_log_t = now
                except Exception as iter_exc:  # noqa: BLE001
                    if aborted_flag[0]:
                        n_aborted = batch_size - completed
                        LOG.warning(
                            "[score] watchdog aborted pool with %d items left "
                            "(treated as None): %r",
                            n_aborted,
                            iter_exc,
                        )
                    else:
                        raise
                finally:
                    stop_watchdog.set()
            _log_progress("process-DONE", completed, t0, n_timeouts, n_errors)
            if n_aborted:
                LOG.warning(
                    "[score] %d items aborted by straggler watchdog "
                    "(%.3f%% of batch); their answer_score = None",
                    n_aborted,
                    100 * n_aborted / max(1, batch_size),
                )
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                "process-pool extract_answer_scores failed (%r); "
                "falling back to ThreadPool legacy path",
                exc,
            )
            # Fallback: legacy ThreadPool path so we don't hard-fail mid-run.
            from concurrent.futures import ThreadPoolExecutor

            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for i, scored in zip(
                    range(batch_size), pool.map(_score_one, range(batch_size))
                ):
                    extracted_answers[i] = scored.extracted_answer
                    per_sample_scores[i] = scored.answer_score
                    primary_scores[i] = scored.primary_score
                    primary_score_names[i] = scored.primary_score_name
                    primary_score_scale_maxes[i] = scored.primary_score_scale_max
                    exec_results[i] = scored.exec_result
                    if scored.primary_score is not None:
                        any_primary_scores = True
                    if scored.exec_result is not None:
                        any_exec_results = True
            _log_progress("thread-fallback-DONE", batch_size, t0, 0, 0)

    # Store per-sample data into non_tensor_batch for downstream consumers
    output.non_tensor_batch["extracted_answer"] = np.array(extracted_answers, dtype=object)
    output.non_tensor_batch["answer_score"] = np.array(per_sample_scores, dtype=object)
    if any_primary_scores:
        output.non_tensor_batch["primary_score"] = np.array(primary_scores, dtype=object)
        output.non_tensor_batch["primary_score_name"] = np.array(primary_score_names, dtype=object)
        output.non_tensor_batch["primary_score_scale_max"] = np.array(
            primary_score_scale_maxes, dtype=object
        )
    if any_exec_results:
        output.non_tensor_batch["exec_result"] = np.array(exec_results, dtype=object)


def group_scores_by_qid(output: "DataProto") -> Dict[str, List[Optional[float]]]:
    """Derive ``{question_id: [score, ...]}`` view from DataProto's non_tensor_batch."""
    from verl import DataProto  # noqa: F811 — deferred to avoid heavy import at module level

    qids = output.non_tensor_batch["question_id"]
    scores = output.non_tensor_batch["answer_score"]
    grouped: Dict[str, List[Optional[float]]] = {}
    for qid, score in zip(qids, scores):
        qid = str(qid)
        s = None if score is None else float(score)
        grouped.setdefault(qid, []).append(s)
    return grouped


__all__ = [
    "build_mcq_messages",
    "extract_answer_scores",
    "extract_boxed_answer",
    "extract_boxed_answer_general",
    "group_scores_by_qid",
    "is_correct",
    "is_correct_general",
]
