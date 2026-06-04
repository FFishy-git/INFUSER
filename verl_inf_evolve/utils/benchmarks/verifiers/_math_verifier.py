"""Shared math verification logic used by multiple benchmark verifiers.

Combines ``math_verify`` (symbolic equivalence) with ``_math_judger``
(``\\dfrac`` normalisation, space-stripped matching, sympy) in a union
strategy: if *either* engine returns ``True``, the answer is accepted.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from typing import Optional

LOG = logging.getLogger(__name__)

# Outer wall-clock budget for a single verification subprocess.  Default 15 s,
# deliberately larger than math_verify's *own* internal timeout (``@timeout(5)``
# on ``compare_single_extraction``) so the library can cap its strategy cascade
# first and our fork wrapper only fires when something sympy cannot interrupt
# hangs the child.  Set to <= 0 to disable the subprocess wrapper entirely
# (useful when the caller already runs each verify in its own process — e.g. a
# persistent ProcessPool worker that uses SIGALRM for timeouts).
#
# Read at *call time* rather than module import so that pool workers can set
# the env var in their initializer without re-importing this module.
_DEFAULT_VERIFY_TIMEOUT_S = 15.0


def _current_timeout_s() -> float:
    raw = os.environ.get("VERL_MATH_VERIFY_TIMEOUT_S")
    if raw is None or raw == "":
        return _DEFAULT_VERIFY_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_VERIFY_TIMEOUT_S


# Per-thread timeout counter so callers (e.g. ``extract_answer_scores``) can
# emit a single "N timeouts in this batch" summary instead of letting every
# call log a WARNING that contends the global Python logging lock under heavy
# parallelism.
_timeout_counter_lock = threading.Lock()
_timeout_counter = 0


def reset_timeout_count() -> int:
    """Atomically read and zero the running timeout counter."""
    global _timeout_counter
    with _timeout_counter_lock:
        n, _timeout_counter = _timeout_counter, 0
    return n


def _record_timeout() -> None:
    global _timeout_counter
    with _timeout_counter_lock:
        _timeout_counter += 1


# Use fork context — child inherits parent memory (sympy caches) without
# pickling.  Avoids GIL contention with vLLM/CUDA/Ray worker threads that
# caused thread.join(timeout) to hang indefinitely.
_mp_ctx = multiprocessing.get_context("fork")


def _run_with_timeout(func, *args):
    """Run ``func(*args)`` in a subprocess with a timeout.

    Uses a forked process so that:
    1. Pathological sympy computation cannot hold the parent's GIL.
    2. Stuck workers are killed with SIGKILL — no zombie thread accumulation.
    3. signal monkey-patching is isolated to the child process.

    When ``VERL_MATH_VERIFY_TIMEOUT_S`` is set to ``0`` (or negative), skips
    the fork entirely and runs ``func`` in-process — the caller is responsible
    for enforcing its own timeout.  This is the path persistent ProcessPool
    workers take to avoid fork-per-call serialization on ``_flush_std_streams``.
    """
    timeout_s = _current_timeout_s()
    if timeout_s <= 0:
        try:
            return func(*args)
        except Exception:
            return None

    parent_conn, child_conn = _mp_ctx.Pipe(duplex=False)

    def _worker(conn):
        try:
            result = func(*args)
            conn.send(result)
        except Exception:
            conn.send(None)
        finally:
            conn.close()

    p = _mp_ctx.Process(target=_worker, args=(child_conn,))
    try:
        p.start()
        child_conn.close()
    except OSError:
        # Fork failed (resource exhaustion) — fall back to in-process.
        parent_conn.close()
        child_conn.close()
        try:
            return func(*args)
        except Exception:
            return None

    p.join(timeout=timeout_s)

    if p.is_alive():
        p.kill()
        p.join(timeout=1.0)
        parent_conn.close()
        _record_timeout()
        LOG.debug("Math verification timed out after %.2fs", timeout_s)
        return None

    try:
        result = parent_conn.recv() if parent_conn.poll(0) else None
    except (EOFError, OSError):
        result = None
    finally:
        parent_conn.close()

    return result


def _check_with_math_verify(predicted: str, ground_truth: str) -> Optional[bool]:
    """Try math_verify; return True/False or None on parse/compare failure.

    Wraps inputs in ``$...$`` and uses explicit extraction configs to match
    the OpenCompass MATHVerifyEvaluator behaviour.  Without the dollar-sign
    wrapper, ``math_verify.parse`` fails on compact LaTeX such as
    ``\\frac43`` (no braces) or ``\\frac 59`` (space-separated).

    Runs inside ``_run_with_timeout`` (forked subprocess).  math_verify's own
    ``@timeout`` decorator installs a SIGALRM handler inside the child, which
    is safe here because the child is fully isolated from the parent's
    threads/GIL/GC — the SIGALRM that originally caused WeakSet-cleanup
    deadlock on the main thread (see commit 450c3dec) cannot leak out.
    Letting math_verify's internal 5 s cap fire first means its strategy
    cascade resolves in ≤5 s rather than running up against the outer wall
    (typically 15 s), which cuts pathological-case latency by ~5-6x.
    """

    def _inner():
        from math_verify import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
            verify,
        )

        _extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()]
        parsed_pred = parse(
            f"${predicted}$",
            extraction_mode="first_match",
            extraction_config=_extraction_config,
        )
        parsed_gt = parse(
            f"${ground_truth}$",
            extraction_mode="first_match",
            extraction_config=_extraction_config,
        )
        return bool(verify(parsed_pred, parsed_gt))

    return _run_with_timeout(_inner)


import re as _re


# Regexes for cheap, unambiguous canonicalizations applied before
# ``math_verify.parse``. We only apply rewrites that have a single defensible
# mathematical interpretation:
#   * ``\%`` and ``N%`` (digit-attached) → ``/100``
#     ``%`` has no other LaTeX meaning; the conversion is exact.
# We deliberately do NOT canonicalize ``X!^k`` even though ``math_verify``
# can't parse it. Postfix factorial followed by an exponent is ambiguous as
# written (could be ``(X!)^k`` or ``X!^k`` rendered with the ``^`` binding
# differently), so silently rewriting could mask a malformed GT. The
# stage-3 judge sees ``gt_self_parseable=False`` and can decide.
_PERCENT_LATEX = _re.compile(r"\\%")
_PERCENT_BARE = _re.compile(r"(?<=\d)%")


def canonicalize_for_math_verify(text: str) -> str:
    """Apply unambiguous rewrites that make ``text`` more parseable by
    ``math_verify``. Idempotent. Returns ``text`` unchanged if nothing applies.
    """
    if not text:
        return text
    text = _PERCENT_LATEX.sub("/100", text)
    text = _PERCENT_BARE.sub("/100", text)
    return text


def is_math_verifier_parseable(text: Optional[str]) -> bool:
    """Cheap in-process check: can ``math_verify`` symbolically parse ``text``?

    Returns True iff ``math_verify.parse('$text$')`` produces at least one
    non-string element (i.e. a sympy/numeric object), which is the signal
    that the verifier could *symbolically* interpret the input. A purely
    string fallback (e.g. ``['x^2 + y']`` for malformed LaTeX) returns False.

    Before parsing, we apply ``canonicalize_for_math_verify`` to recover
    common forms the parser doesn't natively handle, currently only ``%``
    percentages.

    No subprocess fork, no fallback cascade — meant for offline batch
    filtering of generated ground truths where ``verify_math_answer``'s
    online-rollout scaffolding is overkill.
    """
    if not text:
        return False
    canonical = canonicalize_for_math_verify(text)
    try:
        from math_verify import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
        )

        out = parse(
            f"${canonical}$",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        )
    except Exception:
        return False
    if not out:
        return False
    return any(not isinstance(p, str) for p in out)


def _is_math_verifier_parseable_worker(text: Optional[str]) -> bool:
    return is_math_verifier_parseable(text)


def is_math_verifier_parseable_many(
    texts: list[Optional[str]],
    *,
    workers: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> list[bool]:
    """Batch ``is_math_verifier_parseable`` with optional process parallelism.

    ``math_verify.parse`` can be expensive on generated free-form ground
    truths. Use a spawned process pool for batch training/offline paths when
    ``workers`` (or ``VERL_GT_PARSE_WORKERS``) is greater than 1. Spawned
    workers avoid inheriting Ray/vLLM/CUDA state from the parent process.

    Default is 32 workers; the clamp ``min(workers, len(texts))`` keeps small
    batches from over-spawning. Set ``VERL_GT_PARSE_WORKERS=1`` to force the
    legacy serial path.
    """
    if not texts:
        return []

    if workers is None:
        workers = int(os.environ.get("VERL_GT_PARSE_WORKERS", "32") or "32")
    workers = max(1, min(int(workers), len(texts)))
    if workers <= 1:
        return [is_math_verifier_parseable(text) for text in texts]

    if chunksize is None:
        chunksize = int(os.environ.get("VERL_GT_PARSE_CHUNKSIZE", "16") or "16")
    chunksize = max(1, int(chunksize))

    ctx = multiprocessing.get_context("spawn")
    try:
        with ctx.Pool(processes=workers) as pool:
            return list(
                pool.imap(
                    _is_math_verifier_parseable_worker,
                    texts,
                    chunksize=chunksize,
                )
            )
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "Parallel math parseability check failed with %d workers; "
            "falling back to serial path: %r",
            workers,
            exc,
        )
        return [is_math_verifier_parseable(text) for text in texts]


def verify_math_answer(
    predicted: Optional[str],
    ground_truth: str,
    answer_type: str = "",
) -> Optional[bool]:
    """Verify a math answer using math_verify + _math_judger union strategy.

    Args:
        predicted: Extracted answer string (or None if no answer extracted).
        ground_truth: Reference answer string.
        answer_type: OlympiadBench answer type hint (e.g. "Numerical",
            "Expression"). Pass "" for benchmarks that don't use it.

    Returns:
        True/False for correct/incorrect, or None if undetermined.
    """
    if predicted is None:
        return None

    cleaned = predicted.strip()
    if not cleaned:
        return None

    gt = ground_truth.strip()

    # 1. Try math_verify first (runs in a subprocess with timeout).
    mv_result = _check_with_math_verify(cleaned, gt)
    if mv_result is True:
        return True

    # 2. Fall back to custom judge (handles \dfrac, spacing, sympy).
    #    Also runs in a subprocess with timeout.
    from verl_inf_evolve.utils.benchmarks.verifiers._math_judger import judge

    def _judge():
        return judge(
            predicted=cleaned,
            ground_truth=gt,
            answer_type=answer_type,
        )

    custom_result = _run_with_timeout(_judge)
    if custom_result is True:
        return True

    if mv_result is False or custom_result is False:
        return False

    return None
