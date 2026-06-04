"""Per-benchmark custom verifier registry.

Each benchmark can register a verifier function keyed by its ``data_source``
string.  ``extract_answer_scores()`` in ``mcq_utils.py`` looks up a custom
verifier first; when none is registered it falls back to the existing
MCQ / general-form comparison logic.

Verifier signature::

    (predicted: str | None, ground_truth: str, metadata: dict | None) -> bool | None

Return ``True`` / ``False`` for correct / incorrect, or ``None`` if the
verifier cannot determine correctness (treated the same as "no answer
extracted").
"""

from __future__ import annotations

from typing import Callable, Optional

# Type alias for verifier functions.
VerifierFn = Callable[
    [Optional[str], str, Optional[dict]],  # predicted, ground_truth, metadata
    Optional[bool],                         # True/False/None
]

_REGISTRY: dict[str, VerifierFn] = {}
_NEEDS_FULL_RESPONSE: set[str] = set()


def register(name: str, *, needs_full_response: bool = False):
    """Decorator that registers a verifier function under *name*.

    Args:
        name: The data_source string that triggers this verifier.
        needs_full_response: When ``True``, the verifier receives the full
            decoded model response instead of just the extracted boxed answer.

    Usage::

        @register("olympiadbench")
        def verify(predicted, ground_truth, metadata):
            ...

        @register("lpfqa", needs_full_response=True)
        def verify(predicted, ground_truth, metadata):
            ...
    """
    def decorator(fn: VerifierFn) -> VerifierFn:
        if name in _REGISTRY:
            raise ValueError(f"Verifier already registered for '{name}'")
        _REGISTRY[name] = fn
        if needs_full_response:
            _NEEDS_FULL_RESPONSE.add(name)
        return fn
    return decorator


def get_verifier(data_source: str) -> Optional[VerifierFn]:
    """Look up a custom verifier by *data_source*.

    Returns ``None`` when no custom verifier is registered (caller should
    fall back to existing logic).
    """
    return _REGISTRY.get(data_source)


def get_needs_full_response(data_source: str) -> bool:
    """Return whether the verifier for *data_source* needs the full response.

    Returns ``False`` when no verifier is registered or when the verifier
    uses the default (boxed answer extraction).
    """
    return data_source in _NEEDS_FULL_RESPONSE


# Auto-import built-in verifier modules so their @register decorators run.
from verl_inf_evolve.utils.benchmarks.verifiers import aime as _aime  # noqa: F401, E402
from verl_inf_evolve.utils.benchmarks.verifiers import olympiadbench as _olympiadbench  # noqa: F401, E402
from verl_inf_evolve.utils.benchmarks.verifiers import hmmt as _hmmt  # noqa: F401, E402
from verl_inf_evolve.utils.benchmarks.verifiers import math500 as _math500  # noqa: F401, E402
from verl_inf_evolve.utils.benchmarks.verifiers import lpfqa as _lpfqa  # noqa: F401, E402
from verl_inf_evolve.utils.benchmarks.verifiers import bbeh_open as _bbeh_open  # noqa: F401, E402
from verl_inf_evolve.utils.benchmarks.verifiers import phybench as _phybench  # noqa: F401, E402
from verl_inf_evolve.utils.benchmarks.verifiers import amc as _amc  # noqa: F401, E402
from verl_inf_evolve.utils.benchmarks.verifiers import minerva_math as _minerva_math  # noqa: F401, E402
from verl_inf_evolve.utils.benchmarks.verifiers import putnam as _putnam  # noqa: F401, E402
