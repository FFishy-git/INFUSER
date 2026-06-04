"""AIME verifier — integer comparison with normalization.

AIME answers are integers 0-999.  This verifier normalizes both predicted
and ground truth to ``int(float(x))`` to handle edge cases like
``"042"`` vs ``"42"`` or ``"42.0"`` vs ``"42"``.

Ground truth from some HuggingFace datasets (e.g. opencompass/AIME2025)
may contain LaTeX suffixes like ``336^\\circ``.  We strip those before
comparing.
"""

from __future__ import annotations

import re
from typing import Optional

from verl_inf_evolve.utils.benchmarks.verifiers import register

# LaTeX suffixes that may appear after a numeric AIME answer.
_LATEX_SUFFIX_RE = re.compile(
    r"[\s]*"
    r"(?:"
    r"\\?(?:\^\\?circ|°)"  # ^\circ, \circ, °
    r"|\\%|%"              # \%, %
    r"|\\text\{[^}]*\}"   # \text{...}
    r"|\\mathrm\{[^}]*\}" # \mathrm{...}
    r")"
    r"[\s]*$"
)


def _strip_latex_suffix(s: str) -> str:
    """Remove trailing LaTeX unit annotations from a numeric string."""
    return _LATEX_SUFFIX_RE.sub("", s)


@register("aime")
def verify(
    predicted: Optional[str],
    ground_truth: str,
    metadata: Optional[dict] = None,
) -> Optional[bool]:
    """Verify an AIME answer by integer comparison."""
    if predicted is None:
        return None

    predicted = _strip_latex_suffix(predicted.strip())
    ground_truth = _strip_latex_suffix(ground_truth.strip())

    if not predicted:
        return None

    try:
        p = int(float(predicted))
        g = int(float(ground_truth))
        return p == g
    except (ValueError, TypeError, OverflowError):
        # Fall back to exact string comparison
        return predicted == ground_truth
