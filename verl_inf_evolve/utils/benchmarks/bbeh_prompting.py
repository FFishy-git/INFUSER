"""BBEH answer extraction helpers.

BBEH subtasks (e.g. ``sarc_triples``, ``sportqa``) embed their own answer-format
instruction in the question text ("provide your answer as a comma-separated
list"), which Qwen3 follows in preference to the system prompt's
``put your final answer in \\boxed{}`` instruction. The model then emits
``Final Answer:\\n<x>`` style outputs that the plain ``\\boxed{...}`` extractor
cannot read. The fuzzy extractor below tries ``\\boxed{}`` first, then falls
back to ``Final Answer: ...`` / ``The answer is: ...`` / ``Answer: ...`` —
matching the behavior of the General-Reasoner evaluator that we compare to.
"""

from __future__ import annotations

import re
from typing import Optional

from verl_inf_evolve.utils.mcq_utils import extract_boxed_answer_general


_BBEH_FUZZY_PATTERNS: list[re.Pattern[str]] = [
    # Priority 1: "Final Answer" (most authoritative; matches `**Final Answer:**`,
    # `### Final Answer`, `Final Answer -`, etc.)
    re.compile(
        r"(?:^|\n)[#*\s]*Final\s+Answer[#*\s]*[:\-][#*\s]*(.+?)(?:\n\s*\n|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    # Priority 2: "The (final) answer is"
    re.compile(
        r"(?:^|\n)[#*\s]*The\s+(?:final\s+)?answer\s+is[#*\s]*:?[#*\s]*(.+?)(?:\n\s*\n|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    # Priority 3: plain "Answer:" on its own line (last resort)
    re.compile(
        r"(?:^|\n)[#*\s]*Answer[#*\s]*:[#*\s]*(.+?)(?:\n\s*\n|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
]


def extract_bbeh_answer(response_text: str) -> Optional[str]:
    """Extract a BBEH answer from a free-form response.

    Tries ``\\boxed{...}`` first (backward-compat with sol_eval's default system
    prompt), then falls back to the fuzzy ``Final Answer:`` family.
    Returns the captured text (whitespace-stripped, outer markdown removed) or
    ``None`` if nothing matched.
    """
    boxed = extract_boxed_answer_general(response_text)
    if boxed is not None and boxed.strip():
        return boxed

    for pattern in _BBEH_FUZZY_PATTERNS:
        matches = list(pattern.finditer(response_text))
        if not matches:
            continue
        candidate = matches[-1].group(1).strip()
        candidate = candidate.strip("*`_ \n\t").rstrip(":")
        if candidate:
            return candidate
    return None
