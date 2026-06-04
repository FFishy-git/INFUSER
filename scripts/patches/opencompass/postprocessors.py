"""Fixed postprocessors for OpenCompass MCQ evaluation.

OC 0.5.2's first_option_postprocess has a bug with 5-shot CoT configs:
it extracts 'L' from "Answer: Let's think step by step" because the
ANSWER: pattern matches before the actual "The answer is (X)" at the end.

Additionally, base models with temperature>0 often continue generating
new questions after answering, contaminating last-match extraction.

This module provides last_option_postprocess which:
1. Splits on "Question:" to isolate the first answer (fixes continuation)
2. Takes the LAST match position across all patterns (fixes CoT preamble)
"""

from __future__ import annotations

import re

from opencompass.registry import TEXT_POSTPROCESSORS


@TEXT_POSTPROCESSORS.register_module()
def identity_postprocess(text: str) -> str:
    """No-op postprocessor — returns raw prediction unchanged."""
    return text


@TEXT_POSTPROCESSORS.register_module()
def humaneval_extract_code(text: str) -> str:
    """Extract code from markdown blocks but preserve indentation.

    Replaces OC's humaneval_postprocess_v2 which calls lstrip() and destroys
    function-body indentation.  This extracts the code block content (if any)
    without stripping whitespace, so the evaluator receives valid Python that
    the tree-sitter sanitizer can clean up.
    """
    blocks = re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)
    if len(blocks) >= 1:
        text = blocks[0]
    return text


@TEXT_POSTPROCESSORS.register_module()
def last_option_postprocess(text: str, options: str = 'ABCDEFGHIJKLMNOP') -> str:
    """Extract the last option letter from text, splitting on continuation.

    Fixes two bugs in first_option_postprocess:
    - Bug 1: first match returns 'L' from "Answer: Let's think step by step"
    - Bug 2: model continues generating new Q&A, last match is wrong question
    """
    # Fix 2: isolate the answer to the ORIGINAL question
    parts = re.split(r'\nQuestion:', text, maxsplit=1)
    text = parts[0].strip()

    patterns = [
        rf'[Tt]he answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        rf'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        rf'(?i)ANSWER\s*:\s*([{options}])',
        rf'[\s,:：]([{options}])[.。,]?$',
    ]

    # Fix 1: take the LAST match position, not the first
    last_match = None
    last_pos = -1
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.DOTALL):
            if m.start() > last_pos:
                last_pos = m.start()
                last_match = m.group(1)

    return last_match or ''
