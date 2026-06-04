"""LPFQA verifier — GPT API judge for long-form professional QA.

LPFQA answers are evaluated by an LLM judge (GPT-4o-mini) using a
template-based prompt.  The verifier receives the full model response
(via ``needs_full_response=True``) and constructs a judge prompt by
substituting ``{response_reference}`` and ``{response}`` into the
template provided in the per-question metadata.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

from verl_inf_evolve.utils.benchmarks.verifiers import register

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds


def _call_judge(
    judge_prompt: str,
    judge_system_prompt: str,
) -> Optional[bool]:
    """Call GPT-4o-mini to judge a response.  Returns True/False/None."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; LPFQA verifier returning None")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed; LPFQA verifier returning None")
        return None

    client = OpenAI(api_key=api_key)

    backoff = _INITIAL_BACKOFF
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": judge_system_prompt},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.0,
                max_tokens=64,
            )
            judge_text = response.choices[0].message.content or ""
            return _parse_score(judge_text)
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                logger.warning(
                    "LPFQA judge API error (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    e,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.warning(
                    "LPFQA judge API error after %d attempts: %s; returning None",
                    _MAX_RETRIES,
                    e,
                )
                return None
    return None  # unreachable, but satisfies type checker


def _parse_score(judge_text: str) -> Optional[bool]:
    """Extract binary score (0 or 1) from judge response text."""
    # Look for a standalone digit 0 or 1 in the judge output.
    # The judge template asks for a score of 0 or 1.
    matches = re.findall(r"\b([01])\b", judge_text)
    if matches:
        # Use the last match (judges often explain before giving score)
        return matches[-1] == "1"
    logger.warning("Could not parse score from judge response: %r", judge_text)
    return None


@register("lpfqa", needs_full_response=True)
def verify(
    predicted: Optional[str],
    ground_truth: str,
    metadata: Optional[dict] = None,
) -> Optional[bool]:
    """Verify an LPFQA answer using GPT-4o-mini as LLM judge."""
    if predicted is None or not predicted.strip():
        return None

    if metadata is None:
        logger.warning("LPFQA verifier called without metadata; returning None")
        return None

    template = metadata.get("judge_prompt_template", "")
    system_prompt = metadata.get("judge_system_prompt", "")
    reference = metadata.get("response_reference", "")

    if not template:
        logger.warning("LPFQA verifier: missing judge_prompt_template; returning None")
        return None

    # Construct judge prompt from template
    judge_prompt = template.replace("{response_reference}", reference).replace(
        "{response}", predicted
    )

    return _call_judge(judge_prompt, system_prompt)
