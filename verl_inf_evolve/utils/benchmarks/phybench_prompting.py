"""PHYBench prompt helpers aligned to the public OpenCompass integration.

When ``eval.use_public_eval_prompt`` is enabled for ``data_source=phybench``,
we switch from the repo-generic free-form ICL prompt to the public zero-shot
prompt published in OpenCompass:

https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/PHYBench/phybench_gen.py
"""

from __future__ import annotations


PHYBENCH_PUBLIC_ZERO_SHOT_PROMPT = (
    "Solve the following physics problem and return only the final result as a "
    "clean LaTeX expression.Remember to put your final answer within \\boxed{}."
)


def build_phybench_icl_messages(
    question: dict[str, object],
) -> tuple[list[dict[str, str]], str]:
    """Build the public OpenCompass-style zero-shot PHYBench prompt."""

    question_text = str(question.get("question_text", "") or "").strip()
    prompt = (
        f"{PHYBENCH_PUBLIC_ZERO_SHOT_PROMPT}\n\n"
        f"Question: {question_text}\n"
        "Answer: "
    )
    return [{"role": "user", "content": prompt}], str(question.get("ground_truth", ""))


def extract_phybench_answer(response_text: str | None) -> str | None:
    """Extract a PHYBench prediction using OpenCompass-style boxed fallback."""

    from verl_inf_evolve.utils.mcq_utils import extract_boxed_answer_general

    if not response_text:
        return None

    boxed = extract_boxed_answer_general(response_text)
    if boxed is not None:
        return boxed

    lines = str(response_text).strip().splitlines()
    if not lines:
        return None
    last_line = lines[-1].strip()
    return last_line or None
