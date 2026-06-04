"""MATH-500 prompt helpers for the public evaluation path.

When ``eval.use_public_eval_prompt`` is enabled for ``data_source=math500``,
we switch from the repo-generic chat-format prompt to a plain-text prompt that
bypasses the chat template.  This matches the OpenCompass MATHVerifyEvaluator
approach (0-shot, raw text).
"""

from __future__ import annotations

import re
from typing import Any

from verl_inf_evolve.utils.mcq_utils import (
    extract_boxed_answer_general,
    extract_first_boxed_answer_general,
)


_MATH500_FEW_SHOTS: list[dict[str, str]] = [
    {
        "problem": r"Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}",
        "solution": (
            r"The expressions inside each square root must be non-negative. Therefore, "
            r"$x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. "
            r"Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. "
            r"Therefore, the domain of the expression is $\boxed{[2,5)}$.\n"
            r"Final Answer: The final answer is $[2,5)$. I hope it is correct."
        ),
    },
    {
        "problem": r"If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$",
        "solution": (
            r"We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = "
            r"(2)(12) = \boxed{24}.$\n"
            r"Final Answer: The final answer is $24$. I hope it is correct."
        ),
    },
    {
        "problem": (
            "Terrell usually lifts two 20-pound weights 12 times. If he uses two "
            "15-pound weights instead, how many times must Terrell lift them in order "
            "to lift the same total weight?"
        ),
        "solution": (
            r"If Terrell lifts two 20-pound weights 12 times, he lifts a total of "
            r"$2\cdot 12\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights "
            r"instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds "
            r"of weight.  Equating this to 480 pounds, we can solve for $n$:\n"
            r"\begin{align*}"
            r"\n30n&=480\\"
            r"\n\Rightarrow\qquad n&=480/30=\boxed{16}"
            r"\n\end{align*}\n"
            r"Final Answer: The final answer is $16$. I hope it is correct."
        ),
    },
    {
        "problem": (
            "If the system of equations\n\n"
            r"\begin{align*}"
            r"\n6x-4y&=a,\\"
            r"\n6y-9x &=b."
            r"\n\end{align*}"
            "has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\n"
            r"find $\frac{a}{b},$ assuming $b$ is nonzero."
        ),
        "solution": (
            r"If we multiply the first equation by $-\frac{3}{2}$, we obtain\n\n"
            r"$$6y-9x=-\frac{3}{2}a.$$"
            r"Since we also know that $6y-9x=b$, we have\n\n"
            r"$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$\n"
            r"Final Answer: The final answer is $-\frac{2}{3}$. I hope it is correct."
        ),
    },
]

_SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

_REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def build_math500_icl_messages(
    question: dict[str, Any],
    *,
    num_fewshot: int = 0,
) -> tuple[list[dict[str, str]], str]:
    """Build a plain-text MATH-500 prompt aligned with OpenCompass.

    Returns a single ``user`` message so that ``eval_core`` detects the
    ICL shape and sets ``prompt_text``, which bypasses
    ``tokenizer.apply_chat_template()`` in the vLLM runtime.  This is
    critical for base models where the chat-template wrapper degrades
    per-answer consistency.

    When *num_fewshot* is 0 (default), produces the same 0-shot prompt
    that OpenCompass uses::

        {question}
        Please reason step by step, and put your final answer within \\boxed{}.

    When *num_fewshot* > 0, prepends Minerva-style worked exemplars.
    """

    question_text = str(question.get("question_text", "")).strip()

    if num_fewshot == 0:
        prompt = (
            question_text
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
    else:
        prompt_parts: list[str] = []
        for example in _MATH500_FEW_SHOTS[:num_fewshot]:
            prompt_parts.append(
                f"Problem:\n{example['problem']}\n\nSolution:\n{example['solution']}"
            )
        prompt_parts.append(
            "Problem:\n" + question_text + "\n\nSolution:"
        )
        prompt = "\n\n".join(prompt_parts)

    return [{"role": "user", "content": prompt}], str(question.get("ground_truth", ""))


def _normalize_final_answer(final_answer: str) -> str:
    """Normalize a Minerva-style final answer string."""

    answer = final_answer.split("=")[-1]
    for before, after in _SUBSTITUTIONS:
        answer = answer.replace(before, after)
    for expr in _REMOVED_EXPRESSIONS:
        answer = answer.replace(expr, "")

    answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", answer)
    answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", answer)
    answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", answer)
    answer = answer.replace("$", "")
    if answer.replace(",", "").isdigit():
        answer = answer.replace(",", "")
    return answer


def extract_math500_answer(response_text: str | None) -> str | None:
    """Extract a Math500 answer from boxed or Minerva/ANSWER-line output."""

    if not response_text:
        return None

    boxed = extract_boxed_answer_general(response_text)
    if boxed is not None:
        return boxed

    # Some MATH500 traces emit a valid first boxed answer, then continue
    # generating into an incomplete trailing box. Only this extractor retries
    # the first box so other benchmarks keep their current last-box semantics.
    first_boxed = extract_first_boxed_answer_general(response_text)
    if first_boxed is not None:
        return first_boxed

    text = str(response_text)
    match = re.search(
        r"Final Answer:\s*The final answer is(.*?)(?:\. I hope it is correct\.|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        extracted = match.group(1).strip()
        return _normalize_final_answer(extracted) if extracted else None

    matches = re.findall(r"(?im)^ANSWER:\s*(.+)$", text)
    if matches:
        extracted = matches[-1].strip()
        return _normalize_final_answer(extracted) if extracted else None

    return None
