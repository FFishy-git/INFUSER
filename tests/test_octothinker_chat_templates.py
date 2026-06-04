from __future__ import annotations

import os
import sys

import jinja2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.sol_eval.eval_core import build_benchmark_messages
from verl_inf_evolve.templates import load_source, render
from verl_inf_evolve.utils.mcq_utils import build_mcq_messages
from verl_inf_evolve.utils.prompts import (
    MCQ_QUESTION_GENERATION_PROMPT,
    MCQ_QUESTION_GENERATION_SYSTEM_PROMPT,
    build_free_form_messages,
)


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        total_len = sum(len(str(m.get("content", ""))) for m in messages)
        return [0] * total_len


def _render_runtime_template(
    template_name: str,
    *,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    env = jinja2.Environment(
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.from_string(load_source(template_name))
    return template.render(messages=messages, add_generation_prompt=add_generation_prompt)


def test_octothinker_generator_non_icl_matches_golden_template() -> None:
    doc_text = "Doc snippet: Water is composed of hydrogen and oxygen."
    messages = [
        {"role": "system", "content": MCQ_QUESTION_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": MCQ_QUESTION_GENERATION_PROMPT.format(text=doc_text)},
    ]

    rendered = _render_runtime_template(
        "octothinker_shared_chat_template.jinja",
        messages=messages,
    )
    golden = render("octothinker_question_gen.jinja", text=doc_text)

    assert rendered == golden


def test_octothinker_solver_mcq_non_icl_matches_golden_template() -> None:
    question = {
        "question_text": "What is 2 + 3?",
        "choices": ["4", "5", "6", "7"],
        "ground_truth": "5",
    }
    messages, _ = build_mcq_messages(question, use_few_shot_icl=False)

    rendered = _render_runtime_template(
        "octothinker_shared_chat_template.jinja",
        messages=messages,
    )
    golden = render(
        "octothinker_mcq_answer.jinja",
        question_text=question["question_text"],
        choices=question["choices"],
    )

    assert rendered == golden


def test_octothinker_solver_freeform_non_icl_matches_golden_template() -> None:
    question_text = "What is 17 + 25?"
    messages = build_free_form_messages(question_text, use_few_shot_icl=False)

    rendered = _render_runtime_template(
        "octothinker_shared_chat_template.jinja",
        messages=messages,
    )
    golden = render("octothinker_freeform_answer.jinja", question_text=question_text)

    assert rendered == golden


def test_octothinker_solver_mcq_icl_fallback_preserves_all_turns() -> None:
    question = {
        "question_text": "What is 2 + 3?",
        "choices": ["4", "5", "6", "7"],
        "ground_truth": "5",
    }
    messages, _ = build_mcq_messages(question, use_few_shot_icl=True)

    rendered = _render_runtime_template(
        "octothinker_shared_chat_template.jinja",
        messages=messages,
    )

    for message in messages:
        assert message["content"].strip() in rendered
    assert rendered.count("\nUser:") == 1
    assert rendered.count("\nAssistant:") == 1
    assert rendered.endswith("Assistant:\n")


def test_octothinker_solver_freeform_icl_fallback_preserves_all_turns() -> None:
    messages = build_free_form_messages("What is 17 + 25?", use_few_shot_icl=True)

    rendered = _render_runtime_template(
        "octothinker_shared_chat_template.jinja",
        messages=messages,
    )

    for message in messages:
        assert message["content"].strip() in rendered
    assert rendered.count("\nUser:") == 1
    assert rendered.count("\nAssistant:") == 1
    assert rendered.endswith("Assistant:\n")


def test_octothinker_solver_custom_prompt_fallback_preserves_custom_system_and_user() -> None:
    msg_data = build_benchmark_messages(
        questions=[
            {
                "question_id": "custom",
                "choices": ["A", "B", "C", "D"],
                "ground_truth": "A",
                "user_prompt": "Custom benchmark user prompt",
                "system_prompt": "Custom benchmark system prompt",
            }
        ],
        tokenizer=_FakeTokenizer(),
        max_prompt_tokens=10_000,
    )

    rendered = _render_runtime_template(
        "octothinker_shared_chat_template.jinja",
        messages=msg_data.messages_list[0],
    )

    assert "Custom benchmark system prompt" in rendered
    assert "Custom benchmark user prompt" in rendered
    assert rendered.count("\nUser:") == 1
    assert rendered.count("\nAssistant:") == 1
    assert rendered.endswith("Assistant:\n")
