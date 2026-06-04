"""Jinja2 prompt templates for the verl_inf_evolve training pipeline.

Each ``.jinja`` file is a **complete, ready-to-tokenize** prompt string
(chat-template wrapper included) with Jinja2 variables for the dynamic
parts.  Rendering a template produces the same byte sequence that the
current ``build_mcq_messages() → apply_chat_template()`` pipeline does.

Usage::

    from verl_inf_evolve.templates import render

    prompt = render("qwen3_mcq_answer.jinja",
                    question_text="What is 2+2?",
                    choices=["3", "4", "5", "6"])
"""

from __future__ import annotations

from pathlib import Path

import jinja2

_TEMPLATE_DIR = Path(__file__).parent

_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_TEMPLATE_DIR),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _validate_template_name(template_name: str) -> Path:
    path = Path(template_name)
    if path.suffix != ".jinja":
        raise ValueError(f"Template must end with .jinja: {template_name}")
    if path.name != template_name or path.is_absolute():
        raise ValueError(f"Template name must be a filename inside the package: {template_name}")
    return _TEMPLATE_DIR / path.name


def load_source(template_name: str) -> str:
    """Load the raw source text for a packaged template file."""
    template_path = _validate_template_name(template_name)
    return template_path.read_text(encoding="utf-8")


def render(template_name: str, **kwargs: object) -> str:
    """Render a template by name with the given variables."""
    return _ENV.get_template(template_name).render(**kwargs)
