"""Local utility helpers for verl_inf_evolve.

These modules contain the subset required by the verl-native self-evolution
trainer while keeping dependencies local.

Heavy imports (data_utils → numpy/torch) are deferred so that lightweight
consumers (task_center, sol_eval CLI) can import submodules like
``remote_backend`` without pulling in the full training stack.
"""


def __getattr__(name: str):
    """Lazy-load re-exported symbols on first access."""
    _mcq_names = {
        "build_mcq_messages",
        "extract_answer_scores",
        "extract_boxed_answer",
        "group_scores_by_qid",
        "is_correct",
    }
    _prompt_names = {
        "MCQ_QUESTION_GENERATION_PROMPT",
        "MCQ_QUESTION_GENERATION_SYSTEM_PROMPT",
        "FREE_FORM_QUESTION_GENERATION_PROMPT",
        "FREE_FORM_QUESTION_GENERATION_SYSTEM_PROMPT",
    }
    _parser_names = {
        "parse_generated_questions",
        "parse_generated_question_response",
        "parse_mcq_questions",
        "parse_question_response",
    }

    if name in _mcq_names:
        from verl_inf_evolve.utils.mcq_utils import (
            build_mcq_messages,
            extract_answer_scores,
            extract_boxed_answer,
            group_scores_by_qid,
            is_correct,
        )
        return locals()[name]

    if name in _prompt_names:
        from verl_inf_evolve.utils.prompts import (
            FREE_FORM_QUESTION_GENERATION_PROMPT,
            FREE_FORM_QUESTION_GENERATION_SYSTEM_PROMPT,
            MCQ_QUESTION_GENERATION_PROMPT,
            MCQ_QUESTION_GENERATION_SYSTEM_PROMPT,
        )
        return locals()[name]

    if name in _parser_names:
        from verl_inf_evolve.utils.question_parser import (
            parse_generated_questions,
            parse_generated_question_response,
            parse_mcq_questions,
            parse_question_response,
        )
        return locals()[name]

    if name == "scatter_for_dispatch":
        from verl_inf_evolve.utils.data_utils import scatter_for_dispatch
        return scatter_for_dispatch

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_mcq_messages",
    "extract_answer_scores",
    "extract_boxed_answer",
    "group_scores_by_qid",
    "is_correct",
    "MCQ_QUESTION_GENERATION_PROMPT",
    "MCQ_QUESTION_GENERATION_SYSTEM_PROMPT",
    "FREE_FORM_QUESTION_GENERATION_PROMPT",
    "FREE_FORM_QUESTION_GENERATION_SYSTEM_PROMPT",
    "parse_generated_questions",
    "parse_generated_question_response",
    "parse_mcq_questions",
    "parse_question_response",
    "scatter_for_dispatch",
]
