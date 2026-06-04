"""Shared benchmark adapters for sol_eval prompting and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


CODE_GENERATION_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Return only valid Python code."
)


@dataclass
class ScoredResponse:
    extracted_answer: str | None
    answer_score: float | None
    primary_score: float | None = None
    primary_score_name: str | None = None
    primary_score_scale_max: float | None = None
    exec_result: dict[str, Any] | None = None


class BenchmarkAdapter(Protocol):
    task_types: tuple[str, ...]

    def build_messages(
        self,
        question: dict[str, Any],
        *,
        use_public_eval_prompt: bool = False,
        mcq_choice_shuffle_config: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], str, bool]: ...

    def extract_prediction(self, response_text: str) -> str | None: ...


def detect_benchmark_type(question: dict[str, Any]) -> str:
    """Return the benchmark/task type for one question row."""

    explicit = question.get("benchmark_type")
    if explicit:
        return str(explicit)
    return "qa_mcq" if question.get("choices") else "qa_open"


def build_verifier_metadata(question: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize per-question verifier metadata."""

    explicit_vmeta = question.get("verifier_metadata")
    if explicit_vmeta is not None:
        if not isinstance(explicit_vmeta, dict):
            raise ValueError(
                f"verifier_metadata must be a dict when present; got {type(explicit_vmeta)}"
            )
        return dict(explicit_vmeta)

    vmeta: dict[str, Any] = {}
    if question.get("answer_type"):
        vmeta["answer_type"] = question["answer_type"]
    if question.get("unit"):
        vmeta["unit"] = question["unit"]
    return vmeta or None


def _resolve_mcq_choice_shuffle_enabled(
    question: dict[str, Any],
    mcq_choice_shuffle_config: dict[str, Any] | None,
) -> bool:
    """Resolve whether MCQ choice shuffling is active for one question."""
    explicit = (mcq_choice_shuffle_config or {}).get("enabled", None)
    if explicit is not None:
        return bool(explicit)
    return str(question.get("data_source", "") or "") == "gpqa_diamond"


class QAAdapter:
    task_types = ("qa_mcq", "qa_open")

    def build_messages(
        self,
        question: dict[str, Any],
        *,
        use_public_eval_prompt: bool = False,
        mcq_choice_shuffle_config: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], str, bool]:
        from verl_inf_evolve.utils.mcq_utils import build_mcq_messages
        from verl_inf_evolve.utils.prompts import (
            FREE_FORM_ANSWER_GENERATION_SYSTEM_PROMPT,
            MCQ_ANSWER_GENERATION_SYSTEM_PROMPT,
            build_free_form_messages,
        )

        task_type = detect_benchmark_type(question)
        custom_user_prompt = question.get("user_prompt")
        custom_system_prompt = question.get("system_prompt")

        if task_type == "qa_mcq":
            if (
                str(question.get("data_source", "") or "") == "mmlu_pro"
                and use_public_eval_prompt
                and custom_user_prompt is None
            ):
                from verl_inf_evolve.utils.benchmarks.mmlu_pro_prompting import (
                    build_mmlu_pro_icl_messages,
                )

                messages, normalized_gt = build_mmlu_pro_icl_messages(question)
                return messages, normalized_gt, True
            if (
                str(question.get("data_source", "") or "") == "medqa"
                and use_public_eval_prompt
                and custom_user_prompt is None
            ):
                from verl_inf_evolve.utils.benchmarks.medqa_prompting import (
                    build_medqa_icl_messages,
                )

                messages, normalized_gt = build_medqa_icl_messages(question)
                return messages, normalized_gt, True
            if custom_user_prompt is not None:
                system_msg = custom_system_prompt or MCQ_ANSWER_GENERATION_SYSTEM_PROMPT
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": custom_user_prompt},
                ]
                normalized_gt = str(question.get("ground_truth", ""))
            else:
                messages, normalized_gt = build_mcq_messages(
                    question,
                    use_few_shot_icl=use_public_eval_prompt,
                    system_prompt=custom_system_prompt,
                    shuffle_choices=_resolve_mcq_choice_shuffle_enabled(
                        question, mcq_choice_shuffle_config,
                    ),
                    shuffle_seed=(mcq_choice_shuffle_config or {}).get("seed"),
                    shuffle_mode=str(
                        (mcq_choice_shuffle_config or {}).get(
                            "mode", "per_question_seeded"
                        )
                    ),
                )
            return messages, normalized_gt, True

        if (
            str(question.get("data_source", "") or "") == "olympiadbench"
            and use_public_eval_prompt
            and custom_user_prompt is None
        ):
            from verl_inf_evolve.utils.benchmarks.olympiadbench_prompting import (
                build_olympiadbench_icl_messages,
            )

            messages, normalized_gt = build_olympiadbench_icl_messages(question)
            return messages, normalized_gt, False

        if (
            str(question.get("data_source", "") or "")
            in {"math500", "math", "amc", "minerva_math"}
            and use_public_eval_prompt
            and custom_user_prompt is None
        ):
            from verl_inf_evolve.utils.benchmarks.math500_prompting import (
                build_math500_icl_messages,
            )

            # OC / General-Reasoner public eval prompt (0-shot, raw text).
            # Returns a single "user" message; eval_core decides whether
            # to bypass chat template based on model type (base → raw
            # text, instruct → apply_chat_template).
            #
            # All four math benchmarks share the same template
            # (``{question}\nPlease reason step by step, and put your
            # final answer within \\boxed{}.``) — verbatim from
            # ``baselines/general-reasoner/evaluation/simple-evals/
            # {math,minerva,amc}_eval_qwen.py``. Routing them through
            # the same builder keeps the chat-template-bypass path
            # consistent for base models on every math benchmark.
            messages, normalized_gt = build_math500_icl_messages(question)
            return messages, normalized_gt, False

        if (
            str(question.get("data_source", "") or "") in {"hmmt", "putnam"}
            and use_public_eval_prompt
            and custom_user_prompt is None
        ):
            from verl_inf_evolve.utils.benchmarks.hmmt_prompting import (
                build_hmmt_icl_messages,
            )

            messages, normalized_gt = build_hmmt_icl_messages(question)
            return messages, normalized_gt, False

        if (
            str(question.get("data_source", "") or "") == "phybench"
            and use_public_eval_prompt
            and custom_user_prompt is None
        ):
            from verl_inf_evolve.utils.benchmarks.phybench_prompting import (
                build_phybench_icl_messages,
            )

            messages, normalized_gt = build_phybench_icl_messages(question)
            return messages, normalized_gt, False

        if custom_user_prompt is not None:
            system_msg = custom_system_prompt or FREE_FORM_ANSWER_GENERATION_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": custom_user_prompt},
            ]
        else:
            messages = build_free_form_messages(
                question_text=question.get("question_text", ""),
                use_few_shot_icl=use_public_eval_prompt,
                system_prompt=custom_system_prompt,
            )
        return messages, str(question.get("ground_truth", "")), False

    def extract_prediction(self, response_text: str) -> str | None:
        from verl_inf_evolve.utils.mcq_utils import (
            extract_boxed_answer,
            extract_boxed_answer_general,
        )

        task_type = "qa_mcq" if response_text is None else None
        del task_type  # protocol placeholder, decision happens in caller
        raise NotImplementedError("Use extract_prediction_from_question()")


class CodegenAdapter:
    task_types = ("code_functional", "code_stdio")

    def __init__(self, *, task_type: str):
        self.task_types = (task_type,)

    def build_messages(
        self,
        question: dict[str, Any],
        *,
        use_public_eval_prompt: bool = False,
        mcq_choice_shuffle_config: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], str, bool]:
        del mcq_choice_shuffle_config, use_public_eval_prompt

        custom_user_prompt = question.get("user_prompt")
        custom_system_prompt = question.get("system_prompt")
        system_msg = custom_system_prompt or CODE_GENERATION_SYSTEM_PROMPT
        user_prompt = custom_user_prompt or str(question.get("question_text", ""))
        if not user_prompt.strip():
            raise ValueError(
                f"Code benchmark question {question.get('question_id', '<unknown>')} "
                "must provide user_prompt or question_text"
            )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]
        return messages, str(question.get("ground_truth", "")), False

    def extract_prediction(self, response_text: str) -> str | None:
        from verl_inf_evolve.utils.benchmarks.code_exec_utils import extract_python_code

        return extract_python_code(response_text)


_QA_ADAPTER = QAAdapter()
_HUMANEVAL_ADAPTER = CodegenAdapter(task_type="code_functional")
_LIVECODEBENCH_ADAPTER = CodegenAdapter(task_type="code_stdio")
_ADAPTERS: tuple[BenchmarkAdapter, ...] = (
    _QA_ADAPTER,
    _HUMANEVAL_ADAPTER,
    _LIVECODEBENCH_ADAPTER,
)
_ADAPTERS_BY_TASK_TYPE = {
    task_type: adapter
    for adapter in _ADAPTERS
    for task_type in adapter.task_types
}


def get_adapter_for_task_type(task_type: str) -> BenchmarkAdapter:
    try:
        return _ADAPTERS_BY_TASK_TYPE[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark_type={task_type!r}") from exc


def get_adapter_for_question(question: dict[str, Any]) -> BenchmarkAdapter:
    return get_adapter_for_task_type(detect_benchmark_type(question))


def build_messages_for_question(
    question: dict[str, Any],
    *,
    use_public_eval_prompt: bool = False,
    mcq_choice_shuffle_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], str, bool]:
    """Build one question row into chat messages + normalized ground truth."""
    adapter = get_adapter_for_question(question)
    return adapter.build_messages(
        question,
        use_public_eval_prompt=use_public_eval_prompt,
        mcq_choice_shuffle_config=mcq_choice_shuffle_config,
    )


def extract_prediction_from_question(
    question: dict[str, Any],
    response_text: str,
    *,
    allow_code_execution: bool = False,
) -> str | None:
    """Extract the benchmark-native prediction from one model response."""
    task_type = detect_benchmark_type(question)
    if task_type in {"code_functional", "code_stdio"}:
        if not allow_code_execution:
            raise ValueError(
                "Code benchmark execution is disabled for this path. "
                "Enable eval.code_execution.enabled before running standalone code benchmarks."
            )
        adapter = get_adapter_for_task_type(task_type)
        return adapter.extract_prediction(response_text)

    from verl_inf_evolve.utils.mcq_utils import (
        extract_boxed_answer,
        extract_boxed_answer_general,
    )

    if task_type == "qa_mcq":
        if str(question.get("data_source", "") or "") == "mmlu_pro":
            from verl_inf_evolve.utils.benchmarks.mmlu_pro_prompting import (
                extract_mmlu_pro_choice,
            )

            extracted = extract_mmlu_pro_choice(response_text)
            if extracted is not None:
                return extracted
        if str(question.get("data_source", "") or "") == "medqa":
            from verl_inf_evolve.utils.benchmarks.medqa_prompting import (
                extract_medqa_choice,
            )

            extracted = extract_medqa_choice(response_text)
            if extracted is not None:
                return extracted
        return extract_boxed_answer(response_text)
    if str(question.get("data_source", "") or "") in {"math500", "math"}:
        from verl_inf_evolve.utils.benchmarks.math500_prompting import (
            extract_math500_answer,
        )

        extracted = extract_math500_answer(response_text)
        if extracted is not None:
            return extracted
    if str(question.get("data_source", "") or "") == "phybench":
        from verl_inf_evolve.utils.benchmarks.phybench_prompting import (
            extract_phybench_answer,
        )

        extracted = extract_phybench_answer(response_text)
        if extracted is not None:
            return extracted
    if str(question.get("data_source", "") or "") == "bbeh":
        from verl_inf_evolve.utils.benchmarks.bbeh_prompting import extract_bbeh_answer

        extracted = extract_bbeh_answer(response_text)
        if extracted is not None:
            return extracted
    return extract_boxed_answer_general(response_text)


def score_response_for_question(
    question: dict[str, Any],
    response_text: str,
    *,
    allow_code_execution: bool = False,
    mcq_choice_shuffle_config: dict[str, Any] | None = None,
) -> ScoredResponse:
    """Shared scoring seam used by both DataProto and vLLM paths."""
    from verl_inf_evolve.utils.benchmarks.benchmark_scorers import (
        get_needs_full_response as get_scorer_needs_full_response,
        get_scorer,
    )
    from verl_inf_evolve.utils.mcq_utils import (
        format_mcq_question,
        is_correct,
        is_correct_general,
    )
    from verl_inf_evolve.utils.benchmarks.verifiers import (
        get_needs_full_response,
        get_verifier,
    )

    task_type = detect_benchmark_type(question)
    ground_truth = str(question.get("ground_truth", ""))
    data_source = str(question.get("data_source", "") or "")
    verifier_metadata = build_verifier_metadata(question)
    normalized_mcq_ground_truth: str | None = None

    if task_type == "qa_mcq":
        # MCQ benchmark rows commonly store ground_truth as full choice text.
        # Normalize it to the canonical letter so extracted answers can be
        # compared consistently across direct-vLLM and DataProto paths.
        # Training rollouts may carry a reduced MCQ payload that already has
        # canonical-letter ground truth but omits question_text / choices.
        raw_mcq_ground_truth = ground_truth.strip().upper()
        if (
            not question.get("choices")
            and not question.get("question_text")
            and len(raw_mcq_ground_truth) == 1
            and raw_mcq_ground_truth.isalpha()
        ):
            normalized_mcq_ground_truth = raw_mcq_ground_truth
        else:
            _, normalized_mcq_ground_truth = format_mcq_question(
                question,
                shuffle_choices=_resolve_mcq_choice_shuffle_enabled(
                    question, mcq_choice_shuffle_config,
                ),
                shuffle_seed=(mcq_choice_shuffle_config or {}).get("seed"),
                shuffle_mode=str(
                    (mcq_choice_shuffle_config or {}).get(
                        "mode", "per_question_seeded"
                    )
                ),
            )

    predicted = extract_prediction_from_question(
        question,
        response_text,
        allow_code_execution=allow_code_execution,
    )

    custom_verifier = get_verifier(data_source) if data_source else None
    verifier_needs_full = (
        get_needs_full_response(data_source) if data_source else False
    )
    custom_scorer = get_scorer(data_source) if data_source else None
    scorer_needs_full = (
        get_scorer_needs_full_response(data_source) if data_source else False
    )
    exec_result = None

    if exec_result is not None:
        passed = exec_result.get("passed")
        correct = bool(passed) if passed is not None else None
    elif custom_verifier is not None:
        verifier_input = response_text if verifier_needs_full else predicted
        correct = custom_verifier(verifier_input, ground_truth, verifier_metadata)
        # NOTE: the optional LLM-judge cascade upgrade for rule-failed samples
        # happens later via ``_llm_math_judge.apply_to_question_results``, which
        # dispatches all rule-failed eligible samples concurrently. Keeping the
        # call out of this per-sample hot path avoids the serial judge bottleneck.
    elif task_type == "qa_mcq":
        correct = (
            is_correct(predicted, normalized_mcq_ground_truth)
            if predicted is not None and normalized_mcq_ground_truth is not None
            else None
        )
    else:
        correct = (
            is_correct_general(predicted, ground_truth) if predicted is not None else None
        )

    answer_score = 1.0 if correct else (0.0 if correct is not None else None)

    primary_score = None
    primary_score_name = None
    primary_score_scale_max = None
    if custom_scorer is not None:
        scorer_input = response_text if scorer_needs_full else predicted
        score_result = custom_scorer(scorer_input, ground_truth, verifier_metadata)
        if score_result is not None:
            primary_score = (
                float(score_result.score) if score_result.score is not None else None
            )
            primary_score_name = score_result.name
            if score_result.scale_max is not None:
                primary_score_scale_max = float(score_result.scale_max)

    return ScoredResponse(
        extracted_answer=str(predicted) if predicted is not None else None,
        answer_score=answer_score,
        primary_score=primary_score,
        primary_score_name=primary_score_name,
        primary_score_scale_max=primary_score_scale_max,
        exec_result=exec_result,
    )
