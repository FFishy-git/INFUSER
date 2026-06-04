"""Shared benchmark evaluation core.

This module is the single source of truth for:
- benchmark path resolution and loading
- benchmark question -> prompt/DataProto conversion
- rollout output scoring/parsing
- common metric flattening for in-training logging
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from verl_inf_evolve.utils.prompt_length_utils import should_filter_by_prompt_length

logger = logging.getLogger(__name__)

_CODE_BENCHMARK_TYPES = frozenset({"code_functional", "code_stdio"})


@dataclass
class BenchmarkMessageData:
    """Container for benchmark prompts and per-question metadata."""

    messages_list: list[list[dict[str, str]]]
    prompt_texts: list[str | None]
    question_ids: list[Any]
    ground_truths: list[str]
    data_sources: list[str]
    verifier_metadatas: list[dict[str, Any] | None]
    is_mcq_flags: list[bool]
    benchmark_types: list[str]


@dataclass
class BenchmarkEvaluationResult:
    """Evaluation artifacts for one benchmark."""

    output: Any
    question_results: list[dict[str, Any]]
    metrics: dict[str, Any]


def resolve_benchmark_path(benchmark_name: str) -> str:
    """Resolve a benchmark name/path to a local JSON file path."""
    if os.path.isfile(benchmark_name):
        return benchmark_name

    standard_path = os.path.join(
        ".cache",
        "data",
        "preprocessed",
        "benchmarks",
        f"{benchmark_name}.json",
    )
    if os.path.isfile(standard_path):
        return standard_path

    raise FileNotFoundError(
        f"Benchmark '{benchmark_name}' not found. Checked:\n"
        f"  - {benchmark_name}\n"
        f"  - {standard_path}"
    )


def load_benchmark_questions(
    benchmark_name_or_path: str,
    max_questions: int | None = None,
) -> tuple[list[dict], str]:
    """Load and optionally cap benchmark questions.

    Args:
        benchmark_name_or_path: Benchmark short name (e.g. ``aime``) or
            direct JSON path.
        max_questions: Optional max question cap. ``None`` or ``<=0`` means
            no cap.

    Returns:
        ``(questions, resolved_path)``.
    """
    benchmark_path = resolve_benchmark_path(benchmark_name_or_path)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise ValueError(
            f"Benchmark JSON must be a list, got {type(questions)} "
            f"(path={benchmark_path})"
        )

    max_q = int(max_questions or 0)
    if max_q > 0 and len(questions) > max_q:
        logger.info(
            "Benchmark %s: capping %d questions to max_questions=%d",
            benchmark_name_or_path,
            len(questions),
            max_q,
        )
        questions = questions[:max_q]

    return questions, benchmark_path


def detect_question_benchmark_types(questions: list[dict]) -> set[str]:
    """Return the set of normalized benchmark types present in ``questions``."""
    from verl_inf_evolve.sol_eval.benchmark_adapters import detect_benchmark_type

    return {detect_benchmark_type(q) for q in questions}


def contains_code_benchmarks(questions: list[dict]) -> bool:
    """Return whether any question row is a code-execution benchmark."""
    return any(bt in _CODE_BENCHMARK_TYPES for bt in detect_question_benchmark_types(questions))


def resolve_code_execution_enabled(
    questions: list[dict],
    *,
    code_execution_enabled: bool,
    execution_scope: str,
) -> bool:
    """Return whether code execution should be active for this evaluation."""
    if code_execution_enabled:
        return True
    if execution_scope == "standalone" and contains_code_benchmarks(questions):
        return True
    return False


def validate_code_execution_policy(
    questions: list[dict],
    *,
    code_execution_enabled: bool,
    execution_scope: str,
    allow_code_execution_in_training: bool = False,
) -> None:
    """Validate code execution policy for the current benchmark batch."""
    benchmark_types = detect_question_benchmark_types(questions)
    code_types = sorted(bt for bt in benchmark_types if bt in _CODE_BENCHMARK_TYPES)
    if not code_types:
        return

    effective_code_execution_enabled = resolve_code_execution_enabled(
        questions,
        code_execution_enabled=code_execution_enabled,
        execution_scope=execution_scope,
    )

    if not effective_code_execution_enabled:
        joined = ", ".join(code_types)
        raise ValueError(
            f"Code benchmark execution is disabled for {execution_scope}. "
            f"Found benchmark_type(s): {joined}. "
            "Enable eval.code_execution.enabled for standalone sol_eval only."
        )

    if execution_scope != "standalone" and not allow_code_execution_in_training:
        raise ValueError(
            "Code benchmarks are disabled for in-training evaluation by default. "
            "Set code_execution.allow_in_training=true only if you explicitly "
            f"want execution_scope={execution_scope}."
        )


def resolve_use_public_eval_prompt(
    cfg_value: Any,
    data_source: str,
) -> bool:
    """Resolve the ``eval.use_public_eval_prompt`` config value for a benchmark.

    Two config shapes are supported (the second is the preferred style since
    2026-04-16 when per-benchmark toggling was added):

    1. **Plain bool** — legacy global flag, applies to every benchmark:

       .. code-block:: yaml

           eval:
             use_public_eval_prompt: true

    2. **Dict with default + per-benchmark overrides** — recommended style:

       .. code-block:: yaml

           eval:
             use_public_eval_prompt:
               default: false
               benchmarks:
                 math500: true
                 math: true
                 minerva_math: true
                 amc: true

       ``benchmarks[data_source]`` takes precedence when present; otherwise
       falls back to ``default`` (which itself defaults to ``false``).

    Anything else (None, other scalars) collapses to ``False``.
    """
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if cfg_value is None:
        return False
    if isinstance(cfg_value, bool):
        return cfg_value
    # DictConfig from Hydra / OmegaConf.
    if isinstance(cfg_value, DictConfig):
        cfg_value = OmegaConf.to_container(cfg_value, resolve=True)
    if isinstance(cfg_value, ListConfig):
        return bool(cfg_value)  # unusual; collapse to truthiness
    if isinstance(cfg_value, dict):
        overrides = cfg_value.get("benchmarks") or {}
        if data_source in overrides:
            return bool(overrides[data_source])
        return bool(cfg_value.get("default", False))
    return bool(cfg_value)


def build_benchmark_messages(
    questions: list[dict],
    tokenizer: Any,
    max_prompt_tokens: int,
    use_public_eval_prompt: Any = False,
    model_path: str = "",
    mcq_choice_shuffle_config: dict[str, Any] | None = None,
    benchmark_prompts: dict[str, Any] | None = None,
) -> BenchmarkMessageData:
    """Build chat messages from benchmark questions.

    ``benchmark_prompts`` is an optional dict of per-benchmark prompt
    overrides from ``eval.benchmark_prompts`` in the config. When a
    benchmark has an entry, its ``system_prompt`` and/or
    ``user_prompt_template`` override the defaults. This is injected into
    each question dict so existing ``question.get("system_prompt")`` logic
    picks it up without changing the adapter code.

    ``use_public_eval_prompt`` accepts either a plain bool or the dict form
    ``{default: bool, benchmarks: {data_source: bool}}`` — see
    :func:`resolve_use_public_eval_prompt`. Resolution happens once per
    benchmark batch based on ``questions[0]['data_source']`` (all questions
    in a batch share the same data source since sol_eval evaluates one
    benchmark at a time).
    """
    from verl_inf_evolve.sol_eval.benchmark_adapters import (
        build_messages_for_question,
        build_verifier_metadata,
        detect_benchmark_type,
    )
    from verl_inf_evolve.utils.benchmarks.model_type import is_base_model

    _is_base = is_base_model(model_path) if model_path else False

    # Resolve the config value to a plain bool once per benchmark. sol_eval
    # evaluates one benchmark per call site, so reading the first question's
    # data_source is sufficient.
    resolved_data_source = (
        str(questions[0].get("data_source", "") or "") if questions else ""
    )
    use_public_eval_prompt_bool = resolve_use_public_eval_prompt(
        use_public_eval_prompt, resolved_data_source
    )
    use_public_eval_prompt = use_public_eval_prompt_bool

    messages_list: list[list[dict[str, str]]] = []
    prompt_texts: list[str | None] = []
    question_ids: list[Any] = []
    ground_truths: list[str] = []
    data_sources: list[str] = []
    verifier_metadatas: list[dict[str, Any] | None] = []
    is_mcq_flags: list[bool] = []
    benchmark_types: list[str] = []

    skipped = 0
    if use_public_eval_prompt:
        logger.info(
            "Public-eval-aligned prompt is enabled for data_source=%s",
            resolved_data_source,
        )

    # Inject per-benchmark prompt overrides from config into each question
    # dict. This lets the adapter's existing question.get("system_prompt")
    # logic pick them up without code changes. Question-level fields take
    # precedence (we only inject if the field is not already set).
    _bp = {}
    if benchmark_prompts and resolved_data_source:
        _bp_raw = benchmark_prompts
        if hasattr(_bp_raw, "items"):
            _bp = dict(_bp_raw.get(resolved_data_source) or {})
        if _bp:
            logger.info(
                "Per-benchmark prompt override for data_source=%s: %s",
                resolved_data_source,
                {k: v[:60] + "..." if isinstance(v, str) and len(v) > 60 else v for k, v in _bp.items()},
            )

    for q in questions:
        if _bp:
            if "system_prompt" in _bp and "system_prompt" not in q:
                q["system_prompt"] = _bp["system_prompt"]
            if "user_prompt_template" in _bp and "user_prompt" not in q:
                q["user_prompt"] = _bp["user_prompt_template"].replace(
                    "{question}", str(q.get("question_text", ""))
                )
        messages, normalized_gt, question_is_mcq = build_messages_for_question(
            q,
            use_public_eval_prompt=use_public_eval_prompt,
            mcq_choice_shuffle_config=mcq_choice_shuffle_config,
        )
        prompt_text = None
        data_source = str(q.get("data_source", "") or "")

        # Decide whether to bypass ``apply_chat_template()`` by setting
        # ``prompt_text`` (raw text sent directly to vLLM).
        #
        # When ``use_public_eval_prompt`` is on, MATH500/MATH uses the
        # OC-aligned public eval prompt (single user message):
        #   - Base model  → set prompt_text (raw text, mirrors OC type=VLLM).
        #   - Instruct    → leave prompt_text=None (chat template, mirrors
        #                    OC VLLMwithChatTemplate).
        #
        # When the flag is off, MATH500 falls through to the generic
        # training prompt (system+user messages, always chat-templated).
        #
        # Other public-eval benchmarks: set prompt_text when the flag is on
        # (existing behaviour, always bypasses chat template for public eval).
        _is_single_user_msg = (
            len(messages) == 1 and messages[0].get("role") == "user"
        )
        if (
            _is_single_user_msg
            and data_source in {"math500", "math", "amc", "minerva_math"}
            and use_public_eval_prompt
        ):
            if _is_base:
                prompt_text = str(messages[0].get("content", ""))
            # instruct: prompt_text stays None → chat template applied
        elif (
            _is_single_user_msg
            and use_public_eval_prompt
            and data_source in {"mmlu_pro", "medqa", "olympiadbench", "hmmt", "phybench", "putnam"}
        ):
            prompt_text = str(messages[0].get("content", ""))

        if should_filter_by_prompt_length(
            tokenizer=tokenizer,
            messages=messages,
            prompt_text=prompt_text,
            max_prompt_tokens=max_prompt_tokens,
            logger=logger,
            sample_kind="benchmark question",
            sample_id=q.get("question_id", "<unknown>"),
        ):
            skipped += 1
            continue

        messages_list.append(messages)
        prompt_texts.append(prompt_text)
        question_ids.append(q["question_id"])
        ground_truths.append(normalized_gt)
        data_sources.append(q.get("data_source", ""))
        is_mcq_flags.append(question_is_mcq)
        benchmark_types.append(detect_benchmark_type(q))
        verifier_metadatas.append(build_verifier_metadata(q))

    if skipped:
        logger.warning(
            "Filtered %d/%d questions exceeding prompt_length=%d",
            skipped,
            len(questions),
            max_prompt_tokens,
        )

    mcq_count = sum(1 for flag in is_mcq_flags if flag)
    open_count = len(is_mcq_flags) - mcq_count
    if mcq_count and not open_count:
        logger.info("Benchmark type: MCQ (questions have choices)")
    elif open_count and not mcq_count:
        logger.info("Benchmark type: open-ended (no choices)")
    elif mcq_count or open_count:
        logger.info(
            "Benchmark type: mixed (%d MCQ, %d open-ended)",
            mcq_count,
            open_count,
        )

    return BenchmarkMessageData(
        messages_list=messages_list,
        prompt_texts=prompt_texts,
        question_ids=question_ids,
        ground_truths=ground_truths,
        data_sources=data_sources,
        verifier_metadatas=verifier_metadatas,
        is_mcq_flags=is_mcq_flags,
        benchmark_types=benchmark_types,
    )


def messages_to_benchmark_dataproto(
    msg_data: BenchmarkMessageData,
    tokenizer: Any,
) -> Any:
    """Convert benchmark prompt messages into a DataProto batch."""
    from verl_inf_evolve.data.batch_utils import messages_to_dataproto

    meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    non_tensor_metadata: dict[str, Any] = {
        "question_id": msg_data.question_ids,
        "ground_truth": msg_data.ground_truths,
        "is_mcq": msg_data.is_mcq_flags,
        "benchmark_type": msg_data.benchmark_types,
    }

    if any(ds for ds in msg_data.data_sources):
        non_tensor_metadata["data_source"] = msg_data.data_sources
    if any(vm is not None for vm in msg_data.verifier_metadatas):
        non_tensor_metadata["verifier_metadata"] = msg_data.verifier_metadatas

    return messages_to_dataproto(
        messages_list=msg_data.messages_list,
        non_tensor_metadata=non_tensor_metadata,
        meta_info=meta_info,
    )
def generate_with_metadata(manager: Any, batch: Any, num_workers: int) -> Any:
    """Generate sequences and preserve ``non_tensor_batch`` metadata."""
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    saved_ntb = {
        k: v.copy()
        for k, v in batch.non_tensor_batch.items()
        if k not in ("raw_prompt", "agent_name")
    }

    batch_padded, pad_size = pad_dataproto_to_divisor(batch, num_workers)
    output_padded = manager.generate_sequences(batch_padded)
    output = unpad_dataproto(output_padded, pad_size)

    output.meta_info.pop("timing", None)
    for k, v in saved_ntb.items():
        output.non_tensor_batch[k] = v

    return output


def build_question_results(
    output: Any,
    questions: list[dict],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Build per-question result rows from scored rollout output."""
    from verl_inf_evolve.utils.mcq_utils import group_scores_by_qid

    q_lookup = {str(q.get("question_id", "")): q for q in questions}
    grouped_scores = group_scores_by_qid(output)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    primary_scores = output.non_tensor_batch.get("primary_score")
    primary_score_names = output.non_tensor_batch.get("primary_score_name")
    primary_score_scale_maxes = output.non_tensor_batch.get("primary_score_scale_max")
    exec_results = output.non_tensor_batch.get("exec_result")

    question_results: list[dict[str, Any]] = []
    for qid in grouped_scores:
        q = q_lookup.get(qid, {})
        qid_indices = [
            i
            for i, qid_val in enumerate(output.non_tensor_batch["question_id"])
            if str(qid_val) == qid
        ]

        sampled_answers = []
        extracted_answers = []
        answer_scores_list = []
        response_token_lengths = []
        sample_scores = []
        sample_score_name = None
        sample_score_scale_max = None
        sample_exec_results = []

        for idx in qid_indices:
            response_tokens = output.batch["responses"][idx]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            sampled_answers.append(response_text)

            if "response_mask" in output.batch.keys():
                token_len = int(output.batch["response_mask"][idx].sum())
            else:
                token_len = int((response_tokens != pad_token_id).sum())
            response_token_lengths.append(token_len)

            extracted = output.non_tensor_batch["extracted_answer"][idx]
            score = output.non_tensor_batch["answer_score"][idx]
            extracted_answers.append(str(extracted) if extracted is not None else None)
            answer_scores_list.append(float(score) if score is not None else None)
            if primary_scores is not None:
                primary_score = primary_scores[idx]
                sample_scores.append(
                    float(primary_score) if primary_score is not None else None
                )
                if sample_score_name is None and primary_score_names is not None:
                    score_name = primary_score_names[idx]
                    sample_score_name = str(score_name) if score_name is not None else None
                if (
                    sample_score_scale_max is None
                    and primary_score_scale_maxes is not None
                    and primary_score_scale_maxes[idx] is not None
                ):
                    sample_score_scale_max = float(primary_score_scale_maxes[idx])
            if exec_results is not None:
                sample_exec_results.append(exec_results[idx])

        question_result = {
            "question_id": qid,
            "question_text": q.get("question_text", ""),
            "choices": q.get("choices", []),
            "ground_truth": q.get("ground_truth", ""),
            "data_source": q.get("data_source", ""),
            "domain": q.get("domain", ""),
            "sampled_answers": sampled_answers,
            "extracted_answers": extracted_answers,
            "answer_scores": answer_scores_list,
            "response_token_lengths": response_token_lengths,
        }
        if primary_scores is not None:
            question_result["sample_scores"] = sample_scores
            if sample_score_name is not None:
                question_result["sample_score_name"] = sample_score_name
            if sample_score_scale_max is not None:
                question_result["sample_score_scale_max"] = sample_score_scale_max
        if exec_results is not None:
            question_result["sample_exec_results"] = sample_exec_results

        question_results.append(question_result)

    return question_results


def evaluate_benchmark_questions(
    questions: list[dict],
    tokenizer: Any,
    max_prompt_tokens: int,
    n_samples: int,
    generate_batch_fn: Callable[[Any], Any],
    use_public_eval_prompt: Any = False,
    raise_if_all_filtered: bool = True,
    on_generation_output_fn: Callable[[Any], None] | None = None,
    code_execution_enabled: bool = False,
    execution_scope: str = "standalone",
    allow_code_execution_in_training: bool = False,
    model_path: str = "",
    mcq_choice_shuffle_config: dict[str, Any] | None = None,
) -> BenchmarkEvaluationResult | None:
    """Run one benchmark end-to-end.

    Args:
        questions: Benchmark question rows.
        tokenizer: Solver tokenizer.
        max_prompt_tokens: Prompt length filter.
        n_samples: Rollout samples per question.
        generate_batch_fn: Function that takes one DataProto batch and returns
            rollout DataProto output.
        use_public_eval_prompt: Use public-eval-aligned prompt per benchmark.
        raise_if_all_filtered: If true, raise when all questions are filtered
            by prompt length. If false, return ``None``.
        on_generation_output_fn: Optional callback invoked right after rollout
            generation (before answer scoring/parsing). Useful for persisting
            raw rollout output.
        code_execution_enabled: Whether code benchmarks may execute at all.
        execution_scope: ``standalone`` or ``in_training``.
        allow_code_execution_in_training: Explicit override for in-loop code eval.
    """
    from verl_inf_evolve.sol_eval.result_format import compute_eval_metrics
    from verl_inf_evolve.utils.mcq_utils import extract_answer_scores

    validate_code_execution_policy(
        questions,
        code_execution_enabled=code_execution_enabled,
        execution_scope=execution_scope,
        allow_code_execution_in_training=allow_code_execution_in_training,
    )
    effective_code_execution_enabled = resolve_code_execution_enabled(
        questions,
        code_execution_enabled=code_execution_enabled,
        execution_scope=execution_scope,
    )

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        use_public_eval_prompt=use_public_eval_prompt,
        model_path=model_path,
        mcq_choice_shuffle_config=mcq_choice_shuffle_config,
    )
    if not msg_data.messages_list:
        message = "All questions were filtered out (too long)"
        if raise_if_all_filtered:
            raise ValueError(message)
        logger.warning(message)
        return None

    batch = messages_to_benchmark_dataproto(msg_data, tokenizer)
    if n_samples > 1:
        batch = batch.repeat(repeat_times=n_samples, interleave=True)

    output = generate_batch_fn(batch)
    if on_generation_output_fn is not None:
        on_generation_output_fn(output)

    extract_answer_scores(
        output,
        tokenizer,
        allow_code_execution=effective_code_execution_enabled,
    )
    question_results = build_question_results(output, questions, tokenizer)
    # Optional LLM-judge cascade upgrade for rule-failed samples on eligible
    # benchmarks (MATH500, Minerva, etc.). Dispatches concurrent gpt-5-mini
    # calls via ThreadPoolExecutor; see ``_llm_math_judge.apply_to_question_results``.
    from verl_inf_evolve.utils.benchmarks.verifiers import _llm_math_judge
    _llm_math_judge.apply_to_question_results(question_results)
    metrics = compute_eval_metrics(question_results, n_samples=n_samples)
    sub_bench_metrics = compute_sub_bench_metrics(question_results, n_samples=n_samples)
    if sub_bench_metrics:
        metrics["sub_bench_metrics"] = sub_bench_metrics

    return BenchmarkEvaluationResult(
        output=output,
        question_results=question_results,
        metrics=metrics,
    )


def compute_sub_bench_metrics(
    question_results: list[dict[str, Any]],
    n_samples: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Group question results by sub-benchmark and compute per-group metrics.

    Returns ``{sub_bench_name: metrics_dict}``.  Only useful for combined
    benchmarks (e.g. ``combine_2000``) or benchmarks that embed an internal
    partition such as OlympiadBench Math vs Physics.
    """
    from collections import defaultdict
    from verl_inf_evolve.sol_eval.result_format import compute_eval_metrics as _compute

    def _sub_bench_name(qr: dict[str, Any]) -> str:
        ds = str(qr.get("data_source") or "unknown")
        if ds == "olympiadbench":
            domain = str(qr.get("domain") or "").strip()
            if domain == "OlympiadBench_Math":
                return "math"
            if domain == "OlympiadBench_Physics":
                return "physics"
            if domain:
                return domain
        return ds

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qr in question_results:
        grouped[_sub_bench_name(qr)].append(qr)

    # Only meaningful when there are multiple sub-benchmarks.
    if len(grouped) <= 1:
        return {}

    return {name: _compute(qs, n_samples=n_samples) for name, qs in sorted(grouped.items())}


def flatten_eval_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert ``compute_eval_metrics()`` output into flat metric keys."""
    flat: dict[str, Any] = {
        f"{prefix}/accuracy_strict": metrics["accuracy_strict"],
        f"{prefix}/accuracy_lenient": metrics["accuracy_lenient"],
        f"{prefix}/total_questions": metrics["total_questions"],
    }

    score_name = metrics.get("score_name")
    if score_name:
        strict_key = f"{score_name}_score_strict"
        lenient_key = f"{score_name}_score_lenient"
        if strict_key in metrics:
            flat[f"{prefix}/{strict_key}"] = metrics[strict_key]
        if lenient_key in metrics:
            flat[f"{prefix}/{lenient_key}"] = metrics[lenient_key]

    for suffix, values in (
        ("strict", metrics.get("pass_at_k_strict", {})),
        ("lenient", metrics.get("pass_at_k_lenient", {})),
    ):
        if not isinstance(values, dict):
            continue
        for k_str, value in sorted(
            values.items(),
            key=lambda item: (
                0, int(item[0])
            ) if str(item[0]).isdigit() else (1, str(item[0])),
        ):
            flat[f"{prefix}/pass_at_{k_str}_{suffix}"] = value

    resp_tokens = metrics.get("response_length_tokens", {})
    if resp_tokens.get("mean") is not None:
        flat[f"{prefix}/response_length_tokens_mean"] = resp_tokens["mean"]

    return flat
