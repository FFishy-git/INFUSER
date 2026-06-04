"""Pure-function metrics for rollout stages.

Each function accepts trainer-native data structures and returns a flat
``dict[str, float]`` ready for ``tracking_logger.log()``.
"""

from __future__ import annotations

from typing import Any

import math

import numpy as np
import torch

from verl_inf_evolve.utils.metric_utils import add_distribution_stats


OPENER_CLASSES = ("alright", "to", "others")


# ------------------------------------------------------------------
# Private helper
# ------------------------------------------------------------------


def classify_opener(text: str) -> str:
    """Classify a decoded response prefix into a coarse opener basin.

    Three buckets: ``alright`` (the long-thinking basin), ``to`` (any
    "To <verb>..." imperative-infinitive opener, e.g. "To solve this/the",
    "To find", "To determine"), and ``others`` (everything else — "Let's...",
    "We are given", "Step 1:", markdown headers, etc.).
    """
    normalized = str(text or "").lower().lstrip()
    if normalized.startswith("alright"):
        return "alright"
    if normalized.startswith("to "):
        return "to"
    return "others"


def _decode_response_prefix(tokenizer: Any, token_ids: Any, prefix_tokens: int = 4) -> str:
    """Decode the first few response tokens for opener classification."""
    prefix = token_ids[:prefix_tokens]
    try:
        if hasattr(prefix, "detach"):
            prefix = prefix.detach().cpu().tolist()
    except Exception:
        pass
    return tokenizer.decode(prefix, skip_special_tokens=True)


def _valid_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_mid_eos_mask_np(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    answer_scores: Any,
    eos_token_id: int | None,
) -> "np.ndarray":
    """Per-row boolean mask: True iff (answer_score is None) AND last unmasked token == eos.

    Distinguishes "EOS'd without \\boxed{}" (mid-EOS) from "hit max-length truncation"
    (last token != EOS) and from "valid answer then EOS" (answer_score is 0/1, not None).

    Returns all-False when ``answer_scores`` is None or ``eos_token_id`` is None.
    """
    batch_size = int(responses.shape[0])
    if answer_scores is None or eos_token_id is None:
        return np.zeros(batch_size, dtype=bool)
    last_idx = (response_mask.sum(dim=-1) - 1).clamp_min(0).long()
    last_tok = (
        responses.gather(1, last_idx.unsqueeze(1))
        .squeeze(1)
        .detach()
        .cpu()
        .numpy()
    )
    ends_with_eos = last_tok == int(eos_token_id)
    unparsed = np.array([s is None for s in answer_scores], dtype=bool)
    return ends_with_eos & unparsed


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _within_alright_minus_non(
    *,
    question_ids: list[str],
    opener_classes: list[str],
    values: list[float | None],
) -> tuple[float, int, float]:
    """Return equal-question-weighted mean(alright - non) and coverage."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for qid, opener, value in zip(question_ids, opener_classes, values):
        if value is None:
            continue
        bucket = "alright" if opener == "alright" else "non"
        grouped.setdefault(qid, {"alright": [], "non": []})[bucket].append(value)

    deltas: list[float] = []
    for per_q in grouped.values():
        if per_q["alright"] and per_q["non"]:
            deltas.append(_safe_mean(per_q["alright"]) - _safe_mean(per_q["non"]))

    total_questions = len(grouped)
    n_mixed = len(deltas)
    mixed_frac = n_mixed / total_questions if total_questions else 0.0
    return _safe_mean(deltas), n_mixed, mixed_frac


def _within_q_per_class(
    *,
    question_ids: list[str],
    opener_classes: list[str],
    values: list[float | None],
) -> dict[str, dict[str, float]]:
    """Per-class within-question debiased statistics.

    For every question, group rollouts by opener class, then compute two
    per-class summaries that each weight every question equally:

    * ``within_q_mean[c]``  = mean over Qs where class ``c`` appeared of
      ``mean(value | class=c, Q)``. This is the "unbiased" pass-rate /
      advantage estimate for class ``c`` — the question-difficulty mix is
      removed because each Q contributes one number regardless of how many
      class-``c`` rollouts it produced.
    * ``within_q_minus_qmean[c]`` = mean over Qs where class ``c`` appeared
      *and* at least one non-``c`` rollout also appeared, of
      ``mean(value | class=c, Q) - mean(value | Q)``. This is the per-class
      deviation from the question's overall mean and is the per-class
      generalisation of ``alright_minus_non``.

    The accompanying ``n_questions`` counts give the denominator behind
    each statistic so a downstream consumer can ignore classes with low
    coverage.
    """
    grouped: dict[str, dict[str, list[float]]] = {}
    for qid, opener, value in zip(question_ids, opener_classes, values):
        if value is None:
            continue
        per_q = grouped.setdefault(qid, {})
        per_q.setdefault(opener, []).append(value)

    within_mean: dict[str, list[float]] = {cls: [] for cls in OPENER_CLASSES}
    minus_qmean: dict[str, list[float]] = {cls: [] for cls in OPENER_CLASSES}

    for per_q in grouped.values():
        all_values: list[float] = []
        for vals in per_q.values():
            all_values.extend(vals)
        if not all_values:
            continue
        q_mean = _safe_mean(all_values)
        for cls in OPENER_CLASSES:
            cls_vals = per_q.get(cls)
            if not cls_vals:
                continue
            cls_mean = _safe_mean(cls_vals)
            within_mean[cls].append(cls_mean)
            # Only score the deviation when the question has another class
            # in it, otherwise the deviation is trivially zero and would
            # silently bias the estimate towards 0 for dominant classes.
            other_present = any(
                other_cls != cls and per_q.get(other_cls)
                for other_cls in per_q.keys()
            )
            if other_present:
                minus_qmean[cls].append(cls_mean - q_mean)

    return {
        "within_q_mean": {cls: _safe_mean(within_mean[cls]) for cls in OPENER_CLASSES},
        "within_q_minus_qmean": {
            cls: _safe_mean(minus_qmean[cls]) for cls in OPENER_CLASSES
        },
        "n_questions_with_class": {
            cls: float(len(within_mean[cls])) for cls in OPENER_CLASSES
        },
        "n_mixed_questions_with_class": {
            cls: float(len(minus_qmean[cls])) for cls in OPENER_CLASSES
        },
    }

def compute_pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k estimator (Chen et al., "Evaluating Large Language
    Models Trained on Code", 2021).

    Args:
        n: Total number of samples.
        c: Number of correct samples.
        k: k value for pass@k.

    Returns:
        Estimated pass@k probability, or ``None`` if ``n < k``.
    """
    if n < k:
        return None
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def _token_length_stats(
    response_mask: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Return mean/std/min/max token lengths from a response mask.

    Args:
        response_mask: ``[N, resp_len]`` binary mask.
        prefix: Metric key prefix (e.g. ``"dev_rollout/response_length_tokens"``).
    """
    if response_mask.numel() == 0:
        return {
            f"{prefix}/mean": 0.0,
            f"{prefix}/std": 0.0,
            f"{prefix}/min": 0.0,
            f"{prefix}/max": 0.0,
        }
    lengths = response_mask.sum(dim=-1).float().cpu()
    return {
        f"{prefix}/mean": lengths.mean().item(),
        f"{prefix}/std": lengths.std().item() if lengths.numel() > 1 else 0.0,
        f"{prefix}/min": lengths.min().item(),
        f"{prefix}/max": lengths.max().item(),
    }


# ------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------

def compute_answer_rollout_metrics(
    scores: dict[str, list[float | None]],
    response_mask: torch.Tensor,
    rollout_n: int,
    prefix: str,
) -> dict[str, float]:
    """Detailed metrics for an answer rollout stage.

    Args:
        scores: ``{question_id: [score_per_sample]}`` where score is
            ``1.0`` (correct), ``0.0`` (wrong), or ``None`` (extraction failed).
        response_mask: ``[N, resp_len]`` binary mask from the rollout output.
        rollout_n: Number of rollout samples per question.
        prefix: Metric key prefix (e.g. ``"dev_rollout"``).
    """
    metrics: dict[str, float] = {}

    # --- Accuracy & extraction stats ---
    all_scores = [s for sl in scores.values() for s in sl]
    total = len(all_scores)
    valid = [s for s in all_scores if s is not None]
    num_valid = len(valid)
    num_correct = sum(1 for s in valid if s > 0)
    num_failed = total - num_valid

    metrics[f"{prefix}/accuracy_strict"] = num_correct / total if total else 0.0
    metrics[f"{prefix}/accuracy_lenient"] = num_correct / num_valid if num_valid else 0.0
    metrics[f"{prefix}/num_questions"] = float(len(scores))
    metrics[f"{prefix}/num_generated_answers"] = float(total)
    metrics[f"{prefix}/num_valid_answers"] = float(num_valid)
    metrics[f"{prefix}/num_invalid_answers"] = float(num_failed)
    metrics[f"{prefix}/frac_valid_answers"] = num_valid / total if total else 0.0

    # Fraction of questions with diverse scores (not all same after None→0.0)
    num_diverse = 0
    num_questions = len(scores)
    for q_scores in scores.values():
        resolved = [s if s is not None else 0.0 for s in q_scores]
        if resolved and min(resolved) != max(resolved):
            num_diverse += 1
    metrics[f"{prefix}/frac_diverse_questions"] = num_diverse / num_questions if num_questions else 0.0

    # --- Pass@k ---
    # k values: powers of 2 up to rollout_n, plus rollout_n itself
    k_values = set()
    k = 1
    while k <= rollout_n:
        k_values.add(k)
        k *= 2
    k_values.add(rollout_n)

    for k_val in sorted(k_values):
        # Strict: None counts as wrong
        strict_scores_per_q: list[float] = []
        # Lenient: None excluded
        lenient_scores_per_q: list[float] = []

        for qid, q_scores in scores.items():
            # Strict
            n_strict = len(q_scores)
            c_strict = sum(1 for s in q_scores if s is not None and s > 0)
            pak = compute_pass_at_k(n_strict, c_strict, k_val)
            if pak is not None:
                strict_scores_per_q.append(pak)

            # Lenient
            valid_q = [s for s in q_scores if s is not None]
            n_lenient = len(valid_q)
            c_lenient = sum(1 for s in valid_q if s > 0)
            pak_len = compute_pass_at_k(n_lenient, c_lenient, k_val)
            if pak_len is not None:
                lenient_scores_per_q.append(pak_len)

        metrics[f"{prefix}/pass_at_{k_val}_strict"] = (
            float(np.mean(strict_scores_per_q)) if strict_scores_per_q else 0.0
        )
        metrics[f"{prefix}/pass_at_{k_val}_lenient"] = (
            float(np.mean(lenient_scores_per_q)) if lenient_scores_per_q else 0.0
        )

    # --- Response token lengths ---
    metrics.update(_token_length_stats(response_mask, f"{prefix}/response_length_tokens"))

    return metrics


def compute_opener_rollout_metrics(
    output: Any,
    tokenizer: Any,
    prefix: str,
) -> dict[str, float]:
    """Compute opener-basin metrics for a solver rollout and annotate rows.

    Mutates ``output.non_tensor_batch`` by adding ``opener_class`` so later
    solver-update batches can preserve the class for advantage diagnostics.

    Also emits ``{prefix}/mid_eos/...`` metrics — overall and per opener
    class — surfacing the rate of "EOS'd without \\boxed{}" rollouts so
    the impact of any ``algorithm.solver_mid_eos_penalty`` can be tracked
    in wandb.
    """
    responses = output.batch["responses"]
    response_mask = output.batch["response_mask"]
    batch_size = int(responses.shape[0])

    opener_classes = [
        classify_opener(_decode_response_prefix(tokenizer, responses[i]))
        for i in range(batch_size)
    ]
    output.non_tensor_batch["opener_class"] = np.array(opener_classes, dtype=object)

    lengths = response_mask.float().sum(dim=-1).detach().cpu().tolist()
    question_ids_raw = output.non_tensor_batch.get("question_id", [])
    question_ids = [str(qid) for qid in question_ids_raw]
    scores_raw = output.non_tensor_batch.get("answer_score", [None] * batch_size)
    scores = [_valid_score(score) for score in scores_raw]

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    mid_eos_mask = _compute_mid_eos_mask_np(
        responses=responses,
        response_mask=response_mask,
        answer_scores=scores_raw,
        eos_token_id=eos_token_id,
    )

    metrics: dict[str, float] = {}
    counts = {cls: opener_classes.count(cls) for cls in OPENER_CLASSES}
    total = float(batch_size)
    qs_with_class: dict[str, set[str]] = {cls: set() for cls in OPENER_CLASSES}
    for qid, oc in zip(question_ids, opener_classes):
        if oc in qs_with_class:
            qs_with_class[oc].add(qid)
    n_unique_qs = len(set(question_ids)) if question_ids else 0

    for cls in OPENER_CLASSES:
        cls_indices = [i for i, opener in enumerate(opener_classes) if opener == cls]
        metrics[f"{prefix}/opener/share/{cls}"] = counts[cls] / total if total else 0.0
        metrics[f"{prefix}/opener/count/{cls}"] = float(counts[cls])
        metrics[f"{prefix}/opener/mean_response_len/{cls}"] = _safe_mean(
            [float(lengths[i]) for i in cls_indices]
        )
        # Q-level share = # Qs with ≥1 cls rollout / total Qs in this pass.
        # Complements rollout-level share: which classes are appearing
        # *anywhere* vs *most often* across the question pool.
        n_qs_cls = len(qs_with_class[cls])
        metrics[f"{prefix}/opener/q_share/{cls}"] = (
            n_qs_cls / n_unique_qs if n_unique_qs else 0.0
        )
        metrics[f"{prefix}/opener/q_count/{cls}"] = float(n_qs_cls)
        # Per-class mid-EOS share = mid_eos rows in this class / rows in this class.
        cls_n = len(cls_indices)
        cls_mid_eos = int(sum(1 for i in cls_indices if mid_eos_mask[i]))
        metrics[f"{prefix}/mid_eos/share/{cls}"] = (
            cls_mid_eos / cls_n if cls_n else 0.0
        )
        metrics[f"{prefix}/mid_eos/count/{cls}"] = float(cls_mid_eos)

    # Overall mid-EOS metrics (across all classes).
    n_mid_eos = int(mid_eos_mask.sum())
    metrics[f"{prefix}/mid_eos/share"] = n_mid_eos / total if total else 0.0
    metrics[f"{prefix}/mid_eos/count"] = float(n_mid_eos)

    metrics[f"{prefix}/opener/top_share"] = (
        max(counts.values()) / total if total else 0.0
    )
    metrics[f"{prefix}/opener/n_unique_questions"] = float(n_unique_qs)

    delta, n_mixed, mixed_frac = _within_alright_minus_non(
        question_ids=question_ids,
        opener_classes=opener_classes,
        values=scores,
    )
    metrics[f"{prefix}/opener/delta_within/alright_minus_non"] = delta
    metrics[f"{prefix}/opener/delta_within/n_mixed_questions"] = float(n_mixed)
    metrics[f"{prefix}/opener/delta_within/mixed_question_frac"] = mixed_frac

    per_class = _within_q_per_class(
        question_ids=question_ids,
        opener_classes=opener_classes,
        values=scores,
    )
    for cls in OPENER_CLASSES:
        metrics[f"{prefix}/opener/within_q/acc/{cls}"] = per_class["within_q_mean"][cls]
        metrics[f"{prefix}/opener/within_q/acc_minus_qmean/{cls}"] = (
            per_class["within_q_minus_qmean"][cls]
        )
        metrics[f"{prefix}/opener/within_q/n_questions/{cls}"] = (
            per_class["n_questions_with_class"][cls]
        )
        metrics[f"{prefix}/opener/within_q/n_mixed_questions/{cls}"] = (
            per_class["n_mixed_questions_with_class"][cls]
        )
    return metrics


def compute_opener_advantage_metrics(
    batch: Any,
    prefix: str,
) -> dict[str, float]:
    """Compute opener-basin diagnostics from a solver PPO batch with advantages."""
    if "advantages" not in batch.batch.keys():
        return {}
    if "opener_class" not in batch.non_tensor_batch:
        return {}

    advantages = batch.batch["advantages"]
    response_mask = batch.batch["response_mask"].float()
    denom = response_mask.sum(dim=-1).clamp_min(1e-8)
    seq_advantages = (
        (advantages * response_mask).sum(dim=-1) / denom
    ).detach().cpu().tolist()

    opener_classes = [str(v) for v in batch.non_tensor_batch["opener_class"]]
    question_ids = [str(v) for v in batch.non_tensor_batch["uid"]]

    metrics: dict[str, float] = {}
    qs_with_class: dict[str, set[str]] = {cls: set() for cls in OPENER_CLASSES}
    for qid, oc in zip(question_ids, opener_classes):
        if oc in qs_with_class:
            qs_with_class[oc].add(qid)
    n_unique_qs = len(set(question_ids)) if question_ids else 0

    for cls in OPENER_CLASSES:
        vals = [
            float(seq_advantages[i])
            for i, opener in enumerate(opener_classes)
            if opener == cls
        ]
        metrics[f"{prefix}/opener/advantage_mean/{cls}"] = _safe_mean(vals)
        metrics[f"{prefix}/opener/advantage_count/{cls}"] = float(len(vals))
        n_qs_cls = len(qs_with_class[cls])
        metrics[f"{prefix}/opener/q_share/{cls}"] = (
            n_qs_cls / n_unique_qs if n_unique_qs else 0.0
        )
        metrics[f"{prefix}/opener/q_count/{cls}"] = float(n_qs_cls)
    metrics[f"{prefix}/opener/n_unique_questions"] = float(n_unique_qs)

    delta, n_mixed, mixed_frac = _within_alright_minus_non(
        question_ids=question_ids,
        opener_classes=opener_classes,
        values=[float(v) for v in seq_advantages],
    )
    metrics[f"{prefix}/opener/advantage_delta_within/alright_minus_non"] = delta
    metrics[f"{prefix}/opener/advantage_delta_within/n_mixed_questions"] = float(n_mixed)
    metrics[f"{prefix}/opener/advantage_delta_within/mixed_question_frac"] = mixed_frac

    per_class = _within_q_per_class(
        question_ids=question_ids,
        opener_classes=opener_classes,
        values=[float(v) for v in seq_advantages],
    )
    for cls in OPENER_CLASSES:
        metrics[f"{prefix}/opener/within_q/advantage/{cls}"] = (
            per_class["within_q_mean"][cls]
        )
        metrics[f"{prefix}/opener/within_q/advantage_minus_qmean/{cls}"] = (
            per_class["within_q_minus_qmean"][cls]
        )
        metrics[f"{prefix}/opener/within_q/n_questions/{cls}"] = (
            per_class["n_questions_with_class"][cls]
        )
        metrics[f"{prefix}/opener/within_q/n_mixed_questions/{cls}"] = (
            per_class["n_mixed_questions_with_class"][cls]
        )
    return metrics


def compute_question_rollout_metrics(
    gen_questions: list[dict[str, Any]],
    total_samples: int,
    response_mask: torch.Tensor,
    num_documents: int,
    prefix: str,
    reject_reasons: np.ndarray | None = None,
) -> dict[str, float]:
    """Detailed metrics for a question-generation rollout stage.

    Args:
        gen_questions: Parsed generated questions (each has ``question_text``,
            ``choices``, ``doc_id``; free-form rows use ``choices=[]``).
        total_samples: Total number of generated samples (before parsing).
        response_mask: ``[N, resp_len]`` binary mask from the rollout output.
        num_documents: Number of source documents in the batch.
        prefix: Metric key prefix (e.g. ``"gen_question_rollout"``).
        reject_reasons: Optional per-sample reject reasons from
            ``parse_generated_questions()``.  When provided, emits granular
            per-category rejection counts.
    """
    metrics: dict[str, float] = {}

    num_questions = len(gen_questions)
    num_valid = num_questions  # successfully parsed
    num_failed = total_samples - num_valid

    metrics[f"{prefix}/num_documents"] = float(num_documents)
    metrics[f"{prefix}/valid_questions_per_doc"] = (
        num_questions / num_documents if num_documents else 0.0
    )
    metrics[f"{prefix}/num_generated_questions"] = float(total_samples)
    metrics[f"{prefix}/num_valid_questions"] = float(num_valid)
    metrics[f"{prefix}/num_invalid_questions"] = float(num_failed)
    metrics[f"{prefix}/frac_valid_questions"] = (
        num_valid / total_samples if total_samples else 0.0
    )

    # Number of choices per question
    num_choices = [float(len(q.get("choices", []))) for q in gen_questions] if gen_questions else []
    add_distribution_stats(metrics, f"{prefix}/num_choices", num_choices)

    # Question text character length
    char_lens = [float(len(q.get("question_text", ""))) for q in gen_questions] if gen_questions else []
    add_distribution_stats(metrics, f"{prefix}/question_char_len", char_lens)

    # Per-category rejection breakdown
    if reject_reasons is not None:
        reasons = list(reject_reasons)
        metrics[f"{prefix}/num_failed_parses"] = float(reasons.count("failed_parse"))
        metrics[f"{prefix}/num_empty_questions"] = float(reasons.count("empty_question"))
        metrics[f"{prefix}/num_invalid_ground_truth"] = float(reasons.count("invalid_ground_truth"))

    # Response token lengths
    metrics.update(_token_length_stats(response_mask, f"{prefix}/response_length_tokens"))

    return metrics


def compute_reward_metrics(
    rewards: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    """Distribution metrics for per-question rewards.

    Args:
        rewards: ``{question_id: reward_value}``.
        prefix: Metric key prefix (e.g. ``"reward"``).
    """
    metrics: dict[str, float] = {}
    vals = list(rewards.values())

    add_distribution_stats(metrics, prefix, vals, extra_stats=("median",))
    if vals:
        arr = np.array(vals)
        metrics[f"{prefix}/num_scored_questions"] = float(len(vals))
        num_positive = int(np.sum(arr > 0))
        metrics[f"{prefix}/num_positive"] = float(num_positive)
        metrics[f"{prefix}/frac_positive"] = num_positive / len(vals)
    else:
        metrics[f"{prefix}/num_scored_questions"] = 0.0
        metrics[f"{prefix}/num_positive"] = 0.0
        metrics[f"{prefix}/frac_positive"] = 0.0

    return metrics


def compute_ans_loop_summary_metrics(
    gen_loop_results: list[dict[str, Any]],
    prefix: str,
) -> dict[str, float]:
    """Summary metrics across all gen-loops in an ans_loop.

    Args:
        gen_loop_results: List of dicts, each containing ``"rewards"``,
            ``"gen_scores"``, and ``"gen_questions"`` from one gen-loop.
        prefix: Metric key prefix (e.g. ``"ans_loop"``).
    """
    metrics: dict[str, float] = {}

    metrics[f"{prefix}/num_gen_loops"] = float(len(gen_loop_results))

    all_rewards = [v for r in gen_loop_results for v in r["rewards"].values()]
    metrics[f"{prefix}/mean_reward"] = float(np.mean(all_rewards)) if all_rewards else 0.0

    total_questions = sum(len(r["gen_questions"]) for r in gen_loop_results)
    metrics[f"{prefix}/total_questions_generated"] = float(total_questions)

    all_valid = []
    for r in gen_loop_results:
        for sl in r["gen_scores"].values():
            all_valid.extend(s for s in sl if s is not None)
    metrics[f"{prefix}/mean_gen_accuracy"] = float(np.mean(all_valid)) if all_valid else 0.0

    return metrics
