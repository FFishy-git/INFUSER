"""Pure helper utilities for variance sensitivity scoring."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

STATUS_OK = "ok"
STATUS_RETRY_EXHAUSTED = "retry_exhausted"
STATUS_ZERO_VARIANCE = "zero_variance"

CASE_BOTH_NONZERO = "both_nonzero"
CASE_BOTH_ZERO = "both_zero"
CASE_ONLY_A_NONZERO = "only_a_nonzero"
CASE_ONLY_B_NONZERO = "only_b_nonzero"


def normalize_score(score: Any) -> float:
    return float(score) if score is not None else 0.0


def group_row_indices_by_qid(question_ids: list[Any] | np.ndarray) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, qid in enumerate(question_ids):
        grouped[str(qid)].append(i)
    return dict(grouped)


def slice_dataproto(data: DataProto, indices: list[int]) -> DataProto:
    idx_tensor = torch.tensor(indices, dtype=torch.long)

    new_batch = {}
    for key in data.batch.keys():
        new_batch[key] = data.batch[key][idx_tensor]

    new_non_tensor: dict[str, Any] = {}
    for key, arr in data.non_tensor_batch.items():
        if isinstance(arr, np.ndarray):
            new_non_tensor[key] = arr[indices]
        else:
            new_non_tensor[key] = [arr[i] for i in indices]

    for key, val in list(new_non_tensor.items()):
        if isinstance(val, list):
            new_non_tensor[key] = np.array(val, dtype=object)

    td = TensorDict(new_batch, batch_size=len(indices))
    result = DataProto(batch=td, non_tensor_batch=new_non_tensor)
    if hasattr(data, "meta_info") and data.meta_info:
        result.meta_info.update(data.meta_info)
    return result


def validate_rollout_n_values(rollout_n_values: list[int], dp_size: int) -> list[int]:
    if not rollout_n_values:
        raise ValueError("variance_sensitivity.rollout_n_values must be non-empty")

    deduped = sorted(set(int(n) for n in rollout_n_values))
    if any(n <= 0 for n in deduped):
        raise ValueError(
            "variance_sensitivity.rollout_n_values must contain only positive integers"
        )
    if any(n % dp_size != 0 for n in deduped):
        raise ValueError(
            f"All rollout_n_values must be divisible by solver dp_size={dp_size}, got {deduped}"
        )
    return deduped


def validate_replay_path(question_source_mode: str, replay_gen_output_path: Optional[str]) -> None:
    if question_source_mode != "replay":
        return
    if not replay_gen_output_path:
        raise ValueError(
            "variance_sensitivity.replay_gen_output_path must be set when question_source_mode='replay'"
        )
    from pathlib import Path

    if not Path(replay_gen_output_path).exists():
        raise FileNotFoundError(
            f"Replay gen_output.pt not found: {replay_gen_output_path}"
        )


def classify_variance_case(
    dev_batch: Optional[DataProto], gen_batch: Optional[DataProto]
) -> str:
    if dev_batch is None and gen_batch is None:
        return CASE_BOTH_ZERO
    if dev_batch is not None and gen_batch is None:
        return CASE_ONLY_A_NONZERO
    if dev_batch is None and gen_batch is not None:
        return CASE_ONLY_B_NONZERO
    return CASE_BOTH_NONZERO


def retry_score_point(
    *,
    qid: str,
    n: int,
    pool_a: list[int],
    pool_b: list[int],
    rng: random.Random,
    max_sampling_attempts: int,
    evaluate_attempt: Callable[[list[int], list[int]], tuple[str, Optional[dict[str, float]]]],
) -> dict[str, Any]:
    last_case = CASE_BOTH_ZERO
    for attempt in range(1, max_sampling_attempts + 1):
        sampled_a = rng.sample(pool_a, n)
        sampled_b = rng.sample(pool_b, n)
        variance_case, metrics = evaluate_attempt(sampled_a, sampled_b)
        last_case = variance_case
        if variance_case == CASE_BOTH_NONZERO and metrics is not None:
            return {
                "question_id": qid,
                "n": n,
                "cosine": metrics.get("cosine"),
                "dot": metrics.get("dot"),
                "grad_norm": metrics.get("grad_norm"),
                "ref_norm": metrics.get("ref_norm"),
                "status": STATUS_OK,
                "variance_case": variance_case,
                "attempts_used": attempt,
            }

    return {
        "question_id": qid,
        "n": n,
        "cosine": None,
        "dot": None,
        "grad_norm": None,
        "ref_norm": None,
        "status": STATUS_RETRY_EXHAUSTED,
        "variance_case": last_case,
        "attempts_used": max_sampling_attempts,
    }


def summarize_scores(
    scores_by_n: dict[int, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {"per_n": {}}
    thresholds = [0.5, 0.8, 0.9, 0.95]

    for n in sorted(scores_by_n):
        per_q = scores_by_n[n]
        values = [
            float(rec["cosine"])
            for rec in per_q.values()
            if rec.get("status") == STATUS_OK and rec.get("cosine") is not None
        ]
        total_points = len(per_q)
        valid_points = len(values)
        retry_exhausted = sum(
            1 for rec in per_q.values() if rec.get("status") == STATUS_RETRY_EXHAUSTED
        )
        zero_variance = sum(
            1 for rec in per_q.values() if rec.get("status") == STATUS_ZERO_VARIANCE
        )
        skipped_points = total_points - valid_points

        n_key = str(n)
        stats: dict[str, Any] = {
            "total_points": total_points,
            "valid_points": valid_points,
            "skipped_points": skipped_points,
            "retry_exhausted_points": retry_exhausted,
            "zero_variance_points": zero_variance,
        }

        if values:
            arr = np.array(values, dtype=np.float32)
            stats.update(
                {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "median": float(np.median(arr)),
                }
            )
            for thr in thresholds:
                stats[f"frac_gt_{str(thr).replace('.', 'p')}"] = float((arr > thr).mean())
        else:
            stats.update(
                {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "median": None,
                }
            )
            for thr in thresholds:
                stats[f"frac_gt_{str(thr).replace('.', 'p')}"] = 0.0

        summary["per_n"][n_key] = stats

    return summary


def format_summary_table(summary: dict[str, Any]) -> str:
    lines = []
    lines.append(
        "n     | valid/total | mean   | std    | median | min    | max    | >0.90  | >0.95"
    )
    lines.append(
        "------+-------------+--------+--------+--------+--------+--------+--------+--------"
    )
    for n_key in sorted(summary["per_n"], key=lambda x: int(x)):
        s = summary["per_n"][n_key]
        mean = "null" if s["mean"] is None else f"{s['mean']:.4f}"
        std = "null" if s["std"] is None else f"{s['std']:.4f}"
        median = "null" if s["median"] is None else f"{s['median']:.4f}"
        min_v = "null" if s["min"] is None else f"{s['min']:.4f}"
        max_v = "null" if s["max"] is None else f"{s['max']:.4f}"
        gt90 = f"{s['frac_gt_0p9']:.2f}"
        gt95 = f"{s['frac_gt_0p95']:.2f}"
        lines.append(
            f"{int(n_key):<5} | "
            f"{s['valid_points']:>5}/{s['total_points']:<5} | "
            f"{mean:>6} | {std:>6} | {median:>6} | {min_v:>6} | {max_v:>6} | "
            f"{gt90:>6} | {gt95:>6}"
        )
    return "\n".join(lines)
