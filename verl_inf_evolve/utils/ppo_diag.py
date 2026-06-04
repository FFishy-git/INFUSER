from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.metric import reduce_metrics


def format_diag_value(value: Any) -> str:
    """Format diagnostic values for compact one-line logs."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value_f = float(value)
        if not math.isfinite(value_f):
            return str(value_f)
        if abs(value_f) >= 100:
            return f"{value_f:.1f}"
        if abs(value_f) >= 1:
            return f"{value_f:.2f}"
        return f"{value_f:.4f}"
    return str(value)


def render_diag_map(diag: dict[str, Any]) -> str:
    """Render ordered diagnostic values into a compact log payload."""
    return ", ".join(
        f"{key}={format_diag_value(value)}"
        for key, value in diag.items()
    )


def diag_series_sum(value: Any) -> float:
    """Sum tensor/list metric payloads into a scalar for diagnostics."""
    if torch.is_tensor(value):
        return float(value.sum().item())
    if isinstance(value, np.ndarray):
        return float(np.sum(value))
    if isinstance(value, (list, tuple)):
        return float(sum(value))
    return float(value)


def metric_samples_for_reduce(value: Any) -> list[Any]:
    """Normalize worker metric payloads before reduce_metrics()."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return list(value)
    return [value]


def collect_ppo_batch_diag(batch: Optional[DataProto]) -> dict[str, Any]:
    """Extract compact batch diagnostics for PPO stage logs."""
    if batch is None:
        return {}

    diag: dict[str, Any] = {
        "batch_size": len(batch),
    }

    if batch.batch is not None:
        batch_keys = set(batch.batch.keys())
        if "attention_mask" in batch_keys:
            diag["input_tokens"] = int(diag_series_sum(batch.batch["attention_mask"]))
        if "response_mask" in batch_keys:
            response_lengths = batch.batch["response_mask"].sum(dim=-1)
            diag["response_tokens"] = int(response_lengths.sum().item())
            if response_lengths.numel() > 0:
                diag["avg_response_tokens"] = float(response_lengths.float().mean().item())
                diag["max_response_tokens"] = int(response_lengths.max().item())

    if batch.non_tensor_batch and "uid" in batch.non_tensor_batch:
        uid_values = batch.non_tensor_batch["uid"]
        if hasattr(uid_values, "tolist"):
            uid_values = uid_values.tolist()
        diag["uid_groups"] = len(dict.fromkeys(str(uid) for uid in uid_values))

    meta_info = batch.meta_info or {}
    for key in ("ppo_mini_batch_size", "dp_size", "temperature"):
        if key in meta_info:
            diag[key] = meta_info[key]
    if "global_token_num" in meta_info:
        diag["global_tokens"] = int(diag_series_sum(meta_info["global_token_num"]))

    return diag


def collect_ppo_actor_diag(actor_output: Optional[DataProto]) -> dict[str, Any]:
    """Extract compact actor-update diagnostics from worker outputs."""
    if actor_output is None:
        return {}
    raw_metrics = actor_output.meta_info.get("metrics", {})
    if not raw_metrics:
        return {}

    reduced = reduce_metrics(
        {
            key: metric_samples_for_reduce(value)
            for key, value in raw_metrics.items()
        }
    )
    diag: dict[str, Any] = {}
    selected_metrics = (
        ("perf/mfu/actor", "worker_mfu"),
        ("perf/max_memory_allocated_gb", "worker_max_alloc_gb"),
        ("perf/max_memory_reserved_gb", "worker_max_reserved_gb"),
        ("perf/cpu_memory_used_gb", "worker_cpu_mem_gb"),
        ("actor/lr", "worker_lr"),
        ("actor/num_minibatches", "worker_num_minibatches"),
        ("actor/ppo_mini_batch_size", "worker_ppo_mini_batch_size"),
        ("actor/ppo_micro_batch_size_per_gpu", "worker_ppo_micro_batch_size_per_gpu"),
        ("actor/total_batch_size", "worker_total_batch_size"),
    )
    for source_key, target_key in selected_metrics:
        if source_key in reduced:
            diag[target_key] = reduced[source_key]
    return diag


def collect_process_resource_diag() -> dict[str, Any]:
    """Collect trainer-process CPU and controller-GPU diagnostics."""
    diag: dict[str, Any] = {}

    try:
        import psutil

        mem = psutil.virtual_memory()
        diag["sys_mem_used_gb"] = mem.used / 1024**3
        diag["sys_mem_pct"] = mem.percent
        diag["proc_rss_gb"] = psutil.Process().memory_info().rss / 1024**3
    except Exception as exc:
        diag["sys_mem_error"] = exc

    try:
        if torch.cuda.is_available():
            allocs = [torch.cuda.memory_allocated(i) / 1024**3 for i in range(torch.cuda.device_count())]
            reserved = [torch.cuda.memory_reserved(i) / 1024**3 for i in range(torch.cuda.device_count())]
            if allocs:
                diag["ctrl_gpu_max_alloc_gb"] = max(allocs)
                diag["ctrl_gpu_max_reserved_gb"] = max(reserved)
    except Exception as exc:
        diag["ctrl_gpu_error"] = exc

    return diag
