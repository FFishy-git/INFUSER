"""Entry point for Solver Evaluation on Benchmarks.

Supports two backends:
- ``ray`` (default): Uses verl Ray worker group (same inference path as training).
- ``vllm``: Direct vLLM with DP across GPUs (no Ray/verl dependency).

Usage::

    # Ray backend (default)
    python -m verl_inf_evolve.sol_eval.sol_eval sol_eval_experiment=debug-run

    # vLLM backend
    python -m verl_inf_evolve.sol_eval.sol_eval eval.backend=vllm eval.run_name=my-run

    # With overrides
    python -m verl_inf_evolve.sol_eval.sol_eval eval.backend=vllm eval.checkpoints=[0,5,10] eval.n_samples=8
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from verl_inf_evolve.sol_eval.experiment_discovery import (
    canonicalize_model_id,
    detect_model_from_remote_path,
    discover_training_experiment,
)
from verl_inf_evolve.templates import load_source
from verl_inf_evolve.utils.config_resolvers import register_config_template_resolvers
from verl_inf_evolve.utils.env_utils import load_startup_env

load_startup_env()

logger = logging.getLogger(__name__)

# Minimal Llama-3 chat template (Jinja2) for base models that lack a native one.
# Uses special tokens already in the Llama-3 / OctoThinker vocabulary.
_LLAMA3_CHAT_TEMPLATE = (
    "{{bos_token}}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
    " + message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{%- endif %}"
)
_OCTOTHINKER_SHARED_CHAT_TEMPLATE = load_source("octothinker_shared_chat_template.jinja")


def _chat_template_for_model(model_path: str) -> str | None:
    """Return a built-in chat template when the model needs one."""
    if model_path == "meta-llama/Llama-3.1-8B":
        return _LLAMA3_CHAT_TEMPLATE
    if model_path.startswith("OctoThinker/OctoThinker-"):
        return _OCTOTHINKER_SHARED_CHAT_TEMPLATE
    return None


def _detect_model_from_remote_path(remote_sync_path: str) -> tuple[str, str | None] | None:
    """Detect model HF path and chat template from a remote path fragment."""
    model_path = detect_model_from_remote_path(remote_sync_path)
    if model_path is None:
        return None
    return model_path, _chat_template_for_model(model_path)


def _detect_model_from_run_metadata(config) -> tuple[str, str | None] | None:  # type: ignore[no-untyped-def]
    """Prefer ``run_metadata.json`` for model auto-detection."""
    remote_sync_path = config.eval.get("remote_sync_path", None)
    if not remote_sync_path:
        return None

    from verl_inf_evolve.sol_eval.r2_ops import load_run_metadata

    metadata = load_run_metadata(str(remote_sync_path), config.get("remote", {}))
    if not metadata:
        return None

    solver_model_id = metadata.get("solver_model_id")
    if not solver_model_id:
        return None

    model_path = str(solver_model_id)
    return model_path, _chat_template_for_model(model_path)


def _maybe_canonicalize_requested_model(config) -> None:  # type: ignore[no-untyped-def]
    """Normalize user-facing model aliases into canonical HF model IDs."""
    for key in ("eval.model_path", "solver.model.path"):
        current = OmegaConf.select(config, key)
        if not current:
            continue
        canonical = canonicalize_model_id(str(current))
        if canonical and canonical != current:
            print(f"Normalized {key}: {current} -> {canonical}")
            OmegaConf.update(config, key, canonical)


def _discover_eval_remote_sync_path(config) -> None:  # type: ignore[no-untyped-def]
    """Infer ``eval.remote_sync_path`` from local training configs when omitted."""
    from verl_inf_evolve.sol_eval.config import parse_checkpoints

    if config.eval.get("remote_sync_path", None):
        return
    checkpoints = config.eval.get("checkpoints", None)
    try:
        checkpoint_list = parse_checkpoints(checkpoints) if checkpoints is not None else []
    except ValueError:
        checkpoint_list = []
    if checkpoint_list == [-1]:
        logger.info(
            "Skipping eval.remote_sync_path auto-discovery for base-model-only eval "
            "(checkpoints=[-1])"
        )
        return
    run_name = config.eval.get("run_name", None)
    if not run_name:
        return

    default_model = "Qwen/Qwen3-8B"
    requested_model = str(config.eval.get("model_path", None) or config.solver.model.path)
    model_is_default = (
        str(config.eval.get("model_path", default_model)) == default_model
        and str(config.solver.model.path) == default_model
    )
    discovery_model = None if model_is_default else requested_model

    match = discover_training_experiment(
        run_name=str(run_name),
        model_name=discovery_model,
    )
    print(f"Auto-discovered training config: {match.path}")
    print(f"Auto-discovered eval.remote_sync_path: {match.remote_sync_path}")
    OmegaConf.update(config, "eval.remote_sync_path", match.remote_sync_path)
    OmegaConf.update(config, "eval.model_path", match.model_path)
    OmegaConf.update(config, "solver.model.path", match.model_path)


def _resolve_eval_remote_paths(config) -> None:  # type: ignore[no-untyped-def]
    """Resolve HF placeholder URIs up front and seed ``HF_TOKEN`` if needed."""
    from verl_inf_evolve.sol_eval.r2_ops import resolve_remote_uri

    for key in ("eval.remote_sync_path", "eval.remote_eval_base", "eval.r2_eval_base"):
        current = OmegaConf.select(config, key)
        if not current:
            continue
        resolved_uri, hf_token = resolve_remote_uri(str(current), config.get("remote", {}))
        if resolved_uri != current:
            print(f"Resolved {key}: {current} -> {resolved_uri}")
            OmegaConf.update(config, key, resolved_uri)
        if hf_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = hf_token


def _sync_eval_overrides_into_runtime(config) -> None:  # type: ignore[no-untyped-def]
    """Mirror eval-facing overrides into the actual solver rollout config."""
    default_model = "Qwen/Qwen3-8B"
    eval_model = config.eval.get("model_path", None)
    current_model = config.solver.model.path
    if eval_model and current_model == default_model and eval_model != default_model:
        OmegaConf.update(config, "solver.model.path", eval_model)

    OmegaConf.update(config, "solver.rollout.n", config.eval.n_samples)
    OmegaConf.update(
        config,
        "solver.rollout.tensor_model_parallel_size",
        config.eval.tp_size,
    )
    OmegaConf.update(
        config,
        "solver.rollout.gpu_memory_utilization",
        config.eval.gpu_memory_utilization,
    )
    OmegaConf.update(config, "eval.model_path", config.solver.model.path)

    inferred_template = _chat_template_for_model(str(config.solver.model.path))
    if inferred_template and not config.solver.model.get("custom_chat_template", None):
        OmegaConf.update(config, "solver.model.custom_chat_template", inferred_template)


def _validate_hf_model_cache(model_path: str) -> None:
    """Check for corrupted HF model cache and clean it up.

    When a previous download was interrupted (e.g., SLURM job killed,
    network timeout), HF hub leaves behind ``*.incomplete`` files that
    cause ``from_pretrained`` to fail with a misleading "files not found"
    error.  This function detects and removes such partial downloads so
    the next ``from_pretrained`` call triggers a clean re-download.
    """
    if "/" not in model_path or model_path.startswith("/"):
        return  # Local path, not an HF model ID

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = Path(hf_home) / "hub" / f"models--{model_path.replace('/', '--')}"

    if not cache_dir.exists():
        return  # No cache yet, fresh download will happen

    blobs_dir = cache_dir / "blobs"
    if not blobs_dir.exists():
        return

    incomplete = list(blobs_dir.glob("*.incomplete"))
    if not incomplete:
        return

    total_bytes = sum(f.stat().st_size for f in incomplete)
    print(
        f"[sol_eval] WARNING: Found {len(incomplete)} incomplete files "
        f"({total_bytes / 1e9:.1f} GB) in HF cache for {model_path}. "
        f"Removing corrupted cache to force clean re-download."
    )
    shutil.rmtree(cache_dir)
    print(f"[sol_eval] Removed {cache_dir}")


def _pre_download_model_if_needed(model_path: str) -> None:
    """Pre-download an HF model in the driver before Ray workers start.

    When ``pre_download_models.py`` can't detect the runtime-resolved model
    (e.g. sol_eval auto-detection from remote metadata), all Ray workers
    would download concurrently, corrupting the HF cache.  Downloading
    once here in the driver avoids the race.
    """
    if "/" not in model_path or model_path.startswith("/"):
        return  # Local path, skip

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = Path(hf_home) / "hub" / f"models--{model_path.replace('/', '--')}"
    snapshots = cache_dir / "snapshots"

    if snapshots.exists() and any(snapshots.iterdir()):
        return  # Already cached

    print(f"[sol_eval] Pre-downloading {model_path} in driver to avoid worker race...")
    from huggingface_hub import snapshot_download

    snapshot_download(model_path)
    print(f"[sol_eval] Pre-download complete: {model_path}")


@hydra.main(config_path="../config", config_name="sol_eval", version_base=None)
def main(config):  # type: ignore[no-untyped-def]
    run_sol_eval(config)


def _resolve_config(config) -> None:  # type: ignore[no-untyped-def]
    """Shared config resolution for both Ray and vLLM backends."""
    from verl_inf_evolve.sol_eval.config import parse_checkpoints

    register_config_template_resolvers()
    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, False)

    _maybe_canonicalize_requested_model(config)
    _discover_eval_remote_sync_path(config)
    _resolve_eval_remote_paths(config)

    if config.eval.remote_sync_path:
        detected = _detect_model_from_run_metadata(config)
        if detected is None:
            detected = _detect_model_from_remote_path(config.eval.remote_sync_path)
        if detected:
            detected_model, detected_template = detected
            default_model = "Qwen/Qwen3-8B"
            current_model = config.solver.model.path
            if current_model == default_model and detected_model != default_model:
                print(f"Auto-detected model from remote path: {detected_model}")
                OmegaConf.update(config, "solver.model.path", detected_model)
                OmegaConf.update(config, "eval.model_path", detected_model)
                if detected_template:
                    print(f"Auto-detected custom chat template for base model")
                    OmegaConf.update(config, "solver.model.custom_chat_template", detected_template)

    _sync_eval_overrides_into_runtime(config)

    if config.eval.checkpoints is None:
        from verl_inf_evolve.sol_eval.r2_ops import discover_checkpoints_on_r2

        checkpoints = discover_checkpoints_on_r2(
            config.eval.remote_sync_path,
            remote_cfg=config.get("remote", {}),
        )
        print(f"Auto-discovered {len(checkpoints)} checkpoints: {checkpoints}")
    else:
        checkpoints = parse_checkpoints(config.eval.checkpoints)
    OmegaConf.update(config, "eval.checkpoints", checkpoints, force_add=True)

    _validate_hf_model_cache(config.solver.model.path)
    _pre_download_model_if_needed(config.solver.model.path)


def _print_summary(config, results: list[dict]) -> None:
    """Print evaluation summary table."""
    print()
    print("=" * 60)
    print("Solver Evaluation Summary")
    print("=" * 60)
    print(f"  Run:         {config.eval.run_name}")
    print(f"  Backend:     {config.eval.get('backend', 'ray')}")
    print(f"  Evaluations: {len(results)}")
    for result in results:
        meta = result.get("_eval_metadata", {})
        metrics = result.get("metrics", {})
        benchmark = meta.get("benchmark", "?")
        ckpt_num = meta.get("ckpt_num", "?")
        primary_metric_name = metrics.get("primary_metric_name", "accuracy_strict")
        primary_metric_value = metrics.get(
            "primary_metric_value",
            metrics.get("accuracy_strict", 0.0),
        )
        acc_strict = metrics.get("accuracy_strict", 0.0)
        acc_lenient = metrics.get("accuracy_lenient", 0.0)
        summary = (
            f"    checkpoint {ckpt_num} / {benchmark}: "
            f"{primary_metric_name}={primary_metric_value:.4f}, "
            f"strict={acc_strict:.4f}, lenient={acc_lenient:.4f}"
        )
        print(summary)
    print("=" * 60)


def run_sol_eval(config) -> None:  # type: ignore[no-untyped-def]
    """Launch solver evaluation using the configured backend.

    Supports two backends:
    - ``ray`` (default): Uses Ray + verl worker group (``EvalRunner``).
    - ``vllm``: Uses direct vLLM with DP across GPUs (``VllmEvalRunner``).

    Select via ``eval.backend=vllm`` on the command line or in the config.
    """
    from pprint import pprint

    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.root.setLevel(getattr(logging, log_level, logging.INFO))

    # Install a safety SIGALRM handler so stray alarms raise a catchable
    # Python exception instead of killing the process with exit code 14.
    # Benchmark scorers (e.g. PHYBench EED) and math_verify use setitimer /
    # signal.alarm internally; if cleanup races leave a pending alarm, this
    # handler converts it to a warning rather than a process-fatal signal.
    import signal as _signal

    def _safety_sigalrm(signum, frame):  # noqa: ARG001
        logging.getLogger("sol_eval").warning(
            "Caught stray SIGALRM (no active timeout handler) — ignoring"
        )

    _signal.signal(_signal.SIGALRM, _safety_sigalrm)

    print(f"sol_eval hostname: {socket.gethostname()}, PID: {os.getpid()}")

    _resolve_config(config)

    # Configure the optional LLM math judge singleton from sol_eval.yaml
    # before any worker scores a sample. Opt-in via
    # ``eval.math500_llm_judge.enabled`` and scoped to the benchmarks listed
    # in ``applies_to`` (default: math500 / math).
    _judge_cfg_raw = config.eval.get("math500_llm_judge", None)
    if _judge_cfg_raw is not None:
        from verl_inf_evolve.utils.benchmarks.verifiers import _llm_math_judge as _math_judge

        _judge_cfg = OmegaConf.to_container(_judge_cfg_raw, resolve=True) or {}
        _applies_to = _judge_cfg.get("applies_to") or ("math500", "math")
        # temperature is allowed to be None (→ omit the parameter when
        # calling the OpenAI API). Don't wrap None in float().
        _raw_temp = _judge_cfg.get("temperature")
        _math_judge.configure(
            enabled=bool(_judge_cfg.get("enabled", False)),
            model=str(_judge_cfg.get("model", "gpt-4o")),
            api_base=_judge_cfg.get("api_base"),
            temperature=(None if _raw_temp is None else float(_raw_temp)),
            max_retries=int(_judge_cfg.get("max_retries", 2)),
            request_timeout=float(_judge_cfg.get("request_timeout", 30.0)),
            applies_to=tuple(_applies_to),
            max_concurrent=int(_judge_cfg.get("max_concurrent", 16)),
            mode=str(_judge_cfg.get("mode", "direct")),
            prompt_style=str(_judge_cfg.get("prompt_style", "gr_equality")),
        )

    from verl_inf_evolve.storage.remote_backend import redact_config_secrets

    pprint(redact_config_secrets(OmegaConf.to_container(config, resolve=True)))

    backend = str(config.eval.get("backend", "ray")).lower()
    print(f"Using backend: {backend}")

    if backend == "vllm":
        from verl_inf_evolve.sol_eval.runner import VllmEvalRunner

        runner = VllmEvalRunner(config)
    else:
        from verl_inf_evolve.sol_eval.runner import EvalRunner

        runner = EvalRunner(config)

    try:
        results = runner.evaluate_run()
        _print_summary(config, results)
    finally:
        runner.shutdown()

    # Report LLM judge activity if it was enabled for this run.
    if _judge_cfg_raw is not None:
        from verl_inf_evolve.utils.benchmarks.verifiers import _llm_math_judge as _math_judge

        _stats = _math_judge.get_stats()
        if _stats["calls"] > 0 or _stats["upgrades"] > 0 or _stats["api_errors"] > 0:
            print(
                f"LLM math judge stats: calls={_stats['calls']} "
                f"upgrades={_stats['upgrades']} "
                f"api_errors={_stats['api_errors']} "
                f"parse_errors={_stats['parse_errors']}"
            )

    if backend != "vllm":
        timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
        if timeline_json_file:
            import ray

            ray.timeline(filename=timeline_json_file)


if __name__ == "__main__":
    main()
