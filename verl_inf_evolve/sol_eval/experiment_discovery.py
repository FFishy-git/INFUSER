"""Helpers for discovering training experiment configs for solver evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml

# Known path fragments to model IDs. This covers both:
# - legacy R2 paths like s3://.../running-states/V3_2_qwen3_4b_base/<run>
# - HF dataset paths like hf://datasets/<ns>/SER/qwen3_4b_base/<run>
REMOTE_PATH_TO_MODEL = {
    "V3_2": "Qwen/Qwen3-8B",
    "V3_2_qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "V3_2_llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "V3_2_octothinker_3b_hybrid_base": "OctoThinker/OctoThinker-3B-Hybrid-Base",
    "octothinker_3b_hybrid_base": "OctoThinker/OctoThinker-3B-Hybrid-Base",
    "V3_2_octothinker_8b_hybrid_base": "OctoThinker/OctoThinker-8B-Hybrid-Base",
    "octothinker_8b_hybrid_base": "OctoThinker/OctoThinker-8B-Hybrid-Base",
    "V3_2_qwen3_4b_base": "Qwen/Qwen3-4B-Base",
    "qwen3_4b_base": "Qwen/Qwen3-4B-Base",
    "V3_2_qwen3_8b_base": "Qwen/Qwen3-8B-Base",
    "qwen3_8b_base": "Qwen/Qwen3-8B-Base",
    "olmo_3_7b_instruct_sft": "allenai/Olmo-3-7B-Instruct-SFT",
}

_MODEL_ALIAS_GROUPS = {
    "Qwen/Qwen3-8B": {
        "Qwen/Qwen3-8B",
        "Qwen3-8B",
        "qwen3_8b",
        "qwen3-8b",
        "qw8b",
        "experiment",
        "V3_2",
    },
    "Qwen/Qwen3-4B": {
        "Qwen/Qwen3-4B",
        "Qwen3-4B",
        "qwen3_4b",
        "qwen3-4b",
        "qw4b",
        "experiment_qwen3_4b",
        "V3_2_qwen3_4b",
    },
    "Qwen/Qwen3-4B-Base": {
        "Qwen/Qwen3-4B-Base",
        "Qwen3-4B-Base",
        "qwen3_4b_base",
        "qwen3-4b-base",
        "qw4bb",
        "experiment_qwen3_4b_base",
        "V3_2_qwen3_4b_base",
    },
    "Qwen/Qwen3-8B-Base": {
        "Qwen/Qwen3-8B-Base",
        "Qwen3-8B-Base",
        "qwen3_8b_base",
        "qwen3-8b-base",
        "qw8bb",
        "experiment_qwen3_8b_base",
        "V3_2_qwen3_8b_base",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "meta-llama/Llama-3.1-8B-Instruct",
        "Llama-3.1-8B-Instruct",
        "llama31_8b",
        "llama31_8b_instruct",
        "llama-3.1-8b-instruct",
        "ll8bi",
        "experiment_llama31_8b",
        "V3_2_llama31_8b",
    },
    "OctoThinker/OctoThinker-3B-Hybrid-Base": {
        "OctoThinker/OctoThinker-3B-Hybrid-Base",
        "OctoThinker-3B-Hybrid-Base",
        "octothinker_3b_hybrid_base",
        "octothinker-3b-hybrid-base",
        "ot3bhb",
        "experiment_octothinker_3b_hybrid_base",
        "V3_2_octothinker_3b_hybrid_base",
    },
    "OctoThinker/OctoThinker-8B-Hybrid-Base": {
        "OctoThinker/OctoThinker-8B-Hybrid-Base",
        "OctoThinker-8B-Hybrid-Base",
        "octothinker_8b_hybrid_base",
        "octothinker-8b-hybrid-base",
        "ot8bhb",
        "experiment_octothinker_8b_hybrid_base",
        "V3_2_octothinker_8b_hybrid_base",
    },
    "allenai/Olmo-3-7B-Instruct-SFT": {
        "allenai/Olmo-3-7B-Instruct-SFT",
        "Olmo-3-7B-Instruct-SFT",
        "olmo_3_7b_instruct_sft",
        "olmo-3-7b-instruct-sft",
        "ol37is",
        "experiment_olmo_3_7b_instruct_sft",
    },
}


def _normalize_token(value: str) -> str:
    """Collapse a model alias to an alphanumeric key for matching."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


_NORMALIZED_MODEL_ALIAS_TO_CANONICAL = {}
for _canonical_model, _aliases in _MODEL_ALIAS_GROUPS.items():
    for _alias in _aliases | {_canonical_model.split("/")[-1]}:
        _NORMALIZED_MODEL_ALIAS_TO_CANONICAL[_normalize_token(_alias)] = _canonical_model


@dataclass(frozen=True)
class TrainingExperimentMatch:
    """Minimal training-config metadata needed by ``sol_eval``."""

    path: Path
    run_name: str
    model_path: str
    remote_sync_path: str


def canonicalize_model_id(model_name: str | None) -> str | None:
    """Map common model aliases to canonical HF model IDs."""
    if model_name is None:
        return None
    stripped = model_name.strip()
    if not stripped:
        return None
    canonical = _NORMALIZED_MODEL_ALIAS_TO_CANONICAL.get(_normalize_token(stripped))
    return canonical or stripped


def detect_model_from_remote_path(remote_sync_path: str) -> str | None:
    """Infer a canonical model ID from a remote checkpoint path."""
    if not remote_sync_path:
        return None
    parts = [part for part in remote_sync_path.rstrip("/").split("/") if part]
    for part in parts:
        model_path = REMOTE_PATH_TO_MODEL.get(part)
        if model_path:
            return model_path
    return None


def _default_config_root() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def resolve_training_config_path(
    config_path: str | Path,
    config_root: str | Path | None = None,
) -> Path:
    """Resolve a training config path from cwd or the default config root."""
    root = Path(config_root) if config_root is not None else _default_config_root()
    requested = Path(config_path).expanduser()

    candidates = [requested] if requested.is_absolute() else [Path.cwd() / requested, root / requested]
    attempted: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.expanduser()
        attempted.append(normalized)
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.is_file():
            return normalized.resolve()

    attempted_str = "\n".join(f"  - {path}" for path in attempted)
    raise ValueError(
        f"Training config not found for {config_path!r}. Tried:\n{attempted_str}"
    )


def _iter_training_config_paths(config_root: Path) -> list[Path]:
    patterns = (
        "experiment/*.yaml",
        "experiment/**/*.yaml",
        "experiment_*/*.yaml",
    )
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        for path in sorted(config_root.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _extract_nested(config_data: dict, *keys: str) -> str | None:
    current: object = config_data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, str) else None


def _resolve_remote_sync_template(remote_sync_path: str | None, run_name: str) -> str | None:
    if not remote_sync_path:
        return None
    return remote_sync_path.replace("${wandb.group_name}", run_name)


def _load_training_config(path: Path) -> TrainingExperimentMatch | None:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        return None

    run_name = _extract_nested(raw, "wandb", "group_name") or path.stem
    remote_sync_path = _resolve_remote_sync_template(
        _extract_nested(raw, "training", "remote_sync_path"),
        run_name,
    )
    if not remote_sync_path:
        return None

    model_path = (
        _extract_nested(raw, "solver", "model", "path")
        or _extract_nested(raw, "generator", "model", "path")
        or detect_model_from_remote_path(remote_sync_path)
        or canonicalize_model_id(path.parent.name)
    )
    if not model_path:
        return None

    return TrainingExperimentMatch(
        path=path,
        run_name=run_name,
        model_path=canonicalize_model_id(model_path) or model_path,
        remote_sync_path=remote_sync_path,
    )


def load_training_experiment(
    config_path: str | Path,
    config_root: str | Path | None = None,
) -> TrainingExperimentMatch:
    """Load a specific local training YAML by path."""
    resolved_path = resolve_training_config_path(config_path, config_root=config_root)
    match = _load_training_config(resolved_path)
    if match is None:
        raise ValueError(
            f"Training config at {resolved_path} is missing required metadata "
            "(need training.remote_sync_path and a model path or recognizable remote)."
        )
    return match


def _format_candidates(candidates: list[TrainingExperimentMatch]) -> str:
    return "\n".join(
        f"  - {candidate.path} (model={candidate.model_path})"
        for candidate in candidates
    )


def discover_training_experiment(
    run_name: str,
    model_name: str | None = None,
    config_root: str | Path | None = None,
) -> TrainingExperimentMatch:
    """Find the local training YAML matching a run name and optional model."""
    root = Path(config_root) if config_root is not None else _default_config_root()
    requested_model = canonicalize_model_id(model_name)

    candidates = [
        match
        for path in _iter_training_config_paths(root)
        if (match := _load_training_config(path)) is not None and match.run_name == run_name
    ]

    if not candidates:
        raise ValueError(
            f"No training config matched run_name={run_name!r} under {root}."
        )

    if requested_model is not None:
        candidates = [
            candidate for candidate in candidates if candidate.model_path == requested_model
        ]
        if not candidates:
            raise ValueError(
                "No training config matched "
                f"run_name={run_name!r} and model={requested_model!r}.\n"
                "Run-name matches were:\n"
                f"{_format_candidates([match for path in _iter_training_config_paths(root) if (match := _load_training_config(path)) is not None and match.run_name == run_name])}"
            )

    if len(candidates) > 1:
        raise ValueError(
            "Multiple training configs matched the requested evaluation run.\n"
            f"run_name={run_name!r}"
            + (
                f", model={requested_model!r}\n"
                if requested_model is not None
                else "\n"
            )
            + "Candidates:\n"
            + _format_candidates(candidates)
        )

    return candidates[0]
