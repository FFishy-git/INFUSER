"""Utility helpers for generator reward composition and filtering."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Mapping

import numpy as np

LOGGER = logging.getLogger(__name__)

ALLOWED_GENERATOR_REWARD_COMPONENTS = (
    "influence_rewards",
    "spice_rewards",
    "invalid_rewards",
)
ALLOWED_GENERATOR_REWARD_COMPONENT_SET = set(ALLOWED_GENERATOR_REWARD_COMPONENTS)


def _validate_generator_reward_component(component: Any, path: str) -> str:
    """Validate a single reward-component name."""
    comp = str(component)
    if comp not in ALLOWED_GENERATOR_REWARD_COMPONENT_SET:
        raise ValueError(
            f"Invalid component '{comp}' at {path}. "
            f"Allowed values: {sorted(ALLOWED_GENERATOR_REWARD_COMPONENT_SET)}"
        )
    return comp


def _coerce_reward_weight(weight: Any, path: str) -> float:
    """Convert a configured reward weight to a finite float."""
    try:
        resolved = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid weight at {path}: {weight!r}") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"Invalid weight at {path}: {weight!r} is not finite.")
    return resolved


def validate_generator_reward_components(components: list[str]) -> list[str]:
    """Validate and de-duplicate configured generator reward components."""
    normalized: list[str] = []
    seen: set[str] = set()
    for component in components:
        comp = _validate_generator_reward_component(
            component,
            "training.generator_reward_components",
        )
        if comp not in seen:
            normalized.append(comp)
            seen.add(comp)
    if not normalized:
        raise ValueError(
            "training.generator_reward_components must include at least one entry."
        )
    return normalized


def validate_generator_reward_structure(
    structure: list[Any],
) -> list[dict[str, Any]]:
    """Validate and normalize ``training.generator_reward_structure``.

    Supported forms:
    - explicit group objects:
      ``[{group_weight: 1.0, terms: [{name: "influence_rewards", weight: 1.0}]}]``
    - shorthand singleton groups:
      ``["influence_rewards", "invalid_rewards"]``
    - shorthand grouped terms:
      ``["influence_rewards", ["spice_rewards", "invalid_rewards"]]``
    """
    normalized_structure: list[dict[str, Any]] = []
    seen_components: set[str] = set()

    for group_idx, raw_group in enumerate(structure):
        group_path = f"training.generator_reward_structure[{group_idx}]"
        raw_terms: Any
        if isinstance(raw_group, Mapping):
            group_weight = _coerce_reward_weight(
                raw_group.get("group_weight", 1.0),
                f"{group_path}.group_weight",
            )
            raw_terms = raw_group.get("terms", None)
            if raw_terms is None:
                raise ValueError(f"{group_path}.terms must be set.")
        else:
            group_weight = 1.0
            raw_terms = raw_group

        if isinstance(raw_terms, str):
            term_iter = [raw_terms]
        else:
            try:
                term_iter = list(raw_terms)
            except TypeError as exc:
                raise ValueError(
                    f"{group_path} must be a group object, a component name, or a list of terms."
                ) from exc

        if not term_iter:
            raise ValueError(f"{group_path} is empty — each group must have at least one term.")

        normalized_terms: list[dict[str, Any]] = []
        for term_idx, raw_term in enumerate(term_iter):
            term_path = f"{group_path}.terms[{term_idx}]"
            if isinstance(raw_term, Mapping):
                name = _validate_generator_reward_component(
                    raw_term.get("name", None),
                    f"{term_path}.name",
                )
                weight = _coerce_reward_weight(
                    raw_term.get("weight", 1.0),
                    f"{term_path}.weight",
                )
            else:
                name = _validate_generator_reward_component(raw_term, term_path)
                weight = 1.0

            if name in seen_components:
                raise ValueError(
                    f"Component '{name}' appears in multiple reward groups — "
                    "each component may belong to at most one group."
                )
            seen_components.add(name)
            normalized_terms.append({"name": name, "weight": weight})

        group_dict: dict[str, Any] = {
            "group_weight": group_weight,
            "terms": normalized_terms,
        }
        if isinstance(raw_group, Mapping) and "normalization_mode" in raw_group:
            group_dict["normalization_mode"] = str(raw_group["normalization_mode"])
        normalized_structure.append(group_dict)

    if not normalized_structure:
        raise ValueError(
            "training.generator_reward_structure must contain at least one group "
            "when it is provided."
        )
    return normalized_structure


def flatten_generator_reward_structure_components(
    structure: list[dict[str, Any]],
) -> list[str]:
    """Return unique component names in the order they appear in the structure."""
    return [str(term["name"]) for group in structure for term in group["terms"]]


def reward_structure_component_weights(
    structure: list[dict[str, Any]],
) -> dict[str, float]:
    """Return per-component pre-normalization term weights from a reward structure."""
    return {
        str(term["name"]): float(term["weight"])
        for group in structure
        for term in group["terms"]
    }


def _resolve_legacy_generator_reward_components(training_cfg: Any) -> list[str]:
    """Resolve the legacy flat component list."""
    configured = training_cfg.get("generator_reward_components", None)
    if configured is None:
        raise ValueError(
            "training.generator_reward_components must be set when "
            "training.generator_reward_structure is empty. "
            "Example: ['influence_rewards', 'invalid_rewards']"
        )

    if isinstance(configured, str):
        components = [configured]
    else:
        components = [str(x) for x in configured]
    return validate_generator_reward_components(components)


def validate_generator_reward_groups(
    groups: list[list[str]],
    weights: list[float],
    selected_components: list[str],
) -> tuple[list[list[str]], list[float]]:
    """Validate legacy reward groups and weights for decoupled advantage."""
    if not groups:
        flat_groups = [[component] for component in selected_components]
        if weights and len(weights) != len(flat_groups):
            raise ValueError(
                f"generator_reward_group_weights length ({len(weights)}) does not match "
                f"number of inferred flat groups ({len(flat_groups)})."
            )
        flat_weights = [float(w) for w in weights] if weights else [1.0] * len(flat_groups)
        return flat_groups, flat_weights

    seen_components: set[str] = set()
    normalized_groups: list[list[str]] = []
    for group_idx, group in enumerate(groups):
        group_path = f"training.generator_reward_groups[{group_idx}]"
        if not group:
            raise ValueError(f"{group_path} is empty — each group must have at least one component.")
        normalized_group: list[str] = []
        for member in group:
            component = _validate_generator_reward_component(member, group_path)
            if component not in selected_components:
                raise ValueError(
                    f"Component '{component}' in {group_path} is not in "
                    f"training.generator_reward_components ({selected_components})."
                )
            if component in seen_components:
                raise ValueError(
                    f"Component '{component}' appears in multiple reward groups — "
                    "each component may belong to at most one group."
                )
            seen_components.add(component)
            normalized_group.append(component)
        normalized_groups.append(normalized_group)

    ungrouped = [component for component in selected_components if component not in seen_components]
    if ungrouped:
        LOGGER.warning(
            "Components %s are selected but not in any legacy reward group — "
            "they will not participate in decoupled advantage computation.",
            ungrouped,
        )

    if weights:
        if len(weights) != len(normalized_groups):
            raise ValueError(
                f"generator_reward_group_weights length ({len(weights)}) must match "
                f"generator_reward_groups length ({len(normalized_groups)})."
            )
        normalized_weights = [
            _coerce_reward_weight(
                weight,
                f"training.generator_reward_group_weights[{group_idx}]",
            )
            for group_idx, weight in enumerate(weights)
        ]
    else:
        normalized_weights = [1.0] * len(normalized_groups)

    return normalized_groups, normalized_weights


def _resolve_legacy_generator_reward_groups(
    training_cfg: Any,
    selected_components: list[str],
) -> tuple[list[list[str]], list[float]]:
    """Resolve legacy reward groups and weights from config."""
    raw_groups = training_cfg.get("generator_reward_groups", [])
    raw_weights = training_cfg.get("generator_reward_group_weights", [])

    if raw_groups is None:
        groups: list[list[str]] = []
    else:
        groups = [[str(member) for member in group] for group in raw_groups]

    if raw_weights is None:
        weights: list[float] = []
    else:
        weights = [float(weight) for weight in raw_weights]

    return validate_generator_reward_groups(groups, weights, selected_components)


def resolve_generator_reward_components(training_cfg: Any) -> list[str]:
    """Resolve selected reward components from training config."""
    raw_structure = training_cfg.get("generator_reward_structure", None)
    if raw_structure:
        return flatten_generator_reward_structure_components(
            validate_generator_reward_structure(list(raw_structure))
        )
    return _resolve_legacy_generator_reward_components(training_cfg)


def resolve_generator_reward_structure(training_cfg: Any) -> list[dict[str, Any]]:
    """Resolve the canonical reward-structure representation.

    When ``training.generator_reward_structure`` is set, it becomes the source
    of truth and implies decoupled advantage. Otherwise legacy
    ``generator_reward_components`` / ``generator_reward_groups`` /
    ``generator_reward_group_weights`` are translated into the same structure.
    """
    raw_structure = training_cfg.get("generator_reward_structure", None)
    if raw_structure:
        return validate_generator_reward_structure(list(raw_structure))

    selected_components = _resolve_legacy_generator_reward_components(training_cfg)
    reward_combination_mode = str(
        training_cfg.get("generator_reward_combination_mode", "sum_scores")
    )
    if reward_combination_mode == "sum_scores":
        return [
            {
                "group_weight": 1.0,
                "terms": [
                    {"name": component, "weight": 1.0}
                    for component in selected_components
                ],
            }
        ]

    legacy_groups, legacy_weights = _resolve_legacy_generator_reward_groups(
        training_cfg,
        selected_components,
    )
    return [
        {
            "group_weight": float(group_weight),
            "terms": [{"name": component, "weight": 1.0} for component in group],
        }
        for group, group_weight in zip(legacy_groups, legacy_weights)
    ]


def resolve_generator_reward_component_weights(
    training_cfg: Any,
) -> dict[str, float]:
    """Resolve pre-normalization per-component weights from config."""
    return reward_structure_component_weights(
        resolve_generator_reward_structure(training_cfg)
    )


def resolve_generator_reward_combination_mode(training_cfg: Any) -> str:
    """Resolve how selected reward components are combined for generator PPO."""
    raw_structure = training_cfg.get("generator_reward_structure", None)
    if raw_structure:
        validate_generator_reward_structure(list(raw_structure))
        return "decoupled"
    mode = str(training_cfg.get("generator_reward_combination_mode", "sum_scores"))
    allowed = {"sum_scores", "decoupled"}
    if mode not in allowed:
        raise ValueError(
            "Invalid training.generator_reward_combination_mode: "
            f"{mode}. Allowed values: {sorted(allowed)}"
        )
    return mode


def coerce_reward_dict(rewards: Any) -> dict[str, float]:
    """Best-effort coercion of ``{question_id: score}`` mappings."""
    if not isinstance(rewards, Mapping):
        return {}
    out: dict[str, float] = {}
    for k, v in rewards.items():
        if v is None:
            continue
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def normalize_reward_dict_for_valid_questions(
    rewards: dict[str, float],
    valid_question_ids: set[str],
) -> dict[str, float]:
    """Return a dense dict over all valid qids; missing entries become 0.0."""
    dense = {qid: 0.0 for qid in valid_question_ids}
    for qid, score in rewards.items():
        qid_s = str(qid)
        if qid_s in dense:
            dense[qid_s] = float(score)
    return dense


def build_stage4_reward_payload(
    valid_question_ids: set[str],
    influence_rewards: dict[str, float] | None,
    spice_rewards: dict[str, float] | None,
    selected_components: list[str],
    reward_structure: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build stage-4 reward payload with component maps + combined rewards."""
    selected_components = validate_generator_reward_components(list(selected_components))
    resolved_reward_structure = None
    if reward_structure:
        resolved_reward_structure = validate_generator_reward_structure(
            list(reward_structure)
        )
        structured_components = flatten_generator_reward_structure_components(
            resolved_reward_structure
        )
        if set(structured_components) != set(selected_components):
            raise ValueError(
                "selected_components and reward_structure disagree. "
                f"selected_components={selected_components}, "
                f"reward_structure_components={structured_components}"
            )

    influence_dense = normalize_reward_dict_for_valid_questions(
        influence_rewards or {},
        valid_question_ids,
    )
    spice_dense = normalize_reward_dict_for_valid_questions(
        spice_rewards or {},
        valid_question_ids,
    )
    component_weights = (
        reward_structure_component_weights(resolved_reward_structure)
        if resolved_reward_structure is not None
        else {component: 1.0 for component in selected_components}
    )

    combined_rewards = {qid: 0.0 for qid in valid_question_ids}
    if "influence_rewards" in selected_components:
        for qid, score in influence_dense.items():
            combined_rewards[qid] += component_weights.get("influence_rewards", 1.0) * float(score)
    if "spice_rewards" in selected_components:
        for qid, score in spice_dense.items():
            combined_rewards[qid] += component_weights.get("spice_rewards", 1.0) * float(score)

    return {
        "version": 2,
        "selected_components": selected_components,
        "known_valid_question_ids": sorted(valid_question_ids),
        "reward_structure": resolved_reward_structure,
        "component_weights": component_weights,
        "reward_components": {
            "influence_rewards": influence_dense,
            "spice_rewards": spice_dense,
        },
        # Combined over valid questions only (invalid component is row-level).
        "combined_rewards": combined_rewards,
    }


def normalize_reward_payload(
    rewards: Any,
    valid_question_ids: set[str],
    training_cfg: Any,
) -> dict[str, Any]:
    """Normalize legacy/new reward formats for ``prepare_gen_update_batch()``."""
    selected_default = resolve_generator_reward_components(training_cfg)
    component_weight_defaults = resolve_generator_reward_component_weights(training_cfg)
    selected_components = selected_default
    influence_raw: dict[str, float] = {}
    spice_raw: dict[str, float] = {}
    known_valid_qids: set[str] = set(valid_question_ids)
    reward_structure: list[dict[str, Any]] | None = None
    component_weights = {
        component: float(component_weight_defaults.get(component, 1.0))
        for component in selected_components
    }

    if isinstance(rewards, Mapping) and (
        "reward_components" in rewards
        or "selected_components" in rewards
        or "combined_rewards" in rewards
        or "reward_structure" in rewards
    ):
        reward_structure_cfg = rewards.get("reward_structure", None)
        if reward_structure_cfg:
            reward_structure = validate_generator_reward_structure(
                list(reward_structure_cfg)
            )
            selected_components = flatten_generator_reward_structure_components(
                reward_structure
            )
            component_weights = reward_structure_component_weights(reward_structure)
        else:
            selected_cfg = rewards.get("selected_components", selected_default)
            if isinstance(selected_cfg, str):
                selected_components = validate_generator_reward_components([selected_cfg])
            elif selected_cfg is None:
                selected_components = selected_default
            else:
                selected_components = validate_generator_reward_components(
                    [str(x) for x in selected_cfg]
                )
            payload_component_weights = rewards.get("component_weights", {})
            if isinstance(payload_component_weights, Mapping):
                component_weights = {
                    component: _coerce_reward_weight(
                        payload_component_weights.get(
                            component,
                            component_weight_defaults.get(component, 1.0),
                        ),
                        f"rewards.component_weights[{component!r}]",
                    )
                    for component in selected_components
                }
            else:
                component_weights = {
                    component: float(component_weight_defaults.get(component, 1.0))
                    for component in selected_components
                }

        reward_components = rewards.get("reward_components", {})
        if isinstance(reward_components, Mapping):
            influence_raw = coerce_reward_dict(
                reward_components.get("influence_rewards", {})
            )
            spice_raw = coerce_reward_dict(
                reward_components.get("spice_rewards", {})
            )

        known_from_payload = rewards.get("known_valid_question_ids", None)
        if known_from_payload is not None:
            known_valid_qids = {
                str(qid) for qid in known_from_payload if qid is not None
            }
        else:
            inferred = set(influence_raw.keys()) | set(spice_raw.keys())
            if inferred:
                known_valid_qids = inferred

    elif isinstance(rewards, Mapping):
        legacy = coerce_reward_dict(rewards)
        known_valid_qids = set(legacy.keys())
        # Legacy plain dict: route to the first selected component.
        if "influence_rewards" in selected_components:
            LOGGER.warning(
                "Legacy reward dict detected — interpreting as influence_rewards "
                "(selected_components=%s).",
                selected_components,
            )
            influence_raw = legacy
        elif "spice_rewards" in selected_components:
            LOGGER.warning(
                "Legacy reward dict detected — interpreting as spice_rewards "
                "(selected_components=%s).",
                selected_components,
            )
            spice_raw = legacy
        else:
            LOGGER.warning(
                "Legacy reward dict detected — defaulting to influence_rewards "
                "(selected_components=%s has neither influence nor spice).",
                selected_components,
            )
            influence_raw = legacy
    else:
        raise ValueError(
            "Invalid rewards payload. Expected dict-like rewards mapping or "
            "stage-4 payload."
        )

    if not known_valid_qids:
        known_valid_qids = set(valid_question_ids)

    return {
        "selected_components": selected_components,
        "known_valid_question_ids": known_valid_qids,
        "reward_structure": reward_structure,
        "component_weights": component_weights,
        "reward_components": {
            "influence_rewards": normalize_reward_dict_for_valid_questions(
                influence_raw, valid_question_ids
            ),
            "spice_rewards": normalize_reward_dict_for_valid_questions(
                spice_raw, valid_question_ids
            ),
        },
    }


def extract_influence_rewards_for_solver_filter(rewards: Any) -> dict[str, float]:
    """Extract per-question influence rewards for solver-side filtering.

    New-style stage-4 payloads are read-only here: solver filtering uses only
    ``reward_components.influence_rewards``. Legacy plain ``{question_id:
    score}`` payloads are treated as influence rewards.
    """
    if not isinstance(rewards, Mapping):
        return {}

    if "reward_components" in rewards:
        reward_components = rewards.get("reward_components", {})
        if not isinstance(reward_components, Mapping):
            return {}
        return coerce_reward_dict(reward_components.get("influence_rewards", {}))

    return coerce_reward_dict(rewards)


def filter_influence_rewards_by_group_std(
    influence_rewards: dict[str, float],
    parsed_qids: np.ndarray,
    doc_ids: np.ndarray,
    known_valid_qids: set[str],
    influence_cfg: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    """Filter influence rewards by doc-group std and return metrics."""
    quantification_mode = influence_cfg.get("quantification_mode", None)
    mode_to_id = {
        None: 0.0,
        "1bit": 1.0,
        "2bit": 2.0,
        "group_std_top_gamma": 3.0,
        "group_std_fixed_threshold": 4.0,
    }
    if quantification_mode not in mode_to_id:
        raise ValueError(
            "Invalid influence.quantification_mode: "
            f"{quantification_mode}. Must be None, '1bit', '2bit', or "
            "'group_std_top_gamma', 'group_std_fixed_threshold'."
        )

    gamma = float(influence_cfg.get("group_std_gamma", 0.2))
    if quantification_mode == "group_std_top_gamma" and not (0.0 < gamma <= 1.0):
        raise ValueError(
            "Invalid influence.group_std_gamma: "
            f"{gamma}. Must be in the range (0, 1]."
        )
    
    # prepare tau_max if needed; only used for group_std_top_gamma mode, ignored otherwise
    tau_max_cfg = influence_cfg.get("group_std_tau_max", None)
    if quantification_mode == "group_std_top_gamma" and tau_max_cfg is not None:
        tau_max = float(tau_max_cfg)
        if not math.isfinite(tau_max) or tau_max < 0.0:
            raise ValueError(
                "Invalid influence.group_std_tau_max: "
                f"{tau_max_cfg}. Must be a finite float >= 0."
            )
    else:
        tau_max = None

    # prepare fixed tau if needed; only used for group_std_fixed_threshold mode, ignored otherwise
    fixed_tau_cfg = influence_cfg.get("group_std_tau", None)
    if quantification_mode == "group_std_fixed_threshold":
        if fixed_tau_cfg is None:
            raise ValueError(
                "influence.group_std_tau must be set when "
                "quantification_mode='group_std_fixed_threshold'."
            )
        fixed_tau = float(fixed_tau_cfg)
        if not math.isfinite(fixed_tau):
            raise ValueError(
                "Invalid influence.group_std_tau: "
                f"{fixed_tau_cfg}. Must be a finite float."
            )
    else:
        fixed_tau = 0.0

    # Group influence rewards by doc_id.
    doc_influence_rewards: dict[str, list[float]] = defaultdict(list)
    qid_to_doc: dict[str, str] = {}
    for raw_qid, raw_doc_id in zip(parsed_qids, doc_ids):
        doc_id = str(raw_doc_id)
        qid = None if raw_qid is None else str(raw_qid)
        if qid is not None and qid in known_valid_qids:
            influence_r = float(influence_rewards.get(qid, 0.0))
            qid_to_doc[qid] = doc_id
        else:
            influence_r = 0.0
        doc_influence_rewards[doc_id].append(influence_r)

    # Compute stddev of influence rewards within each doc group, and collect stats for quantification.
    n_total_docs = len(doc_influence_rewards)
    doc_influence_stds: dict[str, float] = {}
    centered_abs: list[float] = []
    for doc_id, rlist in doc_influence_rewards.items():
        arr = np.asarray(rlist, dtype=np.float32)
        if arr.size == 0:
            doc_influence_stds[doc_id] = 0.0
            continue
        mean_r = float(arr.mean())
        centered_abs.extend(np.abs(arr - mean_r).tolist())
        doc_influence_stds[doc_id] = float(arr.std())

    metrics: dict[str, float] = {
        "gen_quant/gamma": gamma,
        "gen_quant/tau_quantile": (
            1.0 - gamma if quantification_mode == "group_std_top_gamma" else -1.0
        ),
        "gen_quant/tau_max": float(tau_max) if tau_max is not None else -1.0,
        "gen_quant/mode_id": mode_to_id[quantification_mode],
    }

    std_values = list(doc_influence_stds.values())
    if std_values:
        std_arr = np.asarray(std_values, dtype=np.float32)
        metrics.update({
            "gen_quant/group_std/mean": float(std_arr.mean()),
            "gen_quant/group_std/std": float(std_arr.std()),
            "gen_quant/group_std/min": float(std_arr.min()),
            "gen_quant/group_std/max": float(std_arr.max()),
            "gen_quant/group_std/p50": float(np.percentile(std_arr, 50)),
            "gen_quant/group_std/p90": float(np.percentile(std_arr, 90)),
            "gen_quant/group_std/p95": float(np.percentile(std_arr, 95)),
        })
        # For fixed threshold mode, tau is just the configured fixed value.
        if quantification_mode == "group_std_fixed_threshold":
            tau_raw = fixed_tau
            tau = fixed_tau
        # For top-gamma mode, tau is the (1-gamma) quantile of the std distribution, optionally clamped by tau_max if configured.
        elif quantification_mode == "group_std_top_gamma":
            tau_raw = float(np.quantile(std_arr, 1.0 - gamma))
            tau = min(tau_raw, tau_max) if tau_max is not None else tau_raw
        # For other modes, tau is not used. Set tau_raw to the (1-gamma) quantile for logging/analysis purposes only
        else:
            tau_raw = float(np.quantile(std_arr, 1.0 - gamma))
            tau = tau_raw
    else: # no doc groups with valid qids, or all have empty reward lists (should be rare/edge case); set all std metrics to 0 and tau to 0 or fixed value if in fixed threshold mode
        metrics.update({
            "gen_quant/group_std/mean": 0.0,
            "gen_quant/group_std/std": 0.0,
            "gen_quant/group_std/min": 0.0,
            "gen_quant/group_std/max": 0.0,
            "gen_quant/group_std/p50": 0.0,
            "gen_quant/group_std/p90": 0.0,
            "gen_quant/group_std/p95": 0.0,
        })
        if quantification_mode == "group_std_fixed_threshold":
            tau_raw = fixed_tau
            tau = fixed_tau
        elif quantification_mode == "group_std_top_gamma":
            tau_raw = 0.0
            tau = min(tau_raw, tau_max) if tau_max is not None else tau_raw
        else:
            tau_raw = 0.0
            tau = 0.0

    metrics["gen_quant/group_std_basis_influence"] = 1.0
    metrics["gen_quant/tau_raw"] = tau_raw
    metrics["gen_quant/tau"] = tau
    metrics["gen_quant/tau_was_clamped"] = (
        1.0 if quantification_mode == "group_std_top_gamma" and tau < tau_raw else 0.0
    )
    if centered_abs:
        centered_abs_arr = np.asarray(centered_abs, dtype=np.float32)
        metrics["gen_quant/centered_abs/mean"] = float(centered_abs_arr.mean())
        metrics["gen_quant/centered_abs/p90"] = float(np.percentile(centered_abs_arr, 90))
    else:
        metrics["gen_quant/centered_abs/mean"] = 0.0
        metrics["gen_quant/centered_abs/p90"] = 0.0

    influence_masked_docs: set[str] = set()
    if quantification_mode in ("group_std_top_gamma", "group_std_fixed_threshold"):
        influence_masked_docs = {
            doc_id for doc_id, std in doc_influence_stds.items() if std <= tau
        }
        LOGGER.info(
            "filter_influence_rewards_by_group_std: tau-mask=%d/%d doc groups "
            "(mode=%s, gamma=%.4f, tau=%.6f)",
            len(influence_masked_docs),
            n_total_docs,
            quantification_mode,
            gamma,
            tau,
        )

    metrics["gen_quant/influence_mask/groups_masked"] = float(len(influence_masked_docs))
    metrics["gen_quant/influence_mask/groups_kept"] = float(
        n_total_docs - len(influence_masked_docs)
    )
    metrics["gen_quant/influence_mask/groups_keep_rate"] = (
        float(n_total_docs - len(influence_masked_docs)) / float(n_total_docs)
        if n_total_docs > 0
        else 0.0
    )

    filtered = dict(influence_rewards)
    if influence_masked_docs:
        for qid, doc_id in qid_to_doc.items():
            if doc_id in influence_masked_docs:
                filtered[qid] = 0.0

    return filtered, metrics
