import os
import sys

from omegaconf import OmegaConf

# Add project root to path for local imports (verl, verl_inf_evolve).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.utils.generator_reward_utils import (
    build_stage4_reward_payload,
    extract_influence_rewards_for_solver_filter,
    resolve_generator_reward_combination_mode,
    resolve_generator_reward_component_weights,
    resolve_generator_reward_components,
    resolve_generator_reward_structure,
)


def test_structured_reward_config_overrides_legacy_fields():
    cfg = OmegaConf.create(
        {
            "generator_reward_structure": [
                {
                    "group_weight": 1.5,
                    "terms": [
                        {"name": "influence_rewards", "weight": 2.0},
                    ],
                },
                {
                    "terms": [
                        {"name": "spice_rewards", "weight": 0.25},
                        {"name": "invalid_rewards", "weight": 1.0},
                    ],
                },
            ],
            "generator_reward_components": ["influence_rewards"],
            "generator_reward_combination_mode": "sum_scores",
        }
    )

    structure = resolve_generator_reward_structure(cfg)

    assert resolve_generator_reward_combination_mode(cfg) == "decoupled"
    assert resolve_generator_reward_components(cfg) == [
        "influence_rewards",
        "spice_rewards",
        "invalid_rewards",
    ]
    assert structure == [
        {
            "group_weight": 1.5,
            "terms": [{"name": "influence_rewards", "weight": 2.0}],
        },
        {
            "group_weight": 1.0,
            "terms": [
                {"name": "spice_rewards", "weight": 0.25},
                {"name": "invalid_rewards", "weight": 1.0},
            ],
        },
    ]
    assert resolve_generator_reward_component_weights(cfg) == {
        "influence_rewards": 2.0,
        "spice_rewards": 0.25,
        "invalid_rewards": 1.0,
    }


def test_solver_filter_reads_influence_rewards_only():
    cfg = OmegaConf.create(
        {
            "generator_reward_structure": [
                {
                    "group_weight": 1.0,
                    "terms": [
                        {"name": "influence_rewards", "weight": 2.0},
                    ],
                },
                {
                    "group_weight": 1.0,
                    "terms": [
                        {"name": "spice_rewards", "weight": 0.5},
                        {"name": "invalid_rewards", "weight": 3.0},
                    ],
                },
            ]
        }
    )
    payload = build_stage4_reward_payload(
        valid_question_ids={"q1", "q2"},
        influence_rewards={"q1": 1.0, "q2": 3.0},
        spice_rewards={"q1": 8.0, "q2": 10.0},
        selected_components=resolve_generator_reward_components(cfg),
        reward_structure=resolve_generator_reward_structure(cfg),
    )

    influence = extract_influence_rewards_for_solver_filter(payload)

    assert influence == {
        "q1": 1.0,
        "q2": 3.0,
    }


def test_solver_filter_plain_dict_legacy_fallback():
    assert extract_influence_rewards_for_solver_filter({"q1": 2.0, "q2": -1.0}) == {
        "q1": 2.0,
        "q2": -1.0,
    }


def test_legacy_decoupled_config_maps_to_canonical_structure():
    cfg = OmegaConf.create(
        {
            "generator_reward_components": [
                "influence_rewards",
                "spice_rewards",
                "invalid_rewards",
            ],
            "generator_reward_combination_mode": "decoupled",
            "generator_reward_groups": [
                ["influence_rewards"],
                ["spice_rewards", "invalid_rewards"],
            ],
            "generator_reward_group_weights": [1.0, 3.0],
        }
    )

    structure = resolve_generator_reward_structure(cfg)

    assert structure == [
        {
            "group_weight": 1.0,
            "terms": [{"name": "influence_rewards", "weight": 1.0}],
        },
        {
            "group_weight": 3.0,
            "terms": [
                {"name": "spice_rewards", "weight": 1.0},
                {"name": "invalid_rewards", "weight": 1.0},
            ],
        },
    ]
