from __future__ import annotations

import json

from unittest.mock import patch

from verl_inf_evolve.storage import hf_remote_resolver
from verl_inf_evolve.storage.hf_remote_resolver import (
    load_token_pool_with_warnings,
    resolve_hf_remote_from_pool,
)


def test_returns_none_when_namespace_is_explicit(monkeypatch):
    monkeypatch.setenv("HF_TOKEN_A", "tok-a")
    remote_cfg = {
        "hf_namespace_placeholder": "__namespace__",
        "hf_token_pool": [
            {"namespace": "user_a", "token_env_var": "HF_TOKEN_A"},
        ],
    }
    assert (
        resolve_hf_remote_from_pool(
            "hf://datasets/user_a/SER/V4_qwen3_4b_base/run_a",
            remote_cfg,
        )
        is None
    )


@patch("verl_inf_evolve.storage.hf_remote_resolver._namespace_used_storage")
@patch("verl_inf_evolve.storage.hf_remote_resolver._repo_used_storage")
@patch("verl_inf_evolve.storage.hf_remote_resolver._prefix_exists")
def test_reuses_existing_prefix_first_match(
    mock_prefix_exists,
    mock_repo_used_storage,
    mock_namespace_used_storage,
    monkeypatch,
):
    monkeypatch.setenv("HF_TOKEN_A", "tok-a")
    monkeypatch.setenv("HF_TOKEN_B", "tok-b")
    remote_cfg = {
        "hf_namespace_placeholder": "__namespace__",
        "hf_token_pool": [
            {"namespace": "user_a", "token_env_var": "HF_TOKEN_A"},
            {"namespace": "user_b", "token_env_var": "HF_TOKEN_B"},
        ],
    }

    mock_prefix_exists.side_effect = [True, True]
    mock_repo_used_storage.side_effect = [10, 20]
    mock_namespace_used_storage.side_effect = [100, 200]

    resolved = resolve_hf_remote_from_pool(
        "hf://datasets/__namespace__/SER/V4_qwen3_4b_base/run_a",
        remote_cfg,
    )

    assert resolved is not None
    assert resolved.uri == "hf://datasets/user_a/SER/V4_qwen3_4b_base/run_a"
    assert resolved.namespace == "user_a"
    assert resolved.selection_reason == "existing_prefix_reuse"
    assert resolved.warning is not None


@patch("verl_inf_evolve.storage.hf_remote_resolver._namespace_used_storage")
@patch("verl_inf_evolve.storage.hf_remote_resolver._repo_used_storage")
@patch("verl_inf_evolve.storage.hf_remote_resolver._prefix_exists")
def test_picks_largest_estimated_availability_when_no_match(
    mock_prefix_exists,
    mock_repo_used_storage,
    mock_namespace_used_storage,
    monkeypatch,
):
    monkeypatch.setenv("HF_TOKEN_A", "tok-a")
    monkeypatch.setenv("HF_TOKEN_B", "tok-b")
    remote_cfg = {
        "hf_namespace_placeholder": "__namespace__",
        "hf_token_pool": [
            {
                "namespace": "user_a",
                "token_env_var": "HF_TOKEN_A",
                "capacity_bytes": 100,
            },
            {
                "namespace": "user_b",
                "token_env_var": "HF_TOKEN_B",
                "capacity_bytes": 200,
            },
        ],
    }

    mock_prefix_exists.side_effect = [False, False]
    mock_repo_used_storage.side_effect = [0, 0]
    mock_namespace_used_storage.side_effect = [60, 80]

    resolved = resolve_hf_remote_from_pool(
        "hf://datasets/__namespace__/SER/V4_qwen3_4b_base/run_b",
        remote_cfg,
    )

    assert resolved is not None
    assert resolved.uri == "hf://datasets/user_b/SER/V4_qwen3_4b_base/run_b"
    assert resolved.namespace == "user_b"
    assert resolved.selection_reason == "capacity_minus_namespace_usage"
    assert resolved.warning is None


@patch("verl_inf_evolve.storage.hf_remote_resolver._namespace_used_storage")
@patch("verl_inf_evolve.storage.hf_remote_resolver._repo_used_storage")
@patch("verl_inf_evolve.storage.hf_remote_resolver._prefix_exists")
def test_loads_pool_from_file(
    mock_prefix_exists,
    mock_repo_used_storage,
    mock_namespace_used_storage,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HF_TOKEN_A", "tok-a")
    pool_file = tmp_path / "hf_pool.json"
    pool_file.write_text(
        json.dumps(
            [
                {"namespace": "user_a", "token_env_var": "HF_TOKEN_A"},
            ]
        ),
        encoding="utf-8",
    )
    remote_cfg = {
        "hf_namespace_placeholder": "__namespace__",
        "hf_token_pool_file": str(pool_file),
    }

    mock_prefix_exists.return_value = False
    mock_repo_used_storage.return_value = 0
    mock_namespace_used_storage.return_value = 5

    resolved = resolve_hf_remote_from_pool(
        "hf://datasets/__namespace__/SER/V4_qwen3_4b_base/run_c",
        remote_cfg,
    )

    assert resolved is not None
    assert resolved.uri == "hf://datasets/user_a/SER/V4_qwen3_4b_base/run_c"


@patch("verl_inf_evolve.storage.hf_remote_resolver._discover_namespace_from_token")
@patch("verl_inf_evolve.storage.hf_remote_resolver._namespace_used_storage")
@patch("verl_inf_evolve.storage.hf_remote_resolver._repo_used_storage")
@patch("verl_inf_evolve.storage.hf_remote_resolver._prefix_exists")
def test_loads_raw_tokens_from_file_and_auto_discovers_namespace(
    mock_prefix_exists,
    mock_repo_used_storage,
    mock_namespace_used_storage,
    mock_discover_namespace,
    tmp_path,
):
    pool_file = tmp_path / "hf_pool.json"
    pool_file.write_text(
        json.dumps(["tok-a", "tok-b"]),
        encoding="utf-8",
    )
    remote_cfg = {
        "hf_namespace_placeholder": "__namespace__",
        "hf_token_pool_file": str(pool_file),
    }

    mock_discover_namespace.side_effect = ["user_a", "user_b"]
    mock_prefix_exists.side_effect = [False, False]
    mock_repo_used_storage.side_effect = [0, 0]
    mock_namespace_used_storage.side_effect = [5, 10]

    resolved = resolve_hf_remote_from_pool(
        "hf://datasets/__namespace__/SER/V4_qwen3_4b_base/run_d",
        remote_cfg,
    )

    assert resolved is not None
    assert resolved.uri == "hf://datasets/user_a/SER/V4_qwen3_4b_base/run_d"


@patch("verl_inf_evolve.storage.hf_remote_resolver._discover_namespace_from_token")
@patch("verl_inf_evolve.storage.hf_remote_resolver._namespace_used_storage")
@patch("verl_inf_evolve.storage.hf_remote_resolver._repo_used_storage")
@patch("verl_inf_evolve.storage.hf_remote_resolver._prefix_exists")
def test_loads_pool_from_single_env_var_json(
    mock_prefix_exists,
    mock_repo_used_storage,
    mock_namespace_used_storage,
    mock_discover_namespace,
    monkeypatch,
):
    monkeypatch.setenv(
        "HF_TOKEN_POOL_JSON",
        json.dumps(
            [
                {"token": "tok-a"},
                {"token": "tok-b", "capacity_bytes": 200},
            ]
        ),
    )
    remote_cfg = {
        "hf_namespace_placeholder": "__namespace__",
        "hf_token_pool_env_var": "HF_TOKEN_POOL_JSON",
    }

    mock_discover_namespace.side_effect = ["user_a", "user_b"]
    mock_prefix_exists.side_effect = [False, False]
    mock_repo_used_storage.side_effect = [0, 0]
    mock_namespace_used_storage.side_effect = [120, 20]

    resolved = resolve_hf_remote_from_pool(
        "hf://datasets/__namespace__/SER/V4_qwen3_4b_base/run_e",
        remote_cfg,
    )

    assert resolved is not None
    assert resolved.uri == "hf://datasets/user_b/SER/V4_qwen3_4b_base/run_e"


@patch("verl_inf_evolve.storage.hf_remote_resolver._discover_namespace_from_token")
def test_load_token_pool_with_warnings_skips_invalid_entries(
    mock_discover_namespace,
    monkeypatch,
):
    monkeypatch.setenv(
        "HF_TOKEN_POOL_JSON",
        json.dumps(
            [
                {"token": "tok-a"},
                {"token": "tok-b", "namespace": "user_b"},
            ]
        ),
    )
    remote_cfg = {
        "hf_token_pool_env_var": "HF_TOKEN_POOL_JSON",
    }
    mock_discover_namespace.side_effect = ValueError("Invalid user token.")

    resolved, warnings = load_token_pool_with_warnings(remote_cfg)

    assert [(entry.namespace, entry.token) for entry in resolved] == [("user_b", "tok-b")]
    assert warnings == ["HF token pool entry 0 skipped: Invalid user token."]


def test_discover_namespace_from_token_caches_success():
    hf_remote_resolver._discover_namespace_from_token.cache_clear()
    with patch("verl_inf_evolve.storage.hf_remote_resolver.HfApi") as api_cls:
        api_cls.return_value.whoami.return_value = {"name": "user_a"}

        first = hf_remote_resolver._discover_namespace_from_token("tok-a")
        second = hf_remote_resolver._discover_namespace_from_token("tok-a")

    assert first == "user_a"
    assert second == "user_a"
    assert api_cls.return_value.whoami.call_count == 1
    hf_remote_resolver._discover_namespace_from_token.cache_clear()
