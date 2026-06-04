"""Tests for backend-agnostic remote ops used by sol_eval."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from verl_inf_evolve.sol_eval.r2_ops import (
    check_r2_result_exists,
    discover_checkpoints_on_r2,
    download_checkpoint_from_r2,
    load_run_metadata,
    resolve_remote_uri,
)
from verl_inf_evolve.sol_eval.result_format import compute_eval_metrics, format_result_json


def _complete_result_json() -> dict:
    question_results = [
        {
            "question_id": "q1",
            "question_text": "Question 1",
            "choices": ["A", "B", "C", "D"],
            "ground_truth": "A",
            "sampled_answers": ["A"],
            "extracted_answers": ["A"],
            "answer_scores": [1.0],
            "response_token_lengths": [8],
        }
    ]
    metrics = compute_eval_metrics(question_results, n_samples=1)
    return format_result_json(question_results, metrics)


class TestResolveRemoteUri:
    def test_non_hf_uri_passthrough(self):
        uri, token = resolve_remote_uri("s3://bucket/run", remote_cfg={})
        assert uri == "s3://bucket/run"
        assert token is None

    def test_hf_placeholder_resolution_uses_pool_helper(self):
        resolved = SimpleNamespace(
            uri="hf://datasets/alice/SER/qwen3_4b_base/run1",
            token="hf_test_token",
        )
        with patch(
            "verl_inf_evolve.storage.hf_remote_resolver.resolve_hf_remote_from_pool",
            return_value=resolved,
        ):
            uri, token = resolve_remote_uri(
                "hf://datasets/__namespace__/SER/qwen3_4b_base/run1",
                remote_cfg={"hf_namespace_placeholder": "__namespace__"},
            )

        assert uri == "hf://datasets/alice/SER/qwen3_4b_base/run1"
        assert token == "hf_test_token"

    def test_hf_fixed_namespace_uses_matching_pool_token(self):
        with patch(
            "verl_inf_evolve.storage.hf_remote_resolver.resolve_hf_remote_from_pool",
            return_value=None,
        ), patch(
            "verl_inf_evolve.storage.hf_remote_resolver.load_token_pool_with_warnings",
            return_value=(
                [
                    SimpleNamespace(namespace="alice", token="hf_alice"),
                    SimpleNamespace(namespace="beiningwu7", token="hf_beiningwu7"),
                ],
                [],
            ),
        ):
            uri, token = resolve_remote_uri(
                "hf://datasets/beiningwu7/SER-eval",
                remote_cfg={"hf_token_pool_env_var": "HF_TOKEN_POOL_JSON"},
            )

        assert uri == "hf://datasets/beiningwu7/SER-eval"
        assert token == "hf_beiningwu7"

    def test_hf_explicit_token_wins_over_pool(self):
        with patch(
            "verl_inf_evolve.storage.hf_remote_resolver.resolve_hf_remote_from_pool",
            return_value=None,
        ), patch(
            "verl_inf_evolve.storage.hf_remote_resolver.load_token_pool_with_warnings",
            return_value=([SimpleNamespace(namespace="beiningwu7", token="hf_pool")], []),
        ):
            uri, token = resolve_remote_uri(
                "hf://datasets/beiningwu7/SER-eval",
                remote_cfg={"hf_token": "hf_explicit"},
            )

        assert uri == "hf://datasets/beiningwu7/SER-eval"
        assert token == "hf_explicit"


class TestGenericRemoteOps:
    def test_discover_checkpoints_uses_backend_listing(self):
        backend = MagicMock()
        backend.list_immediate_children.return_value = [
            "global_step_10",
            "notes.txt",
            "global_step_2",
        ]
        with patch(
            "verl_inf_evolve.sol_eval.r2_ops._create_remote_backend",
            return_value=(backend, "hf://datasets/alice/SER/run1"),
        ):
            checkpoints = discover_checkpoints_on_r2("hf://datasets/alice/SER/run1")

        assert checkpoints == [2, 10]

    def test_check_result_exists_downloads_and_validates_json(self):
        backend = MagicMock()
        backend.exists.return_value = True
        backend.download_bytes.return_value = json.dumps(_complete_result_json()).encode("utf-8")
        with patch(
            "verl_inf_evolve.sol_eval.r2_ops._create_remote_backend",
            return_value=(backend, "hf://datasets/alice/eval/results"),
        ):
            assert check_r2_result_exists("hf://datasets/alice/eval/results", "aime_result.json")

    def test_download_checkpoint_uses_backend_directory_download(self, tmp_path):
        backend = MagicMock()
        backend.download_dir.return_value = True
        local_cache_dir = tmp_path / "global_step_5"

        with patch(
            "verl_inf_evolve.sol_eval.r2_ops._create_remote_backend",
            return_value=(backend, "hf://datasets/alice/SER/run1"),
        ):
            result = download_checkpoint_from_r2(
                "hf://datasets/alice/SER/run1",
                ckpt_num=5,
                local_cache_dir=str(local_cache_dir),
            )

        assert result == str(local_cache_dir)
        backend.download_dir.assert_called_once_with(
            "global_step_5/solver",
            str(local_cache_dir / "solver"),
        )

    def test_load_run_metadata_downloads_json(self):
        backend = MagicMock()
        backend.download_bytes.return_value = json.dumps(
            {"solver_model_id": "Qwen/Qwen3-4B-Base"}
        ).encode("utf-8")
        with patch(
            "verl_inf_evolve.sol_eval.r2_ops._create_remote_backend",
            return_value=(backend, "hf://datasets/alice/SER/run1"),
        ):
            metadata = load_run_metadata("hf://datasets/alice/SER/run1")

        assert metadata == {"solver_model_id": "Qwen/Qwen3-4B-Base"}
