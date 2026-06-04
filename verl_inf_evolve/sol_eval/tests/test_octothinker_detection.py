from __future__ import annotations

from verl_inf_evolve.sol_eval.sol_eval import _detect_model_from_remote_path
from verl_inf_evolve.templates import load_source


def test_detect_model_from_remote_path_uses_octothinker_solver_template() -> None:
    model_path, chat_template = _detect_model_from_remote_path(
        "s3://example-bucket/running-states/V3_2_octothinker_3b_hybrid_base/demo"
    )

    assert model_path == "OctoThinker/OctoThinker-3B-Hybrid-Base"
    assert chat_template == load_source("octothinker_shared_chat_template.jinja")


def test_detect_model_from_remote_path_keeps_qwen_template_none() -> None:
    model_path, chat_template = _detect_model_from_remote_path(
        "s3://example-bucket/running-states/V3_2_qwen3_4b/demo"
    )

    assert model_path == "Qwen/Qwen3-4B"
    assert chat_template is None
