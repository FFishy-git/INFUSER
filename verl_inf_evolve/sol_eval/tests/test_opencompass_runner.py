from __future__ import annotations

from pathlib import Path

import pytest

from verl_inf_evolve.sol_eval.opencompass_runner import (
    build_model_id,
    detect_hf_type,
    launch_with_python_config,
    launch_opencompass,
    main,
    pass_at_k_ladder,
    resolve_benchmark_names,
    resolve_tokens,
)
from verl_inf_evolve.utils.benchmarks.model_type import detect_model_type


def test_detect_hf_type_and_utils_model_type_stay_aligned() -> None:
    assert detect_hf_type("Qwen/Qwen3-8B-Base") == "base"
    assert detect_model_type("Qwen/Qwen3-8B-Base") == "base"
    assert detect_hf_type("Qwen/Qwen3-8B-Instruct") == "chat"
    assert detect_model_type("Qwen/Qwen3-8B-Instruct") == "chat"


def test_resolve_benchmark_names_expands_groups() -> None:
    assert resolve_benchmark_names("code") == ["humaneval", "livecodebench"]


def test_resolve_tokens_applies_chat_overrides_only_when_needed() -> None:
    assert resolve_tokens("gpqa", "base") == ["gpqa_few_shot_ppl_4b5a83"]
    assert resolve_tokens("gpqa", "chat") == ["gpqa_gen"]


def test_pass_at_k_ladder_uses_powers_of_two() -> None:
    assert pass_at_k_ladder(1) == [1]
    assert pass_at_k_ladder(8) == [1, 2, 4, 8]
    assert pass_at_k_ladder(10) == [1, 2, 4, 8]


def test_build_model_id_handles_checkpoints_and_plain_models() -> None:
    assert build_model_id("Qwen/Qwen3-8B-Base", None, None, None) == "Qwen__Qwen3-8B-Base"
    assert (
        build_model_id(
            "unused",
            "example/SER",
            "qwen3_4b_base/example_run",
            "50",
        )
        == "qwen3_4b_base/example_run/global_step_50"
    )


def test_main_forwards_custom_chat_template_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "template.jinja"
    template_path.write_text("{{ messages }}", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_launch_opencompass(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(
        "verl_inf_evolve.sol_eval.opencompass_runner.launch_opencompass",
        fake_launch_opencompass,
    )

    exit_code = main(
        [
            "--hf-path",
            "Qwen/Qwen3-8B",
            "--hf-type",
            "chat",
            "--benchmarks",
            "gpqa",
            "--custom-chat-template-file",
            str(template_path),
            "--no-upload",
        ]
    )

    assert exit_code == 0
    assert captured["hf_path"] == "Qwen/Qwen3-8B"
    assert captured["hf_type"] == "chat"
    assert captured["benchmarks"] == "gpqa"
    assert captured["custom_chat_template"] == "{{ messages }}"


def test_launch_opencompass_routes_chat_gpqa_through_python_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "verl_inf_evolve.sol_eval.opencompass_runner.get_visible_gpus",
        lambda: 1,
    )
    monkeypatch.setattr(
        "verl_inf_evolve.sol_eval.opencompass_runner.ensure_opencompass_installed",
        lambda: None,
    )
    monkeypatch.setattr(
        "verl_inf_evolve.sol_eval.opencompass_runner.patch_opencompass_site_packages",
        lambda: None,
    )

    def fake_launch_with_python_config(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(
        "verl_inf_evolve.sol_eval.opencompass_runner.launch_with_python_config",
        fake_launch_with_python_config,
    )

    exit_code = launch_opencompass(
        hf_path="Qwen/Qwen3-8B",
        hf_type="chat",
        benchmarks="gpqa",
        output_dir=".output/test_gpqa_chat",
        no_upload=True,
    )

    assert exit_code == 0
    assert captured["hf_type"] == "chat"
    assert captured["bench_names"] == ["gpqa"]
    assert captured["tokens"] == ["gpqa_gen"]
    assert captured["bench_n_values"] == {"gpqa": 5}


def test_python_config_passes_full_pass_at_k_ladders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Result:
        returncode = 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "verl_inf_evolve.sol_eval.opencompass_runner.subprocess.run",
        lambda cmd: _Result(),
    )

    exit_code = launch_with_python_config(
        hf_path="Qwen/Qwen3-8B-Base",
        hf_type="base",
        tokens=["humaneval_plus_gen_8e312c", "livecodebench_gen"],
        bench_names=["humaneval", "livecodebench"],
        bench_n_values={"humaneval": 8, "livecodebench": 2},
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        num_gpus_int=1,
        max_out_len="4096",
        output_dir=None,
        passthrough=[],
        upload_repo=None,
        no_upload=True,
        hf_repo=None,
        ckpt_prefix=None,
        ckpt_step=None,
    )

    assert exit_code == 0
    config_text = (tmp_path / ".cache" / "opencompass_configs" / "eval_config.py").read_text(
        encoding="utf-8"
    )
    assert "HumanEvalPlusEvaluatorAZR', k=[1, 2, 4, 8])" in config_text
    assert "LCBCodeGenerationEvaluatorOfficial', num_process_evaluate=4, timeout=6" in config_text
    assert "model_type='base', k_list=[1, 2])" in config_text
