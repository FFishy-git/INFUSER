from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, call

from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.trainer.resume_state import ResumeState
from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer
from verl_inf_evolve.trainer.stage_context import StageContext


def _make_trainer_stub(
    *, save_every_n_steps: int, always_save_for_resume: bool, max_ans_loop: int
) -> SelfEvolutionTrainer:
    trainer = object.__new__(SelfEvolutionTrainer)
    trainer.config = OmegaConf.create(
        {
            "training": {
                "save_every_n_steps": save_every_n_steps,
                "always_save_for_resume": always_save_for_resume,
                "max_ans_loop": max_ans_loop,
            }
        }
    )
    return trainer


def test_interval_mode_stage_context_upload_window():
    trainer = _make_trainer_stub(
        save_every_n_steps=5,
        always_save_for_resume=False,
        max_ans_loop=12,
    )

    expected = {
        0: True,
        1: True,
        2: False,
        5: True,
        6: True,
        7: False,
        10: True,
        11: True,
    }

    for ans_loop, should_upload in expected.items():
        assert (
            trainer._should_upload_stage_context_for_ans_loop(ans_loop)
            is should_upload
        )


def test_continuous_mode_uses_same_save_window_rule():
    trainer = _make_trainer_stub(
        save_every_n_steps=5,
        always_save_for_resume=True,
        max_ans_loop=12,
    )

    expected = {
        0: True,
        1: True,
        2: False,
        5: True,
        6: True,
        7: False,
        10: True,
        11: True,
    }

    for ans_loop, should_upload in expected.items():
        assert (
            trainer._should_upload_stage_context_for_ans_loop(ans_loop)
            is should_upload
        )


def test_stage_context_skips_remote_upload_when_policy_disabled(tmp_path):
    resume = ResumeState(ans_loop=2, num_gen_per_ans=1)
    upload_manager = MagicMock()
    upload_manager.remote_configured = True
    upload_manager.upload_enabled = True

    with StageContext(
        name="scoring",
        stage_id=4,
        resume=resume,
        resume_dir=str(tmp_path),
        upload_manager=upload_manager,
        is_done=lambda: False,
        mark_done=lambda: setattr(resume, "stage_1_done", True),
        ans_loop=2,
        should_upload_remote=False,
    ) as ctx:
        assert ctx.should_run is True
        ctx.save_json("rewards", {"score": 1.0})

    assert resume.stage_1_done is True
    assert not os.path.exists(tmp_path / "rewards.json")
    assert not os.path.exists(tmp_path / "state.json")
    upload_manager.submit_memory_upload.assert_not_called()
    upload_manager.submit_dir_upload.assert_not_called()
    upload_manager.submit_file_upload.assert_not_called()


def test_stage_context_uploads_remote_artifacts_when_policy_enabled(tmp_path):
    resume = ResumeState(ans_loop=5, num_gen_per_ans=1)
    upload_manager = MagicMock()
    upload_manager.remote_configured = True
    upload_manager.upload_enabled = True
    upload_manager.submit_memory_upload.side_effect = [
        "artifact-task",
        "state-task",
    ]

    with StageContext(
        name="scoring",
        stage_id=4,
        resume=resume,
        resume_dir=str(tmp_path),
        upload_manager=upload_manager,
        is_done=lambda: False,
        mark_done=lambda: setattr(resume, "stage_1_done", True),
        ans_loop=5,
        should_upload_remote=True,
    ) as ctx:
        ctx.save_json("rewards", {"score": 1.0})

    assert upload_manager.submit_memory_upload.call_args_list == [
        call(
            name="rewards",
            data={"score": 1.0},
            kind="json",
            remote_key="ans_5/rewards.json",
        ),
        call(
            name="scoring_state",
            data=resume.to_dict(),
            kind="json",
            remote_key="ans_5/state.json",
            depends_on=["artifact-task"],
        ),
    ]
    upload_manager.submit_file_upload.assert_not_called()
    assert not os.path.exists(tmp_path / "rewards.json")
    assert not os.path.exists(tmp_path / "state.json")


def test_stage_context_persists_locally_when_uploads_disabled(tmp_path):
    resume = ResumeState(ans_loop=3, num_gen_per_ans=1)
    upload_manager = MagicMock()
    upload_manager.remote_configured = True
    upload_manager.upload_enabled = False

    with StageContext(
        name="scoring",
        stage_id=4,
        resume=resume,
        resume_dir=str(tmp_path),
        upload_manager=upload_manager,
        is_done=lambda: False,
        mark_done=lambda: setattr(resume, "stage_1_done", True),
        ans_loop=3,
        should_upload_remote=True,
    ) as ctx:
        ctx.save_json("rewards", {"score": 2.0})

    assert os.path.exists(tmp_path / "rewards.json")
    assert os.path.exists(tmp_path / "state.json")
    upload_manager.submit_memory_upload.assert_not_called()
    upload_manager.submit_file_upload.assert_not_called()
