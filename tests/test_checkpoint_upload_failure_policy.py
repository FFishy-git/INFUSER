from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer


def _make_trainer(stop_on_failure: bool) -> SelfEvolutionTrainer:
    trainer = object.__new__(SelfEvolutionTrainer)
    trainer._stop_on_checkpoint_upload_failure = stop_on_failure
    trainer._upload_manager = MagicMock()
    trainer._last_remote_checkpointed_ans_loop = None
    trainer._failed_remote_checkpoint_ans_loops = set()
    trainer._pending_ckpt_upload_id = None
    trainer._pending_ckpt_marker_upload_id = None
    trainer._pending_ckpt_ans_loop = None
    return trainer


def test_checkpoint_upload_failure_handler_is_noop_by_default():
    trainer = _make_trainer(stop_on_failure=False)
    task = SimpleNamespace(remote_key="global_step_12")

    with patch("os.kill") as mock_kill:
        trainer._handle_checkpoint_upload_failure(
            task,
            RuntimeError("upload failed"),
        )

    mock_kill.assert_not_called()


def test_checkpoint_upload_failure_handler_stops_job_when_enabled():
    trainer = _make_trainer(stop_on_failure=True)
    task = SimpleNamespace(remote_key="global_step_12")

    with patch("os.kill") as mock_kill:
        trainer._handle_checkpoint_upload_failure(
            task,
            RuntimeError("upload failed"),
        )

    mock_kill.assert_called_once()


def test_finalize_pending_checkpoint_upload_updates_last_remote_on_success():
    trainer = _make_trainer(stop_on_failure=False)
    trainer._last_remote_checkpointed_ans_loop = 4
    trainer._failed_remote_checkpoint_ans_loops = {5}
    trainer._pending_ckpt_upload_id = "ckpt-6"
    trainer._pending_ckpt_marker_upload_id = "marker-6"
    trainer._pending_ckpt_ans_loop = 6
    trainer._upload_manager.wait_for_task.side_effect = [True, True]

    trainer._finalize_pending_checkpoint_upload(
        raise_on_failure=False,
    )

    assert trainer._last_remote_checkpointed_ans_loop == 6
    assert trainer._failed_remote_checkpoint_ans_loops == {5}
    assert trainer._pending_ckpt_upload_id is None
    assert trainer._pending_ckpt_marker_upload_id is None
    assert trainer._pending_ckpt_ans_loop is None


def test_finalize_pending_checkpoint_upload_preserves_last_remote_on_failure():
    trainer = _make_trainer(stop_on_failure=False)
    trainer._last_remote_checkpointed_ans_loop = 4
    trainer._pending_ckpt_upload_id = "ckpt-5"
    trainer._pending_ckpt_marker_upload_id = "marker-5"
    trainer._pending_ckpt_ans_loop = 5
    trainer._upload_manager.wait_for_task.side_effect = [False, False]

    trainer._finalize_pending_checkpoint_upload(
        raise_on_failure=False,
    )

    assert trainer._last_remote_checkpointed_ans_loop == 4
    assert trainer._failed_remote_checkpoint_ans_loops == {5}


def test_finalize_pending_checkpoint_upload_raises_when_enabled():
    trainer = _make_trainer(stop_on_failure=True)
    trainer._pending_ckpt_upload_id = "ckpt-5"
    trainer._pending_ckpt_marker_upload_id = "marker-5"
    trainer._pending_ckpt_ans_loop = 5
    trainer._upload_manager.wait_for_task.side_effect = [False, False]

    try:
        trainer._finalize_pending_checkpoint_upload(
            raise_on_failure=True,
        )
    except RuntimeError as exc:
        assert "ckpt-5" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for failed checkpoint upload")


def test_finalize_pending_checkpoint_upload_handles_marker_only_commit():
    trainer = _make_trainer(stop_on_failure=False)
    trainer._last_remote_checkpointed_ans_loop = 2
    trainer._pending_ckpt_upload_id = None
    trainer._pending_ckpt_marker_upload_id = "marker-3"
    trainer._pending_ckpt_ans_loop = 3
    trainer._upload_manager.wait_for_task.return_value = True

    trainer._finalize_pending_checkpoint_upload(
        raise_on_failure=False,
    )

    assert trainer._last_remote_checkpointed_ans_loop == 3
    assert trainer._failed_remote_checkpoint_ans_loops == set()
