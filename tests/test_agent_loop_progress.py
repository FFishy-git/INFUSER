import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl.experimental.agent_loop.agent_loop import (
    _format_progress_message,
    _gather_indexed_tasks_with_progress,
)


def test_format_progress_message_includes_core_fields():
    message = _format_progress_message("stage1/dev_rollout", 5, 10, 0.0, 10.0)

    assert "stage1/dev_rollout progress: 5/10 (50.0%)" in message
    assert "elapsed=10.0s" in message
    assert "pending=5" in message
    assert "rate=0.50/s" in message
    assert "eta=10.0s" in message


def test_gather_indexed_tasks_with_progress_preserves_output_order():
    async def _indexed_result(index: int, delay: float) -> tuple[int, str]:
        await asyncio.sleep(delay)
        return index, f"value-{index}"

    async def _run():
        progress_logs: list[str] = []
        tasks = [
            asyncio.create_task(_indexed_result(0, 0.03)),
            asyncio.create_task(_indexed_result(1, 0.01)),
            asyncio.create_task(_indexed_result(2, 0.02)),
        ]
        outputs = await _gather_indexed_tasks_with_progress(
            tasks,
            label="stage3/gen_answer_rollout",
            enable_progress=True,
            progress_log_interval_seconds=999.0,
            progress_log_interval_samples=1,
            emit_progress=progress_logs.append,
        )
        return outputs, progress_logs

    outputs, progress_logs = asyncio.run(_run())

    assert outputs == ["value-0", "value-1", "value-2"]
    assert progress_logs[0].startswith("stage3/gen_answer_rollout progress: 0/3")
    assert progress_logs[-1].startswith("stage3/gen_answer_rollout progress: 3/3")
    assert any("1/3" in message for message in progress_logs)
    assert any("2/3" in message for message in progress_logs)
