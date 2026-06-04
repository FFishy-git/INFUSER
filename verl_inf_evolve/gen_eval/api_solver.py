"""API-based MCQ solver for gpt_replay mode.

Replaces the vLLM GPU solver rollout with OpenAI/Gemini API calls.
Supports two modes:
- **sync**: Sends questions concurrently via ThreadPoolExecutor (real-time).
- **batch**: Submits all questions as an OpenAI Batch API job (50% cheaper,
  up to 24h turnaround).

Extracts answers from ``\\boxed{}`` blocks and returns per-question scores
compatible with ``compute_answer_rollout_metrics()``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from verl_inf_evolve.utils.mcq_utils import (
    build_mcq_messages,
    extract_boxed_answer,
    is_correct,
)

logger = logging.getLogger(__name__)

_BATCH_TERMINAL_STATUSES = {"failed", "completed", "expired", "cancelled"}

# Pricing per 1M tokens (USD). Add new models as needed.
# Source: https://openai.com/api/pricing/
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "gpt-5.4": {"input": 2.00, "output": 8.00},
}


class _UsageAccumulator:
    """Thread-safe accumulator for API token usage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.total_tokens = 0
        self.num_calls = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        with self._lock:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.total_tokens += getattr(usage, "total_tokens", 0) or 0
            self.num_calls += 1
            # Reasoning tokens (o1, o3, gpt-5.x models)
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                self.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0

    def estimate_cost(self, model: str) -> Optional[float]:
        """Estimate cost in USD based on model pricing."""
        pricing = _MODEL_PRICING.get(model)
        if pricing is None:
            return None
        input_cost = self.prompt_tokens * pricing["input"] / 1_000_000
        output_cost = self.completion_tokens * pricing["output"] / 1_000_000
        return input_cost + output_cost

    def summary(self, model: str) -> str:
        """Human-readable summary with cost estimate."""
        parts = [
            f"prompt={self.prompt_tokens:,}",
            f"completion={self.completion_tokens:,}",
        ]
        if self.reasoning_tokens > 0:
            visible = self.completion_tokens - self.reasoning_tokens
            parts.append(f"reasoning={self.reasoning_tokens:,}")
            parts.append(f"visible_output={visible:,}")
        parts.append(f"total={self.total_tokens:,}")
        cost = self.estimate_cost(model)
        if cost is not None:
            parts.append(f"est_cost=${cost:.4f}")
        return ", ".join(parts)

    def to_metrics(self, model: str, prefix: str = "api_solver") -> dict[str, float]:
        """Return metrics dict for wandb logging."""
        m: dict[str, float] = {
            f"{prefix}/prompt_tokens": float(self.prompt_tokens),
            f"{prefix}/completion_tokens": float(self.completion_tokens),
            f"{prefix}/reasoning_tokens": float(self.reasoning_tokens),
            f"{prefix}/total_tokens": float(self.total_tokens),
            f"{prefix}/num_calls": float(self.num_calls),
        }
        cost = self.estimate_cost(model)
        if cost is not None:
            m[f"{prefix}/estimated_cost_usd"] = cost
        if self.num_calls > 0:
            m[f"{prefix}/avg_completion_tokens"] = self.completion_tokens / self.num_calls
            m[f"{prefix}/avg_prompt_tokens"] = self.prompt_tokens / self.num_calls
        return m


@dataclass
class APISolverResult:
    """Per-sample result from an API solver call."""

    question_id: str
    ground_truth: str
    extracted_answer: Optional[str]
    is_correct: bool
    raw_response: str
    sample_idx: int = 0
    error: Optional[str] = None


@dataclass
class BatchHandle:
    """Handle returned by ``submit_batch()`` for later collection.

    Stores all state needed to poll and collect results without
    re-deriving questions or re-reading gen_output.pt.
    """

    output_dir: Path
    state_path: Path
    custom_id_to_work: dict  # {custom_id: (question, sample_idx, normalized_gt)}
    work_items: list  # [(question, sample_idx, normalized_gt), ...]
    num_requests: int


def _create_openai_client(config: Any) -> Any:
    """Create an OpenAI client from config."""
    from openai import OpenAI

    api_key_env_var = config.get("api_key_env_var", "OPENAI_API_KEY")
    api_key = config.get("api_key", None) or os.environ.get(api_key_env_var)
    if not api_key:
        raise ValueError(
            f"Missing API key. Set api_solver.api_key or the "
            f"{api_key_env_var} environment variable."
        )

    base_url = config.get("base_url", None)
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _create_gemini_client(config: Any) -> Any:
    """Create a Gemini client (via openai-compatible interface)."""
    from openai import OpenAI

    api_key_env_var = config.get("api_key_env_var", "GEMINI_API_KEY")
    api_key = config.get("api_key", None) or os.environ.get(api_key_env_var)
    if not api_key:
        raise ValueError(
            f"Missing Gemini API key. Set api_solver.api_key or the "
            f"{api_key_env_var} environment variable."
        )

    base_url = config.get(
        "base_url", "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    return OpenAI(api_key=api_key, base_url=base_url)


class APISolverClient:
    """Concurrent API-based MCQ solver.

    Supports OpenAI and Gemini providers via their chat completions API.
    Sends each question ``n`` times (configurable), extracts answers from
    ``\\boxed{}`` blocks, and returns scores in the same format as
    ``group_scores_by_qid()``.

    Args:
        config: OmegaConf node with fields: ``provider``, ``model``,
            ``api_key_env_var``, ``max_concurrent``, ``max_retries``,
            ``retry_backoff_s``, ``timeout_s``, ``temperature``, ``n``,
            ``max_tokens``.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.provider = str(config.get("provider", "openai")).lower()
        self.model = str(config.model)
        self.temperature = float(config.get("temperature", 0.7))
        self.max_tokens = int(config.get("max_tokens", 8192))
        self.n = int(config.get("n", 1))
        self.max_concurrent = max(1, int(config.get("max_concurrent", 16)))
        self.max_retries = max(1, int(config.get("max_retries", 3)))
        self.retry_backoff_s = float(config.get("retry_backoff_s", 1.0))
        self.timeout_s = float(config.get("timeout_s", 120))
        self.mode = str(config.get("mode", "sync")).lower()
        self.batch_poll_interval_s = float(config.get("batch_poll_interval_s", 60.0))
        self.batch_completion_window = str(config.get("batch_completion_window", "24h"))
        self._client: Any = None
        self._usage = _UsageAccumulator()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            self._client = _create_openai_client(self.config)
        elif self.provider == "gemini":
            self._client = _create_gemini_client(self.config)
        else:
            raise ValueError(
                f"Unsupported api_solver.provider={self.provider!r}. "
                "Accepted values: 'openai', 'gemini'."
            )
        return self._client

    def _call_api(self, messages: list[dict[str, str]]) -> str:
        """Call the chat completions API with retries and return the response text."""
        client = self._ensure_client()

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    timeout=self.timeout_s,
                )
                self._usage.add(getattr(response, "usage", None))
                content = response.choices[0].message.content
                return content if content else ""
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                wait = self.retry_backoff_s * (2 ** attempt)
                logger.warning(
                    "API call attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    str(exc)[:200],
                    wait,
                )
                time.sleep(wait)

        # Unreachable, but satisfies type checker
        raise RuntimeError("API call failed after all retries")

    def _solve_single(
        self,
        question: dict[str, Any],
        sample_idx: int,
    ) -> APISolverResult:
        """Solve a single question once, returning an APISolverResult."""
        qid = str(question["question_id"])

        try:
            messages, normalized_gt = build_mcq_messages(question)
        except (ValueError, KeyError) as exc:
            return APISolverResult(
                question_id=qid,
                ground_truth=str(question.get("ground_truth", "")),
                extracted_answer=None,
                is_correct=False,
                raw_response="",
                sample_idx=sample_idx,
                error=f"build_mcq_messages failed: {exc}",
            )

        try:
            raw_response = self._call_api(messages)
        except Exception as exc:
            return APISolverResult(
                question_id=qid,
                ground_truth=normalized_gt,
                extracted_answer=None,
                is_correct=False,
                raw_response="",
                sample_idx=sample_idx,
                error=f"API call failed: {exc}",
            )

        predicted = extract_boxed_answer(raw_response)
        correct = is_correct(predicted, normalized_gt) if predicted is not None else False

        return APISolverResult(
            question_id=qid,
            ground_truth=normalized_gt,
            extracted_answer=predicted,
            is_correct=correct,
            raw_response=raw_response,
            sample_idx=sample_idx,
        )

    def solve_questions(
        self,
        questions: list[dict[str, Any]],
    ) -> tuple[dict[str, list[Optional[float]]], list[APISolverResult]]:
        """Solve all questions, each ``n`` times, returning scores and details.

        Args:
            questions: List of question dicts, each with ``question_id``,
                ``question_text``, ``choices``, ``ground_truth``.

        Returns:
            A tuple of:
            - ``scores_by_qid``: ``{question_id: [score1, score2, ...]}`` where
              score is ``1.0`` (correct), ``0.0`` (wrong), or ``None`` (extraction
              failed). Compatible with ``compute_answer_rollout_metrics()``.
            - ``all_results``: Flat list of ``APISolverResult`` objects for
              detailed logging.
        """
        if not questions:
            return {}, []

        # Build work items: (question, sample_idx) for each question x n
        work_items: list[tuple[dict[str, Any], int]] = []
        for q in questions:
            for sample_idx in range(self.n):
                work_items.append((q, sample_idx))

        total = len(work_items)
        logger.info(
            "APISolverClient: solving %d questions x %d samples = %d API calls "
            "(model=%s, provider=%s, max_concurrent=%d)",
            len(questions),
            self.n,
            total,
            self.model,
            self.provider,
            self.max_concurrent,
        )

        all_results: list[APISolverResult] = []
        completed_count = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_idx = {
                executor.submit(self._solve_single, q, si): idx
                for idx, (q, si) in enumerate(work_items)
            }
            # Pre-allocate results list
            indexed_results: list[Optional[APISolverResult]] = [None] * total

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:
                    q, si = work_items[idx]
                    result = APISolverResult(
                        question_id=str(q["question_id"]),
                        ground_truth=str(q.get("ground_truth", "")),
                        extracted_answer=None,
                        is_correct=False,
                        raw_response="",
                        sample_idx=si,
                        error=f"Unexpected error: {exc}",
                    )
                indexed_results[idx] = result
                completed_count += 1
                if completed_count % max(total // 10, 1) == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "APISolverClient progress: %d/%d (%.1f%%) in %.1fs",
                        completed_count,
                        total,
                        100 * completed_count / total,
                        elapsed,
                    )

        all_results = [r for r in indexed_results if r is not None]
        elapsed = time.time() - t0

        # Summary stats
        num_correct = sum(1 for r in all_results if r.is_correct)
        num_failed = sum(1 for r in all_results if r.error is not None)
        num_no_answer = sum(
            1 for r in all_results if r.extracted_answer is None and r.error is None
        )
        logger.info(
            "APISolverClient complete: %d results in %.1fs — "
            "%d correct, %d failed, %d no answer extracted",
            len(all_results),
            elapsed,
            num_correct,
            num_failed,
            num_no_answer,
        )
        logger.info("APISolverClient token usage: %s", self._usage.summary(self.model))

        return self._results_to_scores(all_results), all_results

    def get_usage_metrics(self, prefix: str = "api_solver") -> dict[str, float]:
        """Return accumulated token usage and cost metrics for wandb logging."""
        return self._usage.to_metrics(self.model, prefix)

    def reset_usage(self) -> None:
        """Reset the usage accumulator (call between checkpoints)."""
        self._usage = _UsageAccumulator()

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------

    def _build_batch_custom_id(
        self,
        question: dict[str, Any],
        sample_idx: int,
    ) -> str:
        qid = str(question["question_id"])
        stable_key = f"{qid}:{sample_idx}"
        return f"solve_{hashlib.md5(stable_key.encode('utf-8')).hexdigest()[:16]}"

    def _build_batch_request_row(
        self,
        messages: list[dict[str, str]],
        custom_id: str,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def submit_batch(
        self,
        questions: list[dict[str, Any]],
        output_dir: Path,
    ) -> Optional[BatchHandle]:
        """Submit questions to OpenAI Batch API without waiting.

        Prepares the JSONL request file, uploads it, and creates the batch
        job. Returns a :class:`BatchHandle` for later collection via
        :meth:`collect_batch`. If an existing ``batch_state.json`` is found
        in ``output_dir``, resumes from that state without re-submitting.

        Returns ``None`` if there are no valid questions to submit.
        """
        if not questions:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        state_path = output_dir / "batch_state.json"
        request_path = output_dir / "batch_requests.jsonl"

        # Build work items and prepare request rows
        work_items: list[tuple[dict[str, Any], int, str]] = []
        request_rows: list[dict[str, Any]] = []
        custom_id_to_work: dict[str, tuple[dict[str, Any], int, str]] = {}

        for q in questions:
            qid = str(q["question_id"])
            try:
                messages, normalized_gt = build_mcq_messages(q)
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping question %s: build_mcq_messages failed: %s", qid, exc)
                continue

            for sample_idx in range(self.n):
                custom_id = self._build_batch_custom_id(q, sample_idx)
                if custom_id in custom_id_to_work:
                    custom_id = f"{custom_id}_{sample_idx}"
                custom_id_to_work[custom_id] = (q, sample_idx, normalized_gt)
                work_items.append((q, sample_idx, normalized_gt))
                request_rows.append(
                    self._build_batch_request_row(messages, custom_id)
                )

        if not request_rows:
            logger.warning("No valid questions to submit for batch.")
            return None

        total = len(request_rows)
        logger.info(
            "APISolverClient batch: %d questions x %d samples = %d requests "
            "(model=%s, 50%% discount via Batch API)",
            len(questions), self.n, total, self.model,
        )

        # Submit or resume batch
        client = self._ensure_client()

        if state_path.is_file():
            with state_path.open("r") as f:
                batch_state = json.load(f)
            logger.info(
                "Resuming batch %s (status=%s)",
                batch_state.get("batch_id"), batch_state.get("status"),
            )
        else:
            with request_path.open("w", encoding="utf-8") as f:
                for row in request_rows:
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")

            with request_path.open("rb") as f:
                uploaded = client.files.create(file=f, purpose="batch")

            batch = client.batches.create(
                completion_window=self.batch_completion_window,
                endpoint="/v1/chat/completions",
                input_file_id=uploaded.id,
                metadata={
                    "solver_model": self.model,
                    "num_requests": str(total),
                },
                timeout=self.timeout_s,
            )
            batch_state = {
                "batch_id": batch.id,
                "status": batch.status,
                "input_file_id": batch.input_file_id,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_batch_state(state_path, batch_state)
            logger.info("Submitted batch %s (%d requests)", batch.id, total)

        return BatchHandle(
            output_dir=output_dir,
            state_path=state_path,
            custom_id_to_work=custom_id_to_work,
            work_items=work_items,
            num_requests=total,
        )

    def collect_batch(
        self,
        handle: BatchHandle,
    ) -> tuple[dict[str, list[Optional[float]]], list[APISolverResult]]:
        """Poll a submitted batch until complete and collect results.

        Blocks until the batch reaches a terminal status, then downloads
        and parses the output.

        Args:
            handle: The :class:`BatchHandle` returned by :meth:`submit_batch`.

        Returns:
            Same as ``solve_questions()``.
        """
        with handle.state_path.open("r") as f:
            batch_state = json.load(f)

        batch_id = str(batch_state["batch_id"])
        if batch_state.get("status") not in _BATCH_TERMINAL_STATUSES:
            batch_state = self._poll_batch(batch_id, handle.state_path)

        if batch_state.get("status") != "completed":
            logger.error(
                "Batch %s finished with status=%s",
                batch_id, batch_state.get("status"),
            )
            all_results = [
                APISolverResult(
                    question_id=str(q["question_id"]),
                    ground_truth=gt,
                    extracted_answer=None,
                    is_correct=False,
                    raw_response="",
                    sample_idx=si,
                    error=f"Batch {batch_state.get('status')}",
                )
                for q, si, gt in handle.work_items
            ]
            return self._results_to_scores(all_results), all_results

        return self._collect_batch_results(
            batch_state,
            handle.custom_id_to_work,
            handle.work_items,
            handle.output_dir,
        )

    def cancel_batch(self, output_dir: Path) -> str:
        """Cancel a running batch using its persisted state.

        Reads ``batch_state.json`` from *output_dir*, sends a cancel
        request to OpenAI, and updates the state file.

        Returns:
            A short status message suitable for logging / display.
        """
        state_path = output_dir / "batch_state.json"
        if not state_path.is_file():
            return f"No batch_state.json found in {output_dir}"

        with state_path.open("r") as f:
            batch_state = json.load(f)

        batch_id = str(batch_state.get("batch_id", ""))
        current_status = batch_state.get("status", "unknown")

        if current_status in _BATCH_TERMINAL_STATUSES:
            return f"Batch {batch_id} already in terminal state: {current_status}"

        client = self._ensure_client()
        batch = client.batches.cancel(batch_id, timeout=self.timeout_s)

        batch_state["status"] = batch.status
        self._save_batch_state(state_path, batch_state)

        return f"Batch {batch_id} cancel requested (status: {batch.status})"

    def cancel_all_batches(self, cache_dir: Path) -> list[str]:
        """Cancel all running batches under ``cache_dir/batch/*/``.

        Returns:
            List of status messages, one per checkpoint directory.
        """
        batch_root = cache_dir / "batch"
        if not batch_root.is_dir():
            return [f"No batch directory found at {batch_root}"]

        messages: list[str] = []
        for child in sorted(batch_root.iterdir()):
            if child.is_dir() and (child / "batch_state.json").is_file():
                msg = self.cancel_batch(child)
                messages.append(f"ans_loop={child.name}: {msg}")
                logger.info(msg)
        return messages if messages else ["No batch_state.json files found"]

    def solve_questions_batch(
        self,
        questions: list[dict[str, Any]],
        output_dir: Path,
    ) -> tuple[dict[str, list[Optional[float]]], list[APISolverResult]]:
        """Submit and collect a batch in one call (convenience wrapper).

        Equivalent to calling :meth:`submit_batch` followed by
        :meth:`collect_batch`.
        """
        handle = self.submit_batch(questions, output_dir)
        if handle is None:
            return {}, []
        return self.collect_batch(handle)

    def _save_batch_state(self, path: Path, state: dict[str, Any]) -> None:
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _poll_batch(
        self,
        batch_id: str,
        state_path: Path,
    ) -> dict[str, Any]:
        """Poll batch until terminal status, updating state file."""
        client = self._ensure_client()
        while True:
            batch = client.batches.retrieve(batch_id, timeout=self.timeout_s)
            state: dict[str, Any] = {
                "batch_id": batch.id,
                "status": batch.status,
                "input_file_id": batch.input_file_id,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
            }
            if batch.request_counts is not None:
                state["request_counts"] = batch.request_counts.model_dump(mode="json")
            self._save_batch_state(state_path, state)

            if batch.status in _BATCH_TERMINAL_STATUSES:
                logger.info("Batch %s reached terminal status: %s", batch_id, batch.status)
                return state

            logger.info(
                "Batch %s status=%s; sleeping %.0fs",
                batch_id, batch.status, self.batch_poll_interval_s,
            )
            time.sleep(self.batch_poll_interval_s)

    def _collect_batch_results(
        self,
        batch_state: dict[str, Any],
        custom_id_to_work: dict[str, tuple[dict[str, Any], int, str]],
        work_items: list[tuple[dict[str, Any], int, str]],
        output_dir: Path,
    ) -> tuple[dict[str, list[Optional[float]]], list[APISolverResult]]:
        """Download batch output and convert to APISolverResults."""
        client = self._ensure_client()
        all_results: list[APISolverResult] = []
        resolved_custom_ids: set[str] = set()

        # Download output file
        output_file_id = batch_state.get("output_file_id")
        if output_file_id:
            raw_text = client.files.content(output_file_id, timeout=self.timeout_s).text
            output_path = output_dir / "batch_output.jsonl"
            output_path.write_text(raw_text, encoding="utf-8")

            for line in raw_text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                custom_id = str(row.get("custom_id", ""))
                work = custom_id_to_work.get(custom_id)
                if work is None:
                    continue

                resolved_custom_ids.add(custom_id)
                q, sample_idx, normalized_gt = work
                qid = str(q["question_id"])

                error_info = row.get("error")
                response_info = row.get("response")

                if error_info is not None or response_info is None:
                    all_results.append(APISolverResult(
                        question_id=qid,
                        ground_truth=normalized_gt,
                        extracted_answer=None,
                        is_correct=False,
                        raw_response="",
                        sample_idx=sample_idx,
                        error=json.dumps(error_info) if error_info else "no response",
                    ))
                    continue

                status_code = int(response_info.get("status_code", 0))
                if status_code != 200:
                    all_results.append(APISolverResult(
                        question_id=qid,
                        ground_truth=normalized_gt,
                        extracted_answer=None,
                        is_correct=False,
                        raw_response="",
                        sample_idx=sample_idx,
                        error=f"HTTP {status_code}",
                    ))
                    continue

                # Extract content from chat completion response body
                body = response_info.get("body", {})
                choices = body.get("choices", [])
                raw_response = ""
                if choices:
                    message = choices[0].get("message", {})
                    raw_response = message.get("content", "") or ""

                predicted = extract_boxed_answer(raw_response)
                correct = is_correct(predicted, normalized_gt) if predicted is not None else False

                all_results.append(APISolverResult(
                    question_id=qid,
                    ground_truth=normalized_gt,
                    extracted_answer=predicted,
                    is_correct=correct,
                    raw_response=raw_response,
                    sample_idx=sample_idx,
                ))

        # Download error file (if any)
        error_file_id = batch_state.get("error_file_id")
        if error_file_id:
            try:
                error_text = client.files.content(error_file_id, timeout=self.timeout_s).text
                error_path = output_dir / "batch_errors.jsonl"
                error_path.write_text(error_text, encoding="utf-8")

                for line in error_text.splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    custom_id = str(row.get("custom_id", ""))
                    if custom_id in resolved_custom_ids:
                        continue
                    work = custom_id_to_work.get(custom_id)
                    if work is None:
                        continue
                    resolved_custom_ids.add(custom_id)
                    q, sample_idx, normalized_gt = work
                    all_results.append(APISolverResult(
                        question_id=str(q["question_id"]),
                        ground_truth=normalized_gt,
                        extracted_answer=None,
                        is_correct=False,
                        raw_response="",
                        sample_idx=sample_idx,
                        error=json.dumps(row.get("error", "batch error")),
                    ))
            except Exception as exc:
                logger.warning("Failed to download batch error file: %s", exc)

        # Fill missing results
        for custom_id, (q, sample_idx, normalized_gt) in custom_id_to_work.items():
            if custom_id not in resolved_custom_ids:
                all_results.append(APISolverResult(
                    question_id=str(q["question_id"]),
                    ground_truth=normalized_gt,
                    extracted_answer=None,
                    is_correct=False,
                    raw_response="",
                    sample_idx=sample_idx,
                    error="missing from batch output",
                ))

        # Summary
        num_correct = sum(1 for r in all_results if r.is_correct)
        num_errors = sum(1 for r in all_results if r.error is not None)
        logger.info(
            "Batch results collected: %d total, %d correct, %d errors",
            len(all_results), num_correct, num_errors,
        )

        return self._results_to_scores(all_results), all_results

    @staticmethod
    def _results_to_scores(
        results: list[APISolverResult],
    ) -> dict[str, list[Optional[float]]]:
        """Convert a list of APISolverResults to scores_by_qid dict."""
        scores_by_qid: dict[str, list[Optional[float]]] = {}
        for result in results:
            if result.error is not None and result.extracted_answer is None:
                score: Optional[float] = None
            elif result.extracted_answer is None:
                score = None
            else:
                score = 1.0 if result.is_correct else 0.0
            scores_by_qid.setdefault(result.question_id, []).append(score)
        return scores_by_qid


def compute_api_answer_metrics(
    scores: dict[str, list[Optional[float]]],
    rollout_n: int,
    prefix: str,
) -> dict[str, float]:
    """Compute answer rollout metrics without a response_mask tensor.

    This is a simplified version of ``compute_answer_rollout_metrics()`` that
    skips response token length statistics (which require a response_mask
    from the vLLM rollout output). All accuracy, extraction, and pass@k
    metrics are identical.

    Args:
        scores: ``{question_id: [score_per_sample]}`` where score is
            ``1.0`` (correct), ``0.0`` (wrong), or ``None`` (extraction failed).
        rollout_n: Number of samples per question.
        prefix: Metric key prefix (e.g. ``"gen_answer_rollout"``).
    """
    import numpy as np

    from verl_inf_evolve.trainer.rollout_metrics import compute_pass_at_k

    metrics: dict[str, float] = {}

    # --- Accuracy & extraction stats ---
    all_scores = [s for sl in scores.values() for s in sl]
    total = len(all_scores)
    valid = [s for s in all_scores if s is not None]
    num_valid = len(valid)
    num_correct = sum(1 for s in valid if s > 0)
    num_failed = total - num_valid

    metrics[f"{prefix}/accuracy_strict"] = num_correct / total if total else 0.0
    metrics[f"{prefix}/accuracy_lenient"] = num_correct / num_valid if num_valid else 0.0
    metrics[f"{prefix}/num_questions"] = float(len(scores))
    metrics[f"{prefix}/num_generated_answers"] = float(total)
    metrics[f"{prefix}/num_valid_answers"] = float(num_valid)
    metrics[f"{prefix}/num_invalid_answers"] = float(num_failed)
    metrics[f"{prefix}/frac_valid_answers"] = num_valid / total if total else 0.0

    # Fraction of questions with diverse scores
    num_diverse = 0
    num_questions = len(scores)
    for q_scores in scores.values():
        resolved = [s if s is not None else 0.0 for s in q_scores]
        if resolved and min(resolved) != max(resolved):
            num_diverse += 1
    metrics[f"{prefix}/frac_diverse_questions"] = (
        num_diverse / num_questions if num_questions else 0.0
    )

    # --- Pass@k ---
    k_values = set()
    k = 1
    while k <= rollout_n:
        k_values.add(k)
        k *= 2
    k_values.add(rollout_n)

    for k_val in sorted(k_values):
        strict_scores_per_q: list[float] = []
        lenient_scores_per_q: list[float] = []

        for qid, q_scores in scores.items():
            # Strict
            n_strict = len(q_scores)
            c_strict = sum(1 for s in q_scores if s is not None and s > 0)
            pak = compute_pass_at_k(n_strict, c_strict, k_val)
            if pak is not None:
                strict_scores_per_q.append(pak)

            # Lenient
            valid_q = [s for s in q_scores if s is not None]
            n_lenient = len(valid_q)
            c_lenient = sum(1 for s in valid_q if s > 0)
            pak_len = compute_pass_at_k(n_lenient, c_lenient, k_val)
            if pak_len is not None:
                lenient_scores_per_q.append(pak_len)

        metrics[f"{prefix}/pass_at_{k_val}_strict"] = (
            float(np.mean(strict_scores_per_q)) if strict_scores_per_q else 0.0
        )
        metrics[f"{prefix}/pass_at_{k_val}_lenient"] = (
            float(np.mean(lenient_scores_per_q)) if lenient_scores_per_q else 0.0
        )

    return metrics


# ---------------------------------------------------------------------------
# CLI: cancel batches
# ---------------------------------------------------------------------------

def cancel_batches_cli() -> None:
    """Cancel all pending batches for a gpt_replay run.

    Usage::

        python -m verl_inf_evolve.gen_eval.api_solver cancel \\
            --cache-dir .cache/gen_eval

    Or cancel a single checkpoint::

        python -m verl_inf_evolve.gen_eval.api_solver cancel \\
            --cache-dir .cache/gen_eval --checkpoint 50
    """
    import argparse

    parser = argparse.ArgumentParser(description="Cancel OpenAI batch jobs")
    sub = parser.add_subparsers(dest="command")
    cancel_parser = sub.add_parser("cancel", help="Cancel pending batches")
    cancel_parser.add_argument(
        "--cache-dir",
        required=True,
        help="Local cache dir (gen_eval.local_cache_dir)",
    )
    cancel_parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Cancel a single checkpoint (ans_loop index). "
        "If omitted, cancels all.",
    )
    cancel_parser.add_argument(
        "--api-key-env-var",
        default="OPENAI_API_KEY",
        help="Env var for API key (default: OPENAI_API_KEY)",
    )

    args = parser.parse_args()
    if args.command != "cancel":
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from omegaconf import OmegaConf

    config = OmegaConf.create({"api_key_env_var": args.api_key_env_var})
    client = APISolverClient(config)

    cache_path = Path(args.cache_dir)
    if args.checkpoint is not None:
        output_dir = cache_path / "batch" / str(args.checkpoint)
        print(client.cancel_batch(output_dir))
    else:
        for msg in client.cancel_all_batches(cache_path):
            print(msg)


if __name__ == "__main__":
    cancel_batches_cli()
