"""Lightweight vLLM evaluation runtime — no Ray, no verl, no DataProto.

Provides:
- ``VllmEvalRuntime``: DP evaluation across GPUs using ``mp.Process`` per GPU.
- ``score_completions``: Score raw vLLM completions into ``question_results``
  format compatible with ``compute_eval_metrics()``.
- ``evaluate_benchmark_vllm``: End-to-end benchmark evaluation using direct
  vLLM inference (drop-in alternative to ``evaluate_benchmark_questions``).

Usage::

    runtime = VllmEvalRuntime.create("Qwen/Qwen3-8B", n_gpus=8)
    result = evaluate_benchmark_vllm(questions, runtime, tokenizer, ...)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Keys dropped from question_results by the scoring bridge when the caller
# does not need full response text.
_TRAJECTORY_KEYS = {"sampled_answers"}


def _blackwell_cudagraph_kwargs() -> dict[str, Any]:
    """Return Blackwell-specific vLLM ``LLM(...)`` ctor kwargs, else ``{}``.

    Also has the side effect of setting ``VLLM_USE_TRTLLM_ATTENTION=1`` in
    ``os.environ`` on Blackwell hardware (see fix #2 below). Both fixes
    target the same root vLLM 0.11.0 + FlashInfer + Blackwell bug:

        File ".../vllm/v1/attention/backends/flashinfer.py", line 972, in forward
            assert decode_wrapper._sm_scale == self.scale
        AssertionError

    The assert lives in the **non-TRT-LLM** decode path:

        if not attn_metadata.decode_use_trtllm:
            assert decode_wrapper._window_left == self.window_left
            assert decode_wrapper._logits_soft_cap == (...)
            assert decode_wrapper._sm_scale == self.scale  # <-- here

    so it fires whenever vLLM picks FlashInfer's plain (non-TRT-LLM) decode
    wrapper on Blackwell. There are two ways to land in that branch:

    Fix #1 — ``compilation_config={"cudagraph_mode": "PIECEWISE"}``
        vLLM 0.11.0's default ``cudagraph_mode=FULL_AND_PIECEWISE`` triggers
        the assert during the FULL-decode CUDA graph CAPTURE phase (the
        dummy-run forward pass calls the same code path). PIECEWISE wraps
        everything except attention into the graph and skips the FULL phase,
        which is enough to dodge the capture-time assert.

    Fix #2 — ``VLLM_USE_TRTLLM_ATTENTION=1`` env var
        Even with PIECEWISE, real inference asserts when vLLM's auto-detector
        picks the non-TRT-LLM decode path. The auto-detector criterion in
        ``vllm/utils/flashinfer.py::use_trtllm_attention`` is::

            use_trtllm = (num_tokens <= 256 and max_seq_len <= 131072
                          and kv_cache_dtype == "auto")

        For tiny models (Qwen2.5-0.5B in our smoke tests) decode batches
        stay under 256 tokens and TRT-LLM is selected → assert is skipped.
        For real sol_eval workloads (Qwen3-8B with chunked-prefill batches)
        ``num_tokens > 256`` → non-TRT-LLM path → assert fires.

        Forcing ``VLLM_USE_TRTLLM_ATTENTION=1`` makes ``use_trtllm_attention``
        return True unconditionally (after a head-ratio check
        ``num_qo_heads % num_kv_heads == 0`` which all GQA Qwen3 models pass),
        which steers decode through the TRT-LLM path that skips the assert
        entirely.

    On Blackwell vLLM also forces the FlashInfer attention backend by
    default ("Using FlashInfer backend with HND KV cache layout on V1
    engine by default for Blackwell (SM 10.0) GPUs"), so we cannot dodge
    either fix by switching backends. TRT-LLM-on-FlashInfer is the
    Blackwell-tuned path anyway, so forcing it isn't a downgrade.

    Other Hopper clusters hit a
    different code path: vLLM picks the FlashAttention backend by default
    on Hopper, so neither fix applies. This helper is a no-op on
    non-Blackwell hardware and the env var is left untouched.

    When to remove
    --------------
    Both fixes can be deleted once we upgrade ``LLM`` to vLLM 0.11.1+ and
    verify both assertion sites no longer fire for the sol_eval workloads
    we run. Note that vLLM 0.11.1 hard-pins ``torch==2.9.0`` and
    ``flashinfer-python==0.5.2``, so the upgrade is non-trivial — see
    ``docs/grace_sol_eval_setup.md`` for the env-rebuild plan.

    The vLLM project tracks the FULL-graph variant of this bug as
    https://github.com/vllm-project/vllm/issues/27057 ; the inference-time
    variant (fix #2 above) does not have its own issue but the same root
    cause (FlashInfer non-TRT-LLM wrapper carrying a stale ``sm_scale``
    on Blackwell).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        major, _minor = torch.cuda.get_device_capability(0)
    except Exception:  # pragma: no cover - defensive: never fail LLM init
        return {}

    # SM 10.0 and above is Blackwell datacenter (B100/B200/GB200). SM 12.0
    # is Blackwell consumer (RTX 50-series, RTX Pro 6000); the same vLLM
    # FlashInfer code path is selected there too, so apply the workaround
    # for both.
    if major < 10:
        return {}

    # Fix #2: force TRT-LLM attention so the non-TRT-LLM decode path with
    # the broken sm_scale assertion is never selected. setdefault leaves
    # the user override alone if they explicitly set VLLM_USE_TRTLLM_ATTENTION=0.
    os.environ.setdefault("VLLM_USE_TRTLLM_ATTENTION", "1")

    logger.info(
        "Blackwell GPU detected (CC=(%d, %d)); applying vLLM 0.11.0 + FlashInfer "
        "Blackwell workarounds: cudagraph_mode=PIECEWISE and "
        "VLLM_USE_TRTLLM_ATTENTION=1. See "
        "verl_inf_evolve/sol_eval/vllm_runtime.py::_blackwell_cudagraph_kwargs "
        "for the full rationale.",
        major,
        _minor,
    )
    return {"compilation_config": {"cudagraph_mode": "PIECEWISE"}}


def _sampling_params_summary(params: Any) -> dict[str, Any]:
    """Extract a stable, log-friendly subset of SamplingParams."""
    return {
        "n": getattr(params, "n", None),
        "temperature": getattr(params, "temperature", None),
        "top_p": getattr(params, "top_p", None),
        "top_k": getattr(params, "top_k", None),
        "min_p": getattr(params, "min_p", None),
        "max_tokens": getattr(params, "max_tokens", None),
        "min_tokens": getattr(params, "min_tokens", None),
        "repetition_penalty": getattr(params, "repetition_penalty", None),
        "presence_penalty": getattr(params, "presence_penalty", None),
        "frequency_penalty": getattr(params, "frequency_penalty", None),
        "seed": getattr(params, "seed", None),
        "stop_token_ids": getattr(params, "stop_token_ids", None),
    }


def _model_sampling_diff(model_path: str, trust_remote_code: bool = True) -> dict[str, Any]:
    """Return the model-side generation-config diff that vLLM may consult."""
    from vllm.transformers_utils.config import try_get_generation_config

    config = try_get_generation_config(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    return {} if config is None else config.to_diff_dict()


def prepare_vllm_requests(
    messages_list: list[list[dict[str, str]]],
    question_ids: list[Any],
    tokenizer: Any,
    prompt_texts: list[str | None] | None = None,
    assistant_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Serialize chat messages into tokenized vLLM requests.

    When ``assistant_prefix`` is set, its tokens are appended after the
    chat-template / raw-text encoding so the model continues from a fixed
    opener (used to probe basin selection — e.g. force "Alright" on a
    To-solve-converged checkpoint to test whether long-reasoning capability
    is preserved through training).
    """
    prefix_ids: list[int] = []
    if assistant_prefix:
        prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)
    prepared: list[dict[str, Any]] = []
    if prompt_texts is None:
        prompt_texts = [None] * len(messages_list)
    for messages, qid, prompt_text in zip(messages_list, question_ids, prompt_texts):
        if prompt_text is not None:
            prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            prompt_token_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
        if prefix_ids:
            prompt_token_ids = list(prompt_token_ids) + list(prefix_ids)
        prepared.append({
            "question_id": str(qid),
            "prompt_token_ids": prompt_token_ids,
        })
    return prepared


# ---------------------------------------------------------------------------
# Single-GPU worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _gpu_worker(
    rank: int,
    model_path: str,
    prompts_file: str,
    output_file: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    eos_token_id: int | None,
    seed: int | None = None,
) -> None:
    """Generate completions for a shard of prompts on a single GPU."""
    # Map ``rank`` onto CUDA_VISIBLE_DEVICES. When the parent process has
    # pinned CVD to a specific subset (e.g. sol_eval running on a shared pod
    # alongside a training job that already owns other GPUs), index into the
    # visible list instead of clobbering it. When CVD is unset, fall back to
    # the historical behavior of binding rank → physical GPU rank.
    _parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _visible = [x.strip() for x in _parent_cvd.split(",") if x.strip()]
    if _visible:
        if rank >= len(_visible):
            raise ValueError(
                f"_gpu_worker rank={rank} exceeds CUDA_VISIBLE_DEVICES="
                f"{_parent_cvd!r} (len={len(_visible)}). Lower "
                f"trainer.n_gpus_per_node or widen CUDA_VISIBLE_DEVICES."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = _visible[rank]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    # Suppress vLLM / torch distributed noise in subprocess
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    # These workers only generate text and write JSON outputs. They should not
    # inherit WandB telemetry state from the parent eval process, which can
    # deadlock under fork/import during vLLM + transformers startup.
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Isolate torchinductor cache per worker to prevent cross-benchmark
    # pickle corruption when sequential vLLM instances load stale compiled
    # artifacts. Each worker gets its own fresh temp dir.
    import tempfile as _tf
    _inductor_dir = _tf.mkdtemp(prefix=f"torchinductor_worker{rank}_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _inductor_dir

    # Give each worker rank its own private VLLM_PORT base so 8
    # concurrently-spawning vLLM workers cannot race each other in vLLM's
    # get_open_port() (which has a TOCTOU window between bind(("",0))/close
    # and TCPStore.bind, see vllm/utils/network_utils.py:177-197). With a
    # distinct starting port per rank, the bind+increment loop is sequential
    # within each worker and the ranks cannot pick the same port at the same
    # time.
    os.environ["VLLM_PORT"] = str(50000 + rank * 100)

    from vllm import LLM, SamplingParams, TokensPrompt

    # Blackwell (B200/B100/GB200) needs PIECEWISE-only CUDA graphs on
    # vLLM 0.11.0; see _blackwell_cudagraph_kwargs() above for the
    # full bug writeup. No-op on Hopper/Ampere/older.
    blackwell_kwargs = _blackwell_cudagraph_kwargs() if not enforce_eager else {}

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        **blackwell_kwargs,
    )
    params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
        seed=seed,
    )
    model_config = getattr(llm, "model_config", None)
    model_default_diff = (
        model_config.get_diff_sampling_param()
        if model_config is not None and hasattr(model_config, "get_diff_sampling_param")
        else {}
    )
    print(
        "Direct vLLM worker "
        f"{rank} model sampling defaults: "
        f"generation_config={getattr(model_config, 'generation_config', None)} "
        f"diff={model_default_diff}",
        flush=True,
    )
    print(
        f"Direct vLLM worker {rank} requested SamplingParams: "
        f"{_sampling_params_summary(params)}",
        flush=True,
    )

    with open(prompts_file, "r") as f:
        shard = json.load(f)

    prompts = [TokensPrompt(prompt_token_ids=item["prompt_token_ids"]) for item in shard]
    # When ``seed`` is provided, build per-prompt SamplingParams with a
    # seed derived from (base_seed, question_id). This keeps outputs
    # reproducible even if prompt order inside a shard changes, and gives
    # every sample of the n=... fan-out its own RNG stream within the
    # shared base.
    if seed is not None:
        from verl_inf_evolve.utils.seeding import derive_sampling_seed
        per_prompt_params = []
        for item in shard:
            qid_hash = int.from_bytes(
                str(item.get("question_id", "")).encode("utf-8")[:8].ljust(8, b"\0"),
                "big",
                signed=False,
            )
            per_prompt_params.append(
                SamplingParams(
                    n=n_samples,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
                    seed=derive_sampling_seed(int(seed), qid_hash),
                )
            )
        outputs = llm.generate(prompts, per_prompt_params)
    else:
        outputs = llm.generate(prompts, params)

    results = []
    for item, vllm_out in zip(shard, outputs):
        completions = []
        for comp in vllm_out.outputs:
            completions.append({
                "text": comp.text,
                "token_len": len(comp.token_ids),
                "finish_reason": comp.finish_reason,
            })
        results.append({
            "question_id": item["question_id"],
            "completions": completions,
        })

    with open(output_file, "w") as f:
        json.dump(results, f)


# ---------------------------------------------------------------------------
# VllmEvalRuntime
# ---------------------------------------------------------------------------

@dataclass
class VllmEvalRuntime:
    """Lightweight DP evaluation runtime using direct vLLM (no Ray).

    Each ``generate()`` call spawns ``n_gpus`` worker processes, each loading
    the model independently on its own GPU.  Workers are ephemeral — there is
    no persistent vLLM process between calls, which makes checkpoint switching
    trivial (just update ``model_path``).
    """

    model_path: str
    n_gpus: int
    tokenizer: Any
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.5
    enforce_eager: bool = False
    seed: int | None = None
    _custom_chat_template: str | None = field(default=None, repr=False)

    @property
    def custom_chat_template(self) -> str | None:
        """Path to the custom chat template file, if configured."""
        return self._custom_chat_template

    @classmethod
    def create(
        cls,
        model_path: str,
        n_gpus: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.5,
        enforce_eager: bool = False,
        custom_chat_template: str | None = None,
        seed: int | None = None,
    ) -> "VllmEvalRuntime":
        """Create a runtime, loading the tokenizer eagerly."""
        import torch
        from transformers import AutoTokenizer

        if n_gpus is None:
            n_gpus = torch.cuda.device_count() or 1

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        if custom_chat_template is not None:
            tokenizer.chat_template = custom_chat_template

        return cls(
            model_path=model_path,
            n_gpus=n_gpus,
            tokenizer=tokenizer,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            seed=seed,
            _custom_chat_template=custom_chat_template,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prepared_questions: list[dict],
        n_samples: int,
        max_tokens: int,
    ) -> list[dict]:
        """Run DP generation across GPUs.

        Args:
            prepared_questions: ``[{question_id, prompt_token_ids}]``
            n_samples: Number of completions per prompt (vLLM ``n``).
            max_tokens: Maximum response tokens.

        Returns:
            ``[{question_id, completions: [{text, token_len, finish_reason}]}]``
        """
        import multiprocessing as mp
        from vllm import SamplingParams

        if not prepared_questions:
            return []

        requested_params = SamplingParams(
            n=n_samples,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id]
            if self.tokenizer.eos_token_id is not None
            else None,
        )
        logger.warning(
            "Direct vLLM parent model sampling defaults: generation_config=%s diff=%s",
            "auto",
            _model_sampling_diff(str(self.model_path), trust_remote_code=True),
        )
        logger.warning(
            "Direct vLLM parent requested SamplingParams: %s",
            _sampling_params_summary(requested_params),
        )

        # Shard round-robin across GPUs
        effective_gpus = min(self.n_gpus, len(prepared_questions))
        shards: list[list[dict]] = [[] for _ in range(effective_gpus)]
        for i, q in enumerate(prepared_questions):
            shards[i % effective_gpus].append(q)

        tmp_dir = tempfile.mkdtemp(prefix="sol_eval_vllm_")
        try:
            processes = []
            output_files = []

            # The parent process may already have imported wandb/transformers.
            # Use `spawn` so worker interpreters start clean instead of inheriting
            # forked import/telemetry state that can hang during vLLM startup.
            mp_ctx = mp.get_context("spawn")

            for rank in range(effective_gpus):
                sf = os.path.join(tmp_dir, f"shard_{rank}.json")
                of = os.path.join(tmp_dir, f"result_{rank}.json")
                with open(sf, "w") as f:
                    json.dump(shards[rank], f)
                output_files.append(of)

                # Give each worker its own deterministic seed (base +
                # rank) so shards differ reproducibly.
                worker_seed = None if self.seed is None else int(self.seed) + rank
                p = mp_ctx.Process(
                    target=_gpu_worker,
                    args=(
                        rank,
                        self.model_path,
                        sf,
                        of,
                        n_samples,
                        self.temperature,
                        self.top_p,
                        self.top_k,
                        max_tokens,
                        self.max_model_len,
                        self.gpu_memory_utilization,
                        self.enforce_eager,
                        self.tokenizer.eos_token_id,
                        worker_seed,
                    ),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # Check for worker failures
            for i, p in enumerate(processes):
                if p.exitcode != 0:
                    raise RuntimeError(
                        f"vLLM worker {i} failed with exit code {p.exitcode}"
                    )

            # Merge results
            all_results: list[dict] = []
            for of in output_files:
                with open(of, "r") as f:
                    all_results.extend(json.load(f))

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return all_results

    # ------------------------------------------------------------------
    # Checkpoint switching
    # ------------------------------------------------------------------

    def update_model(self, new_model_path: str) -> None:
        """Point the runtime at a new checkpoint.

        Workers are ephemeral, so this just stores the new path for the
        next ``generate()`` call.
        """
        self.model_path = new_model_path

    def shutdown(self) -> None:
        """No-op — no persistent resources to release."""
        pass


# ---------------------------------------------------------------------------
# Scoring bridge: vLLM output -> question_results
# ---------------------------------------------------------------------------

def score_completions(
    raw_results: list[dict],
    questions: list[dict],
    *,
    allow_code_execution: bool = False,
    mcq_choice_shuffle_config: dict[str, Any] | None = None,
) -> list[dict]:
    """Score raw vLLM completions into ``question_results`` format.

    Args:
        raw_results: ``[{question_id, completions: [{text, token_len, ...}]}]``
        questions: Original benchmark question dicts (for ground_truth, etc.).

    Returns:
        ``question_results`` list compatible with ``compute_eval_metrics()``.
    """
    from verl_inf_evolve.sol_eval.benchmark_adapters import (
        score_response_for_question,
    )
    from verl_inf_evolve.utils.benchmarks.benchmark_scorers import get_scorer
    from verl_inf_evolve.utils.benchmarks.verifiers import _llm_math_judge

    q_lookup = {str(q["question_id"]): q for q in questions}
    question_results: list[dict] = []

    for item in raw_results:
        qid = str(item["question_id"])
        q = q_lookup.get(qid, {})
        gt = str(q.get("ground_truth", ""))
        data_source = q.get("data_source", "")
        has_custom_scorer = get_scorer(data_source) is not None

        sampled_answers: list[str] = []
        extracted_answers: list[str | None] = []
        answer_scores: list[float | None] = []
        response_token_lengths: list[int] = []
        sample_scores: list[float | None] = []
        sample_score_name: str | None = None
        sample_score_scale_max: float | None = None
        sample_exec_results: list[dict[str, Any] | None] = []

        for comp in item["completions"]:
            text = comp["text"]
            sampled_answers.append(text)
            response_token_lengths.append(comp.get("token_len", 0))

            scored = score_response_for_question(
                q,
                text,
                allow_code_execution=allow_code_execution,
                mcq_choice_shuffle_config=mcq_choice_shuffle_config,
            )
            extracted_answers.append(scored.extracted_answer)
            answer_scores.append(scored.answer_score)
            sample_exec_results.append(scored.exec_result)

            if has_custom_scorer:
                sample_scores.append(scored.primary_score)
                if sample_score_name is None and scored.primary_score_name is not None:
                    sample_score_name = scored.primary_score_name
                if (
                    sample_score_scale_max is None
                    and scored.primary_score_scale_max is not None
                ):
                    sample_score_scale_max = scored.primary_score_scale_max

        question_result: dict[str, Any] = {
            "question_id": qid,
            "question_text": q.get("question_text", ""),
            "choices": q.get("choices", []),
            "ground_truth": gt,
            "data_source": data_source,
            # Preserve sub-benchmark group key (e.g. OlympiadBench_Math vs
            # OlympiadBench_Physics) so compute_sub_bench_metrics can split
            # aggregate scores by domain. Empty string when the upstream
            # question dict doesn't carry a domain.
            "domain": q.get("domain", ""),
            "sampled_answers": sampled_answers,
            "extracted_answers": extracted_answers,
            "answer_scores": answer_scores,
            "response_token_lengths": response_token_lengths,
        }
        if sample_scores:
            question_result["sample_scores"] = sample_scores
            if sample_score_name is not None:
                question_result["sample_score_name"] = sample_score_name
            if sample_score_scale_max is not None:
                question_result["sample_score_scale_max"] = sample_score_scale_max
        if any(item is not None for item in sample_exec_results):
            question_result["sample_exec_results"] = sample_exec_results

        question_results.append(question_result)

    # Optional LLM-judge cascade upgrade for rule-failed samples on eligible
    # benchmarks (MATH500, Minerva, etc.). Dispatches concurrent gpt-5-mini
    # calls via ThreadPoolExecutor so the added wall time is minutes instead
    # of tens of minutes on sampled runs. Fail-soft: judge errors leave the
    # rule-based score unchanged.
    _llm_math_judge.apply_to_question_results(question_results)

    return question_results


# ---------------------------------------------------------------------------
# End-to-end benchmark evaluation via vLLM
# ---------------------------------------------------------------------------

def _save_raw_rollout(
    raw_results: list[dict],
    questions: list[dict],
    *,
    rollout_cache_dir: str | None = None,
) -> None:
    """Save raw vLLM generation output to disk before verification.

    If verification hangs or crashes, the saved rollout can be re-scored
    offline without re-generating.  Saved as ``rollout_<benchmark>.jsonl``
    in ``rollout_cache_dir``.
    """
    if not rollout_cache_dir:
        return
    try:
        import json
        os.makedirs(rollout_cache_dir, exist_ok=True)
        # Infer benchmark name from questions
        benchmark = "unknown"
        if questions:
            benchmark = questions[0].get("data_source", "unknown")
        path = os.path.join(rollout_cache_dir, f"rollout_{benchmark}.jsonl")
        with open(path, "w") as f:
            for item in raw_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Saved raw rollout (%d questions) to %s", len(raw_results), path)
    except Exception as e:
        logger.warning("Failed to save raw rollout: %s", e)


def evaluate_benchmark_vllm(
    questions: list[dict],
    tokenizer: Any,
    max_prompt_tokens: int,
    n_samples: int,
    vllm_runtime: VllmEvalRuntime,
    max_response_tokens: int = 8192,
    use_public_eval_prompt: Any = False,
    raise_if_all_filtered: bool = True,
    code_execution_enabled: bool = False,
    execution_scope: str = "standalone",
    allow_code_execution_in_training: bool = False,
    model_path: str = "",
    rollout_cache_dir: str | None = None,
    mcq_choice_shuffle_config: dict[str, Any] | None = None,
    benchmark_prompts: dict[str, Any] | None = None,
    assistant_prefix: str | None = None,
) -> "BenchmarkEvaluationResult | None":
    """Run one benchmark end-to-end via direct vLLM (no Ray/DataProto).

    Drop-in alternative to ``evaluate_benchmark_questions`` for standalone
    evaluation.

    Args:
        questions: Benchmark question rows.
        tokenizer: Solver tokenizer.
        max_prompt_tokens: Prompt length filter.
        n_samples: Rollout samples per question.
        vllm_runtime: ``VllmEvalRuntime`` instance.
        max_response_tokens: Maximum response token length for vLLM.
        use_public_eval_prompt: Use public-eval-aligned prompt per benchmark.
        raise_if_all_filtered: Raise when all questions are filtered.

    Returns:
        ``BenchmarkEvaluationResult`` or ``None`` if all questions filtered.
    """
    from verl_inf_evolve.sol_eval.eval_core import (
        BenchmarkEvaluationResult,
        build_benchmark_messages,
        compute_sub_bench_metrics,
        resolve_code_execution_enabled,
        validate_code_execution_policy,
    )
    from verl_inf_evolve.sol_eval.result_format import compute_eval_metrics

    validate_code_execution_policy(
        questions,
        code_execution_enabled=code_execution_enabled,
        execution_scope=execution_scope,
        allow_code_execution_in_training=allow_code_execution_in_training,
    )
    effective_code_execution_enabled = resolve_code_execution_enabled(
        questions,
        code_execution_enabled=code_execution_enabled,
        execution_scope=execution_scope,
    )

    msg_data = build_benchmark_messages(
        questions=questions,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        use_public_eval_prompt=use_public_eval_prompt,
        model_path=model_path,
        mcq_choice_shuffle_config=mcq_choice_shuffle_config,
        benchmark_prompts=benchmark_prompts,
    )
    if not msg_data.messages_list:
        message = "All questions were filtered out (too long)"
        if raise_if_all_filtered:
            raise ValueError(message)
        logger.warning(message)
        return None

    # Convert messages -> tokenized prompts via chat template.
    prepared = prepare_vllm_requests(
        msg_data.messages_list,
        msg_data.question_ids,
        tokenizer,
        prompt_texts=msg_data.prompt_texts,
        assistant_prefix=assistant_prefix,
    )

    if assistant_prefix:
        logger.info(
            "Forcing assistant_prefix=%r (tokens appended after add_generation_prompt)",
            assistant_prefix,
        )
    logger.info(
        "Generating %d questions × %d samples on %d GPUs via vLLM",
        len(prepared), n_samples, vllm_runtime.n_gpus,
    )
    raw_results = vllm_runtime.generate(
        prepared_questions=prepared,
        n_samples=n_samples,
        max_tokens=max_response_tokens,
    )

    # Save raw generation output before verification — if the verifier hangs,
    # these can be re-scored offline without re-generating.
    _save_raw_rollout(raw_results, questions, rollout_cache_dir=rollout_cache_dir)

    question_results = score_completions(
        raw_results,
        questions,
        allow_code_execution=effective_code_execution_enabled,
        mcq_choice_shuffle_config=mcq_choice_shuffle_config,
    )
    metrics = compute_eval_metrics(question_results, n_samples=n_samples)

    # Mirror evaluate_benchmark_questions (eval_core.py:514-516): stash the
    # per-domain breakdown onto the metrics dict so the wandb logger can emit
    # olympiadbench/sub_bench/{math,physics}/* keys under result_detail=metrics_only
    # (which drops question_results before the logger sees them).
    sub_bench_metrics = compute_sub_bench_metrics(question_results, n_samples=n_samples)
    if sub_bench_metrics:
        metrics["sub_bench_metrics"] = sub_bench_metrics

    logger.info(
        "Evaluation complete: %d questions, accuracy_strict=%.4f, accuracy_lenient=%.4f",
        metrics["total_questions"],
        metrics["accuracy_strict"],
        metrics["accuracy_lenient"],
    )

    return BenchmarkEvaluationResult(
        output=None,  # no DataProto in vLLM path
        question_results=question_results,
        metrics=metrics,
    )
