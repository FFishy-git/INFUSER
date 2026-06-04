"""
SelfEvolutionActorRolloutRefWorker — extended FSDP hybrid engine worker.

Extends verl's ``AsyncActorRolloutRefWorker`` (fsdp_workers.py:1988) with
gradient-based influence scoring: Phase 1 (dev gradient + momentum EMA)
and Phase 2 (per-question cosine similarity).

Most functionality comes from the parent class.  This file adds:
  - ``compute_dev_gradient()``: Phase 1 — forward+backward on dev data,
    update momentum.
  - ``compute_similarity()``: Phase 2 — per-question gradient similarity
    against the momentum buffer.

Both methods implement the forward+backward loop directly rather than
calling ``DataParallelPPOActor.update_policy()``, because verl's native
``update_policy()`` does not support ``force_on_policy`` or
``skip_optimizer_and_retain_grad`` (those were V2-specific patches in
``inf_evolve/grad_utils/patched_dp_actor.py``).

Reference: v2 ``verl_joint_dev_similarity.py:786-1062``
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
from typing import Any, Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
import verl.workers.fsdp_workers as fsdp_workers
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from verl_inf_evolve.workers.hf_checkpoint_utils import (
    augment_hf_state_dict_for_tied_embeddings,
)
from verl_inf_evolve.workers.similarity_optimizer import SimilarityComputingOptimizer

logger = logging.getLogger(__name__)

# In shared-pool mode (generator + solver colocated), each process should own
# exactly one vLLM rollout engine for a compatible rollout signature.
_SHARED_VLLM_ROLLOUTS: dict[tuple[int, int, str], Any] = {}


def _ensure_cuda_detected():
    """Fix verl's module-level CUDA detection in Ray actor subprocesses.

    verl/utils/device.py caches ``is_cuda_available = torch.cuda.is_available()``
    at import time.  In Ray actors, ``torch.cuda.is_available()`` can return
    False due to CUDA initialization race conditions even though GPUs are
    allocated (CUDA_VISIBLE_DEVICES is set and ``device_count() > 0``).

    This function forces CUDA reinitialization and patches the cached variable.
    """
    import verl.utils.device as device_mod

    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_avail = torch.cuda.is_available()
    print(
        f"[CUDA-DIAG] CUDA_VISIBLE_DEVICES={cuda_vis!r}, "
        f"torch.cuda.is_available()={cuda_avail}, "
        f"device.is_cuda_available={device_mod.is_cuda_available}"
    )

    if device_mod.is_cuda_available:
        return

    # CUDA_VISIBLE_DEVICES is set but torch.cuda.is_available() returned False.
    # This can happen due to CUDA init race in multi-process Ray actors.
    # Force reinit: device_count() often succeeds when is_available() doesn't.
    if cuda_vis and cuda_vis != "-1":
        try:
            count = torch.cuda.device_count()
            if count > 0:
                # CUDA runtime works — force torch to recognize it
                torch.cuda.init()
                device_mod.is_cuda_available = True
                print(f"[CUDA-DIAG] Forced CUDA init: device_count={count}, patched is_cuda_available=True")
            else:
                print(f"[CUDA-DIAG] device_count=0, cannot force CUDA")
        except Exception as e:
            # Last resort: if CUDA_VISIBLE_DEVICES is set, trust the allocator
            print(f"[CUDA-DIAG] CUDA init attempt failed ({e}), forcing is_cuda_available=True anyway")
            device_mod.is_cuda_available = True
    elif cuda_avail:
        device_mod.is_cuda_available = True
        print("[CUDA-DIAG] Patched is_cuda_available=True from torch check")


class SelfEvolutionActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """ActorRolloutRefWorker with influence-score computation capability.

    Inherits the full hybrid engine (FSDP actor + vLLM rollout + optional
    reference policy).  Adds gradient-based influence scoring that reuses
    the resident FSDP model — no subprocess or model re-initialization.

    State attributes:
        momentum: List of per-parameter gradient momentum tensors (one per
            FSDP parameter shard).  Persisted across gen_loops within an
            ans_loop for EMA accumulation.
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        _ensure_cuda_detected()
        super().__init__(config, role, **kwargs)
        # Per-worker reseed: user seed + dist rank so workers diverge
        # deterministically. ``seed`` on the rollout config is already
        # resolved to ``training.seed`` via the self_evolution.yaml
        # interpolation, so both roles share the same base.
        from verl_inf_evolve.utils.seeding import seed_all

        base_seed = int((self.config.rollout.get("seed", 42) if self.config.rollout else 42) or 42)
        seed_all(base_seed, rank=int(getattr(self, "rank", 0)))
        self.momentum: Optional[list[torch.Tensor | None]] = None
        self._warned_zero_init_similarity_state = False

    def _build_rollout_signature(self, model_config: Any) -> str:
        """Build a stable signature for rollout-engine compatibility."""
        tokenizer = getattr(self, "tokenizer", None)
        signature_payload = {
            "rollout_name": self.config.rollout.name,
            "rollout_mode": self.config.rollout.get("mode", None),
            "tensor_model_parallel_size": self.config.rollout.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.config.rollout.pipeline_model_parallel_size,
            "data_parallel_size": self.config.rollout.data_parallel_size,
            "prompt_length": self.config.rollout.prompt_length,
            "response_length": self.config.rollout.response_length,
            "temperature": self.config.rollout.get("temperature", None),
            "top_k": self.config.rollout.get("top_k", None),
            "top_p": self.config.rollout.get("top_p", None),
            "gpu_memory_utilization": self.config.rollout.get("gpu_memory_utilization", None),
            "load_format": self.config.rollout.get("load_format", None),
            "model_path": self.config.model.path,
            "model_dtype": str(getattr(model_config, "torch_dtype", None)),
            "tokenizer_name_or_path": self.config.model.path,
            "chat_template": getattr(tokenizer, "chat_template", None),
        }
        payload = json.dumps(signature_payload, sort_keys=True, default=str)
        sig = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        # Diagnostic: log the full payload (without chat_template which is huge)
        # to help debug rollout-sharing failures across roles.
        diag = {k: v for k, v in signature_payload.items() if k != "chat_template"}
        diag["chat_template_hash"] = (
            hashlib.sha256(signature_payload["chat_template"].encode()).hexdigest()[:12]
            if signature_payload["chat_template"]
            else None
        )
        logger.info(
            "Rollout signature: role=%s pid=%s rank=%s sig=%s payload=%s",
            getattr(self, "role", "?"),
            os.getpid(),
            dist.get_rank() if dist.is_initialized() else "N/A",
            sig[:12],
            json.dumps(diag, sort_keys=True, default=str),
        )
        return sig

    def _build_rollout(self, trust_remote_code=False):
        """Build or reuse a shared rollout engine inside the local process."""
        rollout_config: fsdp_workers.RolloutConfig = fsdp_workers.omega_conf_to_dataclass(self.config.rollout)
        model_config: fsdp_workers.HFModelConfig = fsdp_workers.omega_conf_to_dataclass(
            self.config.model,
            dataclass_type=fsdp_workers.HFModelConfig,
        )
        self.model_config = model_config

        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = fsdp_workers.init_device_mesh(
            fsdp_workers.device_name,
            mesh_shape=(dp, infer_tp, infer_pp),
            mesh_dim_names=["dp", "infer_tp", "infer_pp"],
        )
        rollout_name = self.config.rollout.name
        self.rollout_device_mesh = rollout_device_mesh

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout",
                dp_rank=rollout_device_mesh["dp"].get_local_rank(),
                is_collect=is_collect,
            )

        # Seed the per-rank gen-RNG deterministically from the user seed
        # + dp rank. The torch RNG context switching is a verl framework
        # mechanism for hybrid engines that share a GPU between training
        # and generation; with vLLM rollout + dropout disabled the
        # gen_random_states value has no observable effect, but we still
        # derive it from the user seed so nothing in the RL loop secretly
        # depends on the hardcoded ``rank + 1000`` anchor.
        self.torch_random_states = fsdp_workers.get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        base_seed = int((self.config.rollout.get("seed", 42) if self.config.rollout else 42) or 42)
        fsdp_workers.get_torch_device().manual_seed(base_seed + gen_dp_rank)
        self.gen_random_states = fsdp_workers.get_torch_device().get_rng_state()
        fsdp_workers.get_torch_device().set_rng_state(self.torch_random_states)

        rollout_signature = self._build_rollout_signature(model_config)
        shared_key = (os.getpid(), dist.get_rank(), rollout_signature)
        rollout = _SHARED_VLLM_ROLLOUTS.get(shared_key)
        if rollout is None:
            fsdp_workers.log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=fsdp_workers.logger)
            rollout = fsdp_workers.get_rollout_class(rollout_config.name, rollout_config.mode)(
                config=rollout_config,
                model_config=model_config,
                device_mesh=rollout_device_mesh,
            )
            _SHARED_VLLM_ROLLOUTS[shared_key] = rollout
            fsdp_workers.log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=fsdp_workers.logger)
            logger.info(
                "Created shared rollout pid=%s rank=%s signature=%s",
                shared_key[0],
                shared_key[1],
                rollout_signature[:12],
            )
        else:
            logger.info(
                "Reusing shared rollout pid=%s rank=%s signature=%s",
                shared_key[0],
                shared_key[1],
                rollout_signature[:12],
            )

        self.rollout = rollout

        if dist.get_world_size() == 1 and fsdp_workers.fsdp_version(self.actor_module_fsdp) == 1:
            fsdp_workers.FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=fsdp_workers.StateDictType.FULL_STATE_DICT,
                state_dict_config=fsdp_workers.FullStateDictConfig(),
            )
        elif fsdp_workers.fsdp_version(self.actor_module_fsdp) == 1:
            fsdp_workers.FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=fsdp_workers.StateDictType.SHARDED_STATE_DICT,
                state_dict_config=fsdp_workers.ShardedStateDictConfig(),
            )

        self.base_sync_done = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_rollout_object_id(self) -> int:
        """Return the Python object id of ``self.rollout`` for sharing verification."""
        return id(getattr(self, "rollout", None))

    # ------------------------------------------------------------------
    # Checkpoint saving with configurable HF dtype
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """Save checkpoint, optionally converting HF weights to a target dtype.

        When ``actor.checkpoint.hf_save_dtype`` is set (e.g. ``"bf16"``), the
        HF model weights are converted before saving, halving checkpoint size.
        When unset or ``null``, falls through to the base implementation (fp32).
        """
        hf_save_dtype = self.config.actor.checkpoint.get("hf_save_dtype", None)
        if hf_save_dtype is None:
            return super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

        from verl.utils.torch_dtypes import PrecisionType

        save_dtype = PrecisionType.to_dtype(hf_save_dtype)

        # Temporarily disable HF model saving in the checkpoint manager so the
        # base class saves everything *except* the HF weights.
        cm = self.checkpoint_manager
        orig_contents = cm.checkpoint_save_contents
        cm.checkpoint_save_contents = [c for c in orig_contents if c != "hf_model"]
        try:
            super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        finally:
            cm.checkpoint_save_contents = orig_contents

        # Now save HF model weights with dtype conversion.
        if "hf_model" in orig_contents:
            self._save_hf_model_with_dtype(local_path, save_dtype)

    def _save_hf_model_with_dtype(self, local_path: str, save_dtype: torch.dtype) -> None:
        """Gather FSDP state dict, convert to *save_dtype*, and write HF checkpoint."""
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM, GenerationConfig

        from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict

        import gc

        def _log_hf_save_diag(label: str) -> None:
            parts = [f"rank={dist.get_rank()}"]
            try:
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    free_gpu, _ = torch.cuda.mem_get_info()
                    parts.append(
                        f"gpu: alloc={alloc:.1f}G reserved={reserved:.1f}G free={free_gpu / 1024**3:.1f}G"
                    )
            except Exception:
                pass
            try:
                import psutil

                proc = psutil.Process()
                rss = proc.memory_info().rss / 1024**3
                sys_mem = psutil.virtual_memory()
                parts.append(f"proc_rss={rss:.1f}G")
                parts.append(f"sys_mem={sys_mem.used / 1024**3:.1f}G/{sys_mem.total / 1024**3:.1f}G")
            except Exception:
                pass
            logger.info("[save_checkpoint %s] %s", label, ", ".join(parts))

        gc.collect()
        torch.cuda.empty_cache()
        _log_hf_save_diag("hf-pre-gather")

        model = self.actor_module_fsdp
        gather_t0 = time.time()
        state_dict = get_fsdp_full_state_dict(model, offload_to_cpu=True, rank0_only=True)
        logger.info(
            "[save_checkpoint] Gathered full HF state dict in %.1fs on rank=%d",
            time.time() - gather_t0,
            dist.get_rank(),
        )
        _log_hf_save_diag("hf-post-gather")

        if dist.get_rank() == 0:
            convert_t0 = time.time()
            # Convert in-place to avoid holding both fp32 and bf16 copies of
            # the whole checkpoint on rank 0 at the same time.
            for key in list(state_dict.keys()):
                value = state_dict[key]
                if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != save_dtype:
                    state_dict[key] = value.to(save_dtype)
            gc.collect()
            logger.info(
                "[save_checkpoint] Converted HF state dict to %s in %.1fs",
                save_dtype,
                time.time() - convert_t0,
            )
            _log_hf_save_diag("hf-rank0-post-convert")

            unwrap = model._fsdp_wrapped_module if fsdp_version(model) == 1 else model
            model_config = unwrap.config
            hf_path = os.path.join(local_path, "huggingface")
            os.makedirs(hf_path, exist_ok=True)

            with init_empty_weights():
                save_model = AutoModelForCausalLM.from_config(model_config, torch_dtype=save_dtype)
            save_model.to_empty(device="cpu")
            save_model.config.torch_dtype = save_dtype

            # Attach generation config if available.
            if save_model.can_generate():
                name_or_path = getattr(model_config, "name_or_path", "")
                if name_or_path:
                    try:
                        gen_cfg = GenerationConfig.from_pretrained(name_or_path)
                        # Some models (e.g. OLMo) ship with sampling params
                        # (temperature, top_p) but do_sample=False, which fails
                        # validation on save.  Fix by enabling do_sample when
                        # sampling params are present.
                        has_sampling = (
                            getattr(gen_cfg, "temperature", 1.0) != 1.0
                            or getattr(gen_cfg, "top_p", 1.0) != 1.0
                            or getattr(gen_cfg, "top_k", 50) != 50
                        )
                        if has_sampling and not getattr(gen_cfg, "do_sample", False):
                            gen_cfg.do_sample = True
                        save_model.generation_config = gen_cfg
                    except Exception:
                        pass

            save_t0 = time.time()
            _log_hf_save_diag("hf-rank0-pre-write")
            save_model.save_pretrained(hf_path, state_dict=state_dict)
            logger.info(
                "[save_checkpoint] Saved HF model in %s to %s in %.1fs",
                save_dtype, os.path.abspath(hf_path),
                time.time() - save_t0,
            )
            del save_model

        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        _log_hf_save_diag("hf-pre-barrier")
        dist.barrier()
        _log_hf_save_diag("hf-post-barrier")

    # ------------------------------------------------------------------
    # HuggingFace checkpoint loading (for eval pipeline)
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_hf_checkpoint(self, local_path):
        """Load HuggingFace safetensors checkpoint into the FSDP model.

        Used by the eval pipeline when checkpoints are in HuggingFace format
        (model.safetensors) rather than FSDP sharded format.

        Args:
            local_path: Path to HuggingFace checkpoint directory containing
                model.safetensors (or model.safetensors.index.json + shards).
        """
        from safetensors.torch import load_file
        from verl.utils.fsdp_utils import fsdp2_load_full_state_dict

        if local_path is None:
            if self._is_offload_param:
                fsdp_workers.offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            return

        state_dict = None
        if self._is_offload_param:
            fsdp_workers.load_fsdp_model_to_gpu(self.actor_module_fsdp)

        try:
            load_error: str | None = None
            shard_paths: list[str] = []
            try:
                shard_paths = self._resolve_hf_safetensor_files(local_path)
                state_dict = {}
                for filepath in shard_paths:
                    state_dict.update(load_file(filepath, device="cpu"))
                logger.info(
                    "Loaded HF checkpoint: %d files, %d params from %s",
                    len(shard_paths), len(state_dict), local_path,
                )
                state_dict = augment_hf_state_dict_for_tied_embeddings(
                    local_path=local_path,
                    state_dict=state_dict,
                    logger=logger,
                )
            except Exception as exc:
                load_error = (
                    f"rank={self._dist_rank()} failed to read HF checkpoint from "
                    f"{local_path}: {exc}"
                )
                logger.exception(load_error)

            gathered_errors = self._gather_hf_load_errors(load_error)
            if gathered_errors:
                raise RuntimeError("; ".join(gathered_errors))

            # Load into FSDP model using the unified set_model_state_dict API
            # (works for both FSDP1 and FSDP2, avoids FSDP1 import issues on newer PyTorch)
            _ckpt_keys = len(state_dict)
            fsdp2_load_full_state_dict(self.actor_module_fsdp, state_dict)

            # Verify load via DTensor-based state_dict (the actual path used
            # by rollout_mode to sync weights to vLLM).  named_parameters()
            # returns stale values after fsdp2_load with device_mesh-backed
            # FSDP, so we must NOT use it for verification.
            try:
                from torch.distributed._tensor import DTensor
                _sd = self.actor_module_fsdp.state_dict()
                _first_key = next(iter(_sd))
                _first_val = _sd[_first_key]
                if isinstance(_first_val, DTensor):
                    _sample = _first_val.full_tensor().float().sum().item()
                    _vtype = "DTensor"
                else:
                    _sample = _first_val.float().sum().item() if hasattr(_first_val, 'float') else None
                    _vtype = type(_first_val).__name__
                logger.info(
                    "[CKPT-DIAG] fsdp2_load done: ckpt_keys=%d, sd_type=%s, "
                    "first_key=%s, sample_sum=%.4f, rank=%s",
                    _ckpt_keys, _vtype, _first_key,
                    _sample if _sample is not None else 0.0,
                    self._dist_rank(),
                )
                del _sd
            except Exception as _diag_err:
                logger.info(
                    "[CKPT-DIAG] fsdp2_load done: ckpt_keys=%d, verify_error=%s, rank=%s",
                    _ckpt_keys, _diag_err, self._dist_rank(),
                )

            logger.info("HF checkpoint loaded successfully from %s", local_path)
        finally:
            if state_dict is not None:
                del state_dict
            gc.collect()
            torch.cuda.empty_cache()

            if self._is_offload_param:
                fsdp_workers.offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    def _resolve_hf_safetensor_files(self, local_path: str) -> list[str]:
        """Return validated safetensors shard paths for an HF checkpoint."""
        index_file = os.path.join(local_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            if not weight_map:
                raise ValueError(f"HF checkpoint index has empty weight_map: {index_file}")
            filenames = sorted(set(weight_map.values()))
        else:
            filenames = ["model.safetensors"]

        shard_paths: list[str] = []
        missing_files: list[str] = []
        empty_files: list[str] = []
        for filename in filenames:
            filepath = os.path.join(local_path, filename)
            if not os.path.isfile(filepath):
                missing_files.append(filename)
                continue
            if os.path.getsize(filepath) <= 0:
                empty_files.append(filename)
                continue
            shard_paths.append(filepath)

        if missing_files:
            raise FileNotFoundError(
                f"HF checkpoint is missing {len(missing_files)} shard file(s): {missing_files}"
            )
        if empty_files:
            raise ValueError(
                f"HF checkpoint has empty shard file(s): {empty_files}"
            )
        return shard_paths

    def _gather_hf_load_errors(self, local_error: str | None) -> list[str]:
        """Gather HF checkpoint read errors from all ranks before FSDP load."""
        if not (dist.is_available() and dist.is_initialized()):
            return [local_error] if local_error else []

        gathered: list[dict[str, str | None]] = [
            {"rank": str(idx), "error": None} for idx in range(dist.get_world_size())
        ]
        dist.all_gather_object(
            gathered,
            {"rank": str(self._dist_rank()), "error": local_error},
        )
        return [
            str(item["error"])
            for item in gathered
            if item.get("error")
        ]

    @staticmethod
    def _dist_rank() -> int:
        """Best-effort distributed rank helper for load diagnostics."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    # ------------------------------------------------------------------
    # Phase 1: Dev Gradient + Momentum
    # ------------------------------------------------------------------

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_dev_gradient(self, dev_data: DataProto) -> DataProto:
        """Phase 1: Forward+backward on dev data, update momentum.

        Accumulates gradients from all dev samples in a single pass
        (one mini-batch, multiple micro-batches), then applies EMA to
        the momentum buffer: ``m = beta * m + (1 - beta) * grad``.

        Gradients are cleared after momentum update to free GPU memory.

        Expected ``meta_info`` keys:
            temperature (float): Forward-pass temperature.
            momentum_beta (float): EMA decay factor (default 0.9).
            micro_batch_size (int, optional): Override micro-batch size.
            reset_momentum (bool, optional): Reset momentum before update.
            rollout_corr_config (dict, optional): Rollout correction config
                (from ``algorithm.rollout_correction``).

        Expected ``batch`` keys:
            input_ids, attention_mask, position_ids, responses,
            response_mask, advantages, rollout_log_probs (optional).

        Args:
            dev_data: DataProto with tokenized dev samples, dispatched
                per-rank by the actor mesh.

        Returns:
            DataProto with ``meta_info={"status": "ok"}``.
        """
        import time as _time

        temperature = dev_data.meta_info["temperature"]
        momentum_beta = dev_data.meta_info.get("momentum_beta", 0.9)
        rollout_corr_config = dev_data.meta_info.get("rollout_corr_config", None)

        timing: dict[str, float] = {}

        t0 = _time.monotonic()
        if dev_data.meta_info.get("reset_momentum", False):
            self.momentum = None
        else:
            self._load_momentum_to_gpu()
        timing["load_momentum"] = _time.monotonic() - t0

        # local_samples = dev_data.batch["input_ids"].shape[0]
        # micro_batch_size = dev_data.meta_info.get(
        #     "micro_batch_size",
        #     self._auto_micro_batch_size(local_samples),
        # )
        # micro_batch_size = 1 # hardcode to 1 for better debugging
        micro_batch_size = self.actor.config.ppo_micro_batch_size_per_gpu

        # log all the meta_info for debugging
        logger.info("compute_dev_gradient meta_info: %s", dev_data.meta_info)

        t0 = _time.monotonic()
        with self.ulysses_sharding_manager:
            dev_data = dev_data.to("cpu")
            metrics = self._policy_forward_backward(
                dev_data, temperature=temperature, micro_batch_size=micro_batch_size,
                rollout_corr_config=rollout_corr_config,
            )
        timing["forward_backward"] = _time.monotonic() - t0

        # Compute cumulative gradient norm (global across FSDP shards)
        local_grad_norm_sq = torch.zeros((), device="cuda", dtype=torch.float32)
        for p in self.actor.actor_module.parameters():
            if p.grad is not None:
                local_grad_norm_sq += (p.grad.float().view(-1) ** 2).sum()
        dist.all_reduce(local_grad_norm_sq, op=dist.ReduceOp.SUM)
        dev_grad_norm = local_grad_norm_sq.sqrt().clamp_min(1e-12)
        metrics["grad_norm"] = dev_grad_norm.item()
        logger.info("compute_dev_gradient grad_norm=%.6f", dev_grad_norm.item())

        # Update momentum from accumulated gradients
        t0 = _time.monotonic()
        self._update_momentum(momentum_beta)
        timing["update_momentum"] = _time.monotonic() - t0

        # Clear gradients to free GPU memory before Phase 2
        t0 = _time.monotonic()
        self._clear_gradients()
        timing["clear_gradients"] = _time.monotonic() - t0

        logger.info(
            "compute_dev_gradient timing: %s",
            {k: f"{v:.2f}s" for k, v in timing.items()},
        )

        metrics.update({f"influence_timing_s/{k}": v for k, v in timing.items()})
        return DataProto(meta_info={"status": "ok", "metrics": metrics})

    # ------------------------------------------------------------------
    # Phase 2: Per-Question Similarity
    # ------------------------------------------------------------------

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_similarity(self, gen_data: DataProto) -> DataProto:
        """Phase 2: Compute per-question gradient similarity with momentum.

        Splits ``gen_data`` into mini-batches (one per question group) and
        computes the cosine similarity between each mini-batch's gradient
        and the momentum buffer from Phase 1.

        Within each mini-batch:
          1. Zero gradients via ``SimilarityComputingOptimizer.zero_grad()``.
          2. Accumulate gradients via micro-batch forward+backward.
          3. Compute similarity via ``SimilarityComputingOptimizer.step()``,
             which performs all-reduce across FSDP ranks.

        Expected ``meta_info`` keys:
            temperature (float): Forward-pass temperature.
            mini_batch_size (int): Samples per mini-batch (= rollout_n / dp_size).
            rollout_corr_config (dict, optional): Rollout correction config
                (from ``algorithm.rollout_correction``).

        Expected ``batch`` keys:
            input_ids, attention_mask, position_ids, responses,
            response_mask, advantages, rollout_log_probs (optional).

        Args:
            gen_data: DataProto with tokenized gen-question answers,
                dispatched per-rank. Data must be ordered so that each
                mini-batch corresponds to one question's answers (scattered
                interleaving handled by the trainer).

        Returns:
            DataProto with similarity metrics in ``meta_info``:
                similarity_metrics (dict): always contains
                ``{score, score_mode, grad_norm, ref_norm, num_minibatches}``
                and may additionally contain mode-specific keys such as
                ``dot``/``cosine`` or ``preconditioned_dot``/
                ``preconditioned_cosine`` plus ``gamma_norm``. All
                per-mini-batch values are Python lists; ``ref_norm`` is a
                scalar float.
        """
        import time as _time

        if self.momentum is None:
            raise RuntimeError(
                "Momentum not computed. Call compute_dev_gradient first."
            )

        timing: dict[str, float] = {}
        memory_metrics: dict[str, float] = {}
        similarity_mode = gen_data.meta_info.get("similarity_mode", "cosine")
        needs_optimizer_state = similarity_mode in {
            "preconditioned_dot",
            "preconditioned_cosine",
        }

        self._reset_cuda_peak_memory_stats()
        t0 = _time.monotonic()
        self._load_momentum_to_gpu()
        timing["load_momentum"] = _time.monotonic() - t0
        memory_metrics.update(self._cuda_memory_metric_snapshot("post_load_momentum"))

        if needs_optimizer_state and self.actor_optimizer is None:
            raise RuntimeError(
                f"similarity_mode={similarity_mode!r} requires actor_optimizer."
            )

        optimizer_loaded = False
        if needs_optimizer_state and self._is_offload_optimizer:
            t0 = _time.monotonic()
            fsdp_workers.load_fsdp_optimizer(
                optimizer=self.actor_optimizer,
                device_id=get_device_id(),
            )
            timing["load_optimizer"] = _time.monotonic() - t0
            optimizer_loaded = True
            memory_metrics.update(self._cuda_memory_metric_snapshot("post_load_optimizer"))

        if (
            needs_optimizer_state
            and self.actor_optimizer is not None
            and not self.actor_optimizer.state
            and not self._warned_zero_init_similarity_state
        ):
            logger.warning(
                "compute_similarity: optimizer state is empty; "
                "using AdamW zero-init semantics for %s.",
                similarity_mode,
            )
            self._warned_zero_init_similarity_state = True

        temperature = gen_data.meta_info["temperature"]
        mini_batch_size = gen_data.meta_info["mini_batch_size"]
        micro_batch_size = self._auto_micro_batch_size(mini_batch_size)
        rollout_corr_config = gen_data.meta_info.get("rollout_corr_config", None)

        # Populate global_batch_info for loss normalization
        if "avg_response_tokens" in gen_data.meta_info:
            dp_size = gen_data.meta_info["dp_size"]
            self.actor.config.global_batch_info["batch_num_tokens"] = (
                gen_data.meta_info["avg_response_tokens"] * micro_batch_size * dp_size
            )
            self.actor.config.global_batch_info["dp_size"] = dp_size

        # Create similarity optimizer with momentum as reference gradients
        sim_opt = SimilarityComputingOptimizer(
            fsdp_model=self.actor.actor_module,
            ref_gradients=self.momentum,
            actor_optimizer=self.actor_optimizer,
            similarity_mode=similarity_mode,
        )

        select_keys = [
            "responses", "response_mask", "input_ids",
            "attention_mask", "position_ids", "advantages",
        ]
        if "rollout_log_probs" in gen_data.batch.keys():
            select_keys.append("rollout_log_probs")

        forward_backward_total = 0.0
        sim_step_total = 0.0

        all_metrics: dict[str, list] = {}
        try:
            t_loop_start = _time.monotonic()
            self._reset_cuda_peak_memory_stats()
            memory_metrics.update(self._cuda_memory_metric_snapshot("loop_start"))
            with self.ulysses_sharding_manager:
                gen_data = gen_data.to("cpu")
                data = gen_data.select(batch_keys=select_keys)
                mini_batches = data.split(mini_batch_size)

                self.actor.actor_module.train()
                pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
                entropy_coeff = self.actor.config.entropy_coeff
                loss_agg_mode_cfg = self.actor.config.loss_agg_mode
                calculate_entropy = self.actor.config.calculate_entropy or (entropy_coeff != 0)

                _total_mini = len(mini_batches)
                _sim_start = _time.monotonic()
                _last_progress_time = _sim_start
                _progress_label = gen_data.meta_info.get(
                    "progress_label", "gen_similarity/compute_similarity"
                )

                for _mini_idx, mini_batch in enumerate(mini_batches):
                    micro_batches = mini_batch.split(micro_batch_size)
                    gradient_accumulation = len(micro_batches)

                    sim_opt.zero_grad()

                    t_fb = _time.monotonic()
                    for micro_batch in micro_batches:
                        micro_batch = micro_batch.to(get_device_id())
                        model_inputs = {
                            **micro_batch.batch,
                            **micro_batch.non_tensor_batch,
                            "pad_token_id": pad_token_id,
                        }

                        entropy, log_prob = self.actor._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )
                        old_log_prob = log_prob.detach()  # on-policy

                        response_mask = model_inputs["response_mask"]

                        # Apply rollout correction (RS + veto + IS) on-the-fly
                        response_mask, rollout_is_weights, rc_metrics = (
                            self._apply_rollout_correction(
                                old_log_prob, response_mask, model_inputs,
                                rollout_corr_config,
                            )
                        )

                        advantages = model_inputs["advantages"]

                        loss_fn = get_policy_loss_fn(
                            self.actor.config.policy_loss.get("loss_mode", "vanilla")
                        )
                        pg_loss, pg_metrics = loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=self.actor.config.loss_agg_mode,
                            config=self.actor.config,
                            rollout_is_weights=rollout_is_weights,
                        )

                        policy_loss = pg_loss
                        if calculate_entropy and entropy is not None:
                            entropy_agg = agg_loss(
                                loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode_cfg,
                            )
                            micro_batch_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            if entropy_coeff != 0:
                                policy_loss = policy_loss - entropy_agg * entropy_coeff
                        else:
                            micro_batch_metrics = {}

                        loss = policy_loss / gradient_accumulation
                        loss.backward()

                        # Collect per-micro-batch metrics
                        micro_batch_metrics["pg_loss"] = pg_loss.detach().item()
                        micro_batch_metrics.update(pg_metrics)
                        micro_batch_metrics.update(rc_metrics)
                        append_to_dict(all_metrics, micro_batch_metrics)
                    forward_backward_total += _time.monotonic() - t_fb

                    # Compute similarity for this mini-batch
                    t_sim = _time.monotonic()
                    sim_opt.step()
                    sim_step_total += _time.monotonic() - t_sim

                    # Progress logging (every 10s or first/last mini-batch)
                    _now = _time.monotonic()
                    _done = _mini_idx + 1
                    if _now - _last_progress_time >= 10.0 or _done == _total_mini or _done == 1:
                        _elapsed = _now - _sim_start
                        _pct = 100.0 * _done / _total_mini
                        _parts = [
                            f"[{_time.strftime('%H:%M:%S')}] {_progress_label}",
                            f"progress: {_done}/{_total_mini} ({_pct:.1f}%)",
                            f"elapsed={_elapsed:.1f}s",
                            f"pending={_total_mini - _done}",
                        ]
                        if _done > 0 and _elapsed > 0:
                            _rate = _done / _elapsed
                            _parts.append(f"rate={_rate:.2f}/s")
                            if _done < _total_mini:
                                _parts.append(f"eta={(_total_mini - _done) / _rate:.1f}s")
                        logger.info(" | ".join(_parts))
                        _last_progress_time = _now

            timing["mini_batch_loop"] = _time.monotonic() - t_loop_start
            memory_metrics.update(self._cuda_memory_metric_snapshot("post_loop"))
            timing["forward_backward"] = forward_backward_total
            timing["sim_step"] = sim_step_total
            all_metrics["num_mini_batches"] = float(len(mini_batches))

            sim_metrics = sim_opt.get_concatenated_metrics()
        finally:
            t0 = _time.monotonic()
            self._clear_gradients()
            self._offload_momentum_to_cpu()
            if optimizer_loaded:
                fsdp_workers.offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            timing["cleanup"] = _time.monotonic() - t0
            memory_metrics.update(self._cuda_memory_metric_snapshot("post_cleanup"))

        logger.info(
            "compute_similarity timing: %s",
            {k: f"{v:.2f}s" for k, v in timing.items()},
        )

        all_metrics.update({f"influence_timing_s/{k}": v for k, v in timing.items()})
        all_metrics.update(memory_metrics)
        self.actor.config.global_batch_info.clear()
        similarity_metrics = {
            "score": sim_metrics["score"].cpu().tolist(),
            "grad_norm": sim_metrics["grad_norm"].cpu().tolist(),
            "ref_norm": sim_metrics["ref_norm"].cpu().item(),
            "num_minibatches": sim_metrics["num_minibatches"],
            "score_mode": sim_metrics["score_mode"],
        }
        for key in (
            "dot",
            "cosine",
            "gamma_norm",
            "preconditioned_dot",
            "preconditioned_cosine",
        ):
            if key in sim_metrics:
                similarity_metrics[key] = sim_metrics[key].cpu().tolist()

        return DataProto(
            meta_info={
                "similarity_metrics": similarity_metrics,
                "metrics": all_metrics,
            }
        )

    # ------------------------------------------------------------------
    # Helpers: Rollout Correction
    # ------------------------------------------------------------------

    def _apply_rollout_correction(
        self,
        old_log_prob: torch.Tensor,
        response_mask: torch.Tensor,
        model_inputs: dict,
        rollout_corr_config: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
        """Apply rollout correction on-the-fly during forward+backward.

        Follows the pattern in ``patched_dp_actor.py:190-238``: compute
        IS weights from ``old_log_prob`` (detached) and ``rollout_log_probs``
        during the same forward pass used for gradient computation.

        Delegates to ``compute_rollout_correction_and_rejection_mask`` which
        applies the full correction stack: RS → veto → IS → off-policy metrics.

        Args:
            old_log_prob: Detached log probs from current policy forward pass.
            response_mask: Binary mask for valid tokens.
            model_inputs: Dict containing batch tensors (may include
                ``rollout_log_probs``).
            rollout_corr_config: Rollout correction config dict, or ``None``
                to skip correction.

        Returns:
            ``(modified_response_mask, rollout_is_weights, metrics)``.
            If no correction is configured or ``rollout_log_probs`` is absent,
            returns ``(response_mask, None, {})``.
        """
        if rollout_corr_config is None:
            return response_mask, None, {}

        rollout_log_prob = model_inputs.get("rollout_log_probs", None)
        if rollout_log_prob is None:
            return response_mask, None, {}

        from verl.trainer.ppo.rollout_corr_helper import (
            compute_rollout_correction_and_rejection_mask,
        )

        is_weights_proto, modified_response_mask, metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=old_log_prob,
                rollout_log_prob=rollout_log_prob,
                response_mask=response_mask,
                rollout_is=rollout_corr_config.get("rollout_is", None),
                rollout_is_threshold=rollout_corr_config.get(
                    "rollout_is_threshold", 2.0
                ),
                rollout_rs=rollout_corr_config.get("rollout_rs", None),
                rollout_rs_threshold=rollout_corr_config.get(
                    "rollout_rs_threshold", None
                ),
                rollout_rs_threshold_lower=rollout_corr_config.get(
                    "rollout_rs_threshold_lower", None
                ),
                rollout_token_veto_threshold=rollout_corr_config.get(
                    "rollout_token_veto_threshold", None
                ),
                rollout_is_batch_normalize=rollout_corr_config.get(
                    "rollout_is_batch_normalize", False
                ),
            )
        )

        # Unwrap IS weights from DataProto
        rollout_is_weights = None
        if is_weights_proto is not None:
            rollout_is_weights = is_weights_proto.batch["rollout_is_weights"]

        return modified_response_mask, rollout_is_weights, metrics

    # ------------------------------------------------------------------
    # Helpers: Forward+Backward
    # ------------------------------------------------------------------

    def _policy_forward_backward(
        self,
        data: DataProto,
        temperature: float,
        micro_batch_size: int,
        rollout_corr_config: dict | None = None,
    ) -> dict[str, list]:
        """Run forward+backward without optimizer step.

        Treats all data as a single mini-batch and accumulates gradients
        across micro-batches in ``p.grad``.  Uses on-policy mode
        (``old_log_prob = log_prob.detach()``) to avoid needing pre-computed
        log probabilities.

        When ``rollout_corr_config`` is provided and ``rollout_log_probs``
        is present in the batch, applies rollout correction (RS + veto + IS)
        on-the-fly during the forward pass, following the pattern in
        ``patched_dp_actor.py:190-238``.

        This replaces the V2 approach of calling ``update_policy()`` with
        ``force_on_policy=True, skip_optimizer_and_retain_grad=True``.

        Args:
            data: DataProto on CPU with required batch fields.
            temperature: Forward-pass temperature.
            micro_batch_size: Samples per micro-batch.
            rollout_corr_config: Rollout correction config dict, or ``None``
                to skip correction.

        Returns:
            Per-micro-batch metrics as ``dict[str, list]`` (pg_loss,
            pg_metrics, rollout correction metrics).  Use
            ``reduce_metrics()`` to aggregate into scalars.
        """
        self.actor.actor_module.train()

        # Populate global_batch_info for loss normalization
        if "avg_response_tokens" in data.meta_info:
            dp_size = data.meta_info["dp_size"]
            self.actor.config.global_batch_info["batch_num_tokens"] = (
                data.meta_info["avg_response_tokens"] * micro_batch_size * dp_size
            )
            self.actor.config.global_batch_info["dp_size"] = dp_size

        select_keys = [
            "responses", "response_mask", "input_ids",
            "attention_mask", "position_ids", "advantages",
        ]
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        data = data.select(batch_keys=select_keys)
        micro_batches = data.split(micro_batch_size)
        gradient_accumulation = len(micro_batches)

        # log the number of samples, micro-batch size, and meta_info from data for debugging
        logger.info("Policy forward/backward: total_samples=%d, micro_batch_size=%d, meta_info=%s", data.batch["input_ids"].shape[0], micro_batch_size, data.meta_info)

        # Zero existing gradients before accumulation
        for p in self.actor.actor_module.parameters():
            if p.grad is not None:
                p.grad.zero_()

        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        entropy_coeff = self.actor.config.entropy_coeff
        loss_agg_mode = self.actor.config.loss_agg_mode
        calculate_entropy = self.actor.config.calculate_entropy or (entropy_coeff != 0)
        metrics: dict[str, list[float]] = {}
        # # >>> DEBUG_POLICY_INTERMEDIATES BEGIN <<<
        # _debug_collectors = {
        #     "old_log_prob": [], "log_prob": [], "advantages": [],
        #     "response_mask": [], "rollout_is_weights": [],
        #     "pg_loss": [], "pg_clipfrac": [], "ppo_kl": [],
        # }
        # # >>> DEBUG_POLICY_INTERMEDIATES END <<<

        _progress_label = data.meta_info.get("progress_label", "dev_gradient/forward_backward")
        _total_micro = len(micro_batches)
        _fb_start = time.monotonic()
        _last_progress_time = _fb_start

        for _mb_idx, micro_batch in enumerate(micro_batches):
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {
                **micro_batch.batch,
                **micro_batch.non_tensor_batch,
                "pad_token_id": pad_token_id,
            }

            entropy, log_prob = self.actor._forward_micro_batch(
                model_inputs,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
            )
            old_log_prob = log_prob.detach()  # on-policy

            response_mask = model_inputs["response_mask"]

            # Apply rollout correction (RS + veto + IS) on-the-fly
            response_mask, rollout_is_weights, rc_metrics = (
                self._apply_rollout_correction(
                    old_log_prob, response_mask, model_inputs,
                    rollout_corr_config,
                )
            )

            advantages = model_inputs["advantages"]

            loss_fn = get_policy_loss_fn(
                self.actor.config.policy_loss.get("loss_mode", "vanilla")
            )
            pg_loss, pg_metrics = loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode=self.actor.config.loss_agg_mode,
                config=self.actor.config,
                rollout_is_weights=rollout_is_weights,
            )

            # Progress logging (every 10s or every 10 micro-batches)
            _now = time.monotonic()
            _done = _mb_idx + 1
            if _now - _last_progress_time >= 10.0 or _done == _total_micro or _done == 1:
                _elapsed = _now - _fb_start
                _pct = 100.0 * _done / _total_micro
                _parts = [
                    f"[{time.strftime('%H:%M:%S')}] {_progress_label}",
                    f"progress: {_done}/{_total_micro} ({_pct:.1f}%)",
                    f"elapsed={_elapsed:.1f}s",
                    f"pending={_total_micro - _done}",
                ]
                if _done > 0 and _elapsed > 0:
                    _rate = _done / _elapsed
                    _parts.append(f"rate={_rate:.2f}/s")
                    if _done < _total_micro:
                        _parts.append(f"eta={(_total_micro - _done) / _rate:.1f}s")
                logger.info(" | ".join(_parts))
                _last_progress_time = _now

            # # >>> DEBUG_POLICY_INTERMEDIATES COLLECT <<<
            # _debug_collectors["old_log_prob"].append(old_log_prob.detach().float().cpu())
            # _debug_collectors["log_prob"].append(log_prob.detach().float().cpu())
            # _debug_collectors["advantages"].append(advantages.detach().float().cpu())
            # _debug_collectors["response_mask"].append(response_mask.detach().float().cpu())
            # _debug_collectors["rollout_is_weights"].append(
            #     rollout_is_weights.detach().float().cpu() if rollout_is_weights is not None
            #     else torch.zeros_like(response_mask, dtype=torch.float32, device="cpu")
            # )
            # _debug_collectors["pg_loss"].append(pg_loss.detach().float().cpu().unsqueeze(0))
            # _debug_collectors["pg_clipfrac"].append(
            #     torch.tensor([pg_metrics.get("actor/pg_clipfrac", 0.0)], dtype=torch.float32)
            # )
            # _debug_collectors["ppo_kl"].append(
            #     torch.tensor([pg_metrics.get("actor/ppo_kl", 0.0)], dtype=torch.float32)
            # )
            # # >>> DEBUG_POLICY_INTERMEDIATES END <<<

            policy_loss = pg_loss
            if calculate_entropy and entropy is not None:
                entropy_agg = agg_loss(
                    loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode,
                )
                micro_batch_metrics = {"actor/entropy": entropy_agg.detach().item()}
                if entropy_coeff != 0:
                    policy_loss = policy_loss - entropy_agg * entropy_coeff
            else:
                micro_batch_metrics = {}

            loss = policy_loss / gradient_accumulation
            # Fix uneven last micro-batch weighting: scale by actual/target size
            # so all samples contribute equally. Skip when global_batch_info is set
            # (token-mean mode already handles this via fixed batch_num_tokens).
            # Note: this makes the effective denominator ceil(N/M)*M instead of N,
            # so total gradient magnitude is N/(ceil(N/M)*M) of the true mean.
            # This is negligible for adaptive optimizers (Adam) which normalize by
            # second moments, and for cosine similarity scores which are scale-invariant.
            needs_manual_seq_rescale = (
                self.actor.config.loss_agg_mode in {"seq-mean-token-mean", "seq-mean-token-sum"}
                and self.actor.config.global_batch_info.get("global_batch_size") is None
            )
            if needs_manual_seq_rescale:
                loss = loss * (micro_batch.batch["input_ids"].shape[0] / micro_batch_size)
            loss.backward()

            # Collect per-micro-batch metrics
            micro_batch_metrics["pg_loss"] = pg_loss.detach().item()
            micro_batch_metrics.update(pg_metrics)
            micro_batch_metrics.update(rc_metrics)
            append_to_dict(metrics, micro_batch_metrics)

        # # >>> DEBUG_POLICY_INTERMEDIATES BEGIN <<<
        # import os, pathlib, time as _time_mod
        # _debug_save_dir = pathlib.Path(os.environ.get(
        #     "DEBUG_POLICY_SAVE_DIR", "/tmp/debug_policy_intermediates"
        # ))
        # _debug_save_dir.mkdir(parents=True, exist_ok=True)
        # _rank = dist.get_rank()
        # _timestamp = _time_mod.strftime("%Y%m%d_%H%M%S")
        # _debug_payload = {
        #     # (total_samples, response_length) tensors
        #     "old_log_prob": torch.cat(_debug_collectors["old_log_prob"], dim=0),
        #     "log_prob": torch.cat(_debug_collectors["log_prob"], dim=0),
        #     "advantages": torch.cat(_debug_collectors["advantages"], dim=0),
        #     "response_mask": torch.cat(_debug_collectors["response_mask"], dim=0),
        #     "rollout_is_weights": torch.cat(_debug_collectors["rollout_is_weights"], dim=0),
        #     # (total_micro_batches,) scalars
        #     "pg_loss": torch.cat(_debug_collectors["pg_loss"], dim=0),
        #     "pg_clipfrac": torch.cat(_debug_collectors["pg_clipfrac"], dim=0),
        #     "ppo_kl": torch.cat(_debug_collectors["ppo_kl"], dim=0),
        #     # config scalars
        #     "loss_agg_mode": loss_agg_mode,
        #     "gradient_accumulation": gradient_accumulation,
        #     "loss_scale_factor": 1.0 / gradient_accumulation,
        #     "on_policy": True,
        #     "micro_batch_size": micro_batch_size,
        #     "temperature": temperature,
        #     "rollout_corr_config": rollout_corr_config,
        # }
        # _save_path = _debug_save_dir / f"policy_fwd_bwd_{_timestamp}_rank{_rank}.pt"
        # torch.save(_debug_payload, _save_path)
        # if _rank == 0:
        #     import dataclasses as _dc
        #     _cfg_path = _debug_save_dir / f"actor_config_{_timestamp}.pt"
        #     torch.save(_dc.asdict(self.actor.config), _cfg_path)
        #     logger.info("[DEBUG] Saved policy intermediates to %s "
        #                 "(shapes: old_log_prob=%s, pg_loss=%s), "
        #                 "actor config to %s",
        #                 _save_path, _debug_payload["old_log_prob"].shape,
        #                 _debug_payload["pg_loss"].shape, _cfg_path)
        # # >>> DEBUG_POLICY_INTERMEDIATES END <<<

        self.actor.config.global_batch_info.clear()
        return metrics

    # ------------------------------------------------------------------
    # Helpers: Momentum
    # ------------------------------------------------------------------

    def _update_momentum(self, beta: float) -> None:
        """Update momentum via EMA: ``m = beta * m + (1 - beta) * grad``.

        Reads gradients from ``p.grad`` on the FSDP model.  On first call,
        initializes momentum as a clone of the gradient.

        Reference: v2 ``verl_joint_dev_similarity.py:605-646``.
        """
        params = list(self.actor.actor_module.parameters())

        if self.momentum is None:
            # Initialize from first gradient
            self.momentum = [
                p.grad.detach().clone() if p.grad is not None else None
                for p in params
            ]
        else:
            # EMA update in-place
            for i, (m, p) in enumerate(zip(self.momentum, params)):
                g = p.grad
                if m is not None and g is not None:
                    m.mul_(beta).add_(g, alpha=1 - beta)
                elif g is not None:
                    self.momentum[i] = g.detach().clone()

    def _clear_gradients(self) -> None:
        """Zero out all model gradients and free GPU cache."""
        for p in self.actor.actor_module.parameters():
            if p.grad is not None:
                p.grad.zero_()
        torch.cuda.empty_cache()

    def _cuda_memory_metric_snapshot(self, label: str) -> dict[str, float]:
        """Capture a point-in-time CUDA memory snapshot in GB."""
        if not torch.cuda.is_available():
            return {}

        device = get_device_id()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            f"influence_memory_gb/{label}/allocated": torch.cuda.memory_allocated(device) / 1024**3,
            f"influence_memory_gb/{label}/reserved": torch.cuda.memory_reserved(device) / 1024**3,
            f"influence_memory_gb/{label}/max_allocated": torch.cuda.max_memory_allocated(device) / 1024**3,
            f"influence_memory_gb/{label}/max_reserved": torch.cuda.max_memory_reserved(device) / 1024**3,
            f"influence_memory_gb/{label}/free": free_bytes / 1024**3,
            f"influence_memory_gb/{label}/total": total_bytes / 1024**3,
        }

    def _reset_cuda_peak_memory_stats(self) -> None:
        """Reset peak CUDA memory stats before a scoped measurement window."""
        if not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats(get_device_id())

    def _auto_micro_batch_size(self, total_samples: int) -> int:
        """Pick largest micro-batch size <= configured that divides total_samples."""
        target = self.actor.config.ppo_micro_batch_size_per_gpu
        for candidate in range(min(total_samples, target), 0, -1):
            if total_samples % candidate == 0:
                return candidate
        return 1

    def _offload_momentum_to_cpu(self) -> None:
        """Move momentum tensors to CPU to free GPU memory."""
        if self.momentum is not None:
            self.momentum = [
                m.to("cpu", non_blocking=True) if m is not None else None
                for m in self.momentum
            ]
            torch.cuda.empty_cache()

    def _load_momentum_to_gpu(self) -> None:
        """Move momentum tensors back to GPU for computation."""
        if self.momentum is not None:
            device = get_device_id()
            self.momentum = [
                m.to(device, non_blocking=True) if m is not None else None
                for m in self.momentum
            ]

    def reset_momentum(self) -> None:
        """Reset the gradient momentum buffer.

        Called at the start of each ans_loop when momentum EMA is disabled
        (``influence.use_momentum = False``).
        """
        self.momentum = None

    # ------------------------------------------------------------------
    # Momentum persistence (save/load for crash recovery)
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_momentum(self, path: str) -> None:
        """Save FQN-keyed momentum to ``{path}/momentum_with_fqns_world_size_{W}_rank_{R}.pt``.

        Each rank saves its own shard.  Uses ``named_parameters()`` for
        FQN-keyed format so the checkpoint is resilient to parameter
        reordering across FSDP versions.

        Reference: v2 ``verl_joint_dev_similarity.py:648-769``
        """
        if self.momentum is None:
            logger.info("save_momentum: no momentum to save")
            return

        os.makedirs(path, exist_ok=True)

        # Build name → index mapping from the same iteration order as parameters()
        param_names = [name for name, _ in self.actor.actor_module.named_parameters()]
        momentum_with_names: dict[str, torch.Tensor] = {}

        for i, m in enumerate(self.momentum):
            if m is not None and i < len(param_names):
                momentum_with_names[param_names[i]] = m.cpu()

        filename = f"momentum_with_fqns_world_size_{dist.get_world_size()}_rank_{dist.get_rank()}.pt"
        filepath = os.path.join(path, filename)
        torch.save(momentum_with_names, filepath)
        logger.info(
            "Saved %d momentum tensors to %s", len(momentum_with_names), filepath
        )
        dist.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_momentum(self, path: str) -> None:
        """Load FQN-keyed momentum from ``{path}/momentum_with_fqns_world_size_{W}_rank_{R}.pt``.

        Restores ``self.momentum`` list aligned with the current model's
        ``parameters()`` iteration order.

        Reference: v2 ``verl_joint_dev_similarity.py:554-603``
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        fqn_file = os.path.join(
            path, f"momentum_with_fqns_world_size_{world_size}_rank_{rank}.pt"
        )

        if not os.path.exists(fqn_file):
            logger.warning("load_momentum: file not found %s, starting fresh", fqn_file)
            self.momentum = None
            dist.barrier()
            return

        fqn_data = torch.load(fqn_file, map_location="cpu", weights_only=False)

        # Build name → index mapping
        param_names = [name for name, _ in self.actor.actor_module.named_parameters()]
        name_to_idx = {name: idx for idx, name in enumerate(param_names)}

        num_params = len(param_names)
        self.momentum = [None] * num_params

        loaded = 0
        for name, tensor in fqn_data.items():
            if name in name_to_idx:
                self.momentum[name_to_idx[name]] = tensor
                loaded += 1

        logger.info("Loaded %d/%d momentum tensors from %s", loaded, num_params, fqn_file)
        dist.barrier()
