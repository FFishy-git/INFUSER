"""CPU-only mock runtime for dry-run HF upload/resume validation."""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl_inf_evolve.utils.generator_reward_utils import (
    build_stage4_reward_payload,
    resolve_generator_reward_components,
    resolve_generator_reward_structure,
)

logger = logging.getLogger(__name__)


def is_mock_cpu_dry_run_enabled(config: Any) -> bool:
    """Return True when the config enables the mock CPU dry-run backend."""
    dry_run_cfg = config.get("dry_run", {})
    return bool(dry_run_cfg.get("enabled", False)) and str(
        dry_run_cfg.get("backend", "mock_cpu")
    ) == "mock_cpu"


def make_mock_tokenizer(name_or_path: str) -> "MockTokenizer":
    """Build a trivial tokenizer-like object for dry-run code paths."""
    return MockTokenizer(name_or_path=name_or_path)


class MockDryRunCrash(RuntimeError):
    """Controlled crash used to test remote/local resume behavior."""


@dataclass
class MockTokenizer:
    """Tiny tokenizer stub used to satisfy trainer fields in dry-run mode."""

    name_or_path: str
    eos_token_id: int = 0
    pad_token_id: int = 0
    chat_template: str | None = None


def run_mock_cpu_dry_run(config: Any) -> None:
    """Launch the trainer in CPU-only dry-run mode."""
    from verl_inf_evolve.dry_run.mock_trainer import MockDryRunTrainer

    trainer = MockDryRunTrainer(
        config=config,
        gen_tokenizer=make_mock_tokenizer("dry-run-generator"),
        solver_tokenizer=make_mock_tokenizer("dry-run-solver"),
        role_worker_mapping={},
        resource_pool_manager=None,
        ray_worker_group_cls=None,
    )
    trainer.init_workers()
    trainer.fit()


class MockCPUWorker:
    """CPU-only checkpoint worker that emits structural placeholder artifacts."""

    def __init__(self, role_name: str, config: Any):
        self.role_name = role_name
        self.config = config
        self.world_size = int(config.dry_run.get("mock_world_size", 1))
        self.model_path = str(config.get(role_name, {}).get("model", {}).get("path", role_name))

    def init_model(self) -> None:
        """No-op model initialization."""
        logger.info("MockCPUWorker[%s]: init_model skipped", self.role_name)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
    ) -> None:
        """Write placeholder checkpoint files with real on-disk sizes."""
        del hdfs_path, global_step, max_ckpt_to_keep
        cfg = self.config.dry_run.checkpoint
        os.makedirs(local_path, exist_ok=True)

        model_bytes = _mb_to_bytes(cfg.get("model_shard_mb", 0.0))
        optim_bytes = _mb_to_bytes(cfg.get("optim_shard_mb", 0.0))
        extra_bytes = _kb_to_bytes(cfg.get("extra_state_kb", 0))
        hf_shard_sizes_cfg = list(cfg.get("hf_shard_sizes_mb", []))
        if hf_shard_sizes_cfg:
            shard_sizes = [_mb_to_bytes(size_mb) for size_mb in hf_shard_sizes_cfg]
            hf_num_shards = len(shard_sizes)
            hf_total_bytes = sum(shard_sizes)
        else:
            hf_total_bytes = _mb_to_bytes(cfg.get("hf_total_mb", 0.0))
            hf_num_shards = max(1, int(cfg.get("hf_num_shards", 1)))
            shard_sizes = _split_bytes(hf_total_bytes, hf_num_shards)

        for rank in range(self.world_size):
            _write_exact_bytes(
                os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{rank}.pt"),
                model_bytes,
                seed=f"{self.role_name}:model:{rank}",
            )
            _write_exact_bytes(
                os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"),
                optim_bytes,
                seed=f"{self.role_name}:optim:{rank}",
            )
            _write_exact_bytes(
                os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{rank}.pt"),
                extra_bytes,
                seed=f"{self.role_name}:extra:{rank}",
            )

        with open(os.path.join(local_path, "fsdp_config.json"), "w") as f:
            json.dump({"FSDP_version": 1, "world_size": self.world_size}, f, indent=2)

        hf_dir = os.path.join(local_path, "huggingface")
        os.makedirs(hf_dir, exist_ok=True)
        with open(os.path.join(hf_dir, "config.json"), "w") as f:
            json.dump(
                {
                    "architectures": ["MockForCausalLM"],
                    "model_type": "mock_dry_run",
                    "name_or_path": self.model_path,
                    "torch_dtype": "bfloat16",
                    "dry_run": True,
                },
                f,
                indent=2,
            )

        weight_map: dict[str, str] = {}
        for idx, shard_size in enumerate(shard_sizes, start=1):
            shard_name = f"model-{idx:05d}-of-{hf_num_shards:05d}.safetensors"
            _write_exact_bytes(
                os.path.join(hf_dir, shard_name),
                shard_size,
                seed=f"{self.role_name}:hf:{idx}",
            )
            weight_map[f"mock_weight_{idx:05d}"] = shard_name

        with open(os.path.join(hf_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(
                {
                    "metadata": {"total_size": hf_total_bytes},
                    "weight_map": weight_map,
                },
                f,
                indent=2,
            )

        with open(os.path.join(local_path, "dry_run_manifest.json"), "w") as f:
            json.dump(
                {
                    "mode": "mock_cpu",
                    "role_name": self.role_name,
                    "backend": self.config.dry_run.get("backend", "mock_cpu"),
                    "resume_loader": self.config.dry_run.get("resume_loader", "mock"),
                    "world_size": self.world_size,
                    "model_path": self.model_path,
                    "sizes": {
                        "model_shard_bytes": model_bytes,
                        "optim_shard_bytes": optim_bytes,
                        "extra_state_bytes": extra_bytes,
                        "hf_total_bytes": hf_total_bytes,
                        "hf_num_shards": hf_num_shards,
                        "hf_shard_sizes_bytes": shard_sizes,
                    },
                    "created_at": time.time(),
                },
                f,
                indent=2,
            )

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        del_local_after_load: bool = False,
    ) -> None:
        """Validate placeholder FSDP-style checkpoint shards."""
        del hdfs_path, del_local_after_load
        manifest = _load_manifest(local_path)
        sizes = manifest["sizes"]

        for rank in range(self.world_size):
            _validate_file(
                os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{rank}.pt"),
                int(sizes["model_shard_bytes"]),
            )
            _validate_file(
                os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"),
                int(sizes["optim_shard_bytes"]),
            )
            _validate_file(
                os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{rank}.pt"),
                int(sizes["extra_state_bytes"]),
            )

        with open(os.path.join(local_path, "fsdp_config.json"), "r") as f:
            fsdp_config = json.load(f)
        if int(fsdp_config.get("world_size", -1)) != self.world_size:
            raise RuntimeError(
                f"Mock checkpoint world_size mismatch: {fsdp_config.get('world_size')} != {self.world_size}"
            )

    def load_hf_checkpoint(self, local_path: str) -> None:
        """Validate placeholder HF safetensor checkpoint files."""
        ckpt_root = os.path.dirname(local_path.rstrip(os.sep))
        manifest = _load_manifest(ckpt_root)
        sizes = manifest["sizes"]

        config_path = os.path.join(local_path, "config.json")
        index_path = os.path.join(local_path, "model.safetensors.index.json")
        if not os.path.isfile(config_path) or not os.path.isfile(index_path):
            raise FileNotFoundError(f"Incomplete mock HF checkpoint under {local_path}")

        with open(index_path, "r") as f:
            index_data = json.load(f)
        shard_names = sorted(set(index_data.get("weight_map", {}).values()))
        if len(shard_names) != int(sizes["hf_num_shards"]):
            raise RuntimeError(
                f"Mock HF shard count mismatch: {len(shard_names)} != {sizes['hf_num_shards']}"
            )

        total_size = 0
        expected_shard_sizes = list(sizes.get("hf_shard_sizes_bytes", []))
        for idx, shard_name in enumerate(shard_names):
            shard_path = os.path.join(local_path, shard_name)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Missing mock HF shard: {shard_path}")
            shard_size = os.path.getsize(shard_path)
            total_size += shard_size
            if expected_shard_sizes and shard_size != int(expected_shard_sizes[idx]):
                raise RuntimeError(
                    f"Mock HF shard size mismatch for {shard_name}: "
                    f"{shard_size} != {expected_shard_sizes[idx]}"
                )
        if total_size != int(sizes["hf_total_bytes"]):
            raise RuntimeError(
                f"Mock HF total size mismatch: {total_size} != {sizes['hf_total_bytes']}"
            )

    def update_actor(self, data: Any) -> SimpleNamespace:
        """No-op actor update stub for defensive compatibility."""
        del data
        return SimpleNamespace(meta_info={"metrics": {}})


class MockCPUWorkerGroup:
    """Minimal worker-group facade used by the trainer in dry-run mode."""

    def __init__(self, role_name: str, config: Any):
        self.role_name = role_name
        self.worker = MockCPUWorker(role_name=role_name, config=config)
        self.world_size = self.worker.world_size

    def init_model(self) -> None:
        self.worker.init_model()

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        self.worker.save_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        self.worker.load_checkpoint(*args, **kwargs)

    def load_hf_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        self.worker.load_hf_checkpoint(*args, **kwargs)

    def update_actor(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        return self.worker.update_actor(*args, **kwargs)


def build_mock_dev_output(config: Any, ans_loop: int) -> DataProto:
    """Create a tiny answer-rollout DataProto for Stage 1."""
    rollout_n = max(1, int(config.solver.rollout.n))
    question_ids: list[str] = []
    scores: list[float] = []
    for q_idx in range(2):
        qid = f"dry_dev_ans{ans_loop}_q{q_idx}"
        for sample_idx in range(rollout_n):
            question_ids.append(qid)
            scores.append(1.0 if sample_idx % 2 == 0 else 0.0)

    output = _make_answer_output(question_ids, scores, response_length=8)
    return pad_dataproto_to_size(output, float(config.dry_run.stage_outputs.get("dev_output_mb", 0.0)))


def build_mock_question_output(config: Any, ans_loop: int, local_gen: int) -> DataProto:
    """Create a tiny question-rollout DataProto for Stage 2."""
    rollout_n = max(1, int(config.generator.rollout.n))
    doc_ids: list[str] = []
    question_ids: list[str] = []
    question_texts: list[str] = []
    choices_list: list[list[str]] = []
    ground_truths: list[str] = []
    parsed_ok: list[bool] = []
    reject_reasons: list[str | None] = []

    for doc_idx in range(2):
        doc_id = f"dry_doc_ans{ans_loop}_gen{local_gen}_{doc_idx}"
        for sample_idx in range(rollout_n):
            doc_ids.append(doc_id)
            question_ids.append(f"gen_{doc_id}_{sample_idx}")
            question_texts.append(
                f"Dry-run question {sample_idx} for doc {doc_idx} in ans_loop {ans_loop}."
            )
            choices_list.append(["A", "B", "C", "D"])
            ground_truths.append("A")
            parsed_ok.append(True)
            reject_reasons.append(None)

    batch_size = len(question_ids)
    batch = _make_response_batch(batch_size=batch_size, response_length=12)
    output = DataProto(
        batch=batch,
        non_tensor_batch={
            "doc_id": np.array(doc_ids, dtype=object),
            "question_id": np.array(question_ids, dtype=object),
            "question_text": np.array(question_texts, dtype=object),
            "choices": np.array(choices_list, dtype=object),
            "ground_truth": np.array(ground_truths, dtype=object),
            "parsed_ok": np.array(parsed_ok, dtype=bool),
            "reject_reason": np.array(reject_reasons, dtype=object),
        },
        meta_info={},
    )
    return pad_dataproto_to_size(output, float(config.dry_run.stage_outputs.get("gen_output_mb", 0.0)))


def build_mock_gen_answer_output(config: Any, gen_questions: list[dict[str, Any]]) -> DataProto:
    """Create a tiny answer-rollout DataProto for Stage 3."""
    rollout_n = max(1, int(config.solver.rollout.n))
    question_ids: list[str] = []
    scores: list[float] = []
    for q in gen_questions:
        qid = str(q["question_id"])
        for sample_idx in range(rollout_n):
            question_ids.append(qid)
            scores.append(1.0 if sample_idx % 2 == 0 else 0.0)

    output = _make_answer_output(question_ids, scores, response_length=10)
    return pad_dataproto_to_size(
        output,
        float(config.dry_run.stage_outputs.get("gen_answer_output_mb", 0.0)),
    )


def build_mock_stage4_reward_payload(
    config: Any,
    gen_questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a normal stage-4 reward payload for the dry-run path."""
    valid_question_ids = {str(question["question_id"]) for question in gen_questions}
    ordered_qids = sorted(valid_question_ids)
    selected_components = resolve_generator_reward_components(config.training)
    reward_structure = resolve_generator_reward_structure(config.training)

    influence_rewards = {
        qid: float((idx % 3) + 1) / 10.0 for idx, qid in enumerate(ordered_qids)
    }
    spice_rewards = {
        qid: float((idx % 2) + 2) / 10.0 for idx, qid in enumerate(ordered_qids)
    }

    return build_stage4_reward_payload(
        valid_question_ids=valid_question_ids,
        influence_rewards=influence_rewards,
        spice_rewards=spice_rewards,
        selected_components=selected_components,
        reward_structure=reward_structure,
    )


def pad_dataproto_to_size(data: DataProto, target_mb: float) -> DataProto:
    """Pad a DataProto with ignored bytes until its pickle size reaches the target."""
    target_bytes = _mb_to_bytes(target_mb)
    if target_bytes <= 0:
        return data

    base_size = _pickle_size(data)
    if base_size >= target_bytes:
        return data

    batch_dims = tuple(getattr(data.batch, "batch_size", ()))
    batch_size = int(batch_dims[0]) if batch_dims else 1
    low = 0
    high = max(1, target_bytes - base_size)
    best: DataProto | None = None

    while low <= high:
        mid = (low + high) // 2
        candidate = _copy_with_padding(data, batch_size, mid)
        candidate_size = _pickle_size(candidate)
        if candidate_size >= target_bytes:
            best = candidate
            high = mid - 1
        else:
            low = mid + 1

    if best is None:
        best = _copy_with_padding(data, batch_size, target_bytes - base_size + 1024)

    return best


def _make_answer_output(
    question_ids: list[str],
    scores: list[float],
    response_length: int,
) -> DataProto:
    batch = _make_response_batch(batch_size=len(question_ids), response_length=response_length)
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "question_id": np.array(question_ids, dtype=object),
            "answer_score": np.array(scores, dtype=object),
        },
        meta_info={},
    )


def _make_response_batch(batch_size: int, response_length: int) -> TensorDict:
    responses = torch.arange(batch_size * response_length, dtype=torch.long).reshape(batch_size, response_length)
    response_mask = torch.ones(batch_size, response_length, dtype=torch.float32)
    return TensorDict(
        {
            "responses": responses,
            "response_mask": response_mask,
        },
        batch_size=batch_size,
    )


def _copy_with_padding(data: DataProto, batch_size: int, payload_size: int) -> DataProto:
    padded = copy.deepcopy(data)
    padding = np.empty(batch_size, dtype=object)
    padding[0] = b"x" * payload_size
    for idx in range(1, batch_size):
        padding[idx] = b""
    padded.non_tensor_batch["dry_run_padding"] = padding
    return padded


def _pickle_size(obj: Any) -> int:
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def _load_manifest(checkpoint_dir: str) -> dict[str, Any]:
    manifest_path = os.path.join(checkpoint_dir, "dry_run_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing dry_run_manifest.json under {checkpoint_dir}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if manifest.get("mode") != "mock_cpu":
        raise RuntimeError(f"Unsupported dry-run manifest mode: {manifest.get('mode')}")
    return manifest


def _validate_file(path: str, expected_size: int) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise RuntimeError(f"File size mismatch for {path}: {actual_size} != {expected_size}")


def _split_bytes(total_bytes: int, num_parts: int) -> list[int]:
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    base = total_bytes // num_parts
    remainder = total_bytes % num_parts
    return [base + (1 if idx < remainder else 0) for idx in range(num_parts)]


def _write_exact_bytes(path: str, size_bytes: int, seed: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seed_bytes = seed.encode("utf-8") or b"x"
    repeats = (1024 * 1024) // len(seed_bytes) + 1
    chunk = (seed_bytes * repeats)[: 1024 * 1024]
    with open(path, "wb") as f:
        remaining = size_bytes
        while remaining > 0:
            part = chunk[: min(len(chunk), remaining)]
            f.write(part)
            remaining -= len(part)


def _mb_to_bytes(value: Any) -> int:
    return max(0, int(float(value) * 1024**2))


def _kb_to_bytes(value: Any) -> int:
    return max(0, int(float(value) * 1024))
