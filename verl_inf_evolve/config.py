"""
Configuration dataclasses for the V3 Self-Evolution Pipeline.

These are plain dataclasses used with Hydra/OmegaConf. The generator/solver
model configs reuse verl's existing ActorRolloutRefConfig via OmegaConf and
are NOT duplicated here.

Mirrors v2's TrainingConfig (inf_evolve/train.py:565) but structured for
verl-native integration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SelfEvolutionTrainingConfig:
    """Loop control and top-level training settings.

    Corresponds to v2's TrainingConfig loop fields (train.py:573-611).
    """

    max_ans_loop: int = 100
    max_gen_loop: int = 100
    # Canonical generator reward composition.
    # When non-empty, this becomes the source of truth and implies decoupled
    # advantage. Each top-level entry is a group; terms inside a group are
    # summed before normalization, then group-level advantages are mixed using
    # group_weight.
    generator_reward_structure: list[dict] = field(default_factory=list)
    # Legacy flat component list, kept for backward compatibility when
    # generator_reward_structure is empty.
    # Allowed values: "influence_rewards", "spice_rewards", "invalid_rewards".
    generator_reward_components: list[str] = field(
        default_factory=lambda: ["influence_rewards", "invalid_rewards"]
    )
    # Legacy combination mode used only when generator_reward_structure is
    # empty. Structured configs imply decoupled advantage.
    # - "sum_scores": sum selected component scores, then compute one advantage.
    # - "decoupled": GDPO-style decoupled normalization — per-component GRPO
    #   advantage, sum, then batch-normalize.
    generator_reward_combination_mode: str = "sum_scores"
    dev_only: bool = False
    fix_generator: bool = False
    fix_answer_model: bool = False
    doc_batch_size: int = 32
    repeat_doc_batch: bool = False
    seed: int = 42
    # Dev rollout sampling: 0 uses full dev set every ans_loop (default).
    # >0 samples this many dev questions per ans_loop without replacement.
    dev_rollout_subsample_size: int = 0
    # Optional base seed for dev rollout sampling; when None, uses ``seed``.
    # Effective per-loop seed is ``base_seed + ans_loop``.
    dev_rollout_subsample_seed: Optional[int] = None

    # Dataset paths
    dev_dataset_path: str = ".cache/data/preprocessed/curriculum_pool/supergpqa_science_800.json"
    documents_path: str = ".cache/data/preprocessed/documents.json"

    # Seeded question generation (no-document mode)
    # "document" = generate questions from documents (default).
    # "seeded_dev" = generate questions from few-shot seed examples only.
    question_source_mode: str = "document"
    # Path to seed question JSON (list of dicts with question_id, question_text,
    # choices, ground_truth, domain, difficulty).  Required when
    # question_source_mode = "seeded_dev".
    seed_dataset_path: Optional[str] = None
    # Number of seed examples shown as few-shot demonstrations per prompt.
    seed_examples_per_prompt: int = 4

    # Reward assigned to generated questions that fail MCQ validation in
    # prepare_question_batch (e.g. ground_truth doesn't match any choice).
    # These never enter the solver rollout, so they have no computed reward.
    gen_invalid_penalty: float = 0.0

    # Checkpoint settings
    # Save a checkpoint every N ans_loops.  1 = every loop (default).
    # The final ans_loop is always saved regardless of this setting.
    # When always_save_for_resume=True, FSDP is saved every loop and
    # this setting only controls which HF checkpoints are kept on R2.
    save_every_n_steps: int = 1
    # When True: save FSDP checkpoint every loop for exact resume.
    # save_every_n_steps then controls HF retention on R2 only
    # (HF kept at 0, N, 2N, ... plus final loop; FSDP rotated).
    # When False (default): original behavior — checkpoint only at
    # save_every_n_steps intervals, less R2 upload pressure.
    always_save_for_resume: bool = False
    default_local_dir: str = "checkpoints/"
    # Hadoop Distributed File System dir — verl API compatibility only, always None (unused).
    default_hdfs_dir: Optional[str] = None
    # When True AND remote_sync_path is set: delete local checkpoint dirs
    # after R2 upload (cleanup_after=True) and clear local ans_{N}/ stage
    # output dirs after successful checkpoint.  When remote_sync_path is
    # None (local-only mode), this flag is ignored — local data is never
    # deleted because it's the only copy.
    remove_previous_ckpt: bool = False

    # Remote sync settings (R2)
    # R2 bucket path for stage outputs and checkpoints, e.g. "s3://bucket/experiment_name".
    # If None, remote sync is disabled (local-only mode).
    remote_sync_path: Optional[str] = None
    # When True, _load_checkpoint() downloads latest_checkpointed_iteration.txt
    # from R2 before reading it.  Enables resume on a fresh machine where no
    # local checkpoint marker exists.  Requires remote_sync_path to be set.
    resume_from_remote: bool = False
    # Path to pre-generated questions from a separate gen-only job.
    # When set, Stage 2 downloads gen_questions.json from this path
    # instead of running generator rollout.  Implies fix_generator=true.
    # Format: "hf://datasets/namespace/repo/prefix"
    pregenerated_question_source: Optional[str] = None
    # Backpressure threshold: block new uploads when pending serialized data exceeds this (GB).
    max_pending_upload_gb: float = 100.0

    # Optional role-specific initialization checkpoints (fresh start only).
    # These are loaded AFTER worker/model init and AFTER _load_checkpoint()
    # returns start_ans_loop. They are skipped when start_ans_loop > 0, so
    # existing resume behavior remains unchanged.
    #
    # Supported values:
    # - local directory containing FSDP shards (model_world_size_*.pt, etc.)
    # - local role checkpoint dir containing "huggingface/" subdir
    # - local HuggingFace directory containing model.safetensors
    # - remote R2/S3 directory path (downloaded via rclone)
    init_generator_checkpoint_path: Optional[str] = None
    init_solver_checkpoint_path: Optional[str] = None


@dataclass
class InfluenceConfig:
    """Gradient-based influence scoring settings.

    Controls momentum, filtering, and optional score quantification for the
    joint dev-gradient + per-question similarity computation.

    Corresponds to v2's JointDevSimilarityStageConfig (train.py:467-475)
    and the gradient utility configs (grad_utils/config.py).
    """

    use_momentum: bool = True
    momentum_beta: float = 0.9
    micro_batch_size: int = 4
    # Supported values: "dot", "cosine", "preconditioned_dot",
    # "preconditioned_cosine".
    similarity_mode: str = "cosine"
    # Solver training data filtering
    # Modes: "none", "top_<i>", "random_<i>", "top_alpha", "random_alpha"
    # (Aliases: "top_one" == "top_1", "random_one" == "random_1")
    solver_filter_mode: str = "none"
    # Fraction used in *_alpha modes. Threshold is alpha * total questions.
    solver_filter_alpha: float = 1.0
    # Score quantification applied to influence/SPICE scores in
    # prepare_gen_update_batch before generator PPO update:
    # None, "1bit", "2bit", "group_std_top_gamma", or
    # "group_std_fixed_threshold"
    quantification_mode: Optional[str] = None
    # Keep only doc groups whose score std is above the (1-gamma) quantile of
    # per-group std values. Used when quantification_mode=group_std_top_gamma.
    group_std_gamma: float = 0.2
    # Optional upper cap for tau in group_std_top_gamma mode.
    # Effective tau = min(quantile_tau, group_std_tau_max).
    group_std_tau_max: Optional[float] = None
    # Keep only doc groups whose score std is strictly greater than this value.
    # Used when quantification_mode=group_std_fixed_threshold.
    group_std_tau: Optional[float] = None


@dataclass
class SpiceConfig:
    """SPICE (Self-Play with In-Context Evolution) scoring settings.

    SPICE uses answer variance as a reward signal instead of gradient
    similarity. Higher reward for variance near target_variance.

    Reward formula:
      exp(-(variance - target_variance)^2 / variance_scale)

    Corresponds to v2's SpiceScoringStageConfig (train.py:451-464).
    """

    target_variance: float = 0.25
    variance_scale: float = 0.02


@dataclass
class CurriculumConfig:
    """Curriculum learning settings for dynamic dev set refresh.

    Periodically replaces easy/hard questions in the dev set to maintain
    an appropriately challenging training signal.

    Corresponds to v2's curriculum_* fields in TrainingConfig (train.py:641-653).
    """

    enabled: bool = False
    data_pool_path: str = ".cache/data/preprocessed/curriculum_pool/supergpqa_science_800.json"
    num_dev_questions: int = 150
    refresh_every_n_ans_loops: int = 10
    remove_top_ratio: float = 0.1
    remove_bottom_ratio: float = 0.1


@dataclass
class BenchmarkEvalConfig:
    """In-training benchmark evaluation settings."""

    enabled: bool = False
    eval_every_n_steps: int = 5
    eval_on_first_step: bool = True  # Evaluate at ans_loop=0 (pre-training baseline)
    eval_on_last_step: bool = True  # Always evaluate on final ans_loop
    n_samples: int = 1  # Samples per question (1 for speed)
    # Use public-eval-aligned prompt per benchmark (replaces the generic
    # training prompt with the benchmark's official evaluation prompt).
    use_public_eval_prompt: bool = False
    benchmarks: list[str] = field(
        default_factory=lambda: [
            "supergpqa_2000",
            "supergpqa_science_25_percent",
            "aime",
            "gpqa_diamond",
            "bbeh",
            "mmlu_pro",
            "hmmt",
            "medxpertqa",
            "olympiadbench",
            "encyclok",
        ]
    )
    max_questions: int = 0  # 0 = no limit; cap for large benchmarks


@dataclass
class WandbConfig:
    """Weights & Biases logging settings.

    Corresponds to v2's WandbConfig (train.py:396-403).
    """

    enabled: bool = True
    entity: str = ""
    project_name: str = "self-evolution-v3"
    group_name: Optional[str] = None
    run_name: Optional[str] = None
    run_id: Optional[str] = None


@dataclass
class DryRunCheckpointConfig:
    """Synthetic checkpoint size controls for CPU-only dry runs."""

    model_shard_mb: float = 1.0
    optim_shard_mb: float = 1.0
    hf_total_mb: float = 4.0
    hf_num_shards: int = 2
    # Optional exact HF shard sizes. When non-empty, overrides
    # ``hf_total_mb`` and ``hf_num_shards`` so the dry-run can mimic
    # uneven real-world shard layouts.
    hf_shard_sizes_mb: list[float] = field(default_factory=list)
    extra_state_kb: int = 8


@dataclass
class DryRunStageOutputConfig:
    """Serialized stage-output size targets for dry-run DataProto payloads."""

    dev_output_mb: float = 1.0
    gen_output_mb: float = 1.0
    gen_answer_output_mb: float = 1.0


@dataclass
class DryRunConfig:
    """CPU-only mock runtime for HF upload/resume plumbing validation."""

    enabled: bool = False
    backend: str = "mock_cpu"
    resume_loader: str = "mock"
    mock_world_size: int = 2
    crash_after: str = "none"
    checkpoint: DryRunCheckpointConfig = field(default_factory=DryRunCheckpointConfig)
    stage_outputs: DryRunStageOutputConfig = field(default_factory=DryRunStageOutputConfig)


@dataclass
class SelfEvolutionConfig:
    """Top-level configuration aggregating all sub-configs.

    The generator and solver model configs (ActorRolloutRefConfig) are
    provided via OmegaConf at runtime and are NOT defined here. They
    follow the same schema as verl's ``actor_rollout_ref`` config.

    Usage with Hydra::

        @hydra.main(config_path="config", config_name="self_evolution")
        def main(config):
            # config.training -> SelfEvolutionTrainingConfig
            # config.influence -> InfluenceConfig
            # config.generator -> verl ActorRolloutRefConfig (OmegaConf)
            # config.solver    -> verl ActorRolloutRefConfig (OmegaConf)
            ...
    """

    training: SelfEvolutionTrainingConfig = field(
        default_factory=SelfEvolutionTrainingConfig
    )
    influence: InfluenceConfig = field(default_factory=InfluenceConfig)
    spice: SpiceConfig = field(default_factory=SpiceConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    benchmark_eval: BenchmarkEvalConfig = field(default_factory=BenchmarkEvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    dry_run: DryRunConfig = field(default_factory=DryRunConfig)
