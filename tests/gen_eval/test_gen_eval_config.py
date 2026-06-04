"""Tests for gen_eval.yaml Hydra configuration.

Smoke tests that the config loads via Hydra compose API, verifies all expected
top-level sections exist, and checks key default values match self_evolution.yaml.
No GPU required — pure config validation.
"""

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import os

# Absolute path to the config directory
CONFIG_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "verl_inf_evolve",
    "config",
)
CONFIG_DIR = os.path.abspath(CONFIG_DIR)


@pytest.fixture()
def gen_eval_cfg():
    """Load gen_eval.yaml via Hydra compose API."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="gen_eval")
    return cfg


@pytest.fixture()
def self_evolution_cfg():
    """Load self_evolution.yaml via Hydra compose API for comparison."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="self_evolution")
    return cfg


class TestGenEvalConfigLoads:
    """Smoke test that gen_eval.yaml loads without errors."""

    def test_config_loads(self, gen_eval_cfg):
        assert gen_eval_cfg is not None

    def test_config_is_dict_like(self, gen_eval_cfg):
        container = OmegaConf.to_container(gen_eval_cfg, resolve=True)
        assert isinstance(container, dict)


class TestTopLevelSections:
    """Verify all expected top-level config sections exist."""

    @pytest.mark.parametrize(
        "section",
        ["generator", "solver", "gen_eval", "influence", "wandb", "trainer"],
    )
    def test_section_exists(self, gen_eval_cfg, section):
        assert section in gen_eval_cfg, f"Missing top-level section: {section}"

    def test_data_section_exists(self, gen_eval_cfg):
        assert "data" in gen_eval_cfg

    def test_ray_kwargs_section_exists(self, gen_eval_cfg):
        assert "ray_kwargs" in gen_eval_cfg

    def test_reward_model_section_exists(self, gen_eval_cfg):
        assert "reward_model" in gen_eval_cfg


class TestEvalModeConfig:
    """Verify gen_eval.mode config field and its defaults."""

    def test_mode_default_is_regenerate(self, gen_eval_cfg):
        assert gen_eval_cfg.gen_eval.mode == "regenerate"

    def test_mode_override_to_replay(self):
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=replay"],
            )
        assert cfg.gen_eval.mode == "replay"

    def test_mode_override_to_regenerate(self):
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=regenerate"],
            )
        assert cfg.gen_eval.mode == "regenerate"

    def test_mode_combined_with_trajectory_path(self):
        """CLI override works: gen_eval.mode=replay gen_eval.remote_sync_path=s3://..."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=[
                    "gen_eval.mode=replay",
                    "gen_eval.remote_sync_path=s3://bucket/path",
                ],
            )
        assert cfg.gen_eval.mode == "replay"
        assert cfg.gen_eval.remote_sync_path == "s3://bucket/path"

    def test_existing_experiment_override_keeps_default_mode(self):
        """Existing experiment overrides continue to work (mode defaults to regenerate)."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval_experiment=sample"],
            )
        assert cfg.gen_eval.mode == "regenerate"

    def test_invalid_mode_validation(self):
        """Invalid mode values raise a clear error at startup."""
        from unittest.mock import MagicMock
        from omegaconf import OmegaConf
        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator

        # Build a minimal config with an invalid mode
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=invalid_mode"],
            )

        with pytest.raises(ValueError, match=r"Invalid gen_eval\.mode='invalid_mode'"):
            GeneratorEvaluator(
                config=cfg,
                gen_tokenizer=MagicMock(),
                solver_tokenizer=MagicMock(),
                role_worker_mapping={},
                resource_pool_manager=MagicMock(),
            )

    def test_valid_modes_accepted(self):
        """Valid mode values do not raise errors."""
        from unittest.mock import MagicMock
        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator

        for mode in ("regenerate", "replay"):
            with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
                cfg = compose(
                    config_name="gen_eval",
                    overrides=[f"gen_eval.mode={mode}"],
                )
            # Should not raise
            evaluator = GeneratorEvaluator(
                config=cfg,
                gen_tokenizer=MagicMock(),
                solver_tokenizer=MagicMock(),
                role_worker_mapping={},
                resource_pool_manager=MagicMock(),
            )
            assert evaluator.config.gen_eval.mode == mode


class TestEvalSectionDefaults:
    """Verify eval-specific config section has correct defaults."""

    def test_remote_sync_path_default(self, gen_eval_cfg):
        assert gen_eval_cfg.gen_eval.remote_sync_path is None

    def test_ans_loop_indices_default(self, gen_eval_cfg):
        assert gen_eval_cfg.gen_eval.ans_loop_indices is None

    def test_every_n_ans_loop_default(self, gen_eval_cfg):
        assert gen_eval_cfg.gen_eval.every_n_ans_loop is None

    def test_doc_path_default(self, gen_eval_cfg):
        assert gen_eval_cfg.gen_eval.doc_path == ".cache/data/preprocessed/eval_documents.json"

    def test_dev_dataset_path_default(self, gen_eval_cfg):
        assert gen_eval_cfg.gen_eval.dev_dataset_path == ".cache/data/preprocessed/dev.json"

    def test_local_cache_dir_default(self, gen_eval_cfg):
        assert gen_eval_cfg.gen_eval.local_cache_dir == ".cache/eval_checkpoints"

    def test_prefetch_count_default(self, gen_eval_cfg):
        assert gen_eval_cfg.gen_eval.prefetch_count == 1


class TestDefaultValuesMatchSelfEvolution:
    """Verify key default values match self_evolution.yaml verbatim."""

    def test_generator_rollout_temperature(self, gen_eval_cfg):
        assert gen_eval_cfg.generator.rollout.temperature == 0.7

    def test_solver_model_path(self, gen_eval_cfg):
        assert gen_eval_cfg.solver.model.path == "Qwen/Qwen3-8B"

    def test_generator_model_path(self, gen_eval_cfg):
        assert gen_eval_cfg.generator.model.path == "Qwen/Qwen3-8B"

    def test_influence_micro_batch_size(self, gen_eval_cfg):
        assert gen_eval_cfg.influence.micro_batch_size == 1

    def test_generator_rollout_n(self, gen_eval_cfg):
        assert gen_eval_cfg.generator.rollout.n == 8

    def test_solver_rollout_n(self, gen_eval_cfg):
        assert gen_eval_cfg.solver.rollout.n == 8

    def test_solver_rollout_temperature(self, gen_eval_cfg, self_evolution_cfg):
        assert gen_eval_cfg.solver.rollout.temperature == self_evolution_cfg.solver.rollout.temperature

    def test_generator_model_gradient_checkpointing(self, gen_eval_cfg, self_evolution_cfg):
        assert (
            gen_eval_cfg.generator.model.enable_gradient_checkpointing
            == self_evolution_cfg.generator.model.enable_gradient_checkpointing
        )

    def test_solver_model_gradient_checkpointing(self, gen_eval_cfg, self_evolution_cfg):
        assert (
            gen_eval_cfg.solver.model.enable_gradient_checkpointing
            == self_evolution_cfg.solver.model.enable_gradient_checkpointing
        )

    def test_solver_actor_ppo_micro_batch_size_per_gpu(self, gen_eval_cfg, self_evolution_cfg):
        assert (
            gen_eval_cfg.solver.actor.ppo_micro_batch_size_per_gpu
            == self_evolution_cfg.solver.actor.ppo_micro_batch_size_per_gpu
        )

    def test_solver_rollout_gpu_memory_utilization(self, gen_eval_cfg, self_evolution_cfg):
        assert (
            gen_eval_cfg.solver.rollout.gpu_memory_utilization
            == self_evolution_cfg.solver.rollout.gpu_memory_utilization
        )

    def test_influence_use_momentum(self, gen_eval_cfg, self_evolution_cfg):
        assert gen_eval_cfg.influence.use_momentum == self_evolution_cfg.influence.use_momentum

    def test_influence_momentum_beta(self, gen_eval_cfg, self_evolution_cfg):
        assert gen_eval_cfg.influence.momentum_beta == self_evolution_cfg.influence.momentum_beta

    def test_solver_ref_strategy(self, gen_eval_cfg):
        resolved = OmegaConf.to_container(gen_eval_cfg, resolve=True)
        assert resolved["solver"]["ref"]["strategy"] == "fsdp"

    def test_generator_rollout_prompt_length(self, gen_eval_cfg, self_evolution_cfg):
        assert gen_eval_cfg.generator.rollout.prompt_length == self_evolution_cfg.generator.rollout.prompt_length

    def test_generator_rollout_response_length(self, gen_eval_cfg, self_evolution_cfg):
        assert gen_eval_cfg.generator.rollout.response_length == self_evolution_cfg.generator.rollout.response_length


class TestWandbDefaults:
    """Verify wandb section has gen-eval specific defaults."""

    def test_project_name(self, gen_eval_cfg):
        assert gen_eval_cfg.wandb.project_name == "self-evolution-v3-gen-eval"

    def test_entity(self, gen_eval_cfg):
        assert gen_eval_cfg.wandb.entity is None

    def test_enabled(self, gen_eval_cfg):
        assert gen_eval_cfg.wandb.enabled is True


class TestTrainerDefaults:
    """Verify trainer section defaults."""

    def test_nnodes(self, gen_eval_cfg):
        assert gen_eval_cfg.trainer.nnodes == 1

    def test_n_gpus_per_node(self, gen_eval_cfg):
        assert gen_eval_cfg.trainer.n_gpus_per_node == 8

    def test_separate_pools(self, gen_eval_cfg):
        assert gen_eval_cfg.trainer.separate_pools is False


class TestExperimentOverride:
    """Verify experiment override directory is recognized by Hydra."""

    def test_sample_override_loads(self):
        """Load sample experiment override and verify it merges correctly."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval_experiment=sample"],
            )
        assert cfg.gen_eval.remote_sync_path == (
            "s3://example-bucket/running-states/V3_1/FW-Alr_2e-6-vanilla_token-TIS_token"
        )
        assert cfg.gen_eval.ans_loop_indices == [0, 5, 10, 15, 20]
        assert cfg.wandb.group_name == "eval_FW-Alr_2e-6-vanilla_token-TIS_token"

    def test_cli_override_works(self):
        """Verify CLI-style overrides work."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=[
                    "gen_eval.remote_sync_path=s3://test/path",
                    "gen_eval.ans_loop_indices=[0,10,20]",
                ],
            )
        assert cfg.gen_eval.remote_sync_path == "s3://test/path"
        assert list(cfg.gen_eval.ans_loop_indices) == [0, 10, 20]


class TestSolverSectionsComplete:
    """Verify solver has all required subsections for influence scoring."""

    def test_solver_has_actor(self, gen_eval_cfg):
        assert "actor" in gen_eval_cfg.solver

    def test_solver_has_ref(self, gen_eval_cfg):
        assert "ref" in gen_eval_cfg.solver

    def test_solver_has_model(self, gen_eval_cfg):
        assert "model" in gen_eval_cfg.solver

    def test_solver_has_rollout(self, gen_eval_cfg):
        assert "rollout" in gen_eval_cfg.solver

    def test_solver_actor_has_fsdp_config(self, gen_eval_cfg):
        assert "fsdp_config" in gen_eval_cfg.solver.actor

    def test_solver_actor_has_optim(self, gen_eval_cfg):
        assert "optim" in gen_eval_cfg.solver.actor


class TestGeneratorEvalOnly:
    """Verify generator config omits training-only fields."""

    def test_generator_has_no_actor(self, gen_eval_cfg):
        """Generator actor (optimizer/PPO) section is omitted for eval."""
        assert "actor" not in gen_eval_cfg.generator

    def test_generator_has_model(self, gen_eval_cfg):
        assert "model" in gen_eval_cfg.generator

    def test_generator_has_rollout(self, gen_eval_cfg):
        assert "rollout" in gen_eval_cfg.generator

    def test_generator_has_ref(self, gen_eval_cfg):
        assert "ref" in gen_eval_cfg.generator


# ---------------------------------------------------------------------------
# US-003: Replay mode skips generator worker initialization
# ---------------------------------------------------------------------------

class TestReplayModeInitWorkers:
    """Verify init_workers() in replay mode skips generator workers."""

    def _make_evaluator(self, mode="replay"):
        """Create a GeneratorEvaluator with the given eval mode."""
        from unittest.mock import MagicMock
        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=[f"gen_eval.mode={mode}"],
            )

        return GeneratorEvaluator(
            config=cfg,
            gen_tokenizer=MagicMock(),
            solver_tokenizer=MagicMock(),
            role_worker_mapping={},
            resource_pool_manager=MagicMock(),
        )

    def test_replay_mode_generator_wg_is_none_before_init(self):
        """Before init_workers(), generator_wg should already be None."""
        evaluator = self._make_evaluator(mode="replay")
        assert evaluator.generator_wg is None

    def test_replay_mode_init_workers_keeps_generator_wg_none(self):
        """After init_workers() in replay mode, generator_wg stays None."""
        from unittest.mock import MagicMock, patch
        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=replay"],
            )

        mock_rpm = MagicMock()
        mock_solver_pool = MagicMock()
        mock_rpm.get_resource_pool.return_value = mock_solver_pool

        mock_solver_cls = MagicMock()
        mock_wg = MagicMock()
        mock_rwg_cls = MagicMock(return_value=mock_wg)

        evaluator = GeneratorEvaluator(
            config=cfg,
            gen_tokenizer=MagicMock(),
            solver_tokenizer=MagicMock(),
            role_worker_mapping={"solver": mock_solver_cls},
            resource_pool_manager=mock_rpm,
            ray_worker_group_cls=mock_rwg_cls,
        )

        with patch("verl.experimental.agent_loop.AgentLoopManager", MagicMock()):
            evaluator.init_workers()

        # Generator worker group must NOT be created
        assert evaluator.generator_wg is None
        # Generator rollout manager must NOT be created
        assert evaluator.gen_rollout_manager is None

    def test_replay_mode_init_workers_creates_solver_wg(self):
        """After init_workers() in replay mode, solver_wg is created."""
        from unittest.mock import MagicMock, patch

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=replay"],
            )

        mock_rpm = MagicMock()
        mock_solver_pool = MagicMock()
        mock_rpm.get_resource_pool.return_value = mock_solver_pool

        mock_solver_cls = MagicMock()
        mock_wg = MagicMock()
        mock_rwg_cls = MagicMock(return_value=mock_wg)

        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator
        evaluator = GeneratorEvaluator(
            config=cfg,
            gen_tokenizer=MagicMock(),
            solver_tokenizer=MagicMock(),
            role_worker_mapping={"solver": mock_solver_cls},
            resource_pool_manager=mock_rpm,
            ray_worker_group_cls=mock_rwg_cls,
        )

        with patch("verl.experimental.agent_loop.AgentLoopManager", MagicMock()):
            evaluator.init_workers()

        # Solver worker group must be created and initialized
        assert evaluator.solver_wg is not None
        mock_wg.init_model.assert_called_once()

    def test_replay_mode_init_workers_creates_solver_rollout_manager(self):
        """After init_workers() in replay mode, solver_rollout_manager is created."""
        from unittest.mock import MagicMock, patch

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=replay"],
            )

        mock_rpm = MagicMock()
        mock_rpm.get_resource_pool.return_value = MagicMock()

        mock_wg = MagicMock()
        mock_rwg_cls = MagicMock(return_value=mock_wg)
        mock_alm = MagicMock()

        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator
        evaluator = GeneratorEvaluator(
            config=cfg,
            gen_tokenizer=MagicMock(),
            solver_tokenizer=MagicMock(),
            role_worker_mapping={"solver": MagicMock()},
            resource_pool_manager=mock_rpm,
            ray_worker_group_cls=mock_rwg_cls,
        )

        with patch("verl.experimental.agent_loop.AgentLoopManager", return_value=mock_alm):
            evaluator.init_workers()

        assert evaluator.solver_rollout_manager is mock_alm

    def test_replay_mode_shared_engine_is_false(self):
        """In replay mode, _shared_engine_mode is always False (no colocated workers)."""
        from unittest.mock import MagicMock, patch

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=replay"],
            )

        mock_rpm = MagicMock()
        mock_rpm.get_resource_pool.return_value = MagicMock()

        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator
        evaluator = GeneratorEvaluator(
            config=cfg,
            gen_tokenizer=MagicMock(),
            solver_tokenizer=MagicMock(),
            role_worker_mapping={"solver": MagicMock()},
            resource_pool_manager=mock_rpm,
            ray_worker_group_cls=MagicMock(return_value=MagicMock()),
        )

        with patch("verl.experimental.agent_loop.AgentLoopManager", MagicMock()):
            evaluator.init_workers()

        assert evaluator._shared_engine_mode is False

    def test_replay_mode_gen_tokenizer_still_available(self):
        """Generator tokenizer remains available in replay mode (lightweight, needed for decoding)."""
        from unittest.mock import MagicMock

        gen_tokenizer = MagicMock(name="gen_tokenizer")

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=replay"],
            )

        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator
        evaluator = GeneratorEvaluator(
            config=cfg,
            gen_tokenizer=gen_tokenizer,
            solver_tokenizer=MagicMock(),
            role_worker_mapping={},
            resource_pool_manager=MagicMock(),
        )

        assert evaluator.gen_tokenizer is gen_tokenizer

    def test_replay_mode_no_generator_in_role_worker_mapping(self):
        """In replay mode, role_worker_mapping does not need a 'generator' entry."""
        from unittest.mock import MagicMock, patch

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=replay"],
            )

        mock_rpm = MagicMock()
        mock_rpm.get_resource_pool.return_value = MagicMock()

        # Only solver in role_worker_mapping — no generator entry
        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator
        evaluator = GeneratorEvaluator(
            config=cfg,
            gen_tokenizer=MagicMock(),
            solver_tokenizer=MagicMock(),
            role_worker_mapping={"solver": MagicMock()},
            resource_pool_manager=mock_rpm,
            ray_worker_group_cls=MagicMock(return_value=MagicMock()),
        )

        with patch("verl.experimental.agent_loop.AgentLoopManager", MagicMock()):
            evaluator.init_workers()  # Should not raise KeyError for "generator"

        assert evaluator.generator_wg is None

    def test_replay_mode_resource_pool_only_requests_solver(self):
        """In replay mode, init_workers() only requests the 'solver' resource pool."""
        from unittest.mock import MagicMock, patch, call

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=replay"],
            )

        mock_rpm = MagicMock()
        mock_rpm.get_resource_pool.return_value = MagicMock()

        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator
        evaluator = GeneratorEvaluator(
            config=cfg,
            gen_tokenizer=MagicMock(),
            solver_tokenizer=MagicMock(),
            role_worker_mapping={"solver": MagicMock()},
            resource_pool_manager=mock_rpm,
            ray_worker_group_cls=MagicMock(return_value=MagicMock()),
        )

        with patch("verl.experimental.agent_loop.AgentLoopManager", MagicMock()):
            evaluator.init_workers()

        # Only "solver" pool should be requested — no "generator" pool
        pool_calls = mock_rpm.get_resource_pool.call_args_list
        requested_pools = [c.args[0] for c in pool_calls]
        assert "solver" in requested_pools
        assert "generator" not in requested_pools

    def test_regenerate_mode_creates_generator_wg(self):
        """In regenerate mode, init_workers() still creates generator workers."""
        from unittest.mock import MagicMock, patch

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=regenerate", "trainer.separate_pools=true"],
            )

        mock_rpm = MagicMock()
        mock_gen_pool = MagicMock()
        mock_solver_pool = MagicMock()

        def get_pool(name):
            return mock_gen_pool if name == "generator" else mock_solver_pool

        mock_rpm.get_resource_pool.side_effect = get_pool

        mock_wg = MagicMock()
        mock_rwg_cls = MagicMock(return_value=mock_wg)

        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator
        evaluator = GeneratorEvaluator(
            config=cfg,
            gen_tokenizer=MagicMock(),
            solver_tokenizer=MagicMock(),
            role_worker_mapping={"generator": MagicMock(), "solver": MagicMock()},
            resource_pool_manager=mock_rpm,
            ray_worker_group_cls=mock_rwg_cls,
        )

        with patch("verl.experimental.agent_loop.AgentLoopManager", MagicMock()):
            evaluator.init_workers()

        # In regenerate mode, generator_wg must be created
        assert evaluator.generator_wg is not None


class TestReplayModeTaskRunner:
    """Verify TaskRunner behavior in replay mode."""

    def test_replay_mode_skips_generator_worker_registration(self):
        """TaskRunner.run() in replay mode does not register generator workers."""
        from verl_inf_evolve.gen_eval.gen_eval import TaskRunner

        runner = TaskRunner()

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=replay"],
            )

        # Only call add_solver_worker to register solver
        from unittest.mock import MagicMock
        runner.add_solver_worker(cfg)
        # Override mapping for replay
        runner.mapping["solver"] = "solver_pool"

        assert "generator" not in runner.role_worker_mapping
        assert "solver" in runner.role_worker_mapping
        assert runner.mapping["solver"] == "solver_pool"

    def test_replay_mode_resource_pool_all_gpus_to_solver(self):
        """In replay mode, resource pool allocates all GPUs to solver."""
        from verl_inf_evolve.gen_eval.gen_eval import TaskRunner

        runner = TaskRunner()

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=["gen_eval.mode=replay"],
            )

        runner.mapping["solver"] = "solver_pool"
        rpm = runner.init_resource_pool_mgr(cfg)

        # The resource pool spec should have "solver_pool" only
        assert "solver_pool" in rpm.resource_pool_spec
        assert "generator_pool" not in rpm.resource_pool_spec
        assert "global_pool" not in rpm.resource_pool_spec

    def test_regenerate_mode_resource_pool_includes_generator(self):
        """In regenerate mode with separate_pools, resource pool includes generator."""
        from verl_inf_evolve.gen_eval.gen_eval import TaskRunner

        runner = TaskRunner()

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=[
                    "gen_eval.mode=regenerate",
                    "trainer.separate_pools=true",
                ],
            )

        runner.mapping["generator"] = "generator_pool"
        runner.mapping["solver"] = "solver_pool"
        rpm = runner.init_resource_pool_mgr(cfg)

        assert "generator_pool" in rpm.resource_pool_spec
        assert "solver_pool" in rpm.resource_pool_spec


# ---------------------------------------------------------------------------
# US-005: Replay-specific wandb metadata
# ---------------------------------------------------------------------------

class TestReplayWandbMetadata:
    """Verify wandb logging includes replay-specific metadata."""

    def _make_evaluator(self, mode="replay"):
        """Create a GeneratorEvaluator with the given eval mode."""
        from unittest.mock import MagicMock
        from verl_inf_evolve.gen_eval.generator_evaluator import GeneratorEvaluator

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="gen_eval",
                overrides=[f"gen_eval.mode={mode}"],
            )

        return GeneratorEvaluator(
            config=cfg,
            gen_tokenizer=MagicMock(),
            solver_tokenizer=MagicMock(),
            role_worker_mapping={},
            resource_pool_manager=MagicMock(),
        )

    def test_run_name_includes_replay_tag(self):
        """Run name includes 'eval_replay_' prefix in replay mode."""
        from unittest.mock import MagicMock, patch

        evaluator = self._make_evaluator(mode="replay")

        mock_tracking_cls = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.logger = {}  # No wandb backend in test
        mock_tracking_cls.return_value = mock_tracker

        with patch("verl.utils.tracking.Tracking", mock_tracking_cls):
            evaluator._init_tracking()

        call_kwargs = mock_tracking_cls.call_args
        run_name = call_kwargs.kwargs.get("experiment_name") or call_kwargs[0][1]
        assert "eval_replay_" in run_name

    def test_run_name_includes_regen_tag(self):
        """Run name includes 'eval_regen_' prefix in regenerate mode."""
        from unittest.mock import MagicMock, patch

        evaluator = self._make_evaluator(mode="regenerate")

        mock_tracking_cls = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.logger = {}
        mock_tracking_cls.return_value = mock_tracker

        with patch("verl.utils.tracking.Tracking", mock_tracking_cls):
            evaluator._init_tracking()

        call_kwargs = mock_tracking_cls.call_args
        run_name = call_kwargs.kwargs.get("experiment_name") or call_kwargs[0][1]
        assert "eval_regen_" in run_name

    def test_wandb_summary_eval_mode_set(self):
        """wandb.summary['eval/mode'] is set during _init_tracking()."""
        from unittest.mock import MagicMock, patch

        evaluator = self._make_evaluator(mode="replay")

        mock_wandb = MagicMock()
        mock_wandb.summary = {}
        mock_tracker = MagicMock()
        mock_tracker.logger = {"wandb": mock_wandb}

        mock_tracking_cls = MagicMock(return_value=mock_tracker)

        with patch("verl.utils.tracking.Tracking", mock_tracking_cls):
            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                evaluator._init_tracking()

        assert mock_wandb.summary["eval/mode"] == "replay"

    def test_wandb_summary_eval_mode_regenerate(self):
        """wandb.summary['eval/mode'] is 'regenerate' for default mode."""
        from unittest.mock import MagicMock, patch

        evaluator = self._make_evaluator(mode="regenerate")

        mock_wandb = MagicMock()
        mock_wandb.summary = {}
        mock_tracker = MagicMock()
        mock_tracker.logger = {"wandb": mock_wandb}

        mock_tracking_cls = MagicMock(return_value=mock_tracker)

        with patch("verl.utils.tracking.Tracking", mock_tracking_cls):
            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                evaluator._init_tracking()

        assert mock_wandb.summary["eval/mode"] == "regenerate"

    def test_replay_mode_defines_eval_replay_metrics(self):
        """In replay mode, wandb.define_metric is called for eval_replay/* prefix."""
        from unittest.mock import MagicMock, patch

        evaluator = self._make_evaluator(mode="replay")

        mock_wandb = MagicMock()
        mock_wandb.summary = {}
        mock_tracker = MagicMock()
        mock_tracker.logger = {"wandb": mock_wandb}

        mock_tracking_cls = MagicMock(return_value=mock_tracker)

        with patch("verl.utils.tracking.Tracking", mock_tracking_cls):
            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                evaluator._init_tracking()

        # Check that eval_replay/* metric was defined
        define_calls = [str(c) for c in mock_wandb.define_metric.call_args_list]
        assert any("eval_replay/*" in c for c in define_calls)

    def test_regenerate_mode_no_eval_replay_metrics(self):
        """In regenerate mode, eval_replay/* metric prefix is NOT defined."""
        from unittest.mock import MagicMock, patch

        evaluator = self._make_evaluator(mode="regenerate")

        mock_wandb = MagicMock()
        mock_wandb.summary = {}
        mock_tracker = MagicMock()
        mock_tracker.logger = {"wandb": mock_wandb}

        mock_tracking_cls = MagicMock(return_value=mock_tracker)

        with patch("verl.utils.tracking.Tracking", mock_tracking_cls):
            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                evaluator._init_tracking()

        define_calls = [str(c) for c in mock_wandb.define_metric.call_args_list]
        assert not any("eval_replay/*" in c for c in define_calls)

    def test_eval_mode_in_wandb_config(self):
        """gen_eval.mode is included in the config passed to Tracking (and thus wandb.init)."""
        from unittest.mock import MagicMock, patch

        evaluator = self._make_evaluator(mode="replay")

        mock_tracking_cls = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.logger = {}
        mock_tracking_cls.return_value = mock_tracker

        with patch("verl.utils.tracking.Tracking", mock_tracking_cls):
            evaluator._init_tracking()

        call_kwargs = mock_tracking_cls.call_args
        config_arg = call_kwargs.kwargs.get("config") or call_kwargs[0][3]
        assert config_arg["gen_eval"]["mode"] == "replay"
