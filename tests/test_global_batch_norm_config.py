import os
import sys

from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.trainer.self_evolution_trainer import SelfEvolutionTrainer


def _make_trainer(algorithm_cfg):
    trainer = object.__new__(SelfEvolutionTrainer)
    trainer.config = OmegaConf.create({"algorithm": algorithm_cfg})
    return trainer


def test_global_batch_norm_defaults_to_shared_flag_for_generator_and_solver():
    trainer = _make_trainer({"use_global_batch_norm": False})

    assert trainer._use_global_batch_norm("generator") is False
    assert trainer._use_global_batch_norm("solver") is False


def test_generator_override_wins_over_shared_flag():
    trainer = _make_trainer({
        "use_global_batch_norm": False,
        "generator_use_global_batch_norm": True,
    })

    assert trainer._use_global_batch_norm("generator") is True
    assert trainer._use_global_batch_norm("solver") is False


def test_solver_override_wins_over_shared_flag():
    trainer = _make_trainer({
        "use_global_batch_norm": True,
        "solver_use_global_batch_norm": False,
    })

    assert trainer._use_global_batch_norm("generator") is True
    assert trainer._use_global_batch_norm("solver") is False
