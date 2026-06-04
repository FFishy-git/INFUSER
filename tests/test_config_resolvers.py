from __future__ import annotations

import os
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_inf_evolve.templates import load_source
from verl_inf_evolve.utils.config_resolvers import register_config_template_resolvers


def test_pkg_template_resolver_loads_exact_file_contents() -> None:
    register_config_template_resolvers()

    cfg = OmegaConf.create(
        {"template": "${pkg_template:octothinker_shared_chat_template.jinja}"}
    )
    OmegaConf.resolve(cfg)

    assert cfg.template == load_source("octothinker_shared_chat_template.jinja")


def test_pkg_template_resolver_rejects_path_traversal() -> None:
    register_config_template_resolvers()

    cfg = OmegaConf.create({"template": "${pkg_template:../bad.jinja}"})
    with pytest.raises(ValueError, match="filename inside the package"):
        OmegaConf.resolve(cfg)


def test_pkg_template_resolver_rejects_non_jinja_files() -> None:
    register_config_template_resolvers()

    cfg = OmegaConf.create({"template": "${pkg_template:not_a_template.txt}"})
    with pytest.raises(ValueError, match=r"\.jinja"):
        OmegaConf.resolve(cfg)


def test_pkg_template_resolver_registration_is_idempotent() -> None:
    register_config_template_resolvers()
    register_config_template_resolvers()

    cfg = OmegaConf.create(
        {"template": "${pkg_template:octothinker_generator_chat_template.jinja}"}
    )
    OmegaConf.resolve(cfg)

    assert cfg.template == load_source("octothinker_generator_chat_template.jinja")
