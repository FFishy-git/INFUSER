"""Shared OmegaConf resolvers for package-owned config helpers."""

from __future__ import annotations

from omegaconf import OmegaConf

from verl_inf_evolve.templates import load_source


def _pkg_template_resolver(template_name: str) -> str:
    return load_source(template_name)


def register_config_template_resolvers() -> None:
    """Register config resolvers used by training and evaluation entrypoints."""
    if OmegaConf.has_resolver("pkg_template"):
        return
    OmegaConf.register_new_resolver("pkg_template", _pkg_template_resolver)
