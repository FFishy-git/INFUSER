#!/usr/bin/env python3
"""Resolve training.remote_sync_path from Hydra overrides and config YAMLs.

Called by launcher scripts to determine the remote backend target when
``remote_sync_path`` is defined in YAML configs and/or partially overridden
from the CLI.

Usage:
    python3 launcher/resolve_sync_path.py "+experiment_qwen3_4b=FW-..." "--config-name=foo ..."
"""
import re
import sys

import yaml


def _extract_override(overrides: str, key: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S+)", overrides)
    if match:
        return match.group(1)
    return ""


def main():
    overrides = " ".join(sys.argv[1:])
    explicit_group_name = _extract_override(overrides, "wandb.group_name")
    explicit_sync_path = (
        _extract_override(overrides, "eval.remote_sync_path")
        or _extract_override(overrides, "training.remote_sync_path")
        or _extract_override(overrides, "remote_sync_path")
    )

    # Collect candidate config files: experiment packages first, then base config
    # Matches both "+experiment_xxx=NAME" and "experiment=NAME" (optional + prefix)
    configs = []
    base_config = "verl_inf_evolve/config/self_evolution.yaml"
    for m in re.finditer(r"\+?(\w+)=(\S+)", overrides):
        pkg, name = m.group(1), m.group(2)
        # experiment packages, gen_eval_experiment, sol_eval_experiment
        if pkg.startswith("experiment") or pkg in ("gen_eval_experiment", "sol_eval_experiment"):
            configs.append(f"verl_inf_evolve/config/{pkg}/{name}.yaml")
        if pkg == "gen_eval_experiment":
            base_config = "verl_inf_evolve/config/gen_eval.yaml"
        elif pkg == "sol_eval_experiment":
            base_config = "verl_inf_evolve/config/sol_eval.yaml"
    configs.append(base_config)

    # First pass: read group_name only from experiment configs (not the base
    # config) so the base config's placeholder "default" is never used for
    # interpolation.
    group_name = explicit_group_name
    for path in configs[:-1]:  # experiment configs only
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            if not group_name:
                group_name = (data.get("wandb") or {}).get("group_name", "")
        except Exception:
            continue

    if not group_name:
        try:
            with open(configs[-1]) as f:
                data = yaml.safe_load(f)
            if data:
                group_name = (data.get("wandb") or {}).get("group_name", "")
        except Exception:
            pass

    # Second pass: read sync_path from all configs (experiment first, then base)
    # Check both training.remote_sync_path and eval.remote_sync_path (sol_eval)
    sync_path = explicit_sync_path
    for path in configs:
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            if not sync_path:
                sync_path = (
                    (data.get("training") or {}).get("remote_sync_path", "")
                    or (data.get("eval") or {}).get("remote_sync_path", "")
                )
        except Exception:
            continue

    if sync_path and group_name:
        sync_path = sync_path.replace("${wandb.group_name}", group_name)

    # Only print if fully resolved (no remaining interpolations)
    if sync_path and "${" not in sync_path:
        print(sync_path)


if __name__ == "__main__":
    main()
