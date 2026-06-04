"""Pre-download model files before Ray workers start.

Multiple Ray workers simultaneously downloading a model for the first time
can corrupt the HuggingFace cache, causing 'Unrecognized model' or
'FileNotFoundError' for weight shards.  This script downloads the full
model snapshot (config, tokenizer *and* weights) sequentially so that all
workers find a complete, non-corrupted cache entry.

Downloading only config/tokenizer (ignoring .safetensors) is NOT safe:
it creates a snapshot directory that `from_pretrained()` treats as a
complete cache hit, then fails with FileNotFoundError when the weight
files are missing.
"""

import os
import sys

import yaml
from huggingface_hub import snapshot_download


def find_model_paths(hydra_overrides: str) -> set[str]:
    """Extract model paths from Hydra overrides and referenced config files."""
    config_dir = "verl_inf_evolve/config"
    models: set[str] = set()

    # Keys whose values are HF model IDs (org/model format).
    _model_path_keys = {
        "eval.model_path",
        "solver.model.path",
        "generator.model.path",
    }

    for arg in hydra_overrides.split():
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        key = key.lstrip("+")

        # Direct model path overrides (e.g. eval.model_path=Qwen/Qwen3-8B-Base)
        if key in _model_path_keys and "/" in val and not val.startswith("/"):
            models.add(val)

        config_path = os.path.join(config_dir, key, val + ".yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            for role in ("generator", "solver"):
                mp = (cfg.get(role) or {}).get("model") or {}
                if isinstance(mp, dict) and "path" in mp:
                    models.add(mp["path"])
            # sol_eval experiment configs store model path under eval.model_path
            eval_model = (cfg.get("eval") or {}).get("model_path")
            if eval_model:
                models.add(eval_model)

    # Also check base configs for defaults
    for base in ["gen_eval.yaml", "self_evolution.yaml", "sol_eval.yaml"]:
        bp = os.path.join(config_dir, base)
        if not os.path.exists(bp):
            continue
        with open(bp) as f:
            cfg = yaml.safe_load(f) or {}
        for role in ("generator", "solver"):
            mp = (cfg.get(role) or {}).get("model") or {}
            if isinstance(mp, dict) and "path" in mp:
                models.add(mp["path"])
        # sol_eval stores model path under eval.model_path
        eval_model = (cfg.get("eval") or {}).get("model_path")
        if eval_model:
            models.add(eval_model)

    return models


def main() -> None:
    overrides = sys.argv[1] if len(sys.argv) > 1 else ""
    models = find_model_paths(overrides)

    for m in sorted(models):
        if "/" not in m or m.startswith("/"):
            continue
        print(f"Pre-downloading {m} (full snapshot including weights)...")
        snapshot_download(m)
        print(f"  {m} fully cached.")


if __name__ == "__main__":
    main()
