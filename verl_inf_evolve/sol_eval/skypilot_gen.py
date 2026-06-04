"""SkyPilot task YAML generation for the V3 evaluation pipeline.

Generates one SkyPilot YAML per checkpoint from a config namespace and a
template file with placeholder substitution. Each job evaluates all benchmarks
for that checkpoint.
"""

from __future__ import annotations

import re
import shlex
from datetime import datetime
from pathlib import Path

# Directory containing templates (skypilot_eval.yaml.j2, etc.)
_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Default GPU type and count
_DEFAULT_ACCELERATOR = "H100"
_DEFAULT_WANDB_ENTITY = ""


def _temperature_override(run_config) -> float:
    """Read the rollout temperature from the canonical solver config."""
    solver = getattr(run_config, "solver", None)
    rollout = getattr(solver, "rollout", None)
    if rollout is not None and hasattr(rollout, "temperature"):
        return float(rollout.temperature)
    return float(run_config.temperature)


def _response_length_override(run_config) -> int:
    """Read the rollout response length from the canonical solver config."""
    solver = getattr(run_config, "solver", None)
    rollout = getattr(solver, "rollout", None)
    if rollout is not None and hasattr(rollout, "response_length"):
        return int(rollout.response_length)
    return int(run_config.response_length)


def _load_template(template_name: str = "skypilot_eval") -> str:
    """Load a SkyPilot template file by name.

    Args:
        template_name: Template filename without extension (e.g. 'skypilot_eval').

    Returns:
        Template string with {placeholders}.

    Raises:
        FileNotFoundError: If template file does not exist.
    """
    path = _TEMPLATES_DIR / f"{template_name}.yaml.j2"
    if not path.exists():
        raise FileNotFoundError(
            f"SkyPilot template not found: {path}. "
            f"Available templates: {', '.join(t.stem for t in _TEMPLATES_DIR.glob('*.yaml.j2'))}"
        )
    return path.read_text()


def _slugify(name: str) -> str:
    """Convert a run name to a slug safe for SkyPilot job names.

    Replaces non-alphanumeric characters (except hyphens) with hyphens,
    lowercases, and strips leading/trailing hyphens.
    """
    slug = re.sub(r"[^a-zA-Z0-9-]", "-", name).lower().strip("-")
    # Collapse multiple consecutive hyphens
    slug = re.sub(r"-+", "-", slug)
    return slug


def _build_eval_command(
    run_config,
    ckpt_num: int,
    benchmarks: list[str],
    n_gpus: int,
) -> str:
    """Build the Hydra eval command for a single checkpoint with all benchmarks."""
    benchmarks_str = ",".join(benchmarks)
    overrides = [
        f"eval.run_name={run_config.run_name}",
        f"eval.checkpoints=[{ckpt_num}]",
        f"eval.benchmarks=[{benchmarks_str}]",
        f"eval.model_path={run_config.model_path}",
        f"solver.model.path={run_config.model_path}",
        f"solver.rollout.temperature={_temperature_override(run_config)}",
        f"solver.rollout.response_length={_response_length_override(run_config)}",
        f"eval.n_samples={run_config.n_samples}",
        f"eval.tp_size={run_config.tp_size}",
        f"trainer.n_gpus_per_node={n_gpus}",
    ]

    remote_sync_path = getattr(run_config, "remote_sync_path", None)
    if remote_sync_path:
        overrides.append(f"eval.remote_sync_path={remote_sync_path}")

    gpu_mem_util = getattr(run_config, "gpu_memory_utilization", None)
    if gpu_mem_util is not None:
        overrides.append(f"eval.gpu_memory_utilization={gpu_mem_util}")

    result_detail = getattr(run_config, "result_detail", None)
    if result_detail:
        overrides.append(f"eval.result_detail={result_detail}")

    remote_eval_base = getattr(run_config, "remote_eval_base", None)
    if remote_eval_base:
        overrides.append(f"eval.remote_eval_base={remote_eval_base}")

    return "python -m verl_inf_evolve.sol_eval.sol_eval " + " ".join(
        shlex.quote(override) for override in overrides
    )


def generate_skypilot_tasks(
    run_config,
    output_dir: str | None = None,
    template_name: str = "skypilot_eval",
    n_gpus: int | None = None,
    accelerator: str = _DEFAULT_ACCELERATOR,
    wandb_entity: str = _DEFAULT_WANDB_ENTITY,
) -> str:
    """Generate SkyPilot task YAML files for all checkpoints.

    One YAML is created per checkpoint. Each job evaluates all benchmarks
    for that checkpoint.

    Args:
        run_config: Config namespace with eval params (run_name, checkpoints,
            benchmarks, tp_size). Accepts Hydra config.eval or SimpleNamespace.
        output_dir: Directory to write YAMLs into. If None, uses
            verl_inf_evolve/sol_eval/skypilot_tasks_{timestamp}/.
        template_name: Name of the template file (without .yaml.j2 extension).
        n_gpus: Number of GPUs per job. Defaults to run_config.tp_size.
        accelerator: GPU accelerator type (e.g. 'H100', 'A100').
        wandb_entity: WandB entity for logging.

    Returns:
        Path to the output directory containing generated YAMLs.
    """
    template = _load_template(template_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = f"verl_inf_evolve/sol_eval/skypilot_tasks_{timestamp}"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if n_gpus is None:
        n_gpus = run_config.tp_size

    run_name_slug = _slugify(run_config.run_name)
    benchmarks_str = ", ".join(run_config.benchmarks)

    tasks_generated = 0
    for ckpt_num in run_config.checkpoints:
        command = _build_eval_command(
            run_config=run_config,
            ckpt_num=ckpt_num,
            benchmarks=run_config.benchmarks,
            n_gpus=n_gpus,
        )

        yaml_content = template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_name=run_config.run_name,
            run_name_slug=run_name_slug,
            ckpt_num=ckpt_num,
            benchmarks_str=benchmarks_str,
            accelerator=accelerator,
            n_gpus=n_gpus,
            wandb_entity=wandb_entity,
            command=command,
        )

        yaml_filename = f"eval_{run_config.run_name}_ckpt{ckpt_num}.yaml"
        yaml_path = out_path / yaml_filename
        yaml_path.write_text(yaml_content)
        tasks_generated += 1

    print(f"Generated {tasks_generated} SkyPilot task YAMLs in {out_path}/")
    return str(out_path)
