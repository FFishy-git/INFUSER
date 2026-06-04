"""SLURM job script generation for the V3 evaluation pipeline.

Generates one SLURM script per (checkpoint, benchmark) pair from a config
namespace and a template file with placeholder substitution.
"""

from __future__ import annotations

import os
import shlex
from datetime import datetime
from pathlib import Path

# Directory containing SLURM templates (default.txt, etc.)
_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Defaults for SLURM directives
_DEFAULT_TIME_LIMIT = "48:00:00"
_DEFAULT_PARTITION = "gpu"


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


def _load_template(template_name: str) -> str:
    """Load a SLURM template file by name.

    Args:
        template_name: Template filename without extension (e.g. 'default').

    Returns:
        Template string with {placeholders}.

    Raises:
        FileNotFoundError: If template file does not exist.
    """
    path = _TEMPLATES_DIR / f"{template_name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"SLURM template not found: {path}. "
            f"Available templates: {', '.join(t.stem for t in _TEMPLATES_DIR.glob('*.txt'))}"
        )
    return path.read_text()


def _build_eval_command(
    run_config,
    ckpt_num: int,
    benchmark: str,
    gpus_per_node: int,
) -> str:
    """Build the Hydra eval command for a single (checkpoint, benchmark) pair."""
    overrides = [
        f"eval.run_name={run_config.run_name}",
        f"eval.checkpoints=[{ckpt_num}]",
        f"eval.benchmarks=[{benchmark}]",
        f"eval.model_path={run_config.model_path}",
        f"solver.model.path={run_config.model_path}",
        f"solver.rollout.temperature={_temperature_override(run_config)}",
        f"solver.rollout.response_length={_response_length_override(run_config)}",
        f"eval.n_samples={run_config.n_samples}",
        f"eval.tp_size={run_config.tp_size}",
        f"trainer.n_gpus_per_node={gpus_per_node}",
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


def generate_slurm_scripts(
    run_config,
    template_name: str = "default",
    output_dir: str | None = None,
    gpus_per_node: int | None = None,
    time_limit: str = _DEFAULT_TIME_LIMIT,
    partition: str = _DEFAULT_PARTITION,
) -> str:
    """Generate SLURM scripts for all (checkpoint, benchmark) pairs.

    One script is created per (checkpoint, benchmark) pair.

    Args:
        run_config: Config namespace with eval params (run_name, checkpoints,
            benchmarks, tp_size, model_path) and canonical solver rollout
            overrides such as ``solver.rollout.temperature`` and
            ``solver.rollout.response_length``.
            Accepts Hydra config.eval or SimpleNamespace.
        template_name: Name of the template file (without .txt extension).
        output_dir: Directory to write scripts into. If None, uses
            verl_inf_evolve/sol_eval/slurm_scripts_{timestamp}/.
        gpus_per_node: Number of GPUs per node. Defaults to run_config.tp_size.
        time_limit: SLURM time limit string (e.g. '48:00:00').
        partition: SLURM partition name.

    Returns:
        Path to the output directory containing generated scripts.
    """
    template = _load_template(template_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = f"verl_inf_evolve/sol_eval/slurm_scripts_{timestamp}"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if gpus_per_node is None:
        gpus_per_node = run_config.tp_size

    scripts_generated = 0
    for ckpt_num in run_config.checkpoints:
        for benchmark in run_config.benchmarks:
            job_name = f"eval_{run_config.run_name}_ckpt{ckpt_num}_{benchmark}"
            checkpoint_name = f"global_step_{ckpt_num}"

            command = _build_eval_command(
                run_config=run_config,
                ckpt_num=ckpt_num,
                benchmark=benchmark,
                gpus_per_node=gpus_per_node,
            )

            script_content = template.format(
                job_name=job_name,
                gpus_per_node=gpus_per_node,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                base_path=run_config.model_path,
                checkpoint_name=checkpoint_name,
                benchmark=benchmark,
                temperature=_temperature_override(run_config),
                response_length=_response_length_override(run_config),
                command=command,
                time_limit=time_limit,
                partition=partition,
            )

            script_filename = f"{job_name}.sh"
            script_path = out_path / script_filename
            script_path.write_text(script_content)
            os.chmod(script_path, 0o755)
            scripts_generated += 1

    print(f"Generated {scripts_generated} SLURM scripts in {out_path}/")
    return str(out_path)
