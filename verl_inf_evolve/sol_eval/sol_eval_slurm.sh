#!/bin/bash
#SBATCH --job-name=sol-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch [slurm options] verl_inf_evolve/sol_eval/sol_eval_slurm.sh \
    --run-name RUN_NAME \
    --model MODEL_PATH_OR_ALIAS \
    [--checkpoints CHECKPOINT_SPEC] \
    [--config-path TRAINING_CONFIG_YAML] \
    [--ans-loop N] \
    [--benchmarks BENCHMARK_LIST] \
    [--remote-sync-path REMOTE_URI] \
    [--remote-eval-base REMOTE_URI] \
    [--hydra-override KEY=VALUE] ...

Examples:
  sbatch --gres=gpu:1 verl_inf_evolve/sol_eval/sol_eval_slurm.sh \
    --run-name FW-Alr_2e-6-Glr_2e-6-DrGRPO-TIS_token-dev_800-precond_cos \
    --model qwen3_4b_base \
    --checkpoints '[40]' \
    --benchmarks '[aime,gpqa_diamond]'

  sbatch --gres=gpu:2 --partition=gpu verl_inf_evolve/sol_eval/sol_eval_slurm.sh \
    --run-name my-run \
    --remote-sync-path hf://datasets/my-org/SER/qwen3_4b_base/my-run \
    --checkpoints '0:100:20' \
    --hydra-override eval.force=true

  sbatch --gres=gpu:1 verl_inf_evolve/sol_eval/sol_eval_slurm.sh \
    --config-path verl_inf_evolve/config/experiment_qwen3_4b_base/my-run.yaml \
    --ans-loop 85

Notes:
  - If --remote-sync-path is omitted, sol_eval will auto-discover it from the
    local training experiment YAMLs using --run-name and --model.
  - --config-path can be used instead of --run-name/--model to resolve
    run_name, model_path, and remote_sync_path directly from a training YAML.
  - --ans-loop is a convenience alias for a single solver checkpoint
    (maps directly to global_step_{N}).
  - --model is required when --remote-sync-path is omitted because the same
    run_name may exist under multiple model families.
  - To load a sol_eval experiment config, use:
    --hydra-override sol_eval_experiment=<name>
    (loads verl_inf_evolve/config/sol_eval_experiment/<name>.yaml)
  - trainer.n_gpus_per_node is auto-set from the Slurm allocation.
  - Result JSONs are metrics-only by default. To save full per-question
    outputs, add: --hydra-override eval.result_detail=full
  - The benchmark JSONs are expected under .cache/data/preprocessed/benchmarks.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_value() {
  local opt="$1"
  local value="${2:-}"
  [[ -n "${value}" ]] || die "${opt} requires a value"
}

extract_last_integer() {
  local raw="$1"
  python - "$raw" <<'PY'
import re
import sys

matches = re.findall(r"\d+", sys.argv[1])
print(matches[-1] if matches else "")
PY
}

detect_num_gpus() {
  local raw=""
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    raw="$(extract_last_integer "${SLURM_GPUS_ON_NODE}")"
    if [[ -n "${raw}" && "${raw}" -gt 0 ]]; then
      echo "${raw}"
      return 0
    fi
  fi

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    if [[ "${#gpu_ids[@]}" -gt 0 ]]; then
      echo "${#gpu_ids[@]}"
      return 0
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    raw="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -n "${raw}" && "${raw}" -gt 0 ]]; then
      echo "${raw}"
      return 0
    fi
  fi

  echo "1"
}

resolve_project_root() {
  if [[ -n "${PROJECT_DIR:-}" && -d "${PROJECT_DIR}" ]]; then
    echo "${PROJECT_DIR}"
    return 0
  fi

  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    echo "${SLURM_SUBMIT_DIR}"
    return 0
  fi

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/../.." && pwd
}

PROJECT_ROOT="$(resolve_project_root)"

resolve_training_config_metadata() {
  local config_path="$1"
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python - "${PROJECT_ROOT}" "${config_path}" <<'PY'
from pathlib import Path
import sys

from verl_inf_evolve.sol_eval.experiment_discovery import load_training_experiment

project_root = Path(sys.argv[1])
config_path = sys.argv[2]
match = load_training_experiment(
    config_path,
    config_root=project_root / "verl_inf_evolve" / "config",
)
print(match.path)
print(match.run_name)
print(match.model_path)
print(match.remote_sync_path)
PY
}

RUN_NAME="${RUN_NAME:-}"
MODEL_PATH="${MODEL_PATH:-}"
REMOTE_SYNC_PATH="${REMOTE_SYNC_PATH:-}"
CHECKPOINTS="${CHECKPOINTS:-}"
TRAINING_CONFIG_PATH="${TRAINING_CONFIG_PATH:-}"
ANS_LOOP="${ANS_LOOP:-}"
BENCHMARKS="${BENCHMARKS:-}"
REMOTE_EVAL_BASE="${REMOTE_EVAL_BASE:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-LLM}"
HYDRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      require_value "$1" "${2:-}"
      RUN_NAME="$2"
      shift 2
      ;;
    --model|--model-path)
      require_value "$1" "${2:-}"
      MODEL_PATH="$2"
      shift 2
      ;;
    --remote-sync-path)
      require_value "$1" "${2:-}"
      REMOTE_SYNC_PATH="$2"
      shift 2
      ;;
    --checkpoints)
      require_value "$1" "${2:-}"
      CHECKPOINTS="$2"
      shift 2
      ;;
    --config-path)
      require_value "$1" "${2:-}"
      TRAINING_CONFIG_PATH="$2"
      shift 2
      ;;
    --ans-loop)
      require_value "$1" "${2:-}"
      ANS_LOOP="$2"
      shift 2
      ;;
    --benchmarks)
      require_value "$1" "${2:-}"
      BENCHMARKS="$2"
      shift 2
      ;;
    --remote-eval-base)
      require_value "$1" "${2:-}"
      REMOTE_EVAL_BASE="$2"
      shift 2
      ;;
    --hydra-override)
      require_value "$1" "${2:-}"
      HYDRA_OVERRIDES+=("$2")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        HYDRA_OVERRIDES+=("$1")
        shift
      done
      break
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

if [[ -n "${ANS_LOOP}" && -n "${CHECKPOINTS}" ]]; then
  die "--ans-loop cannot be combined with --checkpoints"
fi

if [[ -n "${ANS_LOOP}" && ! "${ANS_LOOP}" =~ ^[0-9]+$ ]]; then
  die "--ans-loop must be a non-negative integer"
fi

mkdir -p "${PROJECT_ROOT}/logs"

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_NAME:-sol-eval} (${SLURM_JOB_ID:-no-slurm-id})"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Working directory: ${PROJECT_ROOT}"
echo "=========================================="

if [[ "${SOL_EVAL_SKIP_ENV_SETUP:-0}" != "1" ]]; then
  if command -v module >/dev/null 2>&1; then
    module load miniconda CUDA/12.8 GCC/13.3
  fi

  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
  fi
fi

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Some cluster images export ROCm visibility vars even on NVIDIA nodes. Ray
# then adds CUDA_VISIBLE_DEVICES for workers and verl rejects the combination.
if command -v nvidia-smi >/dev/null 2>&1; then
  if [[ -n "${ROCR_VISIBLE_DEVICES:-}" || -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    echo "Clearing ROCm visibility env vars for NVIDIA job"
  fi
  unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
  unset HIP_VISIBLE_DEVICES 2>/dev/null || true
fi

# Ray creates deep session socket paths under its temp dir. Keep the base path
# short or AF_UNIX socket creation can fail on shared filesystems.
if [[ -z "${RAY_TMPDIR:-}" ]]; then
  if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
    export RAY_TMPDIR="${SLURM_TMPDIR%/}/ray"
  else
    export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID:-local}"
  fi
fi

mkdir -p "${RAY_TMPDIR}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE=offline
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  unset HF_TOKEN
fi

if [[ -n "${TRAINING_CONFIG_PATH}" ]]; then
  mapfile -t training_metadata < <(resolve_training_config_metadata "${TRAINING_CONFIG_PATH}")
  [[ "${#training_metadata[@]}" -eq 4 ]] || die "Failed to resolve metadata from --config-path ${TRAINING_CONFIG_PATH}"
  TRAINING_CONFIG_PATH="${training_metadata[0]}"
  [[ -n "${RUN_NAME}" ]] || RUN_NAME="${training_metadata[1]}"
  [[ -n "${MODEL_PATH}" ]] || MODEL_PATH="${training_metadata[2]}"
  [[ -n "${REMOTE_SYNC_PATH}" ]] || REMOTE_SYNC_PATH="${training_metadata[3]}"
fi

if [[ -n "${ANS_LOOP}" ]]; then
  CHECKPOINTS="${ANS_LOOP}"
fi

if [[ -z "${RUN_NAME}" && -n "${REMOTE_SYNC_PATH}" ]]; then
  RUN_NAME="$(python - "${REMOTE_SYNC_PATH}" <<'PY'
import sys

parts = [part for part in sys.argv[1].rstrip("/").split("/") if part]
print(parts[-1] if parts else "")
PY
)"
fi

[[ -n "${RUN_NAME}" ]] || die "--run-name is required (or provide --config-path)"
if [[ -z "${REMOTE_SYNC_PATH}" && -z "${MODEL_PATH}" ]]; then
  die "--model is required when --remote-sync-path is omitted (or provide --config-path)"
fi

NUM_GPUS="$(detect_num_gpus)"
echo "Detected ${NUM_GPUS} GPU(s) for this job"
echo "RUN_NAME=${RUN_NAME}"
echo "MODEL_PATH=${MODEL_PATH:-<auto>}"
echo "REMOTE_SYNC_PATH=${REMOTE_SYNC_PATH:-<auto-discover>}"
echo "TRAINING_CONFIG_PATH=${TRAINING_CONFIG_PATH:-<none>}"
echo "ANS_LOOP=${ANS_LOOP:-<unset>}"
echo "CHECKPOINTS=${CHECKPOINTS:-<default/auto>}"
echo "BENCHMARKS=${BENCHMARKS:-<config default>}"
echo "RAY_TMPDIR=${RAY_TMPDIR}"

if [[ ! -d "${PROJECT_ROOT}/.cache/data/preprocessed/benchmarks" ]]; then
  echo "WARNING: ${PROJECT_ROOT}/.cache/data/preprocessed/benchmarks not found"
fi

CMD=(python -m verl_inf_evolve.sol_eval.sol_eval)

CMD+=("eval.run_name=${RUN_NAME}")
CMD+=("trainer.n_gpus_per_node=${NUM_GPUS}")
CMD+=("hydra.run.dir=.")
CMD+=("hydra.output_subdir=null")

if [[ -n "${MODEL_PATH}" ]]; then
  CMD+=("eval.model_path=${MODEL_PATH}")
fi

if [[ -n "${REMOTE_SYNC_PATH}" ]]; then
  CMD+=("eval.remote_sync_path=${REMOTE_SYNC_PATH}")
fi

if [[ -n "${CHECKPOINTS}" ]]; then
  CMD+=("eval.checkpoints=${CHECKPOINTS}")
fi

if [[ -n "${BENCHMARKS}" ]]; then
  CMD+=("eval.benchmarks=${BENCHMARKS}")
fi

if [[ -n "${REMOTE_EVAL_BASE}" ]]; then
  CMD+=("eval.remote_eval_base=${REMOTE_EVAL_BASE}")
fi

if [[ "${#HYDRA_OVERRIDES[@]}" -gt 0 ]]; then
  CMD+=("${HYDRA_OVERRIDES[@]}")
fi

echo "Launching:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
