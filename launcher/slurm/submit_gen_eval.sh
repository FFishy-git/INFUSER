#!/usr/bin/env bash
# Generic SLURM script for generator evaluation.

#SBATCH --job-name=gen-eval
#SBATCH --output=.logs/gen_eval/%x-%j.out
#SBATCH --error=.logs/gen_eval/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_TYPE="gen_eval"
JOB_TYPES_DIR="${REPO_ROOT}/launcher/k8s/job_types"
JOB_TYPE_CONF="${JOB_TYPES_DIR}/${JOB_TYPE}.conf"
source "${JOB_TYPE_CONF}"

HYDRA_OVERRIDES=("$@")

resolve_num_gpus() {
  local detected=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    detected="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -n "${detected}" && "${detected}" -gt 0 ]]; then
      echo "${detected}"
      return
    fi
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
    echo "${#_gpus[@]}"
    return
  fi
  echo "1"
}

check_required_preprocessed_data() {
  local workspace="$1"
  shift

  local missing=0
  local rel
  for rel in "$@"; do
    if [[ -z "$rel" ]]; then
      continue
    fi
    if [[ ! -e "${workspace}/${rel}" ]]; then
      echo "ERROR: required cache path missing: ${workspace}/${rel}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" -eq 1 ]]; then
    echo "Tip: run the local data preparation path first and retry." >&2
    exit 1
  fi
}

if [[ -v REQUIRED_DATA_PATHS ]]; then
  check_required_preprocessed_data "${REPO_ROOT}" "${REQUIRED_DATA_PATHS[@]}"
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
  PROJECT_DIR="${REPO_ROOT}"
fi

echo "========================================"
echo "SLURM gen_eval"
echo "Job: ${SLURM_JOB_NAME:-gen-eval}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Module: ${PYTHON_MODULE}"
echo "Overrides: ${HYDRA_OVERRIDES[*]-}"
echo "========================================"

cd "${PROJECT_DIR}"

if [[ -n "${HF_TOKEN_POOL_JSON:-}" && -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$(python3 -c "
import json, sys
pool = json.loads(sys.argv[1])
namespace = sys.argv[2]
if namespace:
    for t in pool:
        if isinstance(t, dict) and t.get('namespace') == namespace:
            print(t['token'])
            break
    else:
        sys.exit(0)
else:
    t = pool[0]
    print(t['token'] if isinstance(t, dict) else t)
" "${HF_TOKEN_POOL_JSON}" "${HF_TOKEN_POOL_NAMESPACE:-}" 2>/dev/null || true)"
  export HF_TOKEN
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE=offline
else
  echo "WANDB enabled"
fi

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${SLURM_JOB_ID:-local}}"
mkdir -p "${RAY_TMPDIR}"

CONFIG_NAME_ARG=""
FILTERED_OVERRIDES=()
for arg in "${HYDRA_OVERRIDES[@]}"; do
  if [[ "${arg}" == --config-name=* ]]; then
    CONFIG_NAME_ARG="${arg}"
  else
    FILTERED_OVERRIDES+=("${arg}")
  fi
done

NUM_GPUS="$(resolve_num_gpus)"

CMD=(python -m "${PYTHON_MODULE}")
[[ -n "${CONFIG_NAME_ARG}" ]] && CMD+=("${CONFIG_NAME_ARG}")
CMD+=("trainer.n_gpus_per_node=${NUM_GPUS}" "hydra.run.dir=." "hydra.output_subdir=null")
CMD+=("${FILTERED_OVERRIDES[@]}")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
