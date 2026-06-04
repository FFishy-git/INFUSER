#!/usr/bin/env bash
# =============================================================================
# Shared SLURM launcher
# =============================================================================
# Unified entrypoint for:
#   - training
#   - sol_eval
#   - gen_eval
#
# Dispatches to dedicated submission scripts and builds overrides from
# launcher/k8s/job_types/*.conf so arguments match the K8s launcher.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_TYPES_DIR="${REPO_ROOT}/launcher/k8s/job_types"

JOB_TYPE="training"
CONFIG_GROUP="experiment_qwen3_8b_base"
EXTRA_OVERRIDES=""
OFFLINE_DATA=true

EXPERIMENT=""

SBATCH_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./launcher/slurm/launch.sh <experiment_name> [options]

Options:
  --job-type TYPE        training (default), sol_eval, gen_eval
  --config-group GROUP   Hydra config group (training only)
  --extra-overrides STR  Extra Hydra overrides to append
  --sbatch-arg ARG       Extra arg passed verbatim to `sbatch` (repeatable)
  --offline-data         Enforce local preprocessed-data-only mode
  --help

Examples:
  ./launcher/slurm/launch.sh FW-... --job-type training
  ./launcher/slurm/launch.sh eval-run --job-type sol_eval
  ./launcher/slurm/launch.sh replay-FW-... --job-type gen_eval --extra-overrides "gen_eval.mode=replay"
EOF
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
    echo "Please prepare .cache/data/preprocessed first or run a different mode with synced data." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-type)         JOB_TYPE="$2"; shift 2 ;;
    --config-group)     CONFIG_GROUP="$2"; shift 2 ;;
    --extra-overrides)  EXTRA_OVERRIDES="$2"; shift 2 ;;
    --sbatch-arg)       SBATCH_EXTRA_ARGS+=("$2"); shift 2 ;;
    --offline-data)     OFFLINE_DATA=true; shift ;;
    --help|-h)
      usage
      exit 0
      ;;
    -* )
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -z "${EXPERIMENT}" ]]; then
        EXPERIMENT="$1"
      else
        echo "Unexpected argument: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "${EXPERIMENT}" ]]; then
  usage
  exit 1
fi

JOB_TYPE_CONF="${JOB_TYPES_DIR}/${JOB_TYPE}.conf"
if [[ ! -f "${JOB_TYPE_CONF}" ]]; then
  _AVAILABLE="$(ls "${JOB_TYPES_DIR}"/*.conf 2>/dev/null | xargs -n1 basename -s .conf | paste -sd, )"
  echo "ERROR: Unknown job type '${JOB_TYPE}'. Available: ${_AVAILABLE}" >&2
  exit 1
fi

source "${JOB_TYPE_CONF}"

if [[ "${LAUNCH_MODE}" == "hydra" ]]; then
  if [[ -n "${HYDRA_CONFIG_KEY}" && -n "${EXTRA_OVERRIDES}" ]] \
     && echo "${EXTRA_OVERRIDES}" | grep -q "${HYDRA_CONFIG_KEY}="; then
    HYDRA_OVERRIDES="${EXTRA_OVERRIDES}"
  elif [[ -n "${HYDRA_CONFIG_KEY}" ]]; then
    HYDRA_OVERRIDES="${HYDRA_CONFIG_KEY}=${EXPERIMENT}"
    [[ -n "${EXTRA_OVERRIDES}" ]] && HYDRA_OVERRIDES="${HYDRA_OVERRIDES} ${EXTRA_OVERRIDES}"
  else
    HYDRA_OVERRIDES="${CONFIG_GROUP}=${EXPERIMENT}"
    [[ -n "${EXTRA_OVERRIDES}" ]] && HYDRA_OVERRIDES="${HYDRA_OVERRIDES} ${EXTRA_OVERRIDES}"
  fi
else
  HYDRA_OVERRIDES="${EXTRA_OVERRIDES}"
fi

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ -n "${REQUIRED_DATA_PATHS+x}" ]]; then
  check_required_preprocessed_data "${REPO_ROOT}" "${REQUIRED_DATA_PATHS[@]}"
fi

echo "========================================"
echo "SLURM Launcher"
echo "========================================"
echo "Job type:   ${JOB_TYPE}"
echo "Experiment: ${EXPERIMENT}"
echo "Overrides:  ${HYDRA_OVERRIDES}"
echo "Offline:    $([[ "${OFFLINE_DATA}" == "true" ]] && echo yes || echo no)"
echo "========================================"

read -r -a HYDRA_ARGS <<< "${HYDRA_OVERRIDES}"

case "${JOB_TYPE}" in
  training)
    echo "Submitting: launcher/slurm/submit.sh"
    exec sbatch "${SBATCH_EXTRA_ARGS[@]}" \
      --export=ALL,HYDRA_OVERRIDES="${HYDRA_OVERRIDES}" \
      "${REPO_ROOT}/launcher/slurm/submit.sh"
    ;;
  sol_eval)
    echo "Submitting: launcher/slurm/submit_sol_eval.sh"
    exec sbatch "${SBATCH_EXTRA_ARGS[@]}" \
      --export=ALL \
      "${REPO_ROOT}/launcher/slurm/submit_sol_eval.sh" "${HYDRA_ARGS[@]}"
    ;;
  gen_eval)
    echo "Submitting: launcher/slurm/submit_gen_eval.sh"
    exec sbatch "${SBATCH_EXTRA_ARGS[@]}" \
      --export=ALL \
      "${REPO_ROOT}/launcher/slurm/submit_gen_eval.sh" "${HYDRA_ARGS[@]}"
    ;;
  *)
    echo "Unsupported SLURM job-type '${JOB_TYPE}' for this launcher" >&2
    exit 1
    ;;
esac
