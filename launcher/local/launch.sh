#!/usr/bin/env bash
# =============================================================================
# Local Launcher (training / sol_eval / gen_eval)
# =============================================================================
# Runs the shared launcher setup in a local shell instead of scheduling through a
# remote backend. This script intentionally reuses the k8s launcher runtime
# contract:
#   - reads launcher/k8s/job_types/*.conf
#   - builds HYDRA overrides with the same --job-type behavior
#   - runs launcher/k8s/run_job.sh for consistent process/env handling
#
# This keeps local/remote jobs behavior aligned while avoiding R2 dependency.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_TYPES_DIR="${REPO_ROOT}/launcher/k8s/job_types"
JOB_TYPE="training"
CONFIG_GROUP="experiment_qwen3_8b_base"
EXTRA_OVERRIDES=""
OFFLINE_DATA=true

EXPERIMENT=""

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
    echo "Please ensure the local .cache/data/preprocessed files are present." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-type)         JOB_TYPE="$2"; shift 2 ;;
    --config-group)     CONFIG_GROUP="$2"; shift 2 ;;
    --extra-overrides)  EXTRA_OVERRIDES="$2"; shift 2 ;;
    --offline-data)     OFFLINE_DATA=true; shift ;;
    -h|--help)
      sed -n '1,120p' "${REPO_ROOT}/launcher/k8s/launch.sh"
      exit 0 ;;
    -* )
      echo "Unknown option: $1" >&2
      exit 1 ;;
    *)
      if [[ -z "${EXPERIMENT}" ]]; then
        EXPERIMENT="$1"
      else
        echo "Unexpected argument: $1" >&2
        exit 1
      fi
      shift ;;
  esac
done

if [[ -z "${EXPERIMENT}" ]]; then
  echo "Usage: $0 <experiment_name> [options]" >&2
  echo "Run '$0 --help' for details." >&2
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

WANDB_API_KEY="${WANDB_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
HF_TOKEN_POOL_JSON="${HF_TOKEN_POOL_JSON:-}"
HF_TOKEN_POOL_NAMESPACE="${HF_TOKEN_POOL_NAMESPACE:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

if [[ -n "${HF_TOKEN_POOL_JSON}" && -z "${HF_TOKEN}" ]]; then
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
" "${HF_TOKEN_POOL_JSON}" "${HF_TOKEN_POOL_NAMESPACE}" 2>/dev/null || true)"
fi

if [[ -n "${REQUIRED_DATA_PATHS+x}" ]]; then
  check_required_preprocessed_data "${REPO_ROOT}" "${REQUIRED_DATA_PATHS[@]}"
fi

if [[ "${JOB_TYPE}" == "training" ]]; then
  VERL_MATH_VERIFY_TIMEOUT_S_OVERRIDE="${VERL_MATH_VERIFY_TIMEOUT_S_OVERRIDE:-5}"
else
  VERL_MATH_VERIFY_TIMEOUT_S_OVERRIDE="${VERL_MATH_VERIFY_TIMEOUT_S_OVERRIDE:-}"
fi

DATE_TAG="$(date +%Y%m%d)"
RAND_HEX="$(openssl rand -hex 4)"
if [[ -n "${JOB_NAME_PREFIX:-}" ]]; then
  JOB_NAME="${JOB_NAME_PREFIX}-${EXPERIMENT}-${DATE_TAG}-${RAND_HEX}"
else
  JOB_NAME="${EXPERIMENT}-${DATE_TAG}-${RAND_HEX}"
fi

echo "========================================"
echo "Local Launcher"
echo "========================================"
echo "Job name:  ${JOB_NAME}"
echo "Job type:  ${JOB_TYPE}"
echo "Experiment ${EXPERIMENT}"
echo "Module:    ${PYTHON_MODULE}"
echo "Overrides: ${HYDRA_OVERRIDES}"
echo "Offline data: $([[ "${OFFLINE_DATA}" == "true" ]] && echo yes || echo no)"
echo "========================================"

export WORKSPACE="${REPO_ROOT}"
export JOB_NAME="${JOB_NAME}"
export JOB_TYPE="${JOB_TYPE}"
export EXPERIMENT="${EXPERIMENT}"
export PYTHON_MODULE="${PYTHON_MODULE}"
export LOG_PREFIX="${LOG_PREFIX}"
export HYDRA_OVERRIDES="${HYDRA_OVERRIDES}"
export LAUNCH_MODE="${LAUNCH_MODE}"
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_ENTITY="${WANDB_ENTITY}"
export HF_TOKEN="${HF_TOKEN}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export HF_TOKEN_POOL_JSON="${HF_TOKEN_POOL_JSON}"
export VERL_MATH_VERIFY_TIMEOUT_S="${VERL_MATH_VERIFY_TIMEOUT_S_OVERRIDE}"
export SKIP_RUNTIME_INSTALLS=false
export SKIP_LEGACY_RCLONE_CONFIG=false

bash "${REPO_ROOT}/launcher/k8s/run_job.sh"
