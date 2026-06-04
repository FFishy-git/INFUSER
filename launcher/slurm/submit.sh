#!/bin/bash
# =============================================================================
# verl_inf_evolve Self-Evolution Training (V3) - SLURM Submission Script
# =============================================================================
# Converted from SkyPilot launcher.yaml for university SLURM clusters.
#
# This script runs all training stages on a single GPU node:
#   1. Runs the main training loop (verl_inf_evolve/main.py)
#   2. Executes vLLM inference locally with data parallelism
#   3. Runs gradient computation on local GPUs
#
# Usage:
#   # Basic submission
#   sbatch launcher/slurm/submit.sh
#
#   # With Hydra overrides
#   HYDRA_OVERRIDES="training.doc_batch_size=16" sbatch launcher/slurm/submit.sh
#
#   # Full example with all options
#   HYDRA_OVERRIDES="training.max_ans_loop=2 training.max_gen_loop=2" \
#   sbatch --partition=gpu-a100 launcher/slurm/submit.sh
#
#   # With a different config file under verl_inf_evolve/config/:
#   HYDRA_OVERRIDES="--config-name=my_config training.max_ans_loop=5" \
#   sbatch launcher/slurm/submit.sh
#
# Monitor:
#   squeue -u $USER                    # Check job status
#   scancel <job_id>                   # Cancel job
#   tail -f slurm-<job_id>.out         # View output
#
# =============================================================================

#SBATCH --job-name=inf-evolve-v3
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# =============================================================================
# Resource Configuration - ADJUST THESE FOR YOUR CLUSTER
# =============================================================================
#SBATCH --partition=gpu              # Partition name (e.g., gpu, gpu-a100, gpu-h100)
#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks=1                   # Single task
#SBATCH --cpus-per-task=8            # CPUs per task
#SBATCH --mem=512G                    # Memory
#SBATCH --gres=gpu:h100:4                 # Number of GPUs (adjust: gpu:4 for 4 GPUs)
#SBATCH --time=48:00:00              # Max runtime (HH:MM:SS)

# Optional: Email notifications (uncomment and set your email)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@university.edu

# Optional: Account/QOS if required by your cluster
# #SBATCH --account=your_account
#SBATCH --qos=qos_zhuoran_yang
# #SBATCH --constraint="h100|h200"

# =============================================================================
# Environment Variables
# =============================================================================
# Hydra overrides (can be set via environment before sbatch)
HYDRA_OVERRIDES="${HYDRA_OVERRIDES:-}"
JOB_TYPES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/launcher/k8s/job_types"
JOB_TYPE="training"
JOB_TYPE_CONF="${JOB_TYPES_DIR}/${JOB_TYPE}.conf"
source "${JOB_TYPE_CONF}"

REQUIRED_DATA_PATHS_STR=()
if [[ -v REQUIRED_DATA_PATHS ]]; then
  REQUIRED_DATA_PATHS_STR=("${REQUIRED_DATA_PATHS[@]}")
fi

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
    echo "Please ensure local preprocessed files are available before running." >&2
    echo "Expected files are listed in job_types/${JOB_TYPE}.conf (REQUIRED_DATA_PATHS)." >&2
    exit 1
  fi
}

# API keys (set these before submission or in your ~/.bashrc)
# export OPENAI_API_KEY="your_key"    # Required for GPT-based scoring
# export WANDB_API_KEY="your_key"     # Optional for experiment tracking
# export HF_TOKEN="your_token"        # Optional for private HuggingFace models
# export HF_TOKEN_POOL_JSON='[...]'   # Optional for HF namespace-pool artifact auth

# =============================================================================
# Cluster-Specific Setup - MODIFY THIS SECTION FOR YOUR CLUSTER
# =============================================================================
echo "=============================================="
echo "verl_inf_evolve Self-Evolution Training V3 - SLURM Job Starting"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-8}"
echo "=============================================="

# Source bashrc for environment variables (as per user instructions)
source ~/.bashrc

# Load required modules - ADJUST FOR YOUR CLUSTER
# Common module configurations (uncomment/modify as needed):

# --- Option A: CUDA + Anaconda modules ---
# module purge
# module load cuda/12.8
# module load anaconda3/2023.09

# --- Option B: Singularity/Apptainer with container ---
# module load singularity/3.10

# --- Option C: Custom conda environment ---
module load miniconda
conda activate LLM

# For this script, we assume a conda environment named 'verl' or 'inf_evolve'
# Uncomment and modify:
# conda activate verl

# =============================================================================
# Working Directory Setup
# =============================================================================
# Navigate to project root (adjust if your project is elsewhere)
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# =============================================================================
# GPU Verification
# =============================================================================
echo "=============================================="
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "nvidia-smi not available"
echo "=============================================="

# =============================================================================
# Environment Verification (No Installation - Pre-configured Cluster)
# =============================================================================
echo "=== Ensuring EvalPlus parser dependencies ==="
if ! python - <<'PY'
import importlib.util
import sys
required = ("termcolor", "tree_sitter", "tree_sitter_python")
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print(",".join(missing))
    sys.exit(1)
print("already_present")
PY
then
    python -m pip install termcolor "tree_sitter>=0.22.0" tree-sitter-python --no-cache-dir
fi

echo "=== Verifying environment ==="
python -c "import verl; print(f'verl: {verl.__file__}')"
python -c "import verl_inf_evolve; print(f'verl_inf_evolve: {verl_inf_evolve.__file__}')"
python -c "import termcolor, tree_sitter, tree_sitter_python; print('EvalPlus parser deps: OK')"

# Verify data exists
if [[ "${#REQUIRED_DATA_PATHS_STR[@]}" -gt 0 ]]; then
  check_required_preprocessed_data "${PROJECT_DIR}" "${REQUIRED_DATA_PATHS_STR[@]}"
fi
DATA_DIR="${PROJECT_DIR}/.cache/data/preprocessed"
ls -la "${DATA_DIR}/"

echo "=== Environment ready ==="

# =============================================================================
# Training Phase
# =============================================================================
echo "=============================================="
echo "verl_inf_evolve Self-Evolution Training V3"
echo "=============================================="

# Export environment for Python
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} GPUs"

# Run training using verl_inf_evolve with Hydra config
# Note: Extract --config-name from HYDRA_OVERRIDES so it appears before other overrides
CONFIG_NAME_ARG=""
FILTERED_OVERRIDES=""
for arg in ${HYDRA_OVERRIDES}; do
    if [[ "$arg" == --config-name=* ]]; then
        CONFIG_NAME_ARG="$arg"
    else
        FILTERED_OVERRIDES="${FILTERED_OVERRIDES} $arg"
    fi
done

# Disable logging buffer
export PYTHONUNBUFFERED=1
if [ -z "${HF_TOKEN_POOL_JSON:-}" ]; then
  unset HF_TOKEN_POOL_JSON
fi

# Persistent log file: tee stdout+stderr so logs survive crashes.
LOG_DIR="$(echo ${FILTERED_OVERRIDES} | sed -n 's/.*training\.default_local_dir=\([^ ]*\).*/\1/p' || true)"
if [ -z "${LOG_DIR}" ]; then
  LOG_DIR=".output"
fi
mkdir -p "${LOG_DIR}"
TRAIN_LOG="${LOG_DIR}/train_stdout.log"
echo "=== Training stdout/stderr log: ${TRAIN_LOG} ==="

# ---------------------------------------------------------------
# Unique Run ID — each run gets a unique ID so R2 log uploads
# don't overwrite previous runs.  On restart a new ID is created.
# ---------------------------------------------------------------
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_R2_NAME="train_stdout_${RUN_ID}.log"
echo "=== Run ID: ${RUN_ID} ==="

# ---------------------------------------------------------------
# Resolve remote backend and enable launcher-level log mirroring
# only for the current R2/S3 path shape. HF-backed runs still sync
# training artifacts via the Python backend, but shell-side rclone
# log uploads are intentionally disabled.
# ---------------------------------------------------------------
RESOLVED_REMOTE_SYNC_PATH=$(python3 launcher/resolve_sync_path.py ${CONFIG_NAME_ARG} ${FILTERED_OVERRIDES} 2>/dev/null || true)
REMOTE_BACKEND="none"
LOG_R2_DEST=""
case "${RESOLVED_REMOTE_SYNC_PATH}" in
  hf://*)
    REMOTE_BACKEND="hf"
    ;;
  s3://*)
    REMOTE_BACKEND="r2"
    LOG_R2_DEST="$(echo "${RESOLVED_REMOTE_SYNC_PATH}" | sed 's|^s3://|r2:|')/logs"
    ;;
  r2://*)
    REMOTE_BACKEND="r2"
    LOG_R2_DEST="$(echo "${RESOLVED_REMOTE_SYNC_PATH}" | sed 's|^r2://|r2:|')/logs"
    ;;
esac

echo "Resolved remote_sync_path: ${RESOLVED_REMOTE_SYNC_PATH:-<not set>}"
echo "Resolved remote backend: ${REMOTE_BACKEND}"
if [ "${REMOTE_BACKEND}" = "hf" ]; then
  echo "HF remote detected - disabling launcher-level rclone log sync"
fi

LOG_SYNC_PID=""
if [ -n "${LOG_R2_DEST}" ] && which rclone >/dev/null 2>&1; then
  echo "=== Log sync enabled: ${LOG_R2_DEST}/${LOG_R2_NAME} ==="

  _sync_log_loop() {
    while true; do
      sleep 15
      [ -f "${TRAIN_LOG}" ] && rclone copyto "${TRAIN_LOG}" \
        "${LOG_R2_DEST}/${LOG_R2_NAME}" --s3-no-check-bucket 2>/dev/null
    done
  }
  _sync_log_loop &
  LOG_SYNC_PID=$!

  _cleanup_on_signal() {
    echo "=== Signal received — flushing final log to R2 ===" >> "${TRAIN_LOG}" 2>/dev/null
    [ -f "${TRAIN_LOG}" ] && rclone copyto "${TRAIN_LOG}" \
      "${LOG_R2_DEST}/${LOG_R2_NAME}" --s3-no-check-bucket 2>/dev/null
    [ -n "${LOG_SYNC_PID}" ] && kill "${LOG_SYNC_PID}" 2>/dev/null
    exit 143
  }
  trap _cleanup_on_signal SIGTERM SIGINT
else
  echo "=== Log sync disabled (no EXPERIMENT or rclone) ==="
fi

python -m "${PYTHON_MODULE}" \
    ${CONFIG_NAME_ARG} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    hydra.run.dir=. \
    hydra.output_subdir=null \
    ${FILTERED_OVERRIDES} \
    2>&1 | tee -a "${TRAIN_LOG}"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# Final log upload
if [ -n "${LOG_R2_DEST}" ] && [ -f "${TRAIN_LOG}" ] && which rclone >/dev/null 2>&1; then
  echo "=== Uploading final log to R2 ==="
  rclone copyto "${TRAIN_LOG}" "${LOG_R2_DEST}/${LOG_R2_NAME}" \
    --s3-no-check-bucket 2>/dev/null \
    && echo "Log uploaded to: ${LOG_R2_DEST}/${LOG_R2_NAME}" \
    || echo "WARNING: Final log upload failed"
fi

# Stop background log sync
[ -n "${LOG_SYNC_PID}" ] && kill "${LOG_SYNC_PID}" 2>/dev/null

echo "=============================================="
echo "Training complete!"
echo "Exit code: ${TRAIN_EXIT_CODE}"
echo "=============================================="

exit ${TRAIN_EXIT_CODE}
