#!/bin/bash
# =============================================================================
# SLURM Configuration Helper
# =============================================================================
# Source this file before submitting jobs to set cluster-specific defaults.
#
# Usage:
#   source launcher/slurm/config.sh
#   sbatch launcher/slurm/submit.sh
#
# Or create your own config file:
#   cp launcher/slurm/config.sh ~/.inf_evolve_slurm_config.sh
#   # Edit the file with your settings
#   source ~/.inf_evolve_slurm_config.sh
# =============================================================================

# =============================================================================
# Cluster Presets - Uncomment ONE section that matches your cluster
# =============================================================================

# --- Example shared A100 cluster ---
# export SLURM_PARTITION="gpu-shared"
# export SLURM_ACCOUNT="your_allocation"
# export SLURM_GPUS="4"  # gpu-shared max is 4 A100s
# export SLURM_TIME="48:00:00"
# module_cmds() {
#     module purge
#     module load gpu/0.15.4
#     module load cuda/11.8.0
#     module load anaconda3/2021.05
# }

# --- Example H100 cluster ---
# export SLURM_PARTITION="gpu"
# export SLURM_ACCOUNT="your_allocation"
# export SLURM_GPUS="8"
# export SLURM_TIME="48:00:00"
# module_cmds() {
#     module purge
#     module load cuda/12.2
#     module load anaconda3
# }

# --- Generic University Cluster ---
export SLURM_PARTITION="gpu"
export SLURM_ACCOUNT=""
export SLURM_GPUS="8"
export SLURM_TIME="48:00:00"
export SLURM_MEM="64G"
export SLURM_CPUS="8"

module_cmds() {
    # Uncomment and modify as needed:
    # module purge
    # module load cuda/12.1
    # module load anaconda3
    echo "Using default environment (no module commands)"
}

# =============================================================================
# Conda Environment
# =============================================================================
export CONDA_ENV="verl"  # Name of conda environment with dependencies

# =============================================================================
# Project Paths
# =============================================================================
# Adjust if your project is not in the default location
export INF_EVOLVE_PROJECT_DIR="${HOME}/self-evolution-explore"

# Data cache directory (can be on fast local storage)
export INF_EVOLVE_DATA_DIR="${INF_EVOLVE_PROJECT_DIR}/.cache/data/preprocessed"

# =============================================================================
# Training Defaults
# =============================================================================
# Default Hydra overrides (can be extended at submission time)
export HYDRA_OVERRIDES_DEFAULT=""

# =============================================================================
# API Keys (set these or source from a secure file)
# =============================================================================
# export OPENAI_API_KEY=""
# export WANDB_API_KEY=""
# export HF_TOKEN=""

# =============================================================================
# Helper Function: Submit job with overrides
# =============================================================================
submit_inf_evolve() {
    local extra_overrides="$1"
    local hydra_opts="${HYDRA_OVERRIDES_DEFAULT} ${extra_overrides}"

    echo "Submitting inf_evolve job..."
    echo "  Partition: ${SLURM_PARTITION}"
    echo "  GPUs: ${SLURM_GPUS}"
    echo "  Hydra overrides: ${hydra_opts}"

    HYDRA_OVERRIDES="${hydra_opts}" sbatch \
        --partition="${SLURM_PARTITION}" \
        ${SLURM_ACCOUNT:+--account="${SLURM_ACCOUNT}"} \
        --gres="gpu:${SLURM_GPUS}" \
        --time="${SLURM_TIME}" \
        --mem="${SLURM_MEM}" \
        --cpus-per-task="${SLURM_CPUS}" \
        "${INF_EVOLVE_PROJECT_DIR}/launcher/slurm/submit.sh"
}

echo "SLURM config loaded. Use 'submit_inf_evolve' to submit jobs."
echo "Example: submit_inf_evolve 'schema.training.max_ans_loop=2'"
