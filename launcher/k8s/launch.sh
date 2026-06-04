#!/bin/bash
# =============================================================================
# Reusable K8s Direct Launcher
# =============================================================================
# Launches jobs directly on Kubernetes, bypassing SkyPilot's managed job
# controller. Supports multiple job types with shared infrastructure.
#
# USAGE:
#   ./launch.sh <experiment_name> [options]
#
# JOB TYPES (defined in launcher/k8s/job_types/*.conf):
#   training   (default) — joint generator-solver self-evolution training
#   gen_eval   — generator evaluation with fixed solver (replay or regenerate)
#   sol_eval   — solver benchmark evaluation (multi-checkpoint, multi-benchmark)
#   opencompass — OpenCompass benchmark evaluation
#   training_interactive — training shell with the same setup, but manual submit
#
# EXAMPLES:
#   # Training (default)
#   ./launch.sh FW-Alr_2e-6-Glr_2e-6-DrGRPO-TIS_token-dev_800-precond_cos \
#     --config-group experiment_qwen3_4b_base
#
#   # Gen eval (replay mode)
#   ./launch.sh replay-FW-Alr_2e-6-Glr_2e-6-DrGRPO-TIS_token-dev_800-precond_cos-qwen3_4b_base \
#     --job-type gen_eval
#
#   # Sol eval
#   ./launch.sh sample --job-type sol_eval
#
# OPTIONS:
#   --job-type TYPE        Job type (default: training). See job_types/*.conf
#   --config-group GROUP   Hydra config group (training only)
#   --extra-overrides STR  Additional Hydra overrides
#   --gpu GPU_COUNT        Number of GPUs (default: 8)
#   --memory MEM_GI        Memory in Gi (default: 512)
#   --cpu CPU_COUNT        CPU count (default: 32)
#   --queue QUEUE          Kueue queue name (default: skypilot-jobs-8-gpu)
#   --image IMAGE          Docker image (default: verlai/verl:vllm012.latest)
#   --pvc PVC_NAME         PVC to mount (default: csi-pvc-mounted)
#   --skip-runtime-installs
#                          Skip pod-startup apt/rclone/pip installs; use when
#                          the image already contains runtime deps
#   --dry-run              Print manifest without applying
#   --no-stream            Don't stream logs after launch
#   --skip-sync            Deprecated no-op (legacy PVC sync is inactive)
#   --offline-data         Skip preprocessed-data download from R2 and use local
#                          .cache/data/preprocessed only.
#   --with-r2-sync         Force copying preprocessed files from R2 even when
#                          local cache is already present.
#
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_TYPES_DIR="${REPO_ROOT}/launcher/k8s/job_types"
JOB_TYPE="training"
CONFIG_GROUP="experiment_qwen3_8b_base"
EXTRA_OVERRIDES=""
GPU_COUNT=8
MEMORY_GI=512
CPU_COUNT=32
QUEUE_NAME="skypilot-jobs-8-gpu"
IMAGE="verlai/verl:vllm012.latest"
PVC_NAME="csi-pvc-mounted"
SKIP_RUNTIME_INSTALLS=false
DRY_RUN=false
STREAM_LOGS=true
SKIP_SYNC=false
OFFLINE_DATA="${LAUNCHER_OFFLINE_DATA:-auto}"

# Track which resource flags the user explicitly set (for per-type defaults).
_USER_SET_GPU=false
_USER_SET_MEM=false
_USER_SET_CPU=false
_USER_SET_QUEUE=false

# ── Parse arguments ──────────────────────────────────────────────────────────
EXPERIMENT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-type)         JOB_TYPE="$2"; shift 2 ;;
    --config-group)     CONFIG_GROUP="$2"; shift 2 ;;
    --extra-overrides)  EXTRA_OVERRIDES="$2"; shift 2 ;;
    --gpu)              GPU_COUNT="$2"; _USER_SET_GPU=true; shift 2 ;;
    --memory)           MEMORY_GI="$2"; _USER_SET_MEM=true; shift 2 ;;
    --cpu)              CPU_COUNT="$2"; _USER_SET_CPU=true; shift 2 ;;
    --queue)            QUEUE_NAME="$2"; _USER_SET_QUEUE=true; shift 2 ;;
    --image)            IMAGE="$2"; shift 2 ;;
    --pvc)              PVC_NAME="$2"; shift 2 ;;
    --skip-runtime-installs) SKIP_RUNTIME_INSTALLS=true; shift ;;
    --dry-run)          DRY_RUN=true; shift ;;
    --no-stream)        STREAM_LOGS=false; shift ;;
    --skip-sync)        SKIP_SYNC=true; shift ;;
    --offline-data)     OFFLINE_DATA=true; shift ;;
    --with-r2-sync)     OFFLINE_DATA=false; shift ;;
    -h|--help)
      sed -n '2,/^# =====/p' "$0" | head -n -1 | sed 's/^# \?//'
      exit 0 ;;
    -*)
      echo "Unknown option: $1" >&2; exit 1 ;;
    *)
      if [[ -z "${EXPERIMENT}" ]]; then
        EXPERIMENT="$1"
      else
        echo "Unexpected argument: $1" >&2; exit 1
      fi
      shift ;;
  esac
done

if [[ -z "${EXPERIMENT}" ]]; then
  echo "Usage: $0 <experiment_name> [options]"
  echo "Run '$0 --help' for details."
  exit 1
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
    echo "Hint: either prepare local preprocessed files first or rerun without --offline-data" >&2
    exit 1
  fi
}

required_paths_present() {
  local workspace="$1"
  shift

  for rel in "$@"; do
    [[ -z "${rel}" ]] && continue
    if [[ ! -e "${workspace}/${rel}" ]]; then
      return 1
    fi
  done
  return 0
}

yaml_quote() {
  local value="${1-}"
  value="${value//\'/\'\'}"
  printf "'%s'" "${value}"
}

pod_is_terminating() {
  local pod_name="$1"
  local deletion_timestamp
  deletion_timestamp="$(kubectl get pod "${pod_name}" -o jsonpath='{.metadata.deletionTimestamp}' 2>/dev/null || echo "")"
  [[ -n "${deletion_timestamp}" ]]
}

# ── Load job-type config ─────────────────────────────────────────────────────
JOB_TYPE_CONF="${JOB_TYPES_DIR}/${JOB_TYPE}.conf"
if [[ ! -f "${JOB_TYPE_CONF}" ]]; then
  _AVAILABLE="$(ls "${JOB_TYPES_DIR}"/*.conf 2>/dev/null | xargs -n1 basename -s .conf | paste -sd,)"
  echo "ERROR: Unknown job type '${JOB_TYPE}'. Available: ${_AVAILABLE}" >&2
  exit 1
fi
source "${JOB_TYPE_CONF}"

# Apply per-type resource defaults (only where user didn't explicitly override).
${_USER_SET_GPU}   || GPU_COUNT="${DEFAULT_GPU}"
${_USER_SET_MEM}   || MEMORY_GI="${DEFAULT_MEMORY}"
${_USER_SET_CPU}   || CPU_COUNT="${DEFAULT_CPU}"
${_USER_SET_QUEUE} || QUEUE_NAME="${DEFAULT_QUEUE}"
WATCHER_IDLE_THRESHOLD="${WATCHER_IDLE_THRESHOLD:-1800}"
WATCHER_STUCK_KILL_THRESHOLD="${WATCHER_STUCK_KILL_THRESHOLD:-14400}"

if [[ "${OFFLINE_DATA}" == "auto" ]]; then
  if [[ -n "${REQUIRED_DATA_PATHS+x}" ]] && required_paths_present "${REPO_ROOT}" "${REQUIRED_DATA_PATHS[@]}"; then
    OFFLINE_DATA=true
  else
    OFFLINE_DATA=false
  fi
fi

REQUIRED_DATA_PATHS_LITERAL=""
if [[ -n "${REQUIRED_DATA_PATHS+x}" ]]; then
  printf -v REQUIRED_DATA_PATHS_LITERAL '%q ' "${REQUIRED_DATA_PATHS[@]}"
fi

# ── Build Hydra overrides ───────────────────────────────────────────────────
if [[ "${LAUNCH_MODE}" != "script" ]]; then
  if [[ -n "${HYDRA_CONFIG_KEY}" && -n "${EXTRA_OVERRIDES}" ]] \
     && echo "${EXTRA_OVERRIDES}" | grep -q "${HYDRA_CONFIG_KEY}="; then
    # Extra overrides already specify the config key — skip default to avoid duplicate
    HYDRA_OVERRIDES="${EXTRA_OVERRIDES}"
  elif [[ -n "${HYDRA_CONFIG_KEY}" ]]; then
    HYDRA_OVERRIDES="${HYDRA_CONFIG_KEY}=${EXPERIMENT}"
    [[ -n "${EXTRA_OVERRIDES}" ]] && HYDRA_OVERRIDES="${HYDRA_OVERRIDES} ${EXTRA_OVERRIDES}"
  else
    # Training: uses CONFIG_GROUP variable for experiment selection
    HYDRA_OVERRIDES="${CONFIG_GROUP}=${EXPERIMENT}"
    [[ -n "${EXTRA_OVERRIDES}" ]] && HYDRA_OVERRIDES="${HYDRA_OVERRIDES} ${EXTRA_OVERRIDES}"
  fi
else
  # Script mode: EXTRA_OVERRIDES are raw CLI args
  HYDRA_OVERRIDES="${EXTRA_OVERRIDES}"
fi

# ── Source .env for API keys ─────────────────────────────────────────────────
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

WANDB_API_KEY="${WANDB_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
HF_TOKEN_POOL_JSON="${HF_TOKEN_POOL_JSON:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
R2_ENDPOINT_URL="${R2_ENDPOINT_URL:-${API_ENDPOINT:-}}"
R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:-${ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}}"
R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:-${SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}}"
R2_REGION="${R2_REGION:-${DEFAULT_REGION:-auto}}"
PREPROCESSED_DATA_RCLONE_URI="${PREPROCESSED_DATA_RCLONE_URI:-}"

# ── Config group shorthand ───────────────────────────────────────────────────
_config_group_short() {
  case "$1" in
    experiment)                             echo "qw8b" ;;
    experiment_qwen3_4b)                    echo "qw4b" ;;
    experiment_qwen3_4b_ins)                echo "qw4bi" ;;
    experiment_qwen3_4b_base)               echo "qw4bb" ;;
    experiment_qwen3_8b_base)               echo "qw8bb" ;;
    experiment_llama31_8b)                   echo "ll8b" ;;
    *)                                       echo "$1" ;;
  esac
}

# ── Job naming ───────────────────────────────────────────────────────────────
DATE_TAG="$(date +%Y%m%d)"
RAND_HEX="$(openssl rand -hex 4)"

if [[ -n "${JOB_NAME_PREFIX}" ]]; then
  JOB_NAME="${JOB_NAME_PREFIX}-${EXPERIMENT}-${DATE_TAG}-${RAND_HEX}"
  GROUP_TAG="${JOB_NAME_PREFIX}"
else
  # Training: derive tag from config group shorthand
  GROUP_TAG="$(_config_group_short "${CONFIG_GROUP}")"
  JOB_NAME="${EXPERIMENT}-${GROUP_TAG}-${DATE_TAG}-${RAND_HEX}"
fi

# K8s pod names: lowercase, max 63 chars, alphanumeric + hyphens.
_SUFFIX="-${DATE_TAG}-${RAND_HEX}"
_SUFFIX_LOWER="$(echo "${_SUFFIX}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
_PREFIX="$(echo "${JOB_NAME%${_SUFFIX}}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
_MAX_PREFIX_LEN=$(( 63 - ${#_SUFFIX_LOWER} ))
_PREFIX="${_PREFIX:0:${_MAX_PREFIX_LEN}}"
_PREFIX="${_PREFIX%-}"  # trim trailing hyphen
POD_NAME="${_PREFIX}${_SUFFIX_LOWER}"
SECRET_NAME="${POD_NAME}"

# ── Cap math-verifier subprocess timeout for TRAINING jobs only ──────────────
# Lower per-call wallclock (5 s vs default 15 s) cuts long-tail timeout cost
# in extract_answer_scores during ans_loop scoring.  sol_eval / gen_eval keep
# the historical 15 s default by passing an empty value (mcq_utils reads an
# empty string as "unset" and falls back to the built-in default).
if [[ "${JOB_TYPE}" == "training" ]]; then
  VERL_MATH_VERIFY_TIMEOUT_S_OVERRIDE="${VERL_MATH_VERIFY_TIMEOUT_S_OVERRIDE:-5}"
else
  VERL_MATH_VERIFY_TIMEOUT_S_OVERRIDE="${VERL_MATH_VERIFY_TIMEOUT_S_OVERRIDE:-}"
fi

# ── Print summary ────────────────────────────────────────────────────────────
echo "========================================"
echo "K8s Direct Launcher — ${JOB_TYPE}"
echo "========================================"
echo "Job name:     ${JOB_NAME}"
echo "Pod name:     ${POD_NAME}"
echo "Job type:     ${JOB_TYPE}"
echo "Experiment:   ${EXPERIMENT}"
echo "Module:       ${PYTHON_MODULE}"
echo "Overrides:    ${HYDRA_OVERRIDES}"
echo "Image:        ${IMAGE}"
echo "Resources:    ${GPU_COUNT}x H100, ${CPU_COUNT} CPUs, ${MEMORY_GI}Gi RAM"
echo "Queue:        ${QUEUE_NAME}"
echo "PVC:          ${PVC_NAME}"
echo "Secret:       ${SECRET_NAME}"
echo "Runtime deps: $([[ "${SKIP_RUNTIME_INSTALLS}" == "true" ]] && echo "from image" || echo "install in pod")"
echo "R2 config:    env/image config"
echo "WANDB:        $([[ -n "${WANDB_API_KEY}" ]] && echo "(set)" || echo "(not set)")"
echo "HF_TOKEN:     $([[ -n "${HF_TOKEN}" ]] && echo "(set)" || echo "(not set)")"
echo "HF_POOL:      $([[ -n "${HF_TOKEN_POOL_JSON}" ]] && echo "(set)" || echo "(not set)")"
echo "Data source:  ${PREPROCESSED_DATA_RCLONE_URI:-local cache only}"
echo "Offline data:  $([[ "${OFFLINE_DATA}" == "true" ]] && echo "enabled" || echo "disabled")"
echo "========================================"
echo ""

# ── Resolve PVC repo source ──────────────────────────────────────────────────
HOST_REPO_ROOT="${REPO_ROOT}"
WORKSPACE_MOUNT_PVC_REPO="${K8S_PVC_REPO:-${REPO_ROOT}}"
if [[ -z "${K8S_PVC_REPO:-}" && "${REPO_ROOT}" == /nemo-workspace/* ]]; then
  WORKSPACE_MOUNT_PVC_REPO="/workspace/${REPO_ROOT#/nemo-workspace/}"
fi
LEGACY_PVC_REPO="/workspace/project/self-evolution-explore"
echo "=== Repo source ==="
echo "  Primary: ${HOST_REPO_ROOT}"
echo "  Workspace-mount fallback: ${WORKSPACE_MOUNT_PVC_REPO}"
echo "  Legacy fallback: ${LEGACY_PVC_REPO}"
if [[ "${SKIP_SYNC}" == "true" ]]; then
  echo "  Note: --skip-sync is deprecated; legacy PVC sync is inactive"
fi
echo ""

# ── Check for existing pod ───────────────────────────────────────────────────
if kubectl get pod "${POD_NAME}" &>/dev/null; then
  echo "ERROR: Pod '${POD_NAME}' already exists."
  if pod_is_terminating "${POD_NAME}"; then
    echo "  The pod is still terminating; wait for deletion to finish before relaunching."
  fi
  echo "  Current status:"
  kubectl get pod "${POD_NAME}" 2>/dev/null || true
  echo "  kubectl delete pod ${POD_NAME}"
  exit 1
fi

# ── Build GPU-conditional manifest fragments ────────────────────────────────
_NODE_SELECTOR=""
_GPU_RESOURCE_REQ=""
_GPU_LIMITS_BLOCK=""
if [[ "${GPU_COUNT}" -gt 0 ]]; then
  _NODE_SELECTOR="  nodeSelector:
    skypilot.co/accelerator: h100"
  _GPU_RESOURCE_REQ="
          nvidia.com/gpu: \"${GPU_COUNT}\""
  _GPU_LIMITS_BLOCK="        limits:
          nvidia.com/gpu: \"${GPU_COUNT}\""
fi

# ── Generate manifest ────────────────────────────────────────────────────────
MANIFEST=$(cat <<MANIFEST_EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  labels:
    app: ${APP_LABEL}
    kueue.x-k8s.io/queue-name: ${QUEUE_NAME}
    kueue.x-k8s.io/priority-class: 8-gpu
    kueue.x-k8s.io/pod-group-name: "${POD_NAME}"
    skypilot-user: k8s-direct
    training-job: "${POD_NAME}"
  annotations:
    training-job-full: "${JOB_NAME}"
    kueue.x-k8s.io/pod-group-total-count: "1"
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
${_NODE_SELECTOR}
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
    - key: node.kubernetes.io/not-ready
      operator: Exists
      effect: NoExecute
      tolerationSeconds: 300
    - key: node.kubernetes.io/unreachable
      operator: Exists
      effect: NoExecute
      tolerationSeconds: 300
  containers:
    - name: main
      image: ${IMAGE}
      imagePullPolicy: Always
      env:
        - name: JOB_NAME
          value: $(yaml_quote "${JOB_NAME}")
        - name: JOB_TYPE
          value: $(yaml_quote "${JOB_TYPE}")
        - name: EXPERIMENT
          value: $(yaml_quote "${EXPERIMENT}")
        - name: PYTHON_MODULE
          value: $(yaml_quote "${PYTHON_MODULE}")
        - name: LOG_PREFIX
          value: $(yaml_quote "${LOG_PREFIX}")
        - name: HYDRA_OVERRIDES
          value: $(yaml_quote "${HYDRA_OVERRIDES}")
        - name: LAUNCH_MODE
          value: $(yaml_quote "${LAUNCH_MODE}")
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: WANDB_API_KEY
              optional: true
        - name: WANDB_ENTITY
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: WANDB_ENTITY
              optional: true
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: HF_TOKEN
              optional: true
        - name: HF_TOKEN_POOL_JSON
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: HF_TOKEN_POOL_JSON
              optional: true
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: OPENAI_API_KEY
              optional: true
        - name: R2_ENDPOINT_URL
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: R2_ENDPOINT_URL
              optional: true
        - name: R2_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: R2_ACCESS_KEY_ID
              optional: true
        - name: R2_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: R2_SECRET_ACCESS_KEY
              optional: true
        - name: R2_REGION
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: R2_REGION
              optional: true
        - name: PREPROCESSED_DATA_RCLONE_URI
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: PREPROCESSED_DATA_RCLONE_URI
              optional: true
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: SKIP_RUNTIME_INSTALLS
          value: $(yaml_quote "${SKIP_RUNTIME_INSTALLS}")
        # Cap the per-call math verifier subprocess timeout for TRAINING jobs
        # only (5 s instead of the 15 s default).  Most pathological sympy
        # compares are unrecoverable past 5 s; lowering reclaims ~10 s × N per
        # ans_loop on long-tail timeouts.  sol_eval / gen_eval keep the 15 s
        # default to match historical scoring (set to "" → mcq_utils reads as
        # unset and falls back to the 15 s built-in default).
        - name: VERL_MATH_VERIFY_TIMEOUT_S
          value: $(yaml_quote "${VERL_MATH_VERIFY_TIMEOUT_S_OVERRIDE}")
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -ex

          # ==================================================================
          # PHASE 1: Setup (shared across all job types)
          # ==================================================================
          echo "=============================================="
          echo "K8s Direct — Setup Phase"
          echo "Job type: \${JOB_TYPE}"
          echo "Job: \${JOB_NAME}"
          echo "=============================================="

          WORKSPACE="/root/my_workspace"
          mkdir -p "\${WORKSPACE}"
          cd "\${WORKSPACE}"

          # ── Copy workdir from PVC-backed repo ──
          PRIMARY_PVC_REPO="${HOST_REPO_ROOT}"
          WORKSPACE_MOUNT_PVC_REPO="${WORKSPACE_MOUNT_PVC_REPO}"
          LEGACY_PVC_REPO="${LEGACY_PVC_REPO}"
          if [ -d "\${PRIMARY_PVC_REPO}" ]; then
            PVC_REPO="\${PRIMARY_PVC_REPO}"
            echo "=== Copying workdir from current PVC repo ==="
          elif [ -d "\${WORKSPACE_MOUNT_PVC_REPO}" ]; then
            PVC_REPO="\${WORKSPACE_MOUNT_PVC_REPO}"
            echo "=== Copying workdir from workspace-mount PVC repo ==="
          elif [ -d "\${LEGACY_PVC_REPO}" ]; then
            PVC_REPO="\${LEGACY_PVC_REPO}"
            echo "=== Copying workdir from legacy PVC repo ==="
          else
            echo "ERROR: Repo not found at \${PRIMARY_PVC_REPO}, \${WORKSPACE_MOUNT_PVC_REPO}, or \${LEGACY_PVC_REPO}"
            exit 1
          fi
          if [ -d "\${PVC_REPO}" ]; then
            _SKYIGNORE_PATTERNS=()
            if [[ -f "\${PVC_REPO}/.skyignore" ]]; then
              while IFS= read -r line; do
                [[ -z "\$line" || "\$line" == \#* ]] && continue
                _SKYIGNORE_PATTERNS+=("\$line")
              done < "\${PVC_REPO}/.skyignore"
              echo "Using .skyignore exclusions (\${#_SKYIGNORE_PATTERNS[@]} patterns)"
            else
              _SKYIGNORE_PATTERNS=(
                '.git/' '__pycache__/' '*.pyc' '.output/' '.cache/' '.logs/'
                '.tmp/' 'wandb/' 'outputs/' 'logs/' '.claude/'
              )
              echo "Using fallback exclusions (no .skyignore found)"
            fi

            TAR_EXCLUDES=()
            for pat in "\${_SKYIGNORE_PATTERNS[@]}"; do
              pat="\${pat%/}"
              TAR_EXCLUDES+=(--exclude="\$pat")
            done

            tar cf - -C "\${PVC_REPO}" "\${TAR_EXCLUDES[@]}" . | tar xf - -C "\${WORKSPACE}"
            find "\${WORKSPACE}" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
            find "\${WORKSPACE}" -name '*.pyc' -delete 2>/dev/null || true
            echo "Workdir copied from \${PVC_REPO} (\$(du -sh \${WORKSPACE} 2>/dev/null | cut -f1))"
          fi

          REQUIRED_DATA_PATHS_IN_POD=(${REQUIRED_DATA_PATHS_LITERAL})

          copy_required_preprocessed_data() {
            local workspace="\$1"
            local source_repo="\$2"
            shift 2

            local rel src dst
            for rel in "\$@"; do
              [[ -z "\${rel}" ]] && continue
              src="\${source_repo}/\${rel}"
              dst="\${workspace}/\${rel}"
              if [[ -e "\${dst}" || ! -e "\${src}" ]]; then
                continue
              fi
              echo "Restoring required cache path from PVC repo: \${rel}"
              mkdir -p "\$(dirname "\${dst}")"
              if [[ -d "\${src}" ]]; then
                mkdir -p "\${dst}"
                tar cf - -C "\${src}" . | tar xf - -C "\${dst}"
              else
                cp "\${src}" "\${dst}"
              fi
            done
          }

          check_required_preprocessed_data() {
            local workspace="\$1"
            shift

            local missing=0 rel
            for rel in "\$@"; do
              [[ -z "\${rel}" ]] && continue
              if [[ ! -e "\${workspace}/\${rel}" ]]; then
                echo "ERROR: required cache path missing: \${workspace}/\${rel}" >&2
                missing=1
              fi
            done

            if [[ "\${missing}" -eq 1 ]]; then
              echo "Hint: either prepare local preprocessed files on the mounted PVC repo first or rerun without --offline-data" >&2
              exit 1
            fi
          }

          # ── Install system dependencies ──
          echo "=== Installing system tools ==="
          if [[ "\${SKIP_RUNTIME_INSTALLS:-false}" == "true" || "\${SKIP_RUNTIME_INSTALLS:-0}" == "1" ]]; then
            echo "Skipping system tool installs; using tools baked into image"
            command -v curl >/dev/null
            command -v wget >/dev/null
            command -v git >/dev/null
            command -v unzip >/dev/null
          else
            apt-get update -qq && apt-get install -y unzip curl wget git
          fi

          # ── Install rclone ──
          echo "=== Installing rclone ==="
          if [[ "\${SKIP_RUNTIME_INSTALLS:-false}" == "true" || "\${SKIP_RUNTIME_INSTALLS:-0}" == "1" ]]; then
            command -v rclone >/dev/null
            rclone version
          else
            curl -s https://rclone.org/install.sh | bash
          fi

          # ── Configure rclone for R2 ──
          echo "=== Configuring rclone for R2 ==="
          mkdir -p ~/.config/rclone
          if [[ -n "\${R2_ENDPOINT_URL:-}" && -n "\${R2_ACCESS_KEY_ID:-}" && -n "\${R2_SECRET_ACCESS_KEY:-}" ]]; then
            cat > ~/.config/rclone/rclone.conf << RCLONE_CONF_EOF
          [r2]
          type = s3
          provider = Cloudflare
          access_key_id = \${R2_ACCESS_KEY_ID}
          secret_access_key = \${R2_SECRET_ACCESS_KEY}
          endpoint = \${R2_ENDPOINT_URL}
          acl = private
          RCLONE_CONF_EOF
            echo "Configured rclone from R2_* environment variables"
          elif [[ -f ~/.config/rclone/rclone.conf ]]; then
            echo "Using existing rclone config from image/home"
          else
            echo "WARNING: no rclone config written; R2 data staging/log sync will fail unless configs avoid r2:// paths"
          fi
          rclone version

          # ── Install Python dependencies ──
          echo "=== Installing Python dependencies ==="
          if [[ "\${SKIP_RUNTIME_INSTALLS:-false}" == "true" || "\${SKIP_RUNTIME_INSTALLS:-0}" == "1" ]]; then
            echo "Skipping pod-startup pip installs; using dependencies baked into image"
            python -c "import importlib.util, sys, datasets; required = ('aiohttp', 'boto3', 'huggingface_hub', 'math_verify', 'openai', 'verl', 'termcolor', 'tree_sitter', 'tree_sitter_python', 'tempdir', 'wget', 'appdirs', 'multipledispatch', 'fire', 'rich', 'psutil', 'pebble'); missing = [m for m in required if not importlib.util.find_spec(m)]; sys.exit('Missing: ' + ', '.join(missing)) if missing else print('OK:', ', '.join(required), 'datasets=' + datasets.__version__)"
          else
            python -m pip install boto3 openai aiohttp huggingface_hub sympy stopit tempdir wget appdirs multipledispatch fire rich psutil pebble "datasets==3.6.0" --no-cache-dir || true
            python -m pip install verl --no-cache-dir || true
            python -m pip install "math-verify[antlr4_9_3]" --no-cache-dir || true
            python -m pip install termcolor "tree_sitter>=0.22.0" tree-sitter-python --no-cache-dir || true

            # ── Verify critical packages ──
            python -c "import importlib.util, sys, datasets; required = ('aiohttp', 'boto3', 'huggingface_hub', 'math_verify', 'openai', 'verl', 'termcolor', 'tree_sitter', 'tree_sitter_python', 'tempdir', 'wget', 'appdirs', 'multipledispatch', 'fire', 'rich', 'psutil', 'pebble'); missing = [m for m in required if not importlib.util.find_spec(m)]; sys.exit('Missing: ' + ', '.join(missing)) if missing else print('OK:', ', '.join(required), 'datasets=' + datasets.__version__)"
          fi

          cd "\${WORKSPACE}"

          # ── Source .env from repo for API keys ──
          if [ -f "\${WORKSPACE}/.env" ]; then
            echo "=== Sourcing .env for API keys ==="
            set +x
            set -a
            source "\${WORKSPACE}/.env"
            set +a
            set -x
          fi

          # ── Prepare preprocessed data ──
          echo "=== Preparing preprocessed data ==="
          DATA_DIR="\${WORKSPACE}/.cache/data/preprocessed"
          mkdir -p "\${DATA_DIR}"
          if [[ "${OFFLINE_DATA}" == "true" ]]; then
            echo "Offline mode: skipping preprocessed data rclone copy"
          elif [[ -n "\${PREPROCESSED_DATA_RCLONE_URI:-}" ]]; then
            rclone copy "\${PREPROCESSED_DATA_RCLONE_URI%/}/" "\${DATA_DIR}/" \
              --s3-no-check-bucket --transfers=16 --progress
          else
            echo "No PREPROCESSED_DATA_RCLONE_URI set; using local cache only"
          fi
          ls -la "\${DATA_DIR}/"

          echo "=== Preparing benchmark files from preprocessed cache ==="
          BENCHMARKS_DIR="\${WORKSPACE}/.cache/data/preprocessed/benchmarks"
          mkdir -p "\${BENCHMARKS_DIR}"
          if [[ "${OFFLINE_DATA}" == "true" ]]; then
            echo "Offline mode: skipping benchmark rclone copy"
          elif [[ -n "\${PREPROCESSED_DATA_RCLONE_URI:-}" ]]; then
            rclone copy "\${PREPROCESSED_DATA_RCLONE_URI%/}/benchmarks/" "\${BENCHMARKS_DIR}/" \
              --s3-no-check-bucket --transfers=16 2>/dev/null || echo "No benchmarks (OK)"
          else
            echo "No PREPROCESSED_DATA_RCLONE_URI set; using local benchmark cache only"
          fi

          if [[ "\${#REQUIRED_DATA_PATHS_IN_POD[@]}" -gt 0 ]]; then
            copy_required_preprocessed_data "\${WORKSPACE}" "\${PVC_REPO}" "\${REQUIRED_DATA_PATHS_IN_POD[@]}"
            check_required_preprocessed_data "\${WORKSPACE}" "\${REQUIRED_DATA_PATHS_IN_POD[@]}"
          fi

          # ── Cgroup fix for Ray memory monitor ──
          echo "=== Fixing cgroup memory visibility for Ray ==="
          _CGROUP_PATH="\$(cat /proc/self/cgroup 2>/dev/null | grep -oP '(?<=::).*' | head -1)"
          _CGROUP_FULL="/sys/fs/cgroup\${_CGROUP_PATH}"
          if [ -n "\${_CGROUP_PATH}" ] && [ -f "\${_CGROUP_FULL}/memory.max" ] && [ ! -f /sys/fs/cgroup/memory.max ]; then
            _MEM_TOTAL_KB=\$(awk '/^MemTotal:/ {print \$2}' /proc/meminfo 2>/dev/null || echo 0)
            _MEM_LIMIT_BYTES=\$(( _MEM_TOTAL_KB * 1024 * 90 / 100 ))
            _CUR_MAX="\$(cat \${_CGROUP_FULL}/memory.max 2>/dev/null)"
            if [ "\${_CUR_MAX}" = "max" ] && [ "\${_MEM_LIMIT_BYTES}" -gt 0 ]; then
              echo "\${_MEM_LIMIT_BYTES}" > "\${_CGROUP_FULL}/memory.max" 2>/dev/null || true
            fi
            mount --bind "\${_CGROUP_FULL}" /sys/fs/cgroup/ 2>/dev/null \
              && echo "Cgroup fix applied" \
              || echo "Cgroup bind-mount failed (Ray memory monitor may be limited)"
          elif [ -f /sys/fs/cgroup/memory.max ]; then
            echo "Cgroup already visible — no fix needed"
          fi

          # ── Pre-download model files (avoid race in parallel Ray workers) ──
          echo "=== Pre-downloading model files ==="
          python \${WORKSPACE}/launcher/k8s/pre_download_models.py "\${HYDRA_OVERRIDES}" || echo "WARNING: Model pre-download failed (non-fatal)"

          echo "=== Setup complete ==="

$(cat <<'RUN_PHASE'
          # ==================================================================
          # PHASE 2: Run
          # ==================================================================
          export WORKSPACE
          bash "${WORKSPACE}/launcher/k8s/run_job.sh"
RUN_PHASE
)
      resources:
        requests:
          cpu: "${CPU_COUNT}"
          memory: "${MEMORY_GI}Gi"${_GPU_RESOURCE_REQ}
${_GPU_LIMITS_BLOCK}
      securityContext:
        capabilities:
          add: [SYS_RESOURCE, IPC_LOCK, SYS_PTRACE]
        privileged: true
        runAsUser: 0
        runAsGroup: 0
      volumeMounts:
        - name: dshm
          mountPath: /dev/shm
        - name: workspace
          mountPath: /workspace
        - name: workspace
          mountPath: /nemo-workspace
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory
    - name: workspace
      persistentVolumeClaim:
        claimName: ${PVC_NAME}
MANIFEST_EOF
)

SECRET_MANIFEST=$(cat <<SECRET_MANIFEST_EOF
apiVersion: v1
kind: Secret
metadata:
  name: ${SECRET_NAME}
  labels:
    skypilot-user: k8s-direct
    training-job: "${POD_NAME}"
  annotations:
    training-job-full: "${JOB_NAME}"
type: Opaque
stringData:
  WANDB_API_KEY: $(yaml_quote "${WANDB_API_KEY}")
  WANDB_ENTITY: $(yaml_quote "${WANDB_ENTITY}")
  HF_TOKEN: $(yaml_quote "${HF_TOKEN}")
  HF_TOKEN_POOL_JSON: $(yaml_quote "${HF_TOKEN_POOL_JSON}")
  OPENAI_API_KEY: $(yaml_quote "${OPENAI_API_KEY}")
  R2_ENDPOINT_URL: $(yaml_quote "${R2_ENDPOINT_URL}")
  R2_ACCESS_KEY_ID: $(yaml_quote "${R2_ACCESS_KEY_ID}")
  R2_SECRET_ACCESS_KEY: $(yaml_quote "${R2_SECRET_ACCESS_KEY}")
  R2_REGION: $(yaml_quote "${R2_REGION}")
  PREPROCESSED_DATA_RCLONE_URI: $(yaml_quote "${PREPROCESSED_DATA_RCLONE_URI}")
SECRET_MANIFEST_EOF
)

# ── Apply or print ───────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" == true ]]; then
  echo "=== DRY RUN — secret: ==="
  echo "Secret '${SECRET_NAME}' would be created with runtime credential/config keys (values redacted)."
  echo ""
  echo "=== DRY RUN — manifest: ==="
  echo "${MANIFEST}"
  echo ""
  echo "To apply: rerun without --dry-run"
  exit 0
fi

echo "${SECRET_MANIFEST}" | kubectl apply -f - >/dev/null
echo "${MANIFEST}" | kubectl apply -f -

echo ""
echo "========================================"
echo "Pod '${POD_NAME}' created — ${JOB_TYPE}"
echo "========================================"
echo ""
echo "Job name:  ${JOB_NAME}"
echo "Pod name:  ${POD_NAME}"
echo ""

# ── Archive directory ────────────────────────────────────────────────────────
ARCHIVE_DIR="${REPO_ROOT}/.logs/job_archives/by_job/${JOB_NAME}/watcher"
mkdir -p "${ARCHIVE_DIR}"
cat > "${ARCHIVE_DIR}/state.json" <<STATE_EOF
{
  "job_name": "${JOB_NAME}",
  "pod_name": "${POD_NAME}",
  "job_type": "${JOB_TYPE}",
  "phase": "pending",
  "backend": "k8s-direct",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
STATE_EOF
echo "Archive dir: ${ARCHIVE_DIR}"
echo ""

# ── Log access ───────────────────────────────────────────────────────────────
echo "State file: ${ARCHIVE_DIR}/state.json"
echo "Follow logs with:"
echo "  kubectl logs -f ${POD_NAME}"
echo ""

# ── Wait for pod scheduling ──────────────────────────────────────────────────
echo "Waiting for pod to be scheduled..."
kubectl wait --for=condition=PodScheduled pod/"${POD_NAME}" --timeout=600s 2>/dev/null || true

echo ""
echo "Commands:"
echo "  Logs:    kubectl logs ${POD_NAME} -f"
echo "  Exec:    kubectl exec -it ${POD_NAME} -- bash"
echo "  Status:  kubectl get pod ${POD_NAME}"
echo "  Delete:  kubectl delete pod ${POD_NAME}"
echo ""

# ── Stream logs ──────────────────────────────────────────────────────────────
if [[ "${STREAM_LOGS}" == true ]]; then
  echo "=== Streaming logs (Ctrl+C to detach, pod + watcher continue) ==="
  echo ""

  (
    while ! kubectl logs "${POD_NAME}" --tail=1 &>/dev/null; do
      sleep 2
    done
    kubectl logs "${POD_NAME}" -f 2>/dev/null
  ) &
  LOG_STREAM_PID=$!

  trap "kill ${LOG_STREAM_PID} 2>/dev/null; echo ''; echo 'Detached. Pod + watcher continue running.'; exit 0" INT
  wait ${LOG_STREAM_PID} 2>/dev/null || true
fi
