#!/bin/bash
# Shared in-pod runner for direct K8s jobs.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/my_workspace}"
cd "${WORKSPACE}"

echo "=============================================="
echo "K8s Direct — Run Phase"
echo "Job type: ${JOB_TYPE}"
echo "Job: ${JOB_NAME}"
echo "=============================================="

NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "Detected ${NUM_GPUS} GPUs"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RAY_TMPDIR="${WORKSPACE}/.cache/ray_tmp"
mkdir -p "${RAY_TMPDIR}"

if [ -z "${WANDB_API_KEY:-}" ]; then
  unset WANDB_API_KEY
  export WANDB_MODE=offline
  echo "WANDB offline mode"
else
  echo "WANDB tracking enabled"
fi
[ -z "${HF_TOKEN:-}" ] && unset HF_TOKEN
[ -z "${HF_TOKEN_POOL_JSON:-}" ] && unset HF_TOKEN_POOL_JSON
[ -z "${OPENAI_API_KEY:-}" ] && unset OPENAI_API_KEY

CONFIG_NAME_ARG=""
FILTERED_OVERRIDES=""
for arg in ${HYDRA_OVERRIDES}; do
  if [[ "$arg" == --config-name=* ]]; then
    CONFIG_NAME_ARG="$arg"
  else
    FILTERED_OVERRIDES="${FILTERED_OVERRIDES} $arg"
  fi
done

LOG_DIR=".output"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/${LOG_PREFIX}_stdout.log"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_R2_NAME="${LOG_PREFIX}_stdout_${RUN_ID}.log"
echo "=== Run ID: ${RUN_ID} ==="
echo "=== Log: ${MAIN_LOG} ==="

RESOLVED_REMOTE_SYNC_PATH="$(python3 launcher/resolve_sync_path.py ${CONFIG_NAME_ARG} ${FILTERED_OVERRIDES} 2>/dev/null || true)"
REMOTE_BACKEND="none"
LOG_R2_DEST=""
case "${RESOLVED_REMOTE_SYNC_PATH}" in
  hf://*) REMOTE_BACKEND="hf" ;;
  s3://*) REMOTE_BACKEND="r2"; LOG_R2_DEST="$(echo "${RESOLVED_REMOTE_SYNC_PATH}" | sed 's|^s3://|r2:|')/logs" ;;
  r2://*) REMOTE_BACKEND="r2"; LOG_R2_DEST="$(echo "${RESOLVED_REMOTE_SYNC_PATH}" | sed 's|^r2://|r2:|')/logs" ;;
esac
echo "Remote backend: ${REMOTE_BACKEND}"

LOG_SYNC_PID=""
CGROUP_WATCHDOG_PID=""
LOG_SYNC_ERR_FILE="${LOG_DIR}/log_sync_errors.log"
if [ -n "${LOG_R2_DEST}" ] && command -v rclone >/dev/null 2>&1; then
  echo "=== Log sync enabled: ${LOG_R2_DEST}/${LOG_R2_NAME} ==="
  touch "${MAIN_LOG}"
  rclone copyto "${MAIN_LOG}" "${LOG_R2_DEST}/${LOG_R2_NAME}" \
    --s3-no-check-bucket 2>"${LOG_SYNC_ERR_FILE}" || true

  _sync_log_loop() {
    local fail=0
    while true; do
      sleep 15
      if [ -f "${MAIN_LOG}" ]; then
        if ! rclone copyto "${MAIN_LOG}" "${LOG_R2_DEST}/${LOG_R2_NAME}" \
            --s3-no-check-bucket 2>>"${LOG_SYNC_ERR_FILE}"; then
          fail=$((fail + 1))
          [ $((fail % 20)) -eq 1 ] && echo "WARNING: log sync failed (count=${fail})"
        elif [ "${fail}" -gt 0 ]; then
          echo "=== Log sync recovered after ${fail} failures ==="
          fail=0
        fi
      fi
    done
  }
  _sync_log_loop &
  LOG_SYNC_PID=$!
else
  echo "=== Log sync disabled ==="
fi

cleanup_background() {
  [ -n "${LOG_SYNC_PID}" ] && kill "${LOG_SYNC_PID}" 2>/dev/null || true
  [ -n "${CGROUP_WATCHDOG_PID}" ] && kill "${CGROUP_WATCHDOG_PID}" 2>/dev/null || true
}

flush_final_log() {
  if [ -n "${LOG_R2_DEST}" ] && [ -f "${MAIN_LOG}" ] && command -v rclone >/dev/null 2>&1; then
    rclone copyto "${MAIN_LOG}" "${LOG_R2_DEST}/${LOG_R2_NAME}" \
      --s3-no-check-bucket 2>>"${LOG_SYNC_ERR_FILE}" && echo "Log uploaded" || echo "Upload failed"
  fi
}

on_signal() {
  echo "=== Signal received — flushing final log ===" >> "${MAIN_LOG}" 2>/dev/null || true
  flush_final_log
  cleanup_background
  exit 143
}
trap on_signal SIGTERM SIGINT

_cgroup_mem_watchdog() {
  set +x
  local mem_max_file="" mem_cur_file="" use_procmeminfo=false
  if [ -f /sys/fs/cgroup/memory.max ]; then
    mem_max_file="/sys/fs/cgroup/memory.max"
    mem_cur_file="/sys/fs/cgroup/memory.current"
  elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    mem_max_file="/sys/fs/cgroup/memory/memory.limit_in_bytes"
    mem_cur_file="/sys/fs/cgroup/memory/memory.usage_in_bytes"
  fi
  if [ -z "${mem_max_file}" ]; then
    [ -f /proc/meminfo ] && use_procmeminfo=true || return
  fi
  while true; do
    sleep 30
    if [ "${use_procmeminfo}" = "true" ]; then
      local total avail used pct tg ug
      total="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
      avail="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
      used="$((total - avail))"
      pct="$((used * 100 / total))"
      tg="$(awk "BEGIN {printf \"%.1f\", ${total}/1048576}")"
      ug="$(awk "BEGIN {printf \"%.1f\", ${used}/1048576}")"
      [ "${pct}" -ge 85 ] && echo "[mem-watchdog] CRITICAL: ${ug}GB/${tg}GB (${pct}%)" || \
      [ "${pct}" -ge 70 ] && echo "[mem-watchdog] WARNING: ${ug}GB/${tg}GB (${pct}%)" || \
      echo "[mem-watchdog] OK: ${ug}GB/${tg}GB (${pct}%)"
    else
      local max cur pct cg mg
      max="$(cat "${mem_max_file}" 2>/dev/null || echo "0")"
      cur="$(cat "${mem_cur_file}" 2>/dev/null || echo "0")"
      [ "${max}" = "max" ] || [ "${max}" = "0" ] && continue
      pct="$((cur * 100 / max))"
      cg="$(awk "BEGIN {printf \"%.1f\", ${cur}/1073741824}")"
      mg="$(awk "BEGIN {printf \"%.1f\", ${max}/1073741824}")"
      [ "${pct}" -ge 85 ] && echo "[mem-watchdog] CRITICAL: ${cg}GB/${mg}GB (${pct}%)" || \
      [ "${pct}" -ge 70 ] && echo "[mem-watchdog] WARNING: ${cg}GB/${mg}GB (${pct}%)" || \
      echo "[mem-watchdog] OK: ${cg}GB/${mg}GB (${pct}%)"
    fi
  done
}
_cgroup_mem_watchdog &
CGROUP_WATCHDOG_PID=$!

run_with_logging() {
  if [ "$$" -ne 1 ] && [ -w /proc/1/fd/1 ]; then
    "$@" 2>&1 | tee -a "${MAIN_LOG}" /proc/1/fd/1
    return "${PIPESTATUS[0]}"
  fi
  "$@" 2>&1 | tee -a "${MAIN_LOG}"
  return "${PIPESTATUS[0]}"
}

if [ "${LAUNCH_MODE}" = "interactive" ] && [ "${K8S_MANUAL_SUBMIT:-0}" != "1" ]; then
  cat <<EOF
==============================================
Interactive pod is ready.

Workspace: ${WORKSPACE}
Log file:  ${MAIN_LOG}

Submit training from another shell with:
  kubectl exec -it ${HOSTNAME} -- bash
  cd ${WORKSPACE}
  K8S_MANUAL_SUBMIT=1 ./launcher/k8s/run_job.sh

When run from kubectl exec, this script mirrors stdout into /proc/1/fd/1 so
kubectl logs and the pod watcher can still follow the manual training run.
==============================================
EOF
  while true; do
    sleep 3600
  done
fi

EFFECTIVE_LAUNCH_MODE="${LAUNCH_MODE}"
if [ "${LAUNCH_MODE}" = "interactive" ]; then
  EFFECTIVE_LAUNCH_MODE="hydra"
fi

if [ "${EFFECTIVE_LAUNCH_MODE}" = "script" ]; then
  echo "=============================================="
  echo "Starting: python ${PYTHON_MODULE}"
  echo "  ${HYDRA_OVERRIDES}"
  echo "=============================================="
  run_with_logging python "${PYTHON_MODULE}" ${HYDRA_OVERRIDES}
  JOB_EXIT_CODE=$?
else
  echo "=============================================="
  echo "Starting: python -m ${PYTHON_MODULE}"
  echo "  ${CONFIG_NAME_ARG} trainer.n_gpus_per_node=${NUM_GPUS} ${FILTERED_OVERRIDES}"
  echo "=============================================="
  run_with_logging python -m "${PYTHON_MODULE}" \
    ${CONFIG_NAME_ARG} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    hydra.run.dir=. \
    hydra.output_subdir=null \
    ${FILTERED_OVERRIDES}
  JOB_EXIT_CODE=$?
fi

echo "=== Uploading final log to R2 ==="
flush_final_log
cleanup_background

echo "=============================================="
echo "${JOB_TYPE} finished with exit code: ${JOB_EXIT_CODE}"
echo "=============================================="
exit "${JOB_EXIT_CODE}"
