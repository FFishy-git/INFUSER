#!/bin/bash
# =============================================================================
# Cancel a direct K8s job
# =============================================================================
# Finds and deletes the pod by job name (or pod name), and stops the watcher.
# Supports all k8s-direct job types, including training, gen_eval, sol_eval, and opencompass.
#
# USAGE:
#   ./cancel_training.sh <job-name-or-pod-name>
#   ./cancel_training.sh --all          # cancel all k8s-direct pods
#   ./cancel_training.sh --list         # list running k8s-direct pods
#
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── List k8s-direct pods ─────────────────────────────────────────────────────
list_pods() {
  echo "Running k8s-direct pods:"
  echo ""
  kubectl get pods -l skypilot-user=k8s-direct \
    -o custom-columns='POD:metadata.name,APP:metadata.labels.app,STATUS:status.phase,AGE:metadata.creationTimestamp,JOB:metadata.annotations.training-job-full' \
    2>/dev/null || echo "  (none found)"
  echo ""
}

# ── Cancel by pod name ───────────────────────────────────────────────────────
cancel_pod() {
  local pod_name="$1"
  echo "Deleting pod: ${pod_name}"
  kubectl delete pod "${pod_name}" --grace-period=60 --wait=false 2>/dev/null || true

  # Kueue adds a finalizer (kueue.x-k8s.io/managed) that blocks deletion.
  # Remove it so the pod actually goes away.
  local retries=0
  while kubectl get pod "${pod_name}" &>/dev/null && [[ ${retries} -lt 10 ]]; do
    local finalizers
    finalizers="$(kubectl get pod "${pod_name}" -o jsonpath='{.metadata.finalizers}' 2>/dev/null || echo "")"
    if [[ "${finalizers}" == *"kueue"* ]]; then
      echo "Removing Kueue finalizer..."
      kubectl patch pod "${pod_name}" -p '{"metadata":{"finalizers":null}}' 2>/dev/null || true
    fi
    sleep 2
    retries=$((retries + 1))
  done

  # Verify pod is gone
  if kubectl get pod "${pod_name}" &>/dev/null; then
    echo "WARNING: Pod still exists after delete — may need manual cleanup."
    echo "  kubectl patch pod ${pod_name} -p '{\"metadata\":{\"finalizers\":null}}'"
    echo "  kubectl delete pod ${pod_name} --force --grace-period=0"
  else
    echo "Pod deleted successfully."
  fi

  if kubectl get secret "${pod_name}" &>/dev/null; then
    echo "Deleting runtime secret: ${pod_name}"
    kubectl delete secret "${pod_name}" --wait=false 2>/dev/null || true
  fi

  # Delete the Kueue workload object.  Even after the pod is gone, the
  # workload can linger in Admitted state and hold a GPU reservation,
  # blocking other jobs from being admitted.
  if kubectl get workload "${pod_name}" &>/dev/null; then
    echo "Deleting Kueue workload: ${pod_name}"
    kubectl delete workload "${pod_name}" --wait=false 2>/dev/null || true
    # Remove workload finalizers if stuck
    local wl_retries=0
    while kubectl get workload "${pod_name}" &>/dev/null && [[ ${wl_retries} -lt 5 ]]; do
      kubectl patch workload "${pod_name}" --type=merge \
        -p '{"metadata":{"finalizers":null}}' 2>/dev/null || true
      sleep 2
      wl_retries=$((wl_retries + 1))
    done
    if kubectl get workload "${pod_name}" &>/dev/null; then
      echo "WARNING: Kueue workload still exists — may need manual cleanup."
      echo "  kubectl delete workload ${pod_name}"
    else
      echo "Kueue workload deleted (GPU reservation released)."
    fi
  fi
}

# ── Stop watcher by job name ─────────────────────────────────────────────────
stop_watcher() {
  local job_name="$1"
  local pid_file="${REPO_ROOT}/.logs/job_archives/by_job/${job_name}/watcher/watcher.pid"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping watcher (PID ${pid})..."
      kill "${pid}" 2>/dev/null || true
      # Wait briefly for clean shutdown
      local w=0
      while kill -0 "${pid}" 2>/dev/null && [[ ${w} -lt 5 ]]; do
        sleep 1
        w=$((w + 1))
      done
      if kill -0 "${pid}" 2>/dev/null; then
        echo "Watcher didn't exit cleanly, sending SIGKILL..."
        kill -9 "${pid}" 2>/dev/null || true
      else
        echo "Watcher stopped."
      fi
    else
      echo "Watcher already stopped (stale PID ${pid})."
    fi
  else
    echo "No watcher PID file found."
  fi
}

# ── Resolve job name → pod name ──────────────────────────────────────────────
resolve_pod() {
  local query="$1"

  # Try direct pod name first
  if kubectl get pod "${query}" &>/dev/null; then
    echo "${query}"
    return 0
  fi

  # Search by training-job-full annotation across all k8s-direct pods
  local pod
  pod="$(kubectl get pods -l skypilot-user=k8s-direct \
    -o jsonpath="{range .items[?(@.metadata.annotations.training-job-full==\"${query}\")]}{.metadata.name}{end}" \
    2>/dev/null)"
  if [[ -n "${pod}" ]]; then
    echo "${pod}"
    return 0
  fi

  # Search by state.json in archives
  local state_file="${REPO_ROOT}/.logs/job_archives/by_job/${query}/watcher/state.json"
  if [[ -f "${state_file}" ]]; then
    local pod_from_state
    pod_from_state="$(python3 -c "import json; print(json.load(open('${state_file}')).get('pod_name',''))" 2>/dev/null)"
    if [[ -n "${pod_from_state}" ]]; then
      echo "${pod_from_state}"
      return 0
    fi
  fi

  # Partial match on pod name across all k8s-direct jobs
  pod="$(kubectl get pods -l skypilot-user=k8s-direct -o name 2>/dev/null | grep "${query}" | head -1 | sed 's|pod/||')"
  if [[ -n "${pod}" ]]; then
    echo "${pod}"
    return 0
  fi

  return 1
}

# ── Main ─────────────────────────────────────────────────────────────────────
FORCE=false
ARGS=()
for arg in "$@"; do
  case "$arg" in
    -y|--yes) FORCE=true ;;
    *) ARGS+=("$arg") ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <job-name-or-pod-name>"
  echo "       $0 --list"
  echo "       $0 --all [-y]"
  exit 1
fi

case "$1" in
  --list|-l)
    list_pods
    exit 0
    ;;
  --all|-a)
    list_pods
    if [[ "${FORCE}" != true ]]; then
      read -p "Delete ALL training pods? (y/n) " -n 1 -r
      echo
      [[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    fi
    # Get all k8s-direct pod names for watcher cleanup
    ALL_PODS="$(kubectl get pods -l skypilot-user=k8s-direct -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)"
    ALL_JOBS="$(kubectl get pods -l skypilot-user=k8s-direct -o jsonpath='{range .items[*]}{.metadata.annotations.training-job-full}{"\n"}{end}' 2>/dev/null)"
    # Delete pods
    kubectl delete pods -l skypilot-user=k8s-direct --grace-period=60 --wait=false 2>/dev/null || true
    # Delete per-job runtime secrets created by launcher/k8s/launch.sh.
    kubectl delete secret -l skypilot-user=k8s-direct --wait=false 2>/dev/null || true
    # Remove Kueue finalizers
    for pod in ${ALL_PODS}; do
      kubectl patch pod "${pod}" -p '{"metadata":{"finalizers":null}}' 2>/dev/null || true
    done
    # Stop watchers
    for job in ${ALL_JOBS}; do
      [[ -n "${job}" ]] && stop_watcher "${job}"
    done
    # Verify
    sleep 3
    local_remaining="$(kubectl get pods -l skypilot-user=k8s-direct --no-headers 2>/dev/null | wc -l)"
    if [[ "${local_remaining}" -eq 0 ]]; then
      echo "All k8s-direct pods deleted successfully."
    else
      echo "WARNING: ${local_remaining} pod(s) still remain."
      kubectl get pods -l skypilot-user=k8s-direct 2>/dev/null
    fi
    exit 0
    ;;
  *)
    QUERY="$1"
    POD_NAME="$(resolve_pod "${QUERY}")" || {
      echo "Could not find pod for: ${QUERY}"
      echo ""
      list_pods
      exit 1
    }

    # Try to find job name from annotation
    JOB_NAME="$(kubectl get pod "${POD_NAME}" -o jsonpath='{.metadata.annotations.training-job-full}' 2>/dev/null || echo "${QUERY}")"

    echo "========================================"
    echo "Cancelling training job"
    echo "========================================"
    echo "Job name: ${JOB_NAME}"
    echo "Pod name: ${POD_NAME}"
    echo ""

    cancel_pod "${POD_NAME}"
    stop_watcher "${JOB_NAME}"

    echo ""
    echo "Done. The watcher will archive final logs before exiting."
    ;;
esac
