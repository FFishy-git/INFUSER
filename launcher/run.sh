#!/usr/bin/env bash
# Unified launcher entrypoint for k8s / slurm / local backends.
#
# Backend selection:
#   --backend k8s    -> launcher/k8s/launch.sh (default)
#   --backend slurm  -> launcher/slurm/launch.sh
#   --backend local  -> launcher/local/launch.sh
#
# Remaining args are forwarded directly to the selected backend launcher.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="k8s"

usage() {
  cat <<'EOF'
Usage:
  ./launcher/run.sh [--backend k8s|slurm|local] <experiment_name> [launcher options]

Examples:
  ./launcher/run.sh --backend k8s --job-type training FW-example
  ./launcher/run.sh --backend slurm --job-type sol_eval --offline-data Eval-Run
  ./launcher/run.sh --backend local --job-type gen_eval replay-Run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "${BACKEND}" ]]; then
  echo "ERROR: --backend requires one of k8s|slurm|local" >&2
  usage
  exit 1
fi

case "${BACKEND}" in
  k8s)
    exec "${REPO_ROOT}/launcher/k8s/launch.sh" "$@"
    ;;
  slurm)
    exec "${REPO_ROOT}/launcher/slurm/launch.sh" "$@"
    ;;
  local)
    exec "${REPO_ROOT}/launcher/local/launch.sh" "$@"
    ;;
  *)
    echo "ERROR: unknown backend '${BACKEND}'. Use k8s, slurm, or local." >&2
    usage
    exit 1
    ;;
esac
