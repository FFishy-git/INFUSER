#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${IMAGE:-self-evolution-explore:runtime}"

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage:
  IMAGE=ghcr.io/YOUR_ORG/self-evolution-explore:runtime \
    ./launcher/opensource/launch_k8s.sh EXPERIMENT_OR_EVAL_CONFIG [launcher/k8s/launch.sh options]

This wrapper delegates to launcher/k8s/launch.sh and injects --image unless
you already supplied --image yourself.
EOF
  exit 2
fi

has_image_arg=false
has_skip_installs_arg=false
for arg in "$@"; do
  if [[ "${arg}" == "--image" ]]; then
    has_image_arg=true
  elif [[ "${arg}" == "--skip-runtime-installs" ]]; then
    has_skip_installs_arg=true
  fi
done

args=("$@")
if [[ "${has_image_arg}" != "true" ]]; then
  args+=(--image "${IMAGE}")
fi
if [[ "${has_skip_installs_arg}" != "true" ]]; then
  args+=(--skip-runtime-installs)
fi

exec "${REPO_ROOT}/launcher/k8s/launch.sh" "${args[@]}"
