#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${IMAGE:-self-evolution-explore:runtime}"
BASE_IMAGE="${BASE_IMAGE:-verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d}"
INSTALL_OPENCOMPASS="${INSTALL_OPENCOMPASS:-1}"
PLATFORM="${PLATFORM:-linux/amd64}"
RCLONE_VERSION="${RCLONE_VERSION:-1.72.0}"

docker build \
  --platform "${PLATFORM}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "INSTALL_OPENCOMPASS=${INSTALL_OPENCOMPASS}" \
  --build-arg "RCLONE_VERSION=${RCLONE_VERSION}" \
  -f "${SCRIPT_DIR}/Dockerfile.runtime" \
  -t "${IMAGE}" \
  "${REPO_ROOT}"

if [[ "${PUSH:-0}" == "1" ]]; then
  docker push "${IMAGE}"
fi

echo "Built image: ${IMAGE}"
if [[ "${PUSH:-0}" == "1" ]]; then
  echo "Inspect pushed digest with:"
  echo "  docker buildx imagetools inspect ${IMAGE}"
fi
