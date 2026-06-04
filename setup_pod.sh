#!/bin/bash
set -ex

## ------------------------ ##
# If you find permission issue, run
# kubectl exec -it csi-mounted-fs-path-plugin-12csy -- bash
# chown -R developer:developer /workspace/inf-evolve
# chown -R developer:developer /workspace/inf_evolve
## ------------------------ ##

# cd /workspace/self-evolution-explore

# === Step 1: Setup .output symlink ===
echo "=== Setting up .output symlink ==="
mkdir -p /workspace/inf-evolve/.output
if [ ! -e .output ]; then
  ln -s /workspace/inf-evolve/.output .output
  echo ".output symlink created"
else
  echo ".output already exists: $(ls -la .output)"
fi

# === Step 1.5: Setup .cache symlink ===
echo "=== Setting up .cache symlink ==="
mkdir -p /workspace/inf-evolve/.cache
if [ ! -e .cache ]; then
  ln -s /workspace/inf-evolve/.cache .cache
  echo ".cache symlink created"
else
  echo ".cache already exists: $(ls -la .cache)"
fi

# === Step 2: Download preprocessed data from R2 ===
echo "=== Downloading preprocessed data from R2 ==="
export DATA_DIR=".cache/data/preprocessed"
mkdir -p "${DATA_DIR}"
if [ -n "${PREPROCESSED_DATA_RCLONE_URI:-}" ]; then
  rclone copy "${PREPROCESSED_DATA_RCLONE_URI%/}/" "${DATA_DIR}/" --s3-no-check-bucket --transfers=16 --progress
else
  echo "PREPROCESSED_DATA_RCLONE_URI not set; using existing local cache"
fi

# Verify data
ls -la "${DATA_DIR}/"

# === Step 3: Download benchmark files from R2 ===
echo "=== Downloading benchmark files from R2 ==="
export BENCHMARKS_DIR=".cache/data/preprocessed/benchmarks"
mkdir -p "${BENCHMARKS_DIR}"
if [ -n "${PREPROCESSED_DATA_RCLONE_URI:-}" ]; then
  rclone copy "${PREPROCESSED_DATA_RCLONE_URI%/}/benchmarks/" "${BENCHMARKS_DIR}/" --s3-no-check-bucket --transfers=16 2>/dev/null || echo "No benchmark files found at PREPROCESSED_DATA_RCLONE_URI (OK if not using benchmark datasets)"
else
  echo "PREPROCESSED_DATA_RCLONE_URI not set; using existing local benchmark cache"
fi
if [ -d "${BENCHMARKS_DIR}" ] && [ "$(ls -A ${BENCHMARKS_DIR} 2>/dev/null)" ]; then
  echo "Benchmark files downloaded:"
  ls -la "${BENCHMARKS_DIR}/"
else
  echo "No benchmark files downloaded (directory is empty)"
fi

echo "=== Setup complete ==="
