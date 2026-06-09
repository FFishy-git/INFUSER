#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

INSTALL=1
DOWNLOAD_DATA=1
INSTALL_OPENCOMPASS=0
RUN_CHECK=1
FORCE_DATA=0
DATA_DIR=".cache/data"
HF_REPO="Siyuc/infuser_train"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [options]

Set up INFUSER for open-source use from the repository root.

By default this script:
  1. creates .env from .env.example if needed;
  2. creates local .cache/data and .output directories;
  3. installs INFUSER's Python dependency layer from requirements.txt;
  4. downloads released preprocessed data from Hugging Face;
  5. checks that core runtime imports are visible.

Options:
  --no-install          Skip pip install -r requirements.txt.
  --no-data             Skip downloading released data.
  --force-data          Re-download data even if .cache/data/preprocessed exists.
  --with-opencompass    Install optional HumanEval/LiveCodeBench dependencies.
  --no-check            Skip import checks.
  --data-dir DIR        Data root to populate. Default: .cache/data.
  --hf-repo REPO        Hugging Face dataset repo. Default: Siyuc/infuser_train.
  -h, --help            Show this help.

Notes:
  - Run this inside a CUDA/PyTorch/vLLM/Ray/verl-compatible environment.
    The pinned open-source base image is documented in launcher/opensource/.
  - This script does not install torch, vLLM, Ray, or the base verl runtime.
  - Set HF_TOKEN before running if you need gated/private Hugging Face access.
EOF
}

log() {
  printf '\n==> %s\n' "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-install)
      INSTALL=0
      shift
      ;;
    --no-data)
      DOWNLOAD_DATA=0
      shift
      ;;
    --force-data)
      FORCE_DATA=1
      shift
      ;;
    --with-opencompass)
      INSTALL_OPENCOMPASS=1
      shift
      ;;
    --no-check)
      RUN_CHECK=0
      shift
      ;;
    --data-dir)
      [[ $# -ge 2 ]] || die "--data-dir requires a value"
      DATA_DIR="$2"
      shift 2
      ;;
    --hf-repo)
      [[ $# -ge 2 ]] || die "--hf-repo requires a value"
      HF_REPO="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -f "requirements.txt" ]] || die "requirements.txt not found; run from the INFUSER repository root"
[[ -d "verl_inf_evolve" ]] || die "verl_inf_evolve/ not found; run from the INFUSER repository root"
[[ -d "verl" ]] || die "vendored verl/ directory not found"

log "Repository: ${ROOT_DIR}"

if [[ ! -f ".env" && -f ".env.example" ]]; then
  log "Creating .env from .env.example"
  cp .env.example .env
else
  log ".env already exists or .env.example is missing; leaving credentials untouched"
fi

log "Creating local output/cache directories"
mkdir -p ".output" "${DATA_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"
log "PYTHONPATH includes repository root and vendored verl/"

if [[ "${INSTALL}" -eq 1 ]]; then
  log "Installing INFUSER Python dependency layer"
  python -m pip install -r requirements.txt
else
  log "Skipping dependency install"
fi

if [[ "${INSTALL_OPENCOMPASS}" -eq 1 ]]; then
  log "Installing optional OpenCompass/EvalPlus dependencies"
  python -m pip install -r launcher/opensource/requirements-opencompass.txt
  python -m pip install evalplus==0.3.1 --no-deps
  python -m pip install tree-sitter==0.25.2 tree-sitter-python==0.25.0
fi

PREPROCESSED_DIR="${DATA_DIR%/}/preprocessed"
if [[ "${DOWNLOAD_DATA}" -eq 1 ]]; then
  if [[ -d "${PREPROCESSED_DIR}" && "${FORCE_DATA}" -eq 0 ]]; then
    log "Found ${PREPROCESSED_DIR}; skipping data download"
    echo "Use --force-data to re-download released data."
  else
    log "Downloading released preprocessed data from ${HF_REPO}"
    python launcher/preparation/download_data.py \
      --use-preprocessed \
      --hf-repo "${HF_REPO}" \
      --output-dir "${DATA_DIR}"
  fi
else
  log "Skipping data download"
fi

if [[ "${RUN_CHECK}" -eq 1 ]]; then
  log "Checking core imports"
  python - <<'PY'
import importlib

packages = [
    "datasets",
    "ray",
    "torch",
    "vllm",
    "verl",
    "verl_inf_evolve",
]

for name in packages:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    if name == "torch":
        print(f"{name}: {version} cuda={module.cuda.is_available()}")
    else:
        print(f"{name}: {version}")
PY
fi

cat <<EOF

Setup complete.

For future shell sessions, run commands from this repository root or export:
  export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl:\${PYTHONPATH:-}"

If you created .env for the first time, edit it before launching runs that need
Hugging Face, W&B, R2/S3, or API judge credentials.
EOF
