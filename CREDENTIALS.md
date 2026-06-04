# Credentials and Remote Storage

This repository does not require credentials for code checkout, local unit
tests, or offline runs that use public models and local data. Credentials are
only needed when a workflow talks to a private service: Hugging Face private
repos, R2/S3 remote storage, Weights & Biases, or an external LLM API.

Do not commit real credentials. Put them in a local `.env`, a scheduler secret,
or your shell environment.

## Quick Start

```bash
cp .env.example .env
```

Fill only the variables needed by your workflow. The launchers source `.env`
from the repository root when it exists. Direct Python entrypoints also support
`VERL_INF_EVOLVE_DOTENV_PATH=/path/to/file.env` when you want a non-default
dotenv path.

## Variable Reference

| Variable | Required when | Notes |
| --- | --- | --- |
| `HF_TOKEN` | Accessing gated/private HF models, private HF dataset remotes, or `hf://` artifact paths | Use a read token for downloads; use write permission only when uploading checkpoints/results. |
| `HF_TOKEN_POOL_JSON` | Using `hf://datasets/__namespace__/...` auto-selection | JSON array of entries such as `[{"namespace":"org","token":"hf_..."}]`. |
| `HF_TOKEN_POOL_NAMESPACE` | Selecting one entry from `HF_TOKEN_POOL_JSON` in launchers | Optional. If unset, launchers use the first token in the pool. |
| `WANDB_API_KEY` | Online WandB logging | Optional. If unset, launchers run in offline/disabled logging mode. |
| `WANDB_ENTITY` | Logging to a specific WandB team/entity | Optional. Leave blank for the account default. |
| `OPENAI_API_KEY` | GPT/API judge workflows, LPFQA verifier, LLM math judge | Optional unless those verifier or judge modes are enabled. |
| `GEMINI_API_KEY` | Gemini-backed API solver/judge workflows | Optional unless configured via `api_solver.provider=gemini`. |
| `R2_ENDPOINT_URL` | Using `s3://` or `r2://` paths with the R2 backend | Cloudflare R2 S3-compatible endpoint URL. |
| `R2_ACCESS_KEY_ID` | Using `s3://` or `r2://` paths with the R2 backend | Access key ID. `ACCESS_KEY_ID` and `AWS_ACCESS_KEY_ID` are also accepted by storage utilities. |
| `R2_SECRET_ACCESS_KEY` | Using `s3://` or `r2://` paths with the R2 backend | Secret key. `SECRET_ACCESS_KEY` and `AWS_SECRET_ACCESS_KEY` are also accepted by storage utilities. |
| `R2_REGION` | R2/S3 storage clients | Defaults to `auto` for Cloudflare R2. |
| `PREPROCESSED_DATA_RCLONE_URI` | Launcher-managed data download | Optional rclone source for `.cache/data/preprocessed`, for example `r2:my-bucket/path/to/preprocessed`. Leave unset when data is already local. |

## Workflow Matrix

| Workflow | Entry point | Credentials typically needed |
| --- | --- | --- |
| Local unit tests | `python -m pytest ...` | None. |
| Offline training smoke test | `python -m verl_inf_evolve.main training.remote_sync_path=null wandb.enabled=false` | None, if model and data are public/local. |
| Paper-scale training with remote checkpoint upload | `python -m verl_inf_evolve.main ... training.remote_sync_path=...` | `HF_TOKEN` for `hf://`; R2 env vars for `s3://`/`r2://`; optional `WANDB_API_KEY`. |
| Solver eval | `python -m verl_inf_evolve.sol_eval.sol_eval` | `HF_TOKEN` or R2 env vars when checkpoints/results live on remote storage; optional `WANDB_API_KEY`. |
| Generator eval | `python -m verl_inf_evolve.gen_eval.gen_eval` | Same remote storage credentials as training; `OPENAI_API_KEY` only for API-solver modes. |
| OpenCompass code eval | `scripts/run_opencompass.py` or K8s `opencompass` job type | `HF_TOKEN` for private model/checkpoint repos; optional remote upload token if uploading results. |
| K8s/SLURM launchers | `launcher/k8s/launch.sh`, `launcher/slurm/launch.sh` | Same as the underlying workflow, supplied through `.env`, scheduler secrets, or exported environment variables. |
| Scraper SLURM wrapper | `src/agent/scraper/launch/pdf_to_chunks_slurm.sh` | None unless `JUDGE_BASE_URL` points to a secured API, in which case set `JUDGE_API_KEY`. |

## Remote Path Rules

`training.remote_sync_path`, `eval.remote_sync_path`, `eval.remote_eval_base`,
and `gen_eval.remote_sync_path` can use:

- `null` or empty: local-only run, no remote upload/download.
- `hf://datasets/org/repo/prefix`: Hugging Face dataset backend. Set
  `HF_TOKEN` if the dataset is private or uploads are enabled.
- `hf://datasets/__namespace__/repo/prefix`: namespace auto-selection. Provide
  `HF_TOKEN_POOL_JSON`; the resolver picks a namespace and token.
- `s3://bucket/prefix` or `r2://bucket/prefix`: R2/S3-compatible backend. Set
  `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, and `R2_SECRET_ACCESS_KEY`, or provide
  an existing `[r2]` section in `~/.config/rclone/rclone.conf` for launchers.

For open-source examples, prefer local paths or `hf://datasets/...` placeholders.
Do not hard-code private bucket names, account IDs, or tokens in config files.

## Data Cache

The launchers expect preprocessed data under:

```text
.cache/data/preprocessed/
```

Required files by workflow:

- Training: `documents.json` and
  `curriculum_pool/supergpqa_science_800.json`.
- `sol_eval`: `benchmarks/`.
- `gen_eval`: `eval_documents.json` and
  `curriculum_pool/supergpqa_science_800.json`.

Use one of two modes:

- Local/offline: copy the files into `.cache/data/preprocessed` and launch with
  `--offline-data` or let `LAUNCHER_OFFLINE_DATA=auto` detect the local cache.
- Remote data copy: set `PREPROCESSED_DATA_RCLONE_URI` to an rclone source whose
  contents mirror `.cache/data/preprocessed`, and set the matching R2/S3
  credentials.

## Safe Patterns

Local shell:

```bash
set -a
source .env
set +a
python -m verl_inf_evolve.main training.remote_sync_path=null wandb.enabled=false
```

Kubernetes:

```bash
kubectl create secret generic infuser-secrets \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  --from-literal=WANDB_API_KEY="$WANDB_API_KEY"
```

Then expose those values through your cluster's normal secret-injection
mechanism or export them before running `launcher/k8s/launch.sh`.

SLURM:

```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
./launcher/slurm/launch.sh EXPERIMENT --job-type training
```

## What Not To Do

- Do not add real tokens to YAML configs, shell scripts, notebooks, or tests.
- Do not bake `.env`, rclone configs, `.aws`, `.kube`, or SSH keys into Docker
  images.
- Do not rely on historical private bucket names. Use your own
  `PREPROCESSED_DATA_RCLONE_URI` and remote output paths.
- Rotate any credential that was ever committed to a private history before
  publishing a new clean repository.
