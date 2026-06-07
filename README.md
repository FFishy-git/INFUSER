# INFUSER

INFUSER is a training and evaluation runtime for influence-guided
self-evolution. It co-trains two language-model policies:

- a document-grounded generator that proposes multiple-choice reasoning
  questions from source documents;
- a solver that answers generated questions and is updated with
  verifier-based correctness rewards.

The generator is trained with an influence reward: generated questions are
scored by how well the solver update induced by that question aligns with the
solver's held-out dev-set gradient. This repository contains the runtime used
for INFUSER training and solver benchmark evaluation.

## Repository Contents

```text
verl_inf_evolve/              INFUSER training and evaluation code
verl_inf_evolve/config/       Hydra configs and paper experiment overrides
verl/                         vendored verl runtime used by this project
src/agent/scraper/            document chunking utilities
scripts/patches/opencompass/  OpenCompass patches for code benchmarks
tests/                        unit tests for training, data, and eval helpers
```

Run commands from the repository root so Python can resolve both `verl` and
`verl_inf_evolve`.

## Quickstart

The commands below assume a machine with the model runtime already installed
for multi-GPU inference/training: CUDA, PyTorch, Ray, vLLM, Transformers,
Hugging Face Hub, Hydra/OmegaConf, and the standard `verl` dependencies.

The tested open-source setup starts from the pinned `verlai/verl` base image
used by `launcher/opensource/Dockerfile.runtime`:

```text
verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d
```

That image supplies the CUDA/PyTorch/vLLM/Ray/verl stack. The root
`requirements.txt` installs only INFUSER's additional Python layer and
intentionally does not reinstall `torch`, `vllm`, `ray`, or `verl`.

1. Clone the repository and enter it:

```bash
git clone https://github.com/FFishy-git/INFUSER.git
cd INFUSER
```

2. Make the vendored `verl` package and INFUSER code importable, then install
   the exported INFUSER dependency layer:

```bash
export PYTHONPATH="$PWD:$PWD/verl:${PYTHONPATH:-}"
python -m pip install -r requirements.txt
```

Check that both the base runtime and INFUSER dependency layer are visible:

```bash
python - <<'PY'
import datasets, ray, torch, vllm

print("datasets", datasets.__version__)
print("ray", ray.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("vllm", vllm.__version__)
PY
```

For optional code benchmarks (`humaneval`, `livecodebench`), install the
OpenCompass extras after the base requirements:

```bash
python -m pip install -r launcher/opensource/requirements-opencompass.txt
python -m pip install evalplus==0.3.1 --no-deps
python -m pip install tree-sitter==0.25.2 tree-sitter-python==0.25.0
```

3. Configure optional credentials. Public data/model access can run without
   tokens, but set `HF_TOKEN` for gated/private Hugging Face access and
   `WANDB_API_KEY` if you want online logging.

```bash
cp .env.example .env
# Edit .env as needed.
```

4. Download the released preprocessed data and benchmark files:

```bash
python launcher/preparation/download_data.py \
  --use-preprocessed \
  --hf-repo Siyuc/infuser-data \
  --output-dir .cache/data
```

This creates the local paths expected by the launchers, including
`.cache/data/preprocessed/documents.json`,
`.cache/data/preprocessed/curriculum_pool/supergpqa_science_800.json`, and
`.cache/data/preprocessed/benchmarks/`.

5. Run the validated 8-GPU requirements smoke before launching a paper-scale
   run. This exercises solver rollout, generated-question rollout,
   generated-answer rollout, influence scoring, generator PPO, solver PPO, and
   checkpointing for two answer loops while disabling benchmark eval, remote
   upload, and WandB. A successful run completes `ans_loop=0` and `ans_loop=1`,
   saves `global_step_1`, and exits successfully:

```bash
python -m verl_inf_evolve.main \
  experiment_qwen3_8b_base=FW-Alr_2e-6-Glr_4e-6-DrGRPO-TIS_token-dev_800-precond_cos \
  training.max_ans_loop=2 \
  training.max_gen_loop=2 \
  benchmark_eval.enabled=false \
  training.remote_sync_path=null \
  training.resume_from_remote=false \
  wandb.enabled=false \
  trainer.logger='[console]'
```

6. Reproduce the main Qwen3-8B-Base paper training recipe on an 8-GPU machine:

```bash
python -m verl_inf_evolve.main \
  experiment_qwen3_8b_base=FW-Alr_2e-6-Glr_4e-6-DrGRPO-TIS_token-dev_800-precond_cos \
  training.remote_sync_path=null \
  training.resume_from_remote=false \
  wandb.enabled=false \
  trainer.logger='["console"]'
```

For Qwen3-4B-Base, use:

```bash
python -m verl_inf_evolve.main \
  experiment_qwen3_4b_base=FW-Alr_2e-6-Glr_6e-6-DrGRPO-TIS_token-dev_800-precond_cos \
  training.remote_sync_path=null \
  training.resume_from_remote=false \
  wandb.enabled=false \
  trainer.logger='["console"]'
```

Outputs are written under `training.default_local_dir`, which defaults to an
experiment-specific directory under `.output/`.

## Requirements

The full training pipeline is intended for a multi-GPU environment with
PyTorch, Ray, vLLM, Transformers, Hugging Face Hub, Hydra/OmegaConf, and the
standard `verl` runtime dependencies installed. The paper-scale runs use one
node with `8` H100 GPUs. Smaller smoke tests can be launched by reducing the
number of training iterations, but they still require the model and rollout
runtime dependencies.

Large datasets, model checkpoints, benchmark files, and generated outputs are
not committed to this repository. Place the data files described below at the
same relative paths before launching the paper pipeline.

`requirements.txt` is exported from the same pinned Docker runtime and should
be installed on top of that base image or an equivalent environment that already
provides CUDA/PyTorch/vLLM/Ray/verl. Optional OpenCompass/EvalPlus dependencies
are kept separate under `launcher/opensource/requirements-opencompass.txt`.

## Environment Setup

For local runs, copy the template and fill only the keys your run needs:

```bash
cp .env.example .env
```

The training and solver-eval entrypoints automatically load `.env` from the
current working directory. To use another dotenv file, set
`VERL_INF_EVOLVE_DOTENV_PATH=/path/to/file.env` before launching. For
Kubernetes or other schedulers, provide the same variable names through Secrets
or the scheduler's environment injection mechanism.

Hugging Face setup:

- Set `HF_TOKEN` if you need gated/private model access or private HF dataset
  repos for checkpoints and evaluation artifacts.
- For a single HF account, this is enough:

```bash
HF_TOKEN=hf_your_token
```

For local-only runs with `training.remote_sync_path=null` and public model/data
access, Hugging Face variables can be left blank.

See [CREDENTIALS.md](CREDENTIALS.md) for the complete credential and remote
storage guide, including R2/S3, Hugging Face token pools, WandB, and API judge
keys.

## Data Layout

The GitHub repository is code-only. Download the release data from the public
Hugging Face dataset before launching training or evaluation:

```bash
python launcher/preparation/download_data.py \
  --use-preprocessed \
  --hf-repo Siyuc/infuser-data \
  --output-dir .cache/data
```

This populates `.cache/data/preprocessed/`. For K8s, run the same preparation
step on the repository path mounted by the PVC before using `--offline-data`.

Training expects these files:

```text
.cache/data/preprocessed/documents.json
.cache/data/preprocessed/curriculum_pool/supergpqa_science_800.json
```

- `documents.json`: the 12,260-chunk document pool used by the generator.
- `supergpqa_science_800.json`: the 800-question SuperGPQA science dev set
  used to compute the influence signal.

The release dataset also includes the Putnam/AIME/history variant used by some
configs:

```text
.cache/data/preprocessed/documents_with_putnam_aime_history_math10000.json
.cache/data/preprocessed/curriculum_pool/supergpqa_science_pruned_400_aime_history_400.json
```

Solver benchmark evaluation expects benchmark JSON files under:

```text
.cache/data/preprocessed/benchmarks/
```

The paper evaluation suite uses local JSONs for `math500`, `aime2024`,
`aime2025`, `hmmt`, `olympiadbench`, `mmlu_pro`, `gpqa_diamond`, `supergpqa`,
`bbeh`, `medqa`, and `medxpertqa`.

Coding benchmarks (`humaneval`, `livecodebench`) are evaluated through the
external OpenCompass/EvalPlus path and require those dependencies and datasets
in the runtime environment.

Training and solver evaluation use the same local cache layout:

- `.cache/data/preprocessed/documents.json` and
  `.cache/data/preprocessed/curriculum_pool/supergpqa_science_800.json`
  for training.
- `.cache/data/preprocessed/benchmarks/` for `sol_eval`.

## Launchers

Available launchers are:

- Unified:
  - `launcher/run.sh --backend <k8s|slurm|local>`
  - convenience wrapper for all launch backends listed below
- K8s:
  - `launcher/k8s/launch.sh`
  - supports `--job-type training|training_interactive|sol_eval|opencompass`
  - add `--offline-data` to enforce local cache-only mode
- SLURM:
  - `launcher/slurm/launch.sh`
  - supports `--job-type training|sol_eval`
  - dispatches to:
    - `launcher/slurm/submit.sh` for `training`
    - `launcher/slurm/submit_sol_eval.sh` for `sol_eval`
- Local:
  - `launcher/local/launch.sh`
  - supports `training|sol_eval`
  - runs directly on your machine with the same `REQUIRED_DATA_PATHS` checks

Examples:

```bash
# Local training smoke run (dev-only solver PPO)
./launcher/local/launch.sh \
  FW-Alr_2e-6-DrGRPO-TIS_token-dev_only \
  --job-type training \
  --config-group experiment_qwen3_8b_base \
  --extra-overrides "training.max_ans_loop=2 training.max_gen_loop=2 training.dev_rollout_subsample_size=16 generator.rollout.n=8 solver.rollout.n=8 generator.rollout.response_length=512 solver.rollout.response_length=512 generator.rollout.temperature=0.8 solver.rollout.temperature=0.8 solver.actor.ppo_mini_batch_size=16 solver.actor.ppo_micro_batch_size_per_gpu=1 solver.rollout.log_prob_micro_batch_size_per_gpu=1 benchmark_eval.enabled=false training.remote_sync_path=null training.resume_from_remote=false wandb.enabled=false trainer.logger=[console]"

# K8s training smoke run
./launcher/k8s/launch.sh \
  FW-Alr_2e-6-DrGRPO-TIS_token-dev_only \
  --job-type training \
  --config-group experiment_qwen3_8b_base \
  --gpu 8 \
  --cpu 32 \
  --memory 512 \
  --queue YOUR_QUEUE \
  --pvc YOUR_PVC \
  --offline-data \
  --no-stream \
  --extra-overrides "training.max_ans_loop=2 training.max_gen_loop=2 training.dev_rollout_subsample_size=16 generator.rollout.n=8 solver.rollout.n=8 generator.rollout.response_length=512 solver.rollout.response_length=512 generator.rollout.temperature=0.8 solver.rollout.temperature=0.8 solver.actor.ppo_mini_batch_size=16 solver.actor.ppo_micro_batch_size_per_gpu=1 solver.rollout.log_prob_micro_batch_size_per_gpu=1 benchmark_eval.enabled=false training.remote_sync_path=null training.resume_from_remote=false wandb.enabled=false trainer.logger=[console]"
```

To make the launcher independent of R2 data download, keep the Hugging Face
release data under `.cache/data/preprocessed` and always launch with
`--offline-data` (or set `LAUNCHER_OFFLINE_DATA=auto`, which defaults to
local-first behavior when all required paths are already present). For K8s,
`local` means the cache is present in the repository path mounted by the PVC;
the launcher restores the required cache paths into the pod even when
`.skyignore` excludes the broader `.cache/` directory from the code copy.
To let launchers populate that cache from remote storage, set
`PREPROCESSED_DATA_RCLONE_URI` to an rclone source such as
`r2:my-bucket/.cache/data/preprocessed`.

## Paper Recipe Details

The paper uses these Hydra experiment overrides:

| Model | Override |
| --- | --- |
| Qwen3-4B-Base | `experiment_qwen3_4b_base=FW-Alr_2e-6-Glr_6e-6-DrGRPO-TIS_token-dev_800-precond_cos` |
| Qwen3-8B-Base | `experiment_qwen3_8b_base=FW-Alr_2e-6-Glr_4e-6-DrGRPO-TIS_token-dev_800-precond_cos` |

These overrides set the paper recipe: `training.max_ans_loop=100`,
`training.doc_batch_size=128`, `generator.rollout.n=8`, `solver.rollout.n=8`,
`influence.similarity_mode=preconditioned_cosine`, token-level truncated
importance sampling with threshold `2.0`, AdamW with weight decay `0.01`, solver
learning rate `2e-6`, and generator learning rates `6e-6` for Qwen3-4B-Base and
`4e-6` for Qwen3-8B-Base.

To use remote checkpointing, set `training.remote_sync_path` to an
`hf://datasets/...`, `s3://...`, or `r2://...` URI and provide credentials
through environment variables. Avoid hardcoding account names, tokens, or local
credential paths in committed configs.

## Solver Evaluation

Evaluate trained solver checkpoints with:

```bash
python -m verl_inf_evolve.sol_eval.sol_eval \
  eval.model_path=Qwen/Qwen3-8B-Base \
  eval.remote_sync_path=<TRAINING_OUTPUT_URI> \
  eval.run_name=<RUN_NAME> \
  eval.checkpoints='[95]' \
  eval.benchmarks='[math500,aime2024,aime2025,hmmt,olympiadbench,mmlu_pro,gpqa_diamond,supergpqa,bbeh,medqa,medxpertqa]' \
  eval.no_r2_upload=true \
  eval.no_wandb=true
```

For local HF-format checkpoints, set `eval.checkpoint_cache_dir` or
`eval.model_path` to the local checkpoint/model path and keep result upload
disabled.

## Configuration

Default Hydra configs live in `verl_inf_evolve/config/`:

- `self_evolution.yaml`: INFUSER training
- `sol_eval.yaml`: solver benchmark evaluation

Experiment overrides are grouped under:

- `verl_inf_evolve/config/experiment_qwen3_4b_base/`
- `verl_inf_evolve/config/experiment_qwen3_8b_base/`
- `verl_inf_evolve/config/experiment_olmo_3_7b_instruct_sft/`
- `verl_inf_evolve/config/sol_eval_experiment/`

## Testing

Run the unit tests from the repository root:

```bash
pytest tests
```

Some tests exercise optional integrations and may require the corresponding
runtime dependencies.

## Optional code-benchmark dependencies

Install the optional code-benchmark runtime dependencies before `sol_eval` with
`humaneval`/`livecodebench`:

```bash
python -m pip install -r launcher/opensource/requirements-opencompass.txt
```

This repo avoids `opencompass[vllm]` in the base requirements to prevent
reinstalling the CUDA/vLLM stack, since most environments already provide a
compatible runtime. If you need to install manually and your environment does not
pin vLLM already, use the same versions in that file or run
`python -m pip install opencompass[vllm]` intentionally.

If OpenCompass is missing at runtime, `sol_eval` will try to install it
automatically.

## Scope

This branch is scoped to INFUSER training, solver evaluation, and the document
chunking utilities needed by the released pipeline. Legacy prototypes, local
debug outputs, notebooks, checkpoints, and private service credentials are
intentionally excluded.
