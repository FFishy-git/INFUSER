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
for INFUSER training, solver benchmark evaluation, and generator analysis.

## Repository Contents

```text
verl_inf_evolve/              INFUSER training and evaluation code
verl_inf_evolve/config/       Hydra configs and paper experiment overrides
verl/                         vendored verl runtime used by this project
launcher/opensource/          Docker runtime image definition
src/agent/scraper/            document chunking utilities
scripts/patches/opencompass/  OpenCompass patches for code benchmarks
tests/                        unit tests for training, data, and eval helpers
```

Run commands from the repository root so Python can resolve both `verl` and
`verl_inf_evolve`.

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

## Docker Runtime

`launcher/opensource/Dockerfile.runtime` defines a pinned runtime image for
training and `sol_eval`. It starts from a pinned `verlai/verl` image that owns
the CUDA/PyTorch/vLLM/Ray/verl compatibility stack, then installs only this
project's additional Python dependencies and `rclone`.

Build from the repository root:

```bash
docker build \
  --platform linux/amd64 \
  -f launcher/opensource/Dockerfile.runtime \
  -t self-evolution-explore:runtime \
  .
```

The default build includes OpenCompass/EvalPlus support for `humaneval` and
`livecodebench`. For a smaller image without code-benchmark dependencies:

```bash
docker build \
  --platform linux/amd64 \
  --build-arg INSTALL_OPENCOMPASS=0 \
  -f launcher/opensource/Dockerfile.runtime \
  -t self-evolution-explore:runtime-lite \
  .
```

### Open-source install (non-container)

If you run outside the provided Docker image, install the optional code-benchmark
runtime dependencies before `sol_eval` with `humaneval`/`livecodebench`:

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

The image provides the runtime environment for the Python entrypoints, but it
does not include datasets, model checkpoints, secrets, or a cluster-specific
job launcher. For Kubernetes, use your cluster's launcher or job manifests to
run this image with the data layout and environment variables described below.

## Environment Setup

For local runs, copy the template and fill only the keys your run needs:

```bash
cp .env.example .env
```

The training, solver-eval, and generator-eval entrypoints automatically load
`.env` from the current working directory. To use another dotenv file, set
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

Generator evaluation additionally uses:

```text
.cache/data/preprocessed/eval_documents.json
```

Coding benchmarks (`humaneval`, `livecodebench`) are evaluated through the
external OpenCompass/EvalPlus path and require those dependencies and datasets
in the runtime environment.

Solver and generator benchmarks share the same launcher contract:

- `.cache/data/preprocessed/benchmarks/` directory (for `sol_eval`)
- `.cache/data/preprocessed/documents.json` and
  `.cache/data/preprocessed/curriculum_pool/supergpqa_science_800.json`
  (for training + `gen_eval`)

## Launchers

Available launchers are:

- Unified:
  - `launcher/run.sh --backend <k8s|slurm|local>`
  - convenience wrapper for all launch backends listed below
- K8s:
  - `launcher/k8s/launch.sh` (or `launcher/opensource/launch_k8s.sh` wrapper)
  - supports `--job-type training|training_interactive|sol_eval|gen_eval|opencompass`
  - add `--offline-data` to enforce local cache-only mode
- SLURM:
  - `launcher/slurm/launch.sh`
  - supports `--job-type training|sol_eval|gen_eval`
  - dispatches to:
    - `launcher/slurm/submit.sh` for `training`
    - `launcher/slurm/submit_sol_eval.sh` for `sol_eval`
    - `launcher/slurm/submit_gen_eval.sh` for `gen_eval`
- Local:
  - `launcher/local/launch.sh`
  - supports `training|sol_eval|gen_eval`
  - runs directly on your machine with the same `REQUIRED_DATA_PATHS` checks

Examples:

```bash
# Local training smoke run (dev-only solver PPO)
./launcher/local/launch.sh \
  FW-Alr_2e-6-DrGRPO-TIS_token-dev_only \
  --job-type training \
  --config-group experiment_qwen3_8b_base \
  --extra-overrides "training.max_ans_loop=2 training.max_gen_loop=2 training.dev_rollout_subsample_size=16 generator.rollout.n=8 solver.rollout.n=8 generator.rollout.response_length=512 solver.rollout.response_length=512 generator.rollout.temperature=0.8 solver.rollout.temperature=0.8 solver.actor.ppo_mini_batch_size=16 solver.actor.ppo_micro_batch_size_per_gpu=1 solver.rollout.log_prob_micro_batch_size_per_gpu=1 benchmark_eval.enabled=false training.remote_sync_path=null training.resume_from_remote=false wandb.enabled=false trainer.logger=[console]"

# K8s training smoke run with a pushed runtime image
IMAGE=ghcr.io/YOUR_ORG/self-evolution-explore:runtime-20260503 \
  ./launcher/opensource/launch_k8s.sh \
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

## Quickstart

The main training entrypoint is:

```bash
python -m verl_inf_evolve.main
```

For a bounded two-answer-loop smoke test, use the dev-only solver PPO path and
disable in-training benchmark evaluation. This still runs model startup,
solver rollout, answer scoring, solver PPO update, and checkpointing for two
answer loops, while skipping synthetic question generation. The full
self-evolution generator loop is documented in the paper-recipe section below;
it is not a good first-time smoke because an untrained base generator can
produce questions that fail parser validation.

`training.max_gen_loop` still matters because the trainer derives
`num_gen_per_ans = max_gen_loop // max_ans_loop`.

```bash
python -m verl_inf_evolve.main \
  experiment_qwen3_8b_base=FW-Alr_2e-6-DrGRPO-TIS_token-dev_only \
  training.max_ans_loop=2 \
  training.max_gen_loop=2 \
  training.dev_rollout_subsample_size=16 \
  generator.rollout.n=8 \
  solver.rollout.n=8 \
  generator.rollout.response_length=512 \
  solver.rollout.response_length=512 \
  generator.rollout.temperature=0.8 \
  solver.rollout.temperature=0.8 \
  solver.actor.ppo_mini_batch_size=16 \
  solver.actor.ppo_micro_batch_size_per_gpu=1 \
  solver.rollout.log_prob_micro_batch_size_per_gpu=1 \
  benchmark_eval.enabled=false \
  training.remote_sync_path=null \
  training.resume_from_remote=false \
  wandb.enabled=false \
  'trainer.logger=[console]'
```

Outputs are written under `training.default_local_dir`, which defaults to an
experiment-specific directory under `.output/`.

## Reproduce Paper Training

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

Run Qwen3-8B-Base locally without remote checkpoint upload or WandB:

```bash
python -m verl_inf_evolve.main \
  experiment_qwen3_8b_base=FW-Alr_2e-6-Glr_4e-6-DrGRPO-TIS_token-dev_800-precond_cos \
  training.remote_sync_path=null \
  training.resume_from_remote=false \
  wandb.enabled=false \
  trainer.logger='["console"]'
```

Run Qwen3-4B-Base:

```bash
python -m verl_inf_evolve.main \
  experiment_qwen3_4b_base=FW-Alr_2e-6-Glr_6e-6-DrGRPO-TIS_token-dev_800-precond_cos \
  training.remote_sync_path=null \
  training.resume_from_remote=false \
  wandb.enabled=false \
  trainer.logger='["console"]'
```

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

## Generator Evaluation

Run generator evaluation on selected training iterations:

```bash
python -m verl_inf_evolve.gen_eval.gen_eval \
  gen_eval.remote_sync_path=<TRAINING_OUTPUT_URI> \
  gen_eval.ans_loop_indices='[0,30,60,90]' \
  gen_eval.doc_path=.cache/data/preprocessed/eval_documents.json \
  gen_eval.dev_dataset_path=.cache/data/preprocessed/curriculum_pool/supergpqa_science_800.json
```

`gen_eval.mode` supports regeneration from generator checkpoints and replay from
saved generator outputs. See `verl_inf_evolve/config/gen_eval.yaml` for the full
set of options.

## Configuration

Default Hydra configs live in `verl_inf_evolve/config/`:

- `self_evolution.yaml`: INFUSER training
- `sol_eval.yaml`: solver benchmark evaluation
- `gen_eval.yaml`: generator evaluation

Experiment overrides are grouped under:

- `verl_inf_evolve/config/experiment_qwen3_4b_base/`
- `verl_inf_evolve/config/experiment_qwen3_8b_base/`
- `verl_inf_evolve/config/experiment_olmo_3_7b_instruct_sft/`
- `verl_inf_evolve/config/sol_eval_experiment/`
- `verl_inf_evolve/config/gen_eval_experiment/`

## Testing

Run the unit tests from the repository root:

```bash
pytest tests
```

Some tests exercise optional integrations and may require the corresponding
runtime dependencies.

## Scope

This branch is scoped to INFUSER training, solver evaluation, generator
evaluation, and the document chunking utilities needed by the released
pipeline. Legacy prototypes, local debug outputs, notebooks, checkpoints, and
private service credentials are intentionally excluded.
