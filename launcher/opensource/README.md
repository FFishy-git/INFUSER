# Open-Source Docker Launcher Setup

This folder contains the Docker-based runtime setup intended for open-source
users. It is additive: the existing internal K8s launcher remains the thing
that creates pods, while this folder defines the image that launcher should
run.

## What This Pins

`Dockerfile.runtime` pins the current internal base image by immutable digest:

```text
verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d
```

That base image owns the fragile CUDA/PyTorch/vLLM/Ray/verl compatibility set.
The Dockerfile then installs only this project's extra runtime dependencies.
It intentionally does not reinstall `torch`, `vllm`, `ray`, or `verl`.

The Dockerfile also pins the rclone installer version instead of using the
floating install script.

## Build

Build from the repository root:

```bash
IMAGE=self-evolution-explore:runtime \
  ./launcher/opensource/build_image.sh
```

For K8s, the image must be pushed to a registry visible to the cluster:

```bash
IMAGE=ghcr.io/YOUR_ORG/self-evolution-explore:runtime-20260503 \
PUSH=1 \
  ./launcher/opensource/build_image.sh
```

After pushing, record the pushed digest:

```bash
docker buildx imagetools inspect ghcr.io/YOUR_ORG/self-evolution-explore:runtime-20260503
```

Use that digest in papers, release notes, and reproducibility docs.

## Optional Smaller Image

The default `sol_eval` config includes code benchmarks, so the image installs
OpenCompass/evalplus by default. If your release disables `humaneval` and
`livecodebench`, build without those dependencies:

```bash
INSTALL_OPENCOMPASS=0 IMAGE=self-evolution-explore:runtime-lite \
  ./launcher/opensource/build_image.sh
```

## Smoke Test

On a GPU machine:

```bash
docker run --rm --gpus all self-evolution-explore:runtime \
  python -c "import torch, vllm, verl; print(torch.cuda.is_available())"
```

The build also writes a package snapshot inside the image:

```text
/opt/self-evolution-explore/pip-freeze.txt
```

## Launch On K8s

Use the wrapper here, or call `launcher/k8s/launch.sh` directly with
`--image` and `--skip-runtime-installs`.

The wrapper injects those flags by default so the pod uses the baked image
environment instead of running the pod-startup install block. Runtime R2/S3
configuration is always supplied through environment variables or an existing
rclone config; no credentials are embedded in this repository.

Training smoke:

This smoke uses the dev-only solver PPO path. It validates Kubernetes
scheduling, repo/cache setup, model startup, solver rollout/scoring, solver PPO
update, and checkpointing for two answer loops without depending on synthetic
question generation from an untrained base generator.

```bash
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

Solver eval:

```bash
IMAGE=ghcr.io/YOUR_ORG/self-evolution-explore:runtime-20260503 \
  ./launcher/opensource/launch_k8s.sh SOL_EVAL_CONFIG \
    --job-type sol_eval \
    --gpu 8 \
    --cpu 120 \
    --memory 512 \
    --queue YOUR_QUEUE \
    --pvc YOUR_PVC \
    --no-stream
```

## Credentials

Do not bake credentials into the image. The Docker build is only for code,
system packages, and Python packages. Runtime credentials must be provided
when the container runs, not when it is built.

The Dockerfile uses `COPY .`, so `Dockerfile.runtime.dockerignore` excludes
local secret files such as `.env`, `.aws`, `.kube`, rclone configs, SSH keys,
and non-runtime private docs from the build context. Do not pass secrets as
Docker build args, and do not add a later `COPY .env ...` step: even deleting a
secret in a later layer still leaves it in image history.

For local runs, pass credentials as environment variables or a local `.env`
file consumed by the launcher:

```bash
cp .env.example .env
```

Fill only the values your run needs. `WANDB_API_KEY` is optional; runs fall
back to offline/disabled logging when it is unset. `HF_TOKEN` is needed for
gated models or private HF artifact repos. R2/S3 credentials are needed only
when configs use `s3://` or `r2://` remote paths.

For Kubernetes, prefer Kubernetes Secrets or your cluster's existing secret
injection mechanism. Registry credentials, when needed for private images, are
separate `imagePullSecrets`; they let K8s pull the image but are not part of
the image itself.

See [../../CREDENTIALS.md](../../CREDENTIALS.md) for the full variable matrix
and examples for Hugging Face, R2/S3, WandB, and API judge credentials.

## Release Checklist

- Build from this Dockerfile.
- Run a short training smoke test.
- Run one non-code `sol_eval` benchmark.
- If code benchmarks are in the default config, run one HumanEval or
  LiveCodeBench smoke test.
- Push the image and publish the immutable digest.
- Publish the data source and data preparation command:
  `python launcher/preparation/download_data.py --use-preprocessed --hf-repo Siyuc/infuser-data --output-dir .cache/data`.
