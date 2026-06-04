"""Reusable OpenCompass launcher helpers.

This module contains the logic previously embedded in
``scripts/run_opencompass.py`` so other pipelines can reuse the same
benchmark mapping, model detection, checkpoint download, and launch path.
"""

from __future__ import annotations

import glob as globmod
import importlib.util
import os
import subprocess
import sys
from typing import Sequence

# Tokens for base models. GPQA uses PPL mode.
BASE_TOKENS: dict[str, str] = {
    "supergpqa": "supergpqa_gen",
    "aime2024": "aime2024_gen",
    "aime2025": "aime2025_cascade_eval_gen_5e9f4f",
    "gpqa": "gpqa_few_shot_ppl_4b5a83",
    "bbeh": "bbeh_gen",
    "mmlu_pro": "mmlu_pro_gen_cdbebf",
    "math500": "math_500_gen",
    "math500_cascade": "math_500_cascade_eval_gen_6ff468",
    "olympiadbench": "OlympiadBench_0shot_gen_be8b13",
    "phybench": "phybench_gen",
    "medqa": "MedQA_gen_3bf756",
    "humaneval": "humaneval_plus_gen_8e312c",
    "livecodebench": "livecodebench_gen",
    "medxpertqa": "MedXpertQA_gen",
    # hmmt2026 is not available in OpenCompass 0.5.2.
}

# VLLMwithChatTemplate does not support PPL.
CHAT_TOKEN_OVERRIDES: dict[str, str] = {
    "gpqa": "gpqa_gen",
}

BENCHMARK_N: dict[str, int] = {
    "supergpqa": 1,
    "aime2024": 32,
    "aime2025": 32,
    "gpqa": 5,
    "bbeh": 1,
    "mmlu_pro": 1,
    "math500": 2,
    "math500_cascade": 1,
    "olympiadbench": 1,
    "phybench": 10,
    "medqa": 1,
    "humaneval": 8,
    "livecodebench": 1,
    "medxpertqa": 1,
}

BENCHMARK_MAX_OUT_LEN: dict[str, int] = {
    # Empty by default — all benchmarks use the model-level max_out_len.
    # Entries here are merged (lower priority) with caller-supplied overrides.
}

BENCHMARK_GROUPS: dict[str, list[str]] = {
    "all": list(BASE_TOKENS.keys()),
    "math": ["aime2024", "aime2025", "math500", "olympiadbench"],
    "code": ["humaneval", "livecodebench"],
    "science": ["gpqa", "phybench", "medqa", "medxpertqa"],
    "reasoning": ["supergpqa", "bbeh", "mmlu_pro"],
}


def pass_at_k_ladder(n: int) -> list[int]:
    """Return the powers-of-two pass@k ladder supported by ``n`` samples."""
    values: list[int] = []
    k = 1
    while k <= max(int(n), 0):
        values.append(k)
        k *= 2
    return values or [1]


MODEL_REGISTRY: dict[str, str] = {
    "Qwen/Qwen3-4B-Base": "base",
    "Qwen/Qwen3-8B-Base": "base",
    "Qwen/Qwen3-8B": "chat",
    "meta-llama/Llama-3.1-8B": "base",
    "Qwen/Qwen3-4B-Instruct-2507": "chat",
    "Qwen/Qwen3-8B-Instruct": "chat",
    "Qwen/Qwen3-4B": "chat",
    "meta-llama/Llama-3.1-8B-Instruct": "chat",
}

CHAT_MARKERS: tuple[str, ...] = ("instruct", "-chat", "_chat", "-ins-", "_ins_")


def ensure_hf_token() -> None:
    """Extract ``HF_TOKEN`` from the configured pool when needed."""
    if os.environ.get("HF_TOKEN"):
        return
    pool_json = os.environ.get("HF_TOKEN_POOL_JSON", "")
    if not pool_json:
        return
    import json

    try:
        pool = json.loads(pool_json)
        preferred_namespace = os.environ.get("HF_TOKEN_POOL_NAMESPACE")
        for token_info in pool:
            if (
                preferred_namespace
                and isinstance(token_info, dict)
                and token_info.get("namespace") == preferred_namespace
            ):
                os.environ["HF_TOKEN"] = token_info["token"]
                print(
                    f"Extracted HF_TOKEN from pool (namespace: {preferred_namespace})",
                    flush=True,
                )
                return
        token_info = pool[0]
        token = token_info["token"] if isinstance(token_info, dict) else token_info
        os.environ["HF_TOKEN"] = token
        print("Extracted HF_TOKEN from pool (first entry)", flush=True)
    except (json.JSONDecodeError, IndexError, KeyError):
        return


def ensure_opencompass_installed() -> None:
    """Install OpenCompass lazily when not present."""
    if importlib.util.find_spec("opencompass") is not None:
        return
    print("Installing opencompass...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "opencompass[vllm]"],
        check=True,
    )
    # opencompass depends on tree-sitter-languages which pins tree-sitter<0.22,
    # but evalplus (used by HumanEval+ evaluator) requires tree-sitter>=0.22.
    # Re-upgrade tree-sitter so evalplus.sanitize works correctly.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "tree_sitter>=0.22.0"],
        check=False,
    )


def patch_opencompass_site_packages() -> None:
    """Copy custom patches into OC's site-packages so subprocess configs can resolve them.

    OC's eval subprocess ignores custom_imports, so patched modules must live
    inside the installed ``opencompass.datasets`` package directory.
    """
    import shutil

    try:
        import opencompass
        oc_datasets = os.path.join(os.path.dirname(opencompass.__file__), "datasets")
    except ImportError:
        return

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    patches_dir = os.path.join(repo_root, "scripts", "patches", "opencompass")
    if not os.path.isdir(patches_dir):
        return

    copies = [
        ("lcb_dataset.py", "lcb_dataset.py"),
        ("humaneval_plus_evaluator.py", "humaneval_plus_evaluator.py"),
        ("postprocessors.py", "custom_postprocessors.py"),
        ("MedXpertQA.py", "MedXpertQA.py"),
    ]
    for src_name, dst_name in copies:
        src = os.path.join(patches_dir, src_name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(oc_datasets, dst_name))

    # Copy lcb_few_shot directory
    lcb_few_shot_src = os.path.join(patches_dir, "lcb_few_shot")
    if os.path.isdir(lcb_few_shot_src):
        lcb_few_shot_dst = os.path.join(oc_datasets, "lcb_few_shot")
        if os.path.isdir(lcb_few_shot_dst):
            shutil.rmtree(lcb_few_shot_dst)
        shutil.copytree(lcb_few_shot_src, lcb_few_shot_dst)

    # Patch base evaluator to skip length check when n>1 (pass@k with multiple samples)
    base_eval_path = os.path.join(
        os.path.dirname(oc_datasets), "openicl", "icl_evaluator", "icl_base_evaluator.py"
    )
    if os.path.isfile(base_eval_path):
        with open(base_eval_path, "r") as f:
            content = f.read()
        old = (
            "and score_kwargs['references'] is not None):\n"
            "            if len(score_kwargs['predictions']) != len(\n"
            "                    score_kwargs['references']):"
        )
        new = (
            "and score_kwargs['references'] is not None and n <= 1):\n"
            "            if len(score_kwargs['predictions']) != len(\n"
            "                    score_kwargs['references']):"
        )
        if old in content:
            content = content.replace(old, new)
            with open(base_eval_path, "w") as f:
                f.write(content)
            # Clear pyc cache
            cache_dir = os.path.join(os.path.dirname(base_eval_path), "__pycache__")
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)

    print("Patched OC site-packages with custom modules", flush=True)

    # Ensure evalplus is available (required by HumanEvalPlusEvaluatorAZR).
    # Must use the open-compass/human-eval version (same as launch_opencompass.sh)
    # because the evaluator depends on evalplus.evaluate() and evalplus.sanitize().
    _ensure_oc_evalplus(repo_root)


def _evalplus_is_compatible() -> bool:
    """Check if evalplus is importable and has the ``dataset`` kwarg in evaluate().

    Uses a subprocess to avoid stale import caches in the current process —
    pip installs during the same process are not reliably visible to
    ``importlib`` even after cache invalidation.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "from evalplus.evaluate import evaluate; "
             "import inspect; "
             "sig = inspect.signature(evaluate); "
             "print('dataset' in sig.parameters)"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and result.stdout.strip() == "True"
    except Exception:
        return False


def _pip_install(args: list[str], label: str) -> bool:
    """Run pip install and report success/failure visibly."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + args,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  FAILED ({label}): {result.stderr[-300:]}", flush=True)
        return False
    print(f"  OK ({label})", flush=True)
    return True


def _ensure_oc_evalplus(repo_root: str) -> None:
    """Install evalplus from open-compass/human-eval, matching launch_opencompass.sh.

    Uses non-editable installs so packages persist in site-packages for the
    OC eval subprocess.  Checks return codes and verifies the ``evaluate()``
    function signature after each attempt.
    """
    import tempfile

    if _evalplus_is_compatible():
        print("evalplus already installed and compatible.", flush=True)
        return

    # Install evalplus from PyPI.  The open-compass/human-eval submodule ships
    # an older evalplus with ``def evaluate(flags)`` (argparse-style), which is
    # incompatible with our HumanEvalPlusEvaluatorAZR that calls
    # ``evaluate(dataset=..., samples=...)``.  PyPI evalplus >= 0.2.0 has the
    # correct signature.  Use --force-reinstall to ensure the PyPI version wins
    # even if the OC submodule's version was installed earlier.
    print("Installing evalplus from PyPI...", flush=True)
    _pip_install(
        ["evalplus", "--no-cache-dir", "--force-reinstall", "--no-deps"],
        "evalplus-from-pypi",
    )
    if _evalplus_is_compatible():
        print("evalplus compatible.", flush=True)
        _ensure_evalplus_sanitize(repo_root)
        return

    # Fallback: try AZR vendored copy which also has compatible evaluate().
    azr_evalplus = os.path.join(
        repo_root, "baselines", "absolute-zero-reasoner",
        "evaluation", "code_eval", "coding", "evalplus",
    )
    if os.path.isdir(azr_evalplus):
        print("Trying AZR vendored evalplus...", flush=True)
        _pip_install(
            [azr_evalplus, "--no-cache-dir", "--force-reinstall"],
            "evalplus-from-azr",
        )
        if _evalplus_is_compatible():
            print("evalplus compatible (AZR).", flush=True)
            return

    print(
        "WARNING: evalplus installation failed — humaneval benchmark will crash. "
        "Check network access and pip output above.",
        flush=True,
    )


def _ensure_evalplus_sanitize(repo_root: str) -> None:
    """Ensure evalplus.sanitize is available (needed for tree-sitter code cleaning)."""
    try:
        from evalplus.sanitize import sanitize  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        azr_evalplus = os.path.join(
            repo_root, "baselines", "absolute-zero-reasoner",
            "evaluation", "code_eval", "coding", "evalplus",
        )
        if os.path.isdir(azr_evalplus):
            print("Installing AZR evalplus for sanitize module...", flush=True)
            # Only install if it won't break the existing compatible evalplus.
            # Use --no-deps to avoid overwriting the evaluate module.
            _pip_install(
                [azr_evalplus, "--no-deps", "--no-cache-dir"],
                "evalplus-sanitize-only",
            )


def get_visible_gpus() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def detect_hf_type(hf_path: str) -> str:
    """Auto-detect whether a model should be treated as ``base`` or ``chat``.

    Resolution order:
    1. Exact match in ``MODEL_REGISTRY``.
    2. Local directory introspection — if ``hf_path`` is a directory, treat
       it as ``"chat"`` when it ships a non-trivial chat template (either a
       ``chat_template.jinja`` file or a non-empty ``chat_template`` field in
       ``tokenizer_config.json``). This catches merged checkpoints whose path
       has no chat marker (e.g. R-Zero solver HF dirs).
    3. Case-insensitive substring match against ``CHAT_MARKERS`` in the path.
    4. Default ``"base"``.
    """
    if hf_path in MODEL_REGISTRY:
        return MODEL_REGISTRY[hf_path]
    try:
        if os.path.isdir(hf_path):
            jinja = os.path.join(hf_path, "chat_template.jinja")
            if os.path.isfile(jinja) and os.path.getsize(jinja) > 0:
                return "chat"
            tok_cfg = os.path.join(hf_path, "tokenizer_config.json")
            if os.path.isfile(tok_cfg):
                import json

                with open(tok_cfg, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                tpl = cfg.get("chat_template")
                if isinstance(tpl, str) and tpl.strip():
                    return "chat"
                if isinstance(tpl, list) and tpl:
                    return "chat"
    except Exception:
        pass
    path_lower = hf_path.lower()
    for marker in CHAT_MARKERS:
        if marker in path_lower:
            return "chat"
    return "base"


def resolve_benchmark_names(benchmarks: str) -> list[str]:
    """Resolve a shorthand group or comma-separated list into benchmark names."""
    if benchmarks in BENCHMARK_GROUPS:
        return BENCHMARK_GROUPS[benchmarks]
    return [bench.strip() for bench in benchmarks.split(",") if bench.strip()]


def resolve_tokens(benchmarks: str, hf_type: str) -> list[str]:
    """Resolve benchmark names to OpenCompass dataset tokens."""
    bench_names = resolve_benchmark_names(benchmarks)
    tokens = []
    for name in bench_names:
        if name not in BASE_TOKENS:
            tokens.append(name)
            continue
        if hf_type == "chat" and name in CHAT_TOKEN_OVERRIDES:
            tokens.append(CHAT_TOKEN_OVERRIDES[name])
        else:
            tokens.append(BASE_TOKENS[name])
    return tokens


def download_hf_checkpoint(
    hf_repo: str,
    ckpt_prefix: str,
    ckpt_step: int,
    role: str = "solver",
    cache_dir: str | None = None,
) -> str:
    """Download a checkpoint from an HF dataset repo and return the model dir."""
    from huggingface_hub import snapshot_download

    subdir = f"{ckpt_prefix}/global_step_{ckpt_step}/{role}/huggingface"
    print("Downloading checkpoint from HF dataset repo:", flush=True)
    print(f"  Repo:   {hf_repo}", flush=True)
    print(f"  Path:   {subdir}", flush=True)

    snapshot_root = snapshot_download(
        hf_repo,
        repo_type="dataset",
        allow_patterns=f"{subdir}/*",
        cache_dir=cache_dir,
    )

    local_model_dir = os.path.join(snapshot_root, subdir)
    if not os.path.isfile(os.path.join(local_model_dir, "config.json")):
        raise FileNotFoundError(
            f"config.json not found at {local_model_dir}. "
            f"Check --hf-repo, --ckpt-prefix, and --ckpt-step values."
        )

    size_mb = sum(
        os.path.getsize(os.path.join(local_model_dir, filename))
        for filename in os.listdir(local_model_dir)
        if os.path.isfile(os.path.join(local_model_dir, filename))
    ) / (1024 * 1024)
    n_files = len(os.listdir(local_model_dir))
    print(f"  Downloaded: {n_files} files, {size_mb:.0f} MB -> {local_model_dir}", flush=True)
    return local_model_dir


def extract_flag(
    argv: list[str],
    flag: str,
    default: str | None = None,
) -> tuple[str | None, list[str]]:
    """Extract a flag value from ``argv``, returning ``(value, remaining_argv)``."""
    remaining = []
    value = default
    idx = 0
    while idx < len(argv):
        if argv[idx] == flag and idx + 1 < len(argv):
            value = argv[idx + 1]
            idx += 2
        elif argv[idx].startswith(f"{flag}="):
            value = argv[idx].split("=", 1)[1]
            idx += 1
        else:
            remaining.append(argv[idx])
            idx += 1
    return value, remaining


def build_model_id(
    hf_path: str,
    hf_repo: str | None,
    ckpt_prefix: str | None,
    ckpt_step: str | None,
) -> str:
    """Build a path-safe model identifier for organizing uploaded results."""
    if hf_repo and ckpt_prefix and ckpt_step:
        return f"{ckpt_prefix}/global_step_{ckpt_step}"
    return hf_path.replace("/", "__")


def upload_results_to_hf(
    upload_repo: str,
    model_id: str,
    output_dir: str | None,
) -> None:
    """Upload OpenCompass outputs to an HF dataset repo."""
    from huggingface_hub import HfApi

    run_dir = find_latest_run_dir(output_dir)
    if run_dir is None:
        return

    run_timestamp = os.path.basename(run_dir)
    hf_prefix = f"OC_eval/{model_id}/{run_timestamp}"
    print(f"Uploading results to hf://datasets/{upload_repo}/{hf_prefix}/", flush=True)
    print(f"  Local dir: {run_dir}", flush=True)

    api = HfApi()
    upload_kwargs = dict(
        repo_id=upload_repo,
        repo_type="dataset",
        folder_path=run_dir,
        path_in_repo=hf_prefix,
        commit_message=f"OC eval results: {model_id} ({run_timestamp})",
    )
    try:
        api.upload_folder(**upload_kwargs)
    except Exception:
        # Some HF repos require PRs; retry with create_pr=True
        print("Direct upload failed, retrying with create_pr=True", flush=True)
        api.upload_folder(**upload_kwargs, create_pr=True)
    print(f"Upload complete: hf://datasets/{upload_repo}/{hf_prefix}/", flush=True)


def find_latest_run_dir(output_dir: str | None) -> str | None:
    """Return the most recent OpenCompass run dir under ``output_dir``."""
    search_root = output_dir or "outputs"
    if not os.path.isdir(search_root):
        print(f"WARNING: Output directory '{search_root}' not found", flush=True)
        return None

    summary_files = sorted(
        globmod.glob(f"{search_root}/**/summary/summary_*.csv", recursive=True)
    )
    if not summary_files:
        print(f"WARNING: No summary CSV found under '{search_root}'", flush=True)
        return None

    latest_summary = summary_files[-1]
    return os.path.dirname(os.path.dirname(latest_summary))


def launch_opencompass(
    *,
    hf_path: str,
    hf_type: str,
    benchmarks: str,
    num_runs: str | None = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = -1,
    num_gpus: str | None = None,
    max_out_len: str = "8192",
    prompt_length: int = 4096,
    benchmark_max_out_len: dict[str, int] | None = None,
    max_questions: int | None = None,
    output_dir: str | None = None,
    passthrough: Sequence[str] = (),
    upload_repo: str | None = None,
    no_upload: bool = False,
    hf_repo: str | None = None,
    ckpt_prefix: str | None = None,
    ckpt_step: str | None = None,
    custom_chat_template: str | None = None,
) -> int:
    """Launch OpenCompass with the repo's benchmark defaults and patches."""
    bench_names = resolve_benchmark_names(benchmarks)
    tokens = resolve_tokens(benchmarks, hf_type)

    visible_gpus = get_visible_gpus()
    if not num_gpus:
        num_gpus = str(max(visible_gpus, 1))
    num_gpus_int = int(num_gpus)
    print(
        f"GPUs: {visible_gpus} visible, using {num_gpus_int} workers "
        f"(DP={num_gpus_int}, TP=1)",
        flush=True,
    )

    ensure_opencompass_installed()
    patch_opencompass_site_packages()

    print(f"Benchmarks ({len(tokens)}): {' '.join(tokens)}", flush=True)
    bench_n_values: dict[str, int] = {
        name: BENCHMARK_N[name] for name in bench_names if name in BENCHMARK_N
    }

    if num_runs:
        override_n = int(num_runs)
        bench_n_values = {name: override_n for name in bench_names}
        print(f"Overriding all n values to {override_n} (--dataset-num-runs)", flush=True)
    else:
        n_summary = {key: value for key, value in bench_n_values.items() if value > 1}
        if n_summary:
            print(f"Per-benchmark n: {n_summary}", flush=True)

    # Merge per-benchmark max_out_len: module defaults < caller overrides
    bench_max_out_len: dict[str, int] = dict(BENCHMARK_MAX_OUT_LEN)
    if benchmark_max_out_len:
        bench_max_out_len.update(benchmark_max_out_len)
    # Only keep entries for benchmarks in this run
    bench_max_out_len = {
        name: length for name, length in bench_max_out_len.items()
        if name in bench_names
    }
    if bench_max_out_len:
        print(f"Per-benchmark max_out_len: {bench_max_out_len}", flush=True)

    needs_python_config = any(value > 1 for value in bench_n_values.values())
    has_varying_n = len(set(bench_n_values.values())) > 1
    needs_overrides = "livecodebench" in bench_names or "humaneval" in bench_names
    needs_per_bench_max_out_len = bool(bench_max_out_len)
    needs_python_config_for_chat_gpqa = (
        hf_type == "chat"
        and "gpqa" in bench_names
        and bench_n_values.get("gpqa", 1) > 1
    )

    if (
        (needs_python_config and has_varying_n)
        or needs_overrides
        or needs_per_bench_max_out_len
        or max_questions
        or needs_python_config_for_chat_gpqa
    ):
        return launch_with_python_config(
            hf_path=hf_path,
            hf_type=hf_type,
            tokens=tokens,
            bench_names=bench_names,
            bench_n_values=bench_n_values,
            bench_max_out_len=bench_max_out_len,
            max_questions=max_questions,
            prompt_length=prompt_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_gpus_int=num_gpus_int,
            max_out_len=max_out_len,
            output_dir=output_dir,
            passthrough=list(passthrough),
            upload_repo=upload_repo,
            no_upload=no_upload,
            hf_repo=hf_repo,
            ckpt_prefix=ckpt_prefix,
            ckpt_step=ckpt_step,
            custom_chat_template=custom_chat_template,
        )

    cmd = [
        "opencompass",
        "--hf-type",
        hf_type,
        "--hf-path",
        hf_path,
        "-a",
        "vllm",
        "--hf-num-gpus",
        "1",
        "--max-num-workers",
        num_gpus,
        "--max-out-len",
        max_out_len,
        "--datasets",
        *tokens,
    ]
    if num_runs:
        cmd.extend(["--dataset-num-runs", num_runs])
    elif needs_python_config:
        uniform_n = next(iter(bench_n_values.values()))
        cmd.extend(["--dataset-num-runs", str(uniform_n)])
    if output_dir:
        cmd.extend(["-w", output_dir])
    cmd.extend(passthrough)

    print(f"Launching: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode == 0 and not no_upload and upload_repo:
        model_id = build_model_id(hf_path, hf_repo, ckpt_prefix, ckpt_step)
        try:
            upload_results_to_hf(upload_repo, model_id, output_dir)
        except Exception as exc:
            print(f"WARNING: HF upload failed: {exc}", flush=True)
    return int(result.returncode)


def launch_with_python_config(
    *,
    hf_path: str,
    hf_type: str,
    tokens: list[str],
    bench_names: list[str],
    bench_n_values: dict[str, int],
    bench_max_out_len: dict[str, int] | None = None,
    max_questions: int | None = None,
    prompt_length: int = 4096,
    temperature: float,
    top_p: float,
    top_k: int,
    num_gpus_int: int,
    max_out_len: str,
    output_dir: str | None,
    passthrough: list[str],
    upload_repo: str | None,
    no_upload: bool,
    hf_repo: str | None,
    ckpt_prefix: str | None,
    ckpt_step: str | None,
    custom_chat_template: str | None = None,
) -> int:
    """Generate a Python config with per-benchmark overrides and launch it."""
    _bench_max_out_len = bench_max_out_len or {}
    bench_token_n = []
    for name, token in zip(bench_names, tokens):
        bench_token_n.append((name, token, bench_n_values.get(name, 1)))

    token_import_map: dict[str, tuple[str, str]] = {
        "supergpqa_gen": ("supergpqa", "supergpqa_datasets"),
        "aime2024_gen": ("aime2024", "aime2024_datasets"),
        "aime2025_cascade_eval_gen_5e9f4f": ("aime2025", "aime2025_datasets"),
        "gpqa_few_shot_ppl_4b5a83": ("gpqa", "gpqa_datasets"),
        "gpqa_gen": ("gpqa", "gpqa_datasets"),
        "bbeh_gen": ("bbeh", "bbeh_datasets"),
        "mmlu_pro_gen_cdbebf": ("mmlu_pro", "mmlu_pro_datasets"),
        "math_500_gen": ("math", "math_datasets"),
        "math_500_cascade_eval_gen_6ff468": ("math", "math_datasets"),
        "OlympiadBench_0shot_gen_be8b13": ("OlympiadBench", "olympiadbench_datasets"),
        "phybench_gen": ("PHYBench", "phybench_datasets"),
        "MedQA_gen_3bf756": ("MedQA", "MedQA_datasets"),
        "humaneval_plus_gen_8e312c": ("humaneval_plus", "humaneval_plus_datasets"),
        "livecodebench_gen": ("livecodebench", "LCB_datasets"),
        "MedXpertQA_gen": ("MedXpertQA", "medxpertqa_datasets"),
        "hmmt2026_cascade_eval_gen_6ff468": ("hmmt2026", "hmmt2026_datasets"),
    }

    import_lines = []
    collect_lines = []
    for name, token, n_value in bench_token_n:
        if token not in token_import_map:
            print(f"WARNING: No import mapping for token '{token}', skipping", flush=True)
            continue
        dirname, export_var = token_import_map[token]
        import_lines.append(
            f"from opencompass.configs.datasets.{dirname}.{token} "
            f"import {export_var} as {name}_ds"
        )
        override_lines = []
        if max_questions and name != "humaneval":
            # Skip test_range for HumanEval: evalplus asserts completions
            # cover ALL problems, so partial datasets break its evaluator.
            override_lines.append(
                f"    _ds.setdefault('reader_cfg', {{}})['test_range'] = '[:{max_questions}]'"
            )
        if n_value > 1:
            if name == "livecodebench":
                override_lines.append(
                    f"    if 'test_output' not in _ds.get('abbr', ''): _ds['n'] = {n_value}"
                )
            else:
                override_lines.append(f"    _ds['n'] = {n_value}")
        if name in _bench_max_out_len:
            mol = _bench_max_out_len[name]
            override_lines.append(
                f"    _ds.setdefault('infer_cfg', {{}}).setdefault('inferencer', {{}})['max_out_len'] = {mol}"
            )
        if name == "supergpqa":
            override_lines.append("    _ds['prompt_mode'] = 'five-shot'")
        if name == "livecodebench":
            k_list = pass_at_k_ladder(n_value)
            override_lines.extend(
                [
                    "    if _ds.get('abbr', '') == 'lcb_code_generation': "
                    "_ds['type'] = 'opencompass.datasets.lcb_dataset.LCBCodeGenerationDatasetOfficial'",
                    "    if _ds.get('abbr', '') == 'lcb_code_generation': "
                    "_ds['release_version'] = 'release_v5'",
                    "    if _ds.get('abbr', '') == 'lcb_code_generation': "
                    f"_ds['model_type'] = '{hf_type}'",
                ]
            )
            if hf_type == "base":
                override_lines.append(
                    "    if _ds.get('abbr', '') == 'lcb_code_generation': "
                    "_ds['infer_cfg']['prompt_template']['template'] = "
                    "dict(round=[dict(role='HUMAN', prompt='{question_content}')])"
                )
            else:
                override_lines.append(
                    "    if _ds.get('abbr', '') == 'lcb_code_generation': "
                    "_ds['infer_cfg']['prompt_template']['template'] = "
                    "dict(begin=[dict(role='SYSTEM', fallback_role='HUMAN', "
                    "prompt='You are an expert Python programmer. You will be given a question"
                    " (problem specification) and will generate a correct Python program that"
                    " matches the specification and passes all tests.')], "
                    "round=[dict(role='HUMAN', "
                    "prompt='### Question:\\n{question_content}\\n\\n{format_prompt}"
                    "### Answer: (use the provided format with backticks)\\n\\n')])"
                )
            override_lines.append(
                "    if _ds.get('abbr', '') == 'lcb_code_generation': "
                "_ds['eval_cfg']['evaluator'] = dict("
                "type='opencompass.datasets.lcb_dataset.LCBCodeGenerationEvaluatorOfficial', "
                "num_process_evaluate=4, timeout=6, "
                f"release_version='release_v5', model_type='{hf_type}', k_list={k_list!r})"
            )
        if name == "mmlu_pro":
            override_lines.append(
                "    _ds['eval_cfg']['pred_postprocessor'] = "
                "dict(type=last_option_postprocess, options='ABCDEFGHIJKLMNOP')"
            )
        # HumanEval+: bypass OC's postprocessor (lstrip breaks indent) and
        # use AZR tree-sitter sanitizer in the evaluator instead.
        # NOTE: The evaluator and postprocessor are copied into OC's
        # site-packages by the K8s launcher, so we use full module paths
        # that the eval subprocess can resolve without custom_imports.
        if name == "humaneval":
            k_list = pass_at_k_ladder(n_value)
            override_lines.append(
                "    _ds['eval_cfg']['evaluator'] = "
                "dict("
                "type='opencompass.datasets.humaneval_plus_evaluator.HumanEvalPlusEvaluatorAZR', "
                f"k={k_list!r})"
            )
            override_lines.append(
                "    _ds['eval_cfg']['pred_postprocessor'] = "
                "dict(type='opencompass.datasets.custom_postprocessors.humaneval_extract_code')"
            )

        if override_lines:
            overrides = "\n".join(override_lines)
            if name == "livecodebench":
                collect_lines.append(
                    f"for _ds in list({name}_ds):\n{overrides}\n"
                    f"datasets += [_ds for _ds in {name}_ds if 'code_generation' in _ds.get('abbr', '')]"
                )
            else:
                collect_lines.append(
                    f"for _ds in list({name}_ds):\n{overrides}\ndatasets += list({name}_ds)"
                )
        else:
            collect_lines.append(f"datasets += list({name}_ds)")

    imports_block = "\n    ".join(import_lines)
    collect_block = "\n".join(collect_lines)
    model_type_cls = "VLLM" if hf_type == "base" else "VLLMwithChatTemplate"

    # If a custom chat template is provided (e.g. nonthinking), write it to a
    # file and inject chat_template_kwargs so OC's VLLMwithChatTemplate passes
    # it to tokenizer.apply_chat_template(chat_template=...).
    _chat_template_kwargs_line = ""
    if custom_chat_template and hf_type != "base":
        # custom_chat_template is the template CONTENT (Jinja string),
        # resolved by the pkg_template Hydra resolver.  Write it to a file
        # so the generated OC Python config can read it at import time.
        config_dir = os.path.join(os.getcwd(), ".cache", "opencompass_configs")
        os.makedirs(config_dir, exist_ok=True)
        tpl_path = os.path.join(config_dir, "custom_chat_template.jinja")
        with open(tpl_path, "w") as f:
            f.write(custom_chat_template)
        _escaped_path = tpl_path.replace("\\", "\\\\")
        _chat_template_kwargs_line = (
            f"        chat_template_kwargs=dict("
            f"chat_template=open('{_escaped_path}').read()),\n"
        )
        print(f"OC model config: using custom chat template ({len(custom_chat_template)} chars)", flush=True)

    # Compute max_seq_len = prompt_length + max(max_out_len, per-benchmark overrides)
    _effective_max_out = int(max_out_len)
    if bench_max_out_len:
        _effective_max_out = max(_effective_max_out, max(bench_max_out_len.values()))
    _max_seq_len = prompt_length + _effective_max_out

    needs_postprocessor_patch = any(name == "mmlu_pro" for name, _, _ in bench_token_n)
    needs_humaneval_patch = any(name == "humaneval" for name, _, _ in bench_token_n)
    needs_lcb_patch = any(name == "livecodebench" for name, _, _ in bench_token_n)
    patch_imports = []
    custom_import_modules = []
    if needs_postprocessor_patch or needs_humaneval_patch or needs_lcb_patch:
        patch_imports.append("__import__('sys').path.insert(0, '.')")
    if needs_postprocessor_patch:
        custom_import_modules.append("scripts.patches.opencompass.postprocessors")
        patch_imports.append(
            "from scripts.patches.opencompass.postprocessors import last_option_postprocess"
        )
    if needs_humaneval_patch:
        custom_import_modules.append(
            "scripts.patches.opencompass.humaneval_plus_evaluator"
        )
        patch_imports.append(
            "from scripts.patches.opencompass.humaneval_plus_evaluator import HumanEvalPlusEvaluatorAZR"
        )
        # Also import humaneval_extract_code to replace OC's lstrip() postprocessor.
        if "scripts.patches.opencompass.postprocessors" not in custom_import_modules:
            custom_import_modules.append("scripts.patches.opencompass.postprocessors")
        patch_imports.append(
            "from scripts.patches.opencompass.postprocessors import humaneval_extract_code"
        )
    if needs_lcb_patch:
        custom_import_modules.append("scripts.patches.opencompass.lcb_dataset")
        patch_imports.append(
            "from scripts.patches.opencompass.lcb_dataset import "
            "LCBCodeGenerationDatasetOfficial, LCBCodeGenerationEvaluatorOfficial"
        )
    postprocessor_import = "\n".join(patch_imports) + "\n" if patch_imports else ""
    custom_imports_block = ""
    if custom_import_modules:
        custom_imports_block = (
            "custom_imports = dict(\n"
            f"    imports={custom_import_modules!r},\n"
            "    allow_failed_imports=False,\n"
            ")\n\n"
        )

    config_content = f"""\
# Auto-generated OpenCompass config with per-benchmark n values.
# Do not edit — regenerated on each run.

from mmengine.config import read_base
from opencompass.models import VLLM
from opencompass.models import VLLMwithChatTemplate
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
{custom_imports_block}{postprocessor_import}

with read_base():
    {imports_block}

datasets = []
{collect_block}

models = [
    dict(
        type={model_type_cls},
        abbr="eval-model",
        path="{hf_path}",
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.9, enforce_eager=True),
{_chat_template_kwargs_line}        max_out_len={max_out_len},
        max_seq_len={_max_seq_len},
        batch_size=16,
        generation_kwargs=dict(
            temperature={temperature},
            top_p={top_p},
            top_k={top_k},
        ),
        run_cfg=dict(num_gpus=1),
    ),
]

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker={num_gpus_int}),
    runner=dict(
        type=LocalRunner,
        max_num_workers={num_gpus_int},
        max_workers_per_gpu=1,
        task=dict(type=OpenICLInferTask),
    ),
)
"""

    config_dir = os.path.join(os.getcwd(), ".cache", "opencompass_configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "eval_config.py")
    with open(config_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(config_content)

    print(f"Generated Python config: {config_path}", flush=True)
    print("Per-benchmark n values:", flush=True)
    for name, _, n_value in bench_token_n:
        if n_value > 1:
            print(f"  {name}: n={n_value}", flush=True)

    cmd = ["opencompass", config_path]
    if output_dir:
        cmd.extend(["-w", output_dir])
    cmd.extend(passthrough)

    print(f"Launching: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode == 0 and not no_upload and upload_repo:
        model_id = build_model_id(hf_path, hf_repo, ckpt_prefix, ckpt_step)
        try:
            upload_results_to_hf(upload_repo, model_id, output_dir)
        except Exception as exc:
            print(f"WARNING: HF upload failed: {exc}", flush=True)
    return int(result.returncode)


def usage_text() -> str:
    """Return the CLI usage string for the thin wrapper script."""
    return (
        "Usage: python scripts/run_opencompass.py --hf-path MODEL_PATH [options]\n"
        "       python scripts/run_opencompass.py --hf-repo REPO --ckpt-prefix PREFIX --ckpt-step N\n"
        "\n"
        "Model source (one required):\n"
        "  --hf-path PATH          HuggingFace model ID or local checkpoint path\n"
        "  --hf-repo REPO          HF dataset repo (e.g. ORG/REPO) with training checkpoints\n"
        "  --ckpt-prefix PREFIX    Path prefix in the HF repo (e.g. qwen3_4b_base/FW-Alr_...)\n"
        "  --ckpt-step N           Checkpoint step number (maps to global_step_N)\n"
        "  --ckpt-role ROLE        Model role in checkpoint (default: solver)\n"
        "  --cache-dir DIR         HF download cache directory (default: HF default)\n"
        "\n"
        "Options:\n"
        "  --hf-type base|chat     Model type (auto-detected if omitted)\n"
        "  --benchmarks NAMES      Benchmark selection: all, math, code, science, reasoning,\n"
        "                          or comma-separated names (default: all)\n"
        "  --dataset-num-runs N    Number of evaluation repetitions per problem (default: none)\n"
        "  --num-gpus N            Number of GPUs for data parallelism (auto-detected)\n"
        "  --max-out-len N         Max output tokens (default: 8192)\n"
        "  --custom-chat-template-file PATH\n"
        "                          Jinja chat template file to inject into\n"
        "                          VLLMwithChatTemplate models\n"
        "  --output-dir DIR        Output directory (default: opencompass default)\n"
        "  --upload-repo REPO      HF dataset repo for result upload\n"
        "  --no-upload             Disable HF result upload\n"
        "  -- [flags]              Extra flags passed directly to opencompass\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse wrapper CLI arguments and launch OpenCompass."""
    args = list(sys.argv[1:] if argv is None else argv)

    passthrough: list[str] = []
    if "--" in args:
        sep = args.index("--")
        passthrough = args[sep + 1 :]
        args = args[:sep]

    hf_path, args = extract_flag(args, "--hf-path")
    hf_type, args = extract_flag(args, "--hf-type")
    benchmarks, args = extract_flag(args, "--benchmarks", "all")
    num_runs, args = extract_flag(args, "--dataset-num-runs")
    num_gpus, args = extract_flag(args, "--num-gpus")
    max_out_len, args = extract_flag(args, "--max-out-len", "8192")
    custom_chat_template_file, args = extract_flag(args, "--custom-chat-template-file")
    output_dir, args = extract_flag(args, "--output-dir")
    hf_repo, args = extract_flag(args, "--hf-repo")
    ckpt_prefix, args = extract_flag(args, "--ckpt-prefix")
    ckpt_step, args = extract_flag(args, "--ckpt-step")
    ckpt_role, args = extract_flag(args, "--ckpt-role", "solver")
    cache_dir, args = extract_flag(args, "--cache-dir")
    upload_repo, args = extract_flag(args, "--upload-repo", os.environ.get("HF_UPLOAD_REPO"))
    no_upload = "--no-upload" in args
    args = [arg for arg in args if arg != "--no-upload"]

    if hf_repo:
        if not ckpt_prefix or not ckpt_step:
            print(
                "Error: --hf-repo requires --ckpt-prefix and --ckpt-step.\n"
                "Example:\n"
                "  --hf-repo ORG/REPO \\\n"
                "  --ckpt-prefix qwen3_4b_base/FW-Alr_2e-6-... \\\n"
                "  --ckpt-step 50",
                file=sys.stderr,
            )
            return 2
        hf_path = download_hf_checkpoint(
            hf_repo=hf_repo,
            ckpt_prefix=ckpt_prefix,
            ckpt_step=int(ckpt_step),
            role=ckpt_role or "solver",
            cache_dir=cache_dir,
        )

    if not hf_path:
        print(usage_text(), file=sys.stderr)
        return 2

    ensure_hf_token()
    if not hf_type:
        hf_type = detect_hf_type(hf_path)
        print(f"Auto-detected --hf-type {hf_type} for {hf_path}", flush=True)

    custom_chat_template = None
    if custom_chat_template_file:
        with open(custom_chat_template_file, "r", encoding="utf-8") as file_obj:
            custom_chat_template = file_obj.read()
        print(
            "Loaded custom chat template file "
            f"{custom_chat_template_file} ({len(custom_chat_template)} chars)",
            flush=True,
        )

    assert benchmarks is not None
    return launch_opencompass(
        hf_path=hf_path,
        hf_type=hf_type,
        benchmarks=benchmarks,
        num_runs=num_runs,
        temperature=0.7,
        top_p=1.0,
        top_k=-1,
        num_gpus=num_gpus,
        max_out_len=max_out_len or "8192",
        output_dir=output_dir,
        passthrough=passthrough,
        upload_repo=upload_repo,
        no_upload=no_upload,
        hf_repo=hf_repo,
        ckpt_prefix=ckpt_prefix,
        ckpt_step=ckpt_step,
        custom_chat_template=custom_chat_template,
    )


__all__ = [
    "BASE_TOKENS",
    "BENCHMARK_GROUPS",
    "BENCHMARK_MAX_OUT_LEN",
    "BENCHMARK_N",
    "CHAT_MARKERS",
    "CHAT_TOKEN_OVERRIDES",
    "MODEL_REGISTRY",
    "build_model_id",
    "detect_hf_type",
    "download_hf_checkpoint",
    "ensure_hf_token",
    "ensure_opencompass_installed",
    "extract_flag",
    "find_latest_run_dir",
    "get_visible_gpus",
    "launch_opencompass",
    "launch_with_python_config",
    "main",
    "pass_at_k_ladder",
    "resolve_benchmark_names",
    "resolve_tokens",
    "upload_results_to_hf",
    "usage_text",
]
