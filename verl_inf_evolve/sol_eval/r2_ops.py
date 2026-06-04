"""Remote storage operations for solver evaluation.

Historically the standalone eval pipeline only supported R2-backed runs, so
this module kept its original ``r2_ops`` name. The implementation now routes
through the generic ``RemoteBackend`` abstraction and supports both:

- ``s3://`` / ``r2://`` experiment roots
- ``hf://datasets/...`` experiment roots

The public function names are preserved to avoid churn in callers and tests.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any | None, key: str, default: Any) -> Any:
    """Read a key from a mapping-like config object."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)



def resolve_remote_uri(
    remote_uri: str,
    remote_cfg: Any | None = None,
) -> tuple[str, str | None]:
    """Resolve a remote URI and return ``(resolved_uri, hf_token)``.

    For HF dataset URIs with a namespace placeholder, this reuses the same
    pool-based namespace resolution helper as training. For fixed HF dataset
    namespaces, this also falls back to the configured token pool so writes to
    an existing public repo still use the matching namespace token.

    Token priority (highest first):
      1. Pool-based ``__namespace__`` resolution
      2. Explicit ``remote.hf_token`` in config
      3. Pool-based namespace matching (URI namespace → pool entry)
      4. ``HF_TOKEN`` env var (generic fallback)

    Namespace-specific pool matching (3) is checked *before* the generic
    ``HF_TOKEN`` env var (4) so that each URI gets the token that owns the
    target repo, even when ``HF_TOKEN`` was seeded from a different URI
    (e.g. checkpoint storage vs eval-result storage).
    """
    remote_uri = str(remote_uri)
    if not remote_uri.startswith("hf://"):
        return remote_uri, None

    from verl_inf_evolve.storage.hf_remote_resolver import (
        load_token_pool_with_warnings,
        resolve_hf_remote_from_pool,
    )
    from verl_inf_evolve.storage.remote_backend import _parse_hf_dataset_uri

    # 1. Pool-based __namespace__ resolution (handles placeholder URIs)
    resolved = resolve_hf_remote_from_pool(remote_uri, remote_cfg or {})
    if resolved is not None:
        return resolved.uri, resolved.token

    # 2. Explicit token in config (remote.hf_token)
    explicit_cfg_token = _cfg_get(remote_cfg, "hf_token", None)
    if explicit_cfg_token:
        return remote_uri, str(explicit_cfg_token)

    # 3. Pool-based namespace matching — find the token that owns this repo
    parsed = _parse_hf_dataset_uri(remote_uri)
    namespace, _repo = parsed["repo_id"].split("/", 1)
    pool, _warnings = load_token_pool_with_warnings(remote_cfg or {})
    for entry in pool:
        if entry.namespace == namespace:
            return remote_uri, entry.token

    # 4. Generic HF_TOKEN env var fallback
    env_var_name = str(_cfg_get(remote_cfg, "hf_token_env_var", "HF_TOKEN"))
    env_token = os.environ.get(env_var_name)
    if env_token:
        return remote_uri, env_token
    if env_var_name != "HF_TOKEN":
        env_token = os.environ.get("HF_TOKEN")
        if env_token:
            return remote_uri, env_token

    return remote_uri, None


def _create_remote_backend(
    remote_uri: str,
    remote_cfg: Any | None = None,
    *,
    auto_create_repo: bool = False,
):
    """Create a backend for the given remote URI."""
    from verl_inf_evolve.storage.remote_backend import (
        build_hf_backend_kwargs,
        create_remote_backend,
    )

    resolved_uri, token = resolve_remote_uri(remote_uri, remote_cfg)
    backend_kwargs: dict[str, Any] = {}

    if resolved_uri.startswith("hf://"):
        backend_kwargs = build_hf_backend_kwargs(
            remote_cfg,
            token=token,
            auto_create_repo=auto_create_repo,
        )

    backend = create_remote_backend(resolved_uri, **backend_kwargs)
    return backend, resolved_uri


def _download_json(backend: Any, remote_key: str) -> Optional[dict]:
    """Download and parse a JSON object from remote storage."""
    try:
        payload = backend.download_bytes(remote_key)
        return json.loads(payload.decode("utf-8"))
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("Failed to download/parse remote JSON %s: %s", remote_key, e)
        return None


def load_run_metadata(
    remote_sync_path: str,
    remote_cfg: Any | None = None,
) -> Optional[dict]:
    """Load ``run_metadata.json`` from a remote training run root."""
    try:
        backend, resolved_uri = _create_remote_backend(remote_sync_path, remote_cfg)
    except Exception as e:
        logger.warning("Failed to create backend for run metadata lookup: %s", e)
        return None

    metadata = _download_json(backend, "run_metadata.json")
    if metadata is None:
        logger.debug("No run_metadata.json found at %s", resolved_uri)
    return metadata


def check_r2_result_exists(
    r2_eval_base: str,
    output_filename: str,
    remote_cfg: Any | None = None,
) -> bool:
    """Check if a result file exists remotely and is a valid completed evaluation."""
    from verl_inf_evolve.sol_eval.result_format import is_result_complete

    try:
        backend, resolved_uri = _create_remote_backend(r2_eval_base, remote_cfg)
    except Exception as e:
        logger.warning("Remote backend unavailable; cannot check existing results: %s", e)
        return False

    if not backend.exists(output_filename):
        return False

    result_json = _download_json(backend, output_filename)
    if result_json is None:
        logger.warning(
            "Remote result exists but validation download failed for %s at %s",
            output_filename,
            resolved_uri,
        )
        return False

    try:
        return is_result_complete(result_json)
    except Exception as e:
        logger.warning(
            "Remote result exists but validation failed for %s: %s",
            output_filename,
            e,
        )
        return False


def download_r2_result(
    r2_eval_base: str,
    output_filename: str,
    remote_cfg: Any | None = None,
) -> Optional[dict]:
    """Download an existing result from remote storage and return it as a dict."""
    try:
        backend, _ = _create_remote_backend(r2_eval_base, remote_cfg)
    except Exception as e:
        logger.warning("Failed to create backend for result download: %s", e)
        return None

    return _download_json(backend, output_filename)


def upload_result_to_r2(
    local_path: str,
    r2_eval_base: str,
    output_filename: str,
    remote_cfg: Any | None = None,
) -> bool:
    """Upload a result JSON file to remote storage."""
    if not os.path.isfile(local_path):
        logger.error("Local result file not found: %s", local_path)
        return False

    try:
        backend, resolved_uri = _create_remote_backend(
            r2_eval_base,
            remote_cfg,
            auto_create_repo=True,
        )
    except Exception as e:
        logger.error("Failed to create backend for result upload: %s", e)
        return False

    with open(local_path, "rb") as f:
        ok = backend.upload_bytes(f.read(), output_filename)

    if ok:
        logger.info("Uploaded result to remote storage: %s/%s", resolved_uri, output_filename)
    else:
        logger.error("Failed to upload result to remote storage: %s", output_filename)

    return ok


def download_checkpoint_from_r2(
    remote_sync_path: str,
    ckpt_num: int,
    local_cache_dir: str,
    remote_cfg: Any | None = None,
) -> str:
    """Download a checkpoint from remote storage to local cache."""
    os.makedirs(local_cache_dir, exist_ok=True)

    local_solver_dir = os.path.join(local_cache_dir, "solver")
    os.makedirs(local_solver_dir, exist_ok=True)

    backend, resolved_uri = _create_remote_backend(remote_sync_path, remote_cfg)
    remote_key = f"global_step_{ckpt_num}/solver"

    logger.info(
        "Downloading checkpoint %d from remote storage: %s/%s -> %s",
        ckpt_num,
        resolved_uri,
        remote_key,
        local_solver_dir,
    )
    ok = backend.download_dir(remote_key, local_solver_dir)
    if not ok:
        raise RuntimeError(
            f"Failed to download checkpoint {ckpt_num} from remote storage: "
            f"{resolved_uri}/{remote_key}"
        )
    logger.info("Checkpoint %d downloaded to: %s", ckpt_num, local_solver_dir)
    return local_cache_dir


def discover_checkpoints_on_r2(
    remote_sync_path: str,
    remote_cfg: Any | None = None,
) -> list[int]:
    """List available checkpoint numbers by scanning ``global_step_*`` prefixes."""
    import re

    backend, resolved_uri = _create_remote_backend(remote_sync_path, remote_cfg)
    checkpoint_nums: list[int] = []
    for child in backend.list_immediate_children(""):
        match = re.match(r"^global_step_(\d+)$", child)
        if match:
            checkpoint_nums.append(int(match.group(1)))

    if not checkpoint_nums:
        raise RuntimeError(
            f"No global_step_* checkpoints found at {resolved_uri}. "
            "Please specify eval.checkpoints explicitly."
        )

    checkpoint_nums.sort()
    logger.info(
        "Discovered %d checkpoints at %s: %s",
        len(checkpoint_nums),
        resolved_uri,
        checkpoint_nums,
    )
    return checkpoint_nums


def cleanup_checkpoint(local_cache_dir: str) -> None:
    """Delete a locally cached checkpoint directory."""
    if os.path.isdir(local_cache_dir):
        shutil.rmtree(local_cache_dir, ignore_errors=True)
        logger.info("Cleaned up local checkpoint: %s", local_cache_dir)
