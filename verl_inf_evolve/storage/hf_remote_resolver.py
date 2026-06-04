"""Helpers for resolving HF dataset remote URIs from a token pool."""

from __future__ import annotations

from functools import lru_cache
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

logger = logging.getLogger(__name__)

from verl_inf_evolve.storage.remote_backend import _parse_hf_dataset_uri


@dataclass(frozen=True)
class HfTokenPoolEntry:
    """Resolved pool entry with namespace and usable token."""

    namespace: str
    token: str
    token_source: str
    capacity_bytes: int | None = None


@dataclass(frozen=True)
class HfPoolCandidateStats:
    """Selection stats for one pool entry."""

    namespace: str
    prefix_exists: bool
    repo_used_storage: int | None
    namespace_used_storage: int | None
    capacity_bytes: int | None


@dataclass(frozen=True)
class ResolvedHFRemote:
    """Resolved HF remote target selected from a token pool."""

    uri: str
    token: str
    namespace: str
    repo: str
    prefix: str
    selection_reason: str
    warning: str | None
    candidates: list[HfPoolCandidateStats]


def resolve_hf_remote_from_pool(
    uri: str,
    remote_cfg: dict[str, Any] | Any,
) -> ResolvedHFRemote | None:
    """Resolve ``hf://datasets/<namespace>/...`` using the configured token pool.

    Returns ``None`` when the URI does not use the namespace placeholder.
    """
    parsed = _parse_hf_dataset_uri(uri)
    namespace, repo = parsed["repo_id"].split("/", 1)
    placeholder = str(_cfg_get(remote_cfg, "hf_namespace_placeholder", "<namespace>"))
    if namespace != placeholder:
        return None

    pool = _load_token_pool(remote_cfg)
    if not pool:
        raise ValueError(
            "HF namespace auto-discovery requires at least one valid token pool "
            "entry. Configure remote.hf_token_pool_env_var, "
            "remote.hf_token_pool, and/or remote.hf_token_pool_file."
        )

    prefix = parsed["prefix"]
    revision = str(_cfg_get(remote_cfg, "hf_revision", "main"))

    candidates: list[tuple[HfTokenPoolEntry, HfPoolCandidateStats]] = []
    matches: list[tuple[HfTokenPoolEntry, HfPoolCandidateStats]] = []

    for entry in pool:
        api = HfApi(token=entry.token)
        repo_id = f"{entry.namespace}/{repo}"
        prefix_exists = _prefix_exists(api, repo_id, prefix, revision)
        repo_used_storage = _repo_used_storage(api, repo_id, entry.token)
        namespace_used_storage = _namespace_used_storage(
            api, entry.namespace, entry.token
        )
        stats = HfPoolCandidateStats(
            namespace=entry.namespace,
            prefix_exists=prefix_exists,
            repo_used_storage=repo_used_storage,
            namespace_used_storage=namespace_used_storage,
            capacity_bytes=entry.capacity_bytes,
        )
        candidates.append((entry, stats))
        if prefix_exists:
            matches.append((entry, stats))

    warning: str | None = None
    if matches:
        selected_entry, _stats = matches[0]
        if len(matches) > 1:
            warning = (
                "Found multiple matching HF namespaces for the current run prefix; "
                f"using the first configured namespace: {selected_entry.namespace}"
            )
        return ResolvedHFRemote(
            uri=_build_hf_uri(selected_entry.namespace, repo, prefix),
            token=selected_entry.token,
            namespace=selected_entry.namespace,
            repo=repo,
            prefix=prefix,
            selection_reason="existing_prefix_reuse",
            warning=warning,
            candidates=[stats for _, stats in candidates],
        )

    selected_entry, selected_stats, heuristic = _pick_best_available(candidates)
    warning = None
    if heuristic != "capacity_minus_namespace_usage":
        warning = (
            "HF auto-selection is using a storage heuristic because the Hub does "
            "not expose exact remaining quota here."
        )

    return ResolvedHFRemote(
        uri=_build_hf_uri(selected_entry.namespace, repo, prefix),
        token=selected_entry.token,
        namespace=selected_entry.namespace,
        repo=repo,
        prefix=prefix,
        selection_reason=heuristic,
        warning=warning,
        candidates=[stats for _, stats in candidates],
    )


def _pick_best_available(
    candidates: list[tuple[HfTokenPoolEntry, HfPoolCandidateStats]]
) -> tuple[HfTokenPoolEntry, HfPoolCandidateStats, str]:
    best_entry: HfTokenPoolEntry | None = None
    best_stats: HfPoolCandidateStats | None = None
    best_score: tuple[int, int] | None = None
    best_reason = "first_available"

    for entry, stats in candidates:
        score: tuple[int, int]
        reason: str
        if stats.capacity_bytes is not None and stats.namespace_used_storage is not None:
            score = (2, int(stats.capacity_bytes - stats.namespace_used_storage))
            reason = "capacity_minus_namespace_usage"
        elif stats.namespace_used_storage is not None:
            score = (1, -int(stats.namespace_used_storage))
            reason = "least_namespace_used_storage"
        elif stats.repo_used_storage is not None:
            score = (0, -int(stats.repo_used_storage))
            reason = "least_repo_used_storage"
        else:
            score = (-1, 0)
            reason = "first_available"

        if best_score is None or score > best_score:
            best_entry = entry
            best_stats = stats
            best_score = score
            best_reason = reason

    if best_entry is None or best_stats is None:
        raise ValueError("No usable HF token pool entries were resolved.")

    return best_entry, best_stats, best_reason


def _prefix_exists(api: HfApi, repo_id: str, prefix: str, revision: str) -> bool:
    try:
        if not prefix:
            api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
            return True
        info = api.get_paths_info(
            repo_id=repo_id,
            paths=[prefix],
            repo_type="dataset",
            revision=revision,
        )
        return len(info) > 0
    except (RepositoryNotFoundError, EntryNotFoundError):
        return False
    except Exception:
        return False


def _repo_used_storage(api: HfApi, repo_id: str, token: str) -> int | None:
    try:
        info = api.dataset_info(
            repo_id=repo_id,
            expand=["usedStorage"],
            token=token,
        )
    except RepositoryNotFoundError:
        return 0
    except Exception:
        return None
    return _extract_used_storage(info)


def _namespace_used_storage(api: HfApi, namespace: str, token: str) -> int | None:
    total = 0
    try:
        datasets = api.list_datasets(
            author=namespace,
            expand=["usedStorage"],
            token=token,
        )
        for info in datasets:
            used = _extract_used_storage(info)
            if used is not None:
                total += used
        return total
    except Exception:
        return None


def _extract_used_storage(info: Any) -> int | None:
    value = getattr(info, "usedStorage", None)
    if value is None:
        value = getattr(info, "used_storage", None)
    if value is None and isinstance(info, dict):
        value = info.get("usedStorage", info.get("used_storage"))
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _load_token_pool(remote_cfg: dict[str, Any] | Any) -> list[HfTokenPoolEntry]:
    resolved, _warnings = _load_token_pool_internal(
        remote_cfg,
        ignore_entry_errors=False,
    )
    return resolved


def load_token_pool_with_warnings(
    remote_cfg: dict[str, Any] | Any,
) -> tuple[list[HfTokenPoolEntry], list[str]]:
    return _load_token_pool_internal(remote_cfg, ignore_entry_errors=True)


def _load_token_pool_internal(
    remote_cfg: dict[str, Any] | Any,
    *,
    ignore_entry_errors: bool,
) -> tuple[list[HfTokenPoolEntry], list[str]]:
    raw_entries: list[Any] = []

    pool_env_var = _cfg_get(remote_cfg, "hf_token_pool_env_var", None)
    if pool_env_var:
        payload = os.environ.get(str(pool_env_var))
        if payload:
            raw_entries.extend(_parse_pool_payload(payload, source=str(pool_env_var)))

    pool_file = _cfg_get(remote_cfg, "hf_token_pool_file", None)
    if pool_file:
        with open(os.path.expanduser(str(pool_file)), "r", encoding="utf-8") as f:
            raw_entries.extend(_parse_pool_payload(f.read(), source=str(pool_file)))

    inline_entries = _cfg_get(remote_cfg, "hf_token_pool", []) or []
    if inline_entries:
        raw_entries.extend(list(inline_entries))

    resolved: list[HfTokenPoolEntry] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for idx, raw in enumerate(raw_entries):
        try:
            namespace, token, token_source, capacity_bytes = _resolve_pool_entry(raw, idx)
        except Exception as exc:
            if not ignore_entry_errors:
                raise
            warnings.append(f"HF token pool entry {idx} skipped: {exc}")
            continue
        if token is None:
            continue
        dedupe_key = (namespace, token)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        resolved.append(
            HfTokenPoolEntry(
                namespace=namespace,
                token=token,
                token_source=token_source,
                capacity_bytes=capacity_bytes,
            )
        )
    return resolved, warnings


def _parse_pool_payload(payload: str, *, source: str) -> list[Any]:
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"HF token pool payload from {source} must be valid JSON."
        ) from e
    if isinstance(loaded, dict):
        loaded = loaded.get("entries", [])
    if not isinstance(loaded, list):
        raise ValueError(f"HF token pool payload from {source} must be a JSON list.")
    return loaded


def _resolve_pool_entry(
    raw: Any,
    idx: int,
) -> tuple[str, str | None, str, int | None]:
    if isinstance(raw, str):
        token = raw.strip()
        if not token:
            return "", None, f"entry[{idx}]", None
        namespace = _discover_namespace_from_token(token)
        return namespace, token, f"entry[{idx}]", None

    if not hasattr(raw, "get"):
        raise ValueError(f"HF token pool entry {idx} must be a string or mapping.")

    if _cfg_get(raw, "disabled", False):
        return "", None, f"entry[{idx}]", None

    capacity_bytes = _cfg_get(raw, "capacity_bytes", None)
    if capacity_bytes is not None:
        capacity_bytes = int(capacity_bytes)

    token: str | None = None
    token_source = f"entry[{idx}]"
    token_env_var = str(_cfg_get(raw, "token_env_var", "")).strip()
    if token_env_var:
        token = os.environ.get(token_env_var)
        token_source = token_env_var

    if token is None:
        inline_token = _cfg_get(raw, "token", None)
        if inline_token:
            token = str(inline_token).strip()
            token_source = f"entry[{idx}]"

    if not token:
        return "", None, token_source, capacity_bytes

    namespace = str(_cfg_get(raw, "namespace", "")).strip()
    if not namespace:
        namespace = _discover_namespace_from_token(token)

    return namespace, token, token_source, capacity_bytes


@lru_cache(maxsize=256)
def _discover_namespace_from_token(token: str) -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            api = HfApi(token=token)
            info = api.whoami(token=token)
            namespace = str(info.get("name", "")).strip()
            if not namespace:
                raise ValueError("Unable to discover HF namespace from token.")
            return namespace
        except ValueError:
            raise
        except Exception as exc:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                logger.warning(
                    "HF whoami attempt %d/%d failed (%s), retrying in %ds...",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                raise


def _build_hf_uri(namespace: str, repo: str, prefix: str) -> str:
    if prefix:
        return f"hf://datasets/{namespace}/{repo}/{prefix}"
    return f"hf://datasets/{namespace}/{repo}"


def _cfg_get(cfg: dict[str, Any] | Any, key: str, default: Any) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default
