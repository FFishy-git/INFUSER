#!/usr/bin/env python3
"""Build a document corpus by appending repeated Putnam/AIME problem chunks.

The output preserves the existing documents.json schema:
doc_id, original_doc_idx, chunk_idx, content, field, source_pdf, token_count.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_BASE = Path(".cache/data/preprocessed/documents.json")
DEFAULT_PUTNAM = Path(".cache/data/preprocessed/benchmarks/putnam.json")
DEFAULT_AIME = Path(".cache/data/preprocessed/benchmarks/aime_history.json")
DEFAULT_OUTPUT = Path(".cache/data/preprocessed/documents_with_putnam_aime_history_math10000.json")
DEFAULT_PUTNAM_CHUNKS_DIR = Path(".cache/data/source/putnam_chunks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a repeated Putnam+AIME-history math document pool to documents.json."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--putnam", type=Path, default=DEFAULT_PUTNAM)
    parser.add_argument("--aime-history", type=Path, default=DEFAULT_AIME)
    parser.add_argument("--putnam-chunks-dir", type=Path, default=DEFAULT_PUTNAM_CHUNKS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-math-docs", type=int, default=10000)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--exclude-aime-qids-from",
        type=Path,
        default=None,
        help=(
            "Path to a dev-set JSON. AIME-history qids that appear there "
            "(rows with data_source == 'aime') will be removed from the AIME "
            "pool BEFORE round-robin repetition fills the math-doc quota."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    return data


def load_token_counter(model_name: str):
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return lambda text: len(tokenizer.encode(text))
    except Exception as exc:  # pragma: no cover - fallback for offline/minimal envs
        print(f"Warning: tokenizer {model_name!r} unavailable ({exc}); using regex token estimate.")
        pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
        return lambda text: len(pattern.findall(text))


def strip_putnam_title(content: str) -> str:
    lines = content.strip().splitlines()
    if lines and lines[0].startswith("# Putnam "):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


def load_putnam_chunk_content(chunks_dir: Path) -> dict[str, str]:
    content_by_id: dict[str, str] = {}
    if not chunks_dir.exists():
        print(f"Warning: Putnam chunks directory not found: {chunks_dir}")
        return content_by_id

    for chunk_path in sorted(chunks_dir.glob("putnam_*.chunks.json")):
        chunks = load_json(chunk_path)
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            year = metadata.get("year")
            label = metadata.get("problem_label")
            page_content = chunk.get("page_content", "")
            if year is None or not label or not page_content:
                continue
            question_id = f"putnam_{year}_{label}"
            content_by_id[question_id] = strip_putnam_title(page_content)
    return content_by_id


def make_content(item: dict[str, Any], putnam_chunk_content: dict[str, str]) -> str:
    question_id = item["question_id"]
    if item.get("data_source") == "putnam":
        chunk_content = putnam_chunk_content.get(question_id)
        if chunk_content:
            return chunk_content

    question = item["question_text"].strip()
    answer = str(item.get("ground_truth", "")).strip()
    return f"## Problem\n{question}\n\n## Answer\n{answer}".strip()


def build_math_documents(
    putnam: list[dict[str, Any]],
    aime_history: list[dict[str, Any]],
    *,
    start_doc_id: int,
    target_math_docs: int,
    putnam_chunk_content: dict[str, str],
    count_tokens,
) -> list[dict[str, Any]]:
    pool: list[tuple[str, int, dict[str, Any]]] = (
        [("putnam", idx, item) for idx, item in enumerate(putnam)]
        + [("aime_history", len(putnam) + idx, item) for idx, item in enumerate(aime_history)]
    )
    if not pool:
        raise ValueError("Putnam+AIME math pool is empty")

    docs: list[dict[str, Any]] = []
    for out_idx in range(target_math_docs):
        source_name, pool_idx, item = pool[out_idx % len(pool)]
        content = make_content(item, putnam_chunk_content)
        docs.append(
            {
                "doc_id": str(start_doc_id + out_idx),
                "original_doc_idx": pool_idx,
                "chunk_idx": 0,
                "content": content,
                "field": "Math",
                "source_pdf": f".cache/data/preprocessed/benchmarks/{source_name}.json#{item['question_id']}",
                "token_count": count_tokens(content),
            }
        )
    return docs


def next_doc_id(docs: list[dict[str, Any]]) -> int:
    numeric_ids = [int(doc["doc_id"]) for doc in docs if str(doc.get("doc_id", "")).isdigit()]
    if len(numeric_ids) != len(docs):
        raise ValueError("All existing doc_id values must be numeric strings")
    return max(numeric_ids, default=-1) + 1


def load_excluded_aime_qids(path: Path) -> set[str]:
    """Read a dev-set JSON and return AIME qids (rows with data_source='aime')."""
    rows = load_json(path)
    excluded = {
        row["question_id"]
        for row in rows
        if row.get("data_source") == "aime" and "question_id" in row
    }
    return excluded


def main() -> None:
    args = parse_args()

    base_docs = load_json(args.base)
    putnam = load_json(args.putnam)
    aime_history = load_json(args.aime_history)
    putnam_chunk_content = load_putnam_chunk_content(args.putnam_chunks_dir)
    count_tokens = load_token_counter(args.tokenizer)

    excluded_aime_qids: set[str] = set()
    aime_history_pre_count = len(aime_history)
    if args.exclude_aime_qids_from is not None:
        excluded_aime_qids = load_excluded_aime_qids(args.exclude_aime_qids_from)
        aime_history = [
            item for item in aime_history if item.get("question_id") not in excluded_aime_qids
        ]
        print(
            f"excluded_aime_qids_source={args.exclude_aime_qids_from}\n"
            f"excluded_aime_qids_count={len(excluded_aime_qids)}\n"
            f"aime_history_before_exclude={aime_history_pre_count}\n"
            f"aime_history_after_exclude={len(aime_history)}"
        )

    math_docs = build_math_documents(
        putnam,
        aime_history,
        start_doc_id=next_doc_id(base_docs),
        target_math_docs=args.target_math_docs,
        putnam_chunk_content=putnam_chunk_content,
        count_tokens=count_tokens,
    )
    combined = base_docs + math_docs
    if not args.no_shuffle:
        random.Random(args.shuffle_seed).shuffle(combined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(combined, f, indent=2)

    print(f"base_docs={len(base_docs)}")
    print(f"putnam_unique={len(putnam)}")
    print(f"putnam_chunks_loaded={len(putnam_chunk_content)}")
    print(f"putnam_chunks_matched={sum(item['question_id'] in putnam_chunk_content for item in putnam)}")
    print(f"aime_history_unique={len(aime_history)}")
    print(f"math_pool_unique={len(putnam) + len(aime_history)}")
    print(f"math_docs_appended={len(math_docs)}")
    print(f"shuffled={not args.no_shuffle}")
    if not args.no_shuffle:
        print(f"shuffle_seed={args.shuffle_seed}")
    print(f"total_docs={len(combined)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
