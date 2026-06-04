"""Prepare a balanced subset of documents for generator evaluation.

Samples equal numbers of documents per science field so that every
field has equal representation in the evaluation set.

CLI usage::

    python -m verl_inf_evolve.gen_eval.prepare_eval_docs \
        --num_docs 128 --seed 42 \
        --input_path .cache/data/preprocessed/documents.json \
        --output_path .cache/data/preprocessed/eval_documents.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_documents(input_path: str) -> list[dict]:
    """Load documents from a JSON file."""
    with open(input_path, "r") as f:
        return json.load(f)


def sample_balanced_documents(
    documents: list[dict],
    num_docs: int,
    seed: int,
) -> list[dict]:
    """Sample a balanced subset of documents across science fields.

    Each field receives an equal quota (``num_docs // num_fields``).
    If a field has fewer documents than its quota, all its documents
    are taken and the shortfall is redistributed evenly across the
    remaining fields (repeating until all slots are filled).

    Args:
        documents: Full list of document dicts, each with a ``"field"`` key.
        num_docs: Total number of documents to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of sampled document dicts (length == ``num_docs``).
    """
    rng = random.Random(seed)

    # Group documents by field
    docs_by_field: dict[str, list[dict]] = defaultdict(list)
    for doc in documents:
        docs_by_field[doc["field"]].append(doc)

    fields = sorted(docs_by_field.keys())
    num_fields = len(fields)

    if num_fields == 0:
        return []

    # Calculate per-field quota
    base_quota = num_docs // num_fields
    remainder = num_docs % num_fields

    # Initial quotas: distribute remainder one-per-field in sorted order
    quotas: dict[str, int] = {}
    for i, field in enumerate(fields):
        quotas[field] = base_quota + (1 if i < remainder else 0)

    # Sample with redistribution for fields with fewer docs than quota.
    # Phase 1: Identify short fields (fewer docs than quota), take all their
    #          docs, and redistribute the shortfall to other fields.
    # Phase 2: Sample from the remaining (non-short) fields.
    sampled: list[dict] = []
    active_fields = set(fields)

    # Iteratively handle short fields until no shortfall remains
    changed = True
    while changed:
        changed = False
        shortfall = 0
        exhausted: list[str] = []

        for field in sorted(active_fields):
            available = docs_by_field[field]
            if len(available) < quotas[field]:
                sampled.extend(available)
                shortfall += quotas[field] - len(available)
                docs_by_field[field] = []
                quotas[field] = 0
                exhausted.append(field)

        for field in exhausted:
            active_fields.discard(field)

        if shortfall > 0 and active_fields:
            changed = True
            remaining = sorted(active_fields)
            per_field_extra = shortfall // len(remaining)
            extra_remainder = shortfall % len(remaining)
            for i, field in enumerate(remaining):
                quotas[field] += per_field_extra + (1 if i < extra_remainder else 0)

    # Phase 2: Sample from fields that have enough documents
    for field in sorted(active_fields):
        available = docs_by_field[field]
        quota = quotas[field]
        if quota > 0 and len(available) >= quota:
            chosen = rng.sample(available, quota)
            sampled.extend(chosen)

    return sampled[:num_docs]


def print_summary(sampled: list[dict], num_docs: int) -> None:
    """Print per-field counts and total."""
    counts: dict[str, int] = defaultdict(int)
    for doc in sampled:
        counts[doc["field"]] += 1

    print("=== Eval Document Sampling Summary ===")
    for field in sorted(counts.keys()):
        print(f"  {field}: {counts[field]}")
    print(f"  Total: {len(sampled)} (requested: {num_docs})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a balanced subset of documents for generator evaluation."
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        default=128,
        help="Total number of documents to sample (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=".cache/data/preprocessed/documents.json",
        help="Path to input documents.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".cache/data/preprocessed/eval_documents.json",
        help="Path to output eval_documents.json",
    )
    args = parser.parse_args(argv)

    # Load
    documents = load_documents(args.input_path)
    print(f"Loaded {len(documents)} documents from {args.input_path}")

    # Sample
    sampled = sample_balanced_documents(documents, args.num_docs, args.seed)

    # Summary
    print_summary(sampled, args.num_docs)

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
