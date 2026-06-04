"""
Download and prepare data for inf_evolve on Kubernetes.

This script downloads the required datasets from HuggingFace and prepares them
for use in the inf_evolve training pipeline.

Datasets:
- SuperGPQA: Multiple-choice QA dataset for dev/test evaluation
- ev_dataset: Textbook chunks for question generation

    Usage:
    python -m launch.preparation.download_data \
        --output-dir /workspace/inf-evolve/data
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


DEFAULT_PREPROCESSED_HF_REPO = "Siyuc/infuser-data"

PREPROCESSED_HF_FILES = [
    "documents.json",
    "documents_with_putnam_aime_history_math10000.json",
    "eval_documents.json",
    "curriculum_pool/supergpqa_science_800.json",
    "curriculum_pool/supergpqa_science_pruned_400_aime_history_400.json",
    "benchmarks/aime2024.json",
    "benchmarks/aime2025.json",
    "benchmarks/bbeh.json",
    "benchmarks/combine_2000.json",
    "benchmarks/gpqa_diamond.json",
    "benchmarks/hmmt.json",
    "benchmarks/math500.json",
    "benchmarks/medqa.json",
    "benchmarks/medxpertqa.json",
    "benchmarks/mmlu_pro.json",
    "benchmarks/olympiadbench.json",
    "benchmarks/supergpqa.json",
]


def download_supergpqa(
    output_dir: Path,
    num_dev: int = 150,
    num_test: int = 1170,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Download and split SuperGPQA dataset.

    Args:
        output_dir: Output directory for processed data.
        num_dev: Number of samples for dev set.
        num_test: Number of samples for test set.
        seed: Random seed for splitting.
        verbose: Print progress.

    Returns:
        dict with paths to dev and test files.
    """
    from datasets import load_dataset
    import random

    if verbose:
        print("Downloading SuperGPQA dataset...")

    # Load dataset from HuggingFace
    try:
        dataset = load_dataset("fdtn-ai/SuperGPQA", split="train")
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        print("Trying alternative source...")
        # Fallback: try loading from local file if available
        local_path = Path(".cache/data/source/supergpqa/SuperGPQA-all.jsonl")
        if local_path.exists():
            with open(local_path, 'r') as f:
                data = [json.loads(line) for line in f]
            dataset = data
        else:
            raise ValueError("Could not load SuperGPQA dataset")

    if verbose:
        print(f"  Loaded {len(dataset)} samples")

    # Convert to list and shuffle
    if hasattr(dataset, 'to_list'):
        all_samples = dataset.to_list()
    else:
        all_samples = list(dataset)

    random.seed(seed)
    random.shuffle(all_samples)

    # Split into dev and test
    dev_samples = all_samples[:num_dev]
    test_samples = all_samples[num_dev:num_dev + num_test]

    # Format samples for inf_evolve
    def format_sample(sample, idx):
        return {
            "question_id": sample.get("id", str(idx)),
            "question_text": sample.get("question", sample.get("Question", "")),
            "choices": sample.get("choices", sample.get("Options", [])),
            "ground_truth": sample.get("answer", sample.get("Answer", "")),
            "domain": sample.get("domain", sample.get("Field", "Unknown")),
            "difficulty": sample.get("difficulty", "unknown"),
        }

    dev_formatted = [format_sample(s, i) for i, s in enumerate(dev_samples)]
    test_formatted = [format_sample(s, i) for i, s in enumerate(test_samples)]

    # Save to files
    preprocessed_dir = output_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    dev_path = preprocessed_dir / "dev.json"
    test_path = preprocessed_dir / "test.json"

    with open(dev_path, 'w') as f:
        json.dump(dev_formatted, f, indent=2)

    with open(test_path, 'w') as f:
        json.dump(test_formatted, f, indent=2)

    if verbose:
        print(f"  Saved {len(dev_formatted)} dev samples to {dev_path}")
        print(f"  Saved {len(test_formatted)} test samples to {test_path}")

    return {
        "dev_path": str(dev_path),
        "test_path": str(test_path),
        "num_dev": len(dev_formatted),
        "num_test": len(test_formatted),
    }


def download_textbook_chunks(
    output_dir: Path,
    fields: List[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Download textbook chunks from ev_dataset.

    Args:
        output_dir: Output directory for processed data.
        fields: List of fields to download (default: all).
        verbose: Print progress.

    Returns:
        dict with path to documents file.
    """
    from datasets import load_dataset

    if fields is None:
        fields = ["Astronomy", "Geography", "Physics", "Biochemistry"]

    if verbose:
        print("Downloading textbook chunks from ev_dataset...")

    all_documents = []
    doc_id = 0

    for field in fields:
        if verbose:
            print(f"  Loading {field}...")

        try:
            # Try loading from HuggingFace
            dataset = load_dataset(
                "beiningwu7/ev_dataset",
                data_dir=f"chunks/{field}",
                split="train"
            )

            for sample in dataset:
                all_documents.append({
                    "doc_id": str(doc_id),
                    "content": sample.get("text", sample.get("content", "")),
                    "field": field,
                    "source_file": sample.get("source", "unknown"),
                })
                doc_id += 1

            if verbose:
                print(f"    Loaded {len(dataset)} chunks")

        except Exception as e:
            print(f"  Warning: Failed to load {field}: {e}")
            continue

    if not all_documents:
        print("Warning: No documents loaded. Creating placeholder...")
        all_documents = [{
            "doc_id": "0",
            "content": "Placeholder document for testing.",
            "field": "Test",
            "source_file": "placeholder.json",
        }]

    # Save to file
    preprocessed_dir = output_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    documents_path = preprocessed_dir / "documents.json"

    with open(documents_path, 'w') as f:
        json.dump(all_documents, f, indent=2)

    if verbose:
        print(f"  Saved {len(all_documents)} documents to {documents_path}")

    return {
        "documents_path": str(documents_path),
        "num_documents": len(all_documents),
        "fields": fields,
    }


def download_preprocessed_from_hf(
    output_dir: Path,
    repo_id: str = DEFAULT_PREPROCESSED_HF_REPO,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Download preprocessed data directly from HuggingFace dataset.

    Args:
        output_dir: Output directory for data.
        repo_id: HuggingFace dataset repository ID.
        verbose: Print progress.

    Returns:
        dict with paths to downloaded files.
    """
    from huggingface_hub import hf_hub_download

    if verbose:
        print(f"Downloading preprocessed data from {repo_id}...")

    preprocessed_dir = output_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    missing = []

    for filename in PREPROCESSED_HF_FILES:
        if verbose:
            print(f"  Downloading {filename}...")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=preprocessed_dir,
            )
            target_path = preprocessed_dir / filename
            downloaded[filename.replace("/", "_").replace(".json", "_path")] = str(target_path)
            if verbose:
                print(f"    Saved to {local_path}")
        except Exception as e:
            missing.append(filename)
            print(f"  ERROR: Failed to download {filename}: {e}")

    if missing:
        raise RuntimeError(
            "Missing required preprocessed files from "
            f"{repo_id}: {', '.join(missing)}"
        )

    if verbose:
        print(f"  Downloaded {len(downloaded)} files")

    return downloaded


def download_from_gcs(
    bucket_name: str,
    prefix: str,
    local_dir: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Download files from GCS bucket.

    Args:
        bucket_name: GCS bucket name.
        prefix: Prefix/path within the bucket.
        local_dir: Local directory to download to.
        verbose: Print progress.

    Returns:
        dict with download stats.
    """
    from google.cloud import storage

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading from gs://{bucket_name}/{prefix} to {local_dir}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    downloaded = []
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        filename = Path(blob.name).name
        local_path = local_dir / filename
        if verbose:
            print(f"  Downloading {blob.name} -> {local_path}")
        blob.download_to_filename(str(local_path))
        downloaded.append(str(local_path))

    if verbose:
        print(f"Downloaded {len(downloaded)} files")

    return {
        "local_dir": str(local_dir),
        "num_files": len(downloaded),
        "files": downloaded,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare data for inf_evolve on K8s"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/inf-evolve/data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--num-dev",
        type=int,
        default=150,
        help="Number of dev samples"
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=1170,
        help="Number of test samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--skip-supergpqa",
        action="store_true",
        help="Skip downloading SuperGPQA"
    )
    parser.add_argument(
        "--skip-textbooks",
        action="store_true",
        help="Skip downloading textbook chunks"
    )
    parser.add_argument(
        "--use-preprocessed",
        action="store_true",
        help="Download preprocessed data from HuggingFace instead of processing locally"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_PREPROCESSED_HF_REPO,
        help="HuggingFace dataset repo for preprocessed data"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress"
    )
    # GCS download options
    parser.add_argument(
        "--bucket",
        type=str,
        help="GCS bucket name (enables GCS download mode)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="GCS prefix/path within bucket"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        help="Local directory for GCS download"
    )

    args = parser.parse_args()

    # GCS download mode
    if args.bucket:
        if not args.local_dir:
            parser.error("--local-dir is required when using --bucket")
        result = download_from_gcs(
            bucket_name=args.bucket,
            prefix=args.prefix,
            local_dir=Path(args.local_dir),
            verbose=args.verbose,
        )
        print(json.dumps(result, indent=2))
        return result

    # HuggingFace download mode
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("inf_evolve Data Preparation")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    results = {}

    # Use preprocessed data from HuggingFace if requested
    if args.use_preprocessed:
        results["preprocessed"] = download_preprocessed_from_hf(
            output_dir=output_dir,
            repo_id=args.hf_repo,
            verbose=args.verbose,
        )
    else:
        if not args.skip_supergpqa:
            results["supergpqa"] = download_supergpqa(
                output_dir=output_dir,
                num_dev=args.num_dev,
                num_test=args.num_test,
                seed=args.seed,
                verbose=args.verbose,
            )

        if not args.skip_textbooks:
            results["textbooks"] = download_textbook_chunks(
                output_dir=output_dir,
                verbose=args.verbose,
            )

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    main()
