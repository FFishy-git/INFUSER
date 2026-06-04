import argparse
import gzip
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.documents import Document

from .markdown_chunker import ChunkerConfig, MarkdownChunker
from .paths import CHUNKS_DIR, CONVERTED_DIR

logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(
    pdf_path: Path,
    output_dir: Optional[Path] = None,
    marker_command: str = "marker_single",
) -> Optional[Path]:
    """Convert a PDF into Markdown using marker_single."""

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    if output_dir is None:
        output_dir = CONVERTED_DIR / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_md = {p.resolve() for p in output_dir.rglob("*.md")}
    if existing_md:
        primary = max(existing_md, key=lambda p: p.stat().st_size if p.exists() else 0)
        logger.info("Markdown already exists, skipping conversion: %s", primary)
        return primary

    cmd = [
        marker_command,
        str(pdf_path),
        f"--output_dir={output_dir}",
        "--output_format=markdown",
    ]
    logger.info("Converting PDF to Markdown: %s", " ".join(cmd))

    start = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        elapsed = time.perf_counter() - start
        logger.warning("%s not found in PATH after %.2fs. Skipping conversion.", marker_command, elapsed)
        return None
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - start
        stderr = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        logger.warning("%s failed after %.2fs: %s", marker_command, elapsed, stderr)
        return None
    else:
        elapsed = time.perf_counter() - start
        logger.info("PDF conversion completed for %s in %.2fs", pdf_path, elapsed) 

    md_files = list(output_dir.rglob("*.md"))
    if not md_files:
        return None

    new_files = [p for p in md_files if p.resolve() not in existing_md]
    candidates: Sequence[Path] = new_files or md_files
    primary = max(candidates, key=lambda p: p.stat().st_size if p.exists() else 0)

    logger.info("PDF converted to Markdown: %s", primary)
    return primary


def _build_judge_client(args):
    """Instantiate a vLLM client for the LLM judge, if --judge-base-url is set."""
    base_url = getattr(args, "judge_base_url", None)
    if not base_url:
        return None
    from llm_client.vllm_client import VLLMClient

    return VLLMClient(
        base_url=base_url,
        model_name=getattr(args, "judge_model", None) or "judge-model",
        api_key=getattr(args, "judge_api_key", None) or "EMPTY",
        temperature=0.0,
        max_tokens=512,
        timeout=getattr(args, "judge_timeout", 120),
        chat_template_kwargs={"enable_thinking": False},
    )


def _build_chunker(args) -> MarkdownChunker:
    llm_client = _build_judge_client(args)
    config = ChunkerConfig(
        tokenizer_model=args.tokenizer_model,
        max_prompt_content_tokens=args.max_prompt_content_tokens,
        max_output_tokens=args.max_output_tokens,
        max_chunk_tokens=args.max_chunk_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
        use_llm_judge=bool(llm_client),
        judge_max_workers=getattr(args, "judge_workers", 8),
    )
    return MarkdownChunker(llm_client=llm_client, config=config)


def _serialize_documents(docs: List[Document]) -> List[dict]:
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]


def _determine_output_paths(pdfs: List[Path], args) -> List[Path]:
    if args.output:
        if len(pdfs) != 1:
            raise SystemExit("--output is only valid when processing a single PDF")
        return [args.output]
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return [output_dir / f"{pdf.stem}.chunks.json" for pdf in pdfs]


def _chunk_single_pdf(pdf_path: Path, chunker: MarkdownChunker) -> List[Document]:
    if not pdf_path.exists():
        logger.warning("Skipping missing file: %s", pdf_path)
        return []

    md_path = convert_pdf_to_markdown(pdf_path)
    if not md_path:
        logger.warning("PDF to Markdown conversion failed for %s", pdf_path)
        return []

    docs = chunker.chunk(md_path)
    _log_token_stats(pdf_path, docs, getattr(chunker, "_estimate_tokens", None))
    for doc in docs:
        metadata = dict(doc.metadata)
        metadata.setdefault("source_pdf", _to_relative_path(pdf_path))
        doc.metadata = metadata
    logger.info("Finished chunking %s → %s chunks", pdf_path, len(docs))
    return docs


def _convert_and_chunk_pdfs_sequential(pdf_paths: List[Path], chunker: MarkdownChunker) -> List[tuple[Path, List[Document]]]:
    results: List[tuple[Path, List[Document]]] = []
    for pdf_path in pdf_paths:
        chunks = _chunk_single_pdf(pdf_path, chunker)
        if chunks:
            results.append((pdf_path, chunks))
    if not results:
        raise SystemExit("No chunks produced. Check logs for details.")
    return results


def _process_pdf_in_thread(pdf_path: Path, args: argparse.Namespace) -> List[Document]:
    chunker = _build_chunker(args)
    return _chunk_single_pdf(pdf_path, chunker)


def _convert_and_chunk_pdfs_parallel(pdf_paths: List[Path], args: argparse.Namespace) -> List[tuple[Path, List[Document]]]:
    ordered_chunks: dict[int, tuple[Path, List[Document]]] = {}
    workers = max(1, args.workers)
    logger.info("Processing %s PDFs with %s worker(s)", len(pdf_paths), workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_process_pdf_in_thread, p, args): (idx, p)
            for idx, p in enumerate(pdf_paths)
        }
        for future in as_completed(future_map):
            idx, pdf_path = future_map[future]
            try:
                chunks = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Chunking failed for %s: %s", pdf_path, exc)
                continue
            ordered_chunks[idx] = (pdf_path, chunks)
    if not ordered_chunks:
        raise SystemExit("No chunks produced. Check logs for details.")
    ordered_results: List[tuple[Path, List[Document]]] = []
    for idx in sorted(ordered_chunks):
        pdf_path, chunks = ordered_chunks[idx]
        if chunks:
            ordered_results.append((pdf_path, chunks))
    if not ordered_results:
        raise SystemExit("No chunks produced. Check logs for details.")
    return ordered_results


def _convert_and_chunk_pdfs(pdf_paths: List[Path], args: argparse.Namespace) -> List[tuple[Path, List[Document]]]:
    if getattr(args, "workers", 1) > 1 and len(pdf_paths) > 1:
        return _convert_and_chunk_pdfs_parallel(pdf_paths, args)
    chunker = _build_chunker(args)
    return _convert_and_chunk_pdfs_sequential(pdf_paths, chunker)


def _expand_pdf_inputs(raw_paths: List[Path]) -> List[Path]:
    """Expand incoming paths, allowing directories containing PDFs."""
    expanded: List[Path] = []
    for p in raw_paths:
        if p.is_dir():
            pdfs = sorted(p.glob("*.pdf"))
            if not pdfs:
                logger.warning("Directory has no PDFs, skipping: %s", p)
            else:
                expanded.extend(pdfs)
        else:
            expanded.append(p)
    return expanded


def _write_chunks_to_disk(docs: List[Document], output_path: Path, force_gzip: bool) -> Path:
    gzip_output = force_gzip or output_path.suffix == ".gz"
    data = _serialize_documents(docs)

    if gzip_output and output_path.suffix != ".gz":
        output_path = output_path.with_suffix(output_path.suffix + ".gz")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if gzip_output:
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one or more PDFs into Markdown chunks and save as JSON."
    )
    parser.add_argument("pdfs", nargs="+", help="Path(s) to PDF files to process.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Explicit output JSON/JSON.GZ file (only valid when processing a single PDF).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHUNKS_DIR,
        help="Directory used when --output is omitted (default: datasets/chunks).",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Force gzip compression regardless of the output extension.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen3-32B",
        help="Tokenizer model name for token estimation.",
    )
    parser.add_argument(
        "--max-prompt-content-tokens",
        type=int,
        default=15384,
        help="Maximum tokens allowed when building prompts for structural detection.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16384,
        help="Maximum output tokens for LLM calls (used when an LLM is configured).",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=2048,
        help="Cap for individual chunk token length before splitting.",
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=200,
        help="Drop any chunks below this token count after processing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of PDFs to process in parallel (uses threads; default: 1).",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="vLLM server base URL for per-chunk LLM judge. If omitted, judge is disabled.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model name to send to the judge vLLM server (e.g. Qwen/Qwen3-8B).",
    )
    parser.add_argument(
        "--judge-api-key",
        default=None,
        help="API key for the judge vLLM server (default: EMPTY).",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=8,
        help="Max concurrent judge requests (default: 8).",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=120,
        help="HTTP timeout for judge calls in seconds (default: 120).",
    )
    parser.add_argument(
        "--mode",
        choices=("generic", "math_qa"),
        default="generic",
        help=(
            "generic: header+token markdown chunking (default). "
            "math_qa: route to math_textbook_qa_extractor, which emits one chunk per "
            "(problem, answer) pair from exercise + answer-key sections, drops geometry-diagram problems."
        ),
    )
    return parser


def run_conversion_and_chunking(args: argparse.Namespace) -> Union[Path, List[Path]]:
    if getattr(args, "workers", 1) < 1:
        raise SystemExit("--workers must be >= 1")
    pdf_paths = _expand_pdf_inputs([Path(p).expanduser() for p in args.pdfs])
    output_paths = _determine_output_paths(pdf_paths, args)

    if getattr(args, "mode", "generic") == "math_qa":
        from .math_textbook_qa_extractor import extract_qa_chunks_from_pdf

        per_pdf_chunks: List[tuple[Path, List[Document]]] = []
        for pdf_path in pdf_paths:
            docs, stats = extract_qa_chunks_from_pdf(pdf_path)
            logger.info(
                "math_qa[%s]: exercises=%d answer_sections=%d problems=%d answers=%d "
                "matched=%d dropped_diagram=%d unmatched=%d",
                pdf_path.name,
                stats.n_exercise_sections,
                stats.n_answer_sections,
                stats.n_problems,
                stats.n_answers,
                stats.n_matched,
                stats.n_dropped_diagram,
                stats.n_unmatched,
            )
            if docs:
                per_pdf_chunks.append((pdf_path, docs))
        if not per_pdf_chunks:
            raise SystemExit("No chunks produced. Check logs for details.")
    else:
        per_pdf_chunks = _convert_and_chunk_pdfs(pdf_paths, args)

    # Map outputs by resolved path to ensure we find the right file even if the conversion adjusted paths.
    output_by_pdf = {p.resolve(): out_path for p, out_path in zip(pdf_paths, output_paths)}

    written_paths: List[Path] = []
    for pdf_path, chunks in per_pdf_chunks:
        out_path = output_by_pdf.get(pdf_path.resolve())
        if not out_path:
            logger.warning("No output path resolved for %s; skipping write", pdf_path)
            continue
        final_path = _write_chunks_to_disk(chunks, out_path, args.gzip)
        written_paths.append(final_path)
        logger.info("Wrote %s chunks for %s to %s", len(chunks), pdf_path, final_path)

    if not written_paths:
        raise SystemExit("No outputs were written.")

    return written_paths[0] if len(written_paths) == 1 else written_paths


def _to_relative_path(path: Path) -> str:
    """Prefer a cwd-relative path; fall back to original string on failure."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except Exception:
        try:
            return os.path.relpath(path.resolve(), Path.cwd())
        except Exception:
            return str(path)


def _log_token_stats(pdf_path: Path, docs: List[Document], estimator=None) -> None:
    """Compute and log basic token statistics for the generated chunks."""
    if not docs:
        return

    def _estimate(text: str) -> int:
        try:
            if estimator:
                return int(estimator(text))
        except Exception:
            pass
        return max(1, len(text) // 4)

    token_counts: List[int] = []
    for d in docs:
        md_tokens = d.metadata.get("token_count") if isinstance(d.metadata, dict) else None
        if isinstance(md_tokens, (int, float)):
            token_counts.append(int(md_tokens))
        else:
            token_counts.append(_estimate(d.page_content))

    total_chunks = len(token_counts)
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / total_chunks if total_chunks else 0
    logger.info(
        "Token stats for %s: chunks=%s, avg=%.0f, max=%s, min=%s",
        pdf_path,
        total_chunks,
        avg_tokens,
        max(token_counts),
        min(token_counts),
    )


def main(argv: Optional[List[str]] = None) -> Union[Path, List[Path]]:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    return run_conversion_and_chunking(args)


if __name__ == "__main__":
    main()
