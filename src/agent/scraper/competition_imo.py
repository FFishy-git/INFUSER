"""IMO Shortlist (imo-official.org) → paired-chunk JSON.

Each year ships a single ``IMO{YEAR}SL.pdf`` containing two parts:

1. A *Problems* section, with four categories — Algebra (``A``),
   Combinatorics (``C``), Geometry (``G``), Number Theory (``N``) — and
   problems labelled ``A1.`` ... ``A8.``, ``C1.`` ..., etc.
2. A *Solutions* section that **restates each problem verbatim** and
   then gives the official answer (``Answer:``) plus one or more
   solutions (``Solution 1.``, ``Solution 2.``) and occasional
   ``Comment.`` blocks.

We render the PDF to Markdown via the existing ``convert_pdf_to_markdown``
helper, locate the Solutions section, and slice it on per-problem
labels. Each slice already contains both the problem statement and its
solution(s), giving the document-grounded generator everything it needs
in one chunk.

Usage::

    PYTHONPATH=src python -m agent.scraper.competition_imo \\
        --start-year 2011 --end-year 2024 \\
        --output-dir .cache/data/source/textbooks/Math_Olympiad/imo_sl

Olympiad geometry problems are written to be solved from text alone
(the source PDFs do not contain diagrams), so the text-only Qwen base
solver can consume them as-is.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_BASE = "https://www.imo-official.org/problems"

CATEGORY_NAME = {
    "A": "Algebra",
    "C": "Combinatorics",
    "G": "Geometry",
    "N": "Number Theory",
}

# Header that begins the solutions half of the shortlist. Marker
# typically produces ``# Solutions`` or ``## Solutions``; we accept any
# heading level and an optional trailing ``of ...`` suffix.
SOL_SECTION_RE = re.compile(
    r"^#+\s*Solutions(?:\b[^\n]*)?$",
    re.MULTILINE | re.IGNORECASE,
)

# Per-problem label as marker renders it. Marker emits the label as a
# markdown link whose text is e.g. ``A1.`` and whose href is the page
# anchor of the matching cross-reference (problem ↔ solution). This
# pattern is specific enough to avoid the false positives that crop up
# in plain prose (e.g. "the line AA1. As X..."). If a future marker
# version stops emitting page anchors, fall back to a looser pattern.
LABEL_RE = re.compile(
    r"\[([ACGN])(\d{1,2})\.\]\(#page-\d+-\d+\)",
)

# Plain-text per-problem label as marker renders it for some years
# (e.g. 2016) where the solutions section exists but the per-problem
# anchors are not emitted as inline links. The label shows up at the
# start of a line as ``A1.`` followed by either the problem text on
# the same line or an optional H3/H4 wrapper. Anchored to line-start
# (with optional heading hashes) to avoid matching label fragments
# inside prose.
PLAIN_LABEL_LINE_RE = re.compile(
    r"^(?:#{2,4}\s+)?([ACGN])(\d{1,2})\.(?:\s|$)",
    re.MULTILINE,
)

# Legacy per-problem heading: pre-2014 shortlists ship without an explicit
# ``### Solutions`` divider, and marker emits each problem label as an
# H3/H4 heading instead of an inline link (e.g. ``#### A1``). The label
# can also appear duplicated on the same line (``#### A3 A3``) due to
# back-to-back banners in the source PDF.
LEGACY_LABEL_HEAD_RE = re.compile(
    r"^#{3,4}\s+([ACGN])(\d{1,2})\b(?:\s+[ACGN]\d{1,2})?\s*$",
    re.MULTILINE,
)

# Words that mark a chunk as the solution body (not just the problem
# statement) in the legacy heuristic. The problems half typically
# contains only the statement; the solutions half restates it and adds
# Answer / Solution / Comment / Proof.
LEGACY_SOLUTION_KEYWORD_RE = re.compile(
    r"\b(Solution|Answer|Comment|Proof)\.",
)

# Category-section markers inside the solutions half. Used as a sanity
# check (the slicer ignores them — labels alone determine boundaries).
CATEGORY_HEADER_RE = re.compile(
    r"^#+\s*(Algebra|Combinatorics|Geometry|Number Theory)\b[^\n]*$",
    re.MULTILINE,
)


def _fetch_pdf(year: int, dest: Path, base_url: str = DEFAULT_BASE,
               force: bool = False, timeout: int = 60) -> Path:
    if dest.exists() and not force:
        logger.debug("cached: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base_url}/IMO{year}SL.pdf"
    logger.info("fetching: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "competition-chunker/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        dest.write_bytes(r.read())
    return dest


def slice_solutions(md: str) -> List[Tuple[str, str]]:
    """Return ``[(label, body)]`` from a marker-rendered IMO Shortlist markdown.

    Three typesetting eras are handled:

    * **Modern (link-anchor)** — most years 2014+: a ``### Solutions``
      divider plus per-problem inline links like ``[A1.](#page-X-Y)``.
      Body for each label runs from one anchor to the next inside the
      solutions half.
    * **Modern (plain-label)** — e.g. 2016: same divider, but marker
      emits each label as a plain ``A1.`` line start instead of a link.
      Sliced via :data:`PLAIN_LABEL_LINE_RE`.
    * **Legacy** — e.g. 2011: no divider; labels render as ``#### A1``
      H3/H4 headings, sometimes duplicated. See
      :func:`_slice_solutions_legacy`.
    """
    sec = SOL_SECTION_RE.search(md)
    if sec:
        sol_md = md[sec.end():]
        anchors = [(m.start(), f"{m.group(1)}{m.group(2)}") for m in LABEL_RE.finditer(sol_md)]
        if anchors:
            out: List[Tuple[str, str]] = []
            for i, (pos, label) in enumerate(anchors):
                end = anchors[i + 1][0] if i + 1 < len(anchors) else len(sol_md)
                body = sol_md[pos:end].strip()
                out.append((label, body))
            return out

        plain_anchors = [
            (m.start(), f"{m.group(1)}{m.group(2)}")
            for m in PLAIN_LABEL_LINE_RE.finditer(sol_md)
        ]
        if plain_anchors:
            out2: List[Tuple[str, str]] = []
            for i, (pos, label) in enumerate(plain_anchors):
                end = plain_anchors[i + 1][0] if i + 1 < len(plain_anchors) else len(sol_md)
                body = sol_md[pos:end].strip()
                out2.append((label, body))
            return out2

        raise ValueError(
            "Solutions section found but no per-problem labels inside it "
            "(neither inline-link nor plain-text variant matched)"
        )

    return _slice_solutions_legacy(md)


def _slice_solutions_legacy(md: str) -> List[Tuple[str, str]]:
    """Legacy slicer for shortlists without a ``### Solutions`` divider.

    Two sub-formats are handled, both via the same longest-body
    heuristic: a label may appear in both the problems half and the
    solutions half, plus some marker-induced duplicates, and the
    solution body is reliably the longest occurrence.

    * H3/H4 heading labels (``#### A1`` style — e.g. 2011 SL)
    * Plain ``A1.`` line-start labels (e.g. 2012 SL)

    Whichever variant yields more candidate spans is used. We pick the
    longest body per label and require it to look like a solution
    (contain ``Solution.`` / ``Answer.`` / ``Comment.`` / ``Proof.``).
    Labels whose only occurrences are problem-statement banners are
    skipped with a warning — that pattern indicates marker garbled the
    corresponding solutions section.
    """
    head_spans = [
        (m.start(), f"{m.group(1)}{m.group(2)}")
        for m in LEGACY_LABEL_HEAD_RE.finditer(md)
    ]
    plain_spans = [
        (m.start(), f"{m.group(1)}{m.group(2)}")
        for m in PLAIN_LABEL_LINE_RE.finditer(md)
    ]
    spans = head_spans if len(head_spans) >= len(plain_spans) else plain_spans

    if not spans:
        raise ValueError(
            "no '### Solutions' header and no legacy-format label markers — "
            "re-check marker output or extend the legacy slicer for this year's typesetting"
        )

    starts = [s for s, _ in spans]
    by_label: Dict[str, List[Tuple[int, int]]] = {}
    for i, (start, label) in enumerate(spans):
        end = starts[i + 1] if i + 1 < len(starts) else len(md)
        by_label.setdefault(label, []).append((start, end))

    out: List[Tuple[str, str]] = []
    skipped: List[str] = []
    for label in sorted(by_label, key=lambda l: (l[0], int(l[1:]))):
        # Pick the occurrence with the longest body — solution bodies
        # are typically 1-2 orders of magnitude larger than the
        # problem-statement banner.
        start, end = max(by_label[label], key=lambda se: se[1] - se[0])
        body = md[start:end].strip()
        if not LEGACY_SOLUTION_KEYWORD_RE.search(body):
            skipped.append(label)
            continue
        out.append((label, body))

    if skipped:
        logger.warning(
            "legacy slicer skipped %d labels with no solution body (likely marker "
            "lost their solutions-half headings): %s",
            len(skipped),
            ", ".join(skipped),
        )
    if not out:
        raise ValueError(
            "legacy slicer found label headings but none had a solution body — "
            "marker output may be malformed for this year"
        )
    return out


def make_chunks_for_year(
    year: int,
    base_url: str = DEFAULT_BASE,
    pdf_cache_dir: Optional[Path] = None,
    md_cache_dir: Optional[Path] = None,
    marker_command: str = "marker_single",
) -> List[dict]:
    """Build all problem+solution chunks for one IMO Shortlist year."""
    pdf_cache_dir = pdf_cache_dir or Path(".cache/competition_sources/imo_sl")
    md_cache_dir = md_cache_dir or Path(".cache/competition_sources/imo_sl_md")

    # Lazy import so ``slice_solutions`` can be unit-tested without pulling
    # in marker / langchain.
    from .pdf_to_chunks import convert_pdf_to_markdown

    pdf = _fetch_pdf(year, pdf_cache_dir / f"IMO{year}SL.pdf", base_url)
    md_path = convert_pdf_to_markdown(
        pdf,
        output_dir=md_cache_dir / f"IMO{year}SL",
        marker_command=marker_command,
    )
    if md_path is None:
        raise RuntimeError(
            f"marker_single failed for {pdf}; install marker-pdf or set --marker-command"
        )
    md = md_path.read_text(encoding="utf-8", errors="replace")

    chunks: List[dict] = []
    for label, body in slice_solutions(md):
        cat_letter = label[0]
        cat = CATEGORY_NAME[cat_letter]
        content = f"# IMO {year} Shortlist — {cat} {label}\n\n{body}"
        chunks.append({
            "page_content": content,
            "metadata": {
                "Header 1": f"IMO {year} Shortlist",
                "Header 2": f"{cat} {label}",
                "source_pdf": f"datasets/competition_math/imo_sl/IMO{year}SL.pdf",
                "competition": "imo_shortlist",
                "year": year,
                "category": cat,
                "problem_label": label,
            },
        })
    return chunks


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IMO Shortlist → .chunks.json (one per year).")
    ap.add_argument("--start-year", type=int, default=2011)
    ap.add_argument("--end-year", type=int, default=2024, help="Inclusive.")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/data/source/textbooks/Math_Olympiad/imo_sl"),
        help="Directory for imo_sl_<year>.chunks.json files.",
    )
    ap.add_argument(
        "--pdf-cache-dir",
        type=Path,
        default=Path(".cache/competition_sources/imo_sl"),
        help="Where downloaded PDFs are cached.",
    )
    ap.add_argument(
        "--md-cache-dir",
        type=Path,
        default=Path(".cache/competition_sources/imo_sl_md"),
        help="Where marker-rendered markdown is cached (one subdir per PDF).",
    )
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--marker-command", default="marker_single")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="Seconds to sleep between year fetches.")
    ap.add_argument("--years", type=int, nargs="*",
                    help="If given, overrides --start-year/--end-year.")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    years = args.years if args.years else range(args.start_year, args.end_year + 1)

    total = 0
    failures: List[Tuple[int, str]] = []
    for year in years:
        try:
            chunks = make_chunks_for_year(
                year,
                base_url=args.base_url,
                pdf_cache_dir=args.pdf_cache_dir,
                md_cache_dir=args.md_cache_dir,
                marker_command=args.marker_command,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, RuntimeError) as e:
            logger.error("year %d failed: %s", year, e)
            failures.append((year, str(e)))
            continue

        out = args.output_dir / f"imo_sl_{year}.chunks.json"
        out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
        logger.info("year %d: wrote %d chunks → %s", year, len(chunks), out)
        total += len(chunks)
        if args.sleep:
            time.sleep(args.sleep)

    print(f"\nDone: {total} chunks across {len(list(years)) - len(failures)} years "
          f"({len(failures)} failures)")
    for y, err in failures:
        print(f"  FAIL {y}: {err}")


if __name__ == "__main__":
    main()
