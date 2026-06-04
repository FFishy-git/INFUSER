"""Putnam Archive (kskedlaya.org) → paired-chunk JSON.

Each yearly Putnam competition contributes 12 problems (A1–A6, B1–B6).
The archive ships ``YYYY.tex`` (problems) and ``YYYYs.tex`` (solutions),
both of which use ``\\item[<label>]`` inside an ``itemize`` block. We join
on the label so each output chunk contains one problem paired with its
solution(s) — the form the document-grounded generator expects.

Usage::

    PYTHONPATH=src python -m agent.scraper.competition_putnam \\
        --start-year 1985 --end-year 2025 \\
        --output-dir .cache/data/source/textbooks/Math_Olympiad/putnam

Each year produces ``putnam_<year>.chunks.json`` with the same
``page_content`` / ``metadata`` layout used by ``preprocess/merge_textbooks.py``.
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

DEFAULT_BASE = "https://kskedlaya.org/putnam-archive"
LABELS = [f"{g}{i}" for g in "AB" for i in range(1, 7)]  # A1..A6, B1..B6

# Match ``\item[A1]`` (modern style, ≥ 2003ish) and ``\item[A--1]`` (older
# Kedlaya style with em-dash separator, used in 1995–2010 solutions).
ITEM_RE = re.compile(r"\\item\s*\[([AB])(?:--)?(\d)\]\s*", re.MULTILINE)
ITEMIZE_BEGIN_RE = re.compile(r"\\begin\{itemize\}")
ITEMIZE_END_RE = re.compile(r"\\end\{itemize\}")


def _fetch(url: str, dest: Path, force: bool = False, timeout: int = 30) -> Path:
    if dest.exists() and not force:
        logger.debug("cached: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("fetching: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "competition-chunker/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        dest.write_bytes(r.read())
    return dest


def parse_items(tex: str) -> Dict[str, str]:
    """Extract ``{label: body}`` from a Putnam .tex file.

    The body is everything between one ``\\item[<label>]`` and the next
    label (or the end of the surrounding ``itemize`` block). Whitespace
    is stripped but LaTeX markup is preserved verbatim.

    Some years (e.g. 2009 problems, 2010 solutions) nest ``\\begin{itemize}``
    blocks inside a problem body for sub-parts ``(a)``/``(b)``/``(c)``,
    so a non-greedy ``\\begin{itemize}.*?\\end{itemize}`` match would
    bail out at the first inner ``\\end{itemize}``. We bracket on the
    first ``\\begin{itemize}`` and the last ``\\end{itemize}`` instead,
    capturing the outermost block. Sub-item labels like ``(a)`` simply
    fail to match :data:`ITEM_RE` and are folded into the surrounding
    problem body.
    """
    begin = ITEMIZE_BEGIN_RE.search(tex)
    if not begin:
        raise ValueError("no \\begin{itemize} found")
    end_matches = list(ITEMIZE_END_RE.finditer(tex))
    if not end_matches:
        raise ValueError("no \\end{itemize} found")
    body = tex[begin.end():end_matches[-1].start()]

    matches = list(ITEM_RE.finditer(body))
    if not matches:
        raise ValueError("no \\item[<label>] entries found")

    out: Dict[str, str] = {}
    for i, m in enumerate(matches):
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        label = f"{m.group(1)}{m.group(2)}"
        out[label] = body[m.end():next_start].strip()
    return out


def make_chunks_for_year(
    year: int,
    base_url: str = DEFAULT_BASE,
    cache_dir: Optional[Path] = None,
) -> List[dict]:
    """Build the 12 problem+solution chunks for a single Putnam year."""
    cache_dir = cache_dir or Path(".cache/competition_sources/putnam")
    cache_dir.mkdir(parents=True, exist_ok=True)

    p_tex = _fetch(f"{base_url}/{year}.tex", cache_dir / f"{year}.tex").read_text(
        encoding="utf-8", errors="replace"
    )
    s_tex = _fetch(f"{base_url}/{year}s.tex", cache_dir / f"{year}s.tex").read_text(
        encoding="utf-8", errors="replace"
    )
    problems = parse_items(p_tex)
    solutions = parse_items(s_tex)

    chunks: List[dict] = []
    for label in LABELS:
        if label not in problems:
            logger.warning("year %d: missing problem %s", year, label)
            continue
        sol = solutions.get(label)
        if sol is None:
            logger.warning("year %d: missing solution %s", year, label)
            sol = "[solution missing in archive]"

        section = "A" if label.startswith("A") else "B"
        content = (
            f"# Putnam {year} — Problem {label}\n\n"
            f"## Problem ({section}-series)\n{problems[label]}\n\n"
            f"## Solution\n{sol}\n"
        )
        chunks.append({
            "page_content": content,
            "metadata": {
                "Header 1": f"Putnam {year}",
                "Header 2": f"Problem {label}",
                "source_pdf": f"datasets/competition_math/putnam/{year}.tex",
                "competition": "putnam",
                "year": year,
                "problem_label": label,
            },
        })
    return chunks


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Putnam Archive → .chunks.json (one per year).")
    ap.add_argument("--start-year", type=int, default=1985)
    ap.add_argument("--end-year", type=int, default=2025, help="Inclusive.")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/data/source/textbooks/Math_Olympiad/putnam"),
        help="Directory for putnam_<year>.chunks.json files.",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/competition_sources/putnam"),
        help="Where downloaded .tex files are cached.",
    )
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="Seconds to sleep between year fetches (be polite).")
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
            chunks = make_chunks_for_year(year, args.base_url, args.cache_dir)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as e:
            logger.error("year %d failed: %s", year, e)
            failures.append((year, str(e)))
            continue

        out = args.output_dir / f"putnam_{year}.chunks.json"
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
