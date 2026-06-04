"""Math textbook → per-problem Q+A chunks (word-format problems only).

Pipeline (parallel to ``competition_putnam.py`` but for arbitrary textbooks):

    PDF
      └─► markdown        (marker_single, reused from pdf_to_chunks.py)
            └─► header walk + exercise/answer-key section detection
                  └─► per-problem split + chapter-aware match
                        └─► drop chunks containing geometry-diagram markers
                              └─► one langchain Document per (problem, answer) pair

Each output Document has the same ``{page_content, metadata}`` shape as
``src/agent/scraper/competition_putnam.py`` so ``preprocess/merge_textbooks.py``
ingests them unchanged. ``page_content`` is

    ## Problem
    {question text}

    ## Answer
    {answer text}

Usage::

    PYTHONPATH=src python -m agent.scraper.math_textbook_qa_extractor \\
        path/to/andreescu_102.pdf path/to/engel_pss.pdf \\
        --output-dir .cache/data/source/textbooks/Math

    # Pre-converted markdown (skip marker_single):
    PYTHONPATH=src python -m agent.scraper.math_textbook_qa_extractor \\
        --md path/to/book.md --book-name andreescu_102_combo \\
        --output-dir .cache/data/source/textbooks/Math
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from .paths import CHUNKS_DIR, CONVERTED_DIR

logger = logging.getLogger(__name__)


def _resolve_marker_command(marker_command: str) -> Optional[str]:
    """Find the marker_single executable. Tries PATH first, then the bin/
    directory of the active Python interpreter (which is what you typically
    want when calling this from the marker_chunk conda env)."""
    resolved = shutil.which(marker_command)
    if resolved:
        return resolved
    py_bin_dir = Path(sys.executable).parent
    candidate = py_bin_dir / marker_command
    if candidate.exists():
        return str(candidate)
    return None


def _convert_pdf_to_markdown_inline(
    pdf_path: Path,
    output_dir: Optional[Path] = None,
    marker_command: str = "marker_single",
) -> Optional[Path]:
    """Self-contained PDF→markdown via marker_single.

    Mirrors ``pdf_to_chunks.convert_pdf_to_markdown`` so this extractor does
    not need to import ``pdf_to_chunks`` (which pulls in agent.vibe_func and
    fails in minimal envs). Also resolves the marker binary against the
    active interpreter's bin/ when not on PATH.
    """
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

    resolved = _resolve_marker_command(marker_command)
    if not resolved:
        logger.warning(
            "%s not found in PATH or in %s; skipping conversion.",
            marker_command, Path(sys.executable).parent,
        )
        return None
    cmd = [resolved, str(pdf_path), f"--output_dir={output_dir}", "--output_format=markdown"]
    logger.info("Converting PDF to Markdown: %s", " ".join(cmd))
    start = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        logger.warning("%s failed: %s", marker_command, stderr)
        return None
    logger.info("PDF conversion completed in %.2fs", time.perf_counter() - start)

    md_files = list(output_dir.rglob("*.md"))
    if not md_files:
        return None
    new_files = [p for p in md_files if p.resolve() not in existing_md]
    candidates: Sequence[Path] = new_files or md_files
    primary = max(candidates, key=lambda p: p.stat().st_size if p.exists() else 0)
    logger.info("PDF converted to Markdown: %s", primary)
    return primary


# ---------------------------------------------------------------------------
# Regex catalog
# ---------------------------------------------------------------------------

HEADER_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

# marker_single frequently wraps section titles in markdown bold:
# "## **2 Introductory Problems**". Strip the outer ** before classification.
BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*$")


def _normalize_header_text(text: str) -> str:
    text = text.strip()
    m = BOLD_WRAP_RE.match(text)
    if m:
        text = m.group(1).strip()
    return text

EXERCISE_HEADER_RE = re.compile(
    r"""
    ^\s*
    (?:[§]\s*)?                                           # Evan Chen "§N.M Problems" section marker
    (?:                                                   # optional section-number prefix
        \d+(?:\.\d+)*[.):]?\s+
    )?
    (?:Part\s+[IVX0-9]+[:.]?\s+)?                         # "Part I:", "Part 1." prefix
    (?:                                                   # optional whitelisted modifier
        (?:introductory|advanced|beginning|intermediate|
           challenging|harder|easier|sample|review|warm[-\s]?up|
           proposed)
        \s+
    )?
    (?:
        exercises?
        | problems?
        | practice\s+(?:problems?|exercises?)
        | (?:exercise|problem)\s+set
        | testing\s+questions?                            # Xu Jiagu Junior series
    )
    (?:\s+and\s+(?:exercises?|problems?))?                # "Exercises and Problems"
    (?:\s*\d+(?:\.\d+)*)?                                 # optional trailing number ("Exercises 5.3")
    (?:\s*[A-Za-z]|\s*\(\s*[A-Za-z]\s*\)?)?              # trailing letter, plain or parenthesized;
                                                          # closing paren optional — scanned books
                                                          # (Xu Jiagu Senior) OCR "Testing Questions
                                                          # (A" with the ")" dropped. End-anchored
                                                          # by the \s*[:.]?\s*$ below, so no mid-line
                                                          # false positives.
    \s*[:.]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

ANSWER_HEADER_RE = re.compile(
    r"""
    ^\s*
    (?:[§]\s*)?                                           # Evan Chen "§N.M Solutions" section marker
    (?:                                                   # optional section-number/Part prefix
        \d+(?:\.\d+)*[.):]?\s+
        | Part\s+[IVX0-9]+[:.]?\s+
    )?
    (?:
        answers? (?:\s+(?:to|for|and|key)\b.*)? \s*[:.]?\s*$
        | solutions? (?:\s+(?:to|for|manual)\b.*)? \s*[:.]?\s*$
        | hints? \s+ (?:to|and|for)\b.*$
        | hints? \s*[:.]?\s*$                                  # bare "Hints" / "Hint:" section header
        | suggestions?\b.*\b(?:solutions?|answers?|hints?)\b.*$ # "Suggestions, solutions, and answers" (Skopenkov)
        | answer\s+key.*$
        | selected\s+(?:answers?|solutions?)\b.*$
        | brief\s+answers?\b.*$
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Within either an exercise OR answer-key body, sub-section "Chapter N".
# Also accepts "Lecture N" (Xu Jiagu Junior series) and "Solutions to Testing
# Questions N" (the matching answer-side per-lecture wrapper).
CHAPTER_HEADER_RE = re.compile(
    r"""
    ^\s*
    (?:
        chapter\s+(?P<chap>\d+(?:\.\d+)*)
        | lecture\s+(?P<lect>\d+(?:\.\d+)*)
        # "Solutions to Testing Questions N" — tolerate marker OCR variance:
        # Test / Testing, Question / Questions, and a parenthesized "(N)".
        | solutions?\s+to\s+test(?:ing)?\s+questions?\s+\(?(?P<sol>\d+(?:\.\d+)*)\)?
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Top-level numbered chapter header. Accepts "1. The Invariance Principle"
# AND multi-level "5.1 Distance, Rate, and Time" (AoPS section style).
NUMBERED_CHAPTER_HEADER_RE = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)*)[.):]?\s+[A-Z]",
)

# Section-label modifier — distinguishes parallel exercise/answer sections that
# restart numbering from 1 (e.g. Andreescu books: Part II Introductory Problems
# 1..52, Part III Advanced Problems 1..52, paired with the matching Solutions
# sections that also restart at 1). Without this dimension, keys collide.
SECTION_LABEL_RE = re.compile(
    r"\b(introductory|advanced|beginning|intermediate|challenging|harder|easier|sample|review|warm[-\s]?up)\b"
    # (A) / (B) / (16-A) / (16-B) — Xu Jiagu series. Closing paren is optional
    # ONLY when the label is at the end of the (trimmed) header — scanned books
    # (Xu Jiagu Senior) OCR "Testing Questions (A" with the ")" dropped. The
    # `(?=\s*$)` end-anchor prevents matching a stray "(a word..." mid-text.
    r"|\((?:\d+-)?([A-Ba-b])(?:\)|(?=\s*$))",
    re.IGNORECASE,
)


def _section_label(header_text: str) -> Optional[str]:
    m = SECTION_LABEL_RE.search(header_text)
    if not m:
        return None
    label = m.group(1) or m.group(2)
    return label.lower().replace(" ", "-") if label else None

# Plain numbered list item: "1.", "5.12.", "**5.1.2.**", "1)", "- 1.", "* 1.".
# Anchored at column 0 so indented sub-items (e.g. "\t- (1) Let n...") never
# start a new top-level problem.
ITEM_RE = re.compile(
    r"""
    ^(?:[-*+]\s+)?                      # optional markdown unordered-list bullet
    (?:\*\*)?                           # optional markdown bold open
    (?:<sup>)?                          # optional HTML superscript open (Xu Jiagu markup)
    (?P<num>\d+(?:\.\d+)*)              # 5 or 5.12 or 5.1.2
    (?:</sup>)?                         # optional HTML superscript close
    [.)\]]                              # terminator
    (?:\*\*)?                           # optional bold close
    (?:\s|$)                            # whitespace or end-of-line
    """,
    re.VERBOSE,
)

# Explicit "Problem N.M" / "Exercise N.M" / "Solution N.M" marker. AoPS-style
# textbooks and solutions manuals frequently use this form (often bolded) to
# tag each problem.
EXPLICIT_ITEM_RE = re.compile(
    r"""
    ^(?:[-*+]\s+)?                      # optional markdown list bullet
    (?:\*\*)?                           # optional bold open
    (?:Problem|Exercise|Solution)       # explicit label
    \s+
    (?P<num>\d+(?:\.\d+)*)              # number
    [.)\]:]?                            # optional terminator
    (?:\*\*)?                           # optional bold close
    (?:\s+|$)                           # whitespace or end-of-line
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Order matters: explicit form is more specific, try it first.
ITEM_PATTERNS: Tuple[re.Pattern, ...] = (EXPLICIT_ITEM_RE, ITEM_RE)

# Inline problem marker: a MULTI-PART dotted number (>= 2 levels, e.g. 1.1.1,
# 8.2.14), optionally bold, at line start. Used by books that number problems
# inline within chapter prose WITHOUT a dedicated "Exercises"/"Problems" header
# (Skopenkov "Mathematics via Problems", Andreescu "Problems in Real Analysis",
# Zeitz "Art & Craft"). The >= 2-level requirement is the key discriminator:
# bare "1." / "2." list items, "(1)" equation refs, and "1. CHAPTER" page-header
# artifacts are all single-level and won't match, so this stays conservative.
# The trailing lookahead keeps the problem text out of the match so the marker
# can be cleanly stripped.
INLINE_PROBLEM_RE = re.compile(
    r"""
    ^[ \t]{0,3}
    (?:\*\*|__)?                       # optional bold open
    (?:(?:Example|Problem|Exercise)s?\s+)?  # optional worked-example/problem label
                                       #   prefix (Stevens "Olympiad Number Theory":
                                       #   "Example 1.1.1. Find gcd(110, 490)."). Still
                                       #   gated by the multi-part-dotted <num> + the
                                       #   per-section MIN_INLINE_PROBLEMS guard.
    (?P<num>\d+\.\d+(?:\.\d+)*)        # multi-part: at least N.M
    \.?                                # optional trailing dot
    (?:\*\*|__)?                       # optional bold close
    [ \t]+
    (?=\S)                             # problem text follows (not consumed)
    """,
    re.VERBOSE,
)
# SINGLE-level inline problem marker ("1.", "7.", "Problem 3."). Far less
# specific than the multi-part rule, so it is used ONLY as a last-resort
# fallback inside `_inline_problems_from_section` — when the multi-part scan
# found ~nothing AND the section clearly is a worked-problem set (>=2 inline
# Solution markers). Recovers single-level competition handouts (Po-Shen Loh
# "Telescoping…", Mildorf "Examples") without loosening the global rule:
# books whose multi-part scan succeeds (ONT's "Example N.M.K", Skopenkov,
# Real Analysis) never reach this branch, so they are byte-identical.
INLINE_PROBLEM_SINGLE_RE = re.compile(
    r"""
    ^[ \t]{0,3}
    (?:\*\*|__)?
    (?:(?:Example|Problem|Exercise)s?\s+)?
    (?P<num>\d{1,3})
    \.
    (?:\*\*|__)?
    [ \t]+
    (?=\S)
    """,
    re.VERBOSE,
)

# A normal section needs at least this many inline multi-part items before we
# treat it as problem-bearing (filters front-matter / prose noise).
MIN_INLINE_PROBLEMS = 3

# Header-delimited Problem/Solution pairs. Some Andreescu/AMT books (e.g. the
# "101 Problems in ..." / "10x ... Problems" Enrichment series) render a
# *self-contained* solutions section where each problem is RESTATED under its
# own header and immediately followed by its solution header(s):
#     #### Problem 4 [AIME 1997]
#     <problem restated>
#     #### Solution 4, Alternative 1
#     <solution>
#     #### Solution 4, Alternative 2
#     <alternative solution>
# _split_numbered_items skips header lines by design, so this layout yields
# zero pairs through the normal numbered-list path. These match the *whole*
# header text (after #-strip + bold-unwrap) and tolerate a trailing source tag
# ("[AIME 1997]"), an "Alternative N" qualifier, or stray bled-in problem text.
HEADER_PROBLEM_RE = re.compile(r"^problems?\s+(?P<num>\d+(?:\.\d+)*)\b", re.IGNORECASE)
HEADER_ANSWER_RE = re.compile(
    r"^(?:solutions?|answers?)\s+(?P<num>\d+(?:\.\d+)*)\b", re.IGNORECASE
)
# An answer_key section must exhibit at least this many problem-nums that also
# have a matching solution header before the header-pair harvester engages.
# This structural gate keeps numbered-list answer keys (104 NT, Engel, Xu Jiagu)
# completely untouched — their solution sides have no Problem/Solution headers.
MIN_HEADER_PAIRS = 3

# --- Competition-paper layout (Olympiad collections) -----------------------
# Books like "Mathematical Olympiad in China — Problems & Solutions" have NO
# Exercises/Problems or Solutions headers. They are organised
# competition -> year -> "Part I/II" / "First/Second Day" leaf sections, each
# a flat run of sequentially-numbered problems with the worked solution INLINE
# right after each one ("... Solution. ...", "Solution 1. ... Solution 2. ...").
# Problem-start marker: line start, optional list bullet, a 1-3 digit number,
# optional ")"/"." terminator, then the statement. Years ("2010(Fujian)") and
# in-solution enumerations ("(1) ...", "(2) ...") are rejected by the
# sequential-numbering state machine in _competition_chunks (not by this
# regex), so this stays permissive.
COMP_PROBLEM_RE = re.compile(
    r"^[ \t]{0,3}(?:[-*+][ \t]+)?(?P<num>\d{1,3})[ \t]*[).\]]?[ \t]+(?=\D)"
)
# Inline solution/proof marker — may sit mid-line (marker glues it to the end
# of the problem statement: "... is Solution. It is easy ..."). Requires the
# word boundary + a "." and following content, optionally a "Solution 2"-style
# index. Conservative enough that problem prose ("the solution set") rarely
# trips it; the gpt-5-mini audit catches the residue.
COMP_SOLUTION_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))\*{0,2}(?:Solution|Proof)s?\s*\d{0,2}\s*\*{0,2}\s*[.:]\s*(?=\S)",
    re.IGNORECASE,
)
# A leaf section must yield at least this many sequential problems (each with
# an inline solution) before the competition extractor accepts it. Strong
# structural gate → zero regression: no currently-parsing book has flat
# sequentially-numbered sections with interleaved inline Solution markers.
MIN_COMPETITION_PROBLEMS = 3

# A line that begins an inline solution/proof immediately after a problem
# statement (Andreescu "Problems in Real Analysis", many solutions manuals:
# "1.2.5. <question> **Solution.** <answer>"). When a harvested problem body
# contains such a line, the text before it is the question and the text from
# it onward is the answer — a self-contained Q+A pair with no separate answer
# section needed. Conservative: must be a line that is *only* the marker (plus
# optional bold), so prose mentioning "solution" mid-sentence is not split.
INLINE_SOLUTION_RE = re.compile(
    r"""
    ^[ \t]{0,3}
    (?:\*\*|__)?
    (?:Solution|Proof|Answer)s?
    (?:\s*\([^)]{0,20}\))?              # optional "(a)" / "(First)" qualifier
    \s*[.:]?\s*
    (?:\*\*|__)?
    \s*$                               # marker occupies the whole line
    | ^[ \t]{0,3}(?:\*\*|__)?(?:Solution|Proof|Answer)s?\b[.:]?(?:\*\*|__)?[ \t]+(?=\S)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _split_inline_solution(body: str) -> Tuple[str, Optional[str]]:
    """Split a problem body at the first inline Solution/Proof/Answer marker.
    Returns ``(question, answer)``; ``answer`` is ``None`` when no marker is
    present (the common case for books with a separate answer section, so this
    is a no-op there — zero regression)."""
    lines = body.splitlines()
    in_fence = False
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if i == 0:
            continue  # never treat the problem's own first line as the answer
        if INLINE_SOLUTION_RE.match(line):
            q = "\n".join(lines[:i]).strip()
            # strip the marker token off the first answer line
            a_first = INLINE_SOLUTION_RE.sub("", lines[i], count=1)
            a = "\n".join([a_first] + lines[i + 1:]).strip()
            if q and a:
                return q, a
            return body, None
    return body, None

# Geometric-diagram indicators — drop chunks containing any of these.
DIAGRAM_REGEXES: Tuple[re.Pattern, ...] = (
    re.compile(r"\[asy\]", re.IGNORECASE),
    re.compile(r"\[/asy\]", re.IGNORECASE),
    re.compile(r"\\begin\{asy\}", re.IGNORECASE),
    re.compile(r"\\begin\{tikzpicture\}"),
    re.compile(r"\\includegraphics\b"),
    re.compile(r"!\[[^\]]*\]\([^)]+\)"),   # markdown image
)

ASYMPTOTE_KEYWORDS = ("unitsize(", "draw(", "dot(", "filldraw(", "pair ", "label(", "MP(")

# Fence delimiters we want to skip over when scanning for headers/items
FENCE_RE = re.compile(r"^```")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class _Header:
    line_idx: int
    level: int
    text: str


@dataclass
class _Section:
    """A markdown section starting at a header, ending before the next header
    of same-or-higher level."""
    kind: str                 # "exercise" | "answer_key" | "normal"
    level: int
    header_text: str
    chapter: Optional[str]    # numeric label of the nearest chapter ancestor
    section_label: Optional[str]  # "introductory" / "advanced" / etc. — disambiguates
                                  # parallel exercise/answer sections that restart numbering.
    body: str
    start_line: int
    flat_wrapper: bool = False  # answer-key wrapper that absorbed same-level chapter
                                # siblings; numbered-item parsing should also run in
                                # chapter-agnostic mode for offset-numbered books


# ---------------------------------------------------------------------------
# Geometry filter
# ---------------------------------------------------------------------------

def has_diagram(text: str) -> bool:
    """True if the text references a non-textual figure (Asymptote, image, …)."""
    if not text:
        return False
    for pat in DIAGRAM_REGEXES:
        if pat.search(text):
            return True
    lower = text.lower()
    hits = sum(1 for kw in ASYMPTOTE_KEYWORDS if kw in lower)
    if hits >= 2:
        return True
    return False


# ---------------------------------------------------------------------------
# Non-informative-answer filter
# ---------------------------------------------------------------------------

_CMP_STRIP_RE = re.compile(r"</?(?:sup|sub|b|i|em|strong)>|[*_`#]")
_CMP_WS_RE = re.compile(r"\s+")


def _normalize_for_compare(s: str) -> str:
    """Lowercase, drop markdown emphasis / HTML sup-sub tags, collapse
    whitespace — so a problem and its restatement compare equal despite
    cosmetic marker_single formatting differences."""
    s = _CMP_STRIP_RE.sub("", s)
    s = _CMP_WS_RE.sub(" ", s)
    return s.strip().lower()


def answer_is_noninformative(prob_body: str, ans_body: str) -> bool:
    """True when the 'answer' carries no solution content beyond the problem
    itself — the scraper grabbed a restated problem instead of its solution.

    Common in books that restate each problem before solving it (e.g.
    Skopenkov "Suggestions, solutions, and answers" sections, where ~24% of
    raw pairs were Q==A). Conservative by construction: fires only on
    (near-)identity or full containment, never on short-but-genuine answers
    like "*Answers*: (a) Yes; (b) No." (a verdict is not a substring of the
    question). Verified zero-drop on 104 NT (95) and Engel (702)."""
    nq = _normalize_for_compare(prob_body)
    na = _normalize_for_compare(ans_body)
    if not na:
        return True
    if na == nq:
        return True
    # answer text is wholly contained in the problem → adds nothing
    if len(na) >= 12 and na in nq:
        return True
    # answer is the problem text plus negligible extra → restatement, no solution
    if nq and nq in na and (len(na) - len(nq)) < 30:
        return True
    return False


# ---------------------------------------------------------------------------
# Header walk
# ---------------------------------------------------------------------------

def _collect_headers(lines: List[str]) -> List[_Header]:
    headers: List[_Header] = []
    in_fence = False
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADER_LINE_RE.match(line)
        if not m:
            continue
        headers.append(_Header(line_idx=i, level=len(m.group(1)), text=_normalize_header_text(m.group(2))))
    return headers


def _nearest_chapter(prior: List[_Header]) -> Optional[str]:
    """Walk backward through earlier headers; return the numeric ID of the
    nearest one that looks like a chapter. Exercise and answer-key sections
    are skipped so they don't accidentally become chapter ancestors for the
    sections that follow them (which would scramble the (chapter, problem_no)
    matching across the textbook ↔ solutions sides)."""
    for h in reversed(prior):
        if EXERCISE_HEADER_RE.match(h.text) or ANSWER_HEADER_RE.match(h.text):
            continue
        m = CHAPTER_HEADER_RE.match(h.text)
        if m:
            return m.group("chap") or m.group("lect") or m.group("sol")
        m = NUMBERED_CHAPTER_HEADER_RE.match(h.text)
        if m and h.level <= 2:  # only treat top-level numbered headers as chapters
            return m.group("num")
    return None


def _classify_header(text: str) -> str:
    if EXERCISE_HEADER_RE.match(text):
        return "exercise"
    if ANSWER_HEADER_RE.match(text):
        return "answer_key"
    return "normal"


# marker_single emits empty page-anchor spans like
# ``<span id="page-16-0"></span>`` mid-line and inside headers. They carry no
# content but break header classification (``# <span...>**Chapter 1**``) and
# item splitting (``- <span...>8. Let...``). Strip every span tag globally
# before parsing — math uses <sup>/<sub>, never <span>, so this is lossless.
_SPAN_ARTIFACT_RE = re.compile(r"</?span[^>]*>")

# marker also renders running PAGE-HEADER artifacts as bold headings, e.g.
# ``#### **4. Solutions to Introductory Problems 85**`` — "<chap#>. <chapter
# title> <page no>". Each one matches ANSWER_HEADER_RE / EXERCISE_HEADER_RE and
# spawns a bogus section that fragments the real solution bodies (Andreescu
# "10X" books: 103 Trig had answer_sections=41, only 53/100 matched). The
# signature is specific: bold-wrapped (`**...**`) AND a leading "<n>." (with the
# dot) AND a trailing bare page number. Real section headers are "# 2
# Introductory Problems" (number, NO dot, no trailing page no.) and numbered
# problem items (``#### 19. Let ABC...``) are not bold-wrapped, so neither can
# match. Strip the whole artifact line globally before parsing — lossless, math
# never uses this heading form.
_PAGE_HEADER_ARTIFACT_RE = re.compile(
    r"^#{1,6}[ \t]+\*\*\d{1,3}\.[ \t]+.*?[A-Za-z].*?[ \t]+\d{1,4}\*\*[ \t]*$",
    re.MULTILINE,
)


# marker also over-promotes a numbered list ITEM to a deep heading, e.g.
# ``#### 22. Let ABC be a triangle...`` while its siblings stay plain
# ``21. Let ABC...``. Such a promoted line is then BOTH (a) skipped by
# ``_split_numbered_items`` (which ignores header lines) AND (b) misread by
# ``_answers_from_section`` as a numbered *chapter* header (its
# NUMBERED_CHAPTER_HEADER_RE branch has no level guard) → chapter reassigned →
# the (chapter,label,num) match breaks (103 Trig: ~29 intro/advanced solutions
# lost). Demote H3+ headings whose text is a SINGLE-level numbered item back to
# a plain item line. Tight discriminators keep it zero-regression:
#   • only H3..H6 — real chapter/section headers in these books are H1/H2.
#   • single-level ``\d{1,3}[.)\]]`` then whitespace+text: multi-level
#     ``5.1 Distance`` has no terminator+space after "5." so AoPS section
#     headers are NOT demoted.
#   • a DIGIT must follow the ``###``: word-prefixed ``#### Problem N`` /
#     ``#### Solution N`` (the concurrent header-pair family, 101-Algebra) and
#     bold ``#### **4. ... 85**`` page-headers start with a letter / ``*`` and
#     are untouched — so this is orthogonal to _header_pair_qa_from_section.
_PROMOTED_ITEM_HEADER_RE = re.compile(
    r"^#{3,6}[ \t]+(\d{1,3}[.)\]][ \t]+\S.*)$",
    re.MULTILINE,
)


def _strip_marker_artifacts(md_text: str) -> str:
    md_text = _SPAN_ARTIFACT_RE.sub("", md_text)
    md_text = _PAGE_HEADER_ARTIFACT_RE.sub("", md_text)
    md_text = _PROMOTED_ITEM_HEADER_RE.sub(r"\1", md_text)
    return md_text


def _walk_sections(md_text: str) -> List[_Section]:
    md_text = _strip_marker_artifacts(md_text)
    lines = md_text.splitlines()
    headers = _collect_headers(lines)
    classifications = [_classify_header(h.text) for h in headers]
    sections: List[_Section] = []
    for idx, h in enumerate(headers):
        end_line = len(lines)
        flat_wrapper = False
        for j in range(idx + 1, len(headers)):
            if headers[j].level <= h.level:
                # If this is an answer-key wrapper followed by same-level
                # chapter-shaped siblings (e.g. Andreescu "Solutions to Proposed
                # Problems" wrapping H1 chapter solutions), absorb those into
                # our body so _answers_from_section sees them as sub-sections.
                if (classifications[idx] == "answer_key"
                        and headers[j].level == h.level
                        and classifications[j] == "normal"
                        and (CHAPTER_HEADER_RE.match(headers[j].text)
                             or NUMBERED_CHAPTER_HEADER_RE.match(headers[j].text))):
                    flat_wrapper = True
                    continue
                end_line = headers[j].line_idx
                break
        body = "\n".join(lines[h.line_idx + 1 : end_line]).strip()
        # When an answer wrapper's OWN header self-identifies its lecture
        # ("Solutions to Testing Questions N"), use N as the chapter. Otherwise
        # _nearest_chapter would skip every "Solutions to..." header and assign
        # the last real lecture before the solutions region to *all* wrappers,
        # collapsing them to one (chapter,num) namespace so dedup drops most
        # answers (Xu Jiagu Vol 1: 24 wrappers → only ~62/161 answers survived).
        own = CHAPTER_HEADER_RE.match(h.text)
        if classifications[idx] == "answer_key" and own and own.group("sol"):
            chapter = own.group("sol")
        else:
            chapter = _nearest_chapter(headers[:idx])
        sections.append(_Section(
            kind=classifications[idx],
            level=h.level,
            header_text=h.text,
            chapter=chapter,
            section_label=_section_label(h.text),
            body=body,
            start_line=h.line_idx,
            flat_wrapper=flat_wrapper,
        ))
    return sections


# ---------------------------------------------------------------------------
# Numbered-item splitter
# ---------------------------------------------------------------------------

# marker frequently runs consecutive short problems together on ONE line
# (Engel: "1. … replace any two integers by their difference. … after 4n-2
# steps. 2. Start with the set {3,4,12} …"), defeating the line-anchored item
# splitter: problems 2.. get swallowed into problem 1's body, so the merged
# body mis-matches its single solution — the DOMINANT Engel cross-match cause
# (~150 pairs; diagnostic 2026-05-17). Within exercise/answer section bodies
# ONLY (`_split_numbered_items` is used nowhere else), insert a line break
# before an inline "<N>. <Capital/(>" that directly follows sentence-ending
# punctuation. Tight signature (sentence-ender [.?!)\]] + space + 1-3 digits +
# "." + space + capital/paren) keeps decimals ("x = 5. Then": preceded by a
# digit/`=`, not a sentence-ender) and comma-lists ("n = 1, 2, 3. Then":
# preceded by ",") from splitting. Regression-verified on the 5-book invariant.
_UNGLUE_INLINE_ITEM_RE = re.compile(r"(?<=[.?!\)\]])[ \t]+(\d{1,3}\.[ \t]+[A-Z(])")


def _unglue_inline_items(text: str) -> str:
    return _UNGLUE_INLINE_ITEM_RE.sub(lambda m: "\n" + m.group(1), text)


def _split_numbered_items(text: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(num, body)`` for each numbered list item in ``text``.

    Accepts either a plain numbered list (``1.``, ``5.12.``, ``**5.1.2.**``)
    or an explicit ``Problem N`` / ``Exercise N`` / ``Solution N`` marker.
    Matches at line start so in-prose numerals like ``1.5`` inside a problem
    body never start a new item; an inline run of glued problems is first
    un-glued onto separate lines (see ``_unglue_inline_items``).
    """
    if not text:
        return
    text = _unglue_inline_items(text)
    lines = text.splitlines()
    starts: List[Tuple[int, str, re.Pattern]] = []
    in_fence = False
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or HEADER_LINE_RE.match(line):
            continue
        for pat in ITEM_PATTERNS:
            m = pat.match(line)
            if m:
                starts.append((i, m.group("num"), pat))
                break

    for k, (line_idx, num, pat) in enumerate(starts):
        end_line = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        first = lines[line_idx]
        first_stripped = pat.sub("", first, count=1)
        body_lines = [first_stripped] + lines[line_idx + 1 : end_line]
        body = "\n".join(body_lines).strip()
        if body:
            yield num, body


# ---------------------------------------------------------------------------
# Per-section parsing
# ---------------------------------------------------------------------------

def _problems_from_section(section: _Section) -> List[Tuple[Optional[str], Optional[str], str, str]]:
    """Yield ``(chapter, section_label, num, body)`` per problem."""
    return [(section.chapter, section.section_label, num, body)
            for num, body in _split_numbered_items(section.body)]


def _inline_problems_from_section(
    section: _Section,
) -> List[Tuple[Optional[str], Optional[str], str, str]]:
    """Harvest problems numbered inline in chapter prose (e.g. ``**1.1.1.**``)
    when the book has NO Exercises/Problems header. Conservative: only
    multi-part dotted IDs qualify, and a section must yield at least
    ``MIN_INLINE_PROBLEMS`` of them or it is ignored (front-matter / noise)."""
    lines = section.body.splitlines()
    starts: List[Tuple[int, str]] = []
    in_fence = False
    answer_region_start: Optional[int] = None
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        hm = HEADER_LINE_RE.match(line)
        if hm:
            # Books like Skopenkov interleave a "Suggestions, solutions, and
            # answers" / "Hints" sub-header INSIDE each chapter body. Everything
            # from that sub-header onward is solution text (parsed separately as
            # its own answer_key section) — stop harvesting problems here so we
            # don't capture restated-problem text and pair it with itself.
            if ANSWER_HEADER_RE.match(_normalize_header_text(hm.group(2))):
                answer_region_start = i
                break
            continue
        m = INLINE_PROBLEM_RE.match(line)
        if m:
            starts.append((i, m.group("num")))
    marker_re = INLINE_PROBLEM_RE
    if len(starts) < MIN_INLINE_PROBLEMS:
        # Last-resort single-level fallback: the strict multi-part scan found
        # ~nothing. Only engage if the section is clearly a worked-problem set
        # (>=2 inline Solution/Proof markers) — single-level handouts like
        # Po-Shen Loh "Telescoping" or Mildorf "Examples" number problems
        # "1." "2." with an inline "Solution:" each. The Solution-marker gate
        # keeps prose "1. … 2. …" lists out. Books whose multi-part scan
        # succeeds never reach here (ONT/Skopenkov/Real-Analysis unchanged).
        sol_markers = sum(1 for ln in lines if INLINE_SOLUTION_RE.match(ln))
        if sol_markers >= 2:
            rescan: List[Tuple[int, str]] = []
            in_fence = False
            answer_region_start = None
            for i, line in enumerate(lines):
                if FENCE_RE.match(line):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                hm = HEADER_LINE_RE.match(line)
                if hm:
                    if ANSWER_HEADER_RE.match(_normalize_header_text(hm.group(2))):
                        answer_region_start = i
                        break
                    continue
                m = INLINE_PROBLEM_SINGLE_RE.match(line)
                if m:
                    rescan.append((i, m.group("num")))
            if len(rescan) >= MIN_INLINE_PROBLEMS:
                starts = rescan
                marker_re = INLINE_PROBLEM_SINGLE_RE
    if len(starts) < MIN_INLINE_PROBLEMS:
        return []
    # Cap problem bodies at the answer region so a trailing problem doesn't
    # swallow the solutions block.
    body_limit = answer_region_start if answer_region_start is not None else len(lines)
    out: List[Tuple[Optional[str], Optional[str], str, str]] = []
    for k, (li, num) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else body_limit
        first = marker_re.sub("", lines[li], count=1)
        body = "\n".join([first] + lines[li + 1:end]).strip()
        if body:
            out.append((section.chapter, section.section_label, num, body))
    return out


def _header_pair_qa_from_section(
    section: _Section,
) -> List[Tuple[str, Optional[str], str, str]]:
    """Harvest a self-contained solutions section that restates each problem
    under its own header and follows it with Solution header(s).

    Returns ``(kind, section_label, num, body)`` where ``kind`` is
    ``"problem"`` or ``"answer"``. Multiple ``Solution N, Alternative K``
    headers for the same N are yielded as separate ``answer`` entries (in
    order) so the caller can concatenate them. ``section_label`` is inherited
    from the outer answer section ("Solutions to Introductory Problems" ->
    ``introductory``) so Introductory/Advanced nums that both restart at 1 do
    not collide.

    Conservative / self-gating: returns ``[]`` unless the section has at least
    ``MIN_HEADER_PAIRS`` problem-nums that ALSO have a matching solution
    header. Numbered-list answer keys have no such headers, so this is a no-op
    for every book that already parses (zero regression)."""
    lines = section.body.splitlines()
    label = section.section_label
    marks: List[Tuple[str, str, int]] = []  # (kind, num, header_line_idx)
    in_fence = False
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        hm = HEADER_LINE_RE.match(line)
        if not hm:
            continue
        htext = _normalize_header_text(hm.group(2))
        pm = HEADER_PROBLEM_RE.match(htext)
        if pm:
            marks.append(("problem", pm.group("num"), i))
            continue
        am = HEADER_ANSWER_RE.match(htext)
        if am:
            marks.append(("answer", am.group("num"), i))
    prob_nums = {n for k, n, _ in marks if k == "problem"}
    ans_nums = {n for k, n, _ in marks if k == "answer"}
    if len(prob_nums & ans_nums) < MIN_HEADER_PAIRS:
        return []
    out: List[Tuple[str, Optional[str], str, str]] = []
    for j, (kind, num, li) in enumerate(marks):
        end = marks[j + 1][2] if j + 1 < len(marks) else len(lines)
        body = "\n".join(lines[li + 1 : end]).strip()
        if body:
            out.append((kind, label, num, body))
    return out


def _answers_from_section(section: _Section) -> List[Tuple[Optional[str], Optional[str], str, str]]:
    """Yield ``(chapter, section_label, num, body)`` per answer. Answer-key
    body may contain its own ``# Chapter N`` sub-headers grouping answers;
    split on those before numbered-item parsing so chapters stay aligned with
    the problem side. ``section_label`` is inherited from the outer answer
    section header (so "Solutions to Introductory Problems" propagates
    ``introductory`` to every answer it contains)."""
    body_lines = section.body.splitlines()
    blocks: List[Tuple[Optional[str], Optional[str], str]] = []
    current_chapter = section.chapter
    current_label = section.section_label
    buffer: List[str] = []
    in_fence = False
    for line in body_lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if in_fence:
            buffer.append(line)
            continue
        m = HEADER_LINE_RE.match(line)
        if m:
            sub_text = _normalize_header_text(m.group(2))
            chap_m = CHAPTER_HEADER_RE.match(sub_text)
            num_m = None if chap_m else NUMBERED_CHAPTER_HEADER_RE.match(sub_text)
            if chap_m or num_m:
                if buffer:
                    blocks.append((current_chapter, current_label, "\n".join(buffer)))
                    buffer = []
                if chap_m:
                    current_chapter = chap_m.group("chap") or chap_m.group("lect") or chap_m.group("sol")
                else:
                    current_chapter = num_m.group("num")
                current_label = section.section_label  # reset to outer when chapter changes
                continue
            # Sub-section label header (e.g. "Testing Questions (16-B)" splits
            # an answer wrapper into (A)/(B) blocks within a single lecture).
            sub_label = _section_label(sub_text)
            if sub_label is not None and sub_label != current_label:
                if buffer:
                    blocks.append((current_chapter, current_label, "\n".join(buffer)))
                    buffer = []
                current_label = sub_label
                continue
        buffer.append(line)
    if buffer:
        blocks.append((current_chapter, current_label, "\n".join(buffer)))

    out: List[Tuple[Optional[str], Optional[str], str, str]] = []
    for chap, lab, body in blocks:
        for num, ans in _split_numbered_items(body):
            out.append((chap, lab, num, ans))

    # For answer wrappers that absorbed offset-numbered chapter siblings
    # (e.g. andrica solutions where the wrapper-side "# 12 Divisibility" doesn't
    # match problem-side chapter "1.1"), also emit chapter-agnostic entries
    # keyed by chapter=None — but ONLY for nums that are unique within the
    # wrapper body. Books with bare-numbered problems (1, 2, 3 in each section)
    # would otherwise have the first-occurrence answer mis-matched to every
    # other problem with the same num. _lookup_answer falls back to
    # (None, None, num), so this catches books whose problem nums are
    # book-unique (e.g. "1.1.10", "1.1.11") without polluting bare-num books.
    if section.flat_wrapper:
        flat_items = list(_split_numbered_items(section.body))
        num_counts: dict[str, int] = {}
        for num, _ in flat_items:
            num_counts[num] = num_counts.get(num, 0) + 1
        for num, ans in flat_items:
            # Multi-part nums (e.g. "1.1.10") are book-unique by construction;
            # bare nums (e.g. "2") are likely enumeration markers inside a
            # solution body, not actual problems. Restrict flat fallback to
            # multi-part nums to avoid cross-matching bare-num problems with
            # random "N." enumeration fragments.
            if num_counts[num] == 1 and "." in num:
                out.append((None, section.section_label, num, ans))
    return out


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------

@dataclass
class ExtractionStats:
    n_exercise_sections: int = 0
    n_answer_sections: int = 0
    n_problems: int = 0
    n_answers: int = 0
    n_matched: int = 0
    n_dropped_diagram: int = 0
    n_dropped_noninformative: int = 0
    n_unmatched: int = 0
    n_inline_problems: int = 0
    n_competition_problems: int = 0


_COMP_GROUP_RE = re.compile(
    r"\b(Olympiad|Competition|Selection\s*Test|Team\s*Selection|Tournament)\b",
    re.IGNORECASE,
)
_COMP_YEAR_RE = re.compile(r"^\*{0,2}\s*((?:19|20)\d\d)\b\s*(\([^)]*\))?")
_COMP_PARTDAY_RE = re.compile(
    r"^\*{0,2}\s*(Part\s+[IVX0-9]+\b|(?:First|Second|Third|Fourth)\s*Day\b)",
    re.IGNORECASE,
)
_COMP_SOLHDR_RE = re.compile(r"^\s*(?:Solution|Proof)s?\b", re.IGNORECASE)


def _competition_emit(
    region_lines: List[str],
    label: Optional[str],
    book_name: str,
    source_pdf: str,
    seen: set,
    stats: ExtractionStats,
) -> List[Document]:
    """Run the sequential-problem + inline-solution state machine over one
    region (text between two structural headers, numbering restarts here)."""
    out: List[Document] = []
    cands: List[Tuple[int, int]] = []
    in_fence = False
    for i, l in enumerate(region_lines):
        if FENCE_RE.match(l):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = COMP_PROBLEM_RE.match(l)
        if m:
            cands.append((i, int(m.group("num"))))
    if len(cands) < MIN_COMPETITION_PROBLEMS:
        return out
    # longest strictly-increasing small-gap run (try every start) — rejects
    # in-solution enumeration / equation-tag numbers, which are not monotone.
    best: List[Tuple[int, int]] = []
    for s in range(len(cands)):
        run = [cands[s]]
        for (li, n) in cands[s + 1:]:
            if n > run[-1][1] and n - run[-1][1] <= 3:
                run.append((li, n))
        if len(run) > len(best):
            best = run
    if len(best) < MIN_COMPETITION_PROBLEMS:
        return out
    ctx = label or book_name
    for k, (li, num) in enumerate(best):
        end = best[k + 1][0] if k + 1 < len(best) else len(region_lines)
        span = "\n".join(region_lines[li:end])
        span = COMP_PROBLEM_RE.sub("", span, count=1)
        sm = COMP_SOLUTION_RE.search(span)
        if not sm:
            continue
        q = span[: sm.start()].strip()
        a = span[sm.end():].strip()
        if len(q) < 15 or len(a) < 15:
            continue
        if has_diagram(q) or has_diagram(a):
            stats.n_dropped_diagram += 1
            continue
        if answer_is_noninformative(q, a):
            stats.n_dropped_noninformative += 1
            continue
        dedup_key = (ctx, num)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        out.append(Document(
            page_content=f"## Problem\n{q}\n\n## Answer\n{a}\n",
            metadata={
                "Header 1": f"{book_name} — {ctx}" if ctx != book_name else book_name,
                "Header 2": f"Problem {num}",
                "source_pdf": source_pdf,
                "book": book_name,
                "chapter": ctx if ctx != book_name else None,
                "section_label": None,
                "problem_no": str(num),
                "has_diagram": False,
            },
        ))
    n = len(out)
    stats.n_competition_problems += n
    stats.n_matched += n
    return out


def _competition_chunks(
    md_text: str,
    book_name: str,
    source_pdf: str,
    stats: ExtractionStats,
) -> List[Document]:
    """Extract Q+A from competition-paper books (Olympiad collections).

    Layout: competition -> year -> "Part I/II" / "First/Second Day" regions,
    each a flat run of sequentially-numbered problems with the worked solution
    INLINE right after each one ("... Solution. ...", "Solution 1. ...").

    Works on the RAW markdown, NOT _walk_sections, because marker renders this
    born-digital book with wildly inconsistent header levels AND promotes many
    inline "Solution." lines to ``#### Solution`` headers — both of which
    fragment the section walker. Here, headers are classified as STRUCTURAL
    (competition group / year / Part-Day → region boundary, numbering resets)
    vs Solution/Proof (demoted back to inline text so COMP_SOLUTION_RE sees
    them) vs other (soft region break, no label change). The numbering only
    runs once a competition-group header has been seen, so front matter
    (Preface/Contents/IMO winners) never produces chunks.

    Only invoked as an all-or-nothing fallback when every standard path
    produced zero chunks → zero regression for books that already parse.
    """
    out: List[Document] = []
    seen: set = set()
    comp = year = part = None
    in_comp = False
    region: List[str] = []

    def _flush():
        if in_comp and region:
            label = " ".join(p for p in (comp, year, part) if p) or book_name
            out.extend(_competition_emit(
                region, label, book_name, source_pdf, seen, stats))
        region.clear()

    for raw in md_text.splitlines():
        hm = HEADER_LINE_RE.match(raw)
        if not hm:
            region.append(raw)
            continue
        ht = _normalize_header_text(hm.group(2)).strip()
        ym = _COMP_YEAR_RE.match(ht)
        pdm = _COMP_PARTDAY_RE.match(ht)
        if _COMP_SOLHDR_RE.match(ht):
            # marker promoted an inline "Solution." to a header — demote it
            # back to text so the split machinery still sees it.
            region.append(ht)
            continue
        if _COMP_GROUP_RE.search(ht) and not ym:
            _flush()
            comp, year, part = re.sub(r"\s+", " ", ht), None, None
            in_comp = True
            continue
        if ym:
            _flush()
            year = re.sub(r"\s+", " ", ym.group(0)).strip()
            part = None
            continue
        if pdm:
            _flush()
            part = re.sub(r"\s+", " ", pdm.group(0)).strip()
            continue
        # any other header (front matter, "Contents", stray) — soft break
        _flush()
    _flush()
    return out


# --- IMO Compendium (Djukić et al.) cross-part matcher ---------------------
# The book has a *Problems* part (Ch 3) and a SEPARATE *Solutions* part (Ch 4),
# each organised by year. A year section "3.Y The Nth IMO <city> … <YEAR>"
# contains subsections "3.Y.K {Contest|Shortlisted|Longlisted} Problems"
# (numbering RESTARTS every subsection). The solutions part uses headers
# "4.Z Solutions to the {Contest|Shortlisted|Longlisted} Problems of IMO
# <YEAR>". The 3.Y vs 4.Z numbers do NOT align, so the only reliable join key
# is (year, type, number). The book solves a *selected* set per year — items
# with no matching solution entry are simply left unmatched (Sato/Zeitz
# lesson: cannot extract answers the book does not contain).
_IMO_SOL_HDR_RE = re.compile(
    r"^\d+(?:\.\d+)?\s+Solutions?\s+to\s+the\s+"
    r"(?P<type>Contest|Shortlisted|Longlisted|Selected)\s+Problems\s+of\s+IMO\s+"
    r"(?P<year>(?:19|20)\d\d)\b",
    re.IGNORECASE,
)
_IMO_PROB_SUB_RE = re.compile(
    r"^\d+\.\d+\.\d+\s+(?:Some\s+)?"
    r"(?P<type>Contest|Shortlisted|Longlisted|Selected)\s+Problems\b",
    re.IGNORECASE,
)
_IMO_YEAR_HDR_RE = re.compile(r"^\d+\.\d+\s+The\s+\S.*\bIMO\b", re.IGNORECASE)
_IMO_DAY_DIVIDER_RE = re.compile(
    r"^\*{0,2}(?:First|Second|Third|Fourth)\s+Day\*{0,2}\s*$", re.IGNORECASE
)
MIN_IMO_MATCHES = 10


def _imo_compendium_chunks(
    md_text: str,
    book_name: str,
    source_pdf: str,
    stats: ExtractionStats,
) -> List[Document]:
    """Cross-part (year, type, number) matcher for The IMO Compendium.

    All-or-nothing fallback: only invoked when every standard path produced
    zero chunks, and itself returns ``[]`` unless it finds >= ``MIN_IMO_MATCHES``
    matched pairs — so it never perturbs any other book (zero regression)."""
    md_text = _strip_marker_artifacts(md_text)
    lines = md_text.splitlines()

    problems: dict = {}   # (year, type, num) -> body   (Problems part)
    answers: dict = {}    # (year, type, num) -> body   (Solutions part)
    cur_year: Optional[str] = None
    region_key = None      # ("P"|"S", year, type)
    buf: List[str] = []

    def _flush_region():
        if region_key is None or not buf:
            return
        kind, yr, typ = region_key
        body = "\n".join(
            l for l in buf if not _IMO_DAY_DIVIDER_RE.match(l.strip())
        )
        target = problems if kind == "P" else answers
        for num, item in _split_numbered_items(body):
            k = (yr, typ, num)
            if k not in target:          # first occurrence wins
                target[k] = item

    for line in lines:
        hm = HEADER_LINE_RE.match(line)
        if not hm:
            if region_key is not None:
                buf.append(line)
            continue
        # a header — close the open region, then re-classify
        _flush_region()
        buf = []
        region_key = None
        htext = _normalize_header_text(hm.group(2)).strip()
        ms = _IMO_SOL_HDR_RE.match(htext)
        if ms:
            region_key = ("S", ms.group("year"), ms.group("type").lower())
            continue
        mp = _IMO_PROB_SUB_RE.match(htext)
        if mp and cur_year:
            region_key = ("P", cur_year, mp.group("type").lower())
            continue
        if _IMO_YEAR_HDR_RE.match(htext):
            yrs = re.findall(r"(?:19|20)\d\d", htext)
            if yrs:
                cur_year = yrs[-1]
            continue
        # any other header → region already closed; stay closed
    _flush_region()

    out: List[Document] = []
    seen: set = set()
    for key, qbody in problems.items():
        if key not in answers:
            continue
        abody = answers[key]
        yr, typ, num = key
        if has_diagram(qbody) or has_diagram(abody):
            stats.n_dropped_diagram += 1
            continue
        if answer_is_noninformative(qbody, abody):
            stats.n_dropped_noninformative += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        ctx = f"IMO {yr} {typ}"
        out.append(Document(
            page_content=f"## Problem\n{qbody}\n\n## Answer\n{abody}\n",
            metadata={
                "Header 1": f"{book_name} — {ctx}",
                "Header 2": f"Problem {num}",
                "source_pdf": source_pdf,
                "book": book_name,
                "chapter": ctx,
                "section_label": typ,
                "problem_no": str(num),
                "has_diagram": False,
            },
        ))
    if len(out) < MIN_IMO_MATCHES:
        return []
    stats.n_problems += len(problems)
    stats.n_answers += len(answers)
    stats.n_matched += len(out)
    return out


def extract_qa_chunks(
    md_text: str,
    source_pdf: str,
    book_name: str,
) -> Tuple[List[Document], ExtractionStats]:
    """Parse a markdown rendering of a textbook into ``(problem, answer)`` chunks.

    Returns the list of langchain Documents (one per matched, diagram-free
    pair) and the parsing stats.
    """
    sections = _walk_sections(md_text)
    stats = ExtractionStats()

    # Key: (chapter, section_label, num). The section_label dimension
    # distinguishes parallel sections like Introductory vs Advanced that both
    # restart numbering from 1.
    problems: dict[Tuple[Optional[str], Optional[str], str], str] = {}
    answers: dict[Tuple[Optional[str], Optional[str], str], str] = {}

    # Pre-pass: header-delimited Problem/Solution harvest over answer_key
    # sections. Self-contained "101 Problems in ..." / AMT Enrichment books
    # restate every problem under its own ``#### Problem N`` header and follow
    # it with ``#### Solution N`` header(s). _split_numbered_items skips header
    # lines, so the normal path not only sees zero pairs but also mis-splits
    # each ``#### Problem N`` into a bogus one-item "exercise section" and (when
    # the book title starts with a number) cross-matches a runaway plain-text
    # problem to a real solution. When the harvester fires, the normal
    # exercise/answer parse is pure noise for that book, so the harvester
    # becomes authoritative — all-or-nothing, mirroring the inline-problem
    # fallback. Self-gating via MIN_HEADER_PAIRS keeps numbered-list answer
    # keys (104 NT, Engel, Xu Jiagu) entirely on the normal path: the harvester
    # never fires there, header_pair_mode stays False -> zero regression.
    hp_problems: dict[Tuple[Optional[str], Optional[str], str], str] = {}
    hp_answers: dict[Tuple[Optional[str], Optional[str], str], str] = {}
    for sec in sections:
        if sec.kind != "answer_key":
            continue
        pairs = _header_pair_qa_from_section(sec)
        if not pairs:
            continue
        acc: dict[Tuple[Optional[str], Optional[str], str], str] = {}
        for kind, label, num, body in pairs:
            key = (None, label, num)
            if kind == "problem":
                hp_problems.setdefault(key, body)  # first-wins across the
                                                   # overlapping H1 + "## 3."
                                                   # solutions wrappers
            else:  # accumulate "Solution N, Alternative K" blocks in order
                acc[key] = acc[key] + "\n\n" + body if key in acc else body
        for key, body in acc.items():
            hp_answers.setdefault(key, body)
    header_pair_mode = bool(hp_problems) and bool(hp_answers)

    for sec in sections:
        if sec.kind == "exercise":
            stats.n_exercise_sections += 1
            if header_pair_mode:
                continue  # harvester authoritative; normal split is noise here
            for chapter, label, num, body in _problems_from_section(sec):
                key = (chapter, label, num)
                if key in problems:
                    continue  # keep first occurrence
                problems[key] = body
                stats.n_problems += 1
        elif sec.kind == "answer_key":
            stats.n_answer_sections += 1
            if header_pair_mode:
                continue
            for chapter, label, num, body in _answers_from_section(sec):
                key = (chapter, label, num)
                if key in answers:
                    continue
                answers[key] = body
                stats.n_answers += 1

    if header_pair_mode:
        for key, body in hp_problems.items():
            problems[key] = body
            stats.n_problems += 1
        for key, body in hp_answers.items():
            answers[key] = body
            stats.n_answers += 1

    # Fallback: books that number problems inline in chapter prose with no
    # Exercises/Problems header (Skopenkov, Andreescu Real Analysis, Zeitz),
    # OR books whose explicit Exercises/Problems sections exist but have NO
    # parseable answer surface (Stevens "Olympiad Number Theory": 34 explicit
    # "Problems" with their answers in unparsed "Solutions of Chapter N"
    # blocks, while the real Q+A is the inline "Example N.M. ... Solution."
    # worked examples). The `n_answers == 0` arm is zero-regression for books
    # that already work: 104 NT / Engel / Xu Jiagu V2 all parse answers
    # (n_answers > 0), so the gate stays closed and their output is
    # byte-identical.
    if not problems or stats.n_answers == 0:
        for sec in sections:
            if sec.kind != "normal":
                continue
            for chapter, label, num, body in _inline_problems_from_section(sec):
                key = (chapter, label, num)
                if key in problems:
                    continue
                problems[key] = body
                stats.n_problems += 1
                stats.n_inline_problems += 1

    # Self-contained Q+A: a problem body that itself contains an inline
    # "Solution."/"Proof." marker (Andreescu Real Analysis & many solutions
    # manuals). Split it and register the answer under the same key. No-op for
    # books whose problem bodies have no such marker — zero regression.
    for key, body in list(problems.items()):
        q, a = _split_inline_solution(body)
        if a is not None and key not in answers:
            problems[key] = q
            answers[key] = a
            stats.n_answers += 1

    def _lookup_answer(chapter, label, num):
        for k in [(chapter, label, num), (chapter, None, num),
                  (None, label, num), (None, None, num)]:
            if k in answers:
                return answers[k]
        return None

    chunks: List[Document] = []
    for key, prob_body in problems.items():
        chapter, label, num = key
        ans_body = _lookup_answer(chapter, label, num)
        if ans_body is None:
            stats.n_unmatched += 1
            logger.debug("unmatched problem: chapter=%s label=%s num=%s", chapter, label, num)
            continue

        if has_diagram(prob_body) or has_diagram(ans_body):
            stats.n_dropped_diagram += 1
            continue

        if answer_is_noninformative(prob_body, ans_body):
            stats.n_dropped_noninformative += 1
            continue

        content = f"## Problem\n{prob_body.strip()}\n\n## Answer\n{ans_body.strip()}\n"
        chapter_label = chapter if chapter else ""
        header1 = book_name + (f" — Chapter {chapter}" if chapter_label else "")
        if label:
            header1 += f" ({label.capitalize()})"
        chunks.append(Document(
            page_content=content,
            metadata={
                "Header 1": header1,
                "Header 2": f"Problem {num}",
                "source_pdf": source_pdf,
                "book": book_name,
                "chapter": chapter,
                "section_label": label,
                "problem_no": num,
                "has_diagram": False,
            },
        ))
        stats.n_matched += 1

    # All-or-nothing fallback: competition-paper books (Olympiad collections)
    # have no Exercises/Solutions headers and no numbered-list answer key, so
    # every standard path above yields nothing. Only engage when standard
    # parsing produced ZERO chunks — guarantees byte-identical output (zero
    # regression) for every book that already parses.
    if not chunks:
        chunks = _competition_chunks(md_text, book_name, source_pdf, stats)
    # The IMO Compendium's standard-path output is a small set of mis-read
    # cross-matches (it has a separate Solutions part the normal matcher
    # cannot align), so the dedicated matcher must REPLACE it, not append —
    # i.e. authoritative-when-fires. _imo_compendium_chunks returns [] unless
    # it finds >= MIN_IMO_MATCHES (year,type,number) pairs whose headers match
    # the Compendium's exact "Solutions to the {type} Problems of IMO YYYY" /
    # "N.Y.K {type} Problems" shape — structurally impossible for any other
    # book, so this is zero-regression (verified: 104NT/Engel/XuJiaguV2/Sato/
    # OTIS byte-identical).
    imo = _imo_compendium_chunks(md_text, book_name, source_pdf, stats)
    if imo:
        chunks = imo

    return chunks, stats


def extract_qa_chunks_from_markdown_file(
    md_path: Path,
    source_pdf: Optional[str] = None,
    book_name: Optional[str] = None,
) -> Tuple[List[Document], ExtractionStats]:
    md_text = md_path.read_text(encoding="utf-8")
    return extract_qa_chunks(
        md_text,
        source_pdf=source_pdf or str(md_path),
        book_name=book_name or md_path.stem,
    )


def extract_qa_chunks_cross_pdf(
    textbook_md: str,
    solutions_md: str,
    source_textbook: str,
    source_solutions: str,
    book_name: str,
) -> Tuple[List[Document], ExtractionStats]:
    """Cross-PDF mode: textbook PDF supplies problems (from its *exercise
    sections*), a separate solutions-manual PDF supplies answers (scanned
    *across all sections*, since every section of a solutions manual is
    answer content). Items are matched by ``(chapter, problem_no)``.

    This is the right mode for AoPS Intermediate Algebra / Intermediate
    Counting & Probability, the Stewart-style textbook+ISM pairings, and
    any other book that ships solutions as a separate volume.
    """
    sections_t = _walk_sections(textbook_md)
    sections_s = _walk_sections(solutions_md)
    stats = ExtractionStats()

    problems: dict[Tuple[Optional[str], Optional[str], str], str] = {}
    for sec in sections_t:
        if sec.kind != "exercise":
            continue
        stats.n_exercise_sections += 1
        for chapter, label, num, body in _problems_from_section(sec):
            key = (chapter, label, num)
            if key in problems:
                continue
            problems[key] = body
            stats.n_problems += 1

    answers: dict[Tuple[Optional[str], Optional[str], str], str] = {}
    for sec in sections_s:
        # Every section in the solutions manual is treated as a potential
        # source of answers — no answer_key classifier required, because the
        # whole volume IS the answer key.
        stats.n_answer_sections += 1
        for chapter, label, num, body in _problems_from_section(sec):
            key = (chapter, label, num)
            if key in answers:
                continue
            answers[key] = body
            stats.n_answers += 1

    def _lookup_answer(chapter, label, num):
        for k in [(chapter, label, num), (chapter, None, num),
                  (None, label, num), (None, None, num)]:
            if k in answers:
                return answers[k]
        return None

    chunks: List[Document] = []
    for key, prob_body in problems.items():
        chapter, label, num = key
        ans_body = _lookup_answer(chapter, label, num)
        if ans_body is None:
            stats.n_unmatched += 1
            logger.debug("unmatched problem: chapter=%s label=%s num=%s", chapter, label, num)
            continue
        if has_diagram(prob_body) or has_diagram(ans_body):
            stats.n_dropped_diagram += 1
            continue

        if answer_is_noninformative(prob_body, ans_body):
            stats.n_dropped_noninformative += 1
            continue

        content = f"## Problem\n{prob_body.strip()}\n\n## Answer\n{ans_body.strip()}\n"
        chapter_label = chapter if chapter else ""
        header1 = book_name + (f" — Chapter {chapter}" if chapter_label else "")
        if label:
            header1 += f" ({label.capitalize()})"
        chunks.append(Document(
            page_content=content,
            metadata={
                "Header 1": header1,
                "Header 2": f"Problem {num}",
                "source_pdf": source_textbook,
                "source_pdf_solutions": source_solutions,
                "book": book_name,
                "chapter": chapter,
                "section_label": label,
                "problem_no": num,
                "has_diagram": False,
            },
        ))
        stats.n_matched += 1

    return chunks, stats


def extract_qa_chunks_cross_pdf_from_files(
    textbook_path: Path,
    solutions_path: Path,
    book_name: Optional[str] = None,
    converted_dir: Optional[Path] = None,
) -> Tuple[List[Document], ExtractionStats]:
    """File-level wrapper: accepts either PDF or markdown for each side; if a
    PDF is passed, runs marker_single via the lazy-imported helper."""
    def _to_markdown(path: Path) -> Tuple[str, str]:
        """Return (markdown_text, source_label_for_metadata)."""
        if path.suffix.lower() == ".pdf":
            md_path = _convert_pdf_to_markdown_inline(path, output_dir=converted_dir)
            if not md_path:
                raise RuntimeError(f"marker_single failed on {path}")
            return md_path.read_text(encoding="utf-8"), str(path)
        return path.read_text(encoding="utf-8"), str(path)

    textbook_md, source_textbook = _to_markdown(textbook_path)
    solutions_md, source_solutions = _to_markdown(solutions_path)
    return extract_qa_chunks_cross_pdf(
        textbook_md=textbook_md,
        solutions_md=solutions_md,
        source_textbook=source_textbook,
        source_solutions=source_solutions,
        book_name=book_name or textbook_path.stem,
    )


def extract_qa_chunks_from_pdf(
    pdf_path: Path,
    converted_dir: Optional[Path] = None,
) -> Tuple[List[Document], ExtractionStats]:
    md_path = _convert_pdf_to_markdown_inline(pdf_path, output_dir=converted_dir)
    if not md_path:
        logger.warning("PDF → Markdown failed: %s", pdf_path)
        return [], ExtractionStats()
    return extract_qa_chunks_from_markdown_file(
        md_path, source_pdf=str(pdf_path), book_name=pdf_path.stem
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize(docs: List[Document]) -> List[dict]:
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]


def _write_chunks(docs: List[Document], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_serialize(docs), ensure_ascii=False, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert math textbook(s) into one-Q+A-per-chunk JSON.",
    )
    p.add_argument(
        "pdfs",
        nargs="*",
        help="Path(s) to PDF files to process.",
    )
    p.add_argument(
        "--md",
        type=Path,
        default=None,
        help="Use this pre-converted markdown file instead of running marker_single. "
             "When set, --book-name controls the book label.",
    )
    p.add_argument(
        "--solutions",
        type=Path,
        default=None,
        help="Path to a separate solutions-manual PDF (or markdown). When set, the extractor "
             "runs in cross-PDF mode: problems come from the textbook side, answers come from "
             "this file's all-sections scan. Pairs with the FIRST positional pdf (or --md).",
    )
    p.add_argument(
        "--book-name",
        default=None,
        help="Book label used in metadata when --md is set. Defaults to the md file stem.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=CHUNKS_DIR,
        help="Directory for {stem}.chunks.json files (default: datasets/chunks).",
    )
    p.add_argument(
        "--converted-dir",
        type=Path,
        default=None,
        help="Override the marker_single output directory (default: <CONVERTED_DIR>/<pdf_stem>).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _process_pdf(pdf_path: Path, output_dir: Path, converted_dir: Optional[Path]) -> Optional[Path]:
    docs, stats = extract_qa_chunks_from_pdf(pdf_path, converted_dir=converted_dir)
    logger.info(
        "%s: exercise_sections=%d answer_sections=%d problems=%d (inline=%d) "
        "answers=%d matched=%d dropped_diagram=%d dropped_noninfo=%d unmatched=%d",
        pdf_path.name,
        stats.n_exercise_sections,
        stats.n_answer_sections,
        stats.n_problems,
        stats.n_inline_problems,
        stats.n_answers,
        stats.n_matched,
        stats.n_dropped_diagram,
        stats.n_dropped_noninformative,
        stats.n_unmatched,
    )
    if not docs:
        logger.warning("no chunks produced for %s", pdf_path)
        return None
    return _write_chunks(docs, output_dir / f"{pdf_path.stem}.chunks.json")


def _process_md(md_path: Path, book_name: Optional[str], output_dir: Path) -> Optional[Path]:
    docs, stats = extract_qa_chunks_from_markdown_file(md_path, book_name=book_name)
    logger.info(
        "%s: exercise_sections=%d answer_sections=%d problems=%d (inline=%d) "
        "answers=%d matched=%d dropped_diagram=%d dropped_noninfo=%d unmatched=%d",
        md_path.name,
        stats.n_exercise_sections,
        stats.n_answer_sections,
        stats.n_problems,
        stats.n_inline_problems,
        stats.n_answers,
        stats.n_matched,
        stats.n_dropped_diagram,
        stats.n_dropped_noninformative,
        stats.n_unmatched,
    )
    if not docs:
        logger.warning("no chunks produced for %s", md_path)
        return None
    stem = book_name or md_path.stem
    return _write_chunks(docs, output_dir / f"{stem}.chunks.json")


def _process_cross_pdf(
    textbook_path: Path,
    solutions_path: Path,
    book_name: Optional[str],
    output_dir: Path,
    converted_dir: Optional[Path],
) -> Optional[Path]:
    docs, stats = extract_qa_chunks_cross_pdf_from_files(
        textbook_path=textbook_path,
        solutions_path=solutions_path,
        book_name=book_name,
        converted_dir=converted_dir,
    )
    logger.info(
        "cross_pdf %s + %s: exercise_sections=%d solutions_sections=%d "
        "problems=%d answers=%d matched=%d dropped_diagram=%d dropped_noninfo=%d unmatched=%d",
        textbook_path.name,
        solutions_path.name,
        stats.n_exercise_sections,
        stats.n_answer_sections,
        stats.n_problems,
        stats.n_answers,
        stats.n_matched,
        stats.n_dropped_diagram,
        stats.n_dropped_noninformative,
        stats.n_unmatched,
    )
    if not docs:
        logger.warning("no chunks produced for %s + %s", textbook_path, solutions_path)
        return None
    stem = book_name or textbook_path.stem
    return _write_chunks(docs, output_dir / f"{stem}.chunks.json")


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s - %(message)s")

    if not args.pdfs and not args.md:
        raise SystemExit("Provide one or more PDF paths, or --md <markdown-file>.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []

    if args.solutions is not None:
        # Cross-PDF mode: pair --solutions with the first positional PDF or --md.
        if args.md and not args.pdfs:
            textbook_path = args.md
        elif args.pdfs:
            textbook_path = Path(args.pdfs[0]).expanduser()
            if len(args.pdfs) > 1:
                logger.warning(
                    "--solutions is set; ignoring extra positional PDFs %s",
                    args.pdfs[1:],
                )
        else:
            raise SystemExit("--solutions requires either --md or a positional PDF.")
        out = _process_cross_pdf(
            textbook_path=textbook_path,
            solutions_path=args.solutions,
            book_name=args.book_name,
            output_dir=args.output_dir,
            converted_dir=args.converted_dir,
        )
        if out:
            written.append(out)
    else:
        if args.md:
            out = _process_md(args.md, args.book_name, args.output_dir)
            if out:
                written.append(out)
        for raw in args.pdfs:
            pdf_path = Path(raw).expanduser()
            out = _process_pdf(pdf_path, args.output_dir, args.converted_dir)
            if out:
                written.append(out)

    if not written:
        raise SystemExit("No outputs were written.")
    for w in written:
        logger.info("wrote %s", w)


if __name__ == "__main__":
    main()
