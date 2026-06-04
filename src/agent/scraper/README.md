# agent.scraper — PDF → chunks pipeline

Converts textbook PDFs into `.chunks.json` files suitable for downstream
question-generation / filtering. Uses `marker-pdf` for PDF → Markdown,
`langchain-text-splitters` for header-aware splitting, and an optional
LLM judge (via `llm_client.vllm_client`) to drop non-teaching chunks
(TOC, index, answer keys, etc.).

## Run

From the repo root, with `src/` on `PYTHONPATH`:

```
PYTHONPATH=src python -m agent.scraper.pdf_to_chunks \
    <pdf_or_dir> [<pdf_or_dir> ...] \
    --output-dir datasets/chunks \
    --max-chunk-tokens 1024 \
    --min-chunk-tokens 50 \
    --workers 4
```

Positional args may be individual `.pdf` files or directories containing
PDFs. Default output is `datasets/chunks/<stem>.chunks.json` relative to
cwd. Use `-o/--output` for a single-file override or `--output-dir` to
change the directory. `--gzip` forces gzipped JSON output.

### With LLM judge

Point at a running vLLM server (e.g. Qwen3-8B) to filter out
non-self-contained chunks:

```
PYTHONPATH=src python -m agent.scraper.pdf_to_chunks \
    <pdf> \
    --judge-base-url http://localhost:8000 \
    --judge-model Qwen/Qwen3-8B \
    --judge-workers 16
```

### SLURM

`launch/pdf_to_chunks_slurm.sh` is a batch wrapper that picks up
`PDF`, `PDF_DIR`, or `PDF_LIST` (with `--array`) env vars and forwards
judge config via `JUDGE_BASE_URL` / `JUDGE_MODEL` / `JUDGE_API_KEY`.

## Dependencies

System / CLI:
- `marker_single` CLI for PDF → Markdown (`pip install marker-pdf`);
  see the `marker` env in the repo's `setup_pod.sh` for a working stack.

Python:
- `langchain-core`, `langchain-text-splitters`
- `transformers` (tokenizer-based token counting)
- `tqdm`

## Chunking guarantees

- Every output chunk is `<= max_chunk_tokens` tokens. When a header
  section is too large, the splitter falls back through: paragraph split
  → sentence split → comma/semicolon split → adaptive char split.
- Chunks below `min_chunk_tokens` are dropped (typically page headers /
  stray fragments).
- With `--judge-base-url`, each chunk is scored by an LLM for self-contained
  teaching value. Non-teaching content (TOC, index, answer keys, prefaces)
  is filtered out.

## Competition sources (Putnam, IMO Shortlist)

The unified pipeline above is tuned for textbook-style PDFs and its LLM
judge **discards answer keys and solution manuals** by design. For
competition-math sources where the goal is the opposite — keep each
problem paired with its official solution — use the per-source
extractors instead. They emit the same `.chunks.json` schema so
`preprocess/merge_textbooks.py` picks them up unchanged.

### Putnam Archive (1985–2025)

```
PYTHONPATH=src python -m agent.scraper.competition_putnam \
    --start-year 1985 --end-year 2025 \
    --output-dir .cache/data/source/textbooks/Math_Olympiad/putnam
```

Pulls `YYYY.tex` + `YYYYs.tex` from `kskedlaya.org/putnam-archive`,
parses the `\item[A1]…\item[B6]` blocks, joins on the label, and writes
`putnam_<year>.chunks.json` (12 problems each, raw LaTeX preserved).

### IMO Shortlist (2011–2024)

```
PYTHONPATH=src python -m agent.scraper.competition_imo \
    --start-year 2011 --end-year 2024 \
    --output-dir .cache/data/source/textbooks/Math_Olympiad/imo_sl
```

Pulls each `IMO{YEAR}SL.pdf` from `imo-official.org`, runs
`marker_single` (so the rendering style matches the rest of the corpus),
locates the `Solutions` section, and slices it on per-problem labels
(`A1.`, `C7.`, `G2.`, `N4.`, …). The solutions section already restates
each problem before its solution, so each slice is a complete
problem+solution pair.

Two typesetting eras are handled:

- **Modern** (≥ 2014, plus 2023 verified): the marker output ships an
  explicit `### Solutions` divider and each label appears as an inline
  page-anchor link `[A1.](#page-X-Y)`. Slicing on those anchors gives
  one chunk per problem.
- **Legacy** (≤ 2011 known, exact crossover ~2012/2013): no divider and
  labels render as `#### A1` H4 headings, sometimes duplicated
  (`#### A3 A3`, or two adjacent `#### A4` lines from a banner). The
  legacy path picks the longest body per label and requires it to look
  like a solution (contain `Solution.` / `Answer.` / `Comment.` /
  `Proof.`). Labels whose only occurrences are bare problem statements
  are skipped with a warning — that pattern indicates marker garbled
  the corresponding solutions section (e.g. 2011 SL drops the N
  category's per-problem headings).

Both extractors deliberately bypass `MarkdownChunker.chunk()` and the
LLM judge.
