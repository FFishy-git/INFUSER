"""Unit tests for verl_inf_evolve.gen_eval.prepare_eval_docs."""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

from verl_inf_evolve.gen_eval.prepare_eval_docs import (
    load_documents,
    main,
    print_summary,
    sample_balanced_documents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(field_counts: dict[str, int]) -> list[dict]:
    """Build a synthetic document list with given per-field counts."""
    docs: list[dict] = []
    idx = 0
    for field, count in sorted(field_counts.items()):
        for i in range(count):
            docs.append({
                "doc_id": str(idx),
                "content": f"content for doc {idx}",
                "field": field,
                "source_pdf": f"{field.lower()}_{i}.pdf",
                "token_count": 100 + idx,
            })
            idx += 1
    return docs


# ---------------------------------------------------------------------------
# Tests: balanced sampling
# ---------------------------------------------------------------------------

class TestSampleBalancedDocuments:
    """Tests for sample_balanced_documents()."""

    def test_equal_fields_exact_split(self):
        """128 docs / 4 fields = 32 each when all fields have enough docs."""
        docs = _make_docs({
            "Astronomy": 100,
            "Biochemistry": 100,
            "Geography": 100,
            "Physics": 100,
        })
        sampled = sample_balanced_documents(docs, num_docs=128, seed=42)

        assert len(sampled) == 128

        counts = defaultdict(int)
        for d in sampled:
            counts[d["field"]] += 1
        assert counts["Astronomy"] == 32
        assert counts["Biochemistry"] == 32
        assert counts["Geography"] == 32
        assert counts["Physics"] == 32

    def test_total_count_matches_requested(self):
        """Sampled count always equals requested num_docs."""
        docs = _make_docs({"A": 200, "B": 200, "C": 200})
        sampled = sample_balanced_documents(docs, num_docs=90, seed=0)
        assert len(sampled) == 90

    def test_redistribution_when_field_short(self):
        """When a field has fewer docs than quota, remainder is redistributed."""
        docs = _make_docs({
            "Astronomy": 10,  # only 10, less than 32
            "Biochemistry": 100,
            "Geography": 100,
            "Physics": 100,
        })
        sampled = sample_balanced_documents(docs, num_docs=128, seed=42)

        assert len(sampled) == 128

        counts = defaultdict(int)
        for d in sampled:
            counts[d["field"]] += 1

        # Astronomy should have all 10
        assert counts["Astronomy"] == 10
        # Remaining 118 distributed among 3 fields
        remaining = 128 - 10
        assert counts["Biochemistry"] + counts["Geography"] + counts["Physics"] == remaining

    def test_multiple_fields_short(self):
        """When multiple fields are short, redistribution cascades correctly."""
        docs = _make_docs({
            "A": 5,
            "B": 5,
            "C": 100,
            "D": 100,
        })
        sampled = sample_balanced_documents(docs, num_docs=40, seed=42)

        assert len(sampled) == 40

        counts = defaultdict(int)
        for d in sampled:
            counts[d["field"]] += 1

        # A gets all 5, B gets all 5, remaining 30 split between C and D
        assert counts["A"] == 5
        assert counts["B"] == 5
        assert counts["C"] + counts["D"] == 30

    def test_deterministic_with_seed(self):
        """Same seed produces identical output."""
        docs = _make_docs({"X": 100, "Y": 100})
        s1 = sample_balanced_documents(docs, num_docs=20, seed=123)
        s2 = sample_balanced_documents(docs, num_docs=20, seed=123)

        ids1 = [d["doc_id"] for d in s1]
        ids2 = [d["doc_id"] for d in s2]
        assert ids1 == ids2

    def test_different_seeds_differ(self):
        """Different seeds produce different output (with high probability)."""
        docs = _make_docs({"X": 200, "Y": 200})
        s1 = sample_balanced_documents(docs, num_docs=20, seed=1)
        s2 = sample_balanced_documents(docs, num_docs=20, seed=2)

        ids1 = [d["doc_id"] for d in s1]
        ids2 = [d["doc_id"] for d in s2]
        assert ids1 != ids2

    def test_empty_documents(self):
        """Empty input produces empty output."""
        sampled = sample_balanced_documents([], num_docs=10, seed=42)
        assert sampled == []

    def test_preserves_document_schema(self):
        """Sampled documents retain all original keys."""
        docs = _make_docs({"Physics": 50})
        docs[0]["extra_key"] = "extra_value"
        sampled = sample_balanced_documents(docs, num_docs=10, seed=42)

        for d in sampled:
            assert "doc_id" in d
            assert "content" in d
            assert "field" in d
            assert "source_pdf" in d
            assert "token_count" in d

    def test_single_field(self):
        """Works with only one field."""
        docs = _make_docs({"Physics": 50})
        sampled = sample_balanced_documents(docs, num_docs=20, seed=42)
        assert len(sampled) == 20
        assert all(d["field"] == "Physics" for d in sampled)

    def test_uneven_remainder(self):
        """num_docs not evenly divisible by num_fields allocates extras correctly."""
        docs = _make_docs({"A": 50, "B": 50, "C": 50})
        sampled = sample_balanced_documents(docs, num_docs=10, seed=42)
        assert len(sampled) == 10

        counts = defaultdict(int)
        for d in sampled:
            counts[d["field"]] += 1

        # 10 / 3 = 3 each + 1 extra → one field gets 4
        assert sorted(counts.values()) == [3, 3, 4]

    def test_request_more_than_available(self):
        """When requesting more docs than available, returns all available."""
        docs = _make_docs({"A": 3, "B": 3})
        sampled = sample_balanced_documents(docs, num_docs=100, seed=42)
        # Can only return 6 total
        assert len(sampled) == 6


# ---------------------------------------------------------------------------
# Tests: CLI / main()
# ---------------------------------------------------------------------------

class TestMain:
    """Tests for the CLI entry point."""

    def test_full_pipeline(self, tmp_path: Path):
        """End-to-end: load → sample → save."""
        docs = _make_docs({
            "Astronomy": 50,
            "Biochemistry": 50,
            "Geography": 50,
            "Physics": 50,
        })
        input_file = tmp_path / "documents.json"
        input_file.write_text(json.dumps(docs))

        output_file = tmp_path / "eval_documents.json"

        main([
            "--num_docs", "40",
            "--seed", "42",
            "--input_path", str(input_file),
            "--output_path", str(output_file),
        ])

        assert output_file.exists()
        result = json.loads(output_file.read_text())
        assert len(result) == 40

        counts: defaultdict[str, int] = defaultdict(int)
        for d in result:
            counts[d["field"]] += 1
        assert counts["Astronomy"] == 10
        assert counts["Biochemistry"] == 10
        assert counts["Geography"] == 10
        assert counts["Physics"] == 10

    def test_creates_output_directory(self, tmp_path: Path):
        """Output parent directories are created if they don't exist."""
        docs = _make_docs({"A": 20})
        input_file = tmp_path / "docs.json"
        input_file.write_text(json.dumps(docs))

        output_file = tmp_path / "nested" / "dir" / "out.json"

        main([
            "--num_docs", "5",
            "--input_path", str(input_file),
            "--output_path", str(output_file),
        ])

        assert output_file.exists()


# ---------------------------------------------------------------------------
# Tests: load_documents
# ---------------------------------------------------------------------------

class TestLoadDocuments:

    def test_loads_json_array(self, tmp_path: Path):
        data = [{"doc_id": "0", "field": "X", "content": "text"}]
        p = tmp_path / "docs.json"
        p.write_text(json.dumps(data))
        loaded = load_documents(str(p))
        assert loaded == data
