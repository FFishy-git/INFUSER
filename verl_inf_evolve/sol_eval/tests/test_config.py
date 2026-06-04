"""Unit tests for sol_eval/config.py — parse_checkpoints utility.

Tests cover:
- parse_checkpoints: list, slice, dash-range, mixed, edge cases
"""

from __future__ import annotations

import pytest

from verl_inf_evolve.sol_eval.config import parse_checkpoints


# ---------------------------------------------------------------------------
# parse_checkpoints tests
# ---------------------------------------------------------------------------

class TestParseCheckpoints:
    """Tests for parse_checkpoints() covering all supported formats."""

    @pytest.mark.parametrize(
        "spec, expected",
        [
            (5, [5]),
            ([0, 5, 10], [0, 5, 10]),
            ([10, 0, 5], [0, 5, 10]),  # sorted
            ([5, 5, 10], [5, 10]),  # deduplicated
        ],
        ids=["single-int", "list-passthrough", "list-sorted", "list-deduplicated"],
    )
    def test_list_input(self, spec, expected):
        assert parse_checkpoints(spec) == expected

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("0:20:5", [0, 5, 10, 15]),
            ("0:20", list(range(0, 21))),  # two-part inclusive
        ],
        ids=["three-part-slice", "two-part-inclusive"],
    )
    def test_slice_notation(self, spec, expected):
        assert parse_checkpoints(spec) == expected

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("0-5", [0, 1, 2, 3, 4, 5]),
            ("3-3", [3]),  # single-element range
        ],
        ids=["dash-range", "dash-single"],
    )
    def test_dash_range(self, spec, expected):
        assert parse_checkpoints(spec) == expected

    def test_mixed_format(self):
        assert parse_checkpoints("0-3,7,10-12") == [0, 1, 2, 3, 7, 10, 11, 12]

    def test_single_int_string(self):
        assert parse_checkpoints("5") == [5]

    def test_bracketed_list_string(self):
        assert parse_checkpoints("[-1]") == [-1]
        assert parse_checkpoints("[0,5,10]") == [0, 5, 10]

    @pytest.mark.parametrize(
        "spec",
        [
            "",       # empty string
            [],       # empty list
        ],
        ids=["empty-string", "empty-list"],
    )
    def test_empty_raises(self, spec):
        with pytest.raises(ValueError):
            parse_checkpoints(spec)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid checkpoint spec type"):
            parse_checkpoints(3.14)  # type: ignore[arg-type]

    def test_invalid_part_raises(self):
        with pytest.raises(ValueError, match="Invalid checkpoint part"):
            parse_checkpoints("abc")

    def test_invalid_slice_raises(self):
        with pytest.raises(ValueError, match="Invalid slice notation"):
            parse_checkpoints("1:2:3:4")
