"""Tests for backend-neutral configuration helpers."""

from __future__ import annotations

import pytest

from control_suite_mcp_aps_2idd.common import json_safe, parse_range, validate_position_in_range


def test_parse_range_returns_float_pair() -> None:
    assert parse_range("1, 2") == (1.0, 2.0)


def test_json_safe_converts_tuple_to_list() -> None:
    assert json_safe({"xrf_elms": ("Cr", "Fe")}) == {"xrf_elms": ["Cr", "Fe"]}


def test_validate_position_in_range_rejects_out_of_bounds_value() -> None:
    with pytest.raises(ValueError):
        validate_position_in_range(5.0, (0.0, 4.0), "x")
