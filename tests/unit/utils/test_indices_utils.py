"""
Unit tests for utility functions that handle sequence indices.

This module validates a collection of helper functions from `indices_utils`.
These functions are designed to provide safe and consistent ways to access
sequence data, normalize coordinate pairs, validate intervals, and determine
split points, which are fundamental operations in RNA folding algorithms.
"""
import pytest

from rna_pk_fold.utils.indices_utils import (
    safe_base,
    canonical_pair,
    is_interval_valid,
    get_default_split_point,
)


# ------------------------------
# safe_base
# ------------------------------
def test_safe_base_in_range_and_boundaries():
    """
    Tests successful character retrieval when the index is valid.
    """
    seq = "AUGC"  # Valid indices are 0, 1, 2, 3.
    # Test first element.
    assert safe_base(seq, 0) == "A"
    # Test last element (boundary).
    assert safe_base(seq, 3) == "C"


def test_safe_base_out_of_range_and_empty_seq():
    """
    Tests the "safe" behavior: the function should return `None` instead of
    raising an `IndexError` for out-of-bounds access.
    """
    seq = "AUGC"
    # Test with a negative index.
    assert safe_base(seq, -1) is None
    # Test with an index equal to the sequence length.
    assert safe_base(seq, 4) is None
    # Test with an empty sequence.
    assert safe_base("", 0) is None


# ------------------------------
# canonical_pair
# ------------------------------
@pytest.mark.parametrize(
    "i,j,expected",
    [
        (0, 0, (0, 0)),    # Equal indices are unchanged.
        (1, 5, (1, 5)),    # Already in order.
        (7, 2, (2, 7)),    # Should be swapped to (min, max).
        (-3, 4, (-3, 4)),  # Handles negative numbers, already in order.
        (4, -3, (-3, 4)),  # Handles negative numbers and swaps them.
    ],
)
def test_canonical_pair_orders_non_decreasing(i, j, expected):
    """
    Verifies that `canonical_pair(i, j)` always returns a tuple (min(i,j), max(i,j)).
    This is useful for creating consistent keys for dictionaries or sets where the
    order of the pair should not matter.
    """
    assert canonical_pair(i, j) == expected


# ------------------------------
# is_interval_valid
# ------------------------------
@pytest.mark.parametrize(
    "i,j,n,ok",
    [
        (0, 0, 1, True),   # Smallest possible valid interval.
        (0, 3, 4, True),   # Interval spanning the entire sequence.
        (2, 2, 5, True),   # A single-element interval is valid.
        (-1, 2, 5, False), # Fails because i < 0.
        (1, -1, 5, False), # Fails because j < 0.
        (3, 2, 5, False),  # Fails because i > j.
        (0, 5, 5, False),  # Fails because j is not < n.
        (0, 6, 6, False),  # Fails because j is not < n.
        (0, 0, 0, False),  # Fails because n must be > 0.
    ],
)
def test_is_interval_valid_various(i, j, n, ok):
    """
    Tests the `is_interval_valid` function against a variety of cases.
    This function is a crucial guard to ensure that an interval [i, j] is
    well-formed (i <= j) and lies entirely within the sequence bounds [0, n-1].
    """
    assert is_interval_valid(i, j, n) is ok


# ------------------------------
# get_default_split_point
# ------------------------------
def test_get_default_split_point_midpoint_no_fallback():
    """
    Tests the default behavior of calculating the integer midpoint of an interval.
    """
    # For an even-sized span, the midpoint is straightforward.
    assert get_default_split_point(0, 4) == 2
    # For an odd-sized span, the result should be floor-divided, favoring the lower index.
    assert get_default_split_point(1, 4) == 2
    # In all cases, the calculated midpoint must lie within the original interval.
    i, j = 5, 11
    m = get_default_split_point(i, j)
    assert i <= m <= j


def test_get_default_split_point_respects_fallback():
    """
    Tests that if a `fallback` value is provided, it is returned directly,
    overriding the midpoint calculation.
    """
    # The provided fallback value of 7 should be returned, ignoring the midpoint.
    assert get_default_split_point(0, 100, fallback=7) == 7
    # The function does not validate or clamp the fallback value; it simply returns it.
    assert get_default_split_point(10, 20, fallback=-3) == -3
