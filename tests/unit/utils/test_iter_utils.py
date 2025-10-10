"""
Unit tests for iterator utility functions.

This module validates a collection of helper functions from `iter_utils` that
generate iterators for various index combinations. These iterators are fundamental
building blocks for dynamic programming algorithms in RNA folding, as they provide
a systematic way to loop over all required substructures (spans, holes, etc.).
"""
from rna_pk_fold.utils.iter_utils import (
    iter_spans,
    iter_holes,
    iter_complementary_tuples,
    iter_inner_holes,
)


# ------------------------------
# iter_spans
# ------------------------------
def test_iter_spans_n0_is_empty():
    """Tests the base case of a sequence of length 0, which has no spans."""
    assert list(iter_spans(0)) == []


def test_iter_spans_n1_singleton():
    """Tests the base case of a sequence of length 1, which has one span (0,0)."""
    assert list(iter_spans(1)) == [(0, 0)]


def test_iter_spans_n3_exact_order_and_content():
    """
    Tests that `iter_spans` yields all spans in the correct, predictable order
    (by increasing span length, then by increasing start index).
    """
    # For n=3 the spans by increasing span length `s` should be:
    # s=0: (0,0), (1,1), (2,2)
    # s=1: (0,1), (1,2)
    # s=2: (0,2)
    expected = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    got = list(iter_spans(3))
    assert got == expected


def test_iter_spans_count_matches_formula():
    """
    Tests two properties of `iter_spans` for a range of sequence lengths:
    1. The total number of spans yielded matches the formula N*(N+1)/2.
    2. All yielded spans (i,j) are valid (i.e., 0 <= i <= j < n).
    """
    for n in range(1, 8):
        got = list(iter_spans(n))
        # The number of (i,j) pairs with 0 <= i <= j < n is n*(n+1)/2.
        expected_count = n * (n + 1) // 2
        assert len(got) == expected_count
        # Verify that all yielded (i,j) tuples are within bounds and correctly ordered.
        assert all(0 <= i <= j < n for (i, j) in got)


# ------------------------------
# iter_holes
# ------------------------------
def test_iter_holes_empty_when_interval_too_small():
    """
    Tests that `iter_holes` yields nothing if the interval is too small to
    contain a valid hole structure.
    """
    # An interval of length 1 or 2 cannot contain a hole.
    assert list(iter_holes(0, 0)) == []
    assert list(iter_holes(2, 3)) == []


def test_iter_holes_example_i0_j4_exact():
    """
    Tests that `iter_holes` yields the correct sequence of (k,l) tuples
    for a specific example, in the expected order.
    """
    # For i=0, j=4, the max hole size `h` is 3. The expected output is:
    # h=1: (0,2), (1,3), (2,4)
    # h=2: (0,3), (1,4)
    # h=3: (0,4)
    expected = [(0, 2), (1, 3), (2, 4),
                (0, 3), (1, 4),
                (0, 4)]
    got = list(iter_holes(0, 4))
    assert got == expected


def test_iter_holes_bounds_and_monotonicity():
    """
    Verifies that all holes yielded by the iterator satisfy the required
    index constraints.
    """
    i, j = 3, 9
    for k, l in iter_holes(i, j):
        # The hole (k,l) must be contained within the outer span (i,j).
        assert i <= k < l <= j
        # The hole must contain at least one unpaired base.
        assert (l - k - 1) >= 1


# ------------------------------
# iter_complementary_tuples
# ------------------------------
def test_iter_complementary_tuples_small_example():
    """
    Tests that `iter_complementary_tuples` yields the correct sequence of
    (r,k,l) tuples for a small example.
    """
    # For i=0, j=3, the logic is:
    # r in 1..2
    # r=1: k in 1..1, l in 2..3 -> (1,1,2), (1,1,3)
    # r=2: k in 1..2, l in 3..3 -> (2,1,3), (2,2,3)
    expected = [(1, 1, 2), (1, 1, 3), (2, 1, 3), (2, 2, 3)]
    got = list(iter_complementary_tuples(0, 3))
    assert got == expected


def test_iter_complementary_tuples_empty_when_no_room():
    """
    Tests that the iterator is empty if the interval is too small to fit
    the complex (r,k,l) structure.
    """
    assert list(iter_complementary_tuples(0, 1)) == []  # An interval of length 2 is too small.


def test_iter_complementary_tuples_constraints_hold():
    """
    Verifies that all yielded (r,k,l) tuples satisfy the fundamental
    ordering constraint `i < k <= r < l <= j`.
    """
    i, j = 2, 7
    for r, k, l in iter_complementary_tuples(i, j):
        assert (i < k <= r < l <= j)


# ------------------------------
# iter_inner_holes
# ------------------------------
def test_iter_inner_holes_empty_for_too_small_interval():
    """
    Tests that the iterator is empty if the interval [i,j] is too small
    to contain any inner pair (k,l) where k < l.
    """
    assert list(iter_inner_holes(0, 0)) == []
    assert list(iter_inner_holes(5, 6)) == []


def test_iter_inner_holes_no_min_returns_all_pairs_inclusive():
    """
    Tests that with `min_hole_width=0`, the iterator yields all possible
    ordered pairs (k,l) within the interval [i,j].
    """
    # For i=0, j=4, there are 5 positions. The number of ordered pairs is C(5,2) = 10.
    expected = [(0, 1), (0, 2), (0, 3), (0, 4),
                (1, 2), (1, 3), (1, 4),
                (2, 3), (2, 4),
                (3, 4)]
    got = list(iter_inner_holes(0, 4, min_hole_width=0))
    assert got == expected


def test_iter_inner_holes_min_hole_1_excludes_adjacents():
    """
    Tests that `min_hole_width=1` correctly filters out adjacent pairs.
    An adjacent pair (k, k+1) has a "hole" of width 0 between them, so they
    should be excluded.
    """
    # This is the same list as the test above, but with (0,1), (1,2), (2,3), (3,4) removed.
    expected = [(0, 2), (0, 3), (0, 4),
                (1, 3), (1, 4),
                (2, 4)]
    got = list(iter_inner_holes(0, 4, min_hole_width=1))
    assert got == expected


def test_iter_inner_holes_min_hole_too_large_yields_empty():
    """
    Tests that the iterator is empty if `min_hole_width` is too large
    to be satisfied within the given interval.
    """
    # The largest possible hole in [0,4] is 3 (for pair 0,4). A min_hole_width of 4 is impossible.
    assert list(iter_inner_holes(0, 4, min_hole_width=4)) == []


def test_iter_inner_holes_bounds_hold():
    """
    Verifies that all yielded (k,l) pairs satisfy the boundary and
    `min_hole_width` constraints.
    """
    i, j = 3, 9
    min_width = 2
    for k, l in iter_inner_holes(i, j, min_hole_width=min_width):
        # The pair (k,l) must be within the outer interval [i,j].
        assert i <= k < l <= j
        # The number of unpaired bases between k and l must meet the minimum.
        assert (l - k - 1) >= min_width
