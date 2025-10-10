"""
Unit tests for the triangular matrix data structures.

This module validates the behavior of the specialized upper-triangular matrix
classes used in the Zucker and Rivas-Eddy folding algorithms. These data
structures are optimized for dynamic programming tables where only indices
(i, j) with `j >= i` are relevant.
"""
import math
import pytest

from rna_pk_fold.structures import ZuckerTriMatrix
from rna_pk_fold.structures.tri_matrix import (
    EddyRivasTriMatrix,
    EddyRivasTriBackPointer,
)


# ---------------------------------------------------------------------------
# ZuckerTriMatrix (a strict, error-raising triangular matrix)
# ---------------------------------------------------------------------------
def test_trimatrix_init_shape_and_defaults():
    """
    Tests the constructor, shape, and default fill behavior of `ZuckerTriMatrix`.
    """
    seq_len = 5
    fill = 123.45
    tri_matrix = ZuckerTriMatrix[float](seq_len, fill)

    # Verify the reported shape and size.
    assert tri_matrix.shape == (seq_len, seq_len)
    assert tri_matrix.size == seq_len

    # All cells in the upper triangle should be initialized with the fill value.
    for i in range(seq_len):
        for j in range(i, seq_len):
            assert tri_matrix.get(i, j) == fill


def test_trimatrix_set_get_roundtrip():
    """
    Ensures that `set()` and `get()` work correctly for a single cell.
    This test verifies that a value can be stored and retrieved without
    affecting neighboring cells.
    """
    seq_len = 4
    tri_matrix = ZuckerTriMatrix[float](seq_len, float("inf"))

    # Set a single cell's value.
    tri_matrix.set(1, 3, -7.25)
    assert tri_matrix.get(1, 3) == -7.25

    # Verify that other cells remain at their initial default value.
    assert math.isinf(tri_matrix.get(0, 0))
    assert math.isinf(tri_matrix.get(1, 1))
    assert math.isinf(tri_matrix.get(0, 3))


def test_trimatrix_invalid_indices_raise():
    """
    Tests that accessing invalid indices raises an `IndexError`.
    `ZuckerTriMatrix` enforces strict boundary checks: indices must be within
    the matrix dimensions and in the upper triangle (i <= j).
    """
    seq_len = 3
    tri_matrix = ZuckerTriMatrix[int](seq_len, 0)

    # A list of coordinates that should be invalid.
    bad_indices = [
        (-1, 0),  # i < 0
        (0, -1),  # j < 0
        (3, 0),   # i >= N
        (0, 3),   # j >= N
        (2, 1),   # j < i (lower triangle)
    ]

    # Both get() and set() should raise an error for each invalid index pair.
    for i, j in bad_indices:
        with pytest.raises(IndexError):
            tri_matrix.get(i, j)
        with pytest.raises(IndexError):
            tri_matrix.set(i, j, 1)


def test_trimatrix_iter_upper_indices_count_and_coverage():
    """
    Tests the `iter_upper_indices()` method.
    This iterator should yield every valid (i, j) coordinate pair in the upper
    triangle of the matrix exactly once.
    """
    seq_len = 6
    tri_matrix = ZuckerTriMatrix[int](seq_len, 0)
    seen = set(tri_matrix.iter_upper_indices())

    # The total number of cells in an upper triangular matrix of size N is N*(N+1)/2.
    expected_count = seq_len * (seq_len + 1) // 2
    assert len(seen) == expected_count

    # Verify that all expected indices were present in the iterator's output.
    for i in range(seq_len):
        for j in range(i, seq_len):
            assert (i, j) in seen


def test_trimatrix_generic_object_storage():
    """
    Verifies that `ZuckerTriMatrix` can store arbitrary object types, not just scalars.
    This is essential for backpointer matrices, which store complex objects.
    """
    n = 3
    # Initialize with an empty list as the default value.
    tri_matrix = ZuckerTriMatrix[list](n, fill=[])
    tri_matrix.set(0, 1, ["x", 1])
    tri_matrix.set(1, 2, ["y", 2])

    assert tri_matrix.get(0, 1) == ["x", 1]
    assert tri_matrix.get(1, 2) == ["y", 2]


# ---------------------------------------------------------------------------
# EddyRivasTriMatrix (a more lenient matrix with special conventions)
# ---------------------------------------------------------------------------
def test_re_trimatrix_defaults_and_roundtrip():
    """
    Tests the constructor and basic set/get for `EddyRivasTriMatrix`.
    This matrix variant defaults to +infinity.
    """
    n = 5
    re_tri = EddyRivasTriMatrix(n=n)

    # Unset cells should default to +infinity.
    assert math.isinf(re_tri.get(0, 0))
    assert math.isinf(re_tri.get(2, 4))

    # Test a simple set/get round-trip.
    re_tri.set(1, 3, -2.75)
    assert re_tri.get(1, 3) == -2.75

    # Ensure other cells were not affected.
    assert math.isinf(re_tri.get(1, 4))


def test_re_trimatrix_empty_segment_convenience():
    """
    Tests the special "empty segment" rule for `get()`.
    In the Rivas & Eddy algorithm, accessing an interval (i, j) where `i = j + 1`
    represents an empty subsequence, which has an energy of 0.0 by definition.
    This matrix implements that convenience directly.
    """
    n = 4
    re_tri = EddyRivasTriMatrix(n=n)

    # Accessing (j+1, j) should return 0.0.
    assert re_tri.get(1, 0) == 0.0
    assert re_tri.get(2, 1) == 0.0
    assert re_tri.get(3, 2) == 0.0

    # This rule also applies at the boundary of the matrix.
    assert re_tri.get(n, n - 1) == 0.0

    # Other cases where i > j should still return +infinity.
    assert math.isinf(re_tri.get(3, 1))
    assert math.isinf(re_tri.get(2, 0))


def test_re_trimatrix_out_of_bounds_are_inf_for_normal_cells():
    """
    Tests out-of-bounds access, which should gracefully return +infinity.
    Unlike the stricter `ZuckerTriMatrix`, this class does not raise an error,
    which can simplify the implementation of recurrence relations in the folding engine.
    """
    n = 3
    re_tri = EddyRivasTriMatrix(n=n)

    # Out-of-bounds access should return +inf.
    assert math.isinf(re_tri.get(-1, 0))
    assert math.isinf(re_tri.get(0, n))
    assert math.isinf(re_tri.get(-2, -1))
    # This behavior applies to all invalid indices except the empty-segment case.
    assert math.isinf(re_tri.get(2, 0))


def test_re_trimatrix_overwrite_values():
    """
    Verifies that setting a value in the same cell twice overwrites the old value.
    """
    n = 5
    re_tri = EddyRivasTriMatrix(n=n)
    re_tri.set(0, 4, -1.0)
    assert re_tri.get(0, 4) == -1.0
    re_tri.set(0, 4, -3.5)
    assert re_tri.get(0, 4) == -3.5


# ---------------------------------------------------------------------------
# EddyRivasTriBackPointer
# ---------------------------------------------------------------------------
class _DummyBP:
    """A simple mock object to represent a backpointer for testing."""
    def __init__(self, tag, payload=None):
        self.tag = tag
        self.payload = payload

    def __repr__(self):
        return f"DummyBP(tag={self.tag!r})"


def test_re_tribackpointer_defaults_and_roundtrip():
    """
    Tests the backpointer matrix for default values and set/get integrity.
    Unset cells should return `None`, and it should preserve object identity.
    """
    n = 4
    bp = EddyRivasTriBackPointer(n=n)

    # Unset cells should default to None, indicating no path found yet.
    assert bp.get(0, 0) is None
    assert bp.get(1, 3) is None

    # Test set/get round-trip, verifying object identity with `is`.
    v = _DummyBP("wx", payload=(1, 3))
    bp.set(1, 3, v)
    assert bp.get(1, 3) is v

    # A different cell should remain None.
    assert bp.get(1, 2) is None


def test_re_tribackpointer_invalid_indices_return_none():
    """
    Tests that accessing invalid or out-of-bounds indices returns `None`.
    The backpointer matrix gracefully handles all invalid indices by returning
    `None`, signifying the absence of a traceback path from that cell.
    """
    n = 3
    bp = EddyRivasTriBackPointer(n=n)

    # Out-of-bounds access should return None.
    assert bp.get(-1, 0) is None
    assert bp.get(0, n) is None
    # Lower-triangle access (i > j) should return None.
    assert bp.get(2, 1) is None
    # Note: The "empty segment" convenience of the energy matrix does not apply here.
    assert bp.get(1, 0) is None

