"""
Unit tests for the sparse 4D matrix data structures.

This module validates the `SparseGapMatrix` and `SparseGapBackptr` classes.
These are specialized, dictionary-based data structures designed to efficiently
store the 4-dimensional energy and backpointer information required for folding
algorithms that handle pseudoknots (like the Rivas-Eddy algorithm). Their sparse
nature avoids the extreme memory cost of a dense 4D array.
"""
import math

from rna_pk_fold.structures.gap_matrix import SparseGapMatrix, SparseGapBackptr


def test_sparse_gap_matrix_defaults_and_bounds():
    """
    Tests the default return values and boundary condition handling of the matrix.
    An unset or out-of-bounds cell in an energy matrix should return +infinity,
    representing an infinitely costly or impossible state.
    """
    n = 6
    matrix = SparseGapMatrix(n=n)

    # An unset cell with valid coordinates should return the default value of +infinity.
    assert math.isinf(matrix.get(0, 0, 0, 0))
    assert math.isinf(matrix.get(1, 4, 2, 3))

    # --- Test boundary checks and index constraints ---
    # The matrix should return +infinity if any index violates the required ordering:
    # 0 <= i <= k < l <= j < n
    assert math.isinf(matrix.get(-1, 2, 0, 1))  # i < 0
    assert math.isinf(matrix.get(0, n, 0, 1))   # j >= n
    assert math.isinf(matrix.get(3, 2, 3, 3))   # i > j
    assert math.isinf(matrix.get(2, 5, 1, 4))   # k < i
    assert math.isinf(matrix.get(1, 4, 1, 5))   # l > j
    assert math.isinf(matrix.get(1, 4, 3, 2))   # k > l


def test_sparse_gap_matrix_set_get_roundtrip_and_overwrite():
    """
    Tests the basic integrity of the matrix's set/get functionality.
    It verifies that a value can be stored, retrieved, and correctly overwritten.
    """
    n = 6
    matrix = SparseGapMatrix(n=n)

    # Perform a simple set/get round-trip.
    matrix.set(1, 5, 2, 4, -3.25)
    assert matrix.get(1, 5, 2, 4) == -3.25

    # Because the matrix is sparse, setting one "hole" (k,l) within an outer
    # span (i,j) should not affect other holes in the same span.
    assert math.isinf(matrix.get(1, 5, 2, 3))

    # Verify that overwriting the same cell works as expected.
    matrix.set(1, 5, 2, 4, -7.0)
    assert matrix.get(1, 5, 2, 4) == -7.0


def test_sparse_gap_matrix_row_is_live_view():
    """
    Verifies that the `row()` method returns a live view, not a copy.
    Modifying the dictionary returned by `matrix.row(i, j)` should directly
    mutate the internal state of the matrix. This is an important feature for
    performance, as it avoids unnecessary data copying.
    """
    n = 5
    matrix = SparseGapMatrix(n=n)

    # Get the dictionary for the outer span (1, 3). It should be empty initially.
    row = matrix.row(1, 3)
    assert isinstance(row, dict)
    assert (2, 3) not in row

    # Mutate the dictionary directly.
    row[(2, 3)] = -1.23
    # The change should be reflected when accessing the matrix via the `get` method.
    assert matrix.get(1, 3, 2, 3) == -1.23

    # A second mutation should also be reflected.
    row[(1, 3)] = -0.5
    assert matrix.get(1, 3, 1, 3) == -0.5


def test_sparse_gap_matrix_no_collapse_identity_inside_class():
    """
    Confirms the matrix is a simple data container, not an algorithmic engine.
    The matrix class itself does NOT implement any of the DP algorithm's "collapse"
    identities (e.g., where a 4D state might reduce to a 2D state when k+1==l).
    Such logic resides in the folding engine, not the data structure.
    """
    n = 5
    matrix = SparseGapMatrix(n=n)

    # A cell whose coordinates might imply a collapse (k+1 == l) is treated
    # like any other. If unset, it remains +infinity.
    assert math.isinf(matrix.get(1, 4, 2, 3))

    # After being explicitly set, the value is returned normally.
    matrix.set(1, 4, 2, 3, -0.75)
    assert matrix.get(1, 4, 2, 3) == -0.75


def test_sparse_gap_backptr_defaults_and_roundtrip():
    """
    Tests the `SparseGapBackptr` class for default values and set/get integrity.
    This class is the backpointer equivalent of the energy matrix.
    """
    n = 6
    back_ptr = SparseGapBackptr(n=n)

    # An unset backpointer should default to None.
    assert back_ptr.get(1, 4, 2, 3) is None

    # Perform a set/get round-trip with a sample backpointer object.
    obj = {"op": "WHX_SPLIT", "args": (2,)}
    back_ptr.set(1, 4, 2, 3, obj)
    assert back_ptr.get(1, 4, 2, 3) is obj

    # Verify that overwriting works correctly.
    obj2 = {"op": "WHX_COLLAPSE"}
    back_ptr.set(1, 4, 2, 3, obj2)
    assert back_ptr.get(1, 4, 2, 3) is obj2


def test_sparse_gap_backptr_missing_or_oob_is_none():
    """
    Tests that the backpointer matrix returns `None` for any missing key.
    Unlike the energy matrix, the backpointer table does not perform boundary
    checks. It behaves like a simple multi-level dictionary, where accessing any
    key that hasn't been set (whether the coordinates are valid or not)
    simply returns `None`.
    """
    n = 4
    back_ptr = SparseGapBackptr(n=n)

    # Accessing valid but unset coordinates should return None.
    assert back_ptr.get(0, 3, 1, 2) is None
    assert back_ptr.get(0, 0, 0, 0) is None

    # Accessing "out-of-bounds" coordinates is equivalent to a missing key.
    assert back_ptr.get(-1, 2, 0, 1) is None
    assert back_ptr.get(0, 5, 1, 2) is None
    assert back_ptr.get(2, 1, 1, 2) is None
