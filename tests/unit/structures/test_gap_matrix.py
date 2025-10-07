import math

from rna_pk_fold.structures.gap_matrix import SparseGapMatrix, SparseGapBackptr


def test_sparse_gap_matrix_defaults_and_bounds():
    n = 6
    M = SparseGapMatrix(n=n)

    # Unset but valid coordinates -> +inf
    assert math.isinf(M.get(0, 0, 0, 0))
    assert math.isinf(M.get(1, 4, 2, 3))

    # Bounds / triangular constraints -> +inf
    assert math.isinf(M.get(-1, 2, 0, 1))   # i < 0
    assert math.isinf(M.get(0, n, 0, 1))    # j >= n
    assert math.isinf(M.get(3, 2, 3, 3))    # i > j
    assert math.isinf(M.get(2, 5, 1, 4))    # k < i
    assert math.isinf(M.get(1, 4, 1, 5))    # l > j
    assert math.isinf(M.get(1, 4, 3, 2))    # k > l


def test_sparse_gap_matrix_set_get_roundtrip_and_overwrite():
    n = 6
    M = SparseGapMatrix(n=n)

    M.set(1, 5, 2, 4, -3.25)
    assert M.get(1, 5, 2, 4) == -3.25

    # Different hole in same (i,j) still +inf
    assert math.isinf(M.get(1, 5, 2, 3))

    # Overwrite same cell
    M.set(1, 5, 2, 4, -7.0)
    assert M.get(1, 5, 2, 4) == -7.0


def test_sparse_gap_matrix_row_is_live_view():
    n = 5
    M = SparseGapMatrix(n=n)

    row = M.row(1, 3)          # creates/returns backing dict
    assert isinstance(row, dict)
    assert (2, 3) not in row

    row[(2, 3)] = -1.23        # mutate through live view
    assert M.get(1, 3, 2, 3) == -1.23

    # Adding another entry through row reflects via get()
    row[(1, 3)] = -0.5
    assert M.get(1, 3, 1, 3) == -0.5


def test_sparse_gap_matrix_no_collapse_identity_inside_class():
    """
    The class itself does NOT implement collapse identities for k+1==l.
    If not explicitly set, such entries remain +inf.
    """
    n = 5
    M = SparseGapMatrix(n=n)

    # k+1 == l but not set -> still +inf (no collapse here)
    assert math.isinf(M.get(1, 4, 2, 3))

    # Once set, value is returned normally
    M.set(1, 4, 2, 3, -0.75)
    assert M.get(1, 4, 2, 3) == -0.75


def test_sparse_gap_backptr_defaults_and_roundtrip():
    n = 6
    B = SparseGapBackptr(n=n)

    # Default None
    assert B.get(1, 4, 2, 3) is None

    obj = {"op": "WHX_SPLIT", "args": (2,)}
    B.set(1, 4, 2, 3, obj)
    assert B.get(1, 4, 2, 3) is obj

    # Overwrite
    obj2 = {"op": "WHX_COLLAPSE"}
    B.set(1, 4, 2, 3, obj2)
    assert B.get(1, 4, 2, 3) is obj2


def test_sparse_gap_backptr_missing_or_oob_is_none():
    """
    SparseGapBackptr does not enforce bounds; missing keys simply return None.
    """
    n = 4
    B = SparseGapBackptr(n=n)

    # Missing rows/holes -> None
    assert B.get(0, 3, 1, 2) is None
    assert B.get(0, 0, 0, 0) is None

    # "Out-of-bounds" values are just absent keys -> None
    assert B.get(-1, 2, 0, 1) is None
    assert B.get(0, 5, 1, 2) is None
    assert B.get(2, 1, 1, 2) is None
