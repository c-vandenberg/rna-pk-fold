import math
import pytest

from rna_pk_fold.structures import CoreTriMatrix


def test_trimatrix_init_shape_and_defaults():
    """
    Validate TriMatrix shape and default fill behavior.

    Expected
    --------
    - `shape` equals `(N, N)` and `size == N`.
    - All `i <= j` cells are initialized to the fill value.

    Notes
    -----
    - Shape should be (N, N).
    - All i <= j cells should be initialized to the fill value.
    """
    seq_len = 5
    fill = 123.45
    tri_matrix = CoreTriMatrix[float](seq_len, fill)

    assert tri_matrix.shape == (seq_len, seq_len)
    assert tri_matrix.size == seq_len

    for i in range(seq_len):
        for j in range(i, seq_len):
            assert tri_matrix.get(i, j) == fill


def test_trimatrix_set_get_roundtrip():
    """
    Ensure set/get updates a single cell correctly.

    Expected
    --------
    - Setting `(1,3)` to a value is retrieved exactly via `get(1,3)`.
    - Other cells remain at the initial fill value.
    """
    seq_len = 4
    tri_matrix = CoreTriMatrix[float](seq_len, float("inf"))

    tri_matrix.set(1, 3, -7.25)
    assert tri_matrix.get(1, 3) == -7.25

    # Unchanged neighbors still at initial value
    assert math.isinf(tri_matrix.get(0, 0))
    assert math.isinf(tri_matrix.get(1, 1))
    assert math.isinf(tri_matrix.get(0, 3))


def test_trimatrix_invalid_indices_raise():
    """
    Verify invalid index access raises IndexError.

    Expected
    --------
    - Accessing any of the listed invalid index pairs raises `IndexError`
      for both `get` and `set`.

    Cases
    -----
    - i < 0 or j < 0
    - i >= N or j >= N
    - j < i (lower triangle is not addressable)
    """
    seq_len = 3
    tri_matrix = CoreTriMatrix[int](seq_len, 0)

    bad_indices = [
        (-1, 0),  # i < 0
        (0, -1),  # j < 0
        (3, 0),   # i >= N
        (0, 3),   # j >= N
        (2, 1),   # j < i
    ]

    for i, j in bad_indices:
        with pytest.raises(IndexError):
            tri_matrix.get(i, j)
        with pytest.raises(IndexError):
            tri_matrix.set(i, j, 1)


def test_trimatrix_iter_upper_indices_count_and_coverage():
    """
    Ensure iter_upper_indices() yields exactly N(N+1)/2 cells and covers all i<=j.

    Expected
    --------
    - Set of yielded pairs equals all `(i,j)` with `0 <= i <= j < N`
      and has cardinality `N(N+1)/2`.
    """
    seq_len = 6
    tri_matrix = CoreTriMatrix[int](seq_len, 0)
    seen = set(tri_matrix.iter_upper_indices())

    expected_count = seq_len * (seq_len + 1) // 2
    assert len(seen) == expected_count

    for i in range(seq_len):
        for j in range(i, seq_len):
            assert (i, j) in seen


def test_trimatrix_generic_object_storage():
    """
    TriMatrix should support arbitrary value types (not just scalars).

    Expected
    --------
    - Objects (e.g., lists) can be stored and retrieved unchanged.
    """
    n = 3
    tri_matrix = CoreTriMatrix[list](n, fill=[])
    tri_matrix.set(0, 1, ["x", 1])
    tri_matrix.set(1, 2, ["y", 2])

    assert tri_matrix.get(0, 1) == ["x", 1]
    assert tri_matrix.get(1, 2) == ["y", 2]
