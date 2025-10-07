import math
import pytest

from rna_pk_fold.structures import ZuckerTriMatrix
from rna_pk_fold.structures.tri_matrix import (
    RivasEddyTriMatrix,
    RivasEddyTriBackPointer,
)


# ---------------------------------------------------------------------------
# ZuckerTriMatrix (existing tests, unchanged)
# ---------------------------------------------------------------------------
def test_trimatrix_init_shape_and_defaults():
    """
    Validate TriMatrix shape and default fill behavior.
    """
    seq_len = 5
    fill = 123.45
    tri_matrix = ZuckerTriMatrix[float](seq_len, fill)

    assert tri_matrix.shape == (seq_len, seq_len)
    assert tri_matrix.size == seq_len

    for i in range(seq_len):
        for j in range(i, seq_len):
            assert tri_matrix.get(i, j) == fill


def test_trimatrix_set_get_roundtrip():
    """
    Ensure set/get updates a single cell correctly.
    """
    seq_len = 4
    tri_matrix = ZuckerTriMatrix[float](seq_len, float("inf"))

    tri_matrix.set(1, 3, -7.25)
    assert tri_matrix.get(1, 3) == -7.25

    # Unchanged neighbors still at initial value
    assert math.isinf(tri_matrix.get(0, 0))
    assert math.isinf(tri_matrix.get(1, 1))
    assert math.isinf(tri_matrix.get(0, 3))


def test_trimatrix_invalid_indices_raise():
    """
    Verify invalid index access raises IndexError for get/set.
    """
    seq_len = 3
    tri_matrix = ZuckerTriMatrix[int](seq_len, 0)

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
    """
    seq_len = 6
    tri_matrix = ZuckerTriMatrix[int](seq_len, 0)
    seen = set(tri_matrix.iter_upper_indices())

    expected_count = seq_len * (seq_len + 1) // 2
    assert len(seen) == expected_count

    for i in range(seq_len):
        for j in range(i, seq_len):
            assert (i, j) in seen


def test_trimatrix_generic_object_storage():
    """
    TriMatrix should support arbitrary value types (not just scalars).
    """
    n = 3
    tri_matrix = ZuckerTriMatrix[list](n, fill=[])
    tri_matrix.set(0, 1, ["x", 1])
    tri_matrix.set(1, 2, ["y", 2])

    assert tri_matrix.get(0, 1) == ["x", 1]
    assert tri_matrix.get(1, 2) == ["y", 2]


# ---------------------------------------------------------------------------
# RivasEddyTriMatrix
# ---------------------------------------------------------------------------
def test_re_trimatrix_defaults_and_roundtrip():
    """
    RivasEddyTriMatrix defaults to +inf; set/get roundtrip stores floats.
    """
    n = 5
    re_tri = RivasEddyTriMatrix(n=n)

    # Defaults: +inf for unset valid cells
    assert math.isinf(re_tri.get(0, 0))
    assert math.isinf(re_tri.get(2, 4))

    # Set/get roundtrip
    re_tri.set(1, 3, -2.75)
    assert re_tri.get(1, 3) == -2.75

    # Different cell remains +inf
    assert math.isinf(re_tri.get(1, 4))


def test_re_trimatrix_empty_segment_convenience():
    """
    i == j + 1 should return 0.0 even if i == n (just after end) and j == n-1.
    Any other i > j returns +inf.
    """
    n = 4
    re_tri = RivasEddyTriMatrix(n=n)

    # Exact empty segment just after diagonal → 0.0
    assert re_tri.get(1, 0) == 0.0
    assert re_tri.get(2, 1) == 0.0
    assert re_tri.get(3, 2) == 0.0

    # Boundary: i == n, j == n-1 (order check returns 0.0 before bounds)
    assert re_tri.get(n, n - 1) == 0.0

    # Other i > j are +inf
    assert math.isinf(re_tri.get(3, 1))
    assert math.isinf(re_tri.get(2, 0))


def test_re_trimatrix_out_of_bounds_are_inf_for_normal_cells():
    """
    Out-of-bounds indices (except the empty-segment convenience) return +inf.
    """
    n = 3
    re_tri = RivasEddyTriMatrix(n=n)

    assert math.isinf(re_tri.get(-1, 0))
    assert math.isinf(re_tri.get(0, n))
    assert math.isinf(re_tri.get(-2, -1))
    # For i > j case that's NOT exactly j+1 → +inf even if in range
    assert math.isinf(re_tri.get(2, 0))


def test_re_trimatrix_overwrite_values():
    """
    Setting a value twice overwrites previous value.
    """
    n = 5
    re_tri = RivasEddyTriMatrix(n=n)
    re_tri.set(0, 4, -1.0)
    assert re_tri.get(0, 4) == -1.0
    re_tri.set(0, 4, -3.5)
    assert re_tri.get(0, 4) == -3.5


# ---------------------------------------------------------------------------
# RivasEddyTriBackPointer
# ---------------------------------------------------------------------------
class _DummyBP:
    def __init__(self, tag, payload=None):
        self.tag = tag
        self.payload = payload

    def __repr__(self):
        return f"DummyBP(tag={self.tag!r})"


def test_re_tribackpointer_defaults_and_roundtrip():
    """
    Unset returns None; set/get roundtrip preserves object identity.
    """
    n = 4
    bp = RivasEddyTriBackPointer(n=n)

    # Defaults
    assert bp.get(0, 0) is None
    assert bp.get(1, 3) is None

    # Roundtrip
    v = _DummyBP("wx", payload=(1, 3))
    bp.set(1, 3, v)
    assert bp.get(1, 3) is v

    # Different cell still None
    assert bp.get(1, 2) is None


def test_re_tribackpointer_invalid_indices_return_none():
    """
    i > j or out-of-bounds → None.
    """
    n = 3
    bp = RivasEddyTriBackPointer(n=n)

    assert bp.get(-1, 0) is None
    assert bp.get(0, n) is None
    assert bp.get(2, 1) is None   # i > j
    # Note: no special i==j+1 behavior here; still None
    assert bp.get(1, 0) is None

