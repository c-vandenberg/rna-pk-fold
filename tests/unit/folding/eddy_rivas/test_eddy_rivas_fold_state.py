import math

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import init_eddy_rivas_fold_state
from rna_pk_fold.structures.tri_matrix import (
    EddyRivasTriMatrix,
    EddyRivasTriBackPointer,
)
from rna_pk_fold.structures.gap_matrix import (
    SparseGapMatrix,
    SparseGapBackptr,
)


def test_make_state_constructs_all_matrices_and_slots():
    n = 5
    st = init_eddy_rivas_fold_state(n)

    # dataclass is slotted
    assert hasattr(st, "__slots__")
    assert not hasattr(st, "__dict__")

    # Types (2D matrices and back-pointers)
    assert isinstance(st.wx_matrix, EddyRivasTriMatrix)
    assert isinstance(st.vx_matrix, EddyRivasTriMatrix)
    assert isinstance(st.wxi_matrix, EddyRivasTriMatrix)
    assert isinstance(st.wxu_matrix, EddyRivasTriMatrix)
    assert isinstance(st.wxc_matrix, EddyRivasTriMatrix)
    assert isinstance(st.vxu_matrix, EddyRivasTriMatrix)
    assert isinstance(st.vxc_matrix, EddyRivasTriMatrix)

    assert isinstance(st.wx_back_ptr, EddyRivasTriBackPointer)
    assert isinstance(st.vx_back_ptr, EddyRivasTriBackPointer)

    # Types (4D gap matrices and back-pointers)
    assert isinstance(st.whx_matrix, SparseGapMatrix)
    assert isinstance(st.vhx_matrix, SparseGapMatrix)
    assert isinstance(st.yhx_matrix, SparseGapMatrix)
    assert isinstance(st.zhx_matrix, SparseGapMatrix)

    assert isinstance(st.whx_back_ptr, SparseGapBackptr)
    assert isinstance(st.vhx_back_ptr, SparseGapBackptr)
    assert isinstance(st.yhx_back_ptr, SparseGapBackptr)
    assert isinstance(st.zhx_back_ptr, SparseGapBackptr)

    # Dimensions
    assert st.seq_len == n


def test_non_gap_base_cases_diagonal_and_defaults():
    n = 4
    st = init_eddy_rivas_fold_state(n)

    # Diagonal base cases
    for i in range(n):
        assert st.wx_matrix.get(i, i) == 0.0
        assert st.wxi_matrix.get(i, i) == 0.0
        assert st.wxu_matrix.get(i, i) == 0.0
        assert st.wxc_matrix.get(i, i) == 0.0

        assert math.isinf(st.vx_matrix.get(i, i))
        assert math.isinf(st.vxu_matrix.get(i, i))
        assert math.isinf(st.vxc_matrix.get(i, i))

    # Off-diagonal default (+inf if unset)
    assert math.isinf(st.wx_matrix.get(0, 2))
    assert math.isinf(st.vx_matrix.get(0, 2))

    # Empty-segment convenience (i == j+1) → 0.0 per RivasEddyTriMatrix.get()
    assert st.wx_matrix.get(1, 0) == 0.0
    assert st.vx_matrix.get(1, 0) == 0.0

    # Out-of-range → +inf
    assert math.isinf(st.wx_matrix.get(-1, 0))
    assert math.isinf(st.wx_matrix.get(0, n))


def test_non_gap_backpointers_default_none_and_set_get():
    n = 3
    st = init_eddy_rivas_fold_state(n)

    # Defaults: None
    assert st.wx_back_ptr.get(0, 0) is None
    assert st.vx_back_ptr.get(0, 2) is None

    # Round-trip set/get (payload is Any; use a simple tuple)
    st.wx_back_ptr.set(0, 2, ("WX_OP", (0, 2)))
    st.vx_back_ptr.set(1, 2, ("VX_OP", (1, 2)))

    assert st.wx_back_ptr.get(0, 2) == ("WX_OP", (0, 2))
    assert st.vx_back_ptr.get(1, 2) == ("VX_OP", (1, 2))


def test_gap_matrices_initial_state_and_roundtrip():
    n = 6
    st = init_eddy_rivas_fold_state(n)

    # Data dicts should start empty (lazy fill)
    assert st.whx_matrix.data == {}
    assert st.vhx_matrix.data == {}
    assert st.yhx_matrix.data == {}
    assert st.zhx_matrix.data == {}

    # Default get() → +inf for unset/invalid
    assert math.isinf(st.whx_matrix.get(0, 5, 2, 4))
    assert math.isinf(st.vhx_matrix.get(1, 4, 2, 3))
    assert math.isinf(st.yhx_matrix.get(0, 3, 1, 2))
    assert math.isinf(st.zhx_matrix.get(2, 5, 3, 4))

    # Backptrs default None
    assert st.whx_back_ptr.get(0, 5, 2, 4) is None
    assert st.vhx_back_ptr.get(1, 4, 2, 3) is None

    # Round-trip set/get for energies
    st.whx_matrix.set(0, 5, 2, 4, 7.25)
    st.zhx_matrix.set(2, 5, 3, 4, -1.0)
    assert st.whx_matrix.get(0, 5, 2, 4) == 7.25
    assert st.zhx_matrix.get(2, 5, 3, 4) == -1.0

    # Round-trip set/get for back-pointers
    st.yhx_back_ptr.set(0, 5, 2, 4, ("YHX_OP", (0, 5, 2, 4)))
    st.zhx_back_ptr.set(2, 5, 3, 4, {"op": "ZHX_OP"})
    assert st.yhx_back_ptr.get(0, 5, 2, 4) == ("YHX_OP", (0, 5, 2, 4))
    assert st.zhx_back_ptr.get(2, 5, 3, 4) == {"op": "ZHX_OP"}


def test_setting_values_in_non_gap_matrices_roundtrip():
    n = 5
    st = init_eddy_rivas_fold_state(n)

    st.wx_matrix.set(0, 3, -3.5)
    st.vx_matrix.set(1, 4, 2.0)

    assert st.wx_matrix.get(0, 3) == -3.5
    assert st.vx_matrix.get(1, 4) == 2.0
