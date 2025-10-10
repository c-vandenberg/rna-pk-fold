"""
Unit tests for the Eddy-Rivas folding state initialization.

This module tests the `init_eddy_rivas_fold_state` factory function, which is
responsible for creating and initializing all the dynamic programming (DP)
matrices required by the folding algorithm.

The tests verify:
1.  The overall structure of the fold state object.
2.  The correct types and dimensions of all DP matrices.
3.  The correct initialization of base cases and default values in the matrices.
4.  The basic functionality (get/set) of the matrix data structures.
"""
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
    """
    Ensures the fold state is correctly constructed with all required matrices.

    This is a structural test to verify that `init_eddy_rivas_fold_state`
    builds the state object with the right component types for a given sequence
    length `n`. It also confirms the state dataclass is memory-optimized.
    """
    n = 5
    st = init_eddy_rivas_fold_state(n)

    # The state dataclass should use __slots__ for memory efficiency.
    assert hasattr(st, "__slots__")
    assert not hasattr(st, "__dict__")

    # Verify the types of the 2D triangular matrices for non-PK components.
    assert isinstance(st.wx_matrix, EddyRivasTriMatrix)
    assert isinstance(st.vx_matrix, EddyRivasTriMatrix)
    assert isinstance(st.wxi_matrix, EddyRivasTriMatrix)
    assert isinstance(st.wxu_matrix, EddyRivasTriMatrix)
    assert isinstance(st.wxc_matrix, EddyRivasTriMatrix)
    assert isinstance(st.vxu_matrix, EddyRivasTriMatrix)
    assert isinstance(st.vxc_matrix, EddyRivasTriMatrix)

    # Verify the types of the corresponding 2D backpointer tables.
    assert isinstance(st.wx_back_ptr, EddyRivasTriBackPointer)
    assert isinstance(st.vx_back_ptr, EddyRivasTriBackPointer)

    # Verify the types of the 4D sparse "gap" matrices for PK components.
    assert isinstance(st.whx_matrix, SparseGapMatrix)
    assert isinstance(st.vhx_matrix, SparseGapMatrix)
    assert isinstance(st.yhx_matrix, SparseGapMatrix)
    assert isinstance(st.zhx_matrix, SparseGapMatrix)

    # Verify the types of the corresponding 4D backpointer tables.
    assert isinstance(st.whx_back_ptr, SparseGapBackptr)
    assert isinstance(st.vhx_back_ptr, SparseGapBackptr)
    assert isinstance(st.yhx_back_ptr, SparseGapBackptr)
    assert isinstance(st.zhx_back_ptr, SparseGapBackptr)

    # Verify the sequence length is correctly stored.
    assert st.seq_len == n


def test_non_gap_base_cases_diagonal_and_defaults():
    """
    Checks initial values of the 2D matrices (base cases and defaults).

    This test ensures that the triangular matrices are initialized according to
    the base cases of the DP recurrences.
    """
    n = 4
    st = init_eddy_rivas_fold_state(n)

    # --- Test diagonal base cases (i, i), representing single-nucleotide subsequences ---
    for i in range(n):
        # WX family matrices: energy of a single nucleotide is 0 (no structure).
        assert st.wx_matrix.get(i, i) == 0.0
        assert st.wxi_matrix.get(i, i) == 0.0
        assert st.wxu_matrix.get(i, i) == 0.0
        assert st.wxc_matrix.get(i, i) == 0.0

        # VX family matrices: a single nucleotide cannot form a pair, so energy is infinite.
        assert math.isinf(st.vx_matrix.get(i, i))
        assert math.isinf(st.vxu_matrix.get(i, i))
        assert math.isinf(st.vxc_matrix.get(i, i))

    # --- Test default values for other cases ---

    # Off-diagonal entries should default to +infinity until they are computed.
    assert math.isinf(st.wx_matrix.get(0, 2))
    assert math.isinf(st.vx_matrix.get(0, 2))

    # Empty subsequences (where j < i) should have an energy of 0.0.
    assert st.wx_matrix.get(1, 0) == 0.0
    assert st.vx_matrix.get(1, 0) == 0.0

    # Accessing out-of-bounds indices should gracefully return +infinity.
    assert math.isinf(st.wx_matrix.get(-1, 0))
    assert math.isinf(st.wx_matrix.get(0, n))


def test_non_gap_backpointers_default_none_and_set_get():
    """
    Tests the initial state and basic operation of 2D backpointer matrices.
    """
    n = 3
    st = init_eddy_rivas_fold_state(n)

    # Backpointer tables should be initialized with None values.
    assert st.wx_back_ptr.get(0, 0) is None
    assert st.vx_back_ptr.get(0, 2) is None

    # Test a simple set/get round-trip. The payload can be any object.
    st.wx_back_ptr.set(0, 2, ("WX_OP", (0, 2)))
    st.vx_back_ptr.set(1, 2, ("VX_OP", (1, 2)))

    assert st.wx_back_ptr.get(0, 2) == ("WX_OP", (0, 2))
    assert st.vx_back_ptr.get(1, 2) == ("VX_OP", (1, 2))


def test_gap_matrices_initial_state_and_roundtrip():
    """
    Tests the initial state and basic operation of 4D sparse gap matrices.
    """
    n = 6
    st = init_eddy_rivas_fold_state(n)

    # Sparse matrices are implemented with dicts and should start empty (lazy fill).
    assert st.whx_matrix.data == {}
    assert st.vhx_matrix.data == {}
    assert st.yhx_matrix.data == {}
    assert st.zhx_matrix.data == {}

    # Getting an unset value should default to +infinity.
    assert math.isinf(st.whx_matrix.get(0, 5, 2, 4))
    assert math.isinf(st.vhx_matrix.get(1, 4, 2, 3))
    assert math.isinf(st.yhx_matrix.get(0, 3, 1, 2))
    assert math.isinf(st.zhx_matrix.get(2, 5, 3, 4))

    # Corresponding backpointers should default to None.
    assert st.whx_back_ptr.get(0, 5, 2, 4) is None
    assert st.vhx_back_ptr.get(1, 4, 2, 3) is None

    # Test set/get round-trip for energy values.
    st.whx_matrix.set(0, 5, 2, 4, 7.25)
    st.zhx_matrix.set(2, 5, 3, 4, -1.0)
    assert st.whx_matrix.get(0, 5, 2, 4) == 7.25
    assert st.zhx_matrix.get(2, 5, 3, 4) == -1.0

    # Test set/get round-trip for back-pointers.
    st.yhx_back_ptr.set(0, 5, 2, 4, ("YHX_OP", (0, 5, 2, 4)))
    st.zhx_back_ptr.set(2, 5, 3, 4, {"op": "ZHX_OP"})
    assert st.yhx_back_ptr.get(0, 5, 2, 4) == ("YHX_OP", (0, 5, 2, 4))
    assert st.zhx_back_ptr.get(2, 5, 3, 4) == {"op": "ZHX_OP"}


def test_setting_values_in_non_gap_matrices_roundtrip():
    """
    Provides a simple, direct test of the set/get round-trip for 2D energy matrices.
    """
    n = 5
    st = init_eddy_rivas_fold_state(n)

    # Set values in two different matrices.
    st.wx_matrix.set(0, 3, -3.5)
    st.vx_matrix.set(1, 4, 2.0)

    # Verify that the values can be retrieved correctly.
    assert st.wx_matrix.get(0, 3) == -3.5
    assert st.vx_matrix.get(1, 4) == 2.0
