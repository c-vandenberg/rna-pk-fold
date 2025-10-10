"""
Unit tests for the Zucker folding state and its factory function.

This module validates the `make_fold_state` function, which constructs the
`ZuckerFoldState` object. The `ZuckerFoldState` serves as a container for all
the dynamic programming (DP) matrices required by the Zucker secondary structure
folding algorithm. Tests in this file ensure that the state is initialized
correctly and that its underlying matrix structures function as expected.
"""
import math

from rna_pk_fold.folding.zucker.zucker_fold_state import ZuckerFoldState, make_fold_state
from rna_pk_fold.folding.zucker.zucker_back_pointer import ZuckerBackPointer, ZuckerBacktrackOp


def test_make_fold_state_shapes_and_defaults():
    """
    Verifies the structure and initial state of a newly created `ZuckerFoldState`.

    This test checks several critical properties after calling `make_fold_state`:
    1.  **Shape**: All DP matrices (for energies and backpointers) must have the
        correct dimensions corresponding to the sequence length.
    2.  **Default Energies**: Matrices must be initialized with the correct
        default values, which represent the base cases of the DP recurrences
        (e.g., +infinity for uncalculated energies, 0.0 for empty multiloops).
    3.  **Default Backpointers**: Backpointer matrices must be filled with a
        default "NONE" operation, indicating no path has been determined yet.
    """
    seq_len = 8
    fold_state = make_fold_state(seq_len)

    # --- Verify Type ---
    assert isinstance(fold_state, ZuckerFoldState)

    # --- Verify Matrix Shapes ---
    assert fold_state.w_matrix.shape == (seq_len, seq_len)
    assert fold_state.v_matrix.shape == (seq_len, seq_len)
    assert fold_state.wm_matrix.shape == (seq_len, seq_len)
    assert fold_state.w_back_ptr.shape == (seq_len, seq_len)
    assert fold_state.v_back_ptr.shape == (seq_len, seq_len)
    assert fold_state.wm_back_ptr.shape == (seq_len, seq_len)

    # --- Verify Default Initial Values ---
    for i in range(seq_len):
        for j in range(i, seq_len):
            # W (any structure) and V (paired structure) matrices default to +infinity.
            assert math.isinf(fold_state.w_matrix.get(i, j))
            assert math.isinf(fold_state.v_matrix.get(i, j))

            # WM (multiloop) has a special base case: the diagonal is 0.0.
            if i == j:
                # WM[i,i] represents an empty segment in a multiloop, with 0.0 energy.
                assert fold_state.wm_matrix.get(i, j) == 0.0
            else:
                # Off-diagonal elements are +infinity until calculated.
                assert math.isinf(fold_state.wm_matrix.get(i, j))

            # Backpointer matrices should be initialized with default "NONE" pointers.
            bp_w = fold_state.w_back_ptr.get(i, j)
            bp_v = fold_state.v_back_ptr.get(i, j)
            bp_wm = fold_state.wm_back_ptr.get(i, j)
            assert isinstance(bp_w, ZuckerBackPointer) and bp_w.operation is ZuckerBacktrackOp.NONE
            assert isinstance(bp_v, ZuckerBackPointer) and bp_v.operation is ZuckerBacktrackOp.NONE
            assert isinstance(bp_wm, ZuckerBackPointer) and bp_wm.operation is ZuckerBacktrackOp.NONE


def test_fold_state_set_get_energy_and_backpointer():
    """
    Tests the integrity of matrix data by performing a set/get round trip.

    This test ensures that values and objects stored in the DP matrices via the
    `set()` method can be retrieved identically using the `get()` method,
    confirming that the underlying data structures are working correctly.
    """
    seq_len = 5
    fold_state = make_fold_state(seq_len)

    # --- Set specific energy values in different matrices ---
    fold_state.w_matrix.set(1, 4, -3.25)
    fold_state.v_matrix.set(2, 3, -1.5)
    fold_state.wm_matrix.set(0, 4, 7.0)

    # --- Create and set custom backpointer objects ---
    bp_w = ZuckerBackPointer(operation=ZuckerBacktrackOp.UNPAIRED_LEFT, note="left unpaired")
    bp_v = ZuckerBackPointer(operation=ZuckerBacktrackOp.STACK, inner=(3, 6))
    bp_wm = ZuckerBackPointer(operation=ZuckerBacktrackOp.MULTI_ATTACH, split_k=2)

    fold_state.w_back_ptr.set(1, 4, bp_w)
    fold_state.v_back_ptr.set(2, 3, bp_v)
    fold_state.wm_back_ptr.set(0, 4, bp_wm)

    # --- Verify that retrieved values match the ones that were set ---
    assert fold_state.w_matrix.get(1, 4) == -3.25
    assert fold_state.v_matrix.get(2, 3) == -1.5
    assert fold_state.wm_matrix.get(0, 4) == 7.0

    assert fold_state.w_back_ptr.get(1, 4) is bp_w
    assert fold_state.v_back_ptr.get(2, 3) is bp_v
    assert fold_state.wm_back_ptr.get(0, 4) is bp_wm
