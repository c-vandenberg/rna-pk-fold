import math

from rna_pk_fold.folding.zucker.zucker_fold_state import ZuckerFoldState, make_fold_state
from rna_pk_fold.folding.zucker.zucker_back_pointer import ZuckerBackPointer, ZuckerBacktrackOp


def test_make_fold_state_shapes_and_defaults():
    """
    Create a FoldState and verify matrix sizes and default initialization.

    Expected
    --------
    - W, V, and WM matrices have shape (N, N).
    - W and V matrices default to `+∞`.
    - WM default to +∞ off-diagonal, and 0.0 on the diagonal.
    - All back-pointer matrices default to BackPointer() with operation NONE.

    Notes
    -----
    - WM is a multiloop accumulator; the diagonal WM[i,i] is seeded as 0.0.
    """
    seq_len = 8
    fold_state = make_fold_state(seq_len)

    # Types
    assert isinstance(fold_state, ZuckerFoldState)

    # Shapes
    assert fold_state.w_matrix.shape == (seq_len, seq_len)
    assert fold_state.v_matrix.shape == (seq_len, seq_len)
    assert fold_state.wm_matrix.shape == (seq_len, seq_len)
    assert fold_state.w_back_ptr.shape == (seq_len, seq_len)
    assert fold_state.v_back_ptr.shape == (seq_len, seq_len)
    assert fold_state.wm_back_ptr.shape == (seq_len, seq_len)

    # Default values
    for i in range(seq_len):
        for j in range(i, seq_len):
            assert math.isinf(fold_state.w_matrix.get(i, j))
            assert math.isinf(fold_state.v_matrix.get(i, j))

            # WM default: Diagonal 0.0, off-diagonal +inf
            if i == j:
                assert fold_state.wm_matrix.get(i, j) == 0.0
            else:
                assert math.isinf(fold_state.wm_matrix.get(i, j))

            bp_w = fold_state.w_back_ptr.get(i, j)
            bp_v = fold_state.v_back_ptr.get(i, j)
            bp_wm = fold_state.wm_back_ptr.get(i, j)
            assert isinstance(bp_w, ZuckerBackPointer) and bp_w.operation is ZuckerBacktrackOp.NONE
            assert isinstance(bp_v, ZuckerBackPointer) and bp_v.operation is ZuckerBacktrackOp.NONE
            assert isinstance(bp_wm, ZuckerBackPointer) and bp_wm.operation is ZuckerBacktrackOp.NONE


def test_fold_state_set_get_energy_and_backpointer():
    """
    Round-trip set/get on both energy and back-pointer matrices.

    Expected
    --------
    - Values set via `set(i,j,...)` are retrieved identically via `get(i,j)`.
    - Back-pointers round-trip equivalently.
    """
    seq_len = 5
    fold_state = make_fold_state(seq_len)

    # Set energies
    fold_state.w_matrix.set(1, 4, -3.25)
    fold_state.v_matrix.set(2, 3, -1.5)
    fold_state.wm_matrix.set(0, 4, 7.0)

    # Set back-pointers
    bp_w = ZuckerBackPointer(operation=ZuckerBacktrackOp.UNPAIRED_LEFT, note="left unpaired")
    bp_v = ZuckerBackPointer(operation=ZuckerBacktrackOp.STACK, inner=(3, 6))
    bp_wm = ZuckerBackPointer(operation=ZuckerBacktrackOp.MULTI_ATTACH, split_k=2)

    fold_state.w_back_ptr.set(1, 4, bp_w)
    fold_state.v_back_ptr.set(2, 3, bp_v)
    fold_state.wm_back_ptr.set(0, 4, bp_wm)

    # Verify round-trip
    assert fold_state.w_matrix.get(1, 4) == -3.25
    assert fold_state.v_matrix.get(2, 3) == -1.5
    assert fold_state.wm_matrix.get(0, 4) == 7.0

    assert fold_state.w_back_ptr.get(1, 4) == bp_w
    assert fold_state.v_back_ptr.get(2, 3) == bp_v
    assert fold_state.wm_back_ptr.get(0, 4) == bp_wm
