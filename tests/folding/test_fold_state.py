import math

from rna_pk_fold.folding.fold_state import FoldState, make_fold_state
from rna_pk_fold.folding.back_pointer import BackPointer, BacktrackOp


def test_make_fold_state_shapes_and_defaults():
    """
    Create a FoldState and verify matrix sizes and default initialization.

    Expected
    --------
    - W/V matrices have shape (N, N) and default to `+∞`.
    - Back-pointer matrices have shape (N, N) and default to `BackPointer()` with `operation=NONE`.

    Notes
    -----
    - Energy matrices should be +inf by default.
    - Back-pointer matrices should be BackPointer() with operation NONE.
    """
    seq_len = 8
    fold_state = make_fold_state(seq_len)

    # Types
    assert isinstance(fold_state, FoldState)

    # Shapes
    assert fold_state.w_matrix.shape == (seq_len, seq_len)
    assert fold_state.v_matrix.shape == (seq_len, seq_len)
    assert fold_state.w_back_ptr.shape == (seq_len, seq_len)
    assert fold_state.v_back_ptr.shape == (seq_len, seq_len)

    # Default values
    for i in range(seq_len):
        for j in range(i, seq_len):
            assert math.isinf(fold_state.w_matrix.get(i, j))
            assert math.isinf(fold_state.v_matrix.get(i, j))

            bp_w = fold_state.w_back_ptr.get(i, j)
            bp_v = fold_state.v_back_ptr.get(i, j)
            assert isinstance(bp_w, BackPointer) and bp_w.operation is BacktrackOp.NONE
            assert isinstance(bp_v, BackPointer) and bp_v.operation is BacktrackOp.NONE


def test_fold_state_set_get_energy_and_backpointer():
    """
    Round-trip set/get on both energy and back-pointer matrices.

    Expected
    --------
    - Values set via `set(i,j,...)` are retrieved identically via `get(i,j)`.
    - Back-pointers round-trip equivalently.
    """
    seq_len = 5
    st = make_fold_state(seq_len)

    # Set energies
    st.w_matrix.set(1, 4, -3.25)
    st.v_matrix.set(2, 3, -1.5)

    # Set back-pointers
    bp_w = BackPointer(operation=BacktrackOp.UNPAIRED_LEFT, note="left unpaired")
    bp_v = BackPointer(operation=BacktrackOp.STACK, inner=(3, 6))

    st.w_back_ptr.set(1, 4, bp_w)
    st.v_back_ptr.set(2, 3, bp_v)

    # Verify round-trip
    assert st.w_matrix.get(1, 4) == -3.25
    assert st.v_matrix.get(2, 3) == -1.5

    assert st.w_back_ptr.get(1, 4) == bp_w
    assert st.v_back_ptr.get(2, 3) == bp_v
