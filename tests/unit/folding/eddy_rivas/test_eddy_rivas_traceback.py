"""
Unit tests for the Rivas-Eddy traceback algorithm.

This module tests the `traceback_with_pk` function, which reconstructs the final
RNA structure (including base pairs and dot-bracket notation) by walking through
the backpointer matrices populated during the dynamic programming phase.

The tests simulate various backpointer paths to ensure that different logical
cases in the traceback—such as falling back to a nested structure, composing
pseudoknotted elements, and handling terminal operations—are processed correctly.
"""
from rna_pk_fold.structures import Pair
from rna_pk_fold.folding.common_traceback import TraceResult
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import init_eddy_rivas_fold_state
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences import (
    EddyRivasBackPointer,
    EddyRivasBacktrackOp,
)
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_traceback import traceback_with_pk


# ----------------- Helpers -----------------

def make_nested_tracer():
    """
    Creates a mock tracer for secondary structure (nested) intervals.

    The main traceback algorithm is a hybrid: it uses Rivas-Eddy backpointers for
    pseudoknotted regions and delegates purely nested regions to a standard
    secondary structure tracer. This helper provides a simple, predictable mock
    of that nested tracer for testing purposes.

    Returns:
        A function that, for a given interval (i, j), returns a TraceResult
        containing a single base pair spanning that interval.
    """
    def _trace(seq, nested_state, i, j):
        # If i < j, return a pair (i, j). Otherwise, return no pairs.
        # The dot_bracket string is ignored by the top-level pk traceback.
        return TraceResult(
            pairs=[Pair(i, j)] if i < j else [],
            dot_bracket=""
        )
    return _trace


# ----------------- Tests -----------------

def test_empty_sequence_returns_empty_result():
    """
    Tests the base case of an empty sequence, which should yield an empty result.
    """
    re_state = init_eddy_rivas_fold_state(0)  # n=0 triggers an early return.
    res = traceback_with_pk(
        seq="",
        nested_state=object(),  # The nested state is not used in this path.
        eddy_rivas_fold_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )
    assert res.pairs == []
    assert res.dot_bracket == ""


def test_wx_fallback_to_nested_merges_pairs():
    """
    Tests the fallback case where no WX backpointer exists.

    If the top-level `WX` matrix has no backpointer for an interval, the algorithm
    should treat that interval as a purely nested structure and delegate its
    traceback to the provided `trace_nested_interval` function.
    """
    seq = "GC"
    # An empty state has no backpointers, so get(0,1) will return None.
    re_state = init_eddy_rivas_fold_state(len(seq))

    res = traceback_with_pk(
        seq=seq,
        nested_state=object(),
        eddy_rivas_fold_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )

    # The mock nested tracer should have returned Pair(0, 1).
    assert res.pairs == [Pair(0, 1)]
    assert len(res.dot_bracket) == len(seq)
    # The dot-bracket should correctly render the pair.
    assert res.dot_bracket[0] != "."
    assert res.dot_bracket[1] != "."


def test_wx_compose_whx_two_collapses_yield_two_disjoint_pairs_across_layers():
    """
    Tests a path: WX composes into two WHX subproblems, each collapsing to nested.

    This simulates a traceback for a structure with two disjoint pseudoknotted
    helices. The path is:
    1. WX(0,5) -> RE_PK_COMPOSE_WX -> Pushes WHX(0,2...) and WHX(2,5...).
    2. Each WHX frame -> RE_WHX_COLLAPSE -> Delegates to nested tracer.
    The final structure should contain the two pairs from the two collapses.
    """
    n = 6
    seq = "GCAUGC"
    re_state = init_eddy_rivas_fold_state(n)

    # 1. Set the WX backpointer to split into two WHX subproblems.
    re_state.wx_back_ptr.set(
        0, 5,
        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX,
                             split=2, hole=(1, 4))
    )

    # 2. Set the left WHX subproblem to collapse to a nested pair (0, 1).
    re_state.whx_back_ptr.set(
        0, 2, 1, 4,
        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE,
                             outer=(0, 1))
    )
    # 3. Set the right WHX subproblem to collapse to a nested pair (4, 5).
    re_state.whx_back_ptr.set(
        2, 5, 3, 3,
        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE,
                             outer=(4, 5))
    )

    res = traceback_with_pk(
        seq=seq,
        nested_state=object(),
        eddy_rivas_fold_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )

    # The result should contain exactly the two pairs from the collapse operations.
    assert set(res.pairs) == {Pair(0, 1), Pair(4, 5)}
    assert len(res.dot_bracket) == n
    # Verify that both pairs are correctly rendered in the dot-bracket string.
    assert res.dot_bracket[0] != "." and res.dot_bracket[1] != "."
    assert res.dot_bracket[4] != "." and res.dot_bracket[5] != "."


def test_wx_compose_yhx_overlap_adds_inner_pair_once():
    """
    Tests the YHX overlap case, which should add the central pair exactly once.

    The `RE_PK_COMPOSE_WX_YHX_OVERLAP` operation implies the formation of the
    innermost pseudoknot pair (k,l). The traceback handler for this operation
    should add this pair directly and ensure it's not added again, even though
    two YHX frames are pushed to the stack.
    """
    n = 6
    seq = "GCAUGC"
    re_state = init_eddy_rivas_fold_state(n)

    k, l = 1, 4
    # Set the backpointer for the overlap composition.
    re_state.wx_back_ptr.set(
        0, 5,
        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_OVERLAP,
                             split=2, hole=(k, l))
    )
    # No further backpointers are needed, as the YHX handler adds (k,l)
    # before consulting any subproblem backpointers.

    res = traceback_with_pk(
        seq=seq,
        nested_state=object(),
        eddy_rivas_fold_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )

    # The result should contain only the central pair (k, l).
    assert res.pairs == [Pair(k, l)]
    assert len(res.dot_bracket) == n
    assert res.dot_bracket[k] != "." and res.dot_bracket[l] != "."


def test_yhx_wraps_into_whx_then_collapses_adding_both_inner_and_nested_pairs():
    """
    Tests a multi-step path: WX -> YHX -> WHX -> Collapse.

    This test verifies a complex traceback chain:
    1. A WX op composes a YHX subproblem.
    2. The YHX op adds its inner pair (k,l) and wraps into a WHX subproblem.
    3. The WHX op collapses, delegating an `outer` interval to the nested tracer.
    The final structure should contain pairs from all relevant steps.
    """
    n = 6
    seq = "GCAUGC"
    re_state = init_eddy_rivas_fold_state(n)

    i, j = 0, 5
    r, k, l = 2, 1, 4
    outer_nested = (0, 1)  # The pair to be added by the final collapse.

    # 1. WX -> YHX. This pushes two YHX frames. We'll trace the left one.
    # The right one, YHX(k+1..j, l-1..r+1) = YHX(2..5, 3..3), will also be traced,
    # adding its own inner pair (3,3).
    re_state.wx_back_ptr.set(
        i, j,
        EddyRivasBackPointer(
            op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX,
            split=r, hole=(k, l)
        )
    )

    # 2. YHX -> WHX. The YHX handler adds inner pair (k,l) and pushes a WHX frame.
    re_state.yhx_back_ptr.set(
        i, r, k, l,
        EddyRivasBackPointer(
            op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX,
            outer=(i, r), hole=(k, l)
        )
    )

    # 3. WHX -> Collapse. The WHX handler delegates the `outer` to the nested tracer.
    re_state.whx_back_ptr.set(
        i, r, k, l,
        EddyRivasBackPointer(
            op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE,
            outer=outer_nested
        )
    )

    res = traceback_with_pk(
        seq=seq,
        nested_state=object(),
        eddy_rivas_fold_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )

    # The result should contain three pairs:
    # 1. Pair(k,l) = (1,4) from the YHX step.
    # 2. Pair(*outer_nested) = (0,1) from the WHX collapse.
    # 3. Pair(3,3) from the second, untraced YHX branch's inner pair.
    assert set(res.pairs) == {Pair(k, l), Pair(*outer_nested), Pair(3, 3)}

    # Check dot-bracket rendering for the two proper pairs.
    assert len(res.dot_bracket) == n
    assert res.dot_bracket[k] != "." and res.dot_bracket[l] != "."
    p, q = outer_nested
    assert res.dot_bracket[p] != "." and res.dot_bracket[q] != "."

