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
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_dynamic_programming import (
    EddyRivasBackPointer,
    EddyRivasBacktrackOp,
)
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_traceback import traceback_with_pseudoknots


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
    res = traceback_with_pseudoknots(
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

    res = traceback_with_pseudoknots(
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
    Verify WX→WHX composition where both WHX subproblems collapse to nested pairs.

    This test exercises the traceback path for a PK composition that splits
    WX(0,5) at r=2 with hole (k,l)=(1,4). Under the refactored split semantics,
    the left WHX uses hole (k,r)=(1,2) over outer span (0,2), and the right WHX
    uses hole (r+1,l)=(3,4) over outer span (3,5). Each WHX then collapses and
    delegates its outer span to the nested tracer.

    Notes
    -----
    The mock nested tracer used by these tests returns a single base pair
    spanning the interval it is asked to trace, i.e. Pair(i, j). Therefore:
      * Collapsing WHX(0,2:1,2) yields Pair(0,2).
      * Collapsing WHX(3,5:3,4) yields Pair(3,5).

    The test asserts that the final set of pairs equals {(0,2), (3,5)} and
    that the dot–bracket string marks those indices as paired.
    """
    n = 6
    seq = "GCAUGC"
    re_state = init_eddy_rivas_fold_state(n)

    # WX(0,5) composed with split r=2 and hole (k,l)=(1,4)
    re_state.wx_back_ptr.set_backpointer(
        0, 5,
        EddyRivasBackPointer(
            op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX,
            split=2, hole=(1, 4)
        )
    )

    # Left WHX uses hole (k, r) = (1, 2), outer span is (0,2)
    re_state.whx_back_ptr.set_backpointer(
        0, 2, 1, 2,
        EddyRivasBackPointer(
            op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE,
            outer=(0, 2)  # match actual collapsed interval
        )
    )

    # Right WHX uses hole (r+1, l) = (3, 4), outer span is (3,5)
    re_state.whx_back_ptr.set_backpointer(
        3, 5, 3, 4,
        EddyRivasBackPointer(
            op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE,
            outer=(3, 5)  # match actual collapsed interval
        )
    )

    res = traceback_with_pseudoknots(
        seq=seq,
        nested_state=object(),
        eddy_rivas_fold_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )

    # The nested tracer returns Pair(i,j) for each collapsed WHX outer span:
    assert set(res.pairs) == {Pair(0, 2), Pair(3, 5)}
    assert len(res.dot_bracket) == n
    assert res.dot_bracket[0] != "." and res.dot_bracket[2] != "."
    assert res.dot_bracket[3] != "." and res.dot_bracket[5] != "."




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
    re_state.wx_back_ptr.set_backpointer(
        0, 5,
        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_OVERLAP,
                             split=2, hole=(k, l))
    )
    # No further backpointers are needed, as the YHX handler adds (k,l)
    # before consulting any subproblem backpointers.

    res = traceback_with_pseudoknots(
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
    Verify the YHX→WHX→collapse traceback chain under new split semantics.

    Starting from WX(0,5) with split r=2 and hole (k,l)=(1,4), the refactored
    semantics create YHX frames with inner pairs (1,2) on the left and (3,4)
    on the right. The left YHX wraps into WHX(0,2:1,2), which collapses and
    delegates (0,2) to the nested tracer.

    Notes
    -----
    With the mock nested tracer returning Pair(i, j), collapsing
    WHX(0,2:1,2) contributes Pair(0,2). The YHX handler places the inner pairs
    directly:
      * Left YHX(0,2:1,2) → Pair(1,2)
      * Right YHX(3,5:3,4) → Pair(3,4)

    The test asserts that the final set of pairs equals {(1,2), (3,4), (0,2)}
    and that the dot–bracket string marks those indices as paired.
    """
    n = 6
    seq = "GCAUGC"
    re_state = init_eddy_rivas_fold_state(n)

    i, j = 0, 5
    r, k, l = 2, 1, 4

    # WX -> YHX with split r=2, hole (k,l)=(1,4)
    re_state.wx_back_ptr.set_backpointer(
        i, j,
        EddyRivasBackPointer(
            op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX,
            split=r, hole=(k, l)
        )
    )

    # Left YHX frame is YHX(i, r : k, r) = YHX(0, 2 : 1, 2); it wraps into WHX
    re_state.yhx_back_ptr.set_backpointer(
        i, r, k, r,
        EddyRivasBackPointer(
            op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX,
            outer=(i, r), hole=(k, r)
        )
    )

    # WHX collapse merges its outer span (0,2); nested tracer returns Pair(0,2)
    re_state.whx_back_ptr.set_backpointer(
        i, r, k, r,
        EddyRivasBackPointer(
            op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE,
            outer=(i, r)  # (0,2) to mirror actual collapse interval
        )
    )

    res = traceback_with_pseudoknots(
        seq=seq,
        nested_state=object(),
        eddy_rivas_fold_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )

    # Under split semantics we expect:
    # - (1,2) from left YHX inner pair
    # - (3,4) from right YHX inner pair
    # - (0,2) from WHX collapse→nested tracer over WHX outer span
    assert set(res.pairs) == {Pair(1, 2), Pair(3, 4), Pair(0, 2)}

    assert len(res.dot_bracket) == n
    # YHX inner pairs:
    assert res.dot_bracket[1] != "." and res.dot_bracket[2] != "."
    assert res.dot_bracket[3] != "." and res.dot_bracket[4] != "."
    # Collapsed WHX outer span:
    assert res.dot_bracket[0] != "." and res.dot_bracket[2] != "."

