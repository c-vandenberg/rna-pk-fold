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
    Return a minimal nested tracer:
    trace_nested_interval(seq, nested_state, i, j) -> TraceResult
    Emits a single Pair(i, j) when i<j; nothing otherwise.
    The dot_bracket returned here is ignored by the top-level traceback.
    """
    def _trace(seq, nested_state, i, j):
        return TraceResult(
            pairs=[Pair(i, j)] if i < j else [],
            dot_bracket=""
        )
    return _trace


# ----------------- Tests -----------------
def test_empty_sequence_returns_empty_result():
    re_state = init_eddy_rivas_fold_state(0)  # n == 0 triggers early return
    res = traceback_with_pk(
        seq="",
        nested_state=object(),
        re_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )
    assert res.pairs == []
    assert res.dot_bracket == ""


def test_wx_fallback_to_nested_merges_pairs():
    """
    No WX back-pointer → treat as nested-only on [0..1].
    The nested tracer returns Pair(0,1).
    """
    seq = "GC"
    re_state = init_eddy_rivas_fold_state(len(seq))  # wx_back_ptr.get(0,1) => None

    res = traceback_with_pk(
        seq=seq,
        nested_state=object(),
        re_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )

    assert res.pairs == [Pair(0, 1)]
    assert len(res.dot_bracket) == len(seq)
    # Positions 0 and 1 must be paired (non-dot)
    assert res.dot_bracket[0] != "."
    assert res.dot_bracket[1] != "."


def test_wx_compose_whx_two_collapses_yield_two_disjoint_pairs_across_layers():
    """
    WX op = RE_PK_COMPOSE_WX → pushes two WHX frames (left on layer, right on layer+1).
    Each WHX uses RE_WHX_COLLAPSE with an explicit 'outer' → merge to nested pair.
    We choose (0,1) and (4,5) to avoid any overlap.
    """
    n = 6
    seq = "GCAUGC"
    re_state = init_eddy_rivas_fold_state(n)

    # WX(0,5) → WHX(0,2:1,4) and WHX(2,5:3,3) at layers 0 and 1 respectively
    re_state.wx_back_ptr.set(
        0, 5,
        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX,
                             split=2, hole=(1, 4))
    )

    # Left WHX collapses to outer (0,1)
    re_state.whx_back_ptr.set(
        0, 2, 1, 4,
        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE,
                             outer=(0, 1))
    )
    # Right WHX collapses to outer (4,5)
    re_state.whx_back_ptr.set(
        2, 5, 3, 3,
        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE,
                             outer=(4, 5))
    )

    res = traceback_with_pk(
        seq=seq,
        nested_state=object(),
        re_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )

    assert set(res.pairs) == {Pair(0, 1), Pair(4, 5)}
    assert len(res.dot_bracket) == n
    # Ensure both pairs are rendered (both ends non-dot)
    assert res.dot_bracket[0] != "." and res.dot_bracket[1] != "."
    assert res.dot_bracket[4] != "." and res.dot_bracket[5] != "."


def test_wx_compose_yhx_overlap_adds_inner_pair_once():
    """
    WX op = RE_PK_COMPOSE_WX_YHX_OVERLAP → pushes two YHX frames sharing the same (k,l).
    YHX block adds inner pair (k,l) immediately via add_pair_once, then (with no bp) stops.
    Since add_pair_once deduplicates, the inner pair appears exactly once.
    """
    n = 6
    seq = "GCAUGC"
    re_state = init_eddy_rivas_fold_state(n)

    k, l = 1, 4
    re_state.wx_back_ptr.set(
        0, 5,
        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_OVERLAP,
                             split=2, hole=(k, l))
    )
    # No YHX bps needed; the YHX handler adds (k,l) before consulting bp.

    res = traceback_with_pk(
        seq=seq,
        nested_state=object(),
        re_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )

    assert res.pairs == [Pair(k, l)]
    assert len(res.dot_bracket) == n
    assert res.dot_bracket[k] != "." and res.dot_bracket[l] != "."


def test_yhx_wraps_into_whx_then_collapses_adding_both_inner_and_nested_pairs():
    """
    WX op = RE_PK_COMPOSE_WX_YHX →
      YHX adds inner (k,l), then YHX op = WRAP_WHX pushes a WHX which collapses to
      a nested outer (p,q). Result includes (k,l), (p,q),
      and also the inner pair from the *second* YHX branch:
      YHX(k+1, j, l-1, r+1) → here (3,3), which the traceback adds as a Pair.
    """
    n = 6
    seq = "GCAUGC"
    re_state = init_eddy_rivas_fold_state(n)

    i, j = 0, 5
    r, k, l = 2, 1, 4
    outer_nested = (0, 1)  # WHX collapse will merge this nested pair

    # WX → YHX (left branch); the right YHX branch is also pushed and adds its inner pair.
    re_state.wx_back_ptr.set(
        i, j,
        EddyRivasBackPointer(
            op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX,
            split=r, hole=(k, l)
        )
    )

    # YHX wraps to WHX on the same coordinates
    re_state.yhx_back_ptr.set(
        i, r, k, l,
        EddyRivasBackPointer(
            op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX,
            outer=(i, r), hole=(k, l)
        )
    )

    # WHX collapses to nested (0,1)
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
        re_state=re_state,
        trace_nested_interval=make_nested_tracer(),
    )

    # Expect: (k,l), (outer_nested), and the second YHX branch’s inner pair (3,3)
    assert set(res.pairs) == {Pair(k, l), Pair(*outer_nested), Pair(3, 3)}

    # Keep the rendering checks only for the proper pairs
    assert len(res.dot_bracket) == n
    assert res.dot_bracket[k] != "." and res.dot_bracket[l] != "."
    p, q = outer_nested
    assert res.dot_bracket[p] != "." and res.dot_bracket[q] != "."

