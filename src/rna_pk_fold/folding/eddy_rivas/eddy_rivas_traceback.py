# eddy_rivas_traceback.py
from __future__ import annotations
from typing import List, Tuple, Dict, Set, Callable, Any

from rna_pk_fold.structures import Pair
from rna_pk_fold.folding.common_traceback import pairs_to_multilayer_dotbracket, TraceResult
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBacktrackOp
from rna_pk_fold.utils.traceback_ops_utils import add_pair_once, merge_nested_interval
from rna_pk_fold.utils.back_pointer_utils import wx_bp, whx_bp, yhx_bp, zhx_bp, vhx_bp


def _record_pk_join_pairs(
    pairs: Set[Pair],
    pair_layer: Dict[Tuple[int, int], int],
    k: int,
    r: int,
    l: int,
    base_layer: int,
) -> None:
    """
    Ensure both helix pairs created at the PK join are recorded:

      left  arm: (k, r)      at layer = base_layer
      right arm: (r+1, l)    at layer = base_layer + 1

    We record these at the WX level because WHX frames do not
    intrinsically add any pairs; YHX does, but not every branch
    descends into YHX immediately.
    """
    add_pair_once(pairs, pair_layer, k, r, base_layer)
    add_pair_once(pairs, pair_layer, r + 1, l, base_layer + 1)


def traceback_with_pk(
    seq: str,
    *,
    nested_state: Any,
    re_state: EddyRivasFoldState,
    trace_nested_interval: Callable[[str, Any, int, int], TraceResult],
) -> TraceResult:
    """
    Full Rivas–Eddy traceback (layered dot-bracket) WITHOUT importing Zucker.

    Parameters
    ----------
    seq : str
        RNA sequence
    nested_state : Any
        State object for the nested-only engine (e.g., ZuckerFoldState).
        Opaque here; only passed to `trace_nested_interval`.
    re_state : EddyRivasFoldState
        Filled Rivas–Eddy DP/backpointer state.
    trace_nested_interval : Callable
        Function: (seq, nested_state, i, j) -> TraceResult for the interval [i..j].

    Returns
    -------
    TraceResult
        pairs + multilayer dot-bracket.
    """
    n = re_state.n
    if n == 0:
        return TraceResult(pairs=[], dot_bracket="")

    # Work frames (tagged)
    #   ("WX", i, j, layer)
    #   ("WHX", i, j, k, l, layer)
    #   ("YHX", i, j, k, l, layer)
    #   ("ZHX", i, j, k, l, layer)
    #   ("VHX", i, j, k, l, layer)
    Frame = Tuple  # readability

    pairs: Set[Pair] = set()
    pair_layer: Dict[Tuple[int, int], int] = {}
    stack: List[Frame] = [("WX", 0, n - 1, 0)]

    while stack:
        frame = stack.pop()
        tag = frame[0]

        # ---------------- WX ----------------
        if tag == "WX":
            _, i, j, layer = frame
            bp = wx_bp(re_state, i, j)

            if not bp:
                merge_nested_interval(seq, nested_state, i, j, layer,
                                      trace_nested_interval, pairs, pair_layer)
                continue

            op = bp.op

            # Uncharged selection → treat as nested
            if op is EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED:
                merge_nested_interval(seq, nested_state, i, j, layer,
                                      trace_nested_interval, pairs, pair_layer)
                continue

            # Common extraction; if anything is missing, fall back to nested.
            r = bp.split
            k, l = bp.hole if bp.hole else (None, None)
            if r is None or k is None or l is None:
                merge_nested_interval(seq, nested_state, i, j, layer,
                                      trace_nested_interval, pairs, pair_layer)
                continue

            # --- FIX: Ensure PK-join helix pairs are recorded at WX level ---
            # This prevents losing (k,r) or (r+1,l) when a branch descends via WHX.
            _record_pk_join_pairs(pairs, pair_layer, k, r, l, layer)

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX:
                # WHX on both arms
                stack.append(("WHX", i, r, k, r, layer))
                stack.append(("WHX", r + 1, j, r + 1, l, layer + 1))
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX:
                # YHX on both arms
                stack.append(("YHX", i, r, k, r, layer))
                stack.append(("YHX", r + 1, j, r + 1, l, layer + 1))
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX:
                # Left YHX, right WHX
                stack.append(("YHX", i, r, k, r, layer))
                stack.append(("WHX", r + 1, j, r + 1, l, layer + 1))
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX:
                # Left WHX, right YHX
                stack.append(("WHX", i, r, k, r, layer))
                stack.append(("YHX", r + 1, j, r + 1, l, layer + 1))
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_OVERLAP:
                # Both sides YHX; same hole (k,l) used on both arms
                stack.append(("YHX", i, r, k, l, layer))
                stack.append(("YHX", r + 1, j, k, l, layer + 1))
                continue

            # Fallback: treat as nested
            merge_nested_interval(seq, nested_state, i, j, layer,
                                  trace_nested_interval, pairs, pair_layer)
            continue

        # ---------------- WHX ----------------
        if tag == "WHX":
            _, i, j, k, l, layer = frame
            bp = whx_bp(re_state, i, j, k, l)
            if not bp:
                merge_nested_interval(seq, nested_state, i, j, layer,
                                      trace_nested_interval, pairs, pair_layer)
                continue

            op = bp.op

            if op in (
                EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT, EddyRivasBacktrackOp.RE_WHX_SHRINK_RIGHT,
                EddyRivasBacktrackOp.RE_WHX_TRIM_LEFT, EddyRivasBacktrackOp.RE_WHX_TRIM_RIGHT,
                EddyRivasBacktrackOp.RE_WHX_SS_BOTH,
            ):
                # Continue WHX with the coords provided on the backpointer
                (ni, nj) = bp.outer if bp.outer else (i, j)
                (nk, nl) = bp.hole if bp.hole else (k, l)
                stack.append(("WHX", ni, nj, nk, nl, layer))

            elif op is EddyRivasBacktrackOp.RE_WHX_COLLAPSE:
                # Collapse to nested WX on the provided outer interval
                (ni, nj) = bp.outer if bp.outer else (i, j)
                merge_nested_interval(seq, nested_state, ni, nj, layer,
                                      trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX:
                r = bp.split if bp.split is not None else (i + j) // 2
                stack.append(("WHX", i, r, k, l, layer))
                merge_nested_interval(seq, nested_state, r + 1, j, layer,
                                      trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_WHX_SPLIT_RIGHT_WX_WHX:
                s2 = bp.split if bp.split is not None else (i + j) // 2
                merge_nested_interval(seq, nested_state, i, s2, layer,
                                      trace_nested_interval, pairs, pair_layer)
                stack.append(("WHX", s2 + 1, j, k, l, layer))

            elif op is EddyRivasBacktrackOp.RE_WHX_OVERLAP_SPLIT:
                r = bp.split if bp.split is not None else (i + j) // 2
                stack.append(("WHX", i, r, k, l, layer))
                stack.append(("WHX", r + 1, j, k, l, layer))

            continue

        # ---------------- YHX ----------------
        if tag == "YHX":
            _, i, j, k, l, layer = frame
            # Record inner helix explicitly for outer context
            add_pair_once(pairs, pair_layer, k, l, layer)

            bp = yhx_bp(re_state, i, j, k, l)
            if not bp:
                continue
            op = bp.op

            if op in (
                EddyRivasBacktrackOp.RE_YHX_DANGLE_L, EddyRivasBacktrackOp.RE_YHX_DANGLE_R,
                EddyRivasBacktrackOp.RE_YHX_DANGLE_LR,
                EddyRivasBacktrackOp.RE_YHX_SS_LEFT, EddyRivasBacktrackOp.RE_YHX_SS_RIGHT,
                EddyRivasBacktrackOp.RE_YHX_SS_BOTH,
            ):
                (ni, nj) = bp.outer if bp.outer else (i, j)
                (nk, nl) = bp.hole if bp.hole else (k, l)
                stack.append(("YHX", ni, nj, nk, nl, layer))

            elif op in (
                EddyRivasBacktrackOp.RE_YHX_WRAP_WHX, EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_L,
                EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_R, EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR,
            ):
                (wi, wj) = bp.outer if bp.outer else (i, j)
                (wk, wl) = bp.hole if bp.hole else (k, l)
                stack.append(("WHX", wi, wj, wk, wl, layer))

            elif op is EddyRivasBacktrackOp.RE_YHX_SPLIT_LEFT_YHX_WX:
                r = bp.split if bp.split is not None else (i + j) // 2
                stack.append(("YHX", i, r, k, l, layer))
                merge_nested_interval(seq, nested_state, r + 1, j, layer,
                                      trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_YHX_SPLIT_RIGHT_WX_YHX:
                s2 = bp.split if bp.split is not None else (i + j) // 2
                merge_nested_interval(seq, nested_state, i, s2, layer,
                                      trace_nested_interval, pairs, pair_layer)
                stack.append(("YHX", s2 + 1, j, k, l, layer))

            elif op is EddyRivasBacktrackOp.RE_YHX_IS2_INNER_WHX:
                (wi, wj) = bp.outer if bp.outer else (i, j)
                (wk, wl) = bp.hole if bp.hole else (k, l)
                stack.append(("WHX", wi, wj, wk, wl, layer))

            continue

        # ---------------- ZHX ----------------
        if tag == "ZHX":
            _, i, j, k, l, layer = frame
            # Record outer helix explicitly for inner-anchored context
            add_pair_once(pairs, pair_layer, i, j, layer)

            bp = zhx_bp(re_state, i, j, k, l)
            if not bp:
                continue
            op = bp.op

            if op in (
                EddyRivasBacktrackOp.RE_ZHX_FROM_VHX, EddyRivasBacktrackOp.RE_ZHX_DANGLE_L,
                EddyRivasBacktrackOp.RE_ZHX_DANGLE_R, EddyRivasBacktrackOp.RE_ZHX_DANGLE_LR,
            ):
                (ni, nj) = bp.outer if bp.outer else (i, j)
                (nk, nl) = bp.hole if bp.hole else (k, l)
                stack.append(("VHX", ni, nj, nk, nl, layer))

            elif op in (EddyRivasBacktrackOp.RE_ZHX_SS_LEFT, EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT):
                (ni, nj) = bp.outer if bp.outer else (i, j)
                (nk, nl) = bp.hole if bp.hole else (k, l)
                stack.append(("ZHX", ni, nj, nk, nl, layer))

            elif op is EddyRivasBacktrackOp.RE_ZHX_SPLIT_LEFT_ZHX_WX:
                r = bp.split if bp.split is not None else (i + k) // 2
                stack.append(("ZHX", i, j, r, l, layer))
                merge_nested_interval(seq, nested_state, r + 1, k, layer,
                                      trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_ZHX_SPLIT_RIGHT_ZHX_WX:
                s2 = bp.split if bp.split is not None else (l + j) // 2
                stack.append(("ZHX", i, j, k, s2, layer))
                merge_nested_interval(seq, nested_state, l, s2 - 1, layer,
                                      trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_ZHX_IS2_INNER_VHX:
                (r, s2) = bp.bridge if bp.bridge else (k, l)
                stack.append(("VHX", r, s2, k, l, layer))

            continue

        # ---------------- VHX ----------------
        if tag == "VHX":
            _, i, j, k, l, layer = frame
            # Record inner helix explicitly for outer-anchored (VHX) context
            add_pair_once(pairs, pair_layer, k, l, layer)

            bp = vhx_bp(re_state, i, j, k, l)
            if not bp:
                continue
            op = bp.op

            if op in (
                EddyRivasBacktrackOp.RE_VHX_DANGLE_L, EddyRivasBacktrackOp.RE_VHX_DANGLE_R,
                EddyRivasBacktrackOp.RE_VHX_DANGLE_LR,
            ):
                (ni, nj) = bp.outer if bp.outer else (i, j)
                (nk, nl) = bp.hole if bp.hole else (k, l)
                stack.append(("VHX", ni, nj, nk, nl, layer))

            elif op in (EddyRivasBacktrackOp.RE_VHX_SS_LEFT, EddyRivasBacktrackOp.RE_VHX_SS_RIGHT):
                (ni, nj) = bp.outer if bp.outer else (i, j)
                (nk, nl) = bp.hole if bp.hole else (k, l)
                stack.append(("ZHX", ni, nj, nk, nl, layer))

            elif op is EddyRivasBacktrackOp.RE_VHX_SPLIT_LEFT_ZHX_WX:
                r = bp.split if bp.split is not None else (i + k) // 2
                stack.append(("ZHX", i, j, r, l, layer))
                merge_nested_interval(seq, nested_state, r + 1, k, layer,
                                      trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_VHX_SPLIT_RIGHT_ZHX_WX:
                s2 = bp.split if bp.split is not None else (l + j) // 2
                stack.append(("ZHX", i, j, k, s2, layer))
                merge_nested_interval(seq, nested_state, l, s2 - 1, layer,
                                      trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_VHX_IS2_INNER_ZHX:
                (r, s2) = bp.bridge if bp.bridge else (i, j)
                stack.append(("ZHX", r, s2, k, l, layer))

            elif op is EddyRivasBacktrackOp.RE_VHX_WRAP_WHX:
                (wi, wj) = bp.outer if bp.outer else (i, j)
                stack.append(("WHX", wi, wj, k, l, layer))

            elif op is EddyRivasBacktrackOp.RE_VHX_CLOSE_BOTH:
                # Both outer and inner helix are closed here
                add_pair_once(pairs, pair_layer, i, j, layer)
                add_pair_once(pairs, pair_layer, k, l, layer)
                if bp.outer and bp.hole:
                    wi, wj = bp.outer
                    wk, wl = bp.hole
                    stack.append(("WHX", wi, wj, wk, wl, layer))

            continue

        # Unknown tag → ignore and continue

    ordered = sorted(pairs, key=lambda pr: (pr.base_i, pr.base_j))
    dot = pairs_to_multilayer_dotbracket(n, ordered, pair_layer)

    return TraceResult(pairs=ordered, dot_bracket=dot)
