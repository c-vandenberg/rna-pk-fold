from __future__ import annotations
from typing import List, Tuple, Dict, Set, Callable, Any
import time
import logging

from rna_pk_fold.structures import Pair
from rna_pk_fold.folding.common_traceback import pairs_to_multilayer_dotbracket, TraceResult
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences import EddyRivasBacktrackOp
from rna_pk_fold.utils.traceback_ops_utils import add_pair_once, merge_nested_interval
from rna_pk_fold.utils.back_pointer_utils import wx_bp, whx_bp, yhx_bp, zhx_bp, vhx_bp

logger = logging.getLogger(__name__)


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
    start_time = time.perf_counter()

    n = re_state.seq_len
    if n == 0:
        return TraceResult(pairs=[], dot_bracket="")

    logger.info(f"Starting traceback for sequence length N={n}")

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
    bp0 = wx_bp(re_state, 0, n - 1)
    if bp0:
        print(
            f"[WX WIN] op={bp0.op} split={bp0.split} "
            f"hole={bp0.hole} hole_left={bp0.hole_left} hole_right={bp0.hole_right}",
            flush=True
        )
    else:
        print("[WX WIN] None → nested only", flush=True)

    if bp0 and bp0.op in (
            EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX,
            EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX,
            EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX,
            EddyRivasBacktrackOp.RE_PK_COMPOSE_WX,
    ):
        r = bp0.split
        # left child for YHX / WHX depends on op
        if bp0.hole_left:  # e.g., (k, r)
            k_l, l_l = bp0.hole_left
        else:
            k_l, l_l = (bp0.hole[0], r) if bp0.hole else (None, None)

        if bp0.hole_right:  # e.g., (r+1, l) or (r+1, l-1)
            k_r, l_r = bp0.hole_right
        else:
            _, l = bp0.hole if bp0.hole else (None, None)
            k_r, l_r = (r + 1, l)

        # Probe the exact cells traceback will ask for
        ybp = yhx_bp(re_state, 0, r, k_l, l_l)
        wbp = whx_bp(re_state, r + 1, n - 1, k_r, l_r)

        print(f"[PROBE] YHX[0,{r}:{k_l},{l_l}] BP?",
              "yes" if ybp else "no",
              f"op={getattr(ybp, 'op', None)}", flush=True)
        print(f"[PROBE] WHX[{r + 1},{n - 1}:{k_r},{l_r}] BP?",
              "yes" if wbp else "no",
              f"op={getattr(wbp, 'op', None)}", flush=True)


    while stack:
        frame = stack.pop()
        print(f"[TB POP] {frame}", flush=True)
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

            # Explicit uncharged selection -> nested
            if op is EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED:
                merge_nested_interval(seq, nested_state, i, j, layer,
                                      trace_nested_interval, pairs, pair_layer)
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX:
                # WHX + WHX  (both nested → layer 0)
                r = bp.split
                if bp.hole_left and bp.hole_right:
                    k_l, l_l = bp.hole_left
                    k_r, l_r = bp.hole_right
                else:
                    k, l = bp.hole if bp.hole else (None, None)
                    k_l, l_l = k, r
                    k_r, l_r = r + 1, l
                stack.append(("WHX", i, r, k_l, l_l, 0))
                stack.append(("WHX", r + 1, j, k_r, l_r, 0))
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX:
                # YHX + YHX  (two crossing seams → put on two distinct layers)
                r = bp.split
                if bp.hole_left and bp.hole_right:
                    k_l, l_l = bp.hole_left
                    k_r, l_r = bp.hole_right
                else:
                    k, l = bp.hole if bp.hole else (None, None)
                    k_l, l_l = k, r
                    k_r, l_r = r + 1, l
                stack.append(("YHX", i, r, k_l, l_l, layer + 1))
                stack.append(("YHX", r + 1, j, k_r, l_r, layer + 2))
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX:
                # YHX (crossing) + WHX (nested)
                r = bp.split
                if bp.hole_left and bp.hole_right:
                    k_l, l_l = bp.hole_left
                    k_r, l_r = bp.hole_right
                else:
                    k, l = bp.hole if bp.hole else (None, None)
                    k_l, l_l = k, r
                    k_r, l_r = r + 1, l  # WHX uses l

                # crossing seam on PK layer
                stack.append(("YHX", i, r, k_l, l_l, layer + 1))
                # nested region on base layer
                stack.append(("WHX", r + 1, j, k_r, l_r, 0))
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX:
                # WHX (nested) + YHX (crossing)
                r = bp.split
                if bp.hole_left and bp.hole_right:
                    k_l, l_l = bp.hole_left
                    k_r, l_r = bp.hole_right
                else:
                    k, l = bp.hole if bp.hole else (None, None)
                    k_l, l_l = k, r
                    k_r, l_r = r + 1, l

                # nested region on base layer
                stack.append(("WHX", i, r, k_l, l_l, 0))
                # crossing seam on PK layer
                stack.append(("YHX", r + 1, j, k_r, l_r, layer + 1))
                continue

            # Fallback: treat as nested
            merge_nested_interval(seq, nested_state, i, j, layer,
                                  trace_nested_interval, pairs, pair_layer)
            continue

        # ---------------- WHX ----------------
        if tag == "WHX":
            _, i, j, k, l, layer = frame
            logger.debug(f"\n=== WHX[{i},{j},{k},{l}] layer={layer} ===")
            bp = whx_bp(re_state, i, j, k, l)
            if not bp:
                print(f"[WHX MISS] merging nested [{i},{j}] hole=({k},{l}) layer={layer}", flush=True)
                merge_nested_interval(seq, nested_state, i, j, layer,
                                      trace_nested_interval, pairs, pair_layer)
                continue

            print(f"[WHX] ({i},{j}:{k},{l}) layer={layer} op={bp.op}", flush=True)
            op = bp.op

            # FIX: Each operation explicitly modifies coordinates
            if op is EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT:
                stack.append(("WHX", i, j, k + 1, l, layer))  # ← Explicitly k+1

            elif op is EddyRivasBacktrackOp.RE_WHX_SHRINK_RIGHT:
                stack.append(("WHX", i, j, k, l - 1, layer))  # ← Explicitly l-1

            elif op is EddyRivasBacktrackOp.RE_WHX_TRIM_LEFT:
                stack.append(("WHX", i + 1, j, k, l, layer))  # ← Explicitly i+1

            elif op is EddyRivasBacktrackOp.RE_WHX_TRIM_RIGHT:
                stack.append(("WHX", i, j - 1, k, l, layer))  # ← Explicitly j-1

            elif op is EddyRivasBacktrackOp.RE_WHX_SS_BOTH:
                stack.append(("WHX", i + 1, j - 1, k, l, layer))  # ← Explicitly i+1, j-1

            elif op is EddyRivasBacktrackOp.RE_WHX_COLLAPSE:
                merge_nested_interval(seq, nested_state, i, j, layer,
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

            else:
                logger.warning(f"Unknown WHX op: {op}, falling back to nested")
                merge_nested_interval(seq, nested_state, i, j, layer,
                                      trace_nested_interval, pairs, pair_layer)

            continue

        # ---------------- YHX ----------------
        if tag == "YHX":
            _, i, j, k, l, layer = frame
            logger.debug(f"YHX[{i},{j},{k},{l}] layer={layer}")

            bp = yhx_bp(re_state, i, j, k, l)
            if not bp:
                print(f"[YHX MISS] ({i},{j}:{k},{l}) layer={layer} → no BP", flush=True)
                continue
            print(f"[YHX] ({i},{j}:{k},{l}) layer={layer} op={bp.op}", flush=True)
            op = bp.op

            # Dangle operations → push VHX with modified coordinates
            if op is EddyRivasBacktrackOp.RE_YHX_DANGLE_L:
                stack.append(("VHX", i + 1, j, k, l, layer))

            elif op is EddyRivasBacktrackOp.RE_YHX_DANGLE_R:
                stack.append(("VHX", i, j - 1, k, l, layer))

            elif op is EddyRivasBacktrackOp.RE_YHX_DANGLE_LR:
                stack.append(("VHX", i + 1, j - 1, k, l, layer))

            # SS trims → push YHX with modified coordinates
            elif op is EddyRivasBacktrackOp.RE_YHX_SS_LEFT:
                stack.append(("YHX", i + 1, j, k, l, layer))

            elif op is EddyRivasBacktrackOp.RE_YHX_SS_RIGHT:
                stack.append(("YHX", i, j - 1, k, l, layer))

            elif op is EddyRivasBacktrackOp.RE_YHX_SS_BOTH:
                stack.append(("YHX", i + 1, j - 1, k, l, layer))

            # Wrap operations → push WHX
            elif op is EddyRivasBacktrackOp.RE_YHX_WRAP_WHX:
                place_pair_non_crossing(pairs, pair_layer, k, l, layer)
                stack.append(("WHX", i, j, k - 1, l + 1, 0))

            elif op is EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_L:
                place_pair_non_crossing(pairs, pair_layer, k, l, layer)
                stack.append(("WHX", i + 1, j, k - 1, l + 1, 0))

            elif op is EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_R:
                place_pair_non_crossing(pairs, pair_layer, k, l, layer)
                stack.append(("WHX", i, j - 1, k - 1, l + 1, 0))

            elif op is EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR:
                place_pair_non_crossing(pairs, pair_layer, k, l, layer)
                stack.append(("WHX", i + 1, j - 1, k - 1, l + 1, 0))



            # Split operations
            elif op is EddyRivasBacktrackOp.RE_YHX_SPLIT_LEFT_YHX_WX:
                r = bp.split if bp.split is not None else (i + j) // 2
                stack.append(("YHX", i, r, k, l, layer))
                merge_nested_interval(seq, nested_state, r + 1, j, 0,
                                      trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_YHX_SPLIT_RIGHT_WX_YHX:
                s2 = bp.split if bp.split is not None else (i + j) // 2
                merge_nested_interval(seq, nested_state, i, s2, 0,
                                      trace_nested_interval, pairs, pair_layer)
                stack.append(("YHX", s2 + 1, j, k, l, layer))

            elif op is EddyRivasBacktrackOp.RE_YHX_IS2_INNER_WHX:
                # IS2 outer context then delegate to WHX
                (r, s2) = bp.bridge if bp.bridge else (i, j)
                stack.append(("WHX", r, s2, k, l, layer))

            else:
                logger.warning(f"Unknown YHX op: {op}, skipping")

            continue

        # ---------------- ZHX ----------------
        if tag == "ZHX":
            _, i, j, k, l, layer = frame

            bp = zhx_bp(re_state, i, j, k, l)
            if not bp:
                continue
            op = bp.op

            # FIX: Explicitly compute modified coordinates for each operation
            if op is EddyRivasBacktrackOp.RE_ZHX_FROM_VHX:
                stack.append(("VHX", i, j, k, l, layer))

            elif op is EddyRivasBacktrackOp.RE_ZHX_DANGLE_L:
                stack.append(("VHX", i, j, k, l + 1, layer))  # ← Queries VHX[i,j,k,l+1]

            elif op is EddyRivasBacktrackOp.RE_ZHX_DANGLE_R:
                stack.append(("VHX", i, j, k - 1, l, layer))  # ← Queries VHX[i,j,k-1,l]

            elif op is EddyRivasBacktrackOp.RE_ZHX_DANGLE_LR:
                stack.append(("VHX", i, j, k - 1, l + 1, layer))  # ← Queries VHX[i,j,k-1,l+1]

            # SS operations
            elif op is EddyRivasBacktrackOp.RE_ZHX_SS_LEFT:
                stack.append(("ZHX", i, j, k - 1, l, layer))  # ← Queries ZHX[i,j,k-1,l]

            elif op is EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT:
                stack.append(("ZHX", i, j, k, l + 1, layer))  # ← Queries ZHX[i,j,k,l+1]

            # Split operations
            elif op is EddyRivasBacktrackOp.RE_ZHX_SPLIT_LEFT_ZHX_WX:
                r = bp.split if bp.split is not None else (i + k) // 2
                stack.append(("ZHX", i, j, r, l, layer))
                merge_nested_interval(seq, nested_state, r + 1, k, 0,     trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_ZHX_SPLIT_RIGHT_ZHX_WX:
                s2 = bp.split if bp.split is not None else (l + j) // 2
                stack.append(("ZHX", i, j, k, s2, layer))
                merge_nested_interval(seq, nested_state, l, s2 - 1, 0,     trace_nested_interval, pairs, pair_layer)

            # IS2 operation
            elif op is EddyRivasBacktrackOp.RE_ZHX_IS2_INNER_VHX:
                (r, s2) = bp.bridge if bp.bridge else (k, l)
                stack.append(("VHX", r, s2, k, l, layer))

            else:
                logger.warning(f"Unknown ZHX op: {op}, skipping")

            continue

        # ---------------- VHX ----------------
        if tag == "VHX":
            _, i, j, k, l, layer = frame

            bp = vhx_bp(re_state, i, j, k, l)
            if not bp:
                print(f"[VHX MISS] ({i},{j}:{k},{l}) layer={layer} → no BP", flush=True)
                continue
            print(f"[VHX] ({i},{j}:{k},{l}) layer={layer} op={bp.op}", flush=True)
            op = bp.op

            # FIX: Dangle operations explicitly modify hole coordinates
            if op is EddyRivasBacktrackOp.RE_VHX_DANGLE_L:
                stack.append(("VHX", i, j, k + 1, l, layer))  # ← Explicitly k+1

            elif op is EddyRivasBacktrackOp.RE_VHX_DANGLE_R:
                stack.append(("VHX", i, j, k, l - 1, layer))  # ← Explicitly l-1

            elif op is EddyRivasBacktrackOp.RE_VHX_DANGLE_LR:
                stack.append(("VHX", i, j, k + 1, l - 1, layer))  # ← Explicitly k+1, l-1

            # SS operations push ZHX with modified coordinates
            elif op is EddyRivasBacktrackOp.RE_VHX_SS_LEFT:
                stack.append(("ZHX", i, j, k - 1, l, layer))

            elif op is EddyRivasBacktrackOp.RE_VHX_SS_RIGHT:
                stack.append(("ZHX", i, j, k, l + 1, layer))

            # ... rest of VHX handlers remain the same ...
            elif op is EddyRivasBacktrackOp.RE_VHX_SPLIT_LEFT_ZHX_WX:
                r = bp.split if bp.split is not None else (i + k) // 2
                stack.append(("ZHX", i, j, r, l, layer))
                merge_nested_interval(seq, nested_state, r + 1, k, 0, trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_VHX_SPLIT_RIGHT_ZHX_WX:
                s2 = bp.split if bp.split is not None else (l + j) // 2
                stack.append(("ZHX", i, j, k, s2, layer))
                merge_nested_interval(seq, nested_state, i, s2, 0, trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_VHX_IS2_INNER_ZHX:
                (r, s2) = bp.bridge if bp.bridge else (i, j)
                stack.append(("ZHX", r, s2, k, l, layer))

            elif op is EddyRivasBacktrackOp.RE_VHX_WRAP_WHX:
                (wi, wj) = bp.outer if bp.outer else (i, j)
                stack.append(("WHX", wi, wj, k, l, 0))



            elif op is EddyRivasBacktrackOp.RE_VHX_CLOSE_BOTH:
                # Place both helices at the lowest non-conflicting PK layer (>= current)
                layer_outer = place_pair_non_crossing(pairs, pair_layer, i, j, layer)  # outer helix
                layer_inner = place_pair_non_crossing(pairs, pair_layer, k, l, layer)  # inner helix

                # Optional debug to see where they landed
                if layer_outer != layer or layer_inner != layer:
                    print(f"[BUMP] VHX_CLOSE_BOTH: outer({i},{j})→L{layer_outer}, inner({k},{l})→L{layer_inner}",
                          flush=True)

                # Any surrounding nested content stays on base layer
                if bp.outer and bp.hole:
                    wi, wj = bp.outer
                    wk, wl = bp.hole
                    stack.append(("WHX", wi, wj, wk, wl, 0))

            continue

        # Unknown tag → ignore and continue

    def audit_layer_map(pair_layer: dict[tuple[int, int], int]) -> None:
        by_layer = {}
        for (i, j), lay in pair_layer.items():
            by_layer.setdefault(lay, []).append((i, j))

        def crosses(a, b):
            return (a[0] < b[0] < a[1] < b[1]) or (b[0] < a[0] < b[1] < a[1])

        for lay, ps in sorted(by_layer.items()):
            c = sum(crosses(ps[u], ps[v]) for u in range(len(ps)) for v in range(u + 1, len(ps)))
            print(f"[L{lay}] pairs={len(ps)} crossings_within_layer={c}", flush=True)

    ordered = sorted(pairs, key=lambda pr: (pr.base_i, pr.base_j))
    dot = pairs_to_multilayer_dotbracket(n, ordered, pair_layer)

    audit_layer_map(pair_layer)

    elapsed = time.perf_counter() - start_time
    logger.info(f"Traceback completed in {elapsed:.3f}s")
    logger.info(f"Found {len(ordered)} base pairs")

    return TraceResult(pairs=ordered, dot_bracket=dot)


# --- layer-safe placement (avoids crossings within the same layer) ---
def _crosses(a: tuple[int,int], b: tuple[int,int]) -> bool:
    return (a[0] < b[0] < a[1] < b[1]) or (b[0] < a[0] < b[1] < a[1])

def place_pair_non_crossing(
    pairs: set,
    pair_layer: dict[tuple[int,int], int],
    i: int,
    j: int,
    start_layer: int
) -> int:
    """Place (i,j) at the lowest layer ≥ start_layer that doesn't create
    an intra-layer crossing. Returns the chosen layer."""
    layer = start_layer
    while True:
        conflict = False
        for (pi, pj), L in pair_layer.items():
            if L == layer and _crosses((i, j), (pi, pj)):
                conflict = True
                break
        if not conflict:
            # actually place it
            add_pair_once(pairs, pair_layer, i, j, layer)
            print(f"[PAIR] ({i},{j}) -> L{layer}", flush=True)
            return layer
        layer += 1

