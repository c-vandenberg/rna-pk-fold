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

    n = re_state.n
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

    iteration_count = 0
    max_iterations = 10000

    while stack:
        iteration_count += 1

        # ADD THIS DEBUG BLOCK
        if iteration_count % 100 == 0:
            print(f"[ITER {iteration_count}] Stack size: {len(stack)}, Last 3 frames: {stack[-3:]}")

        if iteration_count > max_iterations:
            print(f"[ERROR] Exceeded {max_iterations} iterations! Infinite loop detected!")
            print(f"Stack size: {len(stack)}")
            print(f"Last 10 frames:")
            for f in stack[-10:]:
                print(f"  {f}")
            raise RuntimeError("Traceback infinite loop")

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

            # NEW: explicit uncharged selection → treat as nested
            if op is EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED:
                merge_nested_interval(seq, nested_state, i, j, layer,
                                      trace_nested_interval, pairs, pair_layer)
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX:
                r = bp.split
                if bp.hole_left and bp.hole_right:
                    k_l, l_l = bp.hole_left
                    k_r, l_r = bp.hole_right
                else:
                    k, l = bp.hole if bp.hole else (None, None)
                    k_l, l_l = k, r
                    k_r, l_r = r + 1, l
                stack.append(("WHX", i, r, k_l, l_l, layer))
                stack.append(("WHX", r + 1, j, k_r, l_r, layer + 1))
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX:
                r = bp.split
                if bp.hole_left and bp.hole_right:
                    k_l, l_l = bp.hole_left
                    k_r, l_r = bp.hole_right
                else:
                    k, l = bp.hole if bp.hole else (None, None)
                    k_l, l_l = k, r
                    k_r, l_r = r + 1, l - 1  # YHX uses l-1
                stack.append(("YHX", i, r, k_l, l_l, layer))
                stack.append(("YHX", r + 1, j, k_r, l_r, layer + 1))
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX:
                r = bp.split
                if bp.hole_left and bp.hole_right:
                    k_l, l_l = bp.hole_left
                    k_r, l_r = bp.hole_right
                else:
                    k, l = bp.hole if bp.hole else (None, None)
                    k_l, l_l = k, r
                    k_r, l_r = r + 1, l  # WHX uses l

                    # ADD THIS DEBUG BLOCK
                    logger.debug(f"\n=== WX Composition at [{i},{j}] layer={layer} ===")
                    logger.debug(f"Split r={r}, hole={bp.hole}")
                    logger.debug(f"hole_left={bp.hole_left}, hole_right={bp.hole_right}")
                    logger.debug(f"Pushing YHX[{i},{r},{k_l},{l_l}] layer={layer}")
                    logger.debug(f"Pushing WHX[{r + 1},{j},{k_r},{l_r}] layer={layer + 1}")

                    # Check if backpointers exist
                    test_yhx = yhx_bp(re_state, i, r, k_l, l_l)
                    test_whx = whx_bp(re_state, r + 1, j, k_r, l_r)
                    logger.debug(f"YHX BP exists: {test_yhx is not None}")
                    logger.debug(f"WHX BP exists: {test_whx is not None}")
                    if test_yhx:
                        logger.debug(f"  YHX BP op: {test_yhx.op}")
                    if test_whx:
                        logger.debug(f"  WHX BP op: {test_whx.op}")
                stack.append(("YHX", i, r, k_l, l_l, layer))
                stack.append(("WHX", r + 1, j, k_r, l_r, layer + 1))
                continue

            if op is EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX:
                r = bp.split
                if bp.hole_left and bp.hole_right:
                    k_l, l_l = bp.hole_left
                    k_r, l_r = bp.hole_right
                else:
                    k, l = bp.hole if bp.hole else (None, None)
                    k_l, l_l = k, r
                    k_r, l_r = r + 1, l - 1  # YHX uses l-1
                stack.append(("WHX", i, r, k_l, l_l, layer))
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
                logger.debug(f"  → NO BACKPOINTER! Falling back to nested...")
                merge_nested_interval(seq, nested_state, i, j, layer,
                                      trace_nested_interval, pairs, pair_layer)
                continue

            logger.debug(f"  → BP found: op={bp.op}")
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
        # ---------------- YHX ----------------
        if tag == "YHX":
            _, i, j, k, l, layer = frame
            logger.debug(f"YHX[{i},{j},{k},{l}] layer={layer}")
            add_pair_once(pairs, pair_layer, k, l, layer)  # ensure inner helix recorded

            bp = yhx_bp(re_state, i, j, k, l)
            if not bp:
                logger.debug(f"  → NO BACKPOINTER! Skipping...")
                continue

            logger.debug(f"  → BP found: op={bp.op}")
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
                stack.append(("WHX", i, j, k - 1, l + 1, layer))

            elif op is EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_L:
                stack.append(("WHX", i + 1, j, k - 1, l + 1, layer))

            elif op is EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_R:
                stack.append(("WHX", i, j - 1, k - 1, l + 1, layer))

            elif op is EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR:
                stack.append(("WHX", i + 1, j - 1, k - 1, l + 1, layer))

            # Split operations
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
                # IS2 outer context then delegate to WHX
                (r, s2) = bp.bridge if bp.bridge else (i, j)
                stack.append(("WHX", r, s2, k, l, layer))

            else:
                logger.warning(f"Unknown YHX op: {op}, skipping")

            continue

        # ---------------- ZHX ----------------
        if tag == "ZHX":
            _, i, j, k, l, layer = frame
            add_pair_once(pairs, pair_layer, i, j, layer)  # ensure outer helix recorded

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
                merge_nested_interval(seq, nested_state, r + 1, k, layer,
                                      trace_nested_interval, pairs, pair_layer)

            elif op is EddyRivasBacktrackOp.RE_ZHX_SPLIT_RIGHT_ZHX_WX:
                s2 = bp.split if bp.split is not None else (l + j) // 2
                stack.append(("ZHX", i, j, k, s2, layer))
                merge_nested_interval(seq, nested_state, l, s2 - 1, layer,
                                      trace_nested_interval, pairs, pair_layer)

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
            add_pair_once(pairs, pair_layer, k, l, layer)

            bp = vhx_bp(re_state, i, j, k, l)
            if not bp:
                continue
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

    elapsed = time.perf_counter() - start_time
    logger.info(f"Traceback completed in {elapsed:.3f}s")
    logger.info(f"Found {len(ordered)} base pairs")

    return TraceResult(pairs=ordered, dot_bracket=dot)


