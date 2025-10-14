from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import math

import numpy as np

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer, EddyRivasBacktrackOp
from rna_pk_fold.energies.energy_pk_ops import (dangle_hole_left, dangle_hole_right, dangle_outer_left,
                                                dangle_outer_right)
from rna_pk_fold.utils.dynamic_programming.matrix_utils import get_whx_with_collapse, get_zhx_with_collapse, get_wxi_or_wx
from rna_pk_fold.utils.energy.is2_utils import is2_outer, is2_outer_yhx


# ---------- Best-value tracker with optional tie-break to RIGHT ----------
@dataclass
class CandTracker:
    best: float = math.inf
    bp: Optional["EddyRivasBackPointer"] = None

    def consider(self, cand: float, bp: "EddyRivasBackPointer") -> None:
        if cand < self.best:
            self.best, self.bp = cand, bp

    def consider_pair_right_biased(
        self,
        left_cand: float,
        right_cand: float,
        left_bp: "EddyRivasBackPointer",
        right_bp: "EddyRivasBackPointer",
    ) -> None:
        # prefer RIGHT on ties
        if right_cand <= left_cand:
            self.consider(right_cand, right_bp)
        else:
            self.consider(left_cand, left_bp)


# ---------- Common guards ----------
def should_skip_cell(
    i: int,
    j: int,
    k: int,
    l: int,
    cfg,
    vxu_get: Callable[[int, int], float],
    can_pair_mask: Optional[list[list[bool]]] = None,
    require_kl_pairable: bool = False,
) -> bool:
    # Hole width guard
    hole_w = (l - k - 1)
    if cfg.min_hole_width and hole_w < cfg.min_hole_width:
        return True
    if cfg.max_hole_width and hole_w > cfg.max_hole_width:
        return True

    # Beam guard
    if cfg.beam_v_threshold != 0.0 and vxu_get(k, l) > cfg.beam_v_threshold:
        return True

    # Optional Watson–Crick mask
    if require_kl_pairable and can_pair_mask is not None and not can_pair_mask[k][l]:
        return True

    return False


# ---------- Splits (generic) ----------
def best_split(
    length: int,
    left_fetch: Callable[[int], float],
    right_fetch: Callable[[int], float],
    penalty: float = 0.0,
) -> Tuple[float, int]:
    """Vectorized min over left[t] + right[t] (+ penalty). Returns (val, t*)."""
    if length <= 0:
        return math.inf, -1
    left_vec = np.full(length, np.inf, dtype=np.float64)
    right_vec = np.full(length, np.inf, dtype=np.float64)
    for t in range(length):
        lv = left_fetch(t)
        rv = right_fetch(t)
        if math.isfinite(lv): left_vec[t] = lv
        if math.isfinite(rv): right_vec[t] = rv
    if penalty == 0.0:
        val_vec = left_vec + right_vec
    else:
        val_vec = left_vec + right_vec + penalty
    t_star = int(np.argmin(val_vec))
    best = float(val_vec[t_star])
    if not math.isfinite(best):
        return math.inf, -1
    return best, t_star


# ---------- IS2 (outer bridge scan) ----------
def scan_is2_outer_simple(
    state,
    cfg,
    seq: str,
    i: int, j: int, k: int, l: int,
    inner_matrix: str,       # "vhx" | "zhx" | "yhx" | "whx"
    bridge_kind: str,        # "default" -> is2_outer(...) ; "yhx" -> is2_outer_yhx(...)
    op,
) -> Tuple[float, Optional[Tuple[int, int]], object]:
    # Accessors (no nested defs)
    if inner_matrix == "vhx":
        inner_get = lambda r, s2: state.vhx_matrix.get(r, s2, k, l)
    elif inner_matrix == "zhx":
        inner_get = lambda r, s2: get_zhx_with_collapse(state.zhx_matrix, state.vxu_matrix, r, s2, k, l)
    elif inner_matrix == "yhx":
        inner_get = lambda r, s2: state.yhx_matrix.get(r, s2, k, l)
    elif inner_matrix == "whx":
        inner_get = lambda r, s2: get_whx_with_collapse(state.whx_matrix, state.wxu_matrix, r, s2, k, l)
    else:
        raise ValueError(f"unsupported inner matrix: {inner_matrix}")

    if bridge_kind == "yhx":
        bridge_get = lambda r, s2: is2_outer_yhx(cfg, seq, i, j, r, s2)
    else:
        bridge_get = lambda r, s2: is2_outer(seq, cfg.tables, i, j, r, s2)

    best = math.inf
    best_bp = None
    for r in range(i, k + 1):
        for s2 in range(l, j + 1):
            if r <= s2:
                inner = inner_get(r, s2)
                if math.isfinite(inner):
                    cand = bridge_get(r, s2) + inner
                    if cand < best:
                        best, best_bp = cand, (r, s2)
    return best, best_bp, op

# ---------- SS helpers (left/right/both with RIGHT tie-break) ----------
def consider_ss_hole_right_biased(
    tracker: CandTracker,
    zhx_get: Callable[[int, int, int, int], float],
    i: int, j: int, k: int, l: int,
    q_hole: float,
    op_left, op_right,
):
    # 3.1. Single Strand Left: Add base on the 5' side.
    vL = zhx_get(i, j, k - 1, l)

    # 3.1. Single Strand Right: Add base on the 3' side.
    vR = zhx_get(i, j, k, l + 1)

    left_cand  = q_hole + vL if math.isfinite(vL) else math.inf
    right_cand = q_hole + vR if math.isfinite(vR) else math.inf
    tracker.consider_pair_right_biased(
        left_cand,
        right_cand,
        EddyRivasBackPointer(op=op_left,  outer=(i, j), hole=(k, l)),
        EddyRivasBackPointer(op=op_right, outer=(i, j), hole=(k, l)),
    )

def consider_ss_outer_right_biased(
    tracker: CandTracker,
    yhx_get: Callable[[int, int, int, int], float],
    i: int, j: int, k: int, l: int,
    q_out: float,
    op_left, op_right,
):
    # 1. Single Strand Left: Trim from the 5' (Left) Side.
    vL = yhx_get(i + 1, j, k, l)

    # 2. Single Strand Right: Trim from the 3' (Left) Side.
    vR = yhx_get(i, j - 1, k, l)

    left_cand  = q_out + vL if math.isfinite(vL) else math.inf
    right_cand = q_out + vR if math.isfinite(vR) else math.inf
    tracker.consider_pair_right_biased(
        left_cand,
        right_cand,
        EddyRivasBackPointer(op=op_left,  outer=(i, j), hole=(k, l)),
        EddyRivasBackPointer(op=op_right, outer=(i, j), hole=(k, l)),
    )


def consider_ss_outer_both(tracker: CandTracker, v: float, q_out: float, i: int, j: int, k: int, l: int, op):
    if math.isfinite(v):
        tracker.consider(2.0 * q_out + v, EddyRivasBackPointer(op=op, outer=(i, j), hole=(k, l)))


# ---------- Dangle wrappers ----------
def consider_dangles_on_hole_from_vhx(
    tracker: CandTracker,
    vhx_get: Callable[[int, int, int, int], float],
    seq: str, costs,
    i: int, j: int, k: int, l: int,
    tilde_p_hole: float,
    internal_pk_penalty: float,
    op_left, op_right, op_left_right,
):
    # LR
    v = vhx_get(i, j, k - 1, l + 1)
    if math.isfinite(v):
        left_hole_energy = dangle_hole_left(seq, k, costs)
        right_hole_energy = dangle_hole_right(seq, l, costs)
        tracker.consider(left_hole_energy + right_hole_energy + tilde_p_hole + v + internal_pk_penalty,
                         EddyRivasBackPointer(op=op_left_right, outer=(i, j), hole=(k, l)))
    # R
    v = vhx_get(i, j, k - 1, l)
    if math.isfinite(v):
        right_hole_energy = dangle_hole_right(seq, l - 1, costs)
        tracker.consider(right_hole_energy + tilde_p_hole + v + internal_pk_penalty,
                         EddyRivasBackPointer(op=op_right, outer=(i, j), hole=(k, l)))
    # L
    v = vhx_get(i, j, k, l + 1)
    if math.isfinite(v):
        left_hole_energy = dangle_hole_left(seq, k + 1, costs)
        tracker.consider(left_hole_energy + tilde_p_hole + v + internal_pk_penalty,
                         EddyRivasBackPointer(op=op_left, outer=(i, j), hole=(k, l)))

def consider_dangles_on_outer_from_vhx(
    tracker: CandTracker,
    vhx_get: Callable[[int, int, int, int], float],
    seq: str, costs,
    i: int, j: int, k: int, l: int,
    tilde_p_out: float,
    internal_pk_penalty: float,
    op_left, op_right, op_left_right,
):
    # Case 1: Dangles on the Left Side of the Outer Pair
    v = vhx_get(i + 1, j, k, l)
    if math.isfinite(v):
        outer_left_dangle_energy = dangle_outer_left(seq, i, costs)
        tracker.consider(outer_left_dangle_energy + tilde_p_out + v + internal_pk_penalty,
                         EddyRivasBackPointer(op=op_left, outer=(i, j), hole=(k, l)))

    # Case 2: Dangles on the Right Side of the Outer Pair
    v = vhx_get(i, j - 1, k, l)
    if math.isfinite(v):
        outer_right_dangle_energy = dangle_outer_right(seq, j, costs)
        tracker.consider(outer_right_dangle_energy + tilde_p_out + v + internal_pk_penalty,
                         EddyRivasBackPointer(op=op_right, outer=(i, j), hole=(k, l)))

    # Case 3: Dangles on Both Sides of the Outer Pair
    v = vhx_get(i + 1, j - 1, k, l)
    if math.isfinite(v):
        outer_left_dangle_energy = dangle_outer_left(seq, i, costs)
        outer_right_dangle_energy = dangle_outer_right(seq, j, costs)
        tracker.consider(outer_left_dangle_energy + outer_right_dangle_energy + tilde_p_out + v + internal_pk_penalty,
                         EddyRivasBackPointer(op=op_left_right, outer=(i, j), hole=(k, l)))


# ---------- VHX internal dangles (no sequence lookup) ----------
def consider_vhx_inner_dangles(
    tracker: CandTracker,
    vhx_get: Callable[[int, int, int, int], float],
    i: int, j: int, k: int, l: int,
    tilde_p_hole: float, tilde_l: float, tilde_r: float,
    opL, opR, opLR,
):
    # -------- Case 1: Dangle on the 5' (Left) Side of The Inner Pair (k,l). --------
    v = vhx_get(i, j, k + 1, l)
    tracker.consider(tilde_p_hole + tilde_l + v,
                     EddyRivasBackPointer(op=opL, outer=(i, j), hole=(k, l)))

    # -------- Case 2: Dangle on the 3' (Right) Side of the Inner Pair (k,l). --------
    v = vhx_get(i, j, k, l - 1)
    tracker.consider(tilde_p_hole + tilde_r + v,
                     EddyRivasBackPointer(op=opR, outer=(i, j), hole=(k, l)))

    # -------- Case 3: Dangles on Both Sides of the Inner Pair (k,l). --------
    v = vhx_get(i, j, k + 1, l - 1)  # LR
    tracker.consider(tilde_p_hole + tilde_l + tilde_r + v,
                     EddyRivasBackPointer(op=opLR, outer=(i, j), hole=(k, l)))


# ---------- Multiloop wrappers (WHX close/wrap) ----------
def consider_vhx_close_and_wrap(
    tracker: CandTracker,
    whx_collapse_get: Callable[[int, int, int, int], float],
    i: int, j: int, k: int, l: int,
    tilde_p_hole: float,
    tilde_m_vhx: float,
    tilde_m_whx: float,
    internal_pk_penalty: float,
    op_close_both,
    op_wrap_whx,
):
    # close both (i+1,j-1 : k-1,l+1)
    close = whx_collapse_get(i + 1, j - 1, k - 1, l + 1)
    if math.isfinite(close):
        cand = 2.0 * tilde_p_hole + tilde_m_vhx + close + internal_pk_penalty + tilde_m_whx
        tracker.consider(cand, EddyRivasBackPointer(op=op_close_both, outer=(i, j), hole=(k, l)))

    # wrap (i+1,j-1 : k,l)
    wrap = whx_collapse_get(i + 1, j - 1, k, l)
    if math.isfinite(wrap):
        cand = tilde_p_hole + tilde_m_vhx + wrap + internal_pk_penalty + tilde_m_whx
        tracker.consider(cand, EddyRivasBackPointer(op=op_wrap_whx, outer=(i, j), hole=(k, l)))


def consider_yhx_wrap_whx(
    tracker: CandTracker,
    whx_get: Callable[[int, int, int, int], float],
    seq: str, costs,
    i: int, j: int, k: int, l: int,
    tilde_p_out: float, tilde_m_yhx: float, tilde_m_whx: float, internal_pk_penalty: float,
    op_plain, op_left, op_right, op_left_right,
):
    # 1. Plain Multiloop Wrap
    v = whx_get(i, j, k - 1, l + 1)
    if math.isfinite(v):
        tracker.consider(
            tilde_p_out + tilde_m_yhx + tilde_m_whx + v + internal_pk_penalty,
            EddyRivasBackPointer(op=op_plain, outer=(i, j), hole=(k, l))
        )

    # 2. Wrap on 5' (Left) Side
    v = whx_get(i + 1, j, k - 1, l + 1)
    if math.isfinite(v):
        outer_left_dangle_energy = dangle_outer_left(seq, i, costs)
        tracker.consider(
            outer_left_dangle_energy + tilde_p_out + tilde_m_yhx + tilde_m_whx + v + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left, outer=(i, j), hole=(k, l))
        )

    # 3. Wrap on 3' (Right) Side
    v = whx_get(i, j - 1, k - 1, l + 1)
    if math.isfinite(v):
        outer_right_dangle_energy = dangle_outer_right(seq, j, costs)
        tracker.consider(
            outer_right_dangle_energy + tilde_p_out + tilde_m_yhx + tilde_m_whx + v + internal_pk_penalty,
            EddyRivasBackPointer(op=op_right, outer=(i, j), hole=(k, l))
        )

    # 4. Wrap on Both Sides
    v = whx_get(i + 1, j - 1, k - 1, l + 1)
    if math.isfinite(v):
        outer_left_dangle_energy = dangle_outer_left(seq, i, costs)
        outer_right_dangle_energy = dangle_outer_right(seq, j, costs)
        tracker.consider(
            outer_left_dangle_energy + outer_right_dangle_energy + tilde_p_out + tilde_m_yhx + tilde_m_whx + v + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left_right, outer=(i, j), hole=(k, l))
        )

import math


# Reuse CandTracker, best_split, scan_is2_outer_simple from earlier utilities

def consider_whx_hole_shrinks(
    tracker: "CandTracker",
    state: "EddyRivasFoldState",
    i: int, j: int, k: int, l: int,
    q_ss: float,
):
    # 1. Add An Unpaired Base at The 5' End of The Hole (Shrink Hole Left: (k+1,l))
    v = get_whx_with_collapse(state.whx_matrix, state.wxu_matrix, i, j, k + 1, l)
    if math.isfinite(v):
        tracker.consider(
            v + q_ss,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT, outer=(i, j), hole=(k, l))
        )

    # 2. Add an Unpaired Base at The 3' End of The Hole (Shrink Hole Right: (k,l-1))
    v = get_whx_with_collapse(state.whx_matrix, state.wxu_matrix, i, j, k, l - 1)
    if math.isfinite(v):
        tracker.consider(
            v + q_ss,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SHRINK_RIGHT, outer=(i, j), hole=(k, l))
        )

def consider_whx_outer_trims(
    tracker: "CandTracker",
    state: "EddyRivasFoldState",
    i: int, j: int, k: int, l: int,
    q_ss: float,
):
    # 1. Add an Unpaired Base at The 5' End of The Outer Span (Trim Left: (i+1,j))
    v = state.whx_matrix.get(i + 1, j, k, l)
    if math.isfinite(v):
        tracker.consider(
            v + q_ss,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_TRIM_LEFT, outer=(i, j), hole=(k, l))
        )

    # 2. Add An Unpaired Base at The 3' End of The Outer Span (Trim Right: (i,j-1))
    v = state.whx_matrix.get(i, j - 1, k, l)
    if math.isfinite(v):
        tracker.consider(
            v + q_ss,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_TRIM_RIGHT, outer=(i, j), hole=(k, l))
        )

def consider_whx_collapse(
    tracker: "CandTracker",
    state: "EddyRivasFoldState",
    i: int, j: int, k: int, l: int,
):
    v = get_whx_with_collapse(state.whx_matrix, state.wxu_matrix, i, j, k, l)
    if math.isfinite(v):
        tracker.consider(
            v,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE, outer=(i, j), hole=(k, l))
        )

def consider_whx_ss_both(
    tracker: "CandTracker",
    state: "EddyRivasFoldState",
    i: int, j: int, k: int, l: int,
    q_ss: float,
):
    v = state.whx_matrix.get(i + 1, j - 1, k, l)
    if math.isfinite(v):
        tracker.consider(
            v + 2.0 * q_ss,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SS_BOTH, outer=(i, j), hole=(k, l))
        )

def consider_whx_splits(
    tracker: "CandTracker",
    state: "EddyRivasFoldState",
    i: int, j: int, k: int, l: int,
):
    span_len = j - i
    if span_len <= 0:
        return

    # 1. Split Left into WHX(i,r:k,l) + WX(r+1,j)
    cand, t = best_split(
        span_len,
        left_fetch=lambda t: state.whx_matrix.get(i, i + t, k, l),
        right_fetch=lambda t: get_wxi_or_wx(state, i + t + 1, j),
    )
    if t >= 0:
        tracker.consider(
            cand,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX,
                                 outer=(i, j), hole=(k, l), split=i + t)
        )

    # 2. Right split into WX(i,s) + WHX(s+1,j:k,l)
    cand, t = best_split(
        span_len,
        left_fetch=lambda t: get_wxi_or_wx(state, i, i + t),
        right_fetch=lambda t: state.whx_matrix.get(i + t + 1, j, k, l),
    )
    if t >= 0:
        tracker.consider(
            cand,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SPLIT_RIGHT_WX_WHX,
                                 outer=(i, j), hole=(k, l), split=i + t)
        )

def consider_whx_overlap_split(
    tracker: "CandTracker",
    state: "EddyRivasFoldState",
    i: int, j: int, k: int, l: int,
    overlap_penalty: float,
):
    span_len = j - i
    if span_len <= 0 or overlap_penalty == 0.0:
        return
    cand, t = best_split(
        span_len,
        left_fetch=lambda t: state.whx_matrix.get(i, i + t, k, l),
        right_fetch=lambda t: state.whx_matrix.get(i + t + 1, j, k, l),
        penalty=float(overlap_penalty),
    )
    if t >= 0:
        tracker.consider(
            cand,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_OVERLAP_SPLIT,
                                 outer=(i, j), hole=(k, l), split=i + t)
        )

def consider_whx_is2(
    tracker: "CandTracker",
    state: "EddyRivasFoldState",
    cfg,
    seq: str,
    i: int, j: int, k: int, l: int,
):
    is2_best, is2_bp, _ = scan_is2_outer_simple(
        state, cfg, seq, i, j, k, l,
        inner_matrix="yhx",     # WHX uses YHX as inner
        bridge_kind="yhx",      # IS2_outer_yhx
        op=EddyRivasBacktrackOp.RE_WHX_IS2_INNER_YHX
    )
    if is2_bp is not None:
        r2, s2 = is2_bp
        tracker.consider(
            is2_best,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_IS2_INNER_YHX,
                                 outer=(i, j), hole=(k, l), bridge=(r2, s2))
        )
