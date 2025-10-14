import math
import numpy as np
from typing import Optional, Tuple

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer, EddyRivasBacktrackOp
from rna_pk_fold.energies.energy_pk_ops import short_hole_penalty, coax_pack
from rna_pk_fold.folding.eddy_rivas.numba_kernels import compose_wx_best_over_r_arrays, compose_vx_best_over_r
from rna_pk_fold.utils.dynamic_programming.matrix_utils import whx_collapse_with, zhx_collapse_with
from rna_pk_fold.utils.sequences.iter_utils import iter_inner_holes


# ---------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------
def publish_2d_cell(matrix, backptr_store, i: int, j: int, val: float, bp) -> None:
    matrix.set(i, j, val)
    if bp is not None:
        backptr_store.set(i, j, bp)


def hole_len(k: int, l: int) -> int:
    return l - k - 1


def subhole_is_nonempty(hole: Tuple[int, int]) -> bool:
    k, l = hole
    return (l - k) > 1


# ---------------------------------------------------------------------
# WX composition: array preparation (one pass over r)
# ---------------------------------------------------------------------
def prepare_wx_r_arrays(
    state,
    cfg,
    i: int, j: int, k: int, l: int,
    can_pair_mask,
):
    """
    Build arrays over r in [k..l-1] for compose_wx_best_over_r_arrays.
    Leaves invalid positions as +inf and flags as 0.
    """
    # --- Pre-computation Step for Numba Kernel ---
    # Create vectors to store energies for every possible split point 'r' between k and l.
    # This vectorization allows for efficient processing by a Numba kernel.
    base_l = l - k
    l_u = np.full(base_l, np.inf, dtype=np.float64)  # WHX left (uncharged, nested sub-problem)
    r_u = np.full(base_l, np.inf, dtype=np.float64)  # WHX right (uncharged)
    l_c = np.full(base_l, np.inf, dtype=np.float64)  # WHX left (charged, pseudoknotted sub-problem)
    r_c = np.full(base_l, np.inf, dtype=np.float64)  # WHX right (charged)

    left_y  = np.full(base_l, np.inf, dtype=np.float64)  # Energies from YHX matrix (left)
    right_y = np.full(base_l, np.inf, dtype=np.float64)  # Energies from YHX matrix (right)
    left_y_is_charged  = np.zeros(base_l, dtype=np.uint8)
    right_y_is_charged = np.zeros(base_l, dtype=np.uint8)

    # Iterate through all possible split points 'r' to populate the energy vectors.
    for t in range(base_l):
        r = k + t

        # Enforce the strict Rivas & Eddy ordering for pseudoknot helices.
        if cfg.strict_complement_order and not (i < k <= r < l <= j):
            continue

        # Enforce minimum lengths for the 5' and 3' outer segments.
        if (r - i) < cfg.min_outer_left or (j - (r + 1)) < cfg.min_outer_right:
            continue

        # Calculate energies for the left and right gapped subproblems.
        # 'whx_collapse_with' gets the energy, handling cases where the subproblem's hole is empty.
        # 'charged=False' (u) gets the nested baseline; 'charged=True' (c) gets the pseudoknotted energy.
        l_u[t] = whx_collapse_with(state, i, r, k, r, charged=False, can_pair_mask=can_pair_mask)
        r_u[t] = whx_collapse_with(state, r + 1, j, r + 1, l, charged=False, can_pair_mask=can_pair_mask)
        l_c[t] = whx_collapse_with(state, i, r, k, r, charged=True,  can_pair_mask=can_pair_mask)
        r_c[t] = whx_collapse_with(state, r + 1, j, r + 1, l, charged=True,  can_pair_mask=can_pair_mask)

        # Get energies from the YHX matrix (only if the pair is allowed)
        # Left YHX
        if can_pair_mask is not None and can_pair_mask[k][r]:
            ly = state.yhx_matrix.get(i, r, k, r)
            if math.isfinite(ly):
                left_y[t] = ly
                bp_ly = state.yhx_back_ptr.get(i, r, k, r)
                if bp_ly is not None and getattr(bp_ly, "charged", False):
                    left_y_is_charged[t] = 1

        # Right YHX
        if can_pair_mask is not None and can_pair_mask[r + 1][l]:
            ry = state.yhx_matrix.get(r + 1, j, r + 1, l)
            if math.isfinite(ry):
                right_y[t] = ry
                bp_ry = state.yhx_back_ptr.get(r + 1, j, r + 1, l)
                if bp_ry is not None and getattr(bp_ry, "charged", False):
                    right_y_is_charged[t] = 1

    return l_u, r_u, l_c, r_c, left_y, right_y, left_y_is_charged, right_y_is_charged


# ---------------------------------------------------------------------
# WX composition: decode/validate kernel case_id
# ---------------------------------------------------------------------
def _wx_case_uses(
    case_id: int,
    t_star: int,
    l_u, r_u, l_c, r_c, left_y, right_y,
) -> Tuple[bool, bool]:
    """
    Determine which vectors contributed based on case_id.

    This is a critical filter: ensure that both the left and right sub-fragments
    have a defined gapped structure. This prevents selecting combinations where
    one side has simply collapsed to a nested structure, which wouldn't form a true
    pseudoknot.

    Which vectors contributed is determined based on case_id.
    Return (left_has_structure, right_has_structure) for the chosen case_id.
    """
    if t_star < 0:
        return False, False

    # Determine which vectors contributed based on case_id
    # This is a critical filter: ensure that both the left and right sub-fragments
    # have a defined gapped structure. This prevents selecting combinations where
    # one side has simply collapsed to a nested structure, which wouldn't form a true pseudoknot.
    # Which vectors contributed is determined based on case_id
    if case_id == 0:   # Lu + Ru (uncharged + uncharged)
        return np.isfinite(l_u[t_star]), np.isfinite(r_u[t_star])
    if case_id == 1:   # Lu + Rc (uncharged + charged)
        return np.isfinite(l_u[t_star]), np.isfinite(r_c[t_star])
    if case_id == 2:   # Lc + Ru (charged + uncharged)
        return np.isfinite(l_c[t_star]), np.isfinite(r_u[t_star])
    if case_id == 3:   # Lc + Rc (charged + charged)
        return np.isfinite(l_c[t_star]), np.isfinite(r_c[t_star])
    if case_id == 4:   # YHX + YHX
        return np.isfinite(left_y[t_star]), np.isfinite(right_y[t_star])
    if case_id == 5:   # YHX + WHX(u)
        return np.isfinite(left_y[t_star]), np.isfinite(r_u[t_star])
    if case_id == 6:   # YHX + WHX(c)
        return np.isfinite(left_y[t_star]), np.isfinite(r_c[t_star])
    if case_id == 7:   # WHX(u) + YHX
        return np.isfinite(l_u[t_star]), np.isfinite(right_y[t_star])
    if case_id == 8:   # WHX(c) + YHX
        return np.isfinite(l_c[t_star]), np.isfinite(right_y[t_star])

    return False, False


def decode_wx_backpointer(
    case_id: int,
    i: int, j: int, k: int, l: int, r_star: int,
    left_y_is_charged, right_y_is_charged, t_star: int
) -> Tuple["EddyRivasBacktrackOp", Tuple[int,int], Tuple[int,int], bool]:
    """
    Decode the 'case_id' from the kernel to determine the backtrack operation
    and map kernel case to `(op, hole_left, hole_right, charged_flag)` tuple.
    """
    hole_left  = (k, r_star)
    hole_right = (r_star + 1, l)

    if case_id in (0, 1, 2, 3):
        # Case 0, 1 & 2: WHX + WHX (uu/cu/uc): no Gw, not charged
        # Case 3: WHX + WHX (cc): Gw applied in kernel; charged
        charged = (case_id == 3)
        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX
        return op, hole_left, hole_right, charged

    if case_id == 4:
        # Case 4: YHX + YHX: Gw added only if both charged (decided in kernel);
        charged = bool(left_y_is_charged[t_star] and right_y_is_charged[t_star])
        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX
        return op, hole_left, hole_right, charged

    if case_id in (5, 6):
        # Case 5: YHX + WHX (u): no Gw, not charged
        # Case 6: YHX + WHX (c): Gw only if left Y is charged
        charged = (case_id == 6) and bool(left_y_is_charged[t_star])
        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX
        return op, hole_left, hole_right, charged

    if case_id in (7, 8):
        # Case 7: WHX (u) + YHX: no Gw, not charged
        # Case 8: WHX (c) + YHX: Gw only if right Y is charged
        charged = (case_id == 8) and bool(right_y_is_charged[t_star])
        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX
        return op, hole_left, hole_right, charged

    # Fallback (shouldn't happen)
    return EddyRivasBacktrackOp.RE_PK_COMPOSE_WX, hole_left, hole_right, False


# ---------------------------------------------------------------------
# WX composition: one-hole composition wrapper
# ---------------------------------------------------------------------
def compose_wx_for_hole(
    state,
    cfg,
    seq: str,
    i: int, j: int, k: int, l: int,
    pseudoknot_penalty: float,
    can_pair_mask,
):
    """
    Returns (cand, bp_or_None). If invalid (no proper PK), returns (inf, None).
    """
    # Build arrays across split point `r`
    (l_u, r_u, l_c, r_c,
     left_y, right_y, left_y_is_charged, right_y_is_charged) = prepare_wx_r_arrays(
        state, cfg, i, j, k, l, can_pair_mask
    )

    # Calculate penalty for very short loops between helices.
    cap_pen = short_hole_penalty(cfg.costs, k, l)

    # --- Kernel Execution ---
    # Pass the energy vectors to the optimized Numba kernel to find the best split point 'r'
    # and the best combination of subproblems (WHX+WHX, YHX+YHX, etc.).
    cand, t_star, case_id = compose_wx_best_over_r_arrays(
        l_u, r_u, l_c, r_c, left_y, right_y,
        left_y_is_charged, right_y_is_charged,
        float(pseudoknot_penalty), float(cap_pen)
    )

    # Invalid split index -> reject (proceed only if the kernel returned a valid split point).
    if t_star < 0:
        return math.inf, None

    # Validate that both sides have gapped structure (not pure collapse)
    left_ok, right_ok = _wx_case_uses(case_id, t_star, l_u, r_u, l_c, r_c, left_y, right_y)
    if not (left_ok and right_ok):
        return math.inf, None

    # If all checks pass, this is a valid, new best pseudoknot candidate.
    r_star = k + t_star # Recalculate best split point.
    hole_left  = (k, r_star)
    hole_right = (r_star + 1, l)
    if not (subhole_is_nonempty(hole_left) and subhole_is_nonempty(hole_right)):
        return math.inf, None

    # # Decode the 'case_id' and backpointer fields from the kernel to determine the backtrack
    # operation.
    op, hole_left, hole_right, charged = decode_wx_backpointer(
        case_id, i, j, k, l, r_star, left_y_is_charged, right_y_is_charged, t_star
    )

    # Create the backpointer for this optimal pseudoknot configuration.
    bp = EddyRivasBackPointer(
        op=op, outer=(i, j), hole=(k, l),
        hole_left=hole_left, hole_right=hole_right,
        split=r_star, charged=charged
    )
    return cand, bp


# ---------------------------------------------------------------------
# WX composition: optional YHX-overlap path
# ---------------------------------------------------------------------
def compose_wx_yhx_overlap_for_span(
    state, cfg, i: int, j: int, g_wh_wx: float
) -> Tuple[float, Optional["EddyRivasBackPointer"]]:
    """
    Optional overlap path: A class of pseudoknots where two YHX fragments are
    overlapping the same hole.

    Returns the best (cand, bp) across all inner holes/splits for this (i,j).
    """
    if (not cfg.enable_wx_overlap) or g_wh_wx == 0.0:
        return math.inf, None

    best = math.inf
    best_bp = None

    # Iterate through a different set of inner holes and split points.
    for (k2, l2) in iter_inner_holes(i, j, min_hole_width=cfg.min_hole_width):
        cap = short_hole_penalty(cfg.costs, k2, l2)
        for r2 in range(i, j):
            left_yv = state.yhx_matrix.get(i, r2, k2, l2)
            right_yv = state.yhx_matrix.get(r2 + 1, j, k2, l2)

            # If both sub-problems have finite energy, calculate the total energy.
            if math.isfinite(left_yv) and math.isfinite(right_yv):
                cand = g_wh_wx + left_yv + right_yv + cap
                # If this is a new best energy, update the backpointer.
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_OVERLAP,
                        outer=(i, j), hole=(k2, l2), split=r2, charged=True
                    )

    return best, best_bp


# ---------------------------------------------------------------------
# VX composition: array preparation and kernel wrapper
# ---------------------------------------------------------------------
def prepare_vx_r_arrays_and_coax(
    state,
    cfg,
    seq: str,
    i: int, j: int, k: int, l: int,
    can_pair_mask,
):
    """
    Build arrays over r in [k..l-1] for compose_vx_best_over_r.
    """
    # --- Pre-computation Step for Numba Kernal ---
    # Create vectors to store energies for every possible split point 'r' between k and l.
    # This vectorization allows for efficient processing by a Numba kernel.
    base_l = l - k
    l_u = np.full(base_l, np.inf, dtype=np.float64)  # ZHX left (uncharged, nested subproblem)
    r_u = np.full(base_l, np.inf, dtype=np.float64)  # ZHX right (uncharged)
    l_c = np.full(base_l, np.inf, dtype=np.float64)  # ZHX left (charged, pseudoknotted subproblem)
    r_c = np.full(base_l, np.inf, dtype=np.float64)  # ZHX right (charged)
    coax_total = np.zeros(base_l, dtype=np.float64)  # Total coaxial stacking energy
    coax_bonus = np.zeros(base_l, dtype=np.float64)  # Coaxial stacking bonus energy

    # Iterate through all possible split points 'r' to populate the energy vectors.
    for t in range(base_l):
        r = k + t # 'r' is the split point.

        # Enforce the strict Rivas & Eddy ordering for pseudoknot helices.
        if cfg.strict_complement_order and not (i < k <= r < l <= j):
            continue

        # Enforce minimum lengths for the 5' and 3' outer segments.
        if (r - i) < cfg.min_outer_left or (j - (r + 1)) < cfg.min_outer_right:
            continue

        # Calculate energies for the left and right gapped sub-problems from the ZHX matrix.
        # 'zhx_collapse_with' gets the energy, handling cases where the sub-problem's hole is empty.
        l_u[t] = zhx_collapse_with(state, i, r, k, r, charged=False, can_pair_mask=can_pair_mask)
        r_u[t] = zhx_collapse_with(state, r + 1, j, r + 1, l, charged=False, can_pair_mask=can_pair_mask)
        l_c[t] = zhx_collapse_with(state, i, r, k, r, charged=True,  can_pair_mask=can_pair_mask)
        r_c[t] = zhx_collapse_with(state, r + 1, j, r + 1, l, charged=True,  can_pair_mask=can_pair_mask)

        # Calculate the coaxial stacking energy bonus for this specific split point 'r'.
        adjacent = (r == k) # Check if the helices are adjacent for a flush stack.
        cx_total, cx_bonus = coax_pack(seq, i, j, r, k, l, cfg, cfg.costs, adjacent)
        coax_total[t] = cx_total
        coax_bonus[t] = cx_bonus

    return l_u, r_u, l_c, r_c, coax_total, coax_bonus


def compose_vx_for_hole(
    state,
    cfg,
    seq: str,
    i: int, j: int, k: int, l: int,
    pseudoknot_penalty: float,
    coaxial_scale: float,
    can_pair_mask,
):
    """
    Returns (cand, bp_or_None). If invalid (no proper PK), returns (inf, None).
    """
    # Build arrays across split point `r`
    (l_u, r_u, l_c, r_c, coax_total, coax_bonus) = prepare_vx_r_arrays_and_coax(
        state, cfg, seq, i, j, k, l, can_pair_mask
    )
    cap_pen = short_hole_penalty(cfg.costs, k, l)

    # --- Kernel Execution ---
    # Pass the energy vectors to the optimized Numba kernel. It efficiently finds the
    # best split point 'r' (returned as t_star) and the minimum energy 'cand'.
    cand, t_star, base_case = compose_vx_best_over_r(
        l_u, r_u, l_c, r_c, coax_total, coax_bonus,
        float(pseudoknot_penalty), float(cap_pen), float(coaxial_scale)
    )

    # Invalid split index -> reject (proceed only if the kernel returned a valid split point).
    if t_star < 0:
        return math.inf, None

    # Decide if this composition truly formed a PK (cc case only).
    # Only consider 'charged' if both charged sides are finite
    charged = (base_case == 3) and (np.isfinite(l_c[t_star]) and np.isfinite(r_c[t_star]))

    r_star = k + t_star
    hole_left  = (k, r_star)
    hole_right = (r_star + 1, l)
    if not (subhole_is_nonempty(hole_left) and subhole_is_nonempty(hole_right)):
        return math.inf, None

    bp = EddyRivasBackPointer(
        op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX,
        outer=(i, j), hole=(k, l),
        split=r_star, charged=charged
    )

    return cand, bp
