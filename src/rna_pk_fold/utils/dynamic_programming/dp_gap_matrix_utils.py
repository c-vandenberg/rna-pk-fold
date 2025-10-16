from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import math

import numpy as np

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer, EddyRivasBacktrackOp
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.energies.energy_pk_ops import (dangle_hole_left, dangle_hole_right, dangle_outer_left,
                                                dangle_outer_right)
from rna_pk_fold.utils.dynamic_programming.matrix_utils import get_whx_energy_with_collapse, get_zhx_energy_with_collapse, get_wxi_or_wx
from rna_pk_fold.utils.energy.is2_utils import compute_is2_outer_bridge_energy, compute_is2_outer_bridge_energy_yhx


# ---------- Best Candidate Tracker with Tie-break to RIGHT ----------
@dataclass
class BestCandidateTracker:
    best_energy: float = math.inf
    backpointer: Optional["EddyRivasBackPointer"] = None

    def update_if_better(self, candidate_energy: float, bp: "EddyRivasBackPointer") -> None:
        if candidate_energy < self.best_energy:
            self.best_energy, self.backpointer = candidate_energy, bp

    def update_pair_with_right_tiebreak(
        self,
        left_energy: float,
        right_energy: float,
        left_bp: "EddyRivasBackPointer",
        right_bp: "EddyRivasBackPointer",
    ) -> None:
        """Prefer RIGHT on ties."""
        if right_energy <= left_energy:
            self.update_if_better(right_energy, right_bp)
        else:
            self.update_if_better(left_energy, left_bp)


# ---------- Common guards ----------
def should_skip_dp_cell(
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    config,
    vxu_lookup: Callable[[int, int], float],
    can_pair_mask: Optional[list[list[bool]]] = None,
    require_kl_pairable: bool = False,
) -> bool:
    # Hole width guard
    hole_width = (l_idx - k_idx - 1)
    if config.min_hole_width != 0 and hole_width < config.min_hole_width:
        return True
    if config.max_hole_width != 0 and hole_width > config.max_hole_width:
        return True

    # Beam guard
    if config.beam_v_threshold != 0.0 and vxu_lookup(k_idx, l_idx) > config.beam_v_threshold:
        return True

    # Optional Watson–Crick mask
    if require_kl_pairable and can_pair_mask is not None and not can_pair_mask[k_idx][l_idx]:
        return True

    return False


# ---------- Splits (Generic) ----------
def compute_best_split_sum(
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
        left_val = left_fetch(t)
        right_val = right_fetch(t)
        if math.isfinite(left_val):
            left_vec[t] = left_val
        if math.isfinite(right_val):
            right_vec[t] = right_val

    value_vec = left_vec + right_vec + (0.0 if penalty == 0.0 else penalty)
    t_star = int(np.argmin(value_vec))
    best_val = float(value_vec[t_star])
    if not math.isfinite(best_val):
        return math.inf, -1

    return best_val, t_star


# ---------- IS2 (Outer Bridge Scan) ----------
def scan_is2_outer_min_bridge(
    state,
    config,
    seq: str,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    inner_matrix: str,  # "vhx" | "zhx" | "yhx" | "whx"
    bridge_kind: str,  # "default"->is2_outer(...) ; "yhx"->is2_outer_yhx(...)
    op,
) -> Tuple[float, Optional[Tuple[int, int]], object]:
    # Accessors (no nested defs)
    if inner_matrix == "vhx":
        inner_get = lambda r, s2: state.vhx_matrix.get(r, s2, k_idx, l_idx)
    elif inner_matrix == "zhx":
        inner_get = lambda r, s2: get_zhx_energy_with_collapse(state.zhx_matrix, state.vxu_matrix, r, s2, k_idx, l_idx)
    elif inner_matrix == "yhx":
        inner_get = lambda r, s2: state.yhx_matrix.get(r, s2, k_idx, l_idx)
    elif inner_matrix == "whx":
        inner_get = lambda r, s2: get_whx_energy_with_collapse(state.whx_matrix, state.wxu_matrix, r, s2, k_idx, l_idx)
    else:
        raise ValueError(f"unsupported inner matrix: {inner_matrix}")

    if bridge_kind == "yhx":
        bridge_get = lambda r, s2: compute_is2_outer_bridge_energy_yhx(config, seq, i_idx, j_idx, r, s2)
    else:
        bridge_get = lambda r, s2: compute_is2_outer_bridge_energy(seq, config.tables, i_idx, j_idx, r, s2)

    best_val = math.inf
    best_bridge: Optional[Tuple[int, int]] = None
    candidate_energy = 0.0

    for r in range(i_idx, k_idx + 1):
        for s2 in range(l_idx, j_idx + 1):
            if r <= s2:
                inner_val = inner_get(r, s2)
                if math.isfinite(inner_val):
                    candidate_energy = bridge_get(r, s2) + inner_val
                if candidate_energy < best_val:
                    best_val, best_bridge = candidate_energy, (r, s2)

    return best_val, best_bridge, op

# ---------- SS Helpers (Left/Right/Both with RIGHT Tie-Break) ----------
def update_tracker_for_hole_ss_right_tiebreak(
    tracker: BestCandidateTracker,
    zhx_get: Callable[[int, int, int, int], float],
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    q_hole: float,
    op_left,
    op_right,
) -> None:
    # 3.1. Single Strand Left: Add base on the 5' side.
    energy_left = zhx_get(i_idx, j_idx, k_idx - 1, l_idx)

    # 3.1. Single Strand Right: Add base on the 3' side.
    energy_right = zhx_get(i_idx, j_idx, k_idx, l_idx + 1)

    left_candidate = q_hole + energy_left if math.isfinite(energy_left) else math.inf
    right_candidate = q_hole + energy_right if math.isfinite(energy_right) else math.inf

    tracker.update_pair_with_right_tiebreak(
        left_candidate,
        right_candidate,
        EddyRivasBackPointer(op=op_left, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        EddyRivasBackPointer(op=op_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
    )

def update_tracker_for_outer_ss_right_tiebreak(
    tracker: BestCandidateTracker,
    yhx_get: Callable[[int, int, int, int], float],
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    q_out: float,
    op_left,
    op_right,
) -> None:
    # 1. Single Strand Left: Trim from the 5' (Left) Side.
    energy_left = yhx_get(i_idx + 1, j_idx, k_idx, l_idx)

    # 2. Single Strand Right: Trim from the 3' (Left) Side.
    energy_right = yhx_get(i_idx, j_idx - 1, k_idx, l_idx)

    left_candidate = q_out + energy_left if math.isfinite(energy_left) else math.inf
    right_candidate = q_out + energy_right if math.isfinite(energy_right) else math.inf

    tracker.update_pair_with_right_tiebreak(
        left_candidate,
        right_candidate,
        EddyRivasBackPointer(op=op_left, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        EddyRivasBackPointer(op=op_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
    )

def update_tracker_for_outer_ss_both(
    tracker: BestCandidateTracker,
    v_energy: float,
    q_out: float,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    op,
) -> None:
    """Add unpaired bases at both outer ends."""
    if math.isfinite(v_energy):
        tracker.update_if_better(
            2.0 * q_out + v_energy,
            EddyRivasBackPointer(op=op, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


# ---------- Dangle Wrappers ----------
def update_tracker_for_hole_dangles_from_vhx(
    tracker: BestCandidateTracker,
    vhx_get: Callable[[int, int, int, int], float],
    seq: str,
    costs,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    tilde_p_hole: float,
    internal_pk_penalty: float,
    op_left,
    op_right,
    op_left_right,
) -> None:
    # Left Right
    energy = vhx_get(i_idx, j_idx, k_idx - 1, l_idx + 1)
    if math.isfinite(energy):
        left_hole_energy = dangle_hole_left(seq, k_idx, costs)
        right_hole_energy = dangle_hole_right(seq, l_idx, costs)
        tracker.update_if_better(
            left_hole_energy + right_hole_energy + tilde_p_hole + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # Right
    energy = vhx_get(i_idx, j_idx, k_idx - 1, l_idx)
    if math.isfinite(energy):
        right_hole_energy = dangle_hole_right(seq, l_idx - 1, costs)
        tracker.update_if_better(
            right_hole_energy + tilde_p_hole + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # Left
    energy = vhx_get(i_idx, j_idx, k_idx, l_idx + 1)
    if math.isfinite(energy):
        left_hole_energy = dangle_hole_left(seq, k_idx + 1, costs)
        tracker.update_if_better(
            left_hole_energy + tilde_p_hole + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


def update_tracker_for_outer_dangles_from_vhx(
    tracker: BestCandidateTracker,
    vhx_get: Callable[[int, int, int, int], float],
    seq: str,
    costs,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    tilde_p_out: float,
    internal_pk_penalty: float,
    op_left,
    op_right,
    op_left_right,
) -> None:
    # Case 1: Dangles on the Left Side of the Outer Pair
    energy = vhx_get(i_idx + 1, j_idx, k_idx, l_idx)
    if math.isfinite(energy):
        outer_left_dangle_energy = dangle_outer_left(seq, i_idx, costs)
        tracker.update_if_better(
            outer_left_dangle_energy + tilde_p_out + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # Case 2: Dangles on the Right Side of the Outer Pair
    energy = vhx_get(i_idx, j_idx - 1, k_idx, l_idx)
    if math.isfinite(energy):
        outer_right_dangle_energy = dangle_outer_right(seq, j_idx, costs)
        tracker.update_if_better(
            outer_right_dangle_energy + tilde_p_out + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # Case 3: Dangles on Both Sides of the Outer Pair
    energy = vhx_get(i_idx + 1, j_idx - 1, k_idx, l_idx)
    if math.isfinite(energy):
        outer_left_dangle_energy = dangle_outer_left(seq, i_idx, costs)
        outer_right_dangle_energy = dangle_outer_right(seq, j_idx, costs)
        tracker.update_if_better(
            outer_left_dangle_energy
            + outer_right_dangle_energy
            + tilde_p_out
            + energy
            + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


# ---------- VHX Internal Dangles (No Sequence Lookup) ----------
def update_tracker_for_vhx_inner_dangles(
    tracker: BestCandidateTracker,
    vhx_get: Callable[[int, int, int, int], float],
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    tilde_p_hole: float,
    tilde_l: float,
    tilde_r: float,
    op_left,
    op_right,
    op_lr,
) -> None:
    # -------- Case 1: Dangle on the 5' (Left) Side of The Inner Pair (k,l). --------
    energy = vhx_get(i_idx, j_idx, k_idx + 1, l_idx)
    tracker.update_if_better(
        tilde_p_hole + tilde_l + energy,
        EddyRivasBackPointer(op=op_left, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
    )

    # -------- Case 2: Dangle on the 3' (Right) Side of the Inner Pair (k,l). --------
    energy = vhx_get(i_idx, j_idx, k_idx, l_idx - 1)
    tracker.update_if_better(
        tilde_p_hole + tilde_r + energy,
        EddyRivasBackPointer(op=op_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
    )

    # -------- Case 3: Dangles on Both Sides of the Inner Pair (k,l). --------
    energy = vhx_get(i_idx, j_idx, k_idx + 1, l_idx - 1)
    tracker.update_if_better(
        tilde_p_hole + tilde_l + tilde_r + energy,
        EddyRivasBackPointer(op=op_lr, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
    )


# ---------- Multiloop Wrappers (WHX Close/Wrap) ----------
def update_tracker_for_vhx_multiloop_close_and_wrap(
    tracker: BestCandidateTracker,
    whx_collapse_get: Callable[[int, int, int, int], float],
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    tilde_p_hole: float,
    tilde_m_vhx: float,
    tilde_m_whx: float,
    internal_pk_penalty: float,
    op_close_both,
    op_wrap_whx,
) -> None:
    # Close both (i+1,j-1 : k-1,l+1)
    energy_close = whx_collapse_get(i_idx + 1, j_idx - 1, k_idx - 1, l_idx + 1)
    if math.isfinite(energy_close):
        candidate = 2.0 * tilde_p_hole + tilde_m_vhx + energy_close + internal_pk_penalty + tilde_m_whx
        tracker.update_if_better(candidate,
                                 EddyRivasBackPointer(op=op_close_both, outer=(i_idx, j_idx), hole=(k_idx, l_idx)))

    # Wrap (i+1,j-1 : k,l)
    energy_wrap = whx_collapse_get(i_idx + 1, j_idx - 1, k_idx, l_idx)
    if math.isfinite(energy_wrap):
        candidate = tilde_p_hole + tilde_m_vhx + energy_wrap + internal_pk_penalty + tilde_m_whx
        tracker.update_if_better(candidate,
                                 EddyRivasBackPointer(op=op_wrap_whx, outer=(i_idx, j_idx), hole=(k_idx, l_idx)))


def update_tracker_for_yhx_wrap_whx(
    tracker: BestCandidateTracker,
    whx_get: Callable[[int, int, int, int], float],
    seq: str,
    costs,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    tilde_p_out: float,
    tilde_m_yhx: float,
    tilde_m_whx: float,
    internal_pk_penalty: float,
    op_plain,
    op_left,
    op_right,
    op_left_right,
) -> None:
    # 1. Plain Multiloop Wrap
    energy = whx_get(i_idx, j_idx, k_idx - 1, l_idx + 1)
    if math.isfinite(energy):
        tracker.update_if_better(
            tilde_p_out + tilde_m_yhx + tilde_m_whx + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_plain, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # 2. Wrap on 5' (Left) Side
    energy = whx_get(i_idx + 1, j_idx, k_idx - 1, l_idx + 1)
    if math.isfinite(energy):
        outer_left_dangle_energy = dangle_outer_left(seq, i_idx, costs)
        tracker.update_if_better(
            outer_left_dangle_energy + tilde_p_out + tilde_m_yhx + tilde_m_whx + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # 3. Wrap on 3' (Right) Side
    energy = whx_get(i_idx, j_idx - 1, k_idx - 1, l_idx + 1)
    if math.isfinite(energy):
        outer_right_dangle_energy = dangle_outer_right(seq, j_idx, costs)
        tracker.update_if_better(
            outer_right_dangle_energy + tilde_p_out + tilde_m_yhx + tilde_m_whx + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # 4. Wrap on Both Sides
    energy = whx_get(i_idx + 1, j_idx - 1, k_idx - 1, l_idx + 1)
    if math.isfinite(energy):
        outer_left_dangle_energy = dangle_outer_left(seq, i_idx, costs)
        outer_right_dangle_energy = dangle_outer_right(seq, j_idx, costs)
        tracker.update_if_better(
            outer_left_dangle_energy
            + outer_right_dangle_energy
            + tilde_p_out
            + tilde_m_yhx
            + tilde_m_whx
            + energy
            + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


# ---------- WHX Helpers ----------
def update_tracker_for_whx_hole_shrinks(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    q_single_strand: float,
) -> None:
    # 1. Add An Unpaired Base at The 5' End of The Hole (Shrink Hole Left: (k+1,l))
    energy = get_whx_energy_with_collapse(state.whx_matrix, state.wxu_matrix, i_idx, j_idx, k_idx + 1, l_idx)
    if math.isfinite(energy):
        tracker.update_if_better(
            energy + q_single_strand,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT, outer=(i_idx, j_idx), hole=(k_idx, l_idx))
        )

    # 2. Add an Unpaired Base at The 3' End of The Hole (Shrink Hole Right: (k,l-1))
    energy = get_whx_energy_with_collapse(state.whx_matrix, state.wxu_matrix, i_idx, j_idx, k_idx, l_idx - 1)
    if math.isfinite(energy):
        tracker.update_if_better(
            energy + q_single_strand,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SHRINK_RIGHT, outer=(i_idx, j_idx),
                                 hole=(k_idx, l_idx))
        )


def update_tracker_for_whx_outer_trims(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    q_single_strand: float,
) -> None:
    # 1. Add an Unpaired Base at The 5' End of The Outer Span (Trim Left: (i+1,j))
    energy = state.whx_matrix.get(i_idx + 1, j_idx, k_idx, l_idx)
    if math.isfinite(energy):
        tracker.update_if_better(
            energy + q_single_strand,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_TRIM_LEFT, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # 2. Add An Unpaired Base at The 3' End of The Outer Span (Trim Right: (i,j-1))
    energy = state.whx_matrix.get(i_idx, j_idx - 1, k_idx, l_idx)
    if math.isfinite(energy):
        tracker.update_if_better(
            energy + q_single_strand,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_TRIM_RIGHT, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


def update_tracker_for_whx_collapse(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
) -> None:
    energy = get_whx_energy_with_collapse(state.whx_matrix, state.wxu_matrix, i_idx, j_idx, k_idx, l_idx)
    if math.isfinite(energy):
        tracker.update_if_better(
            energy,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


def update_tracker_for_whx_single_strand_both(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    q_single_strand: float,
) -> None:
    energy = state.whx_matrix.get(i_idx + 1, j_idx - 1, k_idx, l_idx)
    if math.isfinite(energy):
        tracker.update_if_better(
            energy + 2.0 * q_single_strand,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SS_BOTH, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


def update_tracker_for_whx_splits(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
) -> None:
    span_len = j_idx - i_idx
    if span_len <= 0:
        return

    # 1. Split Left into WHX(i,r:k,l) + WX(r+1,j)
    cand, t_star = compute_best_split_sum(
        span_len,
        left_fetch=lambda t: state.whx_matrix.get(i_idx, i_idx + t, k_idx, l_idx),
        right_fetch=lambda t: get_wxi_or_wx(state, i_idx + t + 1, j_idx),
    )
    if t_star >= 0:
        tracker.update_if_better(
            cand,
            EddyRivasBackPointer(
                op=EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX,
                outer=(i_idx, j_idx),
                hole=(k_idx, l_idx),
                split=i_idx + t_star,
            ),
        )

    # 2. Right split into WX(i,s) + WHX(s+1,j:k,l)
    cand, t_star = compute_best_split_sum(
        span_len,
        left_fetch=lambda t: get_wxi_or_wx(state, i_idx, i_idx + t),
        right_fetch=lambda t: state.whx_matrix.get(i_idx + t + 1, j_idx, k_idx, l_idx),
    )
    if t_star >= 0:
        tracker.update_if_better(
            cand,
            EddyRivasBackPointer(
                op=EddyRivasBacktrackOp.RE_WHX_SPLIT_RIGHT_WX_WHX,
                outer=(i_idx, j_idx),
                hole=(k_idx, l_idx),
                split=i_idx + t_star,
            ),
        )


def update_tracker_for_whx_overlap_split(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    overlap_penalty: float,
) -> None:
    span_len = j_idx - i_idx
    if span_len <= 0 or overlap_penalty == 0.0:
        return

    cand, t_star = compute_best_split_sum(
        span_len,
        left_fetch=lambda t: state.whx_matrix.get(i_idx, i_idx + t, k_idx, l_idx),
        right_fetch=lambda t: state.whx_matrix.get(i_idx + t + 1, j_idx, k_idx, l_idx),
        penalty=float(overlap_penalty),
    )
    if t_star >= 0:
        tracker.update_if_better(
            cand,
            EddyRivasBackPointer(
                op=EddyRivasBacktrackOp.RE_WHX_OVERLAP_SPLIT,
                outer=(i_idx, j_idx),
                hole=(k_idx, l_idx),
                split=i_idx + t_star,
            ),
        )


def update_tracker_for_whx_is2(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    config,
    seq: str,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
) -> None:
    """Evaluate IS2 (YHX inner + YHX bridge) for WHX and update tracker."""
    is2_best, is2_bridge_bp, _ = scan_is2_outer_min_bridge(
        state,
        config,
        seq,
        i_idx,
        j_idx,
        k_idx,
        l_idx,
        inner_matrix="yhx",  # WHX uses YHX as inner
        bridge_kind="yhx",   # IS2_outer_yhx
        op=EddyRivasBacktrackOp.RE_WHX_IS2_INNER_YHX,
    )
    if is2_bridge_bp is not None:
        r_idx, s2_idx = is2_bridge_bp
        tracker.update_if_better(
            is2_best,
            EddyRivasBackPointer(
                op=EddyRivasBacktrackOp.RE_WHX_IS2_INNER_YHX,
                outer=(i_idx, j_idx),
                hole=(k_idx, l_idx),
                bridge=(r_idx, s2_idx),
            ),
        )
