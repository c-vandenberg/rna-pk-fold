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


# =====================================================================
# Best Candidate Tracker with Tie-break to RIGHT
# =====================================================================
@dataclass
class BestCandidateTracker:
    """
    Track the current best candidate energy and its backpointer.

    This helper centralizes the "keep the minimum" logic while providing a
    consistent tie-break policy.

    Attributes
    ----------
    best_energy : float
        Current best energy (initialized to +inf).
    backpointer : Optional[EddyRivasBackPointer]
        Backpointer associated with `best_energy`, or None if none recorded.
    """
    best_energy: float = math.inf
    backpointer: Optional["EddyRivasBackPointer"] = None

    def update_if_better(self, candidate_energy: float, backpointer: "EddyRivasBackPointer") -> None:
        """
        Update the stored candidate if `candidate_energy` is strictly better.

        Parameters
        ----------
        candidate_energy : float
            Energy value to compare against the current best.
        backpointer : EddyRivasBackPointer
            Backpointer associated with `candidate_energy`.
        """
        if candidate_energy < self.best_energy:
            self.best_energy, self.backpointer = candidate_energy, backpointer

    def update_pair_with_right_tiebreak(
        self,
        left_energy: float,
        right_energy: float,
        left_backpointer: "EddyRivasBackPointer",
        right_backpointer: "EddyRivasBackPointer",
    ) -> None:
        """
        Compare two candidates and prefer the right candidate on ties.

        Parameters
        ----------
        left_energy : float
            Energy of the left candidate.
        right_energy : float
            Energy of the right candidate.
        left_backpointer : EddyRivasBackPointer
            Backpointer for the left candidate.
        right_backpointer : EddyRivasBackPointer
            Backpointer for the right candidate.
        """
        if right_energy <= left_energy:
            self.update_if_better(right_energy, right_backpointer)
        else:
            self.update_if_better(left_energy, left_backpointer)


# =====================================================================
# Guards / Cell-Skipping Logic
# =====================================================================
def should_skip_gap_cell(
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    config,
    vxu_lookup: Callable[[int, int], float],
    can_pair_mask: Optional[list[list[bool]]] = None,
    require_kl_pairable: bool = False,
) -> bool:
    """
    Decide whether a 4D gap DP cell `(i,j:k,l)` should be skipped.

    The decision is based on hole width constraints, a V-beam threshold, and an
    optional Watson–Crick pairability mask for the inner pair `(k,l)`.

    Parameters
    ----------
    i_idx, j_idx : int
        Outer span indices.
    k_idx, l_idx : int
        Inner hole indices.
    config : Any
        Folding configuration (expects `min_hole_width`, `max_hole_width`,
        `beam_v_threshold`).
    vxu_lookup : Callable[[int, int], float]
        Function returning VXU energy for `(k, l)` used by the beam guard.
    can_pair_mask : Optional[list[list[bool]]], default=None
        Optional mask that says whether bases at positions can form a pair.
    require_kl_pairable : bool, default=False
        If True, require `can_pair_mask[k][l]` to be True.

    Returns
    -------
    bool
        True if the cell should be skipped; False otherwise.
    """
    # Hole width guard
    hole_width = (l_idx - k_idx - 1)
    if config.pk_energies.min_hole_width != 0 and hole_width < config.pk_energies.min_hole_width:
        return True
    if config.pk_energies.max_hole_width != 0 and hole_width > config.pk_energies.max_hole_width:
        return True

    # Beam guard
    if config.pk_energies.beam_v_threshold != 0.0 and vxu_lookup(k_idx, l_idx) > config.pk_energies.beam_v_threshold:
        return True

    # Optional Watson–Crick mask
    if require_kl_pairable and can_pair_mask is not None and not can_pair_mask[k_idx][l_idx]:
        return True

    return False


# =====================================================================
# Split Search Helpers
# =====================================================================
def find_best_split_sum(
    length: int,
    left_fetch: Callable[[int], float],
    right_fetch: Callable[[int], float],
    penalty: float = 0.0,
) -> Tuple[float, int]:
    """
    Find `min_t (left[t] + right[t] + penalty)` and its index.

    Parameters
    ----------
    length : int
        Number of split candidates `t` to consider.
    left_fetch : Callable[[int], float]
        Function returning the left energy at index `t`.
    right_fetch : Callable[[int], float]
        Function returning the right energy at index `t`.
    penalty : float, default=0.0
        Constant term added to each candidate sum.

    Returns
    -------
    Tuple[float, int]
        `(best_value, t_star)` where `t_star` is the argmin in `[0, length)`,
        or `(-1)` if no finite candidate exists.
    """
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


# =====================================================================
# IS2: Outer-Bridge Scan
# =====================================================================
def scan_is2_best_outer_bridge(
    state: EddyRivasFoldState,
    config,
    seq: str,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    inner_matrix_name: str,
    bridge_energy_kind: str,
    op: EddyRivasBacktrackOp,
) -> Tuple[float, Optional[Tuple[int, int]], object]:
    """
    Scan all outer-bridge coordinates `(r, s)` and return the best IS2 energy.

    This enumerates `r in [i..k]` and `s in [l..j]`, combines the inner-gap
    energy with the corresponding outer-bridge energy, and returns the minimum.

    Parameters
    ----------
    state : EddyRivasFoldState
        Fold state containing gap and triangular matrices.
    config : Any
        Folding configuration (tables required by bridge energy).
    seq : str
        RNA sequence.
    i_idx, j_idx : int
        Outer span indices.
    k_idx, l_idx : int
        Inner hole indices.
    inner_matrix_name : str
        One of "vhx", "zhx", "yhx", "whx" to choose the inner sub-problem.
    bridge_energy_kind : str
        Either "default" (general bridge) or "yhx" (YHX-specific bridge model).
    op : EddyRivasBacktrackOp
        Operation to annotate in the backpointer if selected.

    Returns
    -------
    Tuple[float, Optional[Tuple[int, int]], EddyRivasBacktrackOp]
        `(best_energy, best_bridge, op)`, where `best_bridge` is `(r, s)` or
        None if no finite candidate exists.
    """
    # Inner sub-problem getter
    if inner_matrix_name == "vhx":
        inner_get = lambda r, s2: state.vhx_matrix.get_energy(r, s2, k_idx, l_idx)
    elif inner_matrix_name == "zhx":
        inner_get = lambda r, s2: get_zhx_energy_with_collapse(state.zhx_matrix, state.vxu_matrix, r, s2, k_idx, l_idx)
    elif inner_matrix_name == "yhx":
        inner_get = lambda r, s2: state.yhx_matrix.get_energy(r, s2, k_idx, l_idx)
    elif inner_matrix_name == "whx":
        inner_get = lambda r, s2: get_whx_energy_with_collapse(state.whx_matrix, state.wxu_matrix, r, s2, k_idx, l_idx)
    else:
        raise ValueError(f"unsupported inner matrix: {inner_matrix_name}")

    if bridge_energy_kind == "yhx":
        bridge_get = lambda r, s2: compute_is2_outer_bridge_energy_yhx(config, seq, i_idx, j_idx, r, s2)
    else:
        bridge_get = lambda r, s2: compute_is2_outer_bridge_energy(seq, config, i_idx, j_idx, r, s2)

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


# =====================================================================
# Single Strand Helpers (Right Tie-Break Variants)
# =====================================================================
def update_tracker_for_hole_ss_with_right_tiebreak(
    tracker: BestCandidateTracker,
    zhx_get: Callable[[int, int, int, int], float],
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    q_hole: float,
    op_left: EddyRivasBacktrackOp,
    op_right: EddyRivasBacktrackOp,
) -> None:
    """
    Evaluate adding a single unpaired base to the hole (left or right) with right tie-break.

    Parameters
    ----------
    tracker : BestCandidateTracker
        Accumulator for the best candidate.
    zhx_get : Callable[[int, int, int, int], float]
        Energy accessor for ZHX-like subproblems.
    i_idx, j_idx : int
        Outer span indices.
    k_idx, l_idx : int
        Inner hole indices.
    q_hole : float
        Single-stranded penalty inside the hole.
    op_left, op_right : EddyRivasBacktrackOp
        Backpointer operations for left/right choices.
    """
    # 1. Single Strand Left: Add base on the 5' side.
    energy_left = zhx_get(i_idx, j_idx, k_idx - 1, l_idx)

    # 2. Single Strand Right: Add base on the 3' side.
    energy_right = zhx_get(i_idx, j_idx, k_idx, l_idx + 1)

    left_candidate = q_hole + energy_left if math.isfinite(energy_left) else math.inf
    right_candidate = q_hole + energy_right if math.isfinite(energy_right) else math.inf

    tracker.update_pair_with_right_tiebreak(
        left_candidate,
        right_candidate,
        EddyRivasBackPointer(op=op_left, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        EddyRivasBackPointer(op=op_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
    )


def update_tracker_for_outer_ss_with_right_tiebreak(
    tracker: BestCandidateTracker,
    yhx_get: Callable[[int, int, int, int], float],
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    q_out: float,
    op_left: EddyRivasBacktrackOp,
    op_right: EddyRivasBacktrackOp,
) -> None:
    """
    Evaluate trimming a single unpaired base on the outer span (left or right) with right tie-break.

    Parameters
    ----------
    tracker : BestCandidateTracker
        Accumulator for the best candidate.
    yhx_get : Callable[[int, int, int, int], float]
        Energy accessor for YHX-like subproblems.
    i_idx, j_idx : int
        Outer span indices.
    k_idx, l_idx : int
        Inner hole indices.
    q_out : float
        Single-stranded penalty on the outer span.
    op_left, op_right : EddyRivasBacktrackOp
        Backpointer operations for left/right choices.
    """
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
    op: EddyRivasBacktrackOp,
) -> None:
    """
    Evaluate adding unpaired bases to both outer ends and update the tracker.

    Parameters
    ----------
    tracker : BestCandidateTracker
        Accumulator for the best candidate.
    v_energy : float
        Energy of the inner VHX-like subproblem.
    q_out : float
        Single-stranded penalty per outer base.
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates for the subproblem.
    op : EddyRivasBacktrackOp
        Backpointer operation.
    """
    if math.isfinite(v_energy):
        tracker.update_if_better(
            2.0 * q_out + v_energy,
            EddyRivasBackPointer(op=op, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


# =====================================================================
# Dangle Wrappers (Sequence-Aware)
# =====================================================================
def update_tracker_for_hole_dangles_using_vhx(
    tracker: BestCandidateTracker,
    vhx_get: Callable[[int, int, int, int], float],
    seq: str,
    pk_energies,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    tilde_p_hole: float,
    internal_pk_penalty: float,
    op_left: EddyRivasBacktrackOp,
    op_right: EddyRivasBacktrackOp,
    op_left_right: EddyRivasBacktrackOp,
) -> None:
    """
    Evaluate hole dangles via VHX transitions and update the tracker.

    Considers left, right, and both dangles around the inner pair `(k,l)`.

    Parameters
    ----------
    tracker : BestCandidateTracker
    vhx_get : Callable[[int, int, int, int], float]
        Energy accessor for VHX.
    seq : str
        RNA sequence.
    pk_energies : Any
        Energy parameter tables.
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates of the subproblem.
    tilde_p_hole : float
        Hole P~ penalty.
    internal_pk_penalty : float
        Internal pseudoknot penalty (Gwi).
    op_left, op_right, op_left_right : EddyRivasBacktrackOp
        Backpointer operations for the three cases.
    """
    # Left Right
    energy = vhx_get(i_idx, j_idx, k_idx - 1, l_idx + 1)
    if math.isfinite(energy):
        left_hole_energy = dangle_hole_left(seq, k_idx, pk_energies)
        right_hole_energy = dangle_hole_right(seq, l_idx, pk_energies)
        tracker.update_if_better(
            left_hole_energy + right_hole_energy + tilde_p_hole + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # Right
    energy = vhx_get(i_idx, j_idx, k_idx - 1, l_idx)
    if math.isfinite(energy):
        right_hole_energy = dangle_hole_right(seq, l_idx - 1, pk_energies)
        tracker.update_if_better(
            right_hole_energy + tilde_p_hole + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # Left
    energy = vhx_get(i_idx, j_idx, k_idx, l_idx + 1)
    if math.isfinite(energy):
        left_hole_energy = dangle_hole_left(seq, k_idx + 1, pk_energies)
        tracker.update_if_better(
            left_hole_energy + tilde_p_hole + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


def update_tracker_for_outer_dangles_using_vhx(
    tracker: BestCandidateTracker,
    vhx_get: Callable[[int, int, int, int], float],
    seq: str,
    pk_energies,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    tilde_p_out: float,
    internal_pk_penalty: float,
    op_left: EddyRivasBacktrackOp,
    op_right: EddyRivasBacktrackOp,
    op_left_right: EddyRivasBacktrackOp,
) -> None:
    """
    Evaluate outer dangles via VHX transitions and update the tracker.

    Considers left, right, and both outer dangles around `(i,j)`.

    Parameters
    ----------
    tracker : BestCandidateTracker
    vhx_get : Callable[[int, int, int, int], float]
        Energy accessor for VHX.
    seq : str
        RNA sequence.
    pk_energies : Any
        Energy parameter tables.
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates of the subproblem.
    tilde_p_out : float
        Outer P~ penalty.
    internal_pk_penalty : float
        Internal pseudoknot penalty (Gwi).
    op_left, op_right, op_left_right : EddyRivasBacktrackOp
        Backpointer operations for the three cases.
    """
    # -------- Case 1: Dangles on the Left Side of the Outer Pair --------
    energy = vhx_get(i_idx + 1, j_idx, k_idx, l_idx)
    if math.isfinite(energy):
        outer_left_dangle_energy = dangle_outer_left(seq, i_idx, pk_energies)
        tracker.update_if_better(
            outer_left_dangle_energy + tilde_p_out + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # -------- Case 2: Dangles on the Right Side of the Outer Pair --------
    energy = vhx_get(i_idx, j_idx - 1, k_idx, l_idx)
    if math.isfinite(energy):
        outer_right_dangle_energy = dangle_outer_right(seq, j_idx, pk_energies)
        tracker.update_if_better(
            outer_right_dangle_energy + tilde_p_out + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # -------- Case 3: Dangles on Both Sides of the Outer Pair --------
    energy = vhx_get(i_idx + 1, j_idx - 1, k_idx, l_idx)
    if math.isfinite(energy):
        outer_left_dangle_energy = dangle_outer_left(seq, i_idx, pk_energies)
        outer_right_dangle_energy = dangle_outer_right(seq, j_idx, pk_energies)
        tracker.update_if_better(
            outer_left_dangle_energy
            + outer_right_dangle_energy
            + tilde_p_out
            + energy
            + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


# =====================================================================
# VHX Internal Dangles (No Sequence Lookup)
# =====================================================================
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
    op_left: EddyRivasBacktrackOp,
    op_right: EddyRivasBacktrackOp,
    op_left_right: EddyRivasBacktrackOp,
) -> None:
    """
    Evaluate VHX-internal dangles (left, right, both) without sequence lookup.

    Parameters
    ----------
    tracker : BestCandidateTracker
    vhx_get : Callable[[int, int, int, int], float]
        Energy accessor for VHX.
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates of the sub-problem.
    tilde_p_hole : float
        Hole P~ penalty.
    tilde_l : float
        Fallback left dangle energy (L~).
    tilde_r : float
        Fallback right dangle energy (R~).
    op_left, op_right, op_left_right : EddyRivasBacktrackOp
        Backpointer operations for the three cases.
    """
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
        EddyRivasBackPointer(op=op_left_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
    )


# =====================================================================
# Multiloop Wrappers (WHX Close/Wrap)
# =====================================================================
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
    op_close_both: EddyRivasBacktrackOp,
    op_wrap_whx: EddyRivasBacktrackOp,
) -> None:
    """
    Evaluate multiloop close and wrap transitions from VHX to WHX.

    Parameters
    ----------
    tracker : BestCandidateTracker
    whx_collapse_get : Callable[[int, int, int, int], float]
        Energy accessor that also handles WHX collapse cases.
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates of the sub-problem.
    tilde_p_hole : float
        Hole P~ penalty (counted per added unpaired base).
    tilde_m_vhx : float
        VHX multiloop penalty (M~).
    tilde_m_whx : float
        WHX multiloop penalty (M~) added in combination.
    internal_pk_penalty : float
        Internal pseudoknot penalty (Gwi).
    op_close_both, op_wrap_whx : EddyRivasBacktrackOp
        Backpointer ops for close-both and wrap paths.
    """
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


def update_tracker_for_yhx_multiloop_wrap_whx(
    tracker: BestCandidateTracker,
    whx_get: Callable[[int, int, int, int], float],
    seq: str,
    pk_energies,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    tilde_p_out: float,
    tilde_m_yhx: float,
    tilde_m_whx: float,
    internal_pk_penalty: float,
    op_plain: EddyRivasBacktrackOp,
    op_left: EddyRivasBacktrackOp,
    op_right: EddyRivasBacktrackOp,
    op_left_right: EddyRivasBacktrackOp,
) -> None:
    """
    Evaluate YHX multiloop wrap cases around a WHX sub-problem.

    Parameters
    ----------
    tracker : BestCandidateTracker
    whx_get : Callable[[int, int, int, int], float]
        Energy accessor for WHX.
    seq : str
        RNA sequence.
    pk_energies : Any
        Energy parameter tables.
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates of the sub-problem.
    tilde_p_out : float
        Outer P~ penalty.
    tilde_m_yhx : float
        YHX multiloop penalty (M~).
    tilde_m_whx : float
        WHX multiloop penalty (M~).
    internal_pk_penalty : float
        Internal pseudoknot penalty (Gwi).
    op_plain, op_left, op_right, op_left_right : EddyRivasBacktrackOp
        Backpointer operations for plain/left/right/both cases.
    """
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
        outer_left_dangle_energy = dangle_outer_left(seq, i_idx, pk_energies)
        tracker.update_if_better(
            outer_left_dangle_energy + tilde_p_out + tilde_m_yhx + tilde_m_whx + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_left, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # 3. Wrap on 3' (Right) Side
    energy = whx_get(i_idx, j_idx - 1, k_idx - 1, l_idx + 1)
    if math.isfinite(energy):
        outer_right_dangle_energy = dangle_outer_right(seq, j_idx, pk_energies)
        tracker.update_if_better(
            outer_right_dangle_energy + tilde_p_out + tilde_m_yhx + tilde_m_whx + energy + internal_pk_penalty,
            EddyRivasBackPointer(op=op_right, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # 4. Wrap on Both Sides
    energy = whx_get(i_idx + 1, j_idx - 1, k_idx - 1, l_idx + 1)
    if math.isfinite(energy):
        outer_left_dangle_energy = dangle_outer_left(seq, i_idx, pk_energies)
        outer_right_dangle_energy = dangle_outer_right(seq, j_idx, pk_energies)
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


# =====================================================================
# WHX Helpers
# =====================================================================
def update_tracker_for_whx_shrink_hole(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    q_single_strand: float,
) -> None:
    """
    Consider shrinking the hole on the left or right by adding an unpaired base.

    Parameters
    ----------
    tracker : BestCandidateTracker
    state : EddyRivasFoldState
    i_idx, j_idx, k_idx, l_idx : int
       Coordinates of the sub-problem.
    q_single_strand : float
       Single-stranded penalty per added base in the hole.
    """
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


def update_tracker_for_whx_trim_outer(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    q_single_strand: float,
) -> None:
    """
    Consider trimming the outer span on the left or right by adding an unpaired base.

    Parameters
    ----------
    tracker : BestCandidateTracker
    state : EddyRivasFoldState
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates of the sub-problem.
    q_single_strand : float
        Single-stranded penalty per added base on the outer span.
    """
    # 1. Add an Unpaired Base at The 5' End of The Outer Span (Trim Left: (i+1,j))
    energy = state.whx_matrix.get_energy(i_idx + 1, j_idx, k_idx, l_idx)
    if math.isfinite(energy):
        tracker.update_if_better(
            energy + q_single_strand,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_TRIM_LEFT, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )

    # 2. Add An Unpaired Base at The 3' End of The Outer Span (Trim Right: (i,j-1))
    energy = state.whx_matrix.get_energy(i_idx, j_idx - 1, k_idx, l_idx)
    if math.isfinite(energy):
        tracker.update_if_better(
            energy + q_single_strand,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_TRIM_RIGHT, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


def update_tracker_for_whx_collapse_to_nested(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
) -> None:
    """
    Consider collapsing the WHX subproblem to a purely nested structure.

    Parameters
    ----------
    tracker : BestCandidateTracker
    state : EddyRivasFoldState
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates of the subproblem.
    """
    energy = get_whx_energy_with_collapse(state.whx_matrix, state.wxu_matrix, i_idx, j_idx, k_idx, l_idx)
    if math.isfinite(energy):
        tracker.update_if_better(
            energy,
            EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE, outer=(i_idx, j_idx), hole=(k_idx, l_idx)),
        )


def update_tracker_for_whx_ss_both_outer(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    q_single_strand: float,
) -> None:
    """
    Consider adding unpaired bases to both outer ends `(i+1, j-1)`.

    Parameters
    ----------
    tracker : BestCandidateTracker
    state : EddyRivasFoldState
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates of the subproblem.
    q_single_strand : float
        Single-stranded penalty per added base on the outer span.
    """
    energy = state.whx_matrix.get_energy(i_idx + 1, j_idx - 1, k_idx, l_idx)
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
    """
    Consider WHX bifurcations into smaller WHX + nested WX sub-problems.

    Parameters
    ----------
    tracker : BestCandidateTracker
    state : EddyRivasFoldState
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates of the subproblem.
    """
    span_len = j_idx - i_idx
    if span_len <= 0:
        return

    # 1. Split Left into WHX(i,r:k,l) + WX(r+1,j)
    cand, t_star = find_best_split_sum(
        span_len,
        left_fetch=lambda t: state.whx_matrix.get_energy(i_idx, i_idx + t, k_idx, l_idx),
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
    cand, t_star = find_best_split_sum(
        span_len,
        left_fetch=lambda t: get_wxi_or_wx(state, i_idx, i_idx + t),
        right_fetch=lambda t: state.whx_matrix.get_energy(i_idx + t + 1, j_idx, k_idx, l_idx),
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
    """
    Consider WHX overlap split into two WHX subproblems sharing `(k,l)`.

    Parameters
    ----------
    tracker : BestCandidateTracker
    state : EddyRivasFoldState
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates of the subproblem.
    overlap_penalty : float
        Extra energy term applied to overlap compositions.
    """
    span_len = j_idx - i_idx
    if span_len <= 0 or overlap_penalty == 0.0:
        return

    cand, t_star = find_best_split_sum(
        span_len,
        left_fetch=lambda t: state.whx_matrix.get_energy(i_idx, i_idx + t, k_idx, l_idx),
        right_fetch=lambda t: state.whx_matrix.get_energy(i_idx + t + 1, j_idx, k_idx, l_idx),
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


def update_tracker_for_whx_is2_bridge(
    tracker: "BestCandidateTracker",
    state: "EddyRivasFoldState",
    config,
    seq: str,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
) -> None:
    """
    Evaluate IS2 (YHX inner + YHX outer-bridge) for a WHX frame.

    Parameters
    ----------
    tracker : BestCandidateTracker
    state : EddyRivasFoldState
    config : Any
        Folding configuration (tables and parameters).
    seq : str
        RNA sequence.
    i_idx, j_idx, k_idx, l_idx : int
        Coordinates of the WHX sub-problem.
    """
    is2_best, is2_bridge_bp, _ = scan_is2_best_outer_bridge(
        state,
        config,
        seq,
        i_idx,
        j_idx,
        k_idx,
        l_idx,
        inner_matrix_name="yhx",  # WHX uses YHX as inner
        bridge_energy_kind="yhx",   # IS2_outer_yhx
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
