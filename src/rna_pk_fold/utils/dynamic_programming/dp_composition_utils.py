import math
import numpy as np
from typing import Optional, Tuple

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer, EddyRivasBacktrackOp
from rna_pk_fold.energies.energy_pk_ops import short_hole_penalty, coax_pack
from rna_pk_fold.folding.eddy_rivas.numba_kernels import compose_wx_min_energy_over_splits, compose_vx_min_energy_over_splits
from rna_pk_fold.utils.dynamic_programming.matrix_utils import (whx_collapse_with, zhx_collapse_with,
                                                                get_yhx_energy_with_collapse)
from rna_pk_fold.utils.sequences.iter_utils import iter_inner_holes


# ---------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------
def set_span_cell_with_backpointer(
        matrix, backpointer_store, i_idx: int, j_idx: int, value: float, backpointer
) -> None:
    """
    Set a span cell and its backpointer if provided.

    Parameters
    ----------
    matrix : Any
        Matrix object exposing `set(i, j, value)`.
    backpointer_store : Any
        Backpointer matrix exposing `set(i, j, bp)`.
    i_idx : int
        5' index of the span.
    j_idx : int
        3' index of the span.
    value : float
        Energy value to write at `(i_idx, j_idx)`.
    backpointer : Optional[EddyRivasBackPointer]
        Backpointer to store at `(i_idx, j_idx)` if not `None`.

    Returns
    -------
    None
    """
    matrix.set_energy(i_idx, j_idx, value)
    if backpointer is not None:
        backpointer_store.set_backpointer(i_idx, j_idx, backpointer)


def hole_width(k_idx: int, l_idx: int) -> int:
    """
    Compute the interior width of a hole `[k_idx..l_idx]`.

    Parameters
    ----------
    k_idx : int
        5' index of the hole.
    l_idx : int
        3' index of the hole.

    Returns
    -------
    int
        Number of internal nucleotides, i.e., `l_idx - k_idx - 1`.
    """
    return l_idx - k_idx - 1


def hole_has_internal_bases(hole_span: Tuple[int, int]) -> bool:
    """
    Check whether a hole contains at least one internal nucleotide.

    Parameters
    ----------
    hole_span : tuple of int
        Pair `(k_idx, l_idx)` describing the hole.

    Returns
    -------
    bool
        `True` if `l_idx - k_idx > 1`.
    """
    k_idx, l_idx = hole_span
    return (l_idx - k_idx) > 1


# ---------------------------------------------------------------------
# WX Composition: Array Preparation (One Pass Over Split Point `r`)
# ---------------------------------------------------------------------
def build_wx_split_arrays(
    fold_state,
    config,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    can_pair_mask,
):
    """
    Build WHX/YHX per-split arrays for WX composition.

    For each split point `r` in `[k_idx, l_idx-1]`, this prepares vectors
    of energies used by the WX kernel, leaving invalid points as `+inf` and
    charge flags as `0`.

    Parameters
    ----------
    fold_state : Any
        Eddy-Rivas fold state providing WHX/YHX matrices and backpointers.
    config : Any
        Folding configuration with geometry and feature flags.
    i_idx, j_idx : int
        Outer span indices.
    k_idx, l_idx : int
        Hole (inner) indices.
    can_pair_mask : array-like of bool or None
        Pairability mask; when provided, disallows unpairable pairs.

    Returns
    -------
    tuple of numpy.ndarray
        `(whx_left_uncharged, whx_right_uncharged, whx_left_charged,
        whx_right_charged, yhx_left_energy, yhx_right_energy,
        yhx_left_is_charged, yhx_right_is_charged)`, each of length
        `l_idx - k_idx`. Energies are `float64`; flags are `uint8`.
    """
    # --- Pre-computation Step for Numba Kernel ---
    # Create vectors to store energies for every possible split point 'r' between k and l.
    # This vectorization allows for efficient processing by a Numba kernel.
    split_count = l_idx - k_idx

    # WHX (energies + charged flags)
    whx_left_uncharged = np.full(split_count, np.inf, dtype=np.float64)  # (nested sub-problem)
    whx_right_uncharged = np.full(split_count, np.inf, dtype=np.float64)
    whx_left_charged = np.full(split_count, np.inf, dtype=np.float64)    # (pseudoknotted sub-problem)
    whx_right_charged = np.full(split_count, np.inf, dtype=np.float64)

    # YHX alternative (energies + charged flags)
    yhx_left_energy = np.full(split_count, np.inf, dtype=np.float64)
    yhx_right_energy = np.full(split_count, np.inf, dtype=np.float64)
    yhx_left_is_charged = np.zeros(split_count, dtype=np.uint8)
    yhx_right_is_charged = np.zeros(split_count, dtype=np.uint8)

    # Iterate through all possible split points 'r' to populate the energy vectors.
    for split_offset in range(split_count):
        split_idx = k_idx + split_offset

        # Enforce the strict Rivas & Eddy ordering for pseudoknot helices.
        if config.enable_strict_compliment_order and not (i_idx < k_idx <= split_idx < l_idx <= j_idx):
            continue

        # Enforce minimum lengths for the 5' and 3' outer segments.
        if ((split_idx - i_idx) < config.pk_energies.min_outer_left or
                (j_idx - (split_idx + 1)) < config.pk_energies.min_outer_right):
            continue

        # Calculate energies for the left and right gapped subproblems.
        # 'whx_collapse_with' gets the energy, handling cases where the subproblem's hole is empty.
        # 'charged=False' (u) gets the nested baseline; 'charged=True' (c) gets the pseudoknotted energy.
        whx_left_uncharged[split_offset] = whx_collapse_with(
            fold_state, i_idx, split_idx, k_idx, split_idx, charged=False, can_pair_mask=can_pair_mask
        )
        whx_right_uncharged[split_offset] = whx_collapse_with(
            fold_state, split_idx + 1, j_idx, split_idx + 1, l_idx, charged=False, can_pair_mask=can_pair_mask
        )
        whx_left_charged[split_offset] = whx_collapse_with(
            fold_state, i_idx, split_idx, k_idx, split_idx, charged=True, can_pair_mask=can_pair_mask
        )
        whx_right_charged[split_offset] = whx_collapse_with(
            fold_state, split_idx + 1, j_idx, split_idx + 1, l_idx, charged=True, can_pair_mask=can_pair_mask
        )

        # Get energies from the YHX matrix (only if the pair is allowed)
        # Left YHX
        if can_pair_mask is not None and can_pair_mask[k_idx][split_idx]:
            ly = get_yhx_energy_with_collapse(fold_state.yhx_matrix, i_idx, split_idx, k_idx, split_idx)
            if math.isfinite(ly):
                yhx_left_energy[split_offset] = ly
                bp_ly = fold_state.yhx_back_ptr.get_backpointer(i_idx, split_idx, k_idx, split_idx)
                if bp_ly is not None and getattr(bp_ly, "charged", False):
                    yhx_left_is_charged[split_offset] = 1

        # Right YHX
        if can_pair_mask is not None and can_pair_mask[split_idx + 1][l_idx]:
            ry = get_yhx_energy_with_collapse(fold_state.yhx_matrix, split_idx + 1, j_idx, split_idx + 1, l_idx)
            if math.isfinite(ry):
                yhx_right_energy[split_offset] = ry
                bp_ry = fold_state.yhx_back_ptr.get_backpointer(split_idx + 1, j_idx, split_idx + 1, l_idx)
                if bp_ry is not None and getattr(bp_ry, "charged", False):
                    yhx_right_is_charged[split_offset] = 1

    return (
        whx_left_uncharged,
        whx_right_uncharged,
        whx_left_charged,
        whx_right_charged,
        yhx_left_energy,
        yhx_right_energy,
        yhx_left_is_charged,
        yhx_right_is_charged,
    )


# ---------------------------------------------------------------------
# WX composition: Decode/Validate Kernel `case_id`
# ---------------------------------------------------------------------
def _wx_case_has_gapped_structure(
    kernel_case_id: int,
    split_offset_star: int,
    whx_left_uncharged,
    whx_right_uncharged,
    whx_left_charged,
    whx_right_charged,
    yhx_left_energy,
    yhx_right_energy,
) -> Tuple[bool, bool]:
    """
    Validate that the chosen kernel case uses true gapped structures.

    Ensures both left and right fragments correspond to finite gapped subproblems
    (rather than collapsed/nested-only states) for the winning split.

    Parameters
    ----------
    kernel_case_id : int
        Case identifier returned by the WX composition kernel.
    split_offset_star : int
        Argmin index in the split arrays.
    whx_left_uncharged, whx_right_uncharged, whx_left_charged, whx_right_charged : numpy.ndarray
        WHX energy vectors per split (length `l_idx - k_idx`).
    yhx_left_energy, yhx_right_energy : numpy.ndarray
        YHX energy vectors per split (length `l_idx - k_idx`).

    Returns
    -------
    tuple of bool
        `(left_has_structure, right_has_structure)`.
    """
    if split_offset_star < 0:
        return False, False

    # Determine which vectors contributed based on case_id
    # This is a critical filter: ensure that both the left and right sub-fragments
    # have a defined gapped structure. This prevents selecting combinations where
    # one side has simply collapsed to a nested structure, which wouldn't form a true pseudoknot.
    # Which vectors contributed is determined based on case_id
    if kernel_case_id == 0:  # Lu + Ru (uncharged + uncharged)
        return np.isfinite(whx_left_uncharged[split_offset_star]), np.isfinite(whx_right_uncharged[split_offset_star])
    if kernel_case_id == 1:  # Lu + Rc (uncharged + charged)
        return np.isfinite(whx_left_uncharged[split_offset_star]), np.isfinite(whx_right_charged[split_offset_star])
    if kernel_case_id == 2:  # Lc + Ru (charged + uncharged)
        return np.isfinite(whx_left_charged[split_offset_star]), np.isfinite(whx_right_uncharged[split_offset_star])
    if kernel_case_id == 3:  # Lc + Rc (charged + charged)
        return np.isfinite(whx_left_charged[split_offset_star]), np.isfinite(whx_right_charged[split_offset_star])
    if kernel_case_id == 4:  # YHX + YHX
        return np.isfinite(yhx_left_energy[split_offset_star]), np.isfinite(yhx_right_energy[split_offset_star])
    if kernel_case_id == 5:  # YHX + WHX(u)
        return np.isfinite(yhx_left_energy[split_offset_star]), np.isfinite(whx_right_uncharged[split_offset_star])
    if kernel_case_id == 6:  # YHX + WHX(c)
        return np.isfinite(yhx_left_energy[split_offset_star]), np.isfinite(whx_right_charged[split_offset_star])
    if kernel_case_id == 7:  # WHX(u) + YHX
        return np.isfinite(whx_left_uncharged[split_offset_star]), np.isfinite(yhx_right_energy[split_offset_star])
    if kernel_case_id == 8:  # WHX(c) + YHX
        return np.isfinite(whx_left_charged[split_offset_star]), np.isfinite(yhx_right_energy[split_offset_star])

    return False, False


def decode_wx_case_to_backpointer(
    kernel_case_id: int,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    split_idx_star: int,
    yhx_left_is_charged,
    yhx_right_is_charged,
    split_offset_star: int,
) -> Tuple["EddyRivasBacktrackOp", Tuple[int, int], Tuple[int, int], bool]:
    """
    Map a WX kernel case to a backtrack op and metadata.

    Parameters
    ----------
    kernel_case_id : int
        Case identifier returned by the WX composition kernel.
    i_idx, j_idx : int
        Outer span indices.
    k_idx, l_idx : int
        Hole indices.
    split_idx_star : int
        Winning split index `r` in absolute coordinates.
    yhx_left_is_charged, yhx_right_is_charged : numpy.ndarray
        YHX charge flags (`uint8`) per split.
    split_offset_star : int
        Winning split index offset in `[0, l_idx-k_idx)`.

    Returns
    -------
    tuple
        `(op, hole_left, hole_right, charged)` where
        `op` is an `EddyRivasBacktrackOp`,
        `hole_left` and `hole_right` are `(k, l)` tuples describing
        the two sub-holes, and `charged` indicates whether a PK charge applies.
    """
    hole_left = (k_idx, split_idx_star)
    hole_right = (split_idx_star + 1, l_idx)

    if kernel_case_id in (0, 1, 2, 3):
        # Case 0, 1 & 2: WHX + WHX (uu/cu/uc): no Gw, not charged
        # Case 3: WHX + WHX (cc): Gw applied in kernel; charged
        charged = kernel_case_id == 3
        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX
        return op, hole_left, hole_right, charged

    if kernel_case_id == 4:
        # Case 4: YHX + YHX: Gw added only if both charged (decided in kernel);
        charged = bool(yhx_left_is_charged[split_offset_star] and yhx_right_is_charged[split_offset_star])
        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX
        return op, hole_left, hole_right, charged

    if kernel_case_id in (5, 6):
        # Case 5: YHX + WHX (u): no Gw, not charged
        # Case 6: YHX + WHX (c): Gw only if left Y is charged
        charged = (kernel_case_id == 6) and bool(yhx_left_is_charged[split_offset_star])
        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX
        return op, hole_left, hole_right, charged

    if kernel_case_id in (7, 8):
        # Case 7: WHX (u) + YHX: no Gw, not charged
        # Case 8: WHX (c) + YHX: Gw only if right Y is charged
        charged = (kernel_case_id == 8) and bool(yhx_right_is_charged[split_offset_star])
        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX
        return op, hole_left, hole_right, charged

    # Defensive default
    return EddyRivasBacktrackOp.RE_PK_COMPOSE_WX, hole_left, hole_right, False


# ---------------------------------------------------------------------
# WX composition: One-Hole Composition Wrapper
# ---------------------------------------------------------------------
def evaluate_wx_composition_for_hole(
    fold_state,
    config,
    seq: str,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    pseudoknot_penalty: float,
    can_pair_mask,
):
    """
    Evaluate WX composition for a single hole inside an outer span.

    Builds split arrays, runs the WX composition kernel, validates the chosen
    case, and (if valid) returns the energy and a fully-annotated backpointer.

    Parameters
    ----------
    fold_state : Any
        Eddy-Rivas fold state with WHX/YHX matrices and backpointers.
    config : Any
        Folding configuration with geometry/penalty parameters.
    seq : str
        RNA sequence (not used directly here; present for API symmetry).
    i_idx, j_idx : int
        Outer span indices.
    k_idx, l_idx : int
        Hole indices.
    pseudoknot_penalty : float
        Penalty for initiating a pseudoknot (Gw).
    can_pair_mask : array-like of bool or None
        Pairability mask; when provided, disallows unpairable pairs.

    Returns
    -------
    tuple
        `(candidate_energy, backpointer_or_none)` where energy is `float`
        and backpointer is an `EddyRivasBackPointer` or `None` when
        the case is invalid.
    """
    # Build arrays across split point `r`
    (
        whx_left_uncharged,
        whx_right_uncharged,
        whx_left_charged,
        whx_right_charged,
        yhx_left_energy,
        yhx_right_energy,
        yhx_left_is_charged,
        yhx_right_is_charged,
    ) = build_wx_split_arrays(fold_state, config, i_idx, j_idx, k_idx, l_idx, can_pair_mask)

    # Calculate penalty for very short loops between helices.
    loop_cap_penalty = short_hole_penalty(config.pk_energies, k_idx, l_idx)

    # --- Kernel Execution ---
    # Pass the energy vectors to the optimized Numba kernel to find the best split point 'r'
    # and the best combination of subproblems (WHX+WHX, YHX+YHX, etc.).
    candidate_energy, split_offset_star, kernel_case_id = compose_wx_min_energy_over_splits(
        whx_left_uncharged,
        whx_right_uncharged,
        whx_left_charged,
        whx_right_charged,
        yhx_left_energy,
        yhx_right_energy,
        yhx_left_is_charged,
        yhx_right_is_charged,
        float(pseudoknot_penalty),
        float(loop_cap_penalty),
    )

    # Invalid split index -> reject (proceed only if the kernel returned a valid split point).
    if split_offset_star < 0:
        return math.inf, None

    # Validate that both sides have gapped structure (not pure collapse)
    left_ok, right_ok = _wx_case_has_gapped_structure(
        kernel_case_id,
        split_offset_star,
        whx_left_uncharged,
        whx_right_uncharged,
        whx_left_charged,
        whx_right_charged,
        yhx_left_energy,
        yhx_right_energy,
    )
    if not (left_ok and right_ok):
        return math.inf, None

    # If all checks pass, this is a valid, new best pseudoknot candidate.
    split_idx_star = k_idx + split_offset_star # Recalculate best split point.
    hole_left = (k_idx, split_idx_star)
    hole_right = (split_idx_star + 1, l_idx)
    if not (hole_has_internal_bases(hole_left) and hole_has_internal_bases(hole_right)):
        return math.inf, None

    # # Decode the 'case_id' and backpointer fields from the kernel to determine the backtrack
    # operation.
    op, hole_left, hole_right, charged = decode_wx_case_to_backpointer(
        kernel_case_id,
        i_idx,
        j_idx,
        k_idx,
        l_idx,
        split_idx_star,
        yhx_left_is_charged,
        yhx_right_is_charged,
        split_offset_star,
    )

    # Create the backpointer for this optimal pseudoknot configuration.
    backpointer = EddyRivasBackPointer(
        op=op,
        outer=(i_idx, j_idx),
        hole=(k_idx, l_idx),
        hole_left=hole_left,
        hole_right=hole_right,
        split=split_idx_star,
        charged=charged,
    )
    return candidate_energy, backpointer


# ---------------------------------------------------------------------
# WX composition: Optional YHX-Overlap Path
# ---------------------------------------------------------------------
def evaluate_wx_yhx_overlap_for_span(
    fold_state, config, i_idx: int, j_idx: int, wx_overlap_penalty: float
) -> Tuple[float, Optional["EddyRivasBackPointer"]]:
    """
    Evaluate the WX YHX-overlap path for a given outer span.

    Considers motifs where two YHX fragments overlap the same hole and returns
    the best energy/backpointer among all feasible inner holes and splits.

    Parameters
    ----------
    fold_state : Any
        Eddy-Rivas fold state with YHX matrix/backpointers.
    config : Any
        Folding configuration; must enable `enable_wx_overlap` to activate.
    i_idx, j_idx : int
        Outer span indices.
    wx_overlap_penalty : float
        Penalty term applied to YHX-overlap compositions.

    Returns
    -------
    tuple
        `(best_energy, best_backpointer)` where energy is `float` and
        backpointer is an `EddyRivasBackPointer` or `None` when
        no feasible overlap is found.
    """
    if (not config.enable_wx_overlap) or wx_overlap_penalty == 0.0:
        return math.inf, None

    best_energy = math.inf
    best_backpointer = None

    # Iterate through a different set of inner holes and split points.
    for (k_inner_idx, l_inner_idx) in iter_inner_holes(i_idx, j_idx, min_hole_width=config.pk_energies.min_hole_width):
        loop_cap_penalty = short_hole_penalty(config.pk_energies, k_inner_idx, l_inner_idx)
        for split_idx in range(i_idx, j_idx):
            left_y_energy = fold_state.yhx_matrix.get_energy(i_idx, split_idx, k_inner_idx, l_inner_idx)
            right_y_energy = fold_state.yhx_matrix.get_energy(split_idx + 1, j_idx, k_inner_idx, l_inner_idx)

            # If both sub-problems have finite energy, calculate the total energy.
            if math.isfinite(left_y_energy) and math.isfinite(right_y_energy):
                candidate_energy = wx_overlap_penalty + left_y_energy + right_y_energy + loop_cap_penalty
                # If this is a new best energy, update the backpointer.
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_backpointer = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_OVERLAP,
                        outer=(i_idx, j_idx),
                        hole=(k_inner_idx, l_inner_idx),
                        split=split_idx,
                        charged=True,
                    )

    return best_energy, best_backpointer


# ---------------------------------------------------------------------
# VX composition: Array Preparation and Kernel Wrapper
# ---------------------------------------------------------------------
def build_vx_split_arrays_and_coax(
    fold_state,
    config,
    seq: str,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    can_pair_mask
):
    """
    Build ZHX/coaxial per-split arrays for VX composition.

    For each split point `r` in `[k_idx, l_idx-1]`, prepares ZHX energies
    for left/right × (uncharged/charged) and coaxial stacking terms used by
    the VX kernel.

    Parameters
    ----------
    fold_state : Any
        Eddy-Rivas fold state with ZHX matrix.
    config : Any
        Folding configuration with geometry/feature flags and costs.
    seq : str
        RNA sequence (required for coaxial stacking evaluation).
    i_idx, j_idx : int
        Outer span indices (closing pair).
    k_idx, l_idx : int
        Hole indices.
    can_pair_mask : array-like of bool or None
        Pairability mask; when provided, disallows unpairable pairs.

    Returns
    -------
    tuple of numpy.ndarray
        `(zhx_left_uncharged, zhx_right_uncharged, zhx_left_charged,
        zhx_right_charged, coax_total, coax_bonus)`, each of length
        `l_idx - k_idx` with dtype `float64`.
    """
    # --- Pre-computation Step for Numba Kernal ---
    # Create vectors to store energies for every possible split point 'r' between k and l.
    # This vectorization allows for efficient processing by a Numba kernel.
    split_count = l_idx - k_idx

    # ZHX energies (left/right × uncharged/charged)
    zhx_left_uncharged = np.full(split_count, np.inf, dtype=np.float64) # (Nested sub-problem)
    zhx_right_uncharged = np.full(split_count, np.inf, dtype=np.float64)
    zhx_left_charged = np.full(split_count, np.inf, dtype=np.float64) # (Pseudoknotted sub-problem)
    zhx_right_charged = np.full(split_count, np.inf, dtype=np.float64)

    # Coaxial stacking terms
    coax_total = np.zeros(split_count, dtype=np.float64) # Total coaxial stacking energy
    coax_bonus = np.zeros(split_count, dtype=np.float64) # Coaxial stacking bonus energy

    # Iterate through all possible split points 'r' to populate the energy vectors.
    for split_offset in range(split_count):
        split_idx = k_idx + split_offset

        # Enforce the strict Rivas & Eddy ordering for pseudoknot helices.
        if config.enable_strict_compliment_order and not (i_idx < k_idx <= split_idx < l_idx <= j_idx):
            continue

        # Enforce minimum lengths for the 5' and 3' outer segments.
        if ((split_idx - i_idx) < config.pk_energies.min_outer_left or
                (j_idx - (split_idx + 1)) < config.pk_energies.min_outer_right):
            continue

        # Calculate energies for the left and right gapped sub-problems from the ZHX matrix.
        # 'zhx_collapse_with' gets the energy, handling cases where the sub-problem's hole is empty.
        zhx_left_uncharged[split_offset] = zhx_collapse_with(
            fold_state, i_idx, split_idx, k_idx, split_idx, charged=False, can_pair_mask=can_pair_mask
        )
        zhx_right_uncharged[split_offset] = zhx_collapse_with(
            fold_state, split_idx + 1, j_idx, split_idx + 1, l_idx, charged=False, can_pair_mask=can_pair_mask
        )
        zhx_left_charged[split_offset] = zhx_collapse_with(
            fold_state, i_idx, split_idx, k_idx, split_idx, charged=True, can_pair_mask=can_pair_mask
        )
        zhx_right_charged[split_offset] = zhx_collapse_with(
            fold_state, split_idx + 1, j_idx, split_idx + 1, l_idx, charged=True, can_pair_mask=can_pair_mask
        )

        # Calculate the coaxial stacking energy bonus for this specific split point 'r'.
        adjacent = split_idx == k_idx # Check if the helices are adjacent for a flush stack.
        cx_total, cx_bonus = coax_pack(seq, i_idx, j_idx, split_idx, k_idx, l_idx, config, config.pk_energies, adjacent)
        coax_total[split_offset] = cx_total
        coax_bonus[split_offset] = cx_bonus

    return zhx_left_uncharged, zhx_right_uncharged, zhx_left_charged, zhx_right_charged, coax_total, coax_bonus


def evaluate_vx_composition_for_hole(
    fold_state,
    config,
    seq: str,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    l_idx: int,
    pseudoknot_penalty: float,
    coaxial_scale: float,
    can_pair_mask,
):
    """
    Evaluate VX composition for a single hole inside an outer paired span.

    Builds split arrays (ZHX, coax), runs the VX composition kernel, determines
    charge status for the winning case, and returns the energy with a backpointer.

    Parameters
    ----------
    fold_state : Any
        Eddy-Rivas fold state with ZHX matrix.
    config : Any
        Folding configuration and costs.
    seq : str
        RNA sequence for coaxial stacking evaluation.
    i_idx, j_idx : int
        Outer span indices (closing pair).
    k_idx, l_idx : int
        Hole indices.
    pseudoknot_penalty : float
        Penalty for initiating a pseudoknot (Gw).
    coaxial_scale : float
        Scaling factor applied to coaxial stacking contributions.
    can_pair_mask : array-like of bool or None
        Pairability mask; when provided, disallows unpairable pairs.

    Returns
    -------
    tuple
        `(candidate_energy, backpointer_or_none)` where energy is `float`
        and backpointer is an `EddyRivasBackPointer` or `None` when
        the case is invalid.
    """
    # Build arrays across split point `r`
    (
        zhx_left_uncharged,
        zhx_right_uncharged,
        zhx_left_charged,
        zhx_right_charged,
        coax_total,
        coax_bonus,
    ) = build_vx_split_arrays_and_coax(fold_state, config, seq, i_idx, j_idx, k_idx, l_idx, can_pair_mask)

    loop_cap_penalty = short_hole_penalty(config.pk_energies, k_idx, l_idx)

    # --- Kernel Execution ---
    # Pass the energy vectors to the optimized Numba kernel. It efficiently finds the
    # best split point 'r' (returned as t_star) and the minimum energy 'cand'.
    candidate_energy, split_offset_star, kernel_case_id = compose_vx_min_energy_over_splits(
        zhx_left_uncharged,
        zhx_right_uncharged,
        zhx_left_charged,
        zhx_right_charged,
        coax_total,
        coax_bonus,
        float(pseudoknot_penalty),
        float(loop_cap_penalty),
        float(coaxial_scale),
    )

    # Invalid split index -> reject (proceed only if the kernel returned a valid split point).
    if split_offset_star < 0:
        return math.inf, None

    # Decide if this composition truly formed a PK (cc case only).
    # Only consider 'charged' if both charged sides are finite
    is_cc_case = kernel_case_id == 3
    is_cc_finite = np.isfinite(zhx_left_charged[split_offset_star]) and np.isfinite(
        zhx_right_charged[split_offset_star]
    )
    charged = bool(is_cc_case and is_cc_finite)

    split_idx_star = k_idx + split_offset_star
    hole_left = (k_idx, split_idx_star)
    hole_right = (split_idx_star + 1, l_idx)
    if not (hole_has_internal_bases(hole_left) and hole_has_internal_bases(hole_right)):
        return math.inf, None

    backpointer = EddyRivasBackPointer(
        op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX,
        outer=(i_idx, j_idx),
        hole=(k_idx, l_idx),
        split=split_idx_star,
        charged=charged,
    )

    return candidate_energy, backpointer
