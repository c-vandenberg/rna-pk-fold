import numpy as np
import numba as nb

# Define a float64 representation of infinity for use within Numba-jitted functions.
INF_FLOAT64 = np.float64(np.inf)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@nb.njit(cache=True, fastmath=True)
def _min_of_four_with_index(val_a: float, val_b: float, val_c: float, val_d: float):
    """
    Compute the minimum of four values and return it with the winning index.

    Parameters
    ----------
    val_a : float
        First candidate value.
    val_b : float
        Second candidate value.
    val_c : float
        Third candidate value.
    val_d : float
        Fourth candidate value.

    Returns
    -------
    Tuple[float, int]
        A tuple `(minimum_value, argmin_index)` where `argmin_index` is in `{0, 1, 2, 3}`.
    """
    # Initialize the minimum to the first value.
    minimum = val_a
    which = 0

    # Sequentially compare and update the minimum and its index.
    if val_b < minimum:
        minimum = val_b
        which = 1
    if val_c < minimum:
        minimum = val_c
        which = 2
    if val_d < minimum:
        minimum = val_d
        which = 3

    return minimum, which


@nb.njit(cache=True, fastmath=True)
def min_sum_over_index(left: np.ndarray, right: np.ndarray):
    """
    Calculates `min(left + right)` over a shared index `t` and return the value and index.

    This is a simple, optimized kernel to find the minimum sum of corresponding
    elements from two arrays and the index at which that minimum occurs.

    Parameters
    ----------
    left : np.ndarray
        First vector of energies.
    right : np.ndarray
        Second vector of energies (same length as `left`).

    Returns
    -------
    Tuple[float, int]
        A tuple containing the minimum sum and the index `t` where it was found.
    """
    num_elements = left.shape[0]
    best = INF_FLOAT64
    best_idx = -1

    # Linearly scan through the arrays to find the minimum sum.
    for t in range(num_elements):
        current_sum = left[t] + right[t]
        if current_sum < best:
            best = current_sum
            best_idx = t

    return best, best_idx


@nb.njit(cache=True, fastmath=True)
def min_sum_with_penalty_over_index(left: np.ndarray, right: np.ndarray, penalty: float):
    """
    Calculates `min(left + right + penalty)` over a shared index `t` and return the value and index.

    This kernel is identical to `best_sum` but includes an additional fixed
    penalty term in the sum.

    Parameters
    ----------
    left : np.ndarray
        First vector of energies.
    right : np.ndarray
        Second vector of energies (same length as `left`).
    penalty : float
        Constant additive penalty applied to each sum.

    Returns
    -------
    Tuple[float, int]
        A tuple containing the minimum sum (including penalty) and the index `t`.
    """
    num_elements = left.shape[0]
    best = INF_FLOAT64
    best_idx = -1

    # Linearly scan through the arrays to find the minimum sum.
    for t in range(num_elements):
        current_sum = left[t] + right[t] + penalty
        if current_sum < best:
            best = current_sum
            best_idx = t

    return best, best_idx


# -------------------------
# WX Composition Kernel
# -------------------------
@nb.njit(cache=True, fastmath=True)
def compose_wx_min_energy_over_splits(
    left_whx_uncharged: np.ndarray,
    right_whx_uncharged: np.ndarray,
    left_whx_charged: np.ndarray,
    right_whx_charged: np.ndarray,
    left_yhx_energy: np.ndarray,
    right_yhx_energy: np.ndarray,
    left_yhx_is_charged,
    right_yhx_is_charged,
    gw_penalty: float,
    loop_cap_penalty: float
):
    """
    Choose the optimal split index for WX composition using precomputed arrays.

    This Numba-jitted kernel evaluates, for each candidate split `t`, all valid
    combinations among `WHX` (uncharged/charged) and `YHX` sub-problems and
    returns the minimum free energy, its split index, and an identifying case id.

    Parameters
    ----------
    left_whx_uncharged : np.ndarray
        Energies for the left uncharged `WHX` sub-problem.
    right_whx_uncharged : np.ndarray
        Energies for the right uncharged `WHX` sub-problem.
    left_whx_charged : np.ndarray
        Energies for the left charged `WHX` sub-problem.
    right_whx_charged : np.ndarray
        Energies for the right charged `WHX` sub-problem.
    left_yhx_energy : np.ndarray
        Energies for the left `YHX` sub-problem (may contain `inf` for invalid).
    right_yhx_energy : np.ndarray
        Energies for the right `YHX` sub-problem (may contain `inf` for invalid).
    left_yhx_is_charged : np.ndarray
        `uint8` flags (0/1) indicating whether the left `YHX` is charged.
    right_yhx_is_charged : np.ndarray
        `uint8` flags (0/1) indicating whether the right `YHX` is charged.
    gw_penalty : float
        Pseudoknot introduction penalty `Gw`.
    loop_cap_penalty : float
        Penalty applied for very short loops between helices.

    Returns
    -------
    Tuple[float, int, int]
        `(best_energy, best_split_index, case_id)`, where `case_id` is:
        - 0: `WHX(u) + WHX(u)`
        - 1: `WHX(c) + WHX(u)`
        - 2: `WHX(u) + WHX(c)`
        - 3: `WHX(c) + WHX(c)` (includes `Gw`)
        - 4: `YHX + YHX` (includes `Gw` if both charged)
        - 5: `YHX + WHX(u)`
        - 6: `YHX + WHX(c)` (includes `Gw` if left `YHX` charged)
        - 7: `WHX(u) + YHX`
        - 8: `WHX(c) + YHX` (includes `Gw` if right `YHX` charged)
    """
    # Get the number of possible split points.
    num_splits = left_whx_uncharged.shape[0]
    # Initialize the best energy found so far to infinity.
    best_energy = INF_FLOAT64
    best_idx = -1
    best_case_id = -1

    # Iterate through each possible split point 't' (where r = k + t).
    for t in range(num_splits):
        # --- Case 1: Compositions involving WHX sub-problems ---
        # Calculate the four energy combinations for WHX(left) + WHX(right),
        # considering both charged (c) and uncharged (u) sub-problems.
        cand_uu = left_whx_uncharged[t] + right_whx_uncharged[t] + loop_cap_penalty  # uncharged + uncharged
        cand_cu = left_whx_charged[t] + right_whx_uncharged[t] + loop_cap_penalty  # charged + uncharged
        cand_uc = left_whx_uncharged[t] + right_whx_charged[t] + loop_cap_penalty  # uncharged + charged
        cand_cc = left_whx_charged[t] + right_whx_charged[t] + loop_cap_penalty + gw_penalty  # charged + charged

        # Find the minimum among these four WHX combinations.
        minimum, which = _min_of_four_with_index(cand_uu, cand_cu, cand_uc, cand_cc)
        # If this is the best energy found so far, update the result.
        if minimum < best_energy:
            best_energy = minimum
            best_idx = t
            best_case_id = which

        # --- Case 2: Compositions involving YHX sub-problems (YHX + YHX) ---
        if np.isfinite(left_yhx_energy[t]) and np.isfinite(right_yhx_energy[t]):
            cand = left_yhx_energy[t] + right_yhx_energy[t] + loop_cap_penalty
            if left_yhx_is_charged[t] and right_yhx_is_charged[t]:
                cand += gw_penalty
            if cand < best_energy:
                best_energy = cand
                best_idx = t
                best_case_id = 4

        # --- Case 3 & 4: YHX(left) + WHX(right) ---
        if np.isfinite(left_yhx_energy[t]):
            # YHX(left) + WHX(right, uncharged)
            cand = left_yhx_energy[t] + right_whx_uncharged[t] + loop_cap_penalty
            if cand < best_energy:
                best_energy = cand
                best_idx = t
                best_case_id = 5

            # YHX(left) + WHX(right, charged)
            cand = left_yhx_energy[t] + right_whx_charged[t] + loop_cap_penalty
            if left_yhx_is_charged[t]:
                cand += gw_penalty
            if cand < best_energy:
                best_energy = cand
                best_idx = t
                best_case_id = 6

        # --- Case 5 & 6: WHX(left) + YHX(right) ---
        if np.isfinite(right_yhx_energy[t]):
            # WHX(left, uncharged) + YHX(right)
            cand = right_yhx_energy[t] + left_whx_uncharged[t] + loop_cap_penalty
            if cand < best_energy:
                best_energy = cand
                best_idx = t
                best_case_id = 7
            # WHX(left, charged) + YHX(right)
            cand = left_whx_charged[t] + right_yhx_energy[t] + loop_cap_penalty
            if right_yhx_is_charged[t]:
                cand += gw_penalty
            if cand < best_energy:
                best_energy = cand
                best_idx = t
                best_case_id = 8

    # Return the overall best energy, its index, and the case that produced it.
    return best_energy, best_idx, best_case_id


# -------------------------
# VX Composition Kernel
# -------------------------
@nb.njit(cache=True, fastmath=True)
def compose_vx_min_energy_over_splits(
    left_zhx_uncharged: np.ndarray,
    right_zhx_uncharged: np.ndarray,
    left_zhx_charged: np.ndarray,
    right_zhx_charged: np.ndarray,
    coax_total: np.ndarray,
    coax_bonus: np.ndarray,
    gw_penalty: float,
    loop_cap_penalty: float,
    coaxial_scale: float
):
    """
    Choose the optimal split index for VX composition using precomputed arrays.

    Similar to the WX kernel but for `VHX`/`ZHX` inside a closing pair, this
    kernel evaluates each split `t` over the four `ZHX` base combinations and
    adds coaxial stacking contributions.

    Parameters
    ----------
    left_zhx_uncharged : np.ndarray
        Energies for the left uncharged `ZHX` sub-problem.
    right_zhx_uncharged : np.ndarray
        Energies for the right uncharged `ZHX` sub-problem.
    left_zhx_charged : np.ndarray
        Energies for the left charged `ZHX` sub-problem.
    right_zhx_charged : np.ndarray
        Energies for the right charged `ZHX` sub-problem.
    coax_total : np.ndarray
        Total coaxial stacking energy per split.
    coax_bonus : np.ndarray
        Additional coaxial bonus per split (additive term).
    gw_penalty : float
        Pseudoknot introduction penalty `Gw`.
    loop_cap_penalty : float
        Penalty applied for very short loops between helices.
    coaxial_scale : float
        Scale factor applied to `coax_total`.

    Returns
    -------
    Tuple[float, int, int]
        `(best_energy, best_split_index, base_case_id)` where `base_case_id` is:
        - 0: `ZHX(u) + ZHX(u)`
        - 1: `ZHX(c) + ZHX(u)`
        - 2: `ZHX(u) + ZHX(c)`
        - 3: `ZHX(c) + ZHX(c)` (includes `Gw`)
    """
    # Get the number of possible split points.
    num_splits = left_zhx_uncharged.shape[0]

    # Initialize the best energy found so far to infinity.
    best_energy = INF_FLOAT64
    best_idx = -1
    best_case_id = -1

    # Iterate through each possible split point 't' (where r = k + t).
    for t in range(num_splits):
        # Calculate the four energy combinations for ZHX(left) + ZHX(right),
        # considering both charged (c) and uncharged (u) sub-problems.
        cand_uu = left_zhx_uncharged[t] + right_zhx_uncharged[t] + loop_cap_penalty
        cand_cu = left_zhx_charged[t] + right_zhx_uncharged[t] + loop_cap_penalty
        cand_uc = left_zhx_uncharged[t] + right_zhx_charged[t] + loop_cap_penalty
        cand_cc = left_zhx_charged[t] + right_zhx_charged[t] + loop_cap_penalty + gw_penalty

        # Find the minimum energy from the base ZHX combinations.
        base_energy, which = _min_of_four_with_index(cand_uu, cand_cu, cand_uc, cand_cc)

        # Calculate the final candidate energy by adding the scaled coaxial stacking energies.
        total_energy = base_energy + coaxial_scale * coax_total[t] + coax_bonus[t]

        # If this is the best total energy found so far, update the result.
        if total_energy < best_energy:
            best_energy = total_energy
            best_idx = t
            best_case_id = which

    # Return the overall best energy, its index, and the base case that produced it.
    return best_energy, best_idx, best_case_id
