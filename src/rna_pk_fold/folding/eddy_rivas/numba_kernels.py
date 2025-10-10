import numpy as np
import numba as nb

# Define a float64 representation of infinity for use within Numba-jitted functions.
INF_FLOAT64 = np.float64(np.inf)

@nb.njit(cache=False, fastmath=True)
def _min4(a: float, b: float, c: float, d: float):
    """
    Finds the minimum of four float values and returns it with an index.
    """
    # Initialize the minimum to the first value.
    m = a; w = 0
    # Sequentially compare and update the minimum and its index.
    if b < m: m = b; w = 1
    if c < m: m = c; w = 2
    if d < m: m = d; w = 3

    return m, w


# -------------------------
# WX Composition Kernel
# -------------------------
@nb.njit(cache=False, fastmath=True)
def compose_wx_best_over_r_arrays(
    l_u: np.ndarray, r_u: np.ndarray, l_c: np.ndarray, r_c: np.ndarray,
    left_y: np.ndarray, right_y: np.ndarray,
    gw_penalty: float, cap_penalty: float
):
    """
    Finds the optimal split point `r` for WX composition via array-based minimization.

    This Numba-jitted kernel is a performance-critical function that executes
    the innermost loop of the O(N^6) WX composition step. It takes pre-computed
    energy vectors for various subproblem types (WHX, YHX) and efficiently finds
    the split point `r` (represented by an index `t`) and the combination of
    subproblems that yield the minimum free energy for a given pseudoknot
    configuration.

    Parameters
    ----------
    l_u : np.ndarray
        Vector of energies for the left, "uncharged" (nested) WHX subproblem.
    r_u : np.ndarray
        Vector of energies for the right, "uncharged" (nested) WHX subproblem.
    l_c : np.ndarray
        Vector of energies for the left, "charged" (pseudoknotted) WHX subproblem.
    r_c : np.ndarray
        Vector of energies for the right, "charged" (pseudoknotted) WHX subproblem.
    left_y : np.ndarray
        Vector of energies for the left YHX subproblem.
    right_y : np.ndarray
        Vector of energies for the right YHX subproblem.
    gw_penalty : float
        The energy penalty for initiating a pseudoknot (Gw).
    cap_penalty : float
        An energy penalty applied for short loop lengths.

    Returns
    -------
    Tuple[float, int, int]
        A tuple containing:
        - The minimum energy found.
        - The index `t` of the optimal split point `r`.
        - A `case_id` (0-8) indicating which combination of subproblems was optimal.
    """
    # Get the number of possible split points.
    num_splits = l_u.shape[0]
    # Initialize the best energy found so far to infinity.
    best = INF_FLOAT64
    best_idx = -1
    best_case = -1

    # Iterate through each possible split point 't' (where r = k + t).
    for t in range(num_splits):
        # --- Case Group 1: Compositions involving WHX subproblems ---
        # Calculate the four energy combinations for WHX(left) + WHX(right),
        # considering both charged (c) and uncharged (u) subproblems.
        cand_uu = gw_penalty + l_u[t] + r_u[t] + cap_penalty  # uncharged + uncharged
        cand_cu = l_c[t] + r_u[t] + cap_penalty  # charged + uncharged
        cand_uc = l_u[t] + r_c[t] + cap_penalty  # uncharged + charged
        cand_cc = l_c[t] + r_c[t] + cap_penalty  # charged + charged

        # Find the minimum among these four WHX combinations.
        m, which = _min4(cand_uu, cand_cu, cand_uc, cand_cc)
        # If this is the best energy found so far, update the result.
        if m < best:
            best = m
            best_idx = t
            best_case = which

        # --- Case Group 2: Compositions involving YHX subproblems ---
        # Case 4: YHX(left) + YHX(right)
        if np.isfinite(left_y[t]) and np.isfinite(right_y[t]):
            cand = gw_penalty + left_y[t] + right_y[t] + cap_penalty
            if cand < best:
                best = cand
                best_idx = t
                best_case = 4

        # Case 5 & 6: YHX(left) + WHX(right)
        if np.isfinite(left_y[t]):
            # YHX(left) + WHX(right, uncharged)
            cand = gw_penalty + left_y[t] + r_u[t] + cap_penalty
            if cand < best:
                best = cand
                best_idx = t
                best_case = 5
            # YHX(left) + WHX(right, charged)
            cand = left_y[t] + r_c[t] + cap_penalty
            if cand < best:
                best = cand
                best_idx = t
                best_case = 6

        # Case 7 & 8: WHX(left) + YHX(right)
        if np.isfinite(right_y[t]):
            # WHX(left, uncharged) + YHX(right)
            cand = gw_penalty + right_y[t] + l_u[t] + cap_penalty
            if cand < best:
                best = cand
                best_idx = t
                best_case = 7
            # WHX(left, charged) + YHX(right)
            cand = l_c[t] + right_y[t] + cap_penalty
            if cand < best:
                best = cand
                best_idx = t
                best_case = 8

    # Return the overall best energy, its index, and the case that produced it.
    return best, best_idx, best_case


# -------------------------
# VX Composition Kernel
# -------------------------
@nb.njit(cache=False, fastmath=True)
def compose_vx_best_over_r(
        l_u: np.ndarray, r_u: np.ndarray, l_c: np.ndarray, r_c: np.ndarray,
        coax_total: np.ndarray, coax_bonus: np.ndarray,
        gw_penalty: float, cap_penalty: float, g: float
):
    """
    Finds the optimal split point `r` for VX composition via array-based minimization.

    This Numba-jitted kernel is similar to the WX composer but is used for
    the VX matrix. It finds the optimal split point `r` by combining `ZHX`
    subproblems and incorporates coaxial stacking energies, which are critical
    for pseudoknot stability within a closing pair.

    Parameters
    ----------
    l_u : np.ndarray
        Vector of energies for the left, "uncharged" (nested) ZHX subproblem.
    r_u : np.ndarray
        Vector of energies for the right, "uncharged" (nested) ZHX subproblem.
    l_c : np.ndarray
        Vector of energies for the left, "charged" (pseudoknotted) ZHX subproblem.
    r_c : np.ndarray
        Vector of energies for the right, "charged" (pseudoknotted) ZHX subproblem.
    coax_total : np.ndarray
        Vector of total coaxial stacking energies for each split point.
    coax_bonus : np.ndarray
        Vector of additional coaxial stacking bonuses for each split point.
    gw_penalty : float
        The energy penalty for initiating a pseudoknot (Gw).
    cap_penalty : float
        An energy penalty applied for short loop lengths.
    g : float
        A scaling factor for coaxial stacking energies in pseudoknots.

    Returns
    -------
    Tuple[float, int, int]
        A tuple containing:
        - The minimum energy found.
        - The index `t` of the optimal split point `r`.
        - A `base_case_id` (0-3) indicating the optimal ZHX combination.
    """
    # Get the number of possible split points.
    num_splits = l_u.shape[0]

    # Initialize the best energy found so far to infinity.
    best = INF_FLOAT64
    best_idx = -1
    best_case = -1

    # Iterate through each possible split point 't' (where r = k + t).
    for t in range(num_splits):
        # Calculate the four energy combinations for ZHX(left) + ZHX(right),
        # considering both charged (c) and uncharged (u) subproblems.
        cand_uu = gw_penalty + l_u[t] + r_u[t] + cap_penalty
        cand_cu = l_c[t] + r_u[t] + cap_penalty
        cand_uc = l_u[t] + r_c[t] + cap_penalty
        cand_cc = l_c[t] + r_c[t] + cap_penalty

        # Find the minimum energy from the base ZHX combinations.
        base, which = _min4(cand_uu, cand_cu, cand_uc, cand_cc)
        # Calculate the final candidate energy by adding the scaled coaxial stacking energies.
        cand = base + g * coax_total[t] + coax_bonus[t]

        # If this is the best total energy found so far, update the result.
        if cand < best:
            best = cand
            best_idx = t
            best_case = which

    # Return the overall best energy, its index, and the base case that produced it.
    return best, best_idx, best_case


@nb.njit(cache=False, fastmath=True)
def best_sum(left: np.ndarray, right: np.ndarray):
    """
    Calculates `min(left + right)` over a shared index `t`.

    This is a simple, optimized kernel to find the minimum sum of corresponding
    elements from two arrays and the index at which that minimum occurs.

    Parameters
    ----------
    left : np.ndarray
        The first array of energies.
    right : np.ndarray
        The second array of energies.

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


@nb.njit(cache=False, fastmath=True)
def best_sum_with_penalty(left: np.ndarray, right: np.ndarray, penalty: float):
    """
    Calculates `min(left + right + penalty)` over a shared index `t`.

    This kernel is identical to `best_sum` but includes an additional fixed
    penalty term in the sum.

    Parameters
    ----------
    left : np.ndarray
        The first array of energies.
    right : np.ndarray
        The second array of energies.
    penalty : float
        A constant penalty to add to each sum.

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
