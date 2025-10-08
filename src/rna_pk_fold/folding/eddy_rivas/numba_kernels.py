import numpy as np
import numba as nb

INF64 = np.float64(np.inf)

@nb.njit(cache=True, fastmath=True)
def _min4(a: float, b: float, c: float, d: float):
    m = a; w = 0
    if b < m: m = b; w = 1
    if c < m: m = c; w = 2
    if d < m: m = d; w = 3
    return m, w


# -------------------------
# WX (array-based variant)
# -------------------------
@nb.njit(cache=True, fastmath=True)
def compose_wx_best_over_r_arrays(
    Lu: np.ndarray, Ru: np.ndarray, Lc: np.ndarray, Rc: np.ndarray,
    left_y: np.ndarray, right_y: np.ndarray,
    Gw: float, cap_pen: float
):
    """
    Array-based minimization over r for WX.

    Inputs are length-L vectors where index t corresponds to r = r0 + t (the
    caller tracks the absolute r). We compute the best of:
      - base: Gw + Lu + Ru + cap, Lc + Ru + cap, Lu + Rc + cap, Lc + Rc + cap
      - yhx+yhx: Gw + left_y + right_y + cap
      - yhx+whx (right): Gw + left_y + Ru + cap,   left_y + Rc + cap
      - whx+yhx (left):  Gw + right_y + Lu + cap,  Lc + right_y + cap

    Any INF components will naturally be ignored.
    Returns (best_value, best_index, which_case_id).
    Cases: 0..3 = base, 4 = yhx+yhx, 5 = yhx+Ru, 6 = yhx+Rc, 7 = Lu+yhx, 8 = Lc+yhx
    """
    L = Lu.shape[0]
    best = INF64
    best_idx = -1
    best_case = -1

    for t in range(L):
        a0 = Gw + Lu[t] + Ru[t] + cap_pen
        a1 =      Lc[t] + Ru[t] + cap_pen
        a2 =      Lu[t] + Rc[t] + cap_pen
        a3 =      Lc[t] + Rc[t] + cap_pen

        m, which = _min4(a0, a1, a2, a3)
        if m < best:
            best = m; best_idx = t; best_case = which

        if np.isfinite(left_y[t]) and np.isfinite(right_y[t]):
            cand = Gw + left_y[t] + right_y[t] + cap_pen
            if cand < best:
                best = cand; best_idx = t; best_case = 4

        if np.isfinite(left_y[t]):
            cand = Gw + left_y[t] + Ru[t] + cap_pen
            if cand < best:
                best = cand; best_idx = t; best_case = 5
            cand =      left_y[t] + Rc[t] + cap_pen
            if cand < best:
                best = cand; best_idx = t; best_case = 6

        if np.isfinite(right_y[t]):
            cand = Gw + right_y[t] + Lu[t] + cap_pen
            if cand < best:
                best = cand; best_idx = t; best_case = 7
            cand =      Lc[t] + right_y[t] + cap_pen
            if cand < best:
                best = cand; best_idx = t; best_case = 8

    return best, best_idx, best_case


# -------------------------
# VX (array-based variant)
# -------------------------
@nb.njit(cache=True, fastmath=True)
def compose_vx_best_over_r(
    Lu: np.ndarray, Ru: np.ndarray, Lc: np.ndarray, Rc: np.ndarray,
    coax_total: np.ndarray, coax_bonus: np.ndarray,
    Gw: float, cap_pen: float, g: float
):
    """
    Array-based minimization over r for VX.

    For each r:
      base_min = min( Gw + Lu + Ru + cap,
                          Lc + Ru + cap,
                          Lu + Rc + cap,
                          Lc + Rc + cap )
      total    = base_min + g * coax_total[r] + coax_bonus[r]

    Returns (best_value, best_index, base_case_id 0..3).
    """
    L = Lu.shape[0]
    best = INF64
    best_idx = -1
    best_case = -1

    for t in range(L):
        a0 = Gw + Lu[t] + Ru[t] + cap_pen
        a1 =      Lc[t] + Ru[t] + cap_pen
        a2 =      Lu[t] + Rc[t] + cap_pen
        a3 =      Lc[t] + Rc[t] + cap_pen

        base, which = _min4(a0, a1, a2, a3)
        cand = base + g * coax_total[t] + coax_bonus[t]

        if cand < best:
            best = cand
            best_idx = t
            best_case = which

    return best, best_idx, best_case


@nb.njit(cache=True, fastmath=True)
def best_sum(left: np.ndarray, right: np.ndarray):
    """
    Return (min_t left[t]+right[t], t). Assumes same length.
    """
    n = left.shape[0]
    best = INF64
    best_idx = -1
    for t in range(n):
        v = left[t] + right[t]
        if v < best:
            best = v
            best_idx = t
    return best, best_idx


@nb.njit(cache=True, fastmath=True)
def best_sum_with_penalty(left: np.ndarray, right: np.ndarray, penalty: float):
    """
    Return (min_t left[t]+right[t]+penalty, t). Assumes same length.
    """
    n = left.shape[0]
    best = INF64
    best_idx = -1
    for t in range(n):
        v = left[t] + right[t] + penalty
        if v < best:
            best = v
            best_idx = t
    return best, best_idx
