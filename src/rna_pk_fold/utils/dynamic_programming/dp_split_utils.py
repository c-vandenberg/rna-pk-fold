import math
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.numba_kernels import min_sum_over_index, min_sum_with_penalty_over_index
from rna_pk_fold.utils.dynamic_programming.matrix_utils import get_wxi_or_wx, get_zhx_energy_with_collapse


# ---------------------------------------------------------------------
# Split Modes
# ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WhxSplitMode:
    """
    Split-mode identifiers for WHX bifurcations.

    Constants
    ---------
    LEFT_WHX_WX : int
        Evaluate `WHX(i, r : k, l) + WX(r+1, j)` over `r ∈ [i, j-1]`.
    RIGHT_WX_WHX : int
        Evaluate `WX(i, s) + WHX(s+1, j : k, l)` over `s ∈ [i, j-1]`.
    OVERLAP : int
        Evaluate `WHX(i, r : k, l) + WHX(r+1, j : k, l)` (overlapping hole) over `r`.
    """
    LEFT_WHX_WX = 0
    RIGHT_WX_WHX = 1
    OVERLAP = 2


@dataclass(frozen=True, slots=True)
class VhxSplitMode:
    """
    Split-mode identifiers for ZHX+WX scans used inside VHX recurrences.

    Constants
    ---------
    LEFT_ZHX_WX : int
        Evaluate `ZHX(i, j : r, l) + WX(r+1, k)` over `r ∈ [i, k-1]`.
    RIGHT_ZHX_WX : int
        Evaluate `ZHX(i, j : k, s2) + WX(l, s2-1)` over `s2 ∈ [l+1, j]`.
    """
    LEFT_ZHX_WX  = 1
    RIGHT_ZHX_WX = 2


@dataclass(frozen=True, slots=True)
class ZhxSplitMode:
    """
    Split-mode identifiers for ZHX+WX scans used inside ZHX recurrences.

    Constants
    ---------
    LEFT_ZHX_WX : int
        Evaluate `ZHX(i, j : r, l) + WX(r+1, k)` over `r ∈ [i, k-1]`.
    RIGHT_ZHX_WX : int
        Evaluate `ZHX(i, j : k, s2) + WX(l, s2-1)` over `s2 ∈ [l+1, j]`.
    """
    LEFT_ZHX_WX  = 1
    RIGHT_ZHX_WX = 2


@dataclass(frozen=True, slots=True)
class YhxSplitMode:
    """
    Split-mode identifiers for YHX+WX scans used inside YHX recurrences.

    Constants
    ---------
    LEFT_YHX_WX : int
        Evaluate `YHX(i, r : k, l) + WX(r+1, j)` over `r ∈ [i, j-1]`.
    RIGHT_WX_YHX : int
        Evaluate `WX(i, s2) + YHX(s2+1, j : k, l)` over `s2 ∈ [i, j-1]`.
    """
    LEFT_YHX_WX  = 1
    RIGHT_WX_YHX = 2


# ---------------------------------------------------------------------
# WHX: Build Left/Right Cost Vectors and Compute Best Split
# ---------------------------------------------------------------------
def build_whx_split_cost_vectors(
    mode: int,
    state: EddyRivasFoldState,
    i: int, j: int, k: int, l: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build left/right energy vectors for scanning WHX split points.

    For each split offset `t ∈ [0, j-i)`, this prepares the energy on the
    left side and the right side according to the requested split `mode`
    (e.g., `WHX + WX`, `WX + WHX`, or overlapping `WHX + WHX`).

    Parameters
    ----------
    mode : int
        One of `WhxSplitMode.LEFT_WHX_WX`, `WhxSplitMode.RIGHT_WX_WHX`, or
        `WhxSplitMode.OVERLAP`.
    state : EddyRivasFoldState
        Fold state providing access to WHX and WX energies.
    i, j : int
        Outer span indices.
    k, l : int
        Inner hole indices.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        `(left_costs, right_costs)`; each array has length `j - i` and `dtype=float64`.
        Non-finite sub-problems remain `+inf`.
    """
    span_len = _compute_span_length(i, j)
    left_costs = np.full(span_len, np.inf, dtype=np.float64)
    right_costs = np.full(span_len, np.inf, dtype=np.float64)

    for t in range(span_len):
        split_idx = i + t
        if mode == WhxSplitMode.LEFT_WHX_WX:
            left_val = state.whx_matrix.get_energy(i, split_idx, k, l)
            right_val = get_wxi_or_wx(state, split_idx + 1, j)
        elif mode == WhxSplitMode.RIGHT_WX_WHX:
            left_val = get_wxi_or_wx(state, i, split_idx)
            right_val = state.whx_matrix.get_energy(split_idx + 1, j, k, l)
        else:  # WhxSplitMode.OVERLAP
            left_val = state.whx_matrix.get_energy(i, split_idx, k, l)
            right_val = state.whx_matrix.get_energy(split_idx + 1, j, k, l)

        if math.isfinite(left_val):
            left_costs[t] = left_val
        if math.isfinite(right_val):
            right_costs[t] = right_val

    return left_costs, right_costs


def compute_whx_best_split_energy(
    mode: int,
    state: EddyRivasFoldState,
    i: int, j: int, k: int, l: int,
    overlap_penalty: float = 0.0,
) -> Tuple[float, int]:
    """
    Compute `(min_energy, argmin_t)` for WHX split mode over `t ∈ [0, j-i)`.

    Uses `min_sum_over_index` or `min_sum_with_penalty_over_index` depending on
    whether the `OVERLAP` mode requires an extra penalty term.

    Parameters
    ----------
    mode : int
        One of `WhxSplitMode.LEFT_WHX_WX`, `WhxSplitMode.RIGHT_WX_WHX`, or
        `WhxSplitMode.OVERLAP`.
    state : EddyRivasFoldState
        Fold state providing access to WHX and WX energies.
    i, j, k, l : int
        Coordinates of the sub-problem.
    overlap_penalty : float, default=0.0
        Additional penalty applied only in `WhxSplitMode.OVERLAP`.

    Returns
    -------
    Tuple[float, int]
        `(best_energy, t_star)` where `t_star` is the split offset; `-1` if no
        finite candidate exists.
    """
    if _compute_span_length(i, j) == 0:
        return math.inf, -1

    left_costs, right_costs = build_whx_split_cost_vectors(mode, state, i, j, k, l)

    if mode == WhxSplitMode.OVERLAP and overlap_penalty != 0.0:
        return min_sum_with_penalty_over_index(left_costs, right_costs, float(overlap_penalty))

    return min_sum_over_index(left_costs, right_costs)

# ---------------------------------------------------------------------
# VHX: ZHX+WX Split Scans Used Inside VHX Recurrences
# ---------------------------------------------------------------------
def compute_vhx_best_split_over_zhx_wx(
    mode: VhxSplitMode,
    state,
    i: int, j: int, k: int, l: int,
) -> Tuple[float, int]:
    """
    Compute `(best_energy, t_star)` for ZHX+WX splits in VHX recurrences.

    Parameters
    ----------
    mode : VhxSplitMode
        Either `VhxSplitMode.LEFT_ZHX_WX` or `VhxSplitMode.RIGHT_ZHX_WX`.
    state : Any
        Fold state providing `zhx_matrix`, `vxu_matrix`, and WX access.
    i, j, k, l : int
        Coordinates of the sub-problem.

    Returns
    -------
    Tuple[float, int]
        `(best_energy, t_star)` where `t_star` is the argmin index over the
        split dimension, or `(-1)` if no finite candidate exists.
    """
    if mode == VhxSplitMode.LEFT_ZHX_WX:
        left_range_len = max(0, k - i)
        if left_range_len <= 0:
            return math.inf, -1
        left_costs = np.full(left_range_len, np.inf, dtype=np.float64)
        right_costs = np.full(left_range_len, np.inf, dtype=np.float64)
        for t in range(left_range_len):
            r = i + t
            lv = get_zhx_energy_with_collapse(state.zhx_matrix, state.vxu_matrix, i, j, r, l)
            rv = get_wxi_or_wx(state, r + 1, k)
            if math.isfinite(lv):
                left_costs[t] = lv
            if math.isfinite(rv):
                right_costs[t] = rv
        return min_sum_over_index(left_costs, right_costs)

    # Mode == VhxSplitMode.RIGHT_ZHX_WX
    right_range_len = max(0, j - l)  # = j - l
    if right_range_len <= 0:
        return math.inf, -1
    left_costs = np.full(right_range_len, np.inf, dtype=np.float64)
    right_costs = np.full(right_range_len, np.inf, dtype=np.float64)
    for t in range(right_range_len):
        s2 = (l + 1) + t
        lv = get_zhx_energy_with_collapse(state.zhx_matrix, state.vxu_matrix, i, j, k, s2)
        rv = get_wxi_or_wx(state, l, s2 - 1)
        if math.isfinite(lv):
            left_costs[t] = lv
        if math.isfinite(rv):
            right_costs[t] = rv

    return min_sum_over_index(left_costs, right_costs)


# ---------------------------------------------------------------------
# ZHX: ZHX+WX Split Scans Used Inside ZHX Recurrences (No Collapse on ZHX)
# ---------------------------------------------------------------------
def compute_zhx_best_split_over_zhx_wx(
    mode: ZhxSplitMode,
    state,
    i: int, j: int, k: int, l: int,
) -> Tuple[float, int]:
    """
    Compute `(best_energy, t_star)` for ZHX+WX splits in ZHX recurrences.

    Uses direct `ZHX.get_energy` (no collapse shortcut) for the ZHX term.

    Parameters
    ----------
    mode : ZhxSplitMode
        Either `ZhxSplitMode.LEFT_ZHX_WX` or `ZhxSplitMode.RIGHT_ZHX_WX`.
    state : Any
        Fold state providing `zhx_matrix` and WX access.
    i, j, k, l : int
        Coordinates of the sub-problem.

    Returns
    -------
    Tuple[float, int]
        `(best_energy, t_star)` where `t_star` is the argmin over the split
        index, or `(-1)` if no finite candidate exists.
    """
    if mode == ZhxSplitMode.LEFT_ZHX_WX:
        left_range_len = max(0, k - i)
        if left_range_len <= 0:
            return math.inf, -1
        left_costs = np.full(left_range_len, np.inf, dtype=np.float64)
        right_costs = np.full(left_range_len, np.inf, dtype=np.float64)
        for t in range(left_range_len):
            r = i + t
            lv = state.zhx_matrix.get_energy(i, j, r, l)
            rv = get_wxi_or_wx(state, r + 1, k)
            if math.isfinite(lv):
                left_costs[t] = lv
            if math.isfinite(rv):
                right_costs[t] = rv
        return min_sum_over_index(left_costs, right_costs)

    # mode == ZhxSplitMode.RIGHT_ZHX_WX
    right_range_len = max(0, j - l)
    if right_range_len <= 0:
        return math.inf, -1
    left_costs = np.full(right_range_len, np.inf, dtype=np.float64)
    right_costs = np.full(right_range_len, np.inf, dtype=np.float64)
    for t in range(right_range_len):
        s2 = (l + 1) + t
        lv = state.zhx_matrix.get_energy(i, j, k, s2)
        rv = get_wxi_or_wx(state, l, s2 - 1)
        if math.isfinite(lv):
            left_costs[t] = lv
        if math.isfinite(rv):
            right_costs[t] = rv

    return min_sum_over_index(left_costs, right_costs)


# ---------------------------------------------------------------------
# YHX: YHX+WX Split Scans Used Inside YHX Recurrences
# ---------------------------------------------------------------------
def compute_yhx_best_split_over_yhx_wx(
    mode: YhxSplitMode,
    state,
    i: int, j: int, k: int, l: int,
) -> Tuple[float, int]:
    """
    Compute `(best_energy, t_star)` for YHX+WX splits in YHX recurrences.

    Parameters
    ----------
    mode : YhxSplitMode
        Either `YhxSplitMode.LEFT_YHX_WX` or `YhxSplitMode.RIGHT_WX_YHX`.
    state : Any
        Fold state providing `yhx_matrix` and WX access.
    i, j, k, l : int
        Coordinates of the sub-problem.

    Returns
    -------
    Tuple[float, int]
        `(best_energy, t_star)` where `t_star` is the argmin index, or `(-1)` if none.
    """
    span_len = max(0, j - i)
    if span_len <= 0:
        return math.inf, -1

    if mode == YhxSplitMode.LEFT_YHX_WX:
        left_costs = np.full(span_len, np.inf, dtype=np.float64)
        right_costs = np.full(span_len, np.inf, dtype=np.float64)
        for t in range(span_len):
            r = i + t
            lv = state.yhx_matrix.get_energy(i, r, k, l)
            rv = get_wxi_or_wx(state, r + 1, j)
            if math.isfinite(lv):
                left_costs[t] = lv
            if math.isfinite(rv):
                right_costs[t] = rv
        return min_sum_over_index(left_costs, right_costs)

    # YhxSplitMode.RIGHT_WX_YHX
    left_costs = np.full(span_len, np.inf, dtype=np.float64)
    right_costs = np.full(span_len, np.inf, dtype=np.float64)
    for t in range(span_len):
        s2 = i + t
        lv = get_wxi_or_wx(state, i, s2)
        rv = state.yhx_matrix.get_energy(s2 + 1, j, k, l)
        if math.isfinite(lv):
            left_costs[t] = lv
        if math.isfinite(rv):
            right_costs[t] = rv

    return min_sum_over_index(left_costs, right_costs)


# ---------------------------------------------------------------------
# Generic Helpers for Split Vectors and Selection
# ---------------------------------------------------------------------
def build_generic_split_cost_vectors(
    length: int,
    left_fetch: Callable[[int], float],
    right_fetch: Callable[[int], float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build generic left/right cost vectors with fetch callbacks.

    Parameters
    ----------
    length : int
        Number of split positions `t` to consider.
    left_fetch : Callable[[int], float]
        Callback returning left energy at index `t`.
    right_fetch : Callable[[int], float]
        Callback returning right energy at index `t`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        `(left_costs, right_costs)` arrays of length `length`, `dtype=float64`.
    """
    left_costs = np.full(length, np.inf, dtype=np.float64)
    right_costs = np.full(length, np.inf, dtype=np.float64)
    for t in range(length):
        lv = left_fetch(t)
        rv = right_fetch(t)
        if math.isfinite(lv):
            left_costs[t] = lv
        if math.isfinite(rv):
            right_costs[t] = rv

    return left_costs, right_costs


def compute_best_split_from_cost_vectors(
    left_costs: np.ndarray,
    right_costs: np.ndarray,
    penalty: float = 0.0,
) -> Tuple[float, int]:
    """
    Compute `(min_energy, argmin_t)` from pre-built cost vectors.

    Parameters
    ----------
    left_costs, right_costs : np.ndarray
        Left/right cost arrays over the same split axis.
    penalty : float, default=0.0
        Constant term added to each candidate.

    Returns
    -------
    Tuple[float, int]
        `(best_energy, t_star)` using either `min_sum_over_index` or
        `min_sum_with_penalty_over_index` depending on `penalty`.
    """
    if penalty == 0.0:
        return min_sum_over_index(left_costs, right_costs)

    return min_sum_with_penalty_over_index(left_costs, right_costs, float(penalty))


# ---------------------------------------------------------------------
# Internal: Span-Length Helper
# ---------------------------------------------------------------------
def _compute_span_length(i: int, j: int) -> int:
    """
    Return the number of split positions for an outer span `(i, j)`.

    This is `max(0, j - i)` and corresponds to the number of choices for a
    split index `r` in `[i, j-1]` (or offset `t` in `[0, j-i)`).

    Parameters
    ----------
    i, j : int
        Outer span indices.

    Returns
    -------
    int
        The non-negative span length.
    """
    return max(0, j - i)


def validate_left_split_index(
    outer_start: int,
    outer_end: int,
    split_index: Optional[int],
) -> Optional[int]:
    """
    Validate or synthesize a left-split index that makes progress.

    If ``split_index`` is ``None``, the midpoint is used. The result is valid iff
    ``outer_start < split < outer_end``.

    Parameters
    ----------
    outer_start : int
        5' index (inclusive) of the outer span.
    outer_end : int
        3' index (inclusive) of the outer span.
    split_index : int or None
        Proposed split index; if ``None``, use midpoint.

    Returns
    -------
    int or None
        A valid split index strictly inside ``(outer_start, outer_end)``,
        or ``None`` if no progress is possible.
    """
    valid_split = (outer_start + outer_end) // 2 if split_index is None else split_index
    if valid_split <= outer_start or valid_split >= outer_end:
        return None
    return valid_split


def validate_right_split_index(
    outer_start: int,
    outer_end: int,
    split_index: Optional[int],
) -> Optional[int]:
    """
    Validate or synthesize a right-split index that makes progress.

    If ``split_index`` is ``None``, the midpoint is used. The result is valid iff
    ``outer_start <= split < outer_end``.

    Parameters
    ----------
    outer_start : int
        5' index (inclusive) of the outer span.
    outer_end : int
        3' index (inclusive) of the outer span.
    split_index : int or None
        Proposed split index; if ``None``, use midpoint.

    Returns
    -------
    int or None
        A valid split index inside ``[outer_start, outer_end)`` or ``None`` if
        no progress is possible.
    """
    valid_split = (outer_start + outer_end) // 2 if split_index is None else split_index
    if valid_split < outer_start or valid_split >= outer_end:
        return None
    return valid_split

