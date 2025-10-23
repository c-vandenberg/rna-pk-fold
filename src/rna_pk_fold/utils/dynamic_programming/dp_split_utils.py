import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.numba_kernels import min_sum_over_index, min_sum_with_penalty_over_index
from rna_pk_fold.utils.dynamic_programming.matrix_utils import get_wxi_or_wx, get_zhx_energy_with_collapse


# ---------------------------------------------------------------------
# Split modes
# ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WhxSplitMode:
    LEFT_WHX_WX = 0      # WHX(i,r:k,l)  + WX(r+1,j)
    RIGHT_WX_WHX = 1     # WX(i,s)       + WHX(s+1,j:k,l)
    OVERLAP = 2          # WHX(i,r:k,l)  + WHX(r+1,j:k,l)


@dataclass(frozen=True, slots=True)
class VhxSplitMode:
    LEFT_ZHX_WX  = 1   # ZHX(i,j:r,l) + WX(r+1,k)
    RIGHT_ZHX_WX = 2   # ZHX(i,j:k,s2) + WX(l, s2-1)


@dataclass(frozen=True, slots=True)
class ZhxSplitMode:
    LEFT_ZHX_WX  = 1   # ZHX(i,j:r,l) + WX(r+1,k)
    RIGHT_ZHX_WX = 2   # ZHX(i,j:k,s2) + WX(l, s2-1)


@dataclass(frozen=True, slots=True)
class YhxSplitMode:
    LEFT_YHX_WX  = 1   # YHX(i, r: k, l) + WX(r+1, j)
    RIGHT_WX_YHX = 2   # WX(i, s2) + YHX(s2+1, j: k, l)


# ---------------------------------------------------------------------
# WHX: Build Left/Right Cost Vectors for Splits Across `t` in `[0, j-i)`
# ---------------------------------------------------------------------
def build_whx_split_vectors(
    mode: int,
    state: EddyRivasFoldState,
    i: int, j: int, k: int, l: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build left/right vectors for split scans across t in [0, j-i).
    """
    span_len = _span_length(i, j)
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


def compute_whx_split_min(
    mode: int,
    state: EddyRivasFoldState,
    i: int, j: int, k: int, l: int,
    overlap_penalty: float = 0.0,
) -> Tuple[float, int]:
    """
    Return (min_energy, argmin_t) using best_sum/best_sum_with_penalty
    for the requested WHX split mode.
    """
    if _span_length(i, j) == 0:
        return math.inf, -1

    left_costs, right_costs = build_whx_split_vectors(mode, state, i, j, k, l)

    if mode == WhxSplitMode.OVERLAP and overlap_penalty != 0.0:
        return min_sum_with_penalty_over_index(left_costs, right_costs, float(overlap_penalty))

    return min_sum_over_index(left_costs, right_costs)

# ---------------------------------------------------------------------
# VHX: ZHX+WX split Scans Used Inside VHX Recurrences
# ---------------------------------------------------------------------
def zhx_wx_split_min_vhx(
    mode: VhxSplitMode,
    state,
    i: int, j: int, k: int, l: int,
) -> Tuple[float, int]:
    """
    Compute min over split for ZHX+WX in VHX recurrences.
    Returns (best_energy, t_star) where t_star is the argmin index over the split dimension,
    or (-1) if no finite candidate exists.
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

        # mode == VhxSplitMode.RIGHT_ZHX_WX
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
def compute_zhx_split_min_over_zhx_wx(
    mode: ZhxSplitMode,
    state,
    i: int, j: int, k: int, l: int,
) -> Tuple[float, int]:
    """
    Compute min over split for ZHX+WX recurrences in ZHX (no collapse;
    uses ZHX.get directly). Returns (best_energy, t_star) where t_star is
    the argmin over the split index, or (-1) if no finite candidate exists.
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
def yhx_wx_split_min(
    mode: YhxSplitMode,
    state,
    i: int, j: int, k: int, l: int,
) -> Tuple[float, int]:
    """
    Compute min over splits for YHX+WX recurrences in YHX (plain matrix gets).
    Returns (best_energy, t_star) where t_star is the argmin index, or -1 if none.
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
# Generic helpers for split vectors and selection
# ---------------------------------------------------------------------
def build_split_cost_vectors(
    length: int,
    left_fetch: Callable[[int], float],
    right_fetch: Callable[[int], float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build generic left/right cost arrays of given length using fetchers.
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


def compute_best_split_from_vectors(
    left_costs: np.ndarray,
    right_costs: np.ndarray,
    penalty: float = 0.0,
) -> Tuple[float, int]:
    """
    Return (min_energy, argmin_t) given pre-built cost vectors and an optional penalty.
    """
    if penalty == 0.0:
        return min_sum_over_index(left_costs, right_costs)

    return min_sum_with_penalty_over_index(left_costs, right_costs, float(penalty))


# ---------------------------------------------------------------------
# Internal: Span Length Helper
# ---------------------------------------------------------------------
def _span_length(i: int, j: int) -> int:
    return max(0, j - i)
