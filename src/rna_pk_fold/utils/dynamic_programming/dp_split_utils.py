import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.numba_kernels import best_sum, best_sum_with_penalty
from rna_pk_fold.utils.dynamic_programming.matrix_utils import get_wxi_or_wx, get_zhx_with_collapse


@dataclass(frozen=True, slots=True)
class WHXSplitMode:
    LEFT_WHX_WX = 0      # WHX(i,r:k,l)  + WX(r+1,j)
    RIGHT_WX_WHX = 1     # WX(i,s)       + WHX(s+1,j:k,l)
    OVERLAP = 2          # WHX(i,r:k,l)  + WHX(r+1,j:k,l)


@dataclass(frozen=True, slots=True)
class VHXSplitMode:
    LEFT_ZHX_WX  = 1   # ZHX(i,j:r,l) + WX(r+1,k)
    RIGHT_ZHX_WX = 2   # ZHX(i,j:k,s2) + WX(l, s2-1)


@dataclass(frozen=True, slots=True)
class ZHXSplitMode:
    LEFT_ZHX_WX  = 1   # ZHX(i,j:r,l) + WX(r+1,k)
    RIGHT_ZHX_WX = 2   # ZHX(i,j:k,s2) + WX(l, s2-1)

@dataclass(frozen=True, slots=True)
class YHXSplitMode:
    LEFT_YHX_WX  = 1   # YHX(i, r: k, l) + WX(r+1, j)
    RIGHT_WX_YHX = 2   # WX(i, s2) + YHX(s2+1, j: k, l)


def whx_build_split_vectors(
    mode: int,
    eddy_rivas_fold_state: EddyRivasFoldState,
    i: int, j: int, k: int, l: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build left/right vectors for split scans across t in [0, j-i).
    """
    lr = _span_len(i, j)
    left_vec = np.full(lr, np.inf, dtype=np.float64)
    right_vec = np.full(lr, np.inf, dtype=np.float64)

    for t in range(lr):
        r_or_s = i + t
        if mode == WHXSplitMode.LEFT_WHX_WX:
            lv = eddy_rivas_fold_state.whx_matrix.get(i, r_or_s, k, l)
            rv = get_wxi_or_wx(eddy_rivas_fold_state, r_or_s + 1, j)
        elif mode == WHXSplitMode.RIGHT_WX_WHX:
            lv = get_wxi_or_wx(eddy_rivas_fold_state, i, r_or_s)
            rv = eddy_rivas_fold_state.whx_matrix.get(r_or_s + 1, j, k, l)
        else:  # OVERLAP
            lv = eddy_rivas_fold_state.whx_matrix.get(i, r_or_s, k, l)
            rv = eddy_rivas_fold_state.whx_matrix.get(r_or_s + 1, j, k, l)

        if math.isfinite(lv): left_vec[t] = lv
        if math.isfinite(rv): right_vec[t] = rv

    return left_vec, right_vec


def whx_split_min(
    mode: int,
    eddy_rivas_fold_state: EddyRivasFoldState,
    i: int, j: int, k: int, l: int,
    overlap_penalty: float = 0.0
) -> Tuple[float, int]:
    """
    Return (cand, t_star) using best_sum or best_sum_with_penalty
    for the requested split mode.
    """
    if _span_len(i, j) == 0:
        return math.inf, -1

    left_vec, right_vec = whx_build_split_vectors(mode, eddy_rivas_fold_state, i, j, k, l)

    if mode == WHXSplitMode.OVERLAP and overlap_penalty != 0.0:
        return best_sum_with_penalty(left_vec, right_vec, float(overlap_penalty))
    else:
        return best_sum(left_vec, right_vec)


def zhx_wx_split_min_vhx(
    mode: VHXSplitMode,
    state,
    i: int, j: int, k: int, l: int,
):
    """
    Compute min over split for ZHX+WX in VHX recurrences.
    Returns (best_energy, t_star) where t_star is the argmin index over the split dimension,
    or (-1) if no finite candidate.
    """
    if mode == VHXSplitMode.LEFT_ZHX_WX:
        lr = max(0, k - i)
        if lr <= 0:
            return math.inf, -1
        left_vec  = np.full(lr, np.inf, dtype=np.float64)
        right_vec = np.full(lr, np.inf, dtype=np.float64)
        for t in range(lr):
            r = i + t
            lv = get_zhx_with_collapse(state.zhx_matrix, state.vxu_matrix, i, j, r, l)
            rv = get_wxi_or_wx(state, r + 1, k)
            if math.isfinite(lv): left_vec[t] = lv
            if math.isfinite(rv): right_vec[t] = rv
        return best_sum(left_vec, right_vec)

    # mode == VHXSplitMode.RIGHT_ZHX_WX
    ls = max(0, j - l)  # = j - l
    if ls <= 0:
        return math.inf, -1
    left_vec  = np.full(ls, np.inf, dtype=np.float64)
    right_vec = np.full(ls, np.inf, dtype=np.float64)
    for t in range(ls):
        s2 = (l + 1) + t
        lv = get_zhx_with_collapse(state.zhx_matrix, state.vxu_matrix, i, j, k, s2)
        rv = get_wxi_or_wx(state, l, s2 - 1)
        if math.isfinite(lv): left_vec[t] = lv
        if math.isfinite(rv): right_vec[t] = rv
    return best_sum(left_vec, right_vec)


def zhx_wx_split_min_zhx(
    mode: ZHXSplitMode,
    state,
    i: int, j: int, k: int, l: int,
):
    """
    Compute min over split for ZHX+WX recurrences in ZHX (no collapse;
    uses ZHX.get directly). Returns (best_energy, t_star) where t_star is
    the argmin over the split index, or (-1) if no finite candidate exists.
    """
    if mode == ZHXSplitMode.LEFT_ZHX_WX:
        lr = max(0, k - i)
        if lr <= 0:
            return math.inf, -1
        left_vec  = np.full(lr, np.inf, dtype=np.float64)
        right_vec = np.full(lr, np.inf, dtype=np.float64)
        for t in range(lr):
            r = i + t
            lv = state.zhx_matrix.get(i, j, r, l)
            rv = get_wxi_or_wx(state, r + 1, k)
            if math.isfinite(lv): left_vec[t] = lv
            if math.isfinite(rv): right_vec[t] = rv
        return best_sum(left_vec, right_vec)

    # mode == ZHXSplitMode.RIGHT_ZHX_WX
    ls = max(0, j - l)
    if ls <= 0:
        return math.inf, -1
    left_vec  = np.full(ls, np.inf, dtype=np.float64)
    right_vec = np.full(ls, np.inf, dtype=np.float64)
    for t in range(ls):
        s2 = (l + 1) + t
        lv = state.zhx_matrix.get(i, j, k, s2)
        rv = get_wxi_or_wx(state, l, s2 - 1)
        if math.isfinite(lv): left_vec[t] = lv
        if math.isfinite(rv): right_vec[t] = rv

    return best_sum(left_vec, right_vec)


def yhx_wx_split_min(
    mode: YHXSplitMode,
    state,
    i: int, j: int, k: int, l: int,
):
    """
    Compute min over splits for YHX+WX recurrences in YHX (plain matrix gets).
    Returns (best_energy, t_star) where t_star is the argmin index, or -1 if none.
    """
    span_len = max(0, j - i)
    if span_len <= 0:
        return math.inf, -1

    if mode == YHXSplitMode.LEFT_YHX_WX:
        left_vec  = np.full(span_len, np.inf, dtype=np.float64)
        right_vec = np.full(span_len, np.inf, dtype=np.float64)
        for t in range(span_len):
            r = i + t
            lv = state.yhx_matrix.get(i, r, k, l)
            rv = get_wxi_or_wx(state, r + 1, j)
            if math.isfinite(lv): left_vec[t] = lv
            if math.isfinite(rv): right_vec[t] = rv
        return best_sum(left_vec, right_vec)

    # RIGHT_WX_YHX
    left_vec  = np.full(span_len, np.inf, dtype=np.float64)
    right_vec = np.full(span_len, np.inf, dtype=np.float64)
    for t in range(span_len):
        s2 = i + t
        lv = get_wxi_or_wx(state, i, s2)
        rv = state.yhx_matrix.get(s2 + 1, j, k, l)
        if math.isfinite(lv): left_vec[t] = lv
        if math.isfinite(rv): right_vec[t] = rv
    return best_sum(left_vec, right_vec)


def build_split_vectors(
    seq_len:int,
    left_get: Callable[[int], float],
    right_get: Callable[[int], float]
) -> Tuple[np.ndarray,np.ndarray]:
    left = np.full(seq_len, np.inf, dtype=np.float64)
    right = np.full(seq_len, np.inf, dtype=np.float64)
    for t in range(seq_len):
        lv = left_get(t)
        rv = right_get(t)
        if math.isfinite(lv): left[t] = lv
        if math.isfinite(rv): right[t] = rv

    return left, right


def best_split(left: np.ndarray, right: np.ndarray, penalty: float = 0.0) -> Tuple[float,int]:
    if penalty == 0.0:
        return best_sum(left, right)

    return best_sum_with_penalty(left, right, float(penalty))


def _span_len(i: int, j: int) -> int:
    return max(0, j - i)
