import math
from typing import Callable, Tuple

import numpy as np

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.numba_kernels import best_sum, best_sum_with_penalty
from rna_pk_fold.utils.matrix_utils import get_wxi_or_wx


class WHXSplitMode:
    LEFT_WHX_WX = 0      # WHX(i,r:k,l)  + WX(r+1,j)
    RIGHT_WX_WHX = 1     # WX(i,s)       + WHX(s+1,j:k,l)
    OVERLAP = 2          # WHX(i,r:k,l)  + WHX(r+1,j:k,l)


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
