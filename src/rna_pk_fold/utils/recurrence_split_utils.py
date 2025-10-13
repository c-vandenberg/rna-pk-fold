import math
from typing import Callable, Tuple

import numpy as np

from rna_pk_fold.folding.eddy_rivas.numba_kernels import best_sum, best_sum_with_penalty


def build_split_vectors(n:int,
                        left_get: Callable[[int], float],
                        right_get: Callable[[int], float]) -> Tuple[np.ndarray,np.ndarray]:
    left = np.full(n, np.inf, dtype=np.float64)
    right = np.full(n, np.inf, dtype=np.float64)
    for t in range(n):
        lv = left_get(t)
        rv = right_get(t)
        if math.isfinite(lv): left[t] = lv
        if math.isfinite(rv): right[t] = rv
    return left, right


def best_split(left: np.ndarray, right: np.ndarray, penalty: float = 0.0) -> Tuple[float,int]:
    if penalty == 0.0:
        return best_sum(left, right)

    return best_sum_with_penalty(left, right, float(penalty))
