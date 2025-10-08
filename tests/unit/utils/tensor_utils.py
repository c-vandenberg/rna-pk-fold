import numpy as np
from typing import Dict, Tuple, Optional

Outer = Tuple[int, int]
GapDict = Dict[Tuple[Outer, Outer], float]

def build_dense_gap_tensors(
    n: int,
    whx: GapDict,
    vhx: GapDict,
    zhx: GapDict,
    yhx: GapDict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (n, n, n, n)
    WHX = np.full(shape, np.inf, dtype=np.float64)
    VHX = np.full(shape, np.inf, dtype=np.float64)
    ZHX = np.full(shape, np.inf, dtype=np.float64)
    YHX = np.full(shape, np.inf, dtype=np.float64)

    def backfill(tensor: np.ndarray, src: GapDict) -> None:
        for (i, j), (k, l) in src.keys():
            val = src[(i, j), (k, l)]
            if 0 <= i < n and 0 <= j < n and 0 <= k < n and 0 <= l < n:
                tensor[i, j, k, l] = val

    backfill(WHX, whx)
    backfill(VHX, vhx)
    backfill(ZHX, zhx)
    backfill(YHX, yhx)
    return WHX, VHX, ZHX, YHX
