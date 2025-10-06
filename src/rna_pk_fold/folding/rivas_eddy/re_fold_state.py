from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Tuple

from rna_pk_fold.structures.tri_matrix import ReTriMatrix
from rna_pk_fold.structures.gap_matrix import SparseGapMatrix, SparseGapBackptr

wx_back_ptr: Dict[Tuple[int, int], Tuple[str, Tuple[int, int, int]]]
vx_back_ptr: Dict[Tuple[int, int], Tuple[str, Tuple[int, int, int]]]


@dataclass(slots=True)
class RivasEddyFoldState:
    """
    Holds the non-gap and gap matrices for the R&E algorithm.
    Values only for scaffolding; Step 12 will fill recurrences.
    """
    n: int

    # Non-gap (triangular)
    wx_matrix: ReTriMatrix
    vx_matrix: ReTriMatrix
    wxi_matrix: ReTriMatrix
    wxu_matrix: ReTriMatrix  # uncharged (baseline, nested-only)
    wxc_matrix: ReTriMatrix  # charged   (has paid Gw at least once)
    vxu_matrix: ReTriMatrix
    vxc_matrix: ReTriMatrix
    wx_back_ptr: dict
    vx_back_ptr: dict

    # Gap (sparse 4D)
    whx_matrix: SparseGapMatrix
    vhx_matrix: SparseGapMatrix
    yhx_matrix: SparseGapMatrix
    zhx_matrix: SparseGapMatrix

    whx_back_ptr: SparseGapBackptr
    vhx_back_ptr: SparseGapBackptr
    yhx_back_ptr: SparseGapBackptr
    zhx_back_ptr: SparseGapBackptr


def make_re_fold_state(n: int) -> RivasEddyFoldState:
    re = RivasEddyFoldState(
        n=n,
        # 2D
        wx_matrix=ReTriMatrix(n),
        vx_matrix=ReTriMatrix(n),
        wxi_matrix=ReTriMatrix(n),
        wxu_matrix=ReTriMatrix(n),
        wxc_matrix=ReTriMatrix(n),
        vxu_matrix=ReTriMatrix(n),
        vxc_matrix=ReTriMatrix(n),
        wx_back_ptr={},
        vx_back_ptr={},
        # 4D sparse
        whx_matrix=SparseGapMatrix(n),
        vhx_matrix=SparseGapMatrix(n),
        yhx_matrix=SparseGapMatrix(n),
        zhx_matrix=SparseGapMatrix(n),
        whx_back_ptr=SparseGapBackptr(n),
        vhx_back_ptr=SparseGapBackptr(n),
        yhx_back_ptr=SparseGapBackptr(n),
        zhx_back_ptr=SparseGapBackptr(n),
    )

    # ---------- Initialization (base conditions) ----------
    # Non-gap base cases (R&E Section 5.2):
    #   wx(i,i) = 0; vx(i,i) = +inf
    for i in range(n):
        re.wx_matrix.set(i, i, 0.0)
        re.vx_matrix.set(i, i, math.inf)
        re.wxi_matrix.set(i, i, 0.0)
        re.wxu_matrix.set(i, i, 0.0)
        re.wxc_matrix.set(i, i, 0.0)  # neutral so empty-in-empty stays finite if used
        re.vxu_matrix.set(i, i, math.inf)
        re.vxc_matrix.set(i, i, math.inf)

    # Gap matrices: we *don’t* prefill O(N^4); instead we implement the
    # collapse identities via helpers below (lazy). We do set illegal holes to +inf
    # by default due to SparseGapMatrix.get() behavior.

    return re