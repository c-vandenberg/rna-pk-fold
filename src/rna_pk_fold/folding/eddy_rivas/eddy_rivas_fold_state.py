from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Tuple

from rna_pk_fold.structures.tri_matrix import RivasEddyTriMatrix, RivasEddyTriBackPointer
from rna_pk_fold.structures.gap_matrix import SparseGapMatrix, SparseGapBackptr
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer

wx_back_ptr: Dict[Tuple[int, int], Tuple[str, Tuple[int, int, int]]]
vx_back_ptr: Dict[Tuple[int, int], Tuple[str, Tuple[int, int, int]]]


@dataclass(slots=True)
class EddyRivasFoldState:
    """
    Holds the non-gap and gap matrices for the R&E algorithm.
    Values only for scaffolding; Step 12 will fill recurrences.
    """
    n: int

    # Non-gap (Energies, 2D)
    wx_matrix: RivasEddyTriMatrix
    vx_matrix: RivasEddyTriMatrix
    wxi_matrix: RivasEddyTriMatrix
    wxu_matrix: RivasEddyTriMatrix  # uncharged (baseline, nested-only)
    wxc_matrix: RivasEddyTriMatrix  # charged   (has paid Gw at least once)
    vxu_matrix: RivasEddyTriMatrix
    vxc_matrix: RivasEddyTriMatrix

    # Non-gap (Back-pointers, 2D)
    wx_back_ptr: RivasEddyTriBackPointer
    vx_back_ptr: RivasEddyTriBackPointer

    # Gap (Energies, 4D)
    whx_matrix: SparseGapMatrix
    vhx_matrix: SparseGapMatrix
    yhx_matrix: SparseGapMatrix
    zhx_matrix: SparseGapMatrix

    # Gap (Back-pointers, 4D)
    whx_back_ptr: SparseGapBackptr
    vhx_back_ptr: SparseGapBackptr
    yhx_back_ptr: SparseGapBackptr
    zhx_back_ptr: SparseGapBackptr


def make_re_fold_state(n: int) -> EddyRivasFoldState:
    re = EddyRivasFoldState(
        n=n,

        # Non-gap (Energies, 2D)
        wx_matrix=RivasEddyTriMatrix(n),
        vx_matrix=RivasEddyTriMatrix(n),
        wxi_matrix=RivasEddyTriMatrix(n),
        wxu_matrix=RivasEddyTriMatrix(n),
        wxc_matrix=RivasEddyTriMatrix(n),
        vxu_matrix=RivasEddyTriMatrix(n),
        vxc_matrix=RivasEddyTriMatrix(n),

        # Non-gap (Back-pointers, 2D)
        wx_back_ptr=RivasEddyTriBackPointer(n),
        vx_back_ptr=RivasEddyTriBackPointer(n),

        # Gap (Energies, 4D)
        whx_matrix=SparseGapMatrix(n),
        vhx_matrix=SparseGapMatrix(n),
        yhx_matrix=SparseGapMatrix(n),
        zhx_matrix=SparseGapMatrix(n),

        # Gap (Back-pointers, 4D)
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