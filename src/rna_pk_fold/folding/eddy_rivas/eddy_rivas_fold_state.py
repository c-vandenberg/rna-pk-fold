from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Tuple

from rna_pk_fold.structures.tri_matrix import EddyRivasTriMatrix, EddyRivasTriBackPointer
from rna_pk_fold.structures.gap_matrix import SparseGapMatrix, SparseGapBackptr

wx_back_ptr: Dict[Tuple[int, int], Tuple[str, Tuple[int, int, int]]]
vx_back_ptr: Dict[Tuple[int, int], Tuple[str, Tuple[int, int, int]]]


@dataclass(slots=True)
class EddyRivasFoldState:
    """
    Holds all dynamic programming matrices for the Eddy & Rivas algorithm.

    This class encapsulates both the non-gap (2D triangular) and gap (4D sparse)
    matrices required for the RNA pseudoknot folding algorithm. Upon instantiation,
    it initializes all matrices and sets their base conditions.

    Attributes:
        seq_len: The length of the RNA sequence.
    """
    seq_len: int

    # --- Non-gap Matrices (Energies, 2D) ---
    wx_matrix: EddyRivasTriMatrix
    vx_matrix: EddyRivasTriMatrix
    wxi_matrix: EddyRivasTriMatrix
    wxu_matrix: EddyRivasTriMatrix  # uncharged (baseline, nested-only)
    wxc_matrix: EddyRivasTriMatrix  # charged   (has paid Gw at least once)
    vxu_matrix: EddyRivasTriMatrix
    vxc_matrix: EddyRivasTriMatrix

    # --- Non-gap Matrices (Back-pointers, 2D) ---
    wx_back_ptr: EddyRivasTriBackPointer
    vx_back_ptr: EddyRivasTriBackPointer

    # --- Gap Matrices (Energies, 4D) ---
    whx_matrix: SparseGapMatrix
    vhx_matrix: SparseGapMatrix
    yhx_matrix: SparseGapMatrix
    zhx_matrix: SparseGapMatrix

    # --- Gap Matrices (Back-pointers, 4D) ---
    whx_back_ptr: SparseGapBackptr
    vhx_back_ptr: SparseGapBackptr
    yhx_back_ptr: SparseGapBackptr
    zhx_back_ptr: SparseGapBackptr


def init_eddy_rivas_fold_state(n: int) -> EddyRivasFoldState:
    er_fold_state = EddyRivasFoldState(
        seq_len=n,

        # --- Matrix Instantiation ---
        # Non-gap (Energies)
        wx_matrix=EddyRivasTriMatrix(n),
        vx_matrix=EddyRivasTriMatrix(n),
        wxi_matrix=EddyRivasTriMatrix(n),
        wxu_matrix=EddyRivasTriMatrix(n),
        wxc_matrix=EddyRivasTriMatrix(n),
        vxu_matrix=EddyRivasTriMatrix(n),
        vxc_matrix=EddyRivasTriMatrix(n),

        # Non-gap (Back-pointers)
        wx_back_ptr=EddyRivasTriBackPointer(n),
        vx_back_ptr=EddyRivasTriBackPointer(n),

        # Gap (Back-pointers)
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

    # ---------- Matrix Initialization (base conditions) ----------
    # As per Eddy & Rivas, Section 5.2, for spans of length 0.
    for i in range(n):
        er_fold_state.wx_matrix.set(i, i, 0.0)
        er_fold_state.vx_matrix.set(i, i, math.inf)
        er_fold_state.wxi_matrix.set(i, i, 0.0)
        er_fold_state.wxu_matrix.set(i, i, 0.0)

        # Neutral init, so empty regions don't cause infinite energy
        er_fold_state.wxc_matrix.set(i, i, 0.0)
        er_fold_state.vxu_matrix.set(i, i, math.inf)
        er_fold_state.vxc_matrix.set(i, i, math.inf)

    # Note: Gap matrices are sparse and initialized on-demand.
    # The default `get()` behavior of SparseGapMatrix returns +inf for
    # unassigned entries, correctly handling illegal holes.

    return er_fold_state