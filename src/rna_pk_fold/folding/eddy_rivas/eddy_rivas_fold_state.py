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

    Attributes
    ----------
    seq_len : int
        The length of the RNA sequence (N).
    wx_matrix : EddyRivasTriMatrix
        Final optimal energy for subsequence [i, j].
    vx_matrix : EddyRivasTriMatrix
        Final optimal energy for subsequence [i, j], given i and j form a pair.
    wxi_matrix : EddyRivasTriMatrix
        Optimal energy for [i, j] in the context of a multiloop.
    wxu_matrix : EddyRivasTriMatrix
        "Uncharged" energy for [i, j] from nested-only structures.
    wxc_matrix : EddyRivasTriMatrix
        "Charged" energy for [i, j] from pseudoknotted structures.
    vxu_matrix : EddyRivasTriMatrix
        "Uncharged" energy for paired [i, j] from nested-only structures.
    vxc_matrix : EddyRivasTriMatrix
        "Charged" energy for paired [i, j] from pseudoknotted structures.
    wx_back_ptr : EddyRivasTriBackPointer
        Backpointers for the final wx_matrix.
    vx_back_ptr : EddyRivasTriBackPointer
        Backpointers for the final vx_matrix.
    whx_matrix : SparseGapMatrix
        Energy for a gapped structure on [i..k] and [l..j], with i,j,k,l undetermined.
    vhx_matrix : SparseGapMatrix
        Energy for a gapped structure where (i,j) and (k,l) are both paired.
    yhx_matrix : SparseGapMatrix
        Energy for a gapped structure where (k,l) is paired, (i,j) is undetermined.
    zhx_matrix : SparseGapMatrix
        Energy for a gapped structure where (i,j) is paired, (k,l) is undetermined.
    whx_back_ptr : SparseGapBackptr
        Backpointers for the whx_matrix.
    vhx_back_ptr : SparseGapBackptr
        Backpointers for the vhx_matrix.
    yhx_back_ptr : SparseGapBackptr
        Backpointers for the yhx_matrix.
    zhx_back_ptr : SparseGapBackptr
        Backpointers for the zhx_matrix.
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
    """
    Initializes and returns a new EddyRivasFoldState object.

    This factory function creates an instance of the `EddyRivasFoldState`
    and sets the required base conditions for all its dynamic programming
    matrices according to the algorithm's specifications.

    Parameters
    ----------
    n : int
        The length of the RNA sequence.

    Returns
    -------
    EddyRivasFoldState
        A fully initialized state object ready for the DP algorithm.
    """
    fold_state = EddyRivasFoldState(
        seq_len=n,

        # --- Matrix Instantiation ---
        # Instantiate 2D triangular matrices for non-gapped structures.
        wx_matrix=EddyRivasTriMatrix(n),
        vx_matrix=EddyRivasTriMatrix(n),
        wxi_matrix=EddyRivasTriMatrix(n),
        wxu_matrix=EddyRivasTriMatrix(n),  # Stores nested-only energies for WX
        wxc_matrix=EddyRivasTriMatrix(n),  # Stores pseudoknotted energies for WX
        vxu_matrix=EddyRivasTriMatrix(n),  # Stores nested-only energies for VX
        vxc_matrix=EddyRivasTriMatrix(n),  # Stores pseudoknotted energies for VX

        # Instantiate 2D triangular backpointer matrices.
        wx_back_ptr=EddyRivasTriBackPointer(n),
        vx_back_ptr=EddyRivasTriBackPointer(n),

        # Instantiate 4D sparse matrices for gapped structures.
        whx_matrix=SparseGapMatrix(n),
        vhx_matrix=SparseGapMatrix(n),
        yhx_matrix=SparseGapMatrix(n),
        zhx_matrix=SparseGapMatrix(n),

        # Instantiate 4D sparse backpointer matrices.
        whx_back_ptr=SparseGapBackptr(n),
        vhx_back_ptr=SparseGapBackptr(n),
        yhx_back_ptr=SparseGapBackptr(n),
        zhx_back_ptr=SparseGapBackptr(n),

    )

    # ---------- Matrix Initialization (Base Conditions) ----------
    # Set the initial energies for all subsequences of length 1 (i.e., a single base).
    # These values correspond to the base cases of the DP recursions.
    for i in range(n):
        # WX(i, i) = 0.0: The energy of a single, unpaired nucleotide is zero.
        fold_state.wx_matrix.set(i, i, 0.0)
        fold_state.vx_matrix.set(i, i, math.inf)
        fold_state.wxi_matrix.set(i, i, 0.0)

        # WXC(i, i) is also set to 0.0; a single base has no structure and thus no pseudoknot penalty.
        fold_state.wxu_matrix.set(i, i, 0.0)

        # VX(i, i) = +inf: A single nucleotide cannot form a base pair with itself.
        fold_state.wxc_matrix.set(i, i, 0.0)
        fold_state.vxu_matrix.set(i, i, math.inf)
        fold_state.vxc_matrix.set(i, i, math.inf)

    # Note: Gap matrices (WHX, VHX, etc.) are sparse and do not require explicit initialization.
    # Their `get()` method is designed to return +infinity for any (i,j,k,l) entry that
    # has not been explicitly set, which correctly represents an invalid or un-calculated state.

    return fold_state