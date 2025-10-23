from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Tuple

from rna_pk_fold.structures.tri_matrix import EddyRivasTriangularEnergyMatrix, EddyRivasTriangularBackpointerMatrix
from rna_pk_fold.structures.gap_matrix import SparseGapEnergyMatrix, SparseGapBackpointerMatrix

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
    wx_matrix : EddyRivasTriangularEnergyMatrix
        Final optimal energy for subsequence [i, j].
    vx_matrix : EddyRivasTriangularEnergyMatrix
        Final optimal energy for subsequence [i, j], given i and j form a pair.
    wxi_matrix : EddyRivasTriangularEnergyMatrix
        Optimal energy for [i, j] in the context of a multiloop.
    wxu_matrix : EddyRivasTriangularEnergyMatrix
        "Uncharged" energy for [i, j] from nested-only structures.
    wxc_matrix : EddyRivasTriangularEnergyMatrix
        "Charged" energy for [i, j] from pseudoknotted structures.
    vxu_matrix : EddyRivasTriangularEnergyMatrix
        "Uncharged" energy for paired [i, j] from nested-only structures.
    vxc_matrix : EddyRivasTriangularEnergyMatrix
        "Charged" energy for paired [i, j] from pseudoknotted structures.
    wx_back_ptr : EddyRivasTriangularBackpointerMatrix
        Backpointers for the final wx_matrix.
    vx_back_ptr : EddyRivasTriangularBackpointerMatrix
        Backpointers for the final vx_matrix.
    whx_matrix : SparseGapEnergyMatrix
        Energy for a gapped structure on [i..k] and [l..j], with i,j,k,l undetermined.
    vhx_matrix : SparseGapEnergyMatrix
        Energy for a gapped structure where (i,j) and (k,l) are both paired.
    yhx_matrix : SparseGapEnergyMatrix
        Energy for a gapped structure where (k,l) is paired, (i,j) is undetermined.
    zhx_matrix : SparseGapEnergyMatrix
        Energy for a gapped structure where (i,j) is paired, (k,l) is undetermined.
    whx_back_ptr : SparseGapBackpointerMatrix
        Backpointers for the whx_matrix.
    vhx_back_ptr : SparseGapBackpointerMatrix
        Backpointers for the vhx_matrix.
    yhx_back_ptr : SparseGapBackpointerMatrix
        Backpointers for the yhx_matrix.
    zhx_back_ptr : SparseGapBackpointerMatrix
        Backpointers for the zhx_matrix.
    """
    seq_len: int

    # --- Non-gap Matrices (Energies, 2D) ---
    wx_matrix: EddyRivasTriangularEnergyMatrix
    vx_matrix: EddyRivasTriangularEnergyMatrix
    wxi_matrix: EddyRivasTriangularEnergyMatrix
    wxu_matrix: EddyRivasTriangularEnergyMatrix  # uncharged (baseline, nested-only)
    wxc_matrix: EddyRivasTriangularEnergyMatrix  # charged   (has paid Gw at least once)
    vxu_matrix: EddyRivasTriangularEnergyMatrix
    vxc_matrix: EddyRivasTriangularEnergyMatrix

    # --- Non-gap Matrices (Back-pointers, 2D) ---
    wx_back_ptr: EddyRivasTriangularBackpointerMatrix
    vx_back_ptr: EddyRivasTriangularBackpointerMatrix

    # --- Gap Matrices (Energies, 4D) ---
    whx_matrix: SparseGapEnergyMatrix
    vhx_matrix: SparseGapEnergyMatrix
    yhx_matrix: SparseGapEnergyMatrix
    zhx_matrix: SparseGapEnergyMatrix

    # --- Gap Matrices (Back-pointers, 4D) ---
    whx_back_ptr: SparseGapBackpointerMatrix
    vhx_back_ptr: SparseGapBackpointerMatrix
    yhx_back_ptr: SparseGapBackpointerMatrix
    zhx_back_ptr: SparseGapBackpointerMatrix


def init_eddy_rivas_fold_state(seq_len: int) -> EddyRivasFoldState:
    """
    Initializes and returns a new EddyRivasFoldState object.

    This factory function creates an instance of the `EddyRivasFoldState`
    and sets the required base conditions for all its dynamic programming
    matrices according to the algorithm's specifications.

    Parameters
    ----------
    seq_len : int
        The length of the RNA sequence.

    Returns
    -------
    EddyRivasFoldState
        A fully initialized state object ready for the DP algorithm.
    """
    fold_state = EddyRivasFoldState(
        seq_len=seq_len,

        # --- Matrix Instantiation ---
        # Instantiate 2D triangular matrices for non-gapped structures.
        wx_matrix=EddyRivasTriangularEnergyMatrix(seq_len),
        vx_matrix=EddyRivasTriangularEnergyMatrix(seq_len),
        wxi_matrix=EddyRivasTriangularEnergyMatrix(seq_len),
        wxu_matrix=EddyRivasTriangularEnergyMatrix(seq_len),  # Stores nested-only energies for WX
        wxc_matrix=EddyRivasTriangularEnergyMatrix(seq_len),  # Stores pseudoknotted energies for WX
        vxu_matrix=EddyRivasTriangularEnergyMatrix(seq_len),  # Stores nested-only energies for VX
        vxc_matrix=EddyRivasTriangularEnergyMatrix(seq_len),  # Stores pseudoknotted energies for VX

        # Instantiate 2D triangular backpointer matrices.
        wx_back_ptr=EddyRivasTriangularBackpointerMatrix(seq_len),
        vx_back_ptr=EddyRivasTriangularBackpointerMatrix(seq_len),

        # Instantiate 4D sparse matrices for gapped structures.
        whx_matrix=SparseGapEnergyMatrix(seq_len),
        vhx_matrix=SparseGapEnergyMatrix(seq_len),
        yhx_matrix=SparseGapEnergyMatrix(seq_len),
        zhx_matrix=SparseGapEnergyMatrix(seq_len),

        # Instantiate 4D sparse backpointer matrices.
        whx_back_ptr=SparseGapBackpointerMatrix(seq_len),
        vhx_back_ptr=SparseGapBackpointerMatrix(seq_len),
        yhx_back_ptr=SparseGapBackpointerMatrix(seq_len),
        zhx_back_ptr=SparseGapBackpointerMatrix(seq_len),

    )

    # ---------- Matrix Initialization (Base Conditions) ----------
    # Set the initial energies for all subsequences of length 1 (i.e., a single base).
    # These values correspond to the base cases of the DP recursions.
    for i in range(seq_len):
        # WX(i, i) = 0.0: The energy of a single, unpaired nucleotide is zero.
        fold_state.wx_matrix.set_energy(i, i, 0.0)
        fold_state.vx_matrix.set_energy(i, i, math.inf)
        fold_state.wxi_matrix.set_energy(i, i, 0.0)

        # WXC(i, i) is also set to 0.0; a single base has no structure and thus no pseudoknot penalty.
        fold_state.wxu_matrix.set_energy(i, i, 0.0)

        # VX(i, i) = +inf: A single nucleotide cannot form a base pair with itself.
        fold_state.wxc_matrix.set_energy(i, i, 0.0)
        fold_state.vxu_matrix.set_energy(i, i, math.inf)
        fold_state.vxc_matrix.set_energy(i, i, math.inf)

    # Note: Gap matrices (WHX, VHX, etc.) are sparse and do not require explicit initialization.
    # Their `get()` method is designed to return +infinity for any (i,j,k,l) entry that
    # has not been explicitly set, which correctly represents an invalid or un-calculated state.

    return fold_state