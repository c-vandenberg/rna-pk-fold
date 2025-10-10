from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

from rna_pk_fold.structures import ZuckerTriMatrix
from rna_pk_fold.folding.zucker.zucker_back_pointer import ZuckerBackPointer

wx_back_ptr: Dict[Tuple[int, int], Tuple[str, Tuple[int, int, int]]]
vx_back_ptr: Dict[Tuple[int, int], Tuple[str, Tuple[int, int, int]]]


@dataclass(frozen=True, slots=True)
class ZuckerFoldState:
    """
    Holds all DP matrices for a Zuker-style nested RNA folding algorithm.

    This immutable and memory-efficient data structure serves as a container
    for the energy and backpointer matrices required by the folding algorithm.
    It bundles the related matrices together for easy management and passing
    between functions.

    Attributes
    ----------
    w_matrix : ZuckerTriMatrix[float]
        The main energy matrix. W[i, j] stores the minimum free energy for the
        subsequence from `i` to `j`, considering all possible nested structures
        (unpaired, paired, bifurcated).
    v_matrix : ZuckerTriMatrix[float]
        The helix-closing energy matrix. V[i, j] stores the minimum free
        energy for the subsequence from `i` to `j`, with the constraint that
        `i` and `j` must form a base pair.
    wm_matrix : ZuckerTriMatrix[float]
        The multiloop energy matrix. WM[i, j] stores the minimum free energy
        for a subsequence from `i` to `j` that forms part of a multiloop.
    w_back_ptr : ZuckerTriMatrix[ZuckerBackPointer]
        Backpointers for the `w_matrix`, recording the recursion rule used to
        achieve the optimal energy at each cell.
    v_back_ptr : ZuckerTriMatrix[ZuckerBackPointer]
        Backpointers for the `v_matrix`.
    wm_back_ptr : ZuckerTriMatrix[ZuckerBackPointer]
        Backpointers for the `wm_matrix`.
    """
    w_matrix: ZuckerTriMatrix[float]
    v_matrix: ZuckerTriMatrix[float]
    wm_matrix: ZuckerTriMatrix[float]
    w_back_ptr: ZuckerTriMatrix[ZuckerBackPointer]
    v_back_ptr: ZuckerTriMatrix[ZuckerBackPointer]
    wm_back_ptr: ZuckerTriMatrix[ZuckerBackPointer]


def make_fold_state(seq_len: int, init_energy: float = float("inf")) -> ZuckerFoldState:
    """
    Allocates and initializes the folding matrices for a Zuker-style algorithm.

    This factory function creates all the necessary dynamic programming matrices
    (W, V, WM) and their corresponding backpointer matrices for a sequence of a
    given length. It sets the initial base conditions required by the algorithm.

    Parameters
    ----------
    seq_len : int
        The length of the RNA sequence (N).
    init_energy : float, optional
        The value to which all energy cells are initialized. Defaults to
        positive infinity, ensuring that any calculated finite energy will
        be selected as the minimum.

    Returns
    -------
    ZuckerFoldState
        A new state object containing the initialized matrices.

    Notes
    -----
    - All backpointer cells are initialized with a default `ZuckerBackPointer`.
    - The base case for the multiloop matrix `WM[i, i]` is set to `0.0`,
      representing the zero energy cost of an empty interior segment of a
      multiloop.
    """
    # Allocate the three main energy matrices (W, V, WM) as triangular matrices,
    # filled with the initial energy value (infinity by default).
    w_matrix = ZuckerTriMatrix[float](seq_len, init_energy)
    v_matrix = ZuckerTriMatrix[float](seq_len, init_energy)
    wm_matrix = ZuckerTriMatrix[float](seq_len, init_energy)

    # Allocate the corresponding backpointer matrices, filling them with default, empty backpointers.
    w_back_ptr = ZuckerTriMatrix[ZuckerBackPointer](seq_len, ZuckerBackPointer())
    v_back_ptr = ZuckerTriMatrix[ZuckerBackPointer](seq_len, ZuckerBackPointer())
    wm_back_ptr = ZuckerTriMatrix[ZuckerBackPointer](seq_len, ZuckerBackPointer())

    # Initialize the base case for the WM (multiloop) matrix.
    # The diagonal WM[i, i] represents an empty segment within a multiloop,
    # which has an energy cost of 0.0.
    for i in range(seq_len):
        wm_matrix.set(i, i, 0.0)

    return ZuckerFoldState(
        w_matrix=w_matrix,
        v_matrix=v_matrix,
        wm_matrix=wm_matrix,
        w_back_ptr=w_back_ptr,
        v_back_ptr=v_back_ptr,
        wm_back_ptr=wm_back_ptr
    )
