from __future__ import annotations

from dataclasses import dataclass

from rna_pk_fold.structures import TriMatrix
from rna_pk_fold.folding import BackPointer

__all__ = ["FoldState", "make_fold_state"]


@dataclass(frozen=True, slots=True)
class FoldState:
    """
    Holds all numeric and back-pointer tables for Zuker-style folding.

    Attributes
    ----------
    w_matrix : TriMatrix[float]
        Optimal substructure energy for span i..j (may include unpaired cases
        or bifurcations).
    v_matrix : TriMatrix[float]
        Optimal energy for spans where i pairs with j (pair-closed contributions).
    w_back_ptr : TriMatrix[BackPointer]
        Back-pointers for W matrix cells (i.e. how W[i,j] was derived).
    v_back_ptr : TriMatrix[BackPointer]
        Back-pointers for V matrix cells (i.e. how V[i,j] was derived).
    """
    w_matrix: TriMatrix[float]
    v_matrix: TriMatrix[float]
    w_back_ptr: TriMatrix[BackPointer]
    v_back_ptr: TriMatrix[BackPointer]


def make_fold_state(seq_len: int, init_energy: float = float("inf")) -> FoldState:
    """
    Allocate the folding matrices for a sequence of length N.

    Parameters
    ----------
    seq_len : int
        Sequence length N.
    init_energy : float, optional
        Initial fill value for energy cells (default: +∞), so any real score
        will improve upon it during DP.

    Returns
    -------
    FoldState
        A newly allocated bundle containing matrices W, V, and their parallel
        back-pointer tables.

    Notes
    -----
    - All energy cells are initialized to +∞ (or `init_energy` you pass).
    - All back-pointer cells start as `BackPointer()`.
    - This module only provides storage & indexing; it does not compute energies.
      The “recurrence engine” will later read/write these tables.
    """
    w_matrix = TriMatrix[float](seq_len, init_energy)
    v_matrix = TriMatrix[float](seq_len, init_energy)
    w_back_ptr = TriMatrix[BackPointer](seq_len, BackPointer())
    v_back_ptr = TriMatrix[BackPointer](seq_len, BackPointer())

    return FoldState(
        w_matrix=w_matrix,
        v_matrix=v_matrix,
        w_back_ptr=w_back_ptr,
        v_back_ptr=v_back_ptr,
    )
