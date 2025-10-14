from __future__ import annotations
from typing import Optional
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences import EddyRivasBackPointer

def wx_bp(state: EddyRivasFoldState, i: int, j: int) -> Optional[EddyRivasBackPointer]:
    """
    Retrieves the backpointer from the WX matrix for the span `(i, j)`.

    This is a convenience function that provides a shorthand for accessing the
    `wx_back_ptr` matrix within the main `EddyRivasFoldState` object.

    Parameters
    ----------
    state : EddyRivasFoldState
        The state object containing all filled DP matrices and backpointers.
    i : int
        The 5' index of the subsequence.
    j : int
        The 3' index of the subsequence.

    Returns
    -------
    Optional[EddyRivasBackPointer]
        The backpointer object stored at `WX[i, j]`, or `None` if it does not exist.
    """
    return state.wx_back_ptr.get(i, j)

def whx_bp(state: EddyRivasFoldState, i: int, j: int, k: int, l: int) -> Optional[EddyRivasBackPointer]:
    """
    Retrieves the backpointer from the WHX gap matrix for the coordinates `(i, j, k, l)`.

    This function provides a shorthand for accessing the `whx_back_ptr` matrix.

    Parameters
    ----------
    state : EddyRivasFoldState
        The state object containing all filled DP matrices and backpointers.
    i, j : int
        The indices of the outer span.
    k, l : int
        The indices of the inner hole.

    Returns
    -------
    Optional[EddyRivasBackPointer]
        The backpointer object stored at `WHX[i, j, k, l]`, or `None` if it does not exist.
    """
    return state.whx_back_ptr.get(i, j, k, l)

def yhx_bp(state: EddyRivasFoldState, i: int, j: int, k: int, l: int) -> Optional[EddyRivasBackPointer]:
    """
    Retrieves the backpointer from the YHX gap matrix for the coordinates `(i, j, k, l)`.

    This function provides a shorthand for accessing the `yhx_back_ptr` matrix.

    Parameters
    ----------
    state : EddyRivasFoldState
        The state object containing all filled DP matrices and backpointers.
    i, j : int
        The indices of the outer span.
    k, l : int
        The indices of the inner hole.

    Returns
    -------
    Optional[EddyRivasBackPointer]
        The backpointer object stored at `YHX[i, j, k, l]`, or `None` if it does not exist.
    """
    return state.yhx_back_ptr.get(i, j, k, l)

def zhx_bp(state: EddyRivasFoldState, i: int, j: int, k: int, l: int) -> Optional[EddyRivasBackPointer]:
    """
    Retrieves the backpointer from the ZHX gap matrix for the coordinates `(i, j, k, l)`.

    This function provides a shorthand for accessing the `zhx_back_ptr` matrix.

    Parameters
    ----------
    state : EddyRivasFoldState
        The state object containing all filled DP matrices and backpointers.
    i, j : int
        The indices of the outer span.
    k, l : int
        The indices of the inner hole.

    Returns
    -------
    Optional[EddyRivasBackPointer]
        The backpointer object stored at `ZHX[i, j, k, l]`, or `None` if it does not exist.
    """
    return state.zhx_back_ptr.get(i, j, k, l)

def vhx_bp(state: EddyRivasFoldState, i: int, j: int, k: int, l: int) -> Optional[EddyRivasBackPointer]:
    """
    Retrieves the backpointer from the VHX gap matrix for the coordinates `(i, j, k, l)`.

    This function provides a shorthand for accessing the `vhx_back_ptr` matrix.

    Parameters
    ----------
    state : EddyRivasFoldState
        The state object containing all filled DP matrices and backpointers.
    i, j : int
        The indices of the outer span.
    k, l : int
        The indices of the inner hole.

    Returns
    -------
    Optional[EddyRivasBackPointer]
        The backpointer object stored at `VHX[i, j, k, l]`, or `None` if it does not exist.
    """
    return state.vhx_back_ptr.get(i, j, k, l)
