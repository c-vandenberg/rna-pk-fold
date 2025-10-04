from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Tuple

from rna_pk_fold.structures import CoreTriMatrix
from rna_pk_fold.folding import BackPointer
from rna_pk_fold.folding.rivas_eddy.rivas_eddy_matrices import ReTriMatrix, SparseGapMatrix, SparseGapBackptr

wx_back_ptr: Dict[Tuple[int, int], Tuple[str, Tuple[int, int, int]]]
vx_back_ptr: Dict[Tuple[int, int], Tuple[str, Tuple[int, int, int]]]


@dataclass(frozen=True, slots=True)
class FoldState:
    """
    Holds all numeric and back-pointer tables for Zuker-style folding.

    Attributes
    ----------
    w_matrix : CoreTriMatrix[float]
        Optimal substructure energy for span i..j (may include unpaired cases
        or bifurcations).
    v_matrix : TriMatrix[float]
        Optimal energy for spans where i pairs with j (pair-closed contributions).
    w_back_ptr : TriMatrix[BackPointer]
        Back-pointers for W matrix cells (i.e. how W[i,j] was derived).
    v_back_ptr : TriMatrix[BackPointer]
        Back-pointers for V matrix cells (i.e. how V[i,j] was derived).
    """
    w_matrix: CoreTriMatrix[float]
    v_matrix: CoreTriMatrix[float]
    wm_matrix: CoreTriMatrix[float]
    w_back_ptr: CoreTriMatrix[BackPointer]
    v_back_ptr: CoreTriMatrix[BackPointer]
    wm_back_ptr: CoreTriMatrix[BackPointer]


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
    w_matrix = CoreTriMatrix[float](seq_len, init_energy)
    v_matrix = CoreTriMatrix[float](seq_len, init_energy)
    wm_matrix = CoreTriMatrix[float](seq_len, init_energy)

    w_back_ptr = CoreTriMatrix[BackPointer](seq_len, BackPointer())
    v_back_ptr = CoreTriMatrix[BackPointer](seq_len, BackPointer())
    wm_back_ptr = CoreTriMatrix[BackPointer](seq_len, BackPointer())

    # bBse case for WM diagonals: Zero cost to have an empty interior
    for i in range(seq_len):
        wm_matrix.set(i, i, 0.0)

    return FoldState(
        w_matrix=w_matrix,
        v_matrix=v_matrix,
        wm_matrix=wm_matrix,
        w_back_ptr=w_back_ptr,
        v_back_ptr=v_back_ptr,
        wm_back_ptr=wm_back_ptr
    )


@dataclass(slots=True)
class RivasEddyState:
    """
    Holds the non-gap and gap matrices for the R&E algorithm.
    Values only for scaffolding; Step 12 will fill recurrences.
    """
    n: int

    # Non-gap (triangular)
    wx_matrix: ReTriMatrix
    vx_matrix: ReTriMatrix
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


def make_re_fold_state(n: int) -> RivasEddyState:
    re = RivasEddyState(
        n=n,
        # 2D
        wx_matrix=ReTriMatrix(n),
        vx_matrix=ReTriMatrix(n),
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

    # Gap matrices: we *don’t* prefill O(N^4); instead we implement the
    # collapse identities via helpers below (lazy). We do set illegal holes to +inf
    # by default due to SparseGapMatrix.get() behavior.

    return re