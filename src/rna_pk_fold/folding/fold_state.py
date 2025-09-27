from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Optional, TypeVar, List, Tuple

T = TypeVar("T")


class TriMatrix(Generic[T]):
    """
    Upper-triangular matrix with O(N^2/2) storage.

    Stores entries only for i <= j (or i < j if you prefer).
    We index with (i, j) in 0-based coordinates and map to a compact row layout:
      row i has length N - i; column j is at offset (j - i).

    Parameters
    ----------
    seq_len : int
        Sequence length N.
    fill : T
        Initial fill value for every valid cell.

    Notes
    -----
    - By construction, only (i, j) with i <= j are addressable.
      If you want strictly paired spans, you can enforce i < j at call sites.
    """

    __slots__ = ("_seq_len", "_rows")

    def __init__(self, seq_len: int, fill: T):
        self._seq_len = seq_len
        self._rows: List[List[T]] = [[fill for _ in range(seq_len - i)] for i in range(seq_len)]

    @property
    def size(self) -> int:
        """Return the underlying sequence length N."""
        return self._seq_len

    def _offset(self, base_i: int, base_j: int) -> int:
        if base_i < 0 or base_j < 0 or base_i >= self._seq_len or base_j >= self._seq_len or base_j < base_i:
            raise IndexError(f"TriMatrix index out of range or invalid (i={base_i}, j={base_i})")
        return base_j - base_i

    def get(self, base_i: int, base_j: int) -> T:
        """Get cell value at (i, j)."""
        return self._rows[base_i][self._offset(base_i, base_j)]

    def set(self, base_i: int, base_j: int, value: T) -> None:
        """Set cell value at (i, j)."""
        self._rows[base_i][self._offset(base_i, base_j)] = value

    def safe_range(self) -> range:
        """
        Range helper for i. For each i, valid j are i..N-1.
        Useful for nested loops: for i in tri.safe_range(): for j in range(i, N): ...
        """
        return range(self._seq_len)


class BacktrackOp(Enum):
    """
    Operation chosen at a cell. This is to be used during traceback to
    track what recurrence case produced the optimal value at a given
    cell.

    NONE            : Not set yet.
    HAIRPIN         : V[i,j] formed a hairpin closed by (i,j).
    STACK           : V[i,j] formed by stacking on V[i+1,j-1].
    INTERNAL        : V[i,j] formed an internal/bulge loop with inner pair (k,l).
    BIFURCATION     : W[i,j] split into W[i,k] + W[k+1,j].
    MULTI_ATTACH    : V/W case where a helix attaches into a multiloop context.
    UNPAIRED_LEFT   : W[i,j] best by leaving i unpaired (use W[i+1,j]).
    UNPAIRED_RIGHT  : W[i,j] best by leaving j unpaired (use W[i,j-1]).
    PSEUDOKNOT      : Placeholder for future pseudoknot logic.
    """
    NONE = auto()
    HAIRPIN = auto()
    STACK = auto()
    INTERNAL = auto()
    BIFURCATION = auto()
    MULTI_ATTACH = auto()
    UNPAIRED_LEFT = auto()
    UNPAIRED_RIGHT = auto()
    PSEUDOKNOT = auto()


@dataclass(frozen=True, slots=True)
class BackPointer:
    """
    Back-pointer describing how a cell's value was derived.

    Parameters
    ----------
    operation : BacktrackOp
        The recurrence operation selected for this cell.
    split_k : Optional[int]
        For bifurcations W[i,j] -> W[i,k] + W[k+1,j], record k.
    inner : Optional[Tuple[int, int]]
        For internal/stack cases, the inner paired indices, e.g., (i+1, j-1) for STACK,
        or (k, l) for INTERNAL loops.
    note : Optional[str]
        Free-form metadata (e.g., “tri-tetra hairpin”, “multi enter”).
    """
    operation: BacktrackOp = BacktrackOp.NONE
    split_k: Optional[int] = None
    inner: Optional[Tuple[int, int]] = None
    note: Optional[str] = None


# ---- Folding state bundle -----------------------------------------------------

@dataclass(frozen=True, slots=True)
class FoldState:
    """
    Holds all numeric and choice tables for Zuker-style folding.

    Attributes
    ----------
    w_matrix : TriMatrix[float]
        Optimal substructure energy for span i..j (may include unpaired cases
        or bifurcations).
    v_matrix : TriMatrix[float]
        Optimal energy for spans where i pairs with j (pair-closed contributions).
    v_backptr : TriMatrix[BackPointer]
        Back-pointers for W matrix cells (i.e. how W[i,j] was derived).
    v_backptr : TriMatrix[BackPointer]
        Back-pointers for V matrix cells (i.e. how V[i,j] was derived).
    """
    w_matrix: TriMatrix[float]
    v_matrix: TriMatrix[float]
    w_backptr: TriMatrix[BackPointer]
    v_backptr: TriMatrix[BackPointer]


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
    - All choice cells start as `Choice(kind=ChoiceKind.NONE)`.
    - This module only provides storage & indexing; it does not compute energies.
      The “recurrence engine” will later read/write these tables.
    """
    w_matrix = TriMatrix[float](seq_len, init_energy)
    v_matrix = TriMatrix[float](seq_len, init_energy)
    w_backptr = TriMatrix[BackPointer](seq_len, BackPointer())
    v_backptr = TriMatrix[BackPointer](seq_len, BackPointer())
    return FoldState(
        w_matrix=w_matrix,
        v_matrix=v_matrix,
        w_backptr=w_backptr,
        v_backptr=v_backptr,
    )
