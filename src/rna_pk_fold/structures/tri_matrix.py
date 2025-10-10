from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from typing import Generic, TypeVar, List, Tuple, Iterator, Dict, Any, Optional

Pair = Tuple[int, int]
Hole = Tuple[int, int]
Outer = Tuple[int, int]

INF = np.inf

T = TypeVar("T")


class ZuckerTriMatrix(Generic[T]):
    """
    A memory-efficient, upper-triangular matrix for Zuker-style DP tables.

    This class provides a 2D matrix-like interface but only allocates storage
    for the upper triangle (where `i <= j`), saving nearly half the memory
    compared to a full square matrix. It maps 2D `(i, j)` coordinates to a
    compact 1D list-of-lists representation.
    """
    __slots__ = ("_seq_len", "_rows")

    def __init__(self, seq_len: int, fill: T):
        self._seq_len = seq_len
        self._rows: List[List[T]] = [[fill for _ in range(seq_len - i)] for i in range(seq_len)]

    @property
    def size(self) -> int:
        """Returns the sequence length N that defines the matrix dimensions."""
        return self._seq_len

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns the matrix shape as a tuple `(N, N)`."""
        return self._seq_len, self._seq_len

    def _offset(self, base_i: int, base_j: int) -> int:
        """Calculates the column offset within a row and validates indices."""
        if base_i < 0 or base_j < 0 or base_i >= self._seq_len or base_j >= self._seq_len or base_j < base_i:
            raise IndexError(f"TriMatrix invalid index: (i={base_i}, j={base_j}) for N={self._seq_len}")
        return base_j - base_i

    def get(self, base_i: int, base_j: int) -> T:
        """
        Retrieves the value at cell `(i, j)`.

        Parameters
        ----------
        i : int
            The row index (0-based).
        j : int
            The column index (0-based).

        Returns
        -------
        T
            The value stored at the specified cell.
        """
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

    def iter_upper_indices(self) -> Iterator[Tuple[int, int]]:
        """Yield all valid (i, j) with i <= j in row-major order."""
        n = self._seq_len
        for i in range(n):
            for j in range(i, n):
                yield i, j


@dataclass(slots=True)
class EddyRivasTriMatrix:
    """
    Triangular N x N float matrix with +inf default.
    Only (i <= j) are meaningful. Use get/set.
    """
    n: int
    data: Dict[Outer, float] = field(default_factory=dict)

    _dense: Optional[np.ndarray] = field(default=None, repr=False)
    _dense_enabled: bool = field(default=False, repr=False)

    def enable_dense(self) -> None:
        """Allocate dense NxN mirror and backfill from existing dict entries."""
        if self._dense_enabled:
            return
        self._dense = np.full((self.n, self.n), np.inf, dtype=np.float64)
        for (i, j), v in self.data.items():
            if 0 <= i <= j < self.n:
                self._dense[i, j] = v
        self._dense_enabled = True

    def as_dense(self) -> np.ndarray:
        """Return the dense mirror (allocates it if needed)."""
        if not self._dense_enabled:
            self.enable_dense()
        # mypy: _dense is not None here
        return self._dense  # type: ignore[return-value]

    def get(self, i: int, j: int) -> float:
        if i > j:
            return 0.0 if i == j + 1 else INF  # empty segment convenience
        if i < 0 or j >= self.n:
            return INF
        if self._dense_enabled:
            return float(self._dense[i, j])  # type: ignore[index]
        return self.data.get((i, j), INF)

    def set(self, i: int, j: int, value: float) -> None:
        self.data[(i, j)] = value
        if self._dense_enabled:
            self._dense[i, j] = value  # type: ignore[index]


@dataclass(slots=True)
class EddyRivasTriBackPointer:
    """
    Triangular N x N back-pointer matrix for WX/VX.
    Only (i <= j) are meaningful. Missing entries return None.
    """
    n: int
    data: Dict[Outer, Any] = field(default_factory=dict)  # store RivasEddyBackPointer

    def get(self, i: int, j: int):
        if i > j or i < 0 or j >= self.n:
            return None
        return self.data.get((i, j))

    def set(self, i: int, j: int, value) -> None:
        self.data[(i, j)] = value
