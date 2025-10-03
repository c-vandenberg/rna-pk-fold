from __future__ import annotations

from typing import Generic, TypeVar, List, Tuple, Iterator

T = TypeVar("T")


class CoreTriMatrix(Generic[T]):
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

    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape as (N, N)."""
        return self._seq_len, self._seq_len

    def _offset(self, base_i: int, base_j: int) -> int:
        if base_i < 0 or base_j < 0 or base_i >= self._seq_len or base_j >= self._seq_len or base_j < base_i:
            raise IndexError(f"TriMatrix invalid index: (i={base_i}, j={base_j}) for N={self._seq_len}")
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

    def iter_upper_indices(self) -> Iterator[Tuple[int, int]]:
        """Yield all valid (i, j) with i <= j in row-major order."""
        n = self._seq_len
        for i in range(n):
            for j in range(i, n):
                yield i, j