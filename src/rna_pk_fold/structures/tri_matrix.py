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
        base_i : int
            The row index (0-based).
        base_j : int
            The column index (0-based).

        Returns
        -------
        T
            The value stored at the specified cell.
        """
        return self._rows[base_i][self._offset(base_i, base_j)]

    def set(self, base_i: int, base_j: int, value: T) -> None:
        """
        Sets the `value` at cell `(i, j)`.

        Parameters
        ----------
        base_i : int
            The row index (0-based).
        base_j : int
            The column index (0-based).
        value : T
            The value to store in the cell.
        """
        self._rows[base_i][self._offset(base_i, base_j)] = value

    def safe_range(self) -> range:
        """
        Range helper for i. For each i, valid j are i..N-1.
        Useful for nested loops: for i in tri.safe_range(): for j in range(i, N): ...
        """
        return range(self._seq_len)

    def iter_upper_indices(self) -> Iterator[Tuple[int, int]]:
        """
        Yields all valid `(i, j)` index tuples in the upper triangle.

        This iterator proceeds in row-major order, yielding `(i, j)` for all
        `j >= i`.

        Yields
        ------
        Iterator[Tuple[int, int]]
            An iterator over the `(i, j)` index tuples.
        """
        n = self._seq_len
        for i in range(n):
            for j in range(i, n):
                yield i, j


@dataclass(slots=True)
class EddyRivasTriMatrix:
    """
    A sparse, dictionary-based triangular matrix for Eddy-Rivas DP tables.

    This class provides an `(i, j)`-indexed matrix that defaults to infinity.
    It can be optionally converted to a dense NumPy array for performance-critical
    access patterns after being populated.

    Attributes
    ----------
    n : int
        The length of the sequence (N).
    data : Dict[OuterSpan, float]
        The primary sparse storage, mapping `(i, j)` tuples to float energy values.
    _dense : Optional[np.ndarray]
        A dense NumPy array mirror of the matrix, created on demand.
    _is_dense_enabled : bool
        A flag indicating whether the dense mirror has been created.
    """
    n: int
    data: Dict[Outer, float] = field(default_factory=dict)

    _dense: Optional[np.ndarray] = field(default=None, repr=False)
    _is_dense_enabled: bool = field(default=False, repr=False)

    def enable_dense(self) -> None:
        """
        Converts the internal storage from a sparse dictionary to a dense NumPy array.

        This method allocates a full NxN NumPy array, initializes it with infinity,
        and copies all existing values from the sparse dictionary into it.
        Subsequent `get` and `set` calls will use the faster NumPy array.
        """
        if self._is_dense_enabled:
            return

        self._dense = np.full((self.n, self.n), np.inf, dtype=np.float64)
        for (i, j), v in self.data.items():
            if 0 <= i <= j < self.n:
                self._dense[i, j] = v
        self._is_dense_enabled = True

    def as_dense(self) -> np.ndarray:
        if not self._is_dense_enabled:
            self.enable_dense()
        # mypy: _dense is not None here
        return self._dense  # type: ignore[return-value]

    def get(self, i: int, j: int) -> float:
        """
        Retrieves the energy value for the given coordinates `(i, j)`.

        Parameters
        ----------
        i, j : int
            The indices of the cell.

        Returns
        -------
        float
            The stored energy value. Defaults to `np.inf` for unassigned cells
            or invalid indices. Returns 0.0 for empty segments `(i = j + 1)`.
        """
        if i > j:
            return 0.0 if i == j + 1 else INF  # empty segment convenience
        if i < 0 or j >= self.n:
            return INF
        if self._is_dense_enabled:
            return float(self._dense[i, j])  # type: ignore[index]
        return self.data.get((i, j), INF)

    def set(self, i: int, j: int, value: float) -> None:
        """
        Sets the energy `value` for the given coordinates `(i, j)`.

        The value is written to both the sparse dictionary and the dense array
        (if it has been enabled) to keep them synchronized.

        Parameters
        ----------
        i, j : int
            The indices of the cell.
        value : float
            The energy value to store.
        """
        self.data[(i, j)] = value
        if self._is_dense_enabled:
            self._dense[i, j] = value  # type: ignore[index]


@dataclass(slots=True)
class EddyRivasTriBackPointer:
    """
    A sparse, dictionary-based triangular matrix for WX/VX backpointers.

    This class provides storage for backpointer objects corresponding to the
    `EddyRivasTriMatrix`. It uses a dictionary for sparse storage, as many
    cells may not have a backpointer.

    Attributes
    ----------
    n : int
        The length of the sequence (N).
    data : Dict[OuterSpan, Any]
        The sparse dictionary mapping `(i, j)` tuples to backpointer objects.
    """
    n: int
    data: Dict[Outer, Any] = field(default_factory=dict)  # store RivasEddyBackPointer

    def get(self, i: int, j: int):
        """
        Retrieves the backpointer object for the given coordinates `(i, j)`.

        Parameters
        ----------
        i, j : int
            The indices of the cell.

        Returns
        -------
        Any | None
            The stored backpointer object, or `None` if the indices are invalid
            or no backpointer has been set.
        """
        if i > j or i < 0 or j >= self.n:
            return None
        return self.data.get((i, j))

    def set(self, i: int, j: int, value) -> None:
        """
        Sets the `value` (a backpointer object) for the given coordinates `(i, j)`.

        Parameters
        ----------
        i, j : int
            The indices of the cell.
        value : Any
            The backpointer object to store.
        """
        self.data[(i, j)] = value
