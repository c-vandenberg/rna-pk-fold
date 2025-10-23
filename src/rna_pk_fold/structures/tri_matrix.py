from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from typing import Generic, TypeVar, List, Tuple, Iterator, Dict, Any, Optional

Pair = Tuple[int, int]
Hole = Tuple[int, int]
OuterSpan = Tuple[int, int]

INF = np.inf

T = TypeVar("T")


class ZuckerTriangularMatrix(Generic[T]):
    """
    Upper-triangular matrix optimized for Zuker-style DP tables.

    This structure stores only cells with `i <= j`, reducing memory by ~50% vs.
    a dense `N x N` array. It exposes a simple 2D matrix interface backed by a
    compact list-of-lists where row `i` has length `N - i`.

    Parameters
    ----------
    seq_len : int
        Sequence length `N` defining the matrix dimensions.
    fill : T
        Initial value used to populate all upper-triangle cells.
    """
    __slots__ = ("_seq_len", "_rows")

    def __init__(self, seq_len: int, fill: T):
        self._seq_len = seq_len
        self._rows: List[List[T]] = [[fill for _ in range(seq_len - i)] for i in range(seq_len)]

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------
    @property
    def size(self) -> int:
        """
        Returns the sequence length N that defines the matrix dimensions.

        Returns
        -------
        int :
            Sequence length `N` (matrix is `N x N`, upper triangle used).
        """
        return self._seq_len

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Returns the matrix shape as a tuple `(N, N)`.

        Returns
        -------
        Tuple[int, int] :
            Matrix shape `(N, N)`.
        """
        return self._seq_len, self._seq_len

    # ---------------------------------------------------------------------
    # Internal indexing helper
    # ---------------------------------------------------------------------
    def _offset(self, i_idx: int, j_idx: int) -> int:
        """
        Compute column offset inside row `i_idx` for upper-triangle storage.

        Parameters
        ----------
        i_idx : int
            Row index (0-based).
        j_idx : int
            Column index (0-based), must satisfy `i_idx <= j_idx < N`.

        Returns
        -------
        int
            Offset within row `i_idx` corresponding to column `j_idx`.

        Raises
        ------
        IndexError
            If indices are out of bounds or violate `i_idx <= j_idx`.
        """
        if i_idx < 0 or j_idx < 0 or i_idx >= self._seq_len or j_idx >= self._seq_len or j_idx < i_idx:
            raise IndexError(f"TriMatrix invalid index: (i={i_idx}, j={j_idx}) for N={self._seq_len}")
        return j_idx - i_idx

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def get_energy(self, i_idx: int, j_idx: int) -> T:
        """
        Retrieve the energy at cell `(i, j)`.

        Parameters
        ----------
        i_idx : int
            The row index (0-based).
        j_idx : int
            The column index (0-based).

        Returns
        -------
        T
            The value stored at the specified cell.
        """
        return self._rows[i_idx][self._offset(i_idx, j_idx)]

    def set_energy(self, i_idx: int, j_idx: int, energy: T) -> None:
        """
        Store an energy at cell `(i, j)`.

        Parameters
        ----------
        i_idx : int
            The row index (0-based).
        j_idx : int
            The column index (0-based).
        energy : T
            The energy value to store in the cell.
        """
        self._rows[i_idx][self._offset(i_idx, j_idx)] = energy

    # ---------------------------------------------------------------------
    # Iteration helpers
    # ---------------------------------------------------------------------
    def valid_index_range(self) -> range:
        """
        Range over valid row indices `i` for the upper triangle.

        Returns
        -------
        range
            Range `0..N-1`. For each `i` in this range, valid columns are `j = i..N-1`.

        Notes
        -----
        Often used in nested loops:

        >>> for i in tri.valid_index_range():
        ...     for j in range(i, tri.size):
        ...         ...
        """
        return range(self._seq_len)

    def iter_upper_triangle_indices(self) -> Iterator[Tuple[int, int]]:
        """
        Iterate all `(i, j)` index pairs with `i <= j`.

        Yields
        ------
        Iterator[Tuple[int, int]]
            `(i, j)` pairs in row-major order.
        """
        n = self._seq_len
        for i in range(n):
            for j in range(i, n):
                yield i, j


@dataclass(slots=True)
class EddyRivasTriangularEnergyMatrix:
    """
    Sparse triangular energy matrix for `(i, j)` spans with optional dense mirror.

    Stores energies for Eddy–Rivas DP tables (e.g., `WX`, `VX`) in a sparse
    dictionary keyed by `(i, j)` and can switch to a dense NumPy array for
    faster access once populated.

    Attributes
    ----------
    seq_len : int
        Sequence length `N`.
    data : Dict[OuterSpan, float]
        Sparse mapping from `(i, j)` to energy.
    _dense : Optional[np.ndarray]
        Dense `N x N` mirror (upper triangle used) created on demand.
    _is_dense_enabled : bool
        Whether the dense mirror is active.
    """
    seq_len: int
    data: Dict[OuterSpan, float] = field(default_factory=dict)

    _dense: Optional[np.ndarray] = field(default=None, repr=False)
    _is_dense_enabled: bool = field(default=False, repr=False)

    def enable_dense_storage(self) -> None:
        """
        Converts the internal storage from a sparse dictionary to a dense NumPy array.

        Allocates an `N x N` array initialized to `inf` and copies all values
        from the sparse dictionary into the dense upper triangle. Subsequent
        reads/writes synchronize with the dense array.
        """
        if self._is_dense_enabled:
            return

        self._dense = np.full((self.seq_len, self.seq_len), np.inf, dtype=np.float64)
        for (i, j), v in self.data.items():
            if 0 <= i <= j < self.seq_len:
                self._dense[i, j] = v
        self._is_dense_enabled = True

    def get_dense_view(self) -> np.ndarray:
        """
        Return a dense NumPy view of the matrix, creating it if necessary.

        Returns
        -------
        np.ndarray
            The `N x N` dense mirror (upper triangle relevant).
        """
        if not self._is_dense_enabled:
            self.enable_dense_storage()
        return self._dense  # type: ignore[return-value]

    def get_energy(self, i: int, j: int) -> float:
        """
        Retrieve the energy at `(i, j)` with triangular/ bounds handling.

        Returns `0.0` for the empty segment convenience case `(i == j + 1)`,
        `inf` for out-of-bounds or unset cells, otherwise the stored energy.

        Parameters
        ----------
        i : int
            5' index of the span.
        j : int
            3' index of the span.

        Returns
        -------
        float
            Energy value, `0.0` for empty segment, or `inf` if unset/invalid.
        """
        if i > j:
            return 0.0 if i == j + 1 else INF  # empty segment convenience
        if i < 0 or j >= self.seq_len:
            return INF
        if self._is_dense_enabled:
            return float(self._dense[i, j])  # type: ignore[index]
        return self.data.get((i, j), INF)

    def set_energy(self, i: int, j: int, value: float) -> None:
        """
        Store an energy at `(i, j)` and keep sparse/dense views in sync.

        Parameters
        ----------
        i : int
            5' index of the span.
        j : int
            3' index of the span.
        value : float
            Energy value to write.
        """
        self.data[(i, j)] = value
        if self._is_dense_enabled:
            self._dense[i, j] = value  # type: ignore[index]


@dataclass(slots=True)
class EddyRivasTriangularBackpointerMatrix:
    """
    Sparse triangular backpointer matrix for `(i, j)` spans.

    Mirrors `TriangularEnergyMatrix` but stores backpointer objects instead of
    energies. Remains sparse since many cells have no backpointer.

    Attributes
    ----------
    seq_len : int
        Sequence length.
    data : Dict[OuterSpan, Any]
        Sparse mapping from `(i, j)` to a backpointer object.
    """
    seq_len: int
    data: Dict[OuterSpan, Any] = field(default_factory=dict)  # store RivasEddyBackPointer

    def get_backpointer(self, i: int, j: int):
        """
        Retrieves the backpointer at coordinates `(i, j)`.

        Parameters
        ----------
        i : int
            5' index of the span.
        j : int
            3' index of the span.

        Returns
        -------
        Any | None
            Stored backpointer object, or `None` if unset or indices invalid.
        """
        if i > j or i < 0 or j >= self.seq_len:
            return None
        return self.data.get((i, j))

    def set_backpointer(self, i: int, j: int, value) -> None:
        """
        Store a backpointer at `(i, j)`.

        Parameters
        ----------
        i : int
            5' index of the span.
        j : int
            3' index of the span.
        value : Any
            Backpointer object to store.
        """
        self.data[(i, j)] = value
