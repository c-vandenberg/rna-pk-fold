from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple
import math

Pair = Tuple[int, int]
Hole = Tuple[int, int]
Outer = Tuple[int, int]

INF = math.inf


@dataclass(slots=True)
class SparseGapMatrix:
    """
    A memory-efficient, sparse representation of a 4D gap matrix.

    This data structure is used for the `whx`, `vhx`, `zhx`, and `yhx` matrices
    in the Eddy-Rivas algorithm. Since most combinations of `(i, j, k, l)` are
    invalid or have infinite energy, a dense 4D array would be prohibitively
    large. This class uses a nested dictionary `Dict[(i,j), Dict[(k,l), float]]`
    to store only the finite, calculated energy values.

    Attributes
    ----------
    n : int
        The length of the RNA sequence.
    data : Dict[OuterSpan, Dict[InnerHole, float]]
        The underlying nested dictionary storing the sparse matrix data.
    """
    n: int
    data: Dict[Outer, Dict[Hole, float]] = field(default_factory=dict)

    def get(self, i: int, j: int, k: int, l: int) -> float:
        # --- 1. Bounds Checking ---
        # Enforce valid triangular and nested geometry for the indices.
        if i < 0 or j >= self.n or k < i or l > j or i > j or k > l:
            return INF

        # --- 2. Data Retrieval ---
        # Look up the outer span dictionary.
        row = self.data.get((i, j))
        if row is None:
            return INF

        # Look up the inner hole value, defaulting to infinity if not found.
        return row.get((k, l), INF)

    def set(self, i: int, j: int, k: int, l: int, value: float) -> None:
        """
       Sets the energy `value` for the given 4D coordinates `(i, j, k, l)`.

       Parameters
       ----------
       i, j : int
           The indices of the outer span.
       k, l : int
           The indices of the inner hole.
       value : float
           The energy value to store.
       """
        row = self.data.setdefault((i, j), {})
        row[(k, l)] = value

    def row(self, i: int, j: int) -> Dict[Hole, float]:
        """
        Retrieves the entire dictionary of inner holes for a given outer span `(i, j)`.

        Parameters
        ----------
        i, j : int
            The indices of the outer span.

        Returns
        -------
        Dict[InnerHole, float]
            A dictionary mapping all stored `(k, l)` holes to their energy
            values for the given outer span `(i, j)`.
        """
        return self.data.setdefault((i, j), {})


@dataclass(slots=True)
class SparseGapBackptr:
    """
    A sparse storage for backpointers corresponding to a 4D gap matrix.

    This class mirrors the structure of `SparseGapMatrix` but is designed to
    store backpointer objects instead of float energy values.

    Attributes
    ----------
    n : int
        The length of the RNA sequence.
    data : Dict[OuterSpan, Dict[InnerHole, object]]
        The underlying nested dictionary storing the sparse backpointer data.
    """
    n: int
    data: Dict[Outer, Dict[Hole, object]] = field(default_factory=dict)

    def get(self, i: int, j: int, k: int, l: int):
        """
       Retrieves the backpointer object for the given 4D coordinates.

       Parameters
       ----------
       i, j : int
           The indices of the outer span.
       k, l : int
           The indices of the inner hole.

       Returns
       -------
       object | None
           The stored backpointer object, or `None` if no backpointer has
           been set for these coordinates.
       """
        return self.data.get((i, j), {}).get((k, l))

    def set(self, i: int, j: int, k: int, l: int, back_pointer) -> None:
        """
        Sets the `back_pointer` object for the given 4D coordinates.

        Parameters
        ----------
        i, j : int
            The indices of the outer span.
        k, l : int
            The indices of the inner hole.
        back_pointer : object
            The backpointer object to store.
        """
        self.data.setdefault((i, j), {})[(k, l)] = back_pointer
