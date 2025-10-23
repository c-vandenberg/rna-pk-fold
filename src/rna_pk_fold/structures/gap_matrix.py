from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple
import math

Pair = Tuple[int, int]
Hole = Tuple[int, int]
OuterSpan = Tuple[int, int]

INF = math.inf


@dataclass(slots=True)
class SparseGapEnergyMatrix:
    """
    Sparse 4D energy matrix for gapped sub-problems `(i, j : k, l)`.

    This structure stores finite energies for gap matrices used by the
    Rivas–Eddy algorithm (`whx`, `vhx`, `zhx`, `yhx`) in a memory-efficient
    way. Only computed (finite) entries are kept in a nested dictionary
    keyed by outer span and inner hole: `Dict[(i, j), Dict[(k, l), float]]`.

    Attributes
    ----------
    seq_len : int
        Sequence length.
    data : Dict[OuterSpan, Dict[Hole, float]]
        Nested mapping from outer span `(i, j)` to inner hole `(k, l)` to energy.
    """
    seq_len: int
    data: Dict[OuterSpan, Dict[Hole, float]] = field(default_factory=dict)

    def get_energy(self, outer_i: int, outer_j: int, hole_k: int, hole_l: int) -> float:
        """
        Retrieve the energy at coordinates `(outer_i, outer_j : hole_k, hole_l)`.

        Performs basic geometry checks for a valid triangular/gapped configuration
        and returns `inf` for out-of-bounds or missing entries.

        Parameters
        ----------
        outer_i : int
            5' index of the outer span.
        outer_j : int
            3' index of the outer span.
        hole_k : int
            5' index of the inner hole.
        hole_l : int
            3' index of the inner hole.

        Returns
        -------
        float
            Stored energy if present; `inf` otherwise.
        """
        # --- 1. Bounds Checking ---
        # Enforce valid triangular and nested geometry for the indices.
        if (
            outer_i < 0
            or outer_j >= self.seq_len
            or hole_k < outer_i
            or hole_l > outer_j
            or outer_i > outer_j
            or hole_k > hole_l
        ):
            return INF

        # --- 2. Data Retrieval ---
        # Look up the outer span dictionary.
        row = self.data.get((outer_i, outer_j))
        if row is None:
            return INF

        # Look up the inner hole value, defaulting to infinity if not found.
        return row.get((hole_k, hole_l), INF)

    def set_energy(self, outer_i: int, outer_j: int, hole_k: int, hole_l: int, energy: float) -> None:
        """
        Store an energy value at `(outer_i, outer_j : hole_k, hole_l)`.

        Parameters
        ----------
        outer_i : int
            5' index of the outer span.
        outer_j : int
            3' index of the outer span.
        hole_k : int
            5' index of the inner hole.
        hole_l : int
            3' index of the inner hole.
        energy : float
            Energy value to record.
        """
        row = self.data.setdefault((outer_i, outer_j), {})
        row[(hole_k, hole_l)] = energy

    def get_outer_span_map(self, outer_i: int, outer_j: int) -> Dict[Hole, float]:
        """
        Return (and create if missing) the mapping of holes for an outer span `(i, j)`.

        Parameters
        ----------
        outer_i : int
            5' index of the outer span.
        outer_j : int
            3' index of the outer span.

        Returns
        -------
        Dict[Hole, float]
            Dictionary mapping each stored hole `(k, l)` to its energy
        """
        return self.data.setdefault((outer_i, outer_j), {})


@dataclass(slots=True)
class SparseGapBackpointerMatrix:
    """
    Sparse 4D backpointer matrix for gapped subproblems `(i, j : k, l)`.

    Mirrors `SparseGapEnergyMatrix` but stores backpointer objects instead of
    energies, using the same nested dictionary shape:
    `Dict[(i, j), Dict[(k, l), object]]`.

    Attributes
    ----------
    seq_len : int
        Sequence length.
    data : Dict[OuterSpan, Dict[Hole, object]]
        Nested mapping from outer span `(i, j)` to inner hole `(k, l)` to backpointer.
    """
    seq_len: int
    data: Dict[OuterSpan, Dict[Hole, object]] = field(default_factory=dict)

    def get_backpointer(self, outer_i: int, outer_j: int, hole_k: int, hole_l: int):
        """
        Retrieve the backpointer at `(outer_i, outer_j : hole_k, hole_l)`.

        Parameters
        ----------
        outer_i : int
            5' index of the outer span.
        outer_j : int
            3' index of the outer span.
        hole_k : int
            5' index of the inner hole.
        hole_l : int
            3' index of the inner hole.

        Returns
        -------
        object | None
            Backpointer object if present; `None` otherwise.
        """
        return self.data.get((outer_i, outer_j), {}).get((hole_k, hole_l))

    def set_backpointer(self, outer_i: int, outer_j: int, hole_k: int, hole_l: int, backpointer) -> None:
        """
        Store a backpointer at `(outer_i, outer_j : hole_k, hole_l)`.

        Parameters
        ----------
        outer_i : int
            5' index of the outer span.
        outer_j : int
            3' index of the outer span.
        hole_k : int
            5' index of the inner hole.
        hole_l : int
            3' index of the inner hole.
        backpointer : object
            Backpointer object to record.
        """
        self.data.setdefault((outer_i, outer_j), {})[(hole_k, hole_l)] = backpointer
