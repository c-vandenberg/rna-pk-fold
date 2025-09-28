from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pair:
    """
    Immutable (i, j) index pair used to represent a candidate/base-paired span.

    Parameters
    ----------
    base_i : int
        Left index (0-based).
    base_j : int
        Right index (0-based), must satisfy j > i in valid uses.

    Notes
    -----
    - `span` is the inclusive length (j - i + 1).
    - `loop_len` is the number of unpaired nts between `i` and j`` (`j - i - 1`).
    """
    base_i: int
    base_j: int

    @property
    def span(self) -> int:
        """
        Inclusive span length.

        Computes the number of positions covered by the base pair,
        including both endpoints.

        Returns
        -------
        int
            The inclusive span length, calculated as ``j - i + 1``.
        """
        return self.base_j - self.base_i + 1

    @property
    def loop_len(self) -> int:
        """
        Length of the enclosed loop.

        Number of unpaired nucleotides between paired indices
        `i` and `j`.

        Returns
        -------
        int
            The loop length, calculated as ``j - i - 1``.
        """
        return self.base_j - self.base_i - 1

    def as_tuple(self) -> tuple[int, int]:
        """
        Pair indices as a tuple.

        Useful for dictionary keys, matrix addressing, or interoperability
        with APIs that expect ``(i, j)`` coordinates.

        Returns
        -------
        tuple[int, int]
            The pair ``(i, j)``.
        """
        return self.base_i, self.base_j
