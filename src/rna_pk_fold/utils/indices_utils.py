from __future__ import annotations
from typing import Optional, Tuple


def safe_base(seq: str, index: int) -> Optional[str]:
    """
    Safely retrieves a character from a sequence by its index.

    This function performs bounds checking before accessing the sequence. If the
    index is valid, it returns the character; otherwise, it returns None.

    Parameters
    ----------
    seq : str
        The sequence (e.g., an RNA sequence).
    index : int
        The 0-based index of the character to retrieve.

    Returns
    -------
    Optional[str]
        The character at the specified index, or `None` if the index is out
        of bounds.
    """
    return seq[index] if 0 <= index < len(seq) else None


def canonical_pair(index_i: int, index_j: int) -> Tuple[int, int]:
    """
    Ensures a pair of indices is in canonical order (i <= j).

    This utility is used to standardize the representation of a base pair,
    ensuring that the smaller index always comes first.

    Parameters
    ----------
    index_i : int
        The first index of the pair.
    index_j : int
        The second index of the pair.

    Returns
    -------
    Tuple[int, int]
        A tuple `(min(i, j), max(i, j))`.
    """
    return (index_i, index_j) if index_i <= index_j else (index_j, index_i)


def is_interval_valid(i: int, j: int, n: int) -> bool:
    """
    Checks if an interval `[i, j]` is valid within a sequence of length `n`.

    An interval is considered valid if `0 <= i <= j < n`.

    Parameters
    ----------
    i : int
        The 5' start index of the interval.
    j : int
        The 3' end index of the interval.
    n : int
        The total length of the sequence.

    Returns
    -------
    bool
        `True` if the interval is valid, `False` otherwise.
    """
    return 0 <= i <= j < n


def get_default_split_point(i: int, j: int, fallback: int | None = None) -> int:
    """
    Provides a default split point for an interval `[i, j]`.

    This is a convenience function used in traceback routines. If a specific
    split point (`fallback`) is not provided, it calculates the midpoint of the
    interval.

    Parameters
    ----------
    i : int
        The 5' start index of the interval.
    j : int
        The 3' end index of the interval.
    fallback : Optional[int], optional
        A specific split point to use if available. If `None`, the midpoint
        is calculated, by default `None`.

    Returns
    -------
    int
        The chosen split point index.
    """
    return ((i + j) // 2) if fallback is None else fallback
