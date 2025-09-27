from __future__ import annotations
from typing import Final

from rna_pk_fold.utils.base_utils import normalize_base

# Minimum number of unpaired nucleotides required in a hairpin loop.
# For RNA, most algorithms enforce >= 3.
MIN_HAIRPIN_UNPAIRED: Final[int] = 3

# ---- Pairing rules (RNA) -----------------------------------------------------

# Allowed canonical pairs (including wobble) for RNA.
# Accept both orientations (e.g., "AU" and "UA") for quick membership checks.
_RNA_ALLOWED_PAIRS: Final[frozenset[str]] = frozenset(
    {"AU", "UA", "GC", "CG", "GU", "UG"}
)


def can_pair(base_i: str, base_j: str) -> bool:
    """
    Return True if nucleotides bases `base_i` and `base_j` can base pair
    in RNA.

    We allow canonical Watson–Crick pairs (AU, GC) and GU wobble pairs.

    Parameters
    ----------
    base_i, base_j : str
        Single-character nucleotides, case-insensitive. Expected in {A, U, G, C}.

    Returns
    -------
    bool
        True if (a,b) is in {AU, UA, GC, CG, GU, UG}; False otherwise.
    """
    if not isinstance(base_i, str) or not isinstance(base_j, str):
        return False

    if len(base_i) != 1 or len(base_j) != 1:
        return False

    base_i_norm = normalize_base(base_i)
    base_j_norm = normalize_base(base_j)

    return (base_i_norm + base_j_norm) in _RNA_ALLOWED_PAIRS


def hairpin_size(i: int, j: int) -> int:
    """
    Compute the number of unpaired nucleotides inside a hairpin closed by (i, j).

    For a candidate closing pair at indices i < j, the hairpin loop length is
    `j - i - 1`.

    Parameters
    ----------
    i, j : int
        Zero-based indices with i < j.

    Returns
    -------
    int
        Number of unpaired nucleotides between `i` and `j`.
    """
    return j - i - 1


def is_min_hairpin_size(i: int, j: int, min_unpaired: int = MIN_HAIRPIN_UNPAIRED) -> bool:
    """
    Check whether a candidate closing pair (i, j) satisfies the minimum hairpin size.

    Parameters
    ----------
    i, j : int
        Zero-based indices with i < j.
    min_unpaired : int, optional
        Minimum allowed unpaired nucleotides in the loop. Defaults to 3.

    Returns
    -------
    bool
        True if `j - i - 1 >= min_unpaired`, else False.
    """
    return hairpin_size(i, j) >= min_unpaired
