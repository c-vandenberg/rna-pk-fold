from typing import Optional


def normalize_base(base_raw: str) -> str:
    """
    Upper-case a nucleotide base and map T->U so RNA logic can be applied uniformly.

    Parameters
    ----------
    base_raw : str
        Raw single-character nucleotide base.

    Returns
    -------
    str
        Normalized base in {A, U, G, C, N}.
    """
    if not isinstance(base_raw, str):
        return base_raw

    if len(base_raw) != 1:
        return base_raw

    base_norm = base_raw.upper()

    return "U" if base_norm == "T" else base_norm


def dimer_key(seq: str, base_i: int, base_j: int) -> Optional[str]:
    """
    Build the nearest-neighbor stack key "XY/ZW" for pairs (i,j) and (i+1,j-1).

    The key encodes the two adjacent base pairs used in NN stacking lookups:
        - Left dimer (5'→3' along the left strand): X = seq[i], Y = seq[i+1]
        - Right dimer (3'→5' along the right strand): Z = seq[j], W = seq[j-1]

    All bases are normalized to RNA (uppercase; ``T``→``U``).

    Parameters
    ----------
    seq : str
        Nucleotide sequence, case-insensitive.
    base_i : int
        Index of the left base of the outer pair (must satisfy `i + 1 < len(seq)`).
    base_j : int
        Index of the right base of the outer pair (must satisfy `j - 1 >= 0` and `j > i`).

    Returns
    -------
    str or None
        The NN key "XY/ZW" if indices are valid; otherwise `None`.
    """
    if base_i + 1 >= len(seq) or base_j - 1 < 0:
        return None

    left = normalize_base(seq[base_i]) + normalize_base(seq[base_i + 1])
    right = normalize_base(seq[base_j]) + normalize_base(seq[base_j - 1])

    return f"{left}/{right}"


def pair_key(base_a: str, base_b: str) -> str:
    """
    Build a two-letter base-pair key (RNA-normalized).

    Concatenates two single-character nucleotides after normalization
    (uppercase; "T" → "U") to form a key such as "AU" or "GC".

    Parameters
    ----------
    base_a : str
        First nucleotide (single character).
    base_b : str
        Second nucleotide (single character).

    Returns
    -------
    str
        Two-character key representing the pair, e.g., ``"AU"`` or ``"GU"``.
    """
    return normalize_base(base_a) + normalize_base(base_b)
