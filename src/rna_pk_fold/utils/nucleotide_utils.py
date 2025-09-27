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


def dimer_key(seq: str, i: int, j: int) -> Optional[str]:
    """
    Build the nearest-neighbor key "XY/ZW" for stacking of (i,j) with (i+1, j-1).

    Left dimer is 5'->3' along the left strand:  seq[i] + seq[i+1]
    Right dimer is 3'->5' along the right strand: seq[j] + seq[j-1]

    Returns None if indices are out of range.
    """
    if i + 1 >= len(seq) or j - 1 < 0:
        return None

    left = normalize_base(seq[i]) + normalize_base(seq[i + 1])
    right = normalize_base(seq[j]) + normalize_base(seq[j - 1])

    return f"{left}/{right}"


def pair_key(a: str, b: str) -> str:
    """
    Return two-letter key like 'AU' with RNA normalization.
    """
    return normalize_base(a) + normalize_base(b)
