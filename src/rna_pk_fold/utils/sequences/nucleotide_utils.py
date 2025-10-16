import logging
from typing import Optional

logger = logging.getLogger(__name__)


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
    Build the nearest-neighbor stack key "XY/ZW" for pairs (i,j) and (i+1,j-1) (i.e. two
    adjacent base pairs across the helix).

    The key encodes the two adjacent base pairs used in NN stacking lookups. The convention
    when using Turner tables/sets is to use pair-of-pair keys:
        - Left dimer "XY" (outer closing pair across strands): X = seq[i], Y = seq[j]
        - Right dimer "ZW" (inner pair, reversed across strands) : Z = seq[j-1], W = seq[i+1]

    The final key is "XY/ZW". All bases are normalized to RNA (uppercase; `T`→`U`).

    Example
    -------
    seq = "GGGAAAUCCC", i=0, j=9
    XY = G C -> "GC"
    ZW = C G -> "CG"
    key = "GC/CG"

    Parameters
    ----------
    seq : str
        Nucleotide sequence, case-insensitive.
    base_i : int
        Index of the left base of the outer pair.
    base_j : int
        Index of the right base of the outer pair.

    Returns
    -------
    str or None
        The NN key "XY/ZW" if indices are valid and an inner pair exists
        (i.e. `base_i + 1 <= base_j - 1`); otherwise `None`.
    """
    # Basic bounds and ordering
    if base_i < 0 or base_j >= len(seq) or base_i >= base_j:
        return None

    # A valid inner pair must exist to define a stack step
    if base_i + 1 > base_j - 1:
        return None

    base_x = normalize_base(seq[base_i])
    base_y = normalize_base(seq[base_j])
    base_z = normalize_base(seq[base_j - 1])
    base_w = normalize_base(seq[base_i + 1])

    return f"{base_x}{base_y}/{base_z}{base_w}"


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


def validate_and_normalize_seq(raw_sequence: str) -> str:
    """
    Validates and normalizes an RNA sequence.

    This function strips whitespace, converts the sequence to uppercase,
    replaces 'T' with 'U', and checks for any invalid characters.

    Parameters
    ----------
    raw_sequence : str
        The input RNA sequence string.

    Returns
    -------
    str
        The validated and normalized RNA sequence.

    Raises
    ------
    ValueError
        If the sequence is empty or contains characters other than A, C, G, U, T.
    """
    logger.debug(f"Validating sequence: {raw_sequence[:50]}{'...' if len(raw_sequence) > 50 else ''}")
    # Normalize the sequence: strip whitespace, convert to uppercase, and replace T with U.
    normalized_sequence = raw_sequence.strip().upper().replace("T", "U")

    # Check if the sequence is empty after normalization.
    if not normalized_sequence:
        logger.error("Sequence is empty")
        raise ValueError("Sequence is empty.")

    # Check for any characters that are not in the allowed set (A, C, G, U).
    allowed_bases = set("ACGU")
    invalid_char_indices = [i for i, char in enumerate(normalized_sequence) if char not in allowed_bases]
    if invalid_char_indices:
        pos = invalid_char_indices[0]
        invalid_char = normalized_sequence[pos]
        error_message = f"Invalid character at position {pos} ('{invalid_char}'). Only A,C,G,U (or T) are allowed."
        logger.error(f"Invalid character at position {pos}: '{invalid_char}'")
        raise ValueError(error_message)

    logger.info(f"Sequence validated: length={len(normalized_sequence)}")
    return normalized_sequence


def dangle5_key(nt: str, pair: str) -> str:
    return f"{nt}./{pair}"


def dangle3_key(pair: str, nt: str) -> str:
    return f"{pair}/.{nt}"


def pair_str(seq: str, i: int, j: int) -> str:
    return normalize_base(seq[i]) + normalize_base(seq[j])

