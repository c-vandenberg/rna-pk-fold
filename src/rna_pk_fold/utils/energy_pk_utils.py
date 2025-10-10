from __future__ import annotations
from typing import Optional, Tuple, Mapping
from rna_pk_fold.utils.nucleotide_utils import pair_str


def coax_pair_key(seq: str, index_a: int, index_b: int) -> Optional[str]:
    """
    Safely creates a canonical string representation of a base pair from indices.

    This function checks if the given indices `(index_a, index_b)` are within the
    bounds of the sequence and, if so, returns their canonical pair string
    (e.g., "GC", "AU").

    Parameters
    ----------
    seq : str
        The RNA sequence.
    index_a : int
        The 0-based index of the first base in the pair.
    index_b : int
        The 0-based index of the second base in the pair.

    Returns
    -------
    Optional[str]
        The canonical string for the base pair (e.g., "GC"), or `None` if the
        indices are out of bounds.
    """
    n = len(seq)
    if 0 <= index_a < n and 0 <= index_b < n:
        return pair_str(seq, index_a, index_b)
    return None


def coax_energy_for_join(
    seq: str,
    left_pair: Tuple[int, int],
    right_pair: Tuple[int, int],
    pairs_table: Mapping[Tuple[str, str], float],
) -> float:
    """
    Calculates the coaxial stacking energy between two adjacent base pairs.

    This function looks up the stacking energy for a `left_pair` and a
    `right_pair` from a provided energy table. It handles symmetric lookups,
    meaning it will check for both `(left, right)` and `(right, left)` keys
    in the table.

    Parameters
    ----------
    seq : str
        The RNA sequence.
    left_pair : Tuple[int, int]
        The `(i, j)` indices of the left base pair.
    right_pair : Tuple[int, int]
        The `(k, l)` indices of the right base pair.
    pairs_table : Mapping[Tuple[str, str], float]
        A dictionary mapping a tuple of pair strings (e.g., `("GC", "CG")`)
        to a free energy value in kcal/mol.

    Returns
    -------
    float
        The coaxial stacking energy in kcal/mol. Returns 0.0 if either pair
        is invalid or if no energy is found in the table for the interaction.
    """
    # --- 1. Get Canonical Pair Strings ---
    # Convert the index pairs into their canonical string representations (e.g., "GC").
    left_pair_key = coax_pair_key(seq, *left_pair)
    right_pair_key = coax_pair_key(seq, *right_pair)

    # If either pair is invalid (e.g., due to out-of-bounds indices), there is no stacking energy.
    if left_pair_key is None or right_pair_key is None:
        return 0.0

    # --- 2. Symmetrical Table Lookup ---
    # Look up the energy for the interaction (left_pair, right_pair).
    # If that key is not found, try the reverse order (right_pair, left_pair)
    # to account for symmetric definitions in the parameter file.
    # If neither key is found, default to 0.0 (no energy contribution).
    return pairs_table.get(
        (left_pair_key, right_pair_key),
        pairs_table.get((right_pair_key, left_pair_key), 0.0)
    )
