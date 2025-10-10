"""
Unit tests for coaxial stacking energy utility functions.

This module validates helper functions from `energy_pk_utils` that are used to
construct keys and look up energies for coaxial stacking calculations in the
context of pseudoknotted structures.
"""
import math

from rna_pk_fold.utils.energy_pk_utils import (
    coax_pair_key,
    coax_energy_for_join,
)


# ------------------------------
# coax_pair_key
# ------------------------------
def test_coax_pair_key_in_range_and_boundaries():
    """
    Tests that `coax_pair_key` correctly extracts a two-character string
    representing a base pair from a sequence using valid indices.
    """
    # Define a sequence for testing.
    seq = "AUGC"  # indices: 0:A, 1:U, 2:G, 3:C

    # Test standard in-range extraction.
    assert coax_pair_key(seq, 0, 1) == "AU"
    assert coax_pair_key(seq, 2, 3) == "GC"

    # Test boundary conditions to ensure indices are inclusive.
    assert coax_pair_key(seq, 0, 3) == "AC"
    # Same index is allowed; the function simply concatenates the bases.
    assert coax_pair_key(seq, 3, 3) == "CC"


def test_coax_pair_key_out_of_range_returns_none():
    """
    Tests that `coax_pair_key` gracefully handles out-of-range indices
    by returning `None`.
    """
    seq = "AUGC"  # len = 4

    # A negative index should be out of range.
    assert coax_pair_key(seq, -1, 1) is None
    # An index equal to the sequence length is out of range.
    assert coax_pair_key(seq, 0, 4) is None
    # Both indices being out of range should also return None.
    assert coax_pair_key(seq, -2, 7) is None


# ------------------------------
# coax_energy_for_join
# ------------------------------
def test_coax_energy_for_join_direct_key_hit():
    """
    Tests the primary success case where the energy key is found directly
    in the provided table.
    """
    # The left pair (0,1) -> "AU", and the right pair (2,3) -> "GC".
    seq = "AUGC"
    # The key ("AU", "GC") is present in the table.
    pairs_tbl = {("AU", "GC"): -0.6}  # Stabilizing energy.

    e = coax_energy_for_join(seq, (0, 1), (2, 3), pairs_tbl)
    assert math.isclose(e, -0.6, rel_tol=1e-12)


def test_coax_energy_for_join_reversed_key_hit():
    """
    Tests the symmetric lookup capability of the function.
    If the direct key is not found, the function should try the reversed key.
    """
    seq = "AUGC"
    # The table only contains the reversed key ("GC", "AU").
    pairs_tbl = {("GC", "AU"): -0.8}

    # The function should still find the energy by checking the reversed key.
    e = coax_energy_for_join(seq, (0, 1), (2, 3), pairs_tbl)
    assert math.isclose(e, -0.8, rel_tol=1e-12)


def test_coax_energy_for_join_missing_key_defaults_zero():
    """
    Tests the default behavior when a key is not found in either the direct
    or reversed order. The function should return 0.0.
    """
    seq = "AUGC"
    pairs_tbl = {("AU", "UA"): -0.3} # This table does not contain the required key.

    # The function looks for ("AU","GC") or ("GC","AU"), neither of which are in the table.
    e = coax_energy_for_join(seq, (0, 1), (2, 3), pairs_tbl)
    # The result should be the default value of 0.0.
    assert math.isclose(e, 0.0, rel_tol=1e-12)


def test_coax_energy_for_join_invalid_indices_return_zero():
    """
    Tests that the function returns 0.0 if any of the input indices are invalid.
    This ensures graceful failure without raising an error if out-of-range
    coordinates are passed.
    """
    seq = "AUGC"
    pairs_tbl = {("AU", "GC"): -1.0}

    # Test with the left pair's indices out of range.
    e1 = coax_energy_for_join(seq, (-1, 1), (2, 3), pairs_tbl)
    # Test with the right pair's indices out of range.
    e2 = coax_energy_for_join(seq, (0, 1), (2, 4), pairs_tbl)
    # Test with both pairs' indices out of range.
    e3 = coax_energy_for_join(seq, (-1, 5), (9, 9), pairs_tbl)

    # In all cases of invalid indices, the energy should be 0.0.
    assert e1 == e2 == e3 == 0.0
