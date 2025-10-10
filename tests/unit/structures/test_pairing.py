"""
Unit tests for the `Pair` data structure.

This module validates the behavior of the `Pair` dataclass, which is a
fundamental component for representing a base pair in an RNA structure.
The tests cover its computed properties, methods, and essential dataclass
features like immutability and hashability.
"""
import pytest
from dataclasses import FrozenInstanceError

from rna_pk_fold.structures.pairing import Pair


def test_pair_properties_span_and_loop_len():
    """
    Validates the `span` and `loop_len` computed properties.

    - `span`: The total number of nucleotides from `i` to `j`, inclusive.
    - `loop_len`: The number of unpaired nucleotides enclosed by the pair (i, j).
    """
    # Create a pair from index 2 to 6.
    base_pair = Pair(base_i=2, base_j=6)
    # The span should be j - i + 1 = 6 - 2 + 1 = 5.
    assert base_pair.span == 5
    # The loop length should be j - i - 1 = 6 - 2 - 1 = 3.
    assert base_pair.loop_len == 3


def test_pair_properties_min_loop_edge_case():
    """
    Tests the computed properties for the edge case of adjacent bases.
    This represents a stack with no unpaired bases in between (a loop of length 0).
    """
    p = Pair(0, 1)
    # Span should be 1 - 0 + 1 = 2.
    assert p.span == 2
    # Loop length should be 1 - 0 - 1 = 0.
    assert p.loop_len == 0


def test_pair_as_tuple_returns_coordinates():
    """
    Ensures the `as_tuple` method correctly returns the (i, j) indices.
    This provides a convenient way to get the raw coordinates for functions
    that expect a simple tuple.
    """
    base_pair = Pair(base_i=10, base_j=20)
    assert base_pair.as_tuple() == (10, 20)


def test_pair_is_frozen_and_slotted():
    """
    Confirms that the `Pair` dataclass is immutable and memory-efficient.

    - `frozen=True`: Guarantees that a `Pair` object cannot be changed after
      creation, which is critical for data integrity in folding algorithms.
    - `slots=True`: Reduces the memory footprint of each `Pair` instance, which is
      important when representing large RNA structures.
    """
    base_pair = Pair(base_i=1, base_j=3)

    # Test for immutability: attempting to change a field should raise an error.
    with pytest.raises(FrozenInstanceError):
        base_pair.base_i = 2

    # Test for slotted behavior: adding a new, undeclared attribute should fail.
    with pytest.raises((AttributeError, TypeError)):
        setattr(base_pair, "new_field", 123)


def test_pair_hashable_and_equality():
    """
    Validates that `Pair` objects are hashable and have correct equality semantics.
    Because the dataclass is frozen, its instances can be used in sets and as
    dictionary keys, which is essential for many structure processing algorithms.
    """
    p1 = Pair(5, 9)
    p2 = Pair(5, 9) # Identical to p1.
    p3 = Pair(5, 8) # Different from p1 and p2.

    # Test equality: Pairs with the same coordinates should be equal.
    assert p1 == p2
    assert p1 != p3

    # Test hashability by adding the pairs to a set.
    s = {p1, p2, p3}
    assert p1 in s and p2 in s and p3 in s
    # The set should only contain two elements, as p1 and p2 are duplicates.
    assert len(s) == 2
