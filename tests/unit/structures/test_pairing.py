import pytest
from dataclasses import FrozenInstanceError

from rna_pk_fold.structures.pairing import Pair


def test_pair_properties_span_and_loop_len():
    """
    Validate `span` and `loop_len` computed properties.

    For (i,j)=(2,6): span = 6-2+1 = 5; loop_len = 6-2-1 = 3.
    """
    base_pair = Pair(base_i=2, base_j=6)
    assert base_pair.span == 5
    assert base_pair.loop_len == 3


def test_pair_properties_min_loop_edge_case():
    """
    Edge case: adjacent bases (i=0, j=1) → span=2, loop_len=0.
    """
    p = Pair(0, 1)
    assert p.span == 2
    assert p.loop_len == 0


def test_pair_as_tuple_returns_coordinates():
    """
    Ensure `as_tuple` returns the (i, j) indices.
    """
    base_pair = Pair(base_i=10, base_j=20)
    assert base_pair.as_tuple() == (10, 20)


def test_pair_is_frozen_and_slotted():
    """
    Check frozen dataclass and slots behavior:
    - assigning to a field raises FrozenInstanceError
    - adding a new attribute raises AttributeError/TypeError
    """
    base_pair = Pair(base_i=1, base_j=3)

    with pytest.raises(FrozenInstanceError):
        base_pair.base_i = 2

    with pytest.raises((AttributeError, TypeError)):
        setattr(base_pair, "new_field", 123)


def test_pair_hashable_and_equality():
    """
    Frozen dataclasses are hashable; identical coords are equal.
    """
    p1 = Pair(5, 9)
    p2 = Pair(5, 9)
    p3 = Pair(5, 8)

    # hash/equality semantics
    assert p1 == p2
    assert p1 != p3

    s = {p1, p2, p3}
    assert p1 in s and p2 in s and p3 in s
    assert len(s) == 2  # p1 and p2 collapse to one entry
