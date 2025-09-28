import pytest
from dataclasses import FrozenInstanceError

from rna_pk_fold.rules.pairing import Pair


def test_pair_properties_span_and_loop_len():
    """
    Validate `span` and `loop_len` computed properties.

    Expected
    --------
    - For (i,j)=(2,6): `span` == 5 and `loop_len` == 3.

    Notes
    -----
    For a pair (i, j):
    - span = j - i + 1
    - loop_len = j - i - 1
    """
    base_pair = Pair(base_i=2, base_j=6)
    assert base_pair.span == 5         # 6 - 2 + 1
    assert base_pair.loop_len == 3     # 6 - 2 - 1


def test_pair_as_tuple_returns_coordinates():
    """
    Ensure `as_tuple` returns the (i, j) indices.

    Expected
    --------
    - For (i,j)=(10,20), `as_tuple()` returns `(10, 20)`.
    """
    base_pair = Pair(base_i=10, base_j=20)
    assert base_pair.as_tuple() == (10, 20)


def test_pair_is_frozen_and_slotted():
    """
    Check frozen dataclass and slots behavior.

    Expected
    --------
    - Assigning to a field raises FrozenInstanceError.
    - Adding a new attribute raises AttributeError (slots).
    """
    base_pair = Pair(base_i=1, base_j=3)

    # Frozen: Changing a field should fail
    with pytest.raises(FrozenInstanceError):
        base_pair.base_i = 2

    # Slots: Adding a new attribute should fail
    with pytest.raises((AttributeError, TypeError)):
        setattr(base_pair, "new_field", 123)
