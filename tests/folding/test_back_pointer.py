import pytest
from dataclasses import FrozenInstanceError

from rna_pk_fold.folding.back_pointer import BackPointer, BacktrackOp


def test_backtrackop_members_exist():
    """
    Enum must contain the expected traceback operations.
    """
    # Spot-check a few members
    assert BacktrackOp.NONE.name == "NONE"
    assert BacktrackOp.HAIRPIN.name == "HAIRPIN"
    assert BacktrackOp.BIFURCATION.name == "BIFURCATION"
    assert BacktrackOp.UNPAIRED_LEFT.name == "UNPAIRED_LEFT"
    assert BacktrackOp.PSEUDOKNOT.name == "PSEUDOKNOT"


def test_backpointer_defaults_and_fields():
    """
    BackPointer should default to NONE and have empty optional fields.
    """
    back_ptr = BackPointer()
    assert back_ptr.operation is BacktrackOp.NONE
    assert back_ptr.split_k is None
    assert back_ptr.inner is None
    assert back_ptr.note is None


def test_backpointer_is_frozen_and_slotted():
    """
    Verify frozen dataclass and slotted behavior.

    Expected
    --------
    - Assigning to a field raises FrozenInstanceError.
    - Adding a new attribute raises (AttributeError or TypeError),
      since CPython may raise either for frozen+slots objects.
    """
    back_ptr = BackPointer()

    with pytest.raises(FrozenInstanceError):
        back_ptr.operation = BacktrackOp.HAIRPIN

    with pytest.raises((AttributeError, TypeError)):
        setattr(back_ptr, "new_attr", 123)


def test_backpointer_custom_values():
    """
    Construct a non-default BackPointer and verify its content.
    """
    back_ptr = BackPointer(
        operation=BacktrackOp.BIFURCATION,
        split_k=7,
        inner=(3, 9),
        note="split at k=7",
    )
    assert back_ptr.operation is BacktrackOp.BIFURCATION
    assert back_ptr.split_k == 7
    assert back_ptr.inner == (3, 9)
    assert back_ptr.note == "split at k=7"
