import pytest
from dataclasses import FrozenInstanceError

from rna_pk_fold.folding.zucker.zucker_back_pointer import ZuckerBackPointer, ZuckerBacktrackOp


def test_backtrackop_members_exist():
    """
    Enum must contain the expected traceback operations.

    Expected
    --------
    - Members include NONE, HAIRPIN, BIFURCATION, UNPAIRED_LEFT, PSEUDOKNOT.
    """
    # Spot-check a few members
    assert ZuckerBacktrackOp.NONE.name == "NONE"
    assert ZuckerBacktrackOp.HAIRPIN.name == "HAIRPIN"
    assert ZuckerBacktrackOp.BIFURCATION.name == "BIFURCATION"
    assert ZuckerBacktrackOp.UNPAIRED_LEFT.name == "UNPAIRED_LEFT"


def test_backpointer_defaults_and_fields():
    """
    BackPointer should default to NONE and have empty optional fields.

    Expected
    --------
    - `operation` is `BacktrackOp.NONE`.
    - `split_k`, `inner`, and `note` are `None`.
    """
    back_ptr = ZuckerBackPointer()
    assert back_ptr.operation is ZuckerBacktrackOp.NONE
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
    back_ptr = ZuckerBackPointer()

    with pytest.raises(FrozenInstanceError):
        back_ptr.operation = ZuckerBacktrackOp.HAIRPIN

    with pytest.raises((AttributeError, TypeError)):
        setattr(back_ptr, "new_attr", 123)


def test_backpointer_custom_values():
    """
    Construct a non-default BackPointer and verify its content.

    Expected
    --------
    - Fields reflect the provided `operation`, `split_k`, `inner`, and `note`.
    """
    back_ptr = ZuckerBackPointer(
        operation=ZuckerBacktrackOp.BIFURCATION,
        split_k=7,
        inner=(3, 9),
        note="split at k=7",
    )
    assert back_ptr.operation is ZuckerBacktrackOp.BIFURCATION
    assert back_ptr.split_k == 7
    assert back_ptr.inner == (3, 9)
    assert back_ptr.note == "split at k=7"
