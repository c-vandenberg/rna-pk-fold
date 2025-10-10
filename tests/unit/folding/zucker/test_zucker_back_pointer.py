"""
Unit tests for the Zucker algorithm's backtracking data structures.

This module validates the `ZuckerBacktrackOp` enum and the `ZuckerBackPointer`
dataclass. These tests ensure that the foundational components for storing and
reconstructing the folding path are stable, immutable, and correctly initialized.
"""
import pytest
from dataclasses import FrozenInstanceError

from rna_pk_fold.folding.zucker.zucker_back_pointer import ZuckerBackPointer, ZuckerBacktrackOp


def test_backtrackop_members_exist():
    """
    Ensures the `ZuckerBacktrackOp` enum contains the expected members.

    These members represent the fundamental operations or states in the Zucker
    dynamic programming recurrences, which are essential for a correct traceback.
    """
    # Spot-check that key members exist by accessing them.
    assert ZuckerBacktrackOp.NONE.name == "NONE"
    assert ZuckerBacktrackOp.HAIRPIN.name == "HAIRPIN"
    assert ZuckerBacktrackOp.BIFURCATION.name == "BIFURCATION"
    assert ZuckerBacktrackOp.UNPAIRED_LEFT.name == "UNPAIRED_LEFT"


def test_backpointer_defaults_and_fields():
    """
    Verifies that a default `ZuckerBackPointer` is correctly initialized.

    An empty backpointer should represent a null or terminal state, defaulting
    to the `NONE` operation with no associated split points or other data.
    """
    # Create a backpointer with no arguments.
    back_ptr = ZuckerBackPointer()
    # It should default to the NONE operation.
    assert back_ptr.operation is ZuckerBacktrackOp.NONE
    # All optional fields used to store subproblem coordinates should be None.
    assert back_ptr.split_k is None
    assert back_ptr.inner is None
    assert back_ptr.note is None


def test_backpointer_is_frozen_and_slotted():
    """
    Confirms that the `ZuckerBackPointer` dataclass is immutable and memory-efficient.

    - `frozen=True`: This is a critical safety feature. Once created and stored
      in a DP matrix, a backpointer cannot be accidentally modified, ensuring the
      integrity of the final traceback.
    - `slots=True`: This is a performance optimization. Using slots reduces the
      memory footprint of each instance, which is important for large DP tables.
    """
    back_ptr = ZuckerBackPointer()

    # Test for immutability: attempting to change a field should raise an error.
    with pytest.raises(FrozenInstanceError):
        back_ptr.operation = ZuckerBacktrackOp.HAIRPIN

    # Test for slotted behavior: adding a new, undeclared attribute should fail.
    with pytest.raises((AttributeError, TypeError)):
        setattr(back_ptr, "new_attr", 123)


def test_backpointer_custom_values():
    """
    Tests the creation of a `ZuckerBackPointer` with specific, non-default values.

    This ensures the dataclass constructor correctly assigns all provided
    attributes, which is necessary for storing the results of specific DP
    recurrence choices (e.g., a bifurcation at a particular index `k`).
    """
    # Create a backpointer representing a bifurcation event.
    back_ptr = ZuckerBackPointer(
        operation=ZuckerBacktrackOp.BIFURCATION,
        split_k=7,
        inner=(3, 9),
        note="split at k=7",
    )
    # Verify that all fields were set correctly.
    assert back_ptr.operation is ZuckerBacktrackOp.BIFURCATION
    assert back_ptr.split_k == 7
    assert back_ptr.inner == (3, 9)
    assert back_ptr.note == "split at k=7"
