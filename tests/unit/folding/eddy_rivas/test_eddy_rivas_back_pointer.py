"""
Unit tests for the Eddy-Rivas backtracking data structures.

This module validates the `EddyRivasBacktrackOp` enum and the `EddyRivasBackPointer`
dataclass, which are fundamental components of the backtracking algorithm.
Tests cover data integrity (enum values, immutability), helper methods (builders),
and the serialization/deserialization logic used for debugging and logging.
"""
import pytest
from dataclasses import FrozenInstanceError

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences import (
    EddyRivasBackPointer,
    EddyRivasBacktrackOp,
)


# ---------------------- Enum basics ----------------------

def test_enum_values_are_stable_names():
    """
    Ensures the backtrack op enum uses string names for its values.

    By using `_AutoName`, the `auto()` function assigns the member's name as its
    value (e.g., `RE_PK_COMPOSE_VX.value` is `"RE_PK_COMPOSE_VX"`). This makes the
    values human-readable and stable for serialization, unlike default integer
    values which could change if the enum definition is reordered.
    """
    # Spot-check a few members to verify the auto-naming behavior.
    assert EddyRivasBacktrackOp.RE_PK_COMPOSE_VX.value == "RE_PK_COMPOSE_VX"
    assert EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT.value == "RE_WHX_SHRINK_LEFT"
    assert EddyRivasBacktrackOp.RE_VHX_SS_RIGHT.value == "RE_VHX_SS_RIGHT"
    assert EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR.value == "RE_YHX_WRAP_WHX_LR"


# ---------------------- Frozen/slotted ----------------------

def test_backpointer_is_frozen_and_slotted():
    """
    Verifies the `EddyRivasBackPointer` dataclass is immutable and memory-efficient.

    - `frozen=True`: Essential for backtracking, as it ensures that once a pointer
      is created and stored in the DP matrix, it cannot be accidentally modified.
    - `slots=True`: A crucial optimization. DP matrices can contain millions of
      these objects, so using `__slots__` significantly reduces memory usage
      compared to the standard `__dict__`.
    """
    bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX)

    # Test for immutability (`frozen=True`): re-assigning an attribute should fail.
    with pytest.raises(FrozenInstanceError):
        bp.op = EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED  # type: ignore[attr-defined]

    # Test for memory optimization (`slots=True`): an instance should not have a __dict__.
    assert hasattr(bp, "__slots__")
    assert not hasattr(bp, "__dict__")

    # Consequently, adding a new, undeclared attribute should also fail.
    with pytest.raises((AttributeError, TypeError)):
        setattr(bp, "new_field", 123)


# ---------------------- Builder helpers ----------------------

def test_compose_vx_builder_sets_expected_fields():
    """Tests the `compose_vx` builder for correct field assignment."""
    # This class method simplifies creating a common type of backpointer.
    bp = EddyRivasBackPointer.compose_vx(r=7, k=10, l=14)

    # Verify that the correct operation and arguments are set.
    assert bp.op is EddyRivasBacktrackOp.RE_PK_COMPOSE_VX
    assert bp.split == 7
    assert bp.hole == (10, 14)
    # The `args` tuple should contain all the raw arguments for easy access.
    assert bp.args == (7, 10, 14)


def test_compose_vx_drift_builder_sets_drift_and_args():
    """Tests the `compose_vx_drift` builder, which includes a drift parameter."""
    bp = EddyRivasBackPointer.compose_vx_drift(r=3, k=5, l=8, d=2)

    assert bp.op is EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT
    assert bp.split == 3 and bp.hole == (5, 8)
    assert bp.drift == 2
    assert bp.args == (3, 5, 8, 2)


def test_select_uncharged_singletons_have_empty_args():
    """
    Verifies that builders for terminal operations create argument-less backpointers.
    """
    # These operations represent base cases or transitions that don't depend
    # on subproblems, so they shouldn't carry any split/hole arguments.
    assert EddyRivasBackPointer.vx_select_uncharged().op is EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED
    assert EddyRivasBackPointer.vx_select_uncharged().args == ()
    assert EddyRivasBackPointer.wx_select_uncharged().op is EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED
    assert EddyRivasBackPointer.wx_select_uncharged().args == ()


def test_whx_shrink_and_split_builders():
    """Tests two common builders for the WHX recurrence."""
    # Test the 'shrink' operation builder.
    bp_shrink = EddyRivasBackPointer.whx_shrink_left(i=1, j=9, k1=3, l=8)
    assert bp_shrink.op is EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT
    assert bp_shrink.outer == (1, 9)
    assert bp_shrink.hole == (3, 8)
    assert bp_shrink.args == (1, 9, 3, 8)

    # Test the 'split' operation builder.
    bp_split = EddyRivasBackPointer.whx_split_left_whx_wx(r=6)
    assert bp_split.op is EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX
    assert bp_split.split == 6
    assert bp_split.args == (6,)


# ---------------------- Serialization & Deserialization ----------------------

def test_to_dict_serializes_expected_fields_with_meta():
    """
    Verifies that `to_dict()` correctly serializes a backpointer to a dictionary.
    """
    # Create a complex backpointer with most fields populated.
    bp = EddyRivasBackPointer(
        op=EddyRivasBacktrackOp.RE_VHX_WRAP_WHX,
        outer=(2, 12),
        hole=(4, 10),
        split=7,
        bridge=(1, 3),
        drift=5,
        charged=True,
        note="wrapped",
        args=(2, 12, 4, 10, 7),
    )
    d = bp.to_dict()

    # The enum `op` should be stored as its stable string value.
    assert d["op"] == "RE_VHX_WRAP_WHX"
    assert d["outer"] == (2, 12)
    assert d["hole"] == (4, 10)
    assert d["split"] == 7
    assert d["bridge"] == (1, 3)
    assert d["drift"] == 5
    assert d["charged"] is True
    # Note the intentional key change: the `note` attribute is serialized as `meta`.
    assert d["meta"] == "wrapped"


def test_from_dict_deserializes_with_note_key():
    """
    Verifies that `from_dict()` correctly reconstructs a backpointer from a dictionary.
    """
    d = {
        "op": "RE_ZHX_SS_RIGHT",
        "outer": (0, 8),
        "hole": (3, 6),
        "split": None,
        "bridge": (1, 2),
        "drift": 9,
        "charged": False,
        "note": "right-ss-step",  # Deserialization expects the key "note".
    }
    bp = EddyRivasBackPointer.from_dict(d)

    # The string "op" should be correctly converted back to an enum member.
    assert bp.op is EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT
    assert bp.outer == (0, 8)
    assert bp.hole == (3, 6)
    assert bp.split is None
    assert bp.bridge == (1, 2)
    assert bp.drift == 9
    assert bp.charged is False
    assert bp.note == "right-ss-step"


def test_from_dict_roundtrip_from_to_dict_loses_note_because_key_is_meta():
    """
    Documents an intentional asymmetry in serialization: a round trip loses the note.

    - `to_dict()` serializes the `note` attribute into a key named `"meta"`.
    - `from_dict()` expects a key named `"note"` to populate the `note` attribute.
    This test confirms that a direct round trip (`to_dict` -> `from_dict`)
    will result in the `note` field being lost (set to `None`).
    """
    src = EddyRivasBackPointer(
        op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR,
        outer=(2, 9),
        hole=(4, 7),
        note="lr-wrap",
    )
    # The created dictionary will have `d["meta"] = "lr-wrap"`.
    d = src.to_dict()
    # `from_dict` looks for a "note" key, which is absent.
    bp2 = EddyRivasBackPointer.from_dict(d)

    assert bp2.op is EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR
    assert bp2.outer == (2, 9)
    assert bp2.hole == (4, 7)
    # The note is lost because `from_dict` did not find a "note" key.
    assert bp2.note is None


def test_from_dict_rejects_unknown_op_string():
    """
    Ensures deserialization fails if the operation string is not a valid enum member.
    """
    # This prevents the creation of a corrupt backpointer object with an invalid state.
    with pytest.raises((ValueError, KeyError)):
        EddyRivasBackPointer.from_dict({"op": "NOT_A_REAL_OP"})
