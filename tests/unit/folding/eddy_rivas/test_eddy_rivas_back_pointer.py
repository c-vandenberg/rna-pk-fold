import pytest
from dataclasses import FrozenInstanceError

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences import (
    EddyRivasBackPointer,
    EddyRivasBacktrackOp,
)


# ---------------------- Enum basics ----------------------
def test_enum_values_are_stable_names():
    """_AutoName maps auto() to the member name string (stable/serializable)."""
    # Spot-check a few to keep this lightweight but meaningful
    assert EddyRivasBacktrackOp.RE_PK_COMPOSE_VX.value == "RE_PK_COMPOSE_VX"
    assert EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT.value == "RE_WHX_SHRINK_LEFT"
    assert EddyRivasBacktrackOp.RE_VHX_SS_RIGHT.value == "RE_VHX_SS_RIGHT"
    assert EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR.value == "RE_YHX_WRAP_WHX_LR"


# ---------------------- Frozen/slotted ----------------------
def test_backpointer_is_frozen_and_slotted():
    """Dataclass must be frozen, slotted (no __dict__), and immutable."""
    bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX)

    # Frozen: re-assignment should fail
    with pytest.raises(FrozenInstanceError):
        bp.op = EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED  # type: ignore[attr-defined]

    # Slots present, no __dict__ for instances
    assert hasattr(bp, "__slots__")
    assert not hasattr(bp, "__dict__")

    # Adding a new attribute should fail
    with pytest.raises((AttributeError, TypeError)):
        setattr(bp, "new_field", 123)


# ---------------------- Builder helpers ----------------------
def test_compose_vx_builder_sets_expected_fields():
    bp = EddyRivasBackPointer.compose_vx(r=7, k=10, l=14)
    assert bp.op is EddyRivasBacktrackOp.RE_PK_COMPOSE_VX
    assert bp.split == 7
    assert bp.hole == (10, 14)
    assert bp.args == (7, 10, 14)


def test_compose_vx_drift_builder_sets_drift_and_args():
    bp = EddyRivasBackPointer.compose_vx_drift(r=3, k=5, l=8, d=2)
    assert bp.op is EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT
    assert bp.split == 3 and bp.hole == (5, 8)
    assert bp.drift == 2
    assert bp.args == (3, 5, 8, 2)


def test_select_uncharged_singletons_have_empty_args():
    assert EddyRivasBackPointer.vx_select_uncharged().op is EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED
    assert EddyRivasBackPointer.vx_select_uncharged().args == ()
    assert EddyRivasBackPointer.wx_select_uncharged().op is EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED
    assert EddyRivasBackPointer.wx_select_uncharged().args == ()


def test_whx_shrink_and_split_builders():
    bp_shrink = EddyRivasBackPointer.whx_shrink_left(i=1, j=9, k1=3, l=8)
    assert bp_shrink.op is EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT
    assert bp_shrink.outer == (1, 9)
    assert bp_shrink.hole == (3, 8)
    assert bp_shrink.args == (1, 9, 3, 8)

    bp_split = EddyRivasBackPointer.whx_split_left_whx_wx(r=6)
    assert bp_split.op is EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX
    assert bp_split.split == 6
    assert bp_split.args == (6,)


# ---------------------- Serialization & Deserialization ----------------------
def test_to_dict_serializes_expected_fields_with_meta():
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
    # op is stored as the string value for stable logs/JSON
    assert d["op"] == "RE_VHX_WRAP_WHX"
    assert d["outer"] == (2, 12)
    assert d["hole"] == (4, 10)
    assert d["split"] == 7
    assert d["bridge"] == (1, 3)
    assert d["drift"] == 5
    assert d["charged"] is True
    # Note: to_dict() uses key "meta" (not "note")
    assert d["meta"] == "wrapped"


def test_from_dict_deserializes_with_note_key():
    d = {
        "op": "RE_ZHX_SS_RIGHT",
        "outer": (0, 8),
        "hole": (3, 6),
        "split": None,
        "bridge": (1, 2),
        "drift": 9,
        "charged": False,
        "note": "right-ss-step",
    }
    bp = EddyRivasBackPointer.from_dict(d)
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
    Intentional documentation test: to_dict() writes key 'meta',
    but from_dict() expects 'note'. Round-tripping via to_dict() drops the note.
    """
    src = EddyRivasBackPointer(
        op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR,
        outer=(2, 9),
        hole=(4, 7),
        note="lr-wrap",
    )
    d = src.to_dict()  # contains d["meta"] = "lr-wrap"
    bp2 = EddyRivasBackPointer.from_dict(d)  # expects 'note' -> None
    assert bp2.op is EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR
    assert bp2.outer == (2, 9)
    assert bp2.hole == (4, 7)
    assert bp2.note is None  # note is not read from 'meta'


def test_from_dict_rejects_unknown_op_string():
    with pytest.raises((ValueError, KeyError)):
        EddyRivasBackPointer.from_dict({"op": "NOT_A_REAL_OP"})
