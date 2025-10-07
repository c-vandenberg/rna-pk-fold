import pytest
from types import SimpleNamespace

# Adjust this import path to match your file containing wx_bp/whx_bp/...
from rna_pk_fold.utils.back_pointer_utils import (
    wx_bp, whx_bp, yhx_bp, zhx_bp, vhx_bp
)


class Recorder2:
    """Minimal 2-index back-pointer map with call recording."""
    def __init__(self, ret=None):
        self.last_args = None
        self._ret = ret

    def get(self, i, j):
        self.last_args = (i, j)
        return self._ret


class Recorder4:
    """Minimal 4-index back-pointer map with call recording."""
    def __init__(self, ret=None):
        self.last_args = None
        self._ret = ret

    def get(self, i, j, k, l):
        self.last_args = (i, j, k, l)
        return self._ret


@pytest.fixture()
def state():
    """
    Duck-typed 'state' object with the exact attributes used by the helpers.
    We don't import the real EddyRivasFoldState to keep the test lightweight.
    """
    return SimpleNamespace(
        wx_back_ptr=Recorder2(),
        whx_back_ptr=Recorder4(),
        yhx_back_ptr=Recorder4(),
        zhx_back_ptr=Recorder4(),
        vhx_back_ptr=Recorder4(),
    )


def test_wx_bp_returns_recorded_value_and_calls_get_with_correct_arity_and_order(state):
    sentinel = object()
    state.wx_back_ptr._ret = sentinel

    got = wx_bp(state, 3, 7)
    assert got is sentinel
    assert state.wx_back_ptr.last_args == (3, 7)

    # Miss path (no entry → our Recorder returns its _ret; set to None)
    state.wx_back_ptr._ret = None
    got_none = wx_bp(state, 1, 2)
    assert got_none is None
    assert state.wx_back_ptr.last_args == (1, 2)


@pytest.mark.parametrize(
    "func, attr_name",
    [
        (whx_bp, "whx_back_ptr"),
        (yhx_bp, "yhx_back_ptr"),
        (zhx_bp, "zhx_back_ptr"),
        (vhx_bp, "vhx_back_ptr"),
    ],
)
def test_hole_backpointer_helpers_return_value_and_call_order(state, func, attr_name):
    rec: Recorder4 = getattr(state, attr_name)
    sentinel = object()
    rec._ret = sentinel

    # Concrete coordinates to catch ordering mistakes
    i, j, k, l = (2, 9, 4, 7)
    got = func(state, i, j, k, l)
    assert got is sentinel
    assert rec.last_args == (i, j, k, l)

    # Miss path
    rec._ret = None
    got_none = func(state, 0, 1, 2, 3)
    assert got_none is None
    assert rec.last_args == (0, 1, 2, 3)
