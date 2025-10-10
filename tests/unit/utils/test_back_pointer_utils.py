"""
Unit tests for the backpointer accessor utility functions.

This module validates a set of simple helper functions (`wx_bp`, `whx_bp`, etc.)
that provide convenient, shorthand access to the various backpointer matrices
within a folding state object.

The tests use mock objects ("Recorders") and a duck-typed `SimpleNamespace`
to simulate the folding state, keeping the tests lightweight and focused purely
on the behavior of the accessor functions themselves.
"""
import pytest
from types import SimpleNamespace

# Adjust this import path to match your file containing wx_bp/whx_bp/...
from rna_pk_fold.utils.back_pointer_utils import (
    wx_bp, whx_bp, yhx_bp, zhx_bp, vhx_bp
)


class Recorder2:
    """
    A minimal mock object for a 2-index backpointer map.
    It mimics the `.get(i, j)` method and records the arguments it was called with,
    allowing tests to verify the behavior of accessor functions.
    """
    def __init__(self, ret=None):
        self.last_args = None
        self._ret = ret  # The value to return when `get` is called.

    def get(self, i, j):
        self.last_args = (i, j)
        return self._ret


class Recorder4:
    """
    A minimal mock object for a 4-index (sparse gap) backpointer map.
    It mimics the `.get(i, j, k, l)` method and records the call arguments.
    """
    def __init__(self, ret=None):
        self.last_args = None
        self._ret = ret  # The value to return when `get` is called.

    def get(self, i, j, k, l):
        self.last_args = (i, j, k, l)
        return self._ret


@pytest.fixture()
def state():
    """
    Provides a duck-typed mock of a folding state object.

    This fixture uses `SimpleNamespace` to create an object with the necessary
    backpointer attributes (`wx_back_ptr`, `whx_back_ptr`, etc.). Each attribute
    is assigned a `Recorder` instance, allowing the tests to run without needing
    to import and construct the full, potentially complex `EddyRivasFoldState`.
    """
    return SimpleNamespace(
        wx_back_ptr=Recorder2(),
        whx_back_ptr=Recorder4(),
        yhx_back_ptr=Recorder4(),
        zhx_back_ptr=Recorder4(),
        vhx_back_ptr=Recorder4(),
    )


def test_wx_bp_returns_recorded_value_and_calls_get_with_correct_arity_and_order(state):
    """
    Tests the `wx_bp` helper function.

    This test verifies two key behaviors:
    1. The helper correctly returns whatever value the underlying `.get()` method provides.
    2. The helper calls the `.get()` method with the correct arguments (i, j) and in the correct order.
    """
    # --- Test "hit" path: a backpointer is found ---
    sentinel = object()
    state.wx_back_ptr._ret = sentinel # Configure the recorder to return a specific object.

    # Call the helper function.
    got = wx_bp(state, 3, 7)

    # Assert that the returned value is the one we configured.
    assert got is sentinel
    # Assert that the underlying `get` method was called with the correct arguments.
    assert state.wx_back_ptr.last_args == (3, 7)

    # --- Test "miss" path: no backpointer is found ---
    state.wx_back_ptr._ret = None # Configure the recorder to return None.
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
    """
    Tests all 4-index ("hole") backpointer helper functions using parametrization.

    This single test validates that each of the specified helper functions (`whx_bp`,
    `yhx_bp`, etc.) correctly calls the appropriate `get` method on the state
    object with the four indices (i, j, k, l) in the correct order, and that it
    passes through the return value.
    """
    # Get the correct recorder object from the mock state based on the test parameter.
    rec: Recorder4 = getattr(state, attr_name)

    # --- Test "hit" path ---
    sentinel = object()
    rec._ret = sentinel # Configure the recorder to return our sentinel object.

    # Use distinct coordinates to ensure the argument order is tested correctly.
    i, j, k, l = (2, 9, 4, 7)
    got = func(state, i, j, k, l)

    # Verify the return value and the arguments passed to the underlying `get` method.
    assert got is sentinel
    assert rec.last_args == (i, j, k, l)

    # --- Test "miss" path ---
    rec._ret = None
    got_none = func(state, 0, 1, 2, 3)
    assert got_none is None
    assert rec.last_args == (0, 1, 2, 3)
