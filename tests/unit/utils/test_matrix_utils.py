"""
Unit tests for high-level matrix utility functions.

This module validates helper functions from `matrix_utils` that provide a
logic-aware interface for accessing values from dynamic programming (DP) matrices.
These helpers encapsulate common algorithmic patterns, such as handling "collapse"
identities (where a 4D matrix state reduces to a 2D one) and selecting between
different matrix types (e.g., "charged" vs. "uncharged"), which simplifies the
main recurrence code.

The tests use minimal mock ("Dummy") classes to isolate the behavior of these
utility functions from the full complexity of the folding state and matrix
implementations.
"""
import math
import pytest

from rna_pk_fold.utils.matrix_utils import (
    clear_matrix_caches,
    get_whx_with_collapse,
    get_zhx_with_collapse,
    get_yhx_with_collapse,
    get_vhx_with_collapse,
    get_wxi_or_wx,
    whx_collapse_with,
    zhx_collapse_with,
)


class DummyTri:
    """A minimal mock of a 2D triangular matrix, defaulting to +infinity."""
    def __init__(self):
        self._d = {}
    def set(self, i, j, v):
        self._d[(i, j)] = v
    def get(self, i, j):
        return self._d.get((i, j), math.inf)


class DummyGap:
    """A minimal mock of a 4D sparse gap matrix, defaulting to +infinity."""
    def __init__(self):
        self._d = {}
    def set(self, i, j, k, l, v):
        self._d[(i, j, k, l)] = v
    def get(self, i, j, k, l):
        return self._d.get((i, j, k, l), math.inf)


class DummyState:
    """A duck-typed mock of the `EddyRivasFoldState` for isolated testing."""
    def __init__(self):
        # 2D matrices for non-pseudoknotted (charged/uncharged) states
        self.wx_matrix = DummyTri()
        self.wxu_matrix = DummyTri()
        self.wxc_matrix = DummyTri()
        self.wxi_matrix = None  # Optional matrix, initialized per-test

        self.vxu_matrix = DummyTri()
        self.vxc_matrix = DummyTri()

        # 4D matrices for pseudoknotted states
        self.whx_matrix = DummyGap()
        self.zhx_matrix = DummyGap()


@pytest.fixture(autouse=True)
def auto_clear_matrix_caches():
    """
    An autouse fixture that clears memoization caches before and after each test.
    This is critical for ensuring test isolation, as it prevents results from one
    test run from affecting the outcome of another.
    """
    clear_matrix_caches()
    yield
    clear_matrix_caches()


# -------------------------------
# get_whx_with_collapse / get_zhx_with_collapse
# -------------------------------
def test_get_whx_with_collapse_uses_wx_for_unit_hole():
    """
    Tests the "collapse" identity for WHX: when the hole size is minimal (l = k+1),
    the 4D WHX state is equivalent to a 2D WX state.
    """
    whx, wx = DummyGap(), DummyTri()
    i, j, k = 2, 7, 4
    l = k + 1  # This condition triggers the collapse.
    wx.set(i, j, -2.25)

    # The helper should retrieve the value from the WX matrix, not the WHX matrix.
    assert math.isclose(get_whx_with_collapse(whx, wx, i, j, k, l), -2.25, rel_tol=1e-12)


def test_get_whx_with_collapse_returns_whx_when_noncollapse():
    """
    Tests the standard case: when the hole is larger than a unit hole, the helper
    should retrieve the value directly from the WHX matrix.
    """
    whx, wx = DummyGap(), DummyTri()
    i, j, k, l = 2, 7, 4, 6 # l > k+1, so no collapse.
    whx.set(i, j, k, l, -3.5)
    wx.set(i, j, +1.0)  # This value should be ignored.

    assert math.isclose(get_whx_with_collapse(whx, wx, i, j, k, l), -3.5, rel_tol=1e-12)


def test_get_whx_with_collapse_invalid_geometry_yields_inf():
    """
    Tests that the helper returns +infinity for geometrically invalid indices.
    """
    whx, wx = DummyGap(), DummyTri()
    # An example of invalid geometry where k is outside the [i, j] span.
    assert math.isinf(get_whx_with_collapse(whx, wx, i=2, j=5, k=10, l=11))


def test_get_zhx_with_collapse_mirrors_whx_behavior():
    """
    Tests that `get_zhx_with_collapse` has behavior analogous to its WHX
    counterpart, but collapses to the VX matrix instead of WX.
    """
    zhx, vx = DummyGap(), DummyTri()
    i, j, k = 1, 6, 3

    # Test the collapse case: for a unit hole, it should use the VX matrix.
    vx.set(i, j, -1.1)
    assert math.isclose(get_zhx_with_collapse(zhx, vx, i, j, k, k + 1), -1.1, rel_tol=1e-12)

    # Test the non-collapse case: it should use the ZHX matrix.
    zhx.set(i, j, k, k + 2, -4.4)
    assert math.isclose(get_zhx_with_collapse(zhx, vx, i, j, k, k + 2), -4.4, rel_tol=1e-12)


# -------------------------------
# get_yhx_with_collapse / get_vhx_with_collapse
# -------------------------------
def test_get_yhx_with_collapse_returns_invalid_on_unit_hole_and_reads_value_else():
    """
    Tests the YHX collapse rule: a unit hole is an invalid state for YHX.
    The helper should return a specified `invalid_value` (defaulting to +inf)
    in this case, and otherwise read from the YHX matrix.
    """
    yhx = DummyGap()
    i, j, k, l = 2, 7, 4, 5  # Unit hole (l = k+1) is an invalid state for YHX.

    # With the default invalid_value, it should return +infinity.
    assert math.isinf(get_yhx_with_collapse(yhx, i, j, k, l))

    # With a custom invalid_value, it should return that value.
    assert math.isclose(get_yhx_with_collapse(yhx, i, j, k, l, invalid_value=123.456), 123.456)

    # For a non-collapse case, it should return the stored value from the YHX matrix.
    yhx.set(i, j, 4, 6, -0.75)
    assert math.isclose(get_yhx_with_collapse(yhx, i, j, 4, 6), -0.75, rel_tol=1e-12)


def test_get_vhx_with_collapse_returns_invalid_on_unit_hole_and_reads_value_else():
    """
    Tests that `get_vhx_with_collapse` mirrors the collapse behavior of YHX.
    A unit hole is also an invalid state for VHX.
    """
    vhx = DummyGap()
    i, j, k, l = 0, 4, 1, 2  # Unit hole.
    # Should return +infinity by default.
    assert math.isinf(get_vhx_with_collapse(vhx, i, j, k, l))

    # Should return the custom invalid value when provided.
    assert math.isclose(get_vhx_with_collapse(vhx, i, j, k, l, invalid_value=-9.9), -9.9, rel_tol=1e-12)

    # Should return the stored value for non-collapse cases.
    vhx.set(i, j, 1, 3, -6.0)
    assert math.isclose(get_vhx_with_collapse(vhx, i, j, 1, 3), -6.0, rel_tol=1e-12)


# -------------------------------
# get_wxi_or_wx
# -------------------------------
def test_wxI_prefers_wxi_when_present_else_falls_back_to_wx():
    """
    Tests the `get_wxi_or_wx` helper, which prioritizes the optional WXI matrix.
    If the `wxi_matrix` attribute exists on the state, its value should be used.
    Otherwise, the function should fall back to the standard `wx_matrix`.
    """
    st = DummyState()
    i, j = 2, 6
    st.wx_matrix.set(i, j, -1.0)

    # Case 1: `wxi_matrix` is None, so it should fall back to `wx_matrix`.
    assert math.isclose(get_wxi_or_wx(st, i, j), -1.0, rel_tol=1e-12)

    # Case 2: `wxi_matrix` is present, so its value should be preferred.
    st.wxi_matrix = DummyTri()
    st.wxi_matrix.set(i, j, -3.5)
    assert math.isclose(get_wxi_or_wx(st, i, j), -3.5, rel_tol=1e-12)


# -------------------------------
# whx_collapse_with / zhx_collapse_with
# -------------------------------
def test_whx_collapse_with_switches_between_charged_and_uncharged():
    """
    Tests a helper that combines the WHX collapse logic with selection between
    "charged" (WXC) and "uncharged" (WXU) states.
    """
    st = DummyState()
    i, j, k = 3, 8, 5

    st.wxu_matrix.set(i, j, -1.0) # Uncharged score.
    st.wxc_matrix.set(i, j, -3.3) # Charged score.

    # When collapsing (l=k+1), the `charged` flag determines which matrix to use.
    assert math.isclose(whx_collapse_with(st, i, j, k, k + 1, charged=False), -1.0, rel_tol=1e-12)
    assert math.isclose(whx_collapse_with(st, i, j, k, k + 1, charged=True), -3.3, rel_tol=1e-12)

    # Test fallback behavior: if the target 2D collapse matrix is empty (+inf),
    # the function should fall back to the value in the original 4D WHX matrix.
    st = DummyState()
    st.whx_matrix.set(i, j, k, k + 1, -7.7)
    assert math.isclose(whx_collapse_with(st, i, j, k, k + 1, charged=True), -7.7, rel_tol=1e-12)


def test_zhx_collapse_with_switches_between_charged_and_uncharged():
    """
    Tests that `zhx_collapse_with` mirrors the behavior of `whx_collapse_with`,
    but for the VXC and VXU matrices.
    """
    clear_matrix_caches()
    st = DummyState()
    i, j, k = 0, 4, 1

    st.vxu_matrix.set(i, j, -2.0)
    st.vxc_matrix.set(i, j, -5.0)

    # The `charged` flag should correctly select between VXC and VXU.
    assert math.isclose(zhx_collapse_with(st, i, j, k, k + 1, charged=False), -2.0, rel_tol=1e-12)
    assert math.isclose(zhx_collapse_with(st, i, j, k, k + 1, charged=True), -5.0, rel_tol=1e-12)

    # Test the fallback to the ZHX matrix if VXC/VXU are empty.
    st = DummyState()
    st.zhx_matrix.set(i, j, k, k + 1, -0.123)
    assert math.isclose(zhx_collapse_with(st, i, j, k, k + 1, charged=False), -0.123, rel_tol=1e-12)
