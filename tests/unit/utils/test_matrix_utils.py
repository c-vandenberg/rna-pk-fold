import math

from rna_pk_fold.utils.matrix_utils import (
    get_whx_with_collapse,
    get_zhx_with_collapse,
    get_yhx_with_collapse,
    get_vhx_with_collapse,
    get_wxi_or_wx,
    whx_collapse_with,
    zhx_collapse_with,
)


class DummyTri:
    """Triangle matrix stub with (+inf) default."""
    def __init__(self):
        self._d = {}
    def set(self, i, j, v):
        self._d[(i, j)] = v
    def get(self, i, j):
        return self._d.get((i, j), math.inf)


class DummyGap:
    """4D sparse gap matrix stub with (+inf) default."""
    def __init__(self):
        self._d = {}
    def set(self, i, j, k, l, v):
        self._d[(i, j, k, l)] = v
    def get(self, i, j, k, l):
        return self._d.get((i, j, k, l), math.inf)


class DummyState:
    """Duck-typed state carrying only the matrices we touch."""
    def __init__(self):
        # Outer non-gap (uncharged/charged) and optional insertion matrix
        self.wx_matrix = DummyTri()
        self.wxu_matrix = DummyTri()
        self.wxc_matrix = DummyTri()
        self.wxi_matrix = None  # filled per-test when needed

        self.vxu_matrix = DummyTri()
        self.vxc_matrix = DummyTri()

        # Gap matrices
        self.whx_matrix = DummyGap()
        self.zhx_matrix = DummyGap()


# -------------------------------
# get_whx_with_collapse / get_zhx_with_collapse
# -------------------------------
def test_get_whx_with_collapse_uses_wx_for_unit_hole():
    whx, wx = DummyGap(), DummyTri()
    i, j, k = 2, 7, 4
    l = k + 1  # collapse case
    wx.set(i, j, -2.25)

    assert math.isclose(get_whx_with_collapse(whx, wx, i, j, k, l), -2.25, rel_tol=1e-12)


def test_get_whx_with_collapse_returns_whx_when_noncollapse():
    whx, wx = DummyGap(), DummyTri()
    i, j, k, l = 2, 7, 4, 6
    whx.set(i, j, k, l, -3.5)
    wx.set(i, j, +1.0)  # should be ignored (no collapse)

    assert math.isclose(get_whx_with_collapse(whx, wx, i, j, k, l), -3.5, rel_tol=1e-12)


def test_get_whx_with_collapse_invalid_geometry_yields_inf():
    whx, wx = DummyGap(), DummyTri()
    # k outside [i, j]
    assert math.isinf(get_whx_with_collapse(whx, wx, i=2, j=5, k=10, l=11))


def test_get_zhx_with_collapse_mirrors_whx_behavior():
    zhx, vx = DummyGap(), DummyTri()
    i, j, k = 1, 6, 3

    # collapse → vx
    vx.set(i, j, -1.1)
    assert math.isclose(get_zhx_with_collapse(zhx, vx, i, j, k, k + 1), -1.1, rel_tol=1e-12)

    # non-collapse → zhx
    zhx.set(i, j, k, k + 2, -4.4)
    assert math.isclose(get_zhx_with_collapse(zhx, vx, i, j, k, k + 2), -4.4, rel_tol=1e-12)


# -------------------------------
# get_yhx_with_collapse / get_vhx_with_collapse
# -------------------------------
def test_get_yhx_with_collapse_returns_invalid_on_unit_hole_and_reads_value_else():
    yhx = DummyGap()
    i, j, k, l = 2, 7, 4, 5  # unit hole → collapse case
    # default invalid_value is +inf
    assert math.isinf(get_yhx_with_collapse(yhx, i, j, k, l))

    # Custom invalid_value
    assert math.isclose(get_yhx_with_collapse(yhx, i, j, k, l, invalid_value=123.456), 123.456)

    # Non-collapse returns stored value
    yhx.set(i, j, 4, 6, -0.75)
    assert math.isclose(get_yhx_with_collapse(yhx, i, j, 4, 6), -0.75, rel_tol=1e-12)


def test_get_vhx_with_collapse_returns_invalid_on_unit_hole_and_reads_value_else():
    vhx = DummyGap()
    i, j, k, l = 0, 4, 1, 2  # unit hole
    assert math.isinf(get_vhx_with_collapse(vhx, i, j, k, l))

    assert math.isclose(get_vhx_with_collapse(vhx, i, j, k, l, invalid_value=-9.9), -9.9, rel_tol=1e-12)

    vhx.set(i, j, 1, 3, -6.0)
    assert math.isclose(get_vhx_with_collapse(vhx, i, j, 1, 3), -6.0, rel_tol=1e-12)


# -------------------------------
# wxI
# -------------------------------
def test_wxI_prefers_wxi_when_present_else_falls_back_to_wx():
    st = DummyState()
    i, j = 2, 6
    st.wx_matrix.set(i, j, -1.0)

    # No wxi → uses wx
    assert math.isclose(get_wxi_or_wx(st, i, j), -1.0, rel_tol=1e-12)

    # With wxi present → prefers wxi
    st.wxi_matrix = DummyTri()
    st.wxi_matrix.set(i, j, -3.5)
    assert math.isclose(get_wxi_or_wx(st, i, j), -3.5, rel_tol=1e-12)


# -------------------------------
# whx_collapse_with / zhx_collapse_with
# -------------------------------
def test_whx_collapse_with_switches_between_charged_and_uncharged():
    st = DummyState()
    i, j, k = 3, 8, 5

    st.wxu_matrix.set(i, j, -1.0)
    st.wxc_matrix.set(i, j, -3.3)

    # Collapse path (k+1==l), charged toggle decides which matrix is used
    assert math.isclose(whx_collapse_with(st, i, j, k, k + 1, charged=False), -1.0, rel_tol=1e-12)
    assert math.isclose(whx_collapse_with(st, i, j, k, k + 1, charged=True), -3.3, rel_tol=1e-12)

    # If both collapse matrices are +inf, fallback returns whx(i,j,k,l)
    st = DummyState()
    st.whx_matrix.set(i, j, k, k + 1, -7.7)
    assert math.isclose(whx_collapse_with(st, i, j, k, k + 1, charged=True), -7.7, rel_tol=1e-12)


def test_zhx_collapse_with_switches_between_charged_and_uncharged():
    st = DummyState()
    i, j, k = 0, 4, 1

    st.vxu_matrix.set(i, j, -2.0)
    st.vxc_matrix.set(i, j, -5.0)

    assert math.isclose(zhx_collapse_with(st, i, j, k, k + 1, charged=False), -2.0, rel_tol=1e-12)
    assert math.isclose(zhx_collapse_with(st, i, j, k, k + 1, charged=True), -5.0, rel_tol=1e-12)

    st = DummyState()
    st.zhx_matrix.set(i, j, k, k + 1, -0.123)
    assert math.isclose(zhx_collapse_with(st, i, j, k, k + 1, charged=False), -0.123, rel_tol=1e-12)
