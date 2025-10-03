import math
import pytest

from rna_pk_fold.folding.rivas_eddy.rivas_eddy_matrices import (
    ReTriMatrix, SparseGapMatrix, get_whx_with_collapse
)

@pytest.mark.parametrize("n", [2, 4, 6])
def test_whx_collapse_maps_to_wx(n: int):
    """
    For l == k+1 (zero-width hole), get_whx_with_collapse(i,j:k,k+1) must
    equal wx(i,j) even if WHX is unset.
    """
    wx = ReTriMatrix(n)
    whx = SparseGapMatrix(n)

    # Set distinctive sentinel values on wx(i,j)
    for i in range(n):
        for j in range(i, n):
            wx.set(i, j, 10.0 + i + j / 100.0)

    # For every outer (i,j), and every k in [i, j-1], the collapse identity holds.
    for i in range(n):
        for j in range(i, n):
            for k in range(i, j):
                l = k + 1
                got = get_whx_with_collapse(whx, wx, i, j, k, l)
                assert got == wx.get(i, j), f"collapse mismatch at (i,j,k,l)=({i},{j},{k},{l})"


def test_whx_non_collapse_prefers_stored_value():
    """
    For non-collapsed holes (l >= k+2), the helper must return the stored WHX value,
    not the outer wx(i,j).
    """
    n = 6
    wx = ReTriMatrix(n)
    whx = SparseGapMatrix(n)

    # Outer cell has a different sentinel
    wx.set(1, 5, 123.456)

    # Non-collapsed hole (k=2, l=4 => width = 1)
    whx.set(1, 5, 2, 4, 7.89)

    got = get_whx_with_collapse(whx, wx, 1, 5, 2, 4)
    assert got == 7.89, "Should return stored WHX for non-collapsed holes"
    assert got != wx.get(1, 5), "Must not fall back to wx(i,j) when not collapsed"


def test_whx_non_collapse_unset_is_inf():
    """
    If the hole is non-collapsed and WHX is unset, helper should return +inf.
    """
    n = 5
    wx = ReTriMatrix(n)
    whx = SparseGapMatrix(n)

    # Set wx to something finite to ensure we don't accidentally pick it
    wx.set(0, 4, -3.0)

    # Non-collapsed hole (l >= k+2) but WHX unset → +inf
    got = get_whx_with_collapse(whx, wx, 0, 4, 1, 3)  # width=1
    assert math.isinf(got)
