import math
import pytest

from rna_pk_fold.folding.eddy_rivas.rivas_eddy_matrices import (
    ReTriMatrix, SparseGapMatrix, get_zhx_with_collapse
)

@pytest.mark.parametrize("n", [2, 4, 6])
def test_zhx_collapse_maps_to_vx(n: int):
    vx = ReTriMatrix(n)
    zhx = SparseGapMatrix(n)

    # set distinctive values for vx
    for i in range(n):
        for j in range(i, n):
            vx.set(i, j, 20.0 + i + j / 10.0)

    for i in range(n):
        for j in range(i, n):
            for k in range(i, j):
                l = k + 1
                got = get_zhx_with_collapse(zhx, vx, i, j, k, l)
                assert got == vx.get(i, j)

def test_zhx_non_collapse_returns_stored_or_inf():
    n = 5
    vx = ReTriMatrix(n)
    zhx = SparseGapMatrix(n)
    vx.set(1, 4, 999.0)  # irrelevant baseline

    # width=1 hole
    zhx.set(1, 4, 2, 4, 7.0)
    assert get_zhx_with_collapse(zhx, vx, 1, 4, 2, 4) == 7.0

    # unset non-collapsed -> +inf
    assert math.isinf(get_zhx_with_collapse(zhx, vx, 0, 4, 1, 3))
