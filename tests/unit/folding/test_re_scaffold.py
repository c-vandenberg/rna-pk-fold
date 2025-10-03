# tests/test_re_scaffold.py

import math
import pytest

from rna_pk_fold.folding.rivas_eddy_matrices import (
    TriMatrix, SparseGapMatrix,
    get_whx_with_collapse, get_zhx_with_collapse
)
from rna_pk_fold.folding.fold_state import make_re_fold_state

@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_collapse_identity_maps_to_wx_vx(n: int):
    re = make_re_fold_state(n)

    # For visibility, put a distinctive value on some wx/vx cells
    for i in range(n):
        for j in range(i, n):
            # keep the base policy: wx(i,i)=0, others +inf, vx(i,i)=+inf
            if i == j:
                re.wx_matrix.set(i, j, 0.0)
                re.vx_matrix.set(i, j, math.inf)
            # leave others as +inf per Step 11

    # Collapse case: l = k + 1
    for i in range(n):
        for j in range(i, n):
            for k in range(i, j):           # k in [i, j-1]
                l = k + 1                   # collapse
                w_val = get_whx_with_collapse(re.whx_matrix, re.wx_matrix, i, j, k, l)
                z_val = get_zhx_with_collapse(re.zhx_matrix, re.vx_matrix, i, j, k, l)
                assert w_val == re.wx_matrix.get(i, j)
                assert z_val == re.vx_matrix.get(i, j)

def test_sparse_gap_bounds_and_defaults():
    n = 5
    re = make_re_fold_state(n)

    # Valid but unfilled gap → +inf
    assert math.isinf(re.whx_matrix.get(0, 3, 1, 3))
    assert math.isinf(re.zhx_matrix.get(0, 4, 2, 4))

    # Out-of-bounds / invalid orders → +inf (via get() guards)
    assert math.isinf(re.whx_matrix.get(-1, 2, 0, 1))
    assert math.isinf(re.whx_matrix.get(1, 0, 0, 1))   # i > j
    assert math.isinf(re.whx_matrix.get(0, 3, 4, 2))   # k > l
    assert math.isinf(re.whx_matrix.get(0, 3, -1, 0))  # k < i (after guards)

    # Non-gap base cases are as expected
    for i in range(n):
        assert re.wx_matrix.get(i, i) == 0.0
        assert math.isinf(re.vx_matrix.get(i, i))
