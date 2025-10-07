import math
import pytest

from rna_pk_fold.folding import make_fold_state
from rna_pk_fold.folding.recurrences import SecondaryStructureFoldingEngine, RecurrenceConfig
from rna_pk_fold.folding.fold_state import make_re_fold_state
from rna_pk_fold.folding.eddy_rivas.rivas_eddy_recurrences import RivasEddyEngine, REREConfig
from rna_pk_fold.folding.eddy_rivas.rivas_eddy_matrices import get_whx_with_collapse

@pytest.mark.parametrize("seq", ["GCAU", "GCAUCU", "AUGCUA"])
def test_fill_minimal_makes_whx_finite_via_collapse(seq: str):
    """
    After running the minimal R&E filler, any WHX(i,j:k,l) with l>k+1 should be finite
    (equal to wx(i,j) here, since hole-shrink is zero-cost and collapses to wx).
    """
    n = len(seq)
    # 1) Run your nested Zuker engine to populate W/V (used as anchors)
    st = make_fold_state(n)
    engine = SecondaryStructureFoldingEngine(
        energy_model=None,  # if your engine requires a model, pass your real fixture here
        config=RecurrenceConfig()
    )
    # If your engine requires a real energy model, replace the None above and fill matrices:
    # engine.fill_all_matrices(seq, st)

    # For a pure smoke test, we set a simple finite baseline on W: wx ≈ W
    # If you have real nested results, comment these lines out.
    for i in range(n):
        for j in range(i, n):
            st.w_matrix.set(i, j, float(j - i))  # simple finite baseline
            st.v_matrix.set(i, j, math.inf)      # leave V unused in this smoke

    # 2) Build R&E state and run minimal filler
    re = make_re_fold_state(n)
    re_engine = RivasEddyEngine(REREConfig(pk_penalty_gw=0.0))  # penalty 0 to make it easy to see
    re_engine.fill_minimal(seq, st, re)

    # 3) Pick a couple of non-collapsed holes and ensure WHX equals WX(i,j)
    # e.g., outer [0..n-1], inner (k,l) with l = k+2 (width=1)
    if n >= 4:
        i, j = 0, n - 1
        k, l = 1, 3
        whx = get_whx_with_collapse(re.whx_matrix, re.wx_matrix, i, j, k, l)
        wx  = re.wx_matrix.get(i, j)
        assert math.isfinite(whx)
        assert whx == wx, "With zero-cost shrink, WHX should collapse to WX"

    # also try a smaller outer span
    if n >= 5:
        i, j = 1, n - 2
        k, l = i, j - 1
        if l >= k + 2:
            whx = get_whx_with_collapse(re.whx_matrix, re.wx_matrix, i, j, k, l)
            wx  = re.wx_matrix.get(i, j)
            assert math.isfinite(whx)
            assert whx == wx
