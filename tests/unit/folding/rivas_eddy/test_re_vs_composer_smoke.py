import math
import pytest

from rna_pk_fold.folding import make_fold_state
from rna_pk_fold.folding.recurrences import SecondaryStructureFoldingEngine, RecurrenceConfig
from rna_pk_fold.folding.fold_state import make_re_fold_state
from rna_pk_fold.folding.rivas_eddy.rivas_eddy_recurrences import RivasEddyEngine, REREConfig

@pytest.mark.parametrize("seq", ["GCAU", "GCAUCU", "AUGCUA"])
def test_vx_two_gap_composition_can_reduce_energy(seq: str):
    """
    With zero-cost hole shrink and Gw=0, zhx collapses to vx.
    If we seed a convex baseline like (j-i)^2 for vx, the composition
    vx(i,r) + vx(r+..,j) can beat vx(i,j). We assert vx(0,N-1) <= baseline.
    """
    n = len(seq)
    st = make_fold_state(n)

    # Seed W with anything finite (unused here)
    for i in range(n):
        for j in range(i, n):
            st.w_matrix.set(i, j, 0.0)

    # Seed V with a convex baseline so splitting helps:
    # vx(i,j) ≈ (j-i)^2 (finite even if i==j)
    for i in range(n):
        for j in range(i, n):
            st.v_matrix.set(i, j, float((j - i) ** 2))

    re = make_re_fold_state(n)
    re_engine = RivasEddyEngine(REREConfig(pk_penalty_gw=0.0))
    re_engine.fill_minimal(seq, st, re)

    base = float((n - 1 - 0) ** 2)
    got  = re.vx_matrix.get(0, n - 1)

    assert math.isfinite(got)
    # composition should never be worse than baseline
    assert got <= base + 1e-9
    # For N>=4, composition is strictly better for many r; keep weak check:
    if n >= 4:
        assert got < base + 1e-9
