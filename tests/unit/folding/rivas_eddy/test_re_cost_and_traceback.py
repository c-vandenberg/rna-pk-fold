import math
import pytest

from rna_pk_fold.folding.fold_state import make_re_fold_state, RivasEddyState
from rna_pk_fold.folding import make_fold_state, BackPointer, BacktrackOp
from rna_pk_fold.folding.eddy_rivas.rivas_eddy_recurrences import (
    RivasEddyEngine, REREConfig, RERECosts,
    RE_BP_COMPOSE_WX,
)
from rna_pk_fold.folding.eddy_rivas.rivas_eddy_matrices import get_whx_with_collapse
from rna_pk_fold.folding.eddy_rivas.rivas_eddy_traceback import traceback_re_with_pk
from rna_pk_fold.folding.traceback import traceback_nested_interval, TraceResult
from rna_pk_fold.structures import Pair


# ---------- 1) whx gap costs accumulate with hole width ----------

@pytest.mark.parametrize("n, i, j, k, l, q", [
    # N=8, outer [0..7], hole (1,5) => width = 5-1-1 = 3 → expect 3*q
    (8, 0, 7, 1, 5, 0.5),
    (8, 0, 7, 1, 5, 0.2),
    # Another shape: outer [1..6], hole (2,5) => width = 2
    (8, 1, 6, 2, 5, 0.3),
])
def test_whx_cost_accumulates_with_hole_shrink(n, i, j, k, l, q):
    # Seed a nested state where wx ≈ W; set W = 0 for all (keeps algebra simple)
    nested = make_fold_state(n)
    for a in range(n):
        for b in range(a, n):
            nested.w_matrix.set(a, b, 0.0)
            nested.v_matrix.set(a, b, math.inf)

    # Build R&E state and run costful fill
    re = make_re_fold_state(n)
    cfg = REREConfig(pk_penalty_gw=0.0, costs=RERECosts(q_ss=q, coax_bonus=0.0))
    RivasEddyEngine(cfg).fill_with_costs(seq="N"*n, nested=nested, re=re)

    # Expected: whx(i,j:k,l) = wx(i,j) + q * (l - k - 1)
    h = l - k - 1
    expected = nested.w_matrix.get(i, j) + q * h
    got = re.whx_matrix.get(i, j, k, l)

    assert math.isfinite(got), "whx should be finite after costful fill"
    assert got == pytest.approx(expected, abs=1e-9)


# ---------- 2) two-gap composition into wx beats baseline and sets backpointer ----------

def test_wx_two_gap_composition_improves_and_sets_bp():
    n = 6
    nested = make_fold_state(n)

    # Baseline: big only at (0,5); zero elsewhere
    for a in range(n):
        for b in range(a, n):
            nested.w_matrix.set(a, b, 0.0)
            nested.v_matrix.set(a, b, math.inf)
    nested.w_matrix.set(0, 5, 10.0)  # target to potentially beat

    re = make_re_fold_state(n)
    cfg = REREConfig(pk_penalty_gw=0.0, costs=RERECosts(q_ss=0.5, coax_bonus=0.0))
    RivasEddyEngine(cfg).fill_with_costs(seq="N"*n, nested=nested, re=re)

    wx_baseline = 10.0
    wx_05 = re.wx_matrix.get(0, 5)
    bp = re.wx_back_ptr.get((0, 5))

    if bp is not None:
        # If composition was actually used, it must strictly improve the baseline.
        assert wx_05 < wx_baseline, f"Composition chosen but did not improve: {wx_05} !< {wx_baseline}"
        assert bp[0] == RE_BP_COMPOSE_WX
    else:
        # Otherwise, we expect to keep the baseline.
        assert wx_05 == pytest.approx(wx_baseline, abs=1e-9)


# ---------- 3) R&E traceback composes two nested intervals into layered brackets ----------

def test_re_traceback_layers_and_subintervals():
    """
    Force a composed WX(0,5) using (r,k,l)=(1,0,2), then
    stub nested backpointers so:
      - left interval (0..1) produces pair (0,1),
      - right interval (1..5) eventually produces pair (2,5).
    Expect layered dot-bracket with () at (0,1) and [] at (2,5).
    """
    n = 6
    nested = make_fold_state(n)

    # Seed W energies (not strictly needed for traceback)
    for a in range(n):
        for b in range(a, n):
            nested.w_matrix.set(a, b, 0.0)
            nested.v_matrix.set(a, b, math.inf)

    # --- Stub nested backpointers to produce two pairs ---
    # Left piece: W[0,1] -> PAIR → V[0,1] → (no-op in V branch still records (0,1))
    nested.w_back_ptr.set(0, 1, BackPointer(operation=BacktrackOp.PAIR))
    nested.v_back_ptr.set(0, 1, BackPointer(operation=BacktrackOp.NONE))

    # Right piece: start at W[1,5] → UNPAIRED_LEFT → W[2,5] → PAIR → V[2,5]
    nested.w_back_ptr.set(1, 5, BackPointer(operation=BacktrackOp.UNPAIRED_LEFT))
    nested.w_back_ptr.set(2, 5, BackPointer(operation=BacktrackOp.PAIR))
    nested.v_back_ptr.set(2, 5, BackPointer(operation=BacktrackOp.NONE))

    # Build R&E state that claims composition at top (0,5)
    re = make_re_fold_state(n)
    # Give wx a value consistent with an improvement
    re.wx_matrix.set(0, 5, 0.5)
    re.wx_back_ptr[(0, 5)] = ("RE_PK_COMPOSE_WX", (1, 0, 2))  # r=1, k=0, l=2

    # Run RE traceback
    tr = traceback_re_with_pk(seq="N"*n, nested=nested, re=re)

    # Expect two pairs: (0,1) layer 0 → '(' ')', and (2,5) layer 1 → '[' ']'
    want_pairs = {Pair(0, 1), Pair(2, 5)}
    got_pairs = set(tr.pairs)
    assert want_pairs <= got_pairs, f"pairs missing: want {want_pairs}, got {got_pairs}"

    # Check bracket layers in the rendered string
    db = tr.dot_bracket
    assert len(db) == n
    assert db[0] == '(' and db[1] == ')', f"left piece not on layer 0: {db}"
    assert db[2] == '[' and db[5] == ']', f"right piece not on layer 1: {db}"


# ---------- 4) Composition shouldn’t win if Gw is large ----------

def test_no_composition_when_gw_large():
    n = 6
    nested = make_fold_state(n)

    # Baseline outer is small (1.0); composition would be >= Gw, so make Gw huge.
    for a in range(n):
        for b in range(a, n):
            nested.w_matrix.set(a, b, 0.0)
            nested.v_matrix.set(a, b, math.inf)
    nested.w_matrix.set(0, 5, 1.0)

    re = make_re_fold_state(n)
    cfg = REREConfig(pk_penalty_gw=100.0, costs=RERECosts(q_ss=0.5, coax_bonus=0.0))
    RivasEddyEngine(cfg).fill_with_costs(seq="N"*n, nested=nested, re=re)

    assert re.wx_matrix.get(0, 5) == pytest.approx(1.0, abs=1e-9)
    # No backpointer expected when baseline wins
    assert re.wx_back_ptr.get((0, 5)) is None
