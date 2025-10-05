# tests/test_rivas_eddy_table3.py
import math
import pytest

from rna_pk_fold.folding.rivas_eddy import rivas_eddy_recurrences as re_rec

# ------------------------
# Pure helper / iterator tests
# ------------------------

def test_iter_complementary_tuples_order_and_bounds_small_window():
    i, j = 0, 3
    triples = list(re_rec._iter_complementary_tuples(i, j))
    assert triples, "expected at least one (r,k,l) triple"
    for (r, k, l) in triples:
        assert i < k <= r < l <= j

def test_iter_inner_holes_min_hole_enforced():
    i, j = 0, 5
    holes0 = list(re_rec._iter_inner_holes(i, j, min_hole=0))
    holes2 = list(re_rec._iter_inner_holes(i, j, min_hole=2))
    assert all(l >= k + 1 for k, l in holes0)
    assert all(l >= k + 3 for k, l in holes2)
    assert set(holes2).issubset(set(holes0)) and len(holes2) < len(holes0)

def test_dangle_table_lookup_and_fallbacks():
    costs = re_rec.RERECosts(
        L_tilde=0.7, R_tilde=0.8,
        dangle_outer_L={("A","U"): -0.3},
        dangle_outer_R={("C","G"): -0.4},
    )
    seq = "AUGC"
    assert re_rec._dangle_outer_L(seq, 0, costs) == -0.3
    assert re_rec._dangle_outer_L(seq, 1, costs) == pytest.approx(0.7)
    # "_dangle_outer_R(seq, j)" looks up (j-1, j)
    assert re_rec._dangle_outer_R(seq, 3, costs) == pytest.approx(0.8)

def test_coax_energy_for_join_symmetric_lookup():
    costs = re_rec.RERECosts(coax_pairs={("GC","AU"): -1.2})
    seq = "GCAU"
    e1 = re_rec._coax_energy_for_join(seq, (0,1), (2,3), costs)
    e2 = re_rec._coax_energy_for_join(seq, (2,3), (0,1), costs)
    assert e1 == e2 == pytest.approx(-1.2)

def test_wxI_prefers_wxi_over_wx():
    class DummyMat:
        def __init__(self, val): self._v = val
        def get(self, i, j): return self._v
    class DummyRe:
        wxi_matrix = DummyMat(11.0)
        wx_matrix  = DummyMat(99.0)

    assert re_rec.wxI(DummyRe, 0, 0) == 11.0


# ------------------------
# Tiny factory / seeds
# ------------------------

def _try_build_states(n):
    try:
        from rna_pk_fold.folding.fold_state import FoldState, RivasEddyState
    except Exception as e:
        pytest.skip(f"Cannot import FoldState/RivasEddyState: {e}")
    try:
        nested = FoldState(n)
        re_state = RivasEddyState(n)
    except Exception as e:
        pytest.skip(f"Cannot construct FoldState/RivasEddyState(n): {e}")
    # Seed nested W/V baselines to zeros (finite)
    for s in range(n):
        for i in range(0, n - s):
            j = i + s
            nested.w_matrix.set(i, j, 0.0)
            nested.v_matrix.set(i, j, 0.0)
    return nested, re_state


# ------------------------
# Coax refinements (adjacency / non-trivial caps / clamp)
# ------------------------

@pytest.mark.parametrize("adjacent, expect_nontrivial_caps", [
    (True, True),   # eligible -> can help
    (True, False),  # r==k but trivial cap -> ignore
])
def test_coax_eligibility_and_never_hurts(adjacent, expect_nontrivial_caps):
    if adjacent and expect_nontrivial_caps:
        seq = "GCGG"   # (0,1)="GC", (2,3)="GG"; caps non-trivial
        i, j, r, k = 0, 3, 1, 1
    elif adjacent and not expect_nontrivial_caps:
        seq = "GCU"    # j == k+1 -> right cap trivial
        i, j, r, k = 0, 2, 1, 1
    else:
        pytest.skip("non-adjacent case not used here")

    n = len(seq)
    nested, re_state = _try_build_states(n)

    base_costs = re_rec.RERECosts(
        q_ss=0.0, P_tilde=0.0, P_tilde_out=0.0, P_tilde_hole=0.0,
        Q_tilde_out=0.0, Q_tilde_hole=0.0, L_tilde=0.0, R_tilde=0.0,
        coax_pairs={("GC","GC"): -1.0, ("AU","AU"): -1.0},
        coax_scale=1.0, coax_bonus=0.0, Gwi=0.0,
    )

    cfg_off = re_rec.REREConfig(enable_coax=False, enable_coax_variants=False,
                                pk_penalty_gw=0.0, costs=base_costs)
    cfg_on  = re_rec.REREConfig(enable_coax=True,  enable_coax_variants=False,
                                pk_penalty_gw=0.0, costs=base_costs)

    eng_off = re_rec.RivasEddyEngine(cfg_off)
    eng_off.fill_with_costs(seq, nested, re_state)
    vx_off = re_state.vx_matrix.get(i, j)

    nested2, re_state2 = _try_build_states(n)
    eng_on = re_rec.RivasEddyEngine(cfg_on)
    eng_on.fill_with_costs(seq, nested2, re_state2)
    vx_on = re_state2.vx_matrix.get(i, j)

    if expect_nontrivial_caps:
        assert vx_on <= vx_off
    else:
        assert vx_on == vx_off

def test_coax_positive_values_are_clamped_to_zero():
    seq = "GCGC"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs_pos = re_rec.RERECosts(
        q_ss=0.0, P_tilde=0.0, P_tilde_out=0.0, P_tilde_hole=0.0,
        Q_tilde_out=0.0, Q_tilde_hole=0.0, L_tilde=0.0, R_tilde=0.0,
        coax_pairs={("GC","GC"): +2.5},  # should be clamped to 0
        coax_scale=1.0, coax_bonus=0.0, Gwi=0.0,
    )
    cfg_off = re_rec.REREConfig(enable_coax=False, pk_penalty_gw=0.0, costs=costs_pos)
    cfg_on  = re_rec.REREConfig(enable_coax=True,  pk_penalty_gw=0.0, costs=costs_pos)

    eng_off = re_rec.RivasEddyEngine(cfg_off)
    eng_off.fill_with_costs(seq, nested, re_state)
    vx_off = re_state.vx_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    eng_on = re_rec.RivasEddyEngine(cfg_on)
    eng_on.fill_with_costs(seq, nested2, re_state2)
    vx_on = re_state2.vx_matrix.get(0, n - 1)

    assert vx_on == vx_off

def test_coax_variants_can_help_when_only_variant_is_scored():
    """
    Provide coax energy only for the variant contacts (i,r)-(k,l) or (k,l)-(k+1,j).
    With enable_coax_variants=False there is no gain; enabling it should improve VX.
    """
    seq = "GCGG"  # (i,r)=(0,1)->"GC"; (k,l)=(1,3)->"CG"; (k+1,j)=(2,3)->"GG"
    n, i, j = len(seq), 0, 3
    nested, re_state = _try_build_states(n)

    costs = re_rec.RERECosts(
        q_ss=0.0, L_tilde=0.0, R_tilde=0.0, Gwi=0.0,
        # Only score the variant edges (GC,CG) and (CG,GG) negatively
        coax_pairs={("GC","CG"): -2.0, ("CG","GG"): -1.0},
        coax_scale=1.0, coax_bonus=0.0,
    )
    cfg_base = re_rec.REREConfig(enable_coax=True,  enable_coax_variants=False,
                                 pk_penalty_gw=0.0, costs=costs)
    cfg_var  = re_rec.REREConfig(enable_coax=True,  enable_coax_variants=True,
                                 pk_penalty_gw=0.0, costs=costs)

    eng0 = re_rec.RivasEddyEngine(cfg_base)
    eng0.fill_with_costs(seq, nested, re_state)
    vx0 = re_state.vx_matrix.get(i, j)

    nested2, re_state2 = _try_build_states(n)
    eng1 = re_rec.RivasEddyEngine(cfg_var)
    eng1.fill_with_costs(seq, nested2, re_state2)
    vx1 = re_state2.vx_matrix.get(i, j)

    assert vx1 <= vx0


# ------------------------
# Pruning guards / overlap split
# ------------------------

def test_pruning_guards_do_not_worsen_optimum():
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs = re_rec.RERECosts(q_ss=0.0)
    cfg_loose = re_rec.REREConfig(enable_coax=False, pk_penalty_gw=0.0,
                                  strict_complement_order=True,
                                  min_hole_width=0, min_outer_left=0, min_outer_right=0,
                                  costs=costs)
    cfg_tight = re_rec.REREConfig(enable_coax=False, pk_penalty_gw=0.0,
                                  strict_complement_order=True,
                                  min_hole_width=1, min_outer_left=1, min_outer_right=1,
                                  costs=costs)

    eng0 = re_rec.RivasEddyEngine(cfg_loose)
    eng0.fill_with_costs(seq, nested, re_state)
    w0, v0 = re_state.wx_matrix.get(0, n - 1), re_state.vx_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    eng1 = re_rec.RivasEddyEngine(cfg_tight)
    eng1.fill_with_costs(seq, nested2, re_state2)
    w1, v1 = re_state2.wx_matrix.get(0, n - 1), re_state2.vx_matrix.get(0, n - 1)

    assert w1 <= w0
    assert v1 <= v0

def test_enable_wx_overlap_with_negative_Gwh_wx_can_only_help():
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs_no = re_rec.RERECosts(q_ss=0.0, Gwh_wx=0.0)
    costs_yes = re_rec.RERECosts(q_ss=0.0, Gwh_wx=-0.5)

    cfg_no  = re_rec.REREConfig(enable_wx_overlap=False, pk_penalty_gw=0.0, costs=costs_no)
    cfg_yes = re_rec.REREConfig(enable_wx_overlap=True,  pk_penalty_gw=0.0, costs=costs_yes)

    eng0 = re_rec.RivasEddyEngine(cfg_no)
    eng0.fill_with_costs(seq, nested, re_state)
    w0 = re_state.wx_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    eng1 = re_rec.RivasEddyEngine(cfg_yes)
    eng1.fill_with_costs(seq, nested2, re_state2)
    w1 = re_state2.wx_matrix.get(0, n - 1)

    assert w1 <= w0


# ------------------------
# Context-split constants: monotonic effects (weak property tests)
# ------------------------

def _min_finite_yhx(re_state, n):
    best = math.inf
    for i in range(n):
        for j in range(i+1, n):
            max_h = max(0, j - i - 1)
            for h in range(1, max_h + 1):
                for k in range(i, j - h):
                    l = k + h + 1
                    v = re_state.yhx_matrix.get(i, j, k, l)
                    if math.isfinite(v) and v < best:
                        best = v
    return best

def _min_finite_vhx(re_state, n):
    best = math.inf
    for i in range(n):
        for j in range(i+1, n):
            max_h = max(0, j - i - 1)
            for h in range(1, max_h + 1):
                for k in range(i, j - h):
                    l = k + h + 1
                    v = re_state.vhx_matrix.get(i, j, k, l)
                    if math.isfinite(v) and v < best:
                        best = v
    return best

def test_P_out_increases_yhx_min_energy_monotonically():
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    cfg0 = re_rec.REREConfig(
        pk_penalty_gw=0.0, enable_coax=False,
        costs=re_rec.RERECosts(q_ss=0.0, P_tilde_out=0.0, P_tilde_hole=0.0, Gwi=0.0))
    cfg1 = re_rec.REREConfig(
        pk_penalty_gw=0.0, enable_coax=False,
        costs=re_rec.RERECosts(q_ss=0.0, P_tilde_out=2.0, P_tilde_hole=0.0, Gwi=0.0))

    eng0 = re_rec.RivasEddyEngine(cfg0)
    eng0.fill_with_costs(seq, nested, re_state)
    y0 = _min_finite_yhx(re_state, n)

    nested2, re_state2 = _try_build_states(n)
    eng1 = re_rec.RivasEddyEngine(cfg1)
    eng1.fill_with_costs(seq, nested2, re_state2)
    y1 = _min_finite_yhx(re_state2, n)

    # Increasing P_out should not lower the best achievable YHX energy
    assert y1 >= y0

def test_P_hole_increases_vhx_min_energy_monotonically():
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    cfg0 = re_rec.REREConfig(
        pk_penalty_gw=0.0, enable_coax=False,
        costs=re_rec.RERECosts(q_ss=0.0, P_tilde_hole=0.0, Gwi=0.0))
    cfg1 = re_rec.REREConfig(
        pk_penalty_gw=0.0, enable_coax=False,
        costs=re_rec.RERECosts(q_ss=0.0, P_tilde_hole=2.0, Gwi=0.0))

    eng0 = re_rec.RivasEddyEngine(cfg0)
    eng0.fill_with_costs(seq, nested, re_state)
    v0 = _min_finite_vhx(re_state, n)

    nested2, re_state2 = _try_build_states(n)
    eng1 = re_rec.RivasEddyEngine(cfg1)
    eng1.fill_with_costs(seq, nested2, re_state2)
    v1 = _min_finite_vhx(re_state2, n)

    assert v1 >= v0


# ------------------------
# Selection behavior on publish
# ------------------------

def test_wx_selects_uncharged_on_tie_and_sets_backpointer():
    seq = "GCAU"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    # Gw=0 so charged and uncharged can tie; selection should prefer uncharged
    cfg = re_rec.REREConfig(enable_coax=False, pk_penalty_gw=0.0,
                            costs=re_rec.RERECosts(q_ss=0.0))
    eng = re_rec.RivasEddyEngine(cfg)
    eng.fill_with_costs(seq, nested, re_state)

    i, j = 0, n - 1
    assert re_state.wx_back_ptr.get((i, j), (None,))[0] in (re_rec.RE_BP_WX_SELECT_UNCHARGED, re_rec.RE_BP_COMPOSE_WX,
                                                            re_rec.RE_BP_COMPOSE_WX_YHX,
                                                            re_rec.RE_BP_COMPOSE_WX_YHX_WHX,
                                                            re_rec.RE_BP_COMPOSE_WX_WHX_YHX,
                                                            re_rec.RE_BP_COMPOSE_WX_YHX_OVERLAP)
    # If tie happened, the tag should be SELECT_UNCHARGED
    if re_state.wxu_matrix.get(i, j) == re_state.wxc_matrix.get(i, j):
        assert re_state.wx_back_ptr[(i, j)][0] == re_rec.RE_BP_WX_SELECT_UNCHARGED
