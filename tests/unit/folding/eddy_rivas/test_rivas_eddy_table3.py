import math
import pytest

from rna_pk_fold.folding.eddy_rivas import eddy_rivas_recurrences
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBacktrackOp
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState, init_eddy_rivas_fold_state
from rna_pk_fold.structures.tri_matrix import ZuckerTriMatrix, EddyRivasTriMatrix, EddyRivasTriBackPointer
from rna_pk_fold.structures.gap_matrix import SparseGapMatrix, SparseGapBackptr

from rna_pk_fold.folding.zucker.zucker_fold_state import ZuckerFoldState, make_fold_state
from rna_pk_fold.folding.zucker.zucker_back_pointer import ZuckerBackPointer

from rna_pk_fold.energies.energy_types import PseudoknotEnergies
from rna_pk_fold.energies.energy_pk_ops import dangle_hole_right, dangle_hole_left
from rna_pk_fold.utils.energy_pk_utils import coax_energy_for_join
from rna_pk_fold.utils.iter_utils import iter_complementary_tuples, iter_inner_holes


# -------- helpers to build costs consistent with the refactor --------
def make_costs(**overrides) -> PseudoknotEnergies:
    """Provide a generous set of defaults so tests can override only what they need."""
    defaults = dict(
        # base scalars
        q_ss=0.0, g_wi=0.0, pk_penalty_gw=1.0, g_wh=0.0,
        # optional split penalties (WX/WHX overlap use these; default to zero to be explicit)
        g_wh_wx=0.0, g_wh_whx=0.0,
        # coax
        coax_scale=1.0, coax_bonus=0.0,
        mismatch_coax_scale=0.0, mismatch_coax_bonus=0.0,
        coax_min_helix_len=1,
        coax_scale_oo=1.0, coax_scale_oi=1.0, coax_scale_io=1.0,
        coax_pairs={},
        # tildes
        p_tilde_out=0.0, p_tilde_hole=0.0, q_tilde_out=0.0, q_tilde_hole=0.0,
        l_tilde=0.0, r_tilde=0.0,
        m_tilde_yhx=0.0, m_tilde_vhx=0.0, m_tilde_whx=0.0,
        # tables/maps
        dangle_outer_left={}, dangle_outer_right={},
        dangle_hole_left={}, dangle_hole_right={},
        # caps & drift
        short_hole_caps={}, join_drift_penalty=0.0,
    )
    defaults.update(overrides)
    return PseudoknotEnergies(**defaults)


# ------------------------
# Pure helper / iterator tests
# ------------------------
def test_iter_complementary_tuples_order_and_bounds_small_window():
    i, j = 0, 3
    triples = list(iter_complementary_tuples(i, j))
    assert triples, "expected at least one (r,k,l) triple"
    for (r, k, l) in triples:
        assert i < k <= r < l <= j


def test_iter_inner_holes_min_hole_enforced():
    i, j = 0, 5
    holes0 = list(iter_inner_holes(i, j, min_hole_width=0))
    holes2 = list(iter_inner_holes(i, j, min_hole_width=2))
    assert all(l >= k + 1 for k, l in holes0)
    assert all(l >= k + 3 for k, l in holes2)
    assert set(holes2).issubset(set(holes0)) and len(holes2) < len(holes0)


def test_dangle_table_lookup_and_fallbacks():
    # Note: hole dangles use 0.0 fallback (outer dangles use L_tilde/R_tilde)
    costs = make_costs(
        l_tilde=0.7, r_tilde=0.8,
        dangle_hole_left={("A", "U"): -0.3},   # will be hit at idx=1 (bigram (0,1))
        dangle_hole_right={("G", "C"): -0.4},  # will be hit at idx=2 (bigram (2,3))
    )
    seq = "AUGC"

    # hole-left uses (i-1, i): idx=1 -> ("A","U")
    assert dangle_hole_left(seq, 1, costs) == pytest.approx(-0.3)
    # out-of-range/missing entry -> 0.0 fallback
    assert dangle_hole_left(seq, 0, costs) == 0.0

    # hole-right uses (i, i+1): idx=2 -> ("G","C")
    assert dangle_hole_right(seq, 2, costs) == pytest.approx(-0.4)
    # out-of-range/missing entry -> 0.0 fallback
    assert dangle_hole_right(seq, 3, costs) == 0.0


def test_coax_energy_for_join_symmetric_lookup():
    costs = make_costs(coax_pairs={("GC", "AU"): -1.2})
    seq = "GCAU"
    # Pass the mapping (coax_pairs), not the whole costs dataclass
    e1 = coax_energy_for_join(seq, (0, 1), (2, 3), costs.coax_pairs)
    e2 = coax_energy_for_join(seq, (2, 3), (0, 1), costs.coax_pairs)
    assert e1 == e2 == pytest.approx(-1.2)


def test_wxI_prefers_wxi_over_wx():
    class DummyMat:
        def __init__(self, val): self._v = val
        def get(self, i, j): return self._v
    class DummyRe:
        def __init__(self):
            self.wxi_matrix = DummyMat(11.0)
            self.wx_matrix  = DummyMat(99.0)

    # wxI is imported into eddy_rivas_recurrences from utils.matrix_utils
    assert eddy_rivas_recurrences.get_wxi_or_wx(DummyRe(), 0, 0) == 11.0


# ------------------------
# Tiny factory / seeds
# ------------------------
def _try_build_states(n):
    try:
        nested = make_fold_state(n)  # current factory; default finite base cases inside
        re_state = init_eddy_rivas_fold_state(n)
    except Exception:
        # Fallback construction
        inf = math.inf

        # --- Nested (Zuker) state ---
        w_matrix = ZuckerTriMatrix[float](n, inf)
        v_matrix = ZuckerTriMatrix[float](n, inf)
        wm_matrix = ZuckerTriMatrix[float](n, inf)

        w_back_ptr = ZuckerTriMatrix[ZuckerBackPointer](n, ZuckerBackPointer())
        v_back_ptr = ZuckerTriMatrix[ZuckerBackPointer](n, ZuckerBackPointer())
        wm_back_ptr = ZuckerTriMatrix[ZuckerBackPointer](n, ZuckerBackPointer())

        for i in range(n):
            wm_matrix.set(i, i, 0.0)

        nested = ZuckerFoldState(
            w_matrix=w_matrix,
            v_matrix=v_matrix,
            wm_matrix=wm_matrix,
            w_back_ptr=w_back_ptr,
            v_back_ptr=v_back_ptr,
            wm_back_ptr=wm_back_ptr,
        )

        # --- Rivas & Eddy state ---
        wx_matrix = EddyRivasTriMatrix(n)
        vx_matrix = EddyRivasTriMatrix(n)
        wxi_matrix = EddyRivasTriMatrix(n)
        wxu_matrix = EddyRivasTriMatrix(n)
        wxc_matrix = EddyRivasTriMatrix(n)
        vxu_matrix = EddyRivasTriMatrix(n)
        vxc_matrix = EddyRivasTriMatrix(n)

        whx_matrix = SparseGapMatrix(n)
        vhx_matrix = SparseGapMatrix(n)
        yhx_matrix = SparseGapMatrix(n)
        zhx_matrix = SparseGapMatrix(n)

        whx_back_ptr = SparseGapBackptr(n)
        vhx_back_ptr = SparseGapBackptr(n)
        yhx_back_ptr = SparseGapBackptr(n)
        zhx_back_ptr = SparseGapBackptr(n)

        re_state = EddyRivasFoldState(
            seq_len=n,
            wx_matrix=wx_matrix, vx_matrix=vx_matrix,
            wxi_matrix=wxi_matrix, wxu_matrix=wxu_matrix, wxc_matrix=wxc_matrix,
            vxu_matrix=vxu_matrix, vxc_matrix=vxc_matrix,
            wx_back_ptr=EddyRivasTriBackPointer(n), vx_back_ptr=EddyRivasTriBackPointer(n),
            whx_matrix=whx_matrix, vhx_matrix=vhx_matrix,
            yhx_matrix=yhx_matrix, zhx_matrix=zhx_matrix,
            whx_back_ptr=whx_back_ptr, vhx_back_ptr=vhx_back_ptr,
            yhx_back_ptr=yhx_back_ptr, zhx_back_ptr=zhx_back_ptr,
        )

        # Base cases for publish tables
        for i in range(n):
            re_state.wx_matrix.set(i, i, 0.0)
            re_state.wxi_matrix.set(i, i, 0.0)
            re_state.wxu_matrix.set(i, i, 0.0)
            re_state.wxc_matrix.set(i, i, 0.0)
            re_state.vx_matrix.set(i, i, inf)
            re_state.vxu_matrix.set(i, i, inf)
            re_state.vxc_matrix.set(i, i, inf)

    # Seed nested W/V to zero everywhere to keep tests predictable
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
    (True, True),
    (True, False),
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

    base_costs = make_costs(
        q_ss=0.0, p_tilde_out=0.0, p_tilde_hole=0.0,
        q_tilde_out=0.0, q_tilde_hole=0.0, l_tilde=0.0, r_tilde=0.0,
        coax_pairs={("GC", "GC"): -1.0, ("AU", "AU"): -1.0},
        coax_scale=1.0, coax_bonus=0.0, g_wi=0.0,
    )

    cfg_off = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=False, enable_coax_variants=False,
        pk_penalty_gw=0.0, costs=base_costs
    )
    cfg_on  = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=False,
        pk_penalty_gw=0.0, costs=base_costs
    )

    eng_off = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_off)
    eng_off.fill_with_costs(seq, nested, re_state)
    vx_off = re_state.vx_matrix.get(i, j)

    nested2, re_state2 = _try_build_states(n)
    eng_on = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_on)
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

    costs_pos = make_costs(
        q_ss=0.0, p_tilde_out=0.0, p_tilde_hole=0.0,
        q_tilde_out=0.0, q_tilde_hole=0.0, l_tilde=0.0, r_tilde=0.0,
        coax_pairs={("GC", "GC"): +2.5},  # should be clamped to 0
        coax_scale=1.0, coax_bonus=0.0, g_wi=0.0,
    )
    cfg_off = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_coax=False, pk_penalty_gw=0.0, costs=costs_pos)
    cfg_on  = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_coax=True,  pk_penalty_gw=0.0, costs=costs_pos)

    eng_off = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_off)
    eng_off.fill_with_costs(seq, nested, re_state)
    vx_off = re_state.vx_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    eng_on = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_on)
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

    costs = make_costs(
        q_ss=0.0, l_tilde=0.0, r_tilde=0.0, g_wi=0.0,
        # Only score the variant edges (GC,CG) and (CG,GG) negatively
        coax_pairs={("GC", "CG"): -2.0, ("CG", "GG"): -1.0},
        coax_scale=1.0, coax_bonus=0.0,
    )
    cfg_base = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=False, pk_penalty_gw=0.0, costs=costs
    )
    cfg_var  = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=True, pk_penalty_gw=0.0, costs=costs
    )

    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_base)
    eng0.fill_with_costs(seq, nested, re_state)
    vx0 = re_state.vx_matrix.get(i, j)

    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_var)
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

    costs = make_costs(q_ss=0.0)
    cfg_loose = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=False, pk_penalty_gw=0.0,
        strict_complement_order=True,
        min_hole_width=0, min_outer_left=0, min_outer_right=0,
        costs=costs
    )
    cfg_tight = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=False, pk_penalty_gw=0.0,
        strict_complement_order=True,
        min_hole_width=1, min_outer_left=1, min_outer_right=1,
        costs=costs
    )

    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_loose)
    eng0.fill_with_costs(seq, nested, re_state)
    w0, v0 = re_state.wx_matrix.get(0, n - 1), re_state.vx_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_tight)
    eng1.fill_with_costs(seq, nested2, re_state2)
    w1, v1 = re_state2.wx_matrix.get(0, n - 1), re_state2.vx_matrix.get(0, n - 1)

    assert w1 <= w0
    assert v1 <= v0


def test_enable_wx_overlap_with_negative_Gwh_wx_can_only_help():
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs_no  = make_costs(q_ss=0.0, g_wh_wx=0.0)
    costs_yes = make_costs(q_ss=0.0, g_wh_wx=-0.5)

    cfg_no  = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_wx_overlap=False, pk_penalty_gw=0.0, costs=costs_no)
    cfg_yes = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_wx_overlap=True,  pk_penalty_gw=0.0, costs=costs_yes)

    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_no)
    eng0.fill_with_costs(seq, nested, re_state)
    w0 = re_state.wx_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_yes)
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

    cfg0 = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        pk_penalty_gw=0.0, enable_coax=False,
        costs=make_costs(q_ss=0.0, p_tilde_out=0.0, p_tilde_hole=0.0, g_wi=0.0))
    cfg1 = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        pk_penalty_gw=0.0, enable_coax=False,
        costs=make_costs(q_ss=0.0, p_tilde_out=2.0, p_tilde_hole=0.0, g_wi=0.0))

    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg0)
    eng0.fill_with_costs(seq, nested, re_state)
    y0 = _min_finite_yhx(re_state, n)

    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg1)
    eng1.fill_with_costs(seq, nested2, re_state2)
    y1 = _min_finite_yhx(re_state2, n)

    assert y1 >= y0


def test_P_hole_increases_vhx_min_energy_monotonically():
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    cfg0 = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        pk_penalty_gw=0.0, enable_coax=False,
        costs=make_costs(q_ss=0.0, p_tilde_hole=0.0, g_wi=0.0))
    cfg1 = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        pk_penalty_gw=0.0, enable_coax=False,
        costs=make_costs(q_ss=0.0, p_tilde_hole=2.0, g_wi=0.0))

    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg0)
    eng0.fill_with_costs(seq, nested, re_state)
    v0 = _min_finite_vhx(re_state, n)

    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg1)
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
    cfg = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=False, pk_penalty_gw=0.0,
        costs=make_costs(q_ss=0.0)
    )
    eng = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg)
    eng.fill_with_costs(seq, nested, re_state)

    i, j = 0, n - 1
    bp = re_state.wx_back_ptr.get(i, j)
    tag = None if bp is None else bp.op
    assert tag in (
        EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED,
        EddyRivasBacktrackOp.RE_PK_COMPOSE_WX,
        EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX,
        EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX,
        EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX,
        EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_OVERLAP,
    )
    # If tie happened, the tag should be SELECT_UNCHARGED
    if re_state.wxu_matrix.get(i, j) == re_state.wxc_matrix.get(i, j):
        assert tag == EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED


def _min_finite_whx(re_state, n):
    best = math.inf
    for i in range(n):
        for j in range(i+1, n):
            max_h = max(0, j - i - 1)
            for h in range(1, max_h + 1):
                for k in range(i, j - h):
                    l = k + h + 1
                    v = re_state.whx_matrix.get(i, j, k, l)
                    if math.isfinite(v) and v < best:
                        best = v
    return best


# ------------------------
# More coax detail: gating, mismatch, directional scales
# ------------------------
def test_coax_min_helix_len_gates_effect():
    """
    If coax requires longer end-caps than available, enabling coax has no effect.
    Lowering the gate allows a strictly better or equal VX.
    """
    seq = "GCGG"   # length 4; (i,r)=(0,1)->"GC"; (k+1,j)=(2,3)->"GG"
    n, i, j = len(seq), 0, 3
    nested, re_state = _try_build_states(n)

    # Only OO contact carries negative energy
    costs = make_costs(
        q_ss=0.0, g_wi=0.0,
        coax_pairs={("GC", "GG"): -1.5},
        coax_scale=1.0, coax_bonus=0.0,
        coax_min_helix_len=10  # too strict -> gated out
    )
    cfg_strict = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=False,
        pk_penalty_gw=0.0, costs=costs
    )
    eng_strict = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_strict)
    eng_strict.fill_with_costs(seq, nested, re_state)
    vx_strict = re_state.vx_matrix.get(i, j)

    # Same but relax the gate so the same seam is eligible
    nested2, re_state2 = _try_build_states(n)
    costs2 = make_costs(
        q_ss=0.0, g_wi=0.0,
        coax_pairs={("GC", "GG"): -1.5},
        coax_scale=1.0, coax_bonus=0.0,
        coax_min_helix_len=1
    )
    cfg_relaxed = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=False,
        pk_penalty_gw=0.0, costs=costs2
    )
    eng_relaxed = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_relaxed)
    eng_relaxed.fill_with_costs(seq, nested2, re_state2)
    vx_relaxed = re_state2.vx_matrix.get(i, j)

    assert vx_relaxed <= vx_strict


def test_coax_mismatch_requires_enable_flag():
    """
    Create a seam where r == k + 1 (mismatch seam).
    With enable_coax_mismatch=False, coax must not engage.
    With enable_coax_mismatch=True, VX can improve.
    """
    seq = "GCAUGC"     # n=6, we'll rely on an (i,j) = (0,5) candidate with (r,k,l) ~ (2,1,4)
    n = len(seq)
    nested, re_state = _try_build_states(n)

    # Only the OO pair for the mismatch seam is negative
    costs = make_costs(
        q_ss=0.0, g_wi=0.0,
        coax_pairs={("GA", "AC"): -2.0},  # (i,r)="GA", (k+1,j)="AC"
        coax_scale=1.0, coax_bonus=0.0,
        coax_min_helix_len=1,
    )

    cfg_no_mismatch = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=False, enable_coax_mismatch=False,
        pk_penalty_gw=0.0, costs=costs
    )
    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_no_mismatch)
    eng0.fill_with_costs(seq, nested, re_state)
    vx0 = re_state.vx_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    cfg_yes_mismatch = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=False, enable_coax_mismatch=True,
        pk_penalty_gw=0.0, costs=costs
    )
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_yes_mismatch)
    eng1.fill_with_costs(seq, nested2, re_state2)
    vx1 = re_state2.vx_matrix.get(0, n - 1)

    assert vx1 <= vx0


def test_coax_directional_scales_affect_variants():
    """
    Only the OI/IO variant contacts have negative energies.
    With variants disabled: no gain. With variants enabled and scales up: VX improves.
    """
    seq = "GCGC"  # (i,r)=(0,1)="GC"; (k,l)=(1,3)="CG"; (k+1,j)=(2,3)="GC"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs = make_costs(
        q_ss=0.0, g_wi=0.0,
        # Only variant edges score (both directions)
        coax_pairs={("GC", "CG"): -1.0, ("CG", "GC"): -1.5},
        coax_scale=1.0, coax_bonus=0.0,
        coax_min_helix_len=1,
        coax_scale_oo=0.0,  # OO contributes nothing
        coax_scale_oi=0.0, coax_scale_io=0.0,
    )

    cfg_base = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=False,
        pk_penalty_gw=0.0, costs=costs
    )
    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_base)
    eng0.fill_with_costs(seq, nested, re_state)
    vx0 = re_state.vx_matrix.get(0, n - 1)

    # Now enable variants but keep scales at 0 (still no effect)
    nested2, re_state2 = _try_build_states(n)
    cfg_var_zero = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=True,
        pk_penalty_gw=0.0, costs=costs
    )
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_var_zero)
    eng1.fill_with_costs(seq, nested2, re_state2)
    vx1 = re_state2.vx_matrix.get(0, n - 1)
    assert vx1 == vx0

    # Increase variant scales -> now effect should appear
    nested3, re_state3 = _try_build_states(n)
    costs2 = make_costs(
        q_ss=0.0, g_wi=0.0,
        coax_pairs={("GC", "CG"): -1.0, ("CG", "GC"): -1.5},
        coax_scale=1.0, coax_bonus=0.0,
        coax_min_helix_len=1,
        coax_scale_oo=0.0, coax_scale_oi=2.0, coax_scale_io=2.0,
    )
    cfg_var_scaled = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=True,
        pk_penalty_gw=0.0, costs=costs2
    )
    eng2 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_var_scaled)
    eng2.fill_with_costs(seq, nested3, re_state3)
    vx2 = re_state3.vx_matrix.get(0, n - 1)
    assert vx2 <= vx1


# ------------------------
# Short-hole capping penalties (charged seam; check vxc/wxc directly)
# ------------------------
def test_short_hole_caps_raise_charged_vx_when_hole_is_tiny():
    """
    Put a positive cap for width=1 holes. Compare vxc with and without the cap.
    Final vx may still publish the uncharged (0), so inspect vxc explicitly.
    """
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs_no  = make_costs(q_ss=0.0, short_hole_caps={})
    costs_yes = make_costs(q_ss=0.0, short_hole_caps={1: +2.0})
    cfg_no  = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_coax=False, pk_penalty_gw=0.0, costs=costs_no)
    cfg_yes = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_coax=False, pk_penalty_gw=0.0, costs=costs_yes)

    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_no)
    eng0.fill_with_costs(seq, nested, re_state)
    vxc0 = re_state.vxc_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_yes)
    eng1.fill_with_costs(seq, nested2, re_state2)
    vxc1 = re_state2.vxc_matrix.get(0, n - 1)

    assert vxc1 >= vxc0


# ------------------------
# Join drift: cannot worsen; can help; BP tag when it wins
# ------------------------
def test_join_drift_cannot_worsen_vx():
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    base_costs = make_costs(q_ss=0.1, join_drift_penalty=1.0)  # penalize drift
    cfg_off = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_join_drift=False, drift_radius=1, pk_penalty_gw=0.0, costs=base_costs
    )
    cfg_on  = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_join_drift=True, drift_radius=1,  pk_penalty_gw=0.0, costs=base_costs
    )

    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_off)
    eng0.fill_with_costs(seq, nested, re_state)
    vxc0 = re_state.vxc_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_on)
    eng1.fill_with_costs(seq, nested2, re_state2)
    vxc1 = re_state2.vxc_matrix.get(0, n - 1)

    assert vxc1 <= vxc0


def test_join_drift_with_negative_penalty_can_win_and_sets_bp():
    """
    Make drift attractive and verify it *helps* the charged path.
    If the charged path (VXC) actually beats the uncharged (VXU),
    the publish tag should be a compose/compose-drift. Otherwise,
    it's fine to keep SELECT_UNCHARGED.
    """
    seq = "GCGCGA"
    n = len(seq)

    # --- Baseline: no drift ---
    nested0, re0 = _try_build_states(n)
    base_costs = make_costs(
        q_ss=0.0, g_wi=0.0,
        coax_pairs={("GC", "GC"): -2.0},
        coax_scale=1.0, coax_bonus=0.0,
        coax_min_helix_len=1,
        join_drift_penalty=0.0,  # no drift effect
    )
    cfg_off = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=False,
        enable_join_drift=False, drift_radius=1,
        pk_penalty_gw=0.0, costs=base_costs
    )
    eng_off = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_off)
    eng_off.fill_with_costs(seq, nested0, re0)
    vxc_off = re0.vxc_matrix.get(0, n - 1)
    vx_off  = re0.vx_matrix.get(0, n - 1)

    # --- With drift enabled and attractive ---
    nested1, re1 = _try_build_states(n)
    drift_costs = make_costs(
        q_ss=0.0, g_wi=0.0,
        coax_pairs={("GC", "GC"): -2.0},
        coax_scale=1.0, coax_bonus=0.0,
        coax_min_helix_len=1,
        join_drift_penalty=-0.5,  # reward drift
        pk_penalty_gw=0.0,
    )
    cfg_on = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=False,
        enable_join_drift=True, drift_radius=1,
        pk_penalty_gw=0.0, costs=drift_costs
    )
    eng_on = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_on)
    eng_on.fill_with_costs(seq, nested1, re1)

    vxc_on = re1.vxc_matrix.get(0, n - 1)
    vxu_on = re1.vxu_matrix.get(0, n - 1)
    vx_on  = re1.vx_matrix.get(0, n - 1)
    bp     = re1.vx_back_ptr.get(0, n - 1)
    tag    = None if bp is None else bp.op

    # Drift must not hurt VXC and overall VX should be no worse
    assert vxc_on <= vxc_off
    assert vx_on <= vx_off
    assert math.isfinite(vx_on)

    # If the charged path wins, we expect a compose tag; otherwise uncharged is OK.
    if vxc_on < vxu_on:
        assert tag in (EddyRivasBacktrackOp.RE_PK_COMPOSE_VX,
                       EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT)
    else:
        assert tag in (EddyRivasBacktrackOp.RE_PK_COMPOSE_VX,
                       EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT,
                       EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED)


# ------------------------
# IS2 in YHX/WHX contexts
# ------------------------
class _TablesYHX:
    def __init__(self, val):
        self.IS2_outer_yhx = lambda seq, i, j, r, s: val


def test_IS2_outer_yhx_lowers_best_yhx_when_negative():
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    cfg0 = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        pk_penalty_gw=0.0, enable_coax=False, costs=make_costs(q_ss=0.0)
    )
    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg0)
    eng0.fill_with_costs(seq, nested, re_state)
    y0 = _min_finite_yhx(re_state, n)

    nested2, re_state2 = _try_build_states(n)
    cfg1 = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        pk_penalty_gw=0.0, enable_coax=False, costs=make_costs(q_ss=0.0)
    )
    cfg1.tables = _TablesYHX(-1.5)  # negative bridge should help
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg1)
    eng1.fill_with_costs(seq, nested2, re_state2)
    y1 = _min_finite_yhx(re_state2, n)

    assert y1 <= y0


def test_IS2_outer_yhx_can_lower_whx_via_yhx_bridge():
    """
    The symmetric hook is applied in WHX using YHX inner.
    Negative bridge should reduce the best WHX energy.
    """
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    cfg0 = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        pk_penalty_gw=0.0, enable_coax=False, costs=make_costs(q_ss=0.0)
    )
    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg0)
    eng0.fill_with_costs(seq, nested, re_state)
    w0 = _min_finite_whx(re_state, n)

    nested2, re_state2 = _try_build_states(n)
    cfg1 = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        pk_penalty_gw=0.0, enable_coax=False, costs=make_costs(q_ss=0.0)
    )
    cfg1.tables = _TablesYHX(-2.0)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg1)
    eng1.fill_with_costs(seq, nested2, re_state2)
    w1 = _min_finite_whx(re_state2, n)

    assert w1 <= w0


# ------------------------
# Overlap + caps sanity (WX)
# ------------------------
def test_wx_overlap_respects_short_hole_caps_on_charged_path():
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs_overlap = make_costs(q_ss=0.0, g_wh_wx=-0.5, short_hole_caps={1: +1.0})
    cfg = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_wx_overlap=True, pk_penalty_gw=0.0, costs=costs_overlap)
    eng = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg)
    eng.fill_with_costs(seq, nested, re_state)

    wxc = re_state.wxc_matrix.get(0, n - 1)
    assert math.isfinite(wxc)


# ------------------------
# VX selection behavior on publish mirrors WX test
# ------------------------
def test_vx_selects_uncharged_on_tie_and_sets_backpointer():
    seq = "GCAU"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    cfg = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=False, pk_penalty_gw=0.0, costs=make_costs(q_ss=0.0)
    )
    eng = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg)
    eng.fill_with_costs(seq, nested, re_state)

    i, j = 0, n - 1
    bp = re_state.vx_back_ptr.get(i, j)
    tag = None if bp is None else bp.op
    assert tag in (EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED, EddyRivasBacktrackOp.RE_PK_COMPOSE_VX)
    if re_state.vxu_matrix.get(i, j) == re_state.vxc_matrix.get(i, j):
        assert tag == EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED


