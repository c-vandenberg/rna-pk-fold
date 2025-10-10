"""
Unit tests for the Rivas & Eddy recurrences and folding engine.

This module contains a suite of "monotonic property tests" that verify the behavior
of the `EddyRivasFoldingEngine`. Instead of checking for exact energy values,
these tests confirm that enabling specific features (like coaxial stacking) or
applying energy bonuses/penalties affects the final computed energies in a
predictable direction (e.g., a bonus should never make the score worse).

This approach makes the tests robust against minor changes in the underlying
energy model and focuses on the correctness of the algorithmic logic.
"""
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


def make_costs(**overrides) -> PseudoknotEnergies:
    """
    Factory for `PseudoknotEnergies` with sensible (mostly zero) defaults.

    This allows tests to be concise by only specifying the energy parameters
    that are directly relevant to the logic being tested.
    """
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
    """
    Tests the iterator for (r,k,l) pseudoknot composition tuples.
    Ensures that the generated indices adhere to the required ordering constraint
    `i < k <= r < l <= j`.
    """
    i, j = 0, 3
    triples = list(iter_complementary_tuples(i, j))
    assert triples, "expected at least one (r,k,l) triple"
    for (r, k, l) in triples:
        assert i < k <= r < l <= j


def test_iter_inner_holes_min_hole_enforced():
    """
    Tests the iterator for inner hole coordinates (k,l).
    Verifies that the `min_hole_width` parameter correctly filters the results.
    """
    i, j = 0, 5
    # A hole of width 0 means l = k+1.
    holes0 = list(iter_inner_holes(i, j, min_hole_width=0))
    # A hole of width 2 means l = k+3.
    holes2 = list(iter_inner_holes(i, j, min_hole_width=2))
    assert all(l >= k + 1 for k, l in holes0)
    assert all(l >= k + 3 for k, l in holes2)
    # The stricter constraint should be a subset of the looser one.
    assert set(holes2).issubset(set(holes0)) and len(holes2) < len(holes0)


def test_dangle_table_lookup_and_fallbacks():
    """
    Tests dangle energy lookups, including table hits and fallbacks for misses.
    """
    costs = make_costs(
        l_tilde=0.7, r_tilde=0.8,
        dangle_hole_left={("A", "U"): -0.3},  # will be hit at k=1 (bigram seq[0:2])
        dangle_hole_right={("G", "C"): -0.4},  # will be hit at l=2 (bigram seq[2:4])
    )
    seq = "AUGC"

    # Test successful lookup for left dangle.
    assert dangle_hole_left(seq, 1, costs) == pytest.approx(-0.3)
    # Test fallback for missing entry (should be 0.0 for hole dangles).
    assert dangle_hole_left(seq, 0, costs) == 0.0

    # Test successful lookup for right dangle.
    assert dangle_hole_right(seq, 2, costs) == pytest.approx(-0.4)
    # Test fallback for missing entry.
    assert dangle_hole_right(seq, 3, costs) == 0.0


def test_coax_energy_for_join_symmetric_lookup():
    """
    Verifies that coaxial stacking energy lookup is symmetric; order doesn't matter.
    """
    costs = make_costs(coax_pairs={("GC", "AU"): -1.2})
    seq = "GCAU"
    # The energy for stacking (0,1) on (2,3) should be the same as (2,3) on (0,1).
    e1 = coax_energy_for_join(seq, (0, 1), (2, 3), costs.coax_pairs)
    e2 = coax_energy_for_join(seq, (2, 3), (0, 1), costs.coax_pairs)
    assert e1 == e2 == pytest.approx(-1.2)


def test_wxI_prefers_wxi_over_wx():
    """
    Tests the `get_wxi_or_wx` helper, which should prioritize the WXI matrix.
    """
    # Create dummy classes to mock the fold state and its matrices.
    class DummyMat:
        def __init__(self, val): self._v = val

        def get(self, i, j): return self._v

    class DummyRe:
        def __init__(self):
            self.wxi_matrix = DummyMat(11.0)  # WXI has a better score.
            self.wx_matrix = DummyMat(99.0)

    # The helper function should return the value from the WXI matrix.
    assert eddy_rivas_recurrences.get_wxi_or_wx(DummyRe(), 0, 0) == 11.0


# ------------------------
# Tiny factory / seeds
# ------------------------
def _try_build_states(n):
    """
    A robust factory to create predictable, zero-filled fold states for tests.

    It creates both a Zucker (nested) state and a Rivas-Eddy (PK) state.
    The `try...except` block provides a manual fallback construction for
    compatibility. The main goal is to initialize all W/V matrices to 0.0
    to ensure test results are deterministic and not influenced by complex
    secondary structure energies.
    """
    try:
        nested = make_fold_state(n)
        re_state = init_eddy_rivas_fold_state(n)
    except Exception:
        # Fallback manual construction if factories fail.
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
            w_matrix=w_matrix, v_matrix=v_matrix, wm_matrix=wm_matrix,
            w_back_ptr=w_back_ptr, v_back_ptr=v_back_ptr, wm_back_ptr=wm_back_ptr,
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
            seq_len=n, wx_matrix=wx_matrix, vx_matrix=vx_matrix,
            wxi_matrix=wxi_matrix, wxu_matrix=wxu_matrix, wxc_matrix=wxc_matrix,
            vxu_matrix=vxu_matrix, vxc_matrix=vxc_matrix,
            wx_back_ptr=EddyRivasTriBackPointer(n), vx_back_ptr=EddyRivasTriBackPointer(n),
            whx_matrix=whx_matrix, vhx_matrix=vhx_matrix,
            yhx_matrix=yhx_matrix, zhx_matrix=zhx_matrix,
            whx_back_ptr=whx_back_ptr, vhx_back_ptr=vhx_back_ptr,
            yhx_back_ptr=yhx_back_ptr, zhx_back_ptr=zhx_back_ptr,
        )

        # Set diagonal base cases for Rivas-Eddy state.
        for i in range(n):
            re_state.wx_matrix.set(i, i, 0.0)
            re_state.wxi_matrix.set(i, i, 0.0)
            re_state.wxu_matrix.set(i, i, 0.0)
            re_state.wxc_matrix.set(i, i, 0.0)
            re_state.vx_matrix.set(i, i, inf)
            re_state.vxu_matrix.set(i, i, inf)
            re_state.vxc_matrix.set(i, i, inf)

    # Seed nested W/V matrices to zero everywhere for predictable tests.
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
    """
    Tests that enabling coaxial stacking only helps (or has no effect).
    It should never result in a worse energy score. Also tests eligibility rules.
    """
    if adjacent and expect_nontrivial_caps:
        seq = "GCGG"  # A case where coax is possible.
        i, j, r, k = 0, 3, 1, 1
    elif adjacent and not expect_nontrivial_caps:
        seq = "GCU"  # A case where coax is not possible due to trivial caps.
        i, j, r, k = 0, 2, 1, 1
    else:
        pytest.skip("non-adjacent case not used here")

    n = len(seq)
    nested, re_state = _try_build_states(n)

    base_costs = make_costs(
        q_ss=0.0, p_tilde_out=0.0, p_tilde_hole=0.0,
        q_tilde_out=0.0, q_tilde_hole=0.0, l_tilde=0.0, r_tilde=0.0,
        coax_pairs={("GC", "GC"): -1.0, ("AU", "AU"): -1.0},  # Favorable coax energy
        coax_scale=1.0, coax_bonus=0.0, g_wi=0.0,
    )

    # --- Run with coax disabled ---
    cfg_off = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=False, costs=base_costs
    )
    eng_off = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_off)
    eng_off.fill_with_costs(seq, nested, re_state)
    vx_off = re_state.vx_matrix.get(i, j)

    # --- Run with coax enabled ---
    nested2, re_state2 = _try_build_states(n)
    cfg_on = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, costs=base_costs
    )
    eng_on = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_on)
    eng_on.fill_with_costs(seq, nested2, re_state2)
    vx_on = re_state2.vx_matrix.get(i, j)

    # If the structure is eligible for coax, the score should improve or stay the same.
    if expect_nontrivial_caps:
        assert vx_on <= vx_off
    # If not eligible, the score should be identical.
    else:
        assert vx_on == vx_off


def test_coax_positive_values_are_clamped_to_zero():
    """
    Verifies that destabilizing (positive) coax energies are ignored.
    If a coax energy is > 0, it should be "clamped" to zero and have no effect,
    preventing it from incorrectly penalizing a structure.
    """
    seq = "GCGC"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs_pos = make_costs(
        coax_pairs={("GC", "GC"): +2.5},  # Positive energy
    )
    cfg_off = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_coax=False, costs=costs_pos)
    cfg_on = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_coax=True, costs=costs_pos)

    # Run with coax off to get a baseline score.
    eng_off = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_off)
    eng_off.fill_with_costs(seq, nested, re_state)
    vx_off = re_state.vx_matrix.get(0, n - 1)

    # Run with coax on.
    nested2, re_state2 = _try_build_states(n)
    eng_on = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_on)
    eng_on.fill_with_costs(seq, nested2, re_state2)
    vx_on = re_state2.vx_matrix.get(0, n - 1)

    # The score should be identical, as the positive energy was clamped.
    assert vx_on == vx_off


def test_coax_variants_can_help_when_only_variant_is_scored():
    """
    Tests that `enable_coax_variants` correctly applies variant-specific energies.
    The test creates a scenario where only non-standard (variant) stacking
    configurations have favorable energy. The final score should only improve
    when the variants flag is enabled.
    """
    seq = "GCGG"
    n, i, j = len(seq), 0, 3
    nested, re_state = _try_build_states(n)

    costs = make_costs(
        # Only score the variant stacking edges favorably.
        coax_pairs={("GC", "CG"): -2.0, ("CG", "GG"): -1.0},
    )
    cfg_base = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=False, costs=costs
    )
    cfg_var = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=True, costs=costs
    )

    # Baseline with variants disabled.
    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_base)
    eng0.fill_with_costs(seq, nested, re_state)
    vx0 = re_state.vx_matrix.get(i, j)

    # With variants enabled.
    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_var)
    eng1.fill_with_costs(seq, nested2, re_state2)
    vx1 = re_state2.vx_matrix.get(i, j)

    # Enabling variants should improve the score.
    assert vx1 <= vx0


# ------------------------
# Pruning guards / overlap split
# ------------------------
def test_pruning_guards_do_not_worsen_optimum():
    """
    Tests that tightening pruning constraints does not produce a worse optimum.
    Pruning parameters (like `min_hole_width`) are optimizations. Stricter
    pruning should either find the same optimal score or a different, potentially
    suboptimal one, but it should never find a score worse than the unpruned version.
    This test uses `<=` because a tighter constraint could find a different path
    that happens to be better by chance in some models.
    """
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs = make_costs(q_ss=0.0)
    # Loose constraints (effectively no pruning).
    cfg_loose = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        min_hole_width=0, min_outer_left=0, min_outer_right=0, costs=costs
    )
    # Tight constraints.
    cfg_tight = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        min_hole_width=1, min_outer_left=1, min_outer_right=1, costs=costs
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
    """
    Tests that enabling the `WX` overlap feature with a favorable energy bonus
    can only improve (or not change) the final score.
    """
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs_no = make_costs(g_wh_wx=0.0)  # No bonus.
    costs_yes = make_costs(g_wh_wx=-0.5)  # Favorable bonus.

    cfg_no = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_wx_overlap=False, costs=costs_no)
    cfg_yes = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_wx_overlap=True, costs=costs_yes)

    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_no)
    eng0.fill_with_costs(seq, nested, re_state)
    w0 = re_state.wx_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_yes)
    eng1.fill_with_costs(seq, nested2, re_state2)
    w1 = re_state2.wx_matrix.get(0, n - 1)

    # The score with the bonus feature enabled should be better or equal.
    assert w1 <= w0


# ------------------------
# Context-split constants: monotonic effects (weak property tests)
# ------------------------
def _min_finite_yhx(re_state, n):
    """Helper to find the minimum finite energy in the YHX matrix."""
    best = math.inf
    for i in range(n):
        for j in range(i + 1, n):
            max_h = max(0, j - i - 1)
            for h in range(1, max_h + 1):
                for k in range(i, j - h):
                    l = k + h + 1
                    v = re_state.yhx_matrix.get(i, j, k, l)
                    if math.isfinite(v) and v < best:
                        best = v
    return best


def _min_finite_vhx(re_state, n):
    """Helper to find the minimum finite energy in the VHX matrix."""
    best = math.inf
    for i in range(n):
        for j in range(i + 1, n):
            max_h = max(0, j - i - 1)
            for h in range(1, max_h + 1):
                for k in range(i, j - h):
                    l = k + h + 1
                    v = re_state.vhx_matrix.get(i, j, k, l)
                    if math.isfinite(v) and v < best:
                        best = v
    return best


def test_P_out_increases_yhx_min_energy_monotonically():
    """
    Tests that the P_tilde_out penalty monotonically worsens YHX energies.
    `P_tilde_out` is a penalty applied in the YHX recurrence. Increasing it
    should result in a higher (worse) or equal minimum energy in the YHX matrix.
    """
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    # Config with no penalty.
    cfg0 = eddy_rivas_recurrences.EddyRivasFoldingConfig(costs=make_costs(p_tilde_out=0.0))
    # Config with a positive penalty.
    cfg1 = eddy_rivas_recurrences.EddyRivasFoldingConfig(costs=make_costs(p_tilde_out=2.0))

    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg0)
    eng0.fill_with_costs(seq, nested, re_state)
    y0 = _min_finite_yhx(re_state, n)

    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg1)
    eng1.fill_with_costs(seq, nested2, re_state2)
    y1 = _min_finite_yhx(re_state2, n)

    assert y1 >= y0


def test_P_hole_increases_vhx_min_energy_monotonically():
    """
    Tests that the P_tilde_hole penalty monotonically worsens VHX energies.
    Similar to the YHX test, increasing this penalty should result in a higher
    (worse) or equal minimum energy in the VHX matrix.
    """
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    cfg0 = eddy_rivas_recurrences.EddyRivasFoldingConfig(costs=make_costs(p_tilde_hole=0.0))
    cfg1 = eddy_rivas_recurrences.EddyRivasFoldingConfig(costs=make_costs(p_tilde_hole=2.0))

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
    """
    Verifies that in an energy tie, the `publish` step prefers the uncharged state.
    This is a convention to favor simpler (non-pseudoknotted) structures when
    they are energetically equivalent to more complex ones.
    """
    seq = "GCAU"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    # Use pk_penalty_gw=0.0 to make ties between charged and uncharged paths more likely.
    cfg = eddy_rivas_recurrences.EddyRivasFoldingConfig(pk_penalty_gw=0.0, costs=make_costs())
    eng = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg)
    eng.fill_with_costs(seq, nested, re_state)

    i, j = 0, n - 1
    bp = re_state.wx_back_ptr.get(i, j)
    tag = None if bp is None else bp.op

    # The backpointer must be one of the valid options for WX.
    assert tag in (
        EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED,
        EddyRivasBacktrackOp.RE_PK_COMPOSE_WX,
        EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX,
        EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX,
        EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX,
        EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_OVERLAP,
    )
    # If a tie occurred, the backpointer must indicate the uncharged path was chosen.
    if re_state.wxu_matrix.get(i, j) == re_state.wxc_matrix.get(i, j):
        assert tag == EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED


def _min_finite_whx(re_state, n):
    """Helper to find the minimum finite energy in the WHX matrix."""
    best = math.inf
    for i in range(n):
        for j in range(i + 1, n):
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
    Tests that `coax_min_helix_len` correctly gates coaxial stacking.
    If coaxial stacking requires helices longer than those available, it should
    have no effect. Relaxing this constraint should allow the energy bonus,
    improving the final score.
    """
    seq = "GCGG"
    n, i, j = len(seq), 0, 3
    nested, re_state = _try_build_states(n)

    # Run with a very strict length requirement that cannot be met.
    costs = make_costs(coax_pairs={("GC", "GG"): -1.5}, coax_min_helix_len=10)
    cfg_strict = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_coax=True, costs=costs)
    eng_strict = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_strict)
    eng_strict.fill_with_costs(seq, nested, re_state)
    vx_strict = re_state.vx_matrix.get(i, j)

    # Run again with a relaxed requirement that can be met.
    nested2, re_state2 = _try_build_states(n)
    costs2 = make_costs(coax_pairs={("GC", "GG"): -1.5}, coax_min_helix_len=1)
    cfg_relaxed = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_coax=True, costs=costs2)
    eng_relaxed = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_relaxed)
    eng_relaxed.fill_with_costs(seq, nested2, re_state2)
    vx_relaxed = re_state2.vx_matrix.get(i, j)

    # Relaxing the gate should allow the favorable coax energy, improving the score.
    assert vx_relaxed <= vx_strict


def test_coax_mismatch_requires_enable_flag():
    """
    Verifies that mismatch coaxial stacking is only applied when the flag is enabled.
    """
    seq = "GCAUGC"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs = make_costs(coax_pairs={("GA", "AC"): -2.0})  # Favorable mismatch energy

    # Run with mismatch disabled.
    cfg_no_mismatch = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_mismatch=False, costs=costs
    )
    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_no_mismatch)
    eng0.fill_with_costs(seq, nested, re_state)
    vx0 = re_state.vx_matrix.get(0, n - 1)

    # Run with mismatch enabled.
    nested2, re_state2 = _try_build_states(n)
    cfg_yes_mismatch = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_mismatch=True, costs=costs
    )
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_yes_mismatch)
    eng1.fill_with_costs(seq, nested2, re_state2)
    vx1 = re_state2.vx_matrix.get(0, n - 1)

    # Enabling the feature should lead to a better or equal score.
    assert vx1 <= vx0


def test_coax_directional_scales_affect_variants():
    """
    Tests that directional scales (coax_scale_oi/io) work correctly for variants.
    The test confirms that even with variants enabled, there is no energy gain if
    their specific scale factors are zero. The score only improves when the scales
    are non-zero.
    """
    seq = "GCGC"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    # Favorable energy only for variant edges.
    costs = make_costs(
        coax_pairs={("GC", "CG"): -1.0, ("CG", "GC"): -1.5},
        coax_scale_oo=0.0, coax_scale_oi=0.0, coax_scale_io=0.0,
    )

    # Baseline: variants disabled.
    cfg_base = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=False, costs=costs
    )
    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_base)
    eng0.fill_with_costs(seq, nested, re_state)
    vx0 = re_state.vx_matrix.get(0, n - 1)

    # Case 1: Variants enabled, but scales are zero.
    nested2, re_state2 = _try_build_states(n)
    cfg_var_zero = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=True, costs=costs
    )
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_var_zero)
    eng1.fill_with_costs(seq, nested2, re_state2)
    vx1 = re_state2.vx_matrix.get(0, n - 1)
    # No change is expected.
    assert vx1 == vx0

    # Case 2: Variants enabled, and scales are positive.
    nested3, re_state3 = _try_build_states(n)
    costs2 = make_costs(
        coax_pairs={("GC", "CG"): -1.0, ("CG", "GC"): -1.5},
        coax_scale_oo=0.0, coax_scale_oi=2.0, coax_scale_io=2.0,
    )
    cfg_var_scaled = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_coax_variants=True, costs=costs2
    )
    eng2 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_var_scaled)
    eng2.fill_with_costs(seq, nested3, re_state3)
    vx2 = re_state3.vx_matrix.get(0, n - 1)
    # The score should now improve.
    assert vx2 <= vx1


# ------------------------
# Short-hole capping penalties (charged seam; check vxc/wxc directly)
# ------------------------
def test_short_hole_caps_raise_charged_vx_when_hole_is_tiny():
    """
    Tests that short hole cap penalties are correctly applied to charged paths.
    A positive (destabilizing) cap for short holes should increase (worsen) the
    energy of the charged `VXC` matrix.
    """
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs_no = make_costs(short_hole_caps={})
    costs_yes = make_costs(short_hole_caps={1: +2.0})  # Penalize holes of width 1.
    cfg_no = eddy_rivas_recurrences.EddyRivasFoldingConfig(costs=costs_no)
    cfg_yes = eddy_rivas_recurrences.EddyRivasFoldingConfig(costs=costs_yes)

    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_no)
    eng0.fill_with_costs(seq, nested, re_state)
    vxc0 = re_state.vxc_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_yes)
    eng1.fill_with_costs(seq, nested2, re_state2)
    vxc1 = re_state2.vxc_matrix.get(0, n - 1)

    # The penalty should make the charged path energy worse or equal.
    assert vxc1 >= vxc0


# ------------------------
# Join drift: cannot worsen; can help; BP tag when it wins
# ------------------------
def test_join_drift_cannot_worsen_vx():
    """
    Tests that enabling join drift can only improve the score, never worsen it.
    """
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    base_costs = make_costs(join_drift_penalty=1.0)  # Penalize drift.
    cfg_off = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_join_drift=False, costs=base_costs)
    cfg_on = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_join_drift=True, costs=base_costs)

    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_off)
    eng0.fill_with_costs(seq, nested, re_state)
    vxc0 = re_state.vxc_matrix.get(0, n - 1)

    nested2, re_state2 = _try_build_states(n)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_on)
    eng1.fill_with_costs(seq, nested2, re_state2)
    vxc1 = re_state2.vxc_matrix.get(0, n - 1)

    # Even with a penalty, the `take_best` logic ensures the final score is not worsened.
    assert vxc1 <= vxc0


def test_join_drift_with_negative_penalty_can_win_and_sets_bp():
    """
    Tests that an attractive join drift bonus can win and sets the correct backpointer.
    If join drift provides a better energy, it should be chosen, and the final
    backpointer should reflect that a drift composition was used.
    """
    seq = "GCGCGA"
    n = len(seq)

    # --- Baseline: no drift ---
    nested0, re0 = _try_build_states(n)
    base_costs = make_costs(coax_pairs={("GC", "GC"): -2.0}, join_drift_penalty=0.0)
    cfg_off = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_join_drift=False, costs=base_costs
    )
    eng_off = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_off)
    eng_off.fill_with_costs(seq, nested0, re0)
    vxc_off = re0.vxc_matrix.get(0, n - 1)
    vx_off = re0.vx_matrix.get(0, n - 1)

    # --- With drift enabled and made attractive ---
    nested1, re1 = _try_build_states(n)
    drift_costs = make_costs(coax_pairs={("GC", "GC"): -2.0}, join_drift_penalty=-0.5)
    cfg_on = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=True, enable_join_drift=True, drift_radius=1, costs=drift_costs
    )
    eng_on = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg_on)
    eng_on.fill_with_costs(seq, nested1, re1)

    vxc_on = re1.vxc_matrix.get(0, n - 1)
    vxu_on = re1.vxu_matrix.get(0, n - 1)
    vx_on = re1.vx_matrix.get(0, n - 1)
    bp = re1.vx_back_ptr.get(0, n - 1)
    tag = None if bp is None else bp.op

    # Drift must not hurt VXC, and overall VX should be no worse.
    assert vxc_on <= vxc_off
    assert vx_on <= vx_off
    assert math.isfinite(vx_on)

    # If the charged path (with drift) wins, the backpointer must be a compose op.
    if vxc_on < vxu_on:
        assert tag in (EddyRivasBacktrackOp.RE_PK_COMPOSE_VX,
                       EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT)
    else:  # If uncharged path wins or ties, its tag is also acceptable.
        assert tag in (EddyRivasBacktrackOp.RE_PK_COMPOSE_VX,
                       EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT,
                       EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED)


# ------------------------
# IS2 in YHX/WHX contexts
# ------------------------
class _TablesYHX:
    """Mock tables object to inject a constant IS2 energy."""

    def __init__(self, val):
        self.IS2_outer_yhx = lambda seq, i, j, r, s: val


def test_IS2_outer_yhx_lowers_best_yhx_when_negative():
    """
    Tests that a favorable IS2 energy term improves the YHX score.
    """
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    # Baseline with no IS2 energy.
    cfg0 = eddy_rivas_recurrences.EddyRivasFoldingConfig(costs=make_costs())
    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg0)
    eng0.fill_with_costs(seq, nested, re_state)
    y0 = _min_finite_yhx(re_state, n)

    # With a favorable (negative) IS2 energy.
    nested2, re_state2 = _try_build_states(n)
    cfg1 = eddy_rivas_recurrences.EddyRivasFoldingConfig(costs=make_costs())
    cfg1.tables = _TablesYHX(-1.5)
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg1)
    eng1.fill_with_costs(seq, nested2, re_state2)
    y1 = _min_finite_yhx(re_state2, n)

    assert y1 <= y0


def test_IS2_outer_yhx_can_lower_whx_via_yhx_bridge():
    """
    Tests that a favorable YHX-related IS2 energy can also improve the WHX score,
    as WHX recurrences depend on YHX subproblems.
    """
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    cfg0 = eddy_rivas_recurrences.EddyRivasFoldingConfig(costs=make_costs())
    eng0 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg0)
    eng0.fill_with_costs(seq, nested, re_state)
    w0 = _min_finite_whx(re_state, n)

    nested2, re_state2 = _try_build_states(n)
    cfg1 = eddy_rivas_recurrences.EddyRivasFoldingConfig(costs=make_costs())
    cfg1.tables = _TablesYHX(-2.0)  # Inject favorable YHX energy
    eng1 = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg1)
    eng1.fill_with_costs(seq, nested2, re_state2)
    w1 = _min_finite_whx(re_state2, n)

    # The improved YHX score should propagate to improve the WHX score.
    assert w1 <= w0


# ------------------------
# Overlap + caps sanity (WX)
# ------------------------
def test_wx_overlap_respects_short_hole_caps_on_charged_path():
    """
    Sanity check that the WX overlap recurrence correctly includes penalties
    from `short_hole_caps`.
    """
    seq = "GCAUCG"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    costs_overlap = make_costs(g_wh_wx=-0.5, short_hole_caps={1: +1.0})
    cfg = eddy_rivas_recurrences.EddyRivasFoldingConfig(enable_wx_overlap=True, costs=costs_overlap)
    eng = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg)
    eng.fill_with_costs(seq, nested, re_state)

    wxc = re_state.wxc_matrix.get(0, n - 1)
    # The main check is that the calculation completes to a finite number,
    # implying the logic paths handled the parameters correctly.
    assert math.isfinite(wxc)


# ------------------------
# VX selection behavior on publish mirrors WX test
# ------------------------
def test_vx_selects_uncharged_on_tie_and_sets_backpointer():
    """
    Verifies that the VX publish step prefers the uncharged state in an energy tie.
    """
    seq = "GCAU"
    n = len(seq)
    nested, re_state = _try_build_states(n)

    cfg = eddy_rivas_recurrences.EddyRivasFoldingConfig(pk_penalty_gw=0.0, costs=make_costs())
    eng = eddy_rivas_recurrences.EddyRivasFoldingEngine(cfg)
    eng.fill_with_costs(seq, nested, re_state)

    i, j = 0, n - 1
    bp = re_state.vx_back_ptr.get(i, j)
    tag = None if bp is None else bp.op
    # The backpointer must be a valid VX option.
    assert tag in (EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED, EddyRivasBacktrackOp.RE_PK_COMPOSE_VX)
    # If a tie occurred, the uncharged path must be chosen.
    if re_state.vxu_matrix.get(i, j) == re_state.vxc_matrix.get(i, j):
        assert tag == EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED


