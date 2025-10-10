import math
from dataclasses import replace
from types import SimpleNamespace
import pytest

from rna_pk_fold.energies.energy_types import PseudoknotEnergies
from rna_pk_fold.energies import energy_pk_ops as pk


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture()
def pk_costs() -> PseudoknotEnergies:
    # Minimal but expressive defaults; maps use bigram tuple keys per your types
    return PseudoknotEnergies(
        q_ss=0.2,
        p_tilde_out=1.0,
        p_tilde_hole=1.0,
        q_tilde_out=0.2,
        q_tilde_hole=0.2,
        l_tilde=0.1,
        r_tilde=0.2,
        m_tilde_yhx=0.0,
        m_tilde_vhx=0.0,
        m_tilde_whx=0.0,
        dangle_hole_left={("A", "U"): -0.30},
        dangle_hole_right={("G", "C"): -0.40},
        dangle_outer_left={("A", "U"): -0.10},
        dangle_outer_right={("G", "C"): -0.20},
        coax_pairs=None,             # we'll monkeypatch coax_energy_for_join
        coax_bonus=-0.15,
        coax_scale_oo=1.0,
        coax_scale_oi=1.0,
        coax_scale_io=1.0,
        coax_min_helix_len=2,
        coax_scale=1.0,
        mismatch_coax_scale=0.5,
        mismatch_coax_bonus=-0.2,
        join_drift_penalty=0.0,
        short_hole_caps={1: 0.7, 3: 0.3},
        g_wh=0.0,
        g_wi=0.0,
        g_wh_wx=0.0,
        g_wh_whx=0.0,
        pk_penalty_gw=1.0,
    )


@pytest.fixture()
def cfg():
    # Use a simple namespace with the expected flags to avoid importing heavy configs
    return SimpleNamespace(
        enable_coax=True,
        enable_coax_mismatch=False,
        enable_coax_variants=False,
    )


# -------------------------
# DANGLE HELPERS
# -------------------------

def test_dangle_hole_left_and_right_with_fallbacks(pk_costs):
    seq = "AUGC"
    # hole-left looks at (k-1, k); here (0,1) = ("A","U") → -0.30
    assert math.isclose(pk.dangle_hole_left(seq, k=1, costs=pk_costs), -0.30, rel_tol=1e-12)
    # fallback when bigram not in table → L_tilde (0.1)
    assert math.isclose(pk.dangle_hole_left(seq, k=3, costs=pk_costs), 0.1, rel_tol=1e-12)

    # hole-right looks at (l, l+1); here (2,3) = ("G","C") → -0.40
    assert math.isclose(pk.dangle_hole_right(seq, l=2, costs=pk_costs), -0.40, rel_tol=1e-12)
    # fallback → R_tilde (0.2)
    assert math.isclose(pk.dangle_hole_right(seq, l=0, costs=pk_costs), 0.2, rel_tol=1e-12)


def test_dangle_outer_left_and_right_with_fallbacks(pk_costs):
    seq = "AUGC"
    # outer-left uses (i, i+1); (0,1) = ("A","U") → -0.10
    assert math.isclose(pk.dangle_outer_left(seq, i=0, costs=pk_costs), -0.10, rel_tol=1e-12)
    # fallback → L_tilde
    assert math.isclose(pk.dangle_outer_left(seq, i=2, costs=pk_costs), 0.1, rel_tol=1e-12)

    # outer-right uses (j-1, j); (2,3) = ("G","C") → -0.20
    assert math.isclose(pk.dangle_outer_right(seq, j=3, costs=pk_costs), -0.20, rel_tol=1e-12)
    # fallback → R_tilde
    assert math.isclose(pk.dangle_outer_right(seq, j=1, costs=pk_costs), 0.2, rel_tol=1e-12)


# -------------------------
# COAX PACKING
# -------------------------

def test_coax_pack_min_helix_length_gate(pk_costs, cfg, monkeypatch):
    # left helix length = r-i+1; make it 1 (< coax_min_helix_len=2)
    i, r = 0, 0
    k, l, j = 2, 2, 3
    seq = "GCCG"

    # even if everything else is enabled and adjacent, gate should zero it
    cfg.enable_coax = True
    total, bonus = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=True)
    assert math.isclose(total, 0.0)
    assert math.isclose(bonus, 0.0)


def test_coax_pack_gating_adjacent_vs_mismatch(pk_costs, cfg, monkeypatch):
    # Valid helix lengths (2 and 2)
    seq = "GCCGC"  # len = 5
    i, r = 0, 1
    k, l, j = 2, 2, 4

    # Stub the join energy for outer↔outer seam
    def fake_join(seq_, seg1, seg2, pairs_tbl):  # only the OO call matters for this test
        return -1.0  # stabilizing
    monkeypatch.setattr(pk, "coax_energy_for_join", fake_join)

    # No coax enabled → zero
    cfg.enable_coax = False
    total, bonus = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=True)
    assert math.isclose(total, 0.0)
    assert math.isclose(bonus, 0.0)

    # Coax enabled but not adjacent and not mismatch → zero
    cfg.enable_coax = True
    cfg.enable_coax_mismatch = False
    total, _ = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=False)
    assert math.isclose(total, 0.0)

    # Adjacent seam → uses OO energy; total = scale_oo * e; bonus returned
    total, bonus = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=True)
    assert math.isclose(total, -1.0 * pk_costs.coax_scale_oo, rel_tol=1e-12)
    assert math.isclose(bonus, pk_costs.coax_bonus, rel_tol=1e-12)


def test_coax_pack_mismatch_scaling_and_clamp(pk_costs, cfg, monkeypatch):
    # Make it a mismatch seam: abs(k - r) == 1
    seq = "GCCGC"
    i, r = 0, 1
    k, l, j = 2, 3, 4

    # OO join energy (pre-scaling)
    def fake_join(seq_, seg1, seg2, pairs_tbl):
        # Only the OO call is used here: (i,r) ↔ (k+1,j) == (0,1) ↔ (3,4)
        return -1.0
    monkeypatch.setattr(pk, "coax_energy_for_join", fake_join)

    cfg.enable_coax = True
    cfg.enable_coax_mismatch = True

    total, bonus = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=False)

    # mismatch scaling: e' = e * mismatch_coax_scale + mismatch_coax_bonus
    e_prime = -1.0 * pk_costs.mismatch_coax_scale + pk_costs.mismatch_coax_bonus  # -0.5 - 0.2 = -0.7
    expected_total = pk_costs.coax_scale_oo * e_prime
    assert math.isclose(total, expected_total, rel_tol=1e-12)
    assert math.isclose(bonus, pk_costs.coax_bonus, rel_tol=1e-12)

    # If the pre-scaling energy were positive, it should clamp to 0
    def fake_join_pos(seq_, seg1, seg2, pairs_tbl):
        return +1.0  # will be clamped to 0.0
    monkeypatch.setattr(pk, "coax_energy_for_join", fake_join_pos)
    total2, _ = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=False)
    assert math.isclose(total2, 0.0, rel_tol=1e-12)


def test_coax_pack_variants_oi_io_and_clamping(pk_costs, cfg, monkeypatch):
    seq = "GCCGC"  # len = 5
    i, r = 0, 1
    k, l, j = 2, 2, 4

    # Distinguish the three variant calls by the (seg1, seg2) pairs
    def fake_join(seq_, seg1, seg2, pairs_tbl):
        # OO: (i,r) ↔ (k+1,j) == (0,1) ↔ (3,3)
        if seg1 == (i, r) and seg2 == (k + 1, j):
            return -1.00
        # OI: (i,r) ↔ (k,l) == (0,1) ↔ (2,2)
        if seg1 == (i, r) and seg2 == (k, l):
            return +0.30  # positive → should clamp to 0
        # IO: (k,l) ↔ (k+1,j) == (2,2) ↔ (3,3)
        if seg1 == (k, l) and seg2 == (k + 1, j):
            return -0.25
        return 0.0

    monkeypatch.setattr(pk, "coax_energy_for_join", fake_join)

    cfg.enable_coax = True
    cfg.enable_coax_variants = True
    cfg.enable_coax_mismatch = False

    # Adjust scales to make the arithmetic visible
    costs = replace(pk_costs, coax_scale_oo=1.0, coax_scale_oi=1.5, coax_scale_io=2.0)
    total, bonus = pk.coax_pack(seq, i, j, r, k, l, cfg, costs, adjacent=True)
    expected_total = (-1.00 * 1.0) + (0.0 * 1.5) + (-0.25 * 2.0)  # = -1.5
    assert math.isclose(total, expected_total, rel_tol=1e-12)

    total, bonus = pk.coax_pack(seq, i, j, r, k, l, cfg, costs, adjacent=True)
    # OI was positive so it should clamp to 0; expect OO + IO only
    expected_total = (-1.00 * 1.0) + (0.0 * 1.5) + (-0.25 * 2.0)  # = -1.5
    assert math.isclose(total, expected_total, rel_tol=1e-12)
    assert math.isclose(bonus, costs.coax_bonus, rel_tol=1e-12)


# -------------------------
# SHORT HOLE PENALTY
# -------------------------

def test_short_hole_penalty_caps_and_default(pk_costs):
    # width h = l - k - 1
    assert math.isclose(pk.short_hole_penalty(pk_costs, k=5, l=7), 0.7, rel_tol=1e-12)  # h=1
    assert math.isclose(pk.short_hole_penalty(pk_costs, k=5, l=9), 0.3, rel_tol=1e-12)  # h=3
    assert math.isclose(pk.short_hole_penalty(pk_costs, k=5, l=6), 0.0, rel_tol=1e-12)  # h=0 not in caps

    # When caps absent, default to 0.0
    costs2 = replace(pk_costs, short_hole_caps=None)
    assert math.isclose(pk.short_hole_penalty(costs2, k=5, l=7), 0.0, rel_tol=1e-12)
