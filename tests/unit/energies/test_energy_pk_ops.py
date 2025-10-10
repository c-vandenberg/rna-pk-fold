"""
Unit tests for pseudoknot energy calculation helpers in `energy_pk_ops`.

This module tests the various components of pseudoknot energy calculations,
including dangle-end penalties, coaxial stacking energies under different
configurations, and penalties for short interior loops (holes).

Fixtures are used to provide:
1.  `pk_costs`: A `PseudoknotEnergies` dataclass with minimal, non-zero values
    to make the tests expressive and sensitive to which parameters are used.
2.  `cfg`: A lightweight mock of a configuration object using `SimpleNamespace`
    to simulate feature flags that control energy calculations (e.g., enabling
    coaxial stacking).
"""

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
    """
    Provides a default `PseudoknotEnergies` instance for testing.

    The values are chosen to be minimal but non-zero and distinct, making it
    easy to trace which energy terms are being applied in the tests and to
    verify correct fallback behavior.
    """
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
        coax_pairs=None,  # we'll monkeypatch coax_energy_for_join
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
    """
    Provides a mock configuration object using `SimpleNamespace`.

    This avoids importing potentially heavy configuration modules and allows tests
    to easily toggle feature flags that control which energy calculations are enabled,
    such as `enable_coax` or `enable_coax_mismatch`.
    """
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
    """
    Tests dangle energies for unpaired bases adjacent to the inner helix (k, l).

    It verifies two scenarios for both left and right dangles:
    1.  The specific dinucleotide is found in the energy table (`dangle_hole_*`).
    2.  The dinucleotide is not found, triggering a fallback to a default penalty
        (`L_tilde` for left, `R_tilde` for right).
    """
    seq = "AUGC"
    # hole-left looks at (k-1, k); here (0,1) = ("A","U") → -0.30 (found in table)
    assert math.isclose(pk.dangle_hole_left(seq, k=1, costs=pk_costs), -0.30, rel_tol=1e-12)
    # k=3 means we look at (2,3) = ("G", "C"), not in the `dangle_hole_left` table.
    # Should fall back to `L_tilde` (0.1).
    assert math.isclose(pk.dangle_hole_left(seq, k=3, costs=pk_costs), 0.1, rel_tol=1e-12)

    # hole-right looks at (l, l+1); here (2,3) = ("G","C") → -0.40 (found in table)
    assert math.isclose(pk.dangle_hole_right(seq, l=2, costs=pk_costs), -0.40, rel_tol=1e-12)
    # l=0 means we look at (0,1) = ("A", "U"), not in the `dangle_hole_right` table.
    # Should fall back to `R_tilde` (0.2).
    assert math.isclose(pk.dangle_hole_right(seq, l=0, costs=pk_costs), 0.2, rel_tol=1e-12)


def test_dangle_outer_left_and_right_with_fallbacks(pk_costs):
    """
    Tests dangle energies for unpaired bases adjacent to the outer helix (i, j).

    This is analogous to the `dangle_hole` test but for the outer pairing context.
    It verifies successful lookups and fallbacks to `L_tilde`/`R_tilde`.
    """
    seq = "AUGC"
    # outer-left uses (i, i+1); (0,1) = ("A","U") → -0.10 (found in table)
    assert math.isclose(pk.dangle_outer_left(seq, i=0, costs=pk_costs), -0.10, rel_tol=1e-12)
    # i=2 means we look at (2,3) = ("G", "C"), not in the `dangle_outer_left` table.
    # Fallback to `L_tilde`.
    assert math.isclose(pk.dangle_outer_left(seq, i=2, costs=pk_costs), 0.1, rel_tol=1e-12)

    # outer-right uses (j-1, j); (2,3) = ("G","C") → -0.20 (found in table)
    assert math.isclose(pk.dangle_outer_right(seq, j=3, costs=pk_costs), -0.20, rel_tol=1e-12)
    # j=1 means we look at (0,1) = ("A", "U"), not in the `dangle_outer_right` table.
    # Fallback to `R_tilde`.
    assert math.isclose(pk.dangle_outer_right(seq, j=1, costs=pk_costs), 0.2, rel_tol=1e-12)


# -------------------------
# COAX PACKING
# -------------------------

def test_coax_pack_min_helix_length_gate(pk_costs, cfg, monkeypatch):
    """
    Tests that coax energy is zeroed if either helix is too short.

    The `coax_pack` function should immediately return zero if the length of the
    left helix (r-i+1) or the right helix (j-l+1) is less than the configured
    `coax_min_helix_len`, regardless of other settings.
    """
    # left helix length = r-i+1; set to 1, which is less than `coax_min_helix_len` (2).
    i, r = 0, 0
    # right helix indices
    k, l, j = 2, 2, 3
    seq = "GCCG"

    # Even if coax is enabled and the helices are adjacent, the length gate should
    # prevent any energy from being calculated.
    cfg.enable_coax = True
    total, bonus = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=True)
    assert math.isclose(total, 0.0)
    assert math.isclose(bonus, 0.0)


def test_coax_pack_gating_adjacent_vs_mismatch(pk_costs, cfg, monkeypatch):
    """
    Tests the main control flow gates for `coax_pack`.

    Verifies that:
    1.  If `enable_coax` is False, the result is always zero.
    2.  If `enable_coax` is True but the seam is not adjacent and `enable_coax_mismatch`
        is False, the result is zero.
    3.  If `enable_coax` is True and the seam is adjacent, the correct energy
        and bonus are calculated.
    """
    # Setup with valid helix lengths (2 and 2)
    seq = "GCCGC"  # len = 5
    i, r = 0, 1
    k, l, j = 2, 2, 4

    # Stub the underlying join energy function to return a constant, stable value.
    # For an adjacent stack, this will be called for the outer-outer (OO) seam.
    def fake_join(seq_, seg1, seg2, pairs_tbl):
        return -1.0  # stabilizing

    monkeypatch.setattr(pk, "coax_energy_for_join", fake_join)

    # Case 1: No coax enabled → should be zero.
    cfg.enable_coax = False
    total, bonus = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=True)
    assert math.isclose(total, 0.0)
    assert math.isclose(bonus, 0.0)

    # Case 2: Coax enabled, but seam is not adjacent and mismatch is disabled → zero.
    cfg.enable_coax = True
    cfg.enable_coax_mismatch = False
    total, _ = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=False)
    assert math.isclose(total, 0.0)

    # Case 3: Adjacent seam with coax enabled → uses OO energy.
    # Total energy should be the join energy times the OO scale factor.
    # Bonus should be the configured `coax_bonus`.
    total, bonus = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=True)
    assert math.isclose(total, -1.0 * pk_costs.coax_scale_oo, rel_tol=1e-12)
    assert math.isclose(bonus, pk_costs.coax_bonus, rel_tol=1e-12)


def test_coax_pack_mismatch_scaling_and_clamp(pk_costs, cfg, monkeypatch):
    """
    Tests the specific energy calculation for a mismatched coaxial stack.

    This test validates a specific "mismatch" case where the helices are
    consecutive (0-nucleotide gap), but are processed via the mismatch logic
    because `adjacent=False` is passed.

    Verifies that:
    1.  The base energy is correctly scaled by `mismatch_coax_scale` and
        the `mismatch_coax_bonus` is added.
    2.  The final result is scaled by the `coax_scale_oo` factor.
    3.  If the base join energy is positive (destabilizing), it is clamped to
        zero before any mismatch scaling is applied.
    """
    # Define a seam where helices are consecutive (k-r-1 = 0), but we will
    # treat it as a mismatch by passing `adjacent=False`.
    seq = "GCCGC"
    i, r = 0, 1
    k, l, j = 2, 3, 4 # Correct indices: k=2, r=1 -> 0-nt gap

    # Mock the join energy function to return a stable value.
    def fake_join(seq_, seg1, seg2, pairs_tbl):
        # Mismatch stack calls join on (i,r) ↔ (k+1,j) == (0,1) ↔ (3,4)
        return -1.0
    monkeypatch.setattr(pk, "coax_energy_for_join", fake_join)

    cfg.enable_coax = True
    cfg.enable_coax_mismatch = True

    # Call the function with `adjacent=False` to force the mismatch path.
    total, bonus = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=False)

    # Expected mismatch energy calculation: e' = e * mismatch_scale + mismatch_bonus
    e_prime = -1.0 * pk_costs.mismatch_coax_scale + pk_costs.mismatch_coax_bonus  # -0.5 - 0.2 = -0.7
    expected_total = pk_costs.coax_scale_oo * e_prime # Final energy is scaled by OO factor
    assert math.isclose(total, expected_total, rel_tol=1e-12)

    # The bonus should be the standard coax bonus, unaffected by mismatch scaling.
    assert math.isclose(bonus, pk_costs.coax_bonus, rel_tol=1e-12)

    # Test clamping: if the base join energy is positive, the final energy should be zero.
    def fake_join_pos(seq_, seg1, seg2, pairs_tbl):
        return +1.0  # this destabilizing energy should be clamped to 0.0
    monkeypatch.setattr(pk, "coax_energy_for_join", fake_join_pos)
    total2, _ = pk.coax_pack(seq, i, j, r, k, l, cfg, pk_costs, adjacent=False)
    assert math.isclose(total2, 0.0, rel_tol=1e-12)


def test_coax_pack_variants_oi_io_and_clamping(pk_costs, cfg, monkeypatch):
    """
    Tests energy calculation when coaxial stacking variants are enabled.

    When `enable_coax_variants` is True, `coax_pack` should calculate energies
    for Outer-Outer (OO), Outer-Inner (OI), and Inner-Outer (IO) stacking
    configurations. This test verifies that:
    1.  All three variant energies are calculated and scaled by their respective
        scale factors (`coax_scale_oo`, `coax_scale_oi`, `coax_scale_io`).
    2.  The total energy is the sum of these three components.
    3.  Positive (destabilizing) energies for any variant are clamped to zero
        before being added to the total.
    """
    seq = "GCCGC"
    i, r = 0, 1
    k, l, j = 2, 2, 4

    # Mock the join function to return different values for each variant type,
    # allowing us to check if the correct logic is applied to each.
    def fake_join(seq_, seg1, seg2, pairs_tbl):
        # OO: outer helix (i,r) stacks on outer part of inner helix (k+1,j)
        if seg1 == (i, r) and seg2 == (k + 1, j):  # (0,1) ↔ (3,4)
            return -1.00
        # OI: outer helix (i,r) stacks on inner part of inner helix (k,l)
        if seg1 == (i, r) and seg2 == (k, l):  # (0,1) ↔ (2,2)
            return +0.30  # positive value, should be clamped to 0
        # IO: inner part of inner helix (k,l) stacks on outer part (k+1,j)
        if seg1 == (k, l) and seg2 == (k + 1, j):  # (2,2) ↔ (3,4)
            return -0.25
        return 0.0  # Default return

    monkeypatch.setattr(pk, "coax_energy_for_join", fake_join)

    # Enable coax and variants
    cfg.enable_coax = True
    cfg.enable_coax_variants = True
    cfg.enable_coax_mismatch = False  # Not testing mismatch here

    # Use custom scales to make the arithmetic easy to verify
    costs = replace(pk_costs, coax_scale_oo=1.0, coax_scale_oi=1.5, coax_scale_io=2.0)
    total, bonus = pk.coax_pack(seq, i, j, r, k, l, cfg, costs, adjacent=True)

    # Calculate expected total:
    # OO energy: -1.00 * 1.0 = -1.00
    # OI energy: +0.30 is clamped to 0, so 0.0 * 1.5 = 0.0
    # IO energy: -0.25 * 2.0 = -0.50
    # Total = -1.00 + 0.0 + -0.50 = -1.5
    expected_total = (-1.00 * 1.0) + (0.0 * 1.5) + (-0.25 * 2.0)
    assert math.isclose(total, expected_total, rel_tol=1e-12)
    assert math.isclose(bonus, costs.coax_bonus, rel_tol=1e-12)


# -------------------------
# SHORT HOLE PENALTY
# -------------------------

def test_short_hole_penalty_caps_and_default(pk_costs):
    """
    Tests the penalty for short "holes" (unpaired regions between helices).

    Verifies that:
    1.  The correct penalty is retrieved from the `short_hole_caps` dictionary
        based on the calculated hole width (`h = l - k - 1`).
    2.  If the hole width is not in the dictionary, the penalty is 0.
    3.  If the `short_hole_caps` table is `None`, the penalty is always 0.
    """
    # Hole width `h` is calculated as `l - k - 1`.

    # h = 7 - 5 - 1 = 1. Penalty for h=1 is 0.7 in the fixture.
    assert math.isclose(pk.short_hole_penalty(pk_costs, k=5, l=7), 0.7, rel_tol=1e-12)
    # h = 9 - 5 - 1 = 3. Penalty for h=3 is 0.3.
    assert math.isclose(pk.short_hole_penalty(pk_costs, k=5, l=9), 0.3, rel_tol=1e-12)
    # h = 6 - 5 - 1 = 0. h=0 is not in the caps dictionary, should default to 0.0.
    assert math.isclose(pk.short_hole_penalty(pk_costs, k=5, l=6), 0.0, rel_tol=1e-12)

    # When the `short_hole_caps` attribute is None, the function should always return 0.
    costs2 = replace(pk_costs, short_hole_caps=None)
    assert math.isclose(pk.short_hole_penalty(costs2, k=5, l=7), 0.0, rel_tol=1e-12)
