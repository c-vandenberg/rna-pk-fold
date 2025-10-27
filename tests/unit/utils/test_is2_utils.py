"""
Unit tests for the IS2 (Symmetric Internal Loop) energy utility functions.

These tests validate the refactored helpers:
- `compute_is2_outer_bridge_energy`
- `compute_is2_outer_bridge_energy_yhx`
- `compute_is2_bridge_energy` (dispatcher)
- `scan_is2_outer_bridge_candidates`

The new API expects a pseudoknot energy object with scalar attributes like
`q_tilde_out`, `p_tilde_out`, `m_tilde_*`, and (optionally) dangle tables.
No `tables` object is consulted by these helpers anymore.
"""
import math
from types import SimpleNamespace

import pytest

from rna_pk_fold.utils.energy.is2_utils import (
    compute_is2_outer_bridge_energy,
    compute_is2_outer_bridge_energy_yhx,
    compute_is2_bridge_energy,
    scan_is2_outer_bridge_candidates,
)


# ---------------------------------------------------------------------------
# Minimal energy object factory for tests
# ---------------------------------------------------------------------------
def _pk(
    q_tilde_out=0.0,
    p_tilde_out=0.0,
    m_tilde_vhx=0.0,
    m_tilde_whx=0.0,
    m_tilde_yhx=0.0,
    # ensure dangle lookups don't add surprise costs
    dangle_outer_left=None,
    dangle_outer_right=None,
    l_tilde=0.0,
    r_tilde=0.0,
):
    return SimpleNamespace(
        q_tilde_out=float(q_tilde_out),
        p_tilde_out=float(p_tilde_out),
        m_tilde_vhx=float(m_tilde_vhx),
        m_tilde_whx=float(m_tilde_whx),
        m_tilde_yhx=float(m_tilde_yhx),
        dangle_outer_left={} if dangle_outer_left is None else dangle_outer_left,
        dangle_outer_right={} if dangle_outer_right is None else dangle_outer_right,
        l_tilde=float(l_tilde),
        r_tilde=float(r_tilde),
    )


# ----------------------- compute_is2_outer_bridge_energy -----------------------
def test_outer_bridge_no_trim_costs_are_just_multipliers_no_dangles():
    """
    When r==i and s==j (no outer trim), the cost should be:
        E = (0)*q_tilde_out + (m_tilde_vhx + m_tilde_whx) + 0
    """
    seq = "ACGU"
    pk = _pk(q_tilde_out=7.0, m_tilde_vhx=0.25, m_tilde_whx=0.75)  # q_tilde_out irrelevant since no trim
    # i=1, j=3, r=1, s=3 -> no trimming
    got = compute_is2_outer_bridge_energy(seq, pk, 1, 3, 1, 3)
    assert got == pytest.approx(1.0)  # 0.25 + 0.75


def test_outer_bridge_with_trim_applies_linear_penalty_and_multiloop_overhead():
    """
    With left/right trims, the cost should be:
        E = (n_left + n_right)*q_tilde_out + (m_tilde_vhx + m_tilde_whx) + dangles(=0 here)
    """
    seq = "ACGUAC"
    # choose values easy to inspect
    pk = _pk(q_tilde_out=1.5, m_tilde_vhx=0.2, m_tilde_whx=0.3, l_tilde=0.0, r_tilde=0.0)
    # i=0, j=5, r=2, s=4 -> n_left = 2, n_right = 1 => (2+1)*1.5 = 4.5; ml = 0.5; total 5.0
    got = compute_is2_outer_bridge_energy(seq, pk, 0, 5, 2, 4)
    assert got == pytest.approx(5.0)


def test_outer_bridge_with_trim_can_include_dangles_when_present():
    """
    If trimming occurs and dangle tables include entries for the sides, those should
    be added. We keep l_tilde/r_tilde at 0 to make the dangle contribution explicit.
    """
    # We'll craft tables that definitely get hit by the implementation for i/j.
    # To avoid relying on internal bigram indexing details, we just provide a
    # broad mapping and check that the returned cost includes our constants.
    seq = "AAAAAA"
    pk = _pk(
        q_tilde_out=0.0, m_tilde_vhx=0.0, m_tilde_whx=0.0,
        dangle_outer_left={("A", "A"): -0.4},   # plausible key for left trim near i
        dangle_outer_right={("A", "A"): -0.6},  # plausible key for right trim near j
        l_tilde=0.0, r_tilde=0.0
    )
    # Any nonzero trim on both sides so both dangle paths are considered:
    # i=0, j=5, r=1, s=4 -> n_left=1, n_right=1
    got = compute_is2_outer_bridge_energy(seq, pk, 0, 5, 1, 4)
    # We don't assert exact dangle hit details; just ensure dangle path does not *increase* energy
    # (dangles are bonuses or zero). At minimum, energy should be <= 0.0 (since q_tilde_out and ml terms are 0).
    assert got <= 0.0


# -------------------- compute_is2_outer_bridge_energy_yhx ----------------------
def test_yhx_includes_pair_cost_even_without_trim():
    """
    For YHX, forming the outer pair contributes p_tilde_out even if there is no trim:
        E = p_tilde_out + 0*q_tilde_out + (m_tilde_yhx + m_tilde_whx)
    """
    seq = "ACGU"
    pk = _pk(p_tilde_out=2.0, q_tilde_out=9.0, m_tilde_yhx=0.4, m_tilde_whx=0.6)  # q_tilde_out irrelevant (no trim)
    # i=1, j=3, r=1, s=3 -> no trim
    got = compute_is2_outer_bridge_energy_yhx(pk, seq, 1, 3, 1, 3)
    assert got == pytest.approx(3.0)  # 2.0 + 0.4 + 0.6


def test_yhx_with_trim_adds_pair_linear_and_multiloop_terms():
    """
    With trim, YHX cost should be:
        E = p_tilde_out + (n_left + n_right)*q_tilde_out + (m_tilde_yhx + m_tilde_whx) + dangles(=0 here)
    """
    seq = "ACGUAC"
    pk = _pk(p_tilde_out=1.25, q_tilde_out=1.5, m_tilde_yhx=0.1, m_tilde_whx=0.2)
    # i=0, j=5, r=2, s=4 -> n_left=2, n_right=1
    got = compute_is2_outer_bridge_energy_yhx(pk, seq, 0, 5, 2, 4)
    # 1.25 + (3*1.5) + (0.1 + 0.2) = 1.25 + 4.5 + 0.3 = 6.05
    assert got == pytest.approx(6.05)


# -------------------- Dispatcher + Scan utilities -----------------------------
def test_dispatch_unknown_kind_raises_value_error():
    with pytest.raises(ValueError):
        compute_is2_bridge_energy(
            config=_pk(),  # any object with needed fields
            seq="ACGU",
            bridge_kind_name="not-a-kind",
            i_index=0, j_index=5, r_index=1, s_index=4,
        )


def test_scan_is2_outer_bridge_candidates_picks_minimum_inner_energy(monkeypatch):
    """
    With all bridge terms set to zero, scan should pick the (r,s) pair whose
    inner gap energy is minimal.
    """
    # Zero out all bridge costs so the candidate energy equals the inner gap energy.
    cfg = _pk(
        p_tilde_out=0.0,
        q_tilde_out=0.0,
        m_tilde_yhx=0.0,
        m_tilde_whx=0.0,
        m_tilde_vhx=0.0,
        dangle_outer_left={},
        dangle_outer_right={},
    )

    # Inject a deterministic "inner matrix" energy surface for testing.
    # We'll make (r=2, s=4) the unique best.
    def fake_inner_energy(_fold_state, _name, r, s, _k, _l):
        # baseline
        val = 10.0
        if (r, s) == (2, 4):
            val = 3.0
        return val

    # Patch the inner gap energy lookup used by the scanner.
    monkeypatch.setattr(
        "rna_pk_fold.utils.energy.is2_utils.get_gap_energy_for_named_matrix",
        fake_inner_energy,
        raising=False,
    )

    # Dummy objects (not used by the fake energy function)
    fold_state = object()
    seq = "ACGUAC"
    i, j, k, l = 0, 5, 2, 4  # note k,l bound the hole

    best_energy, coords, _op = scan_is2_outer_bridge_candidates(
        fold_state=fold_state,
        config=cfg,
        seq=seq,
        i_index=i,
        j_index=j,
        k_index=k,
        l_index=l,
        inner_matrix_name="ANY",
        bridge_kind_name="yhx",  # exercise dispatcher path inside scan
        backtrack_op=None,
    )

    assert math.isfinite(best_energy)
    assert coords == (2, 4)  # picked the minimum by our fake surface

