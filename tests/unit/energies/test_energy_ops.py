import math
import pytest

from rna_pk_fold.energies import SecondaryStructureEnergyLoader
from rna_pk_fold.energies.energy_ops import (
    hairpin_energy,
    stack_energy,
    internal_loop_energy,
    multiloop_linear_energy,
)
from rna_pk_fold.utils.energy_utils import calculate_delta_g


@pytest.fixture(scope="module")
def rna_energy_bundle():
    """
    Return the current RNA energy bundle used by the adapters.

    Expected
    --------
    - A `SecondaryStructureEnergies` instance with populated tables.
    """
    return SecondaryStructureEnergyLoader.load("RNA")


def test_hairpin_energy_size_3_matches_table(rna_energy_bundle):
    """
    For a 3-nt hairpin (j - i - 1 = 3), ΔG should be ΔH - T*(ΔS/1000) from
    rna_energy_bundle.HAIRPIN[3].

    Expected
    --------
    - `hairpin_energy(0, 4, "AAAAA", E, 310.15)` ≈
      `calculate_delta_g(rna_energy_bundle.HAIRPIN[3], 310.15)`
      within tight tolerance.
    """
    seq = "AAAAA"  # length 5; i=0, j=4 => loop length = 3
    base_i, base_j = 0, 4
    temp = 310.15

    expected = calculate_delta_g(rna_energy_bundle.HAIRPIN[3], temp)
    got = hairpin_energy(base_i, base_j, seq, rna_energy_bundle, temp_k=temp)

    assert math.isclose(got, expected, rel_tol=1e-12)


def test_hairpin_energy_below_min_size_returns_inf(rna_energy_bundle):
    """
    Loops smaller than the minimum hairpin size must be disallowed.

    Expected
    --------
    - If `j - i - 1 < MIN_HAIRPIN_UNPAIRED`, `hairpin_energy(...)` returns `+∞`.
    """
    seq = "AAAA"  # i=0, j=2 => loop length = 1
    base_i, base_j = 0, 2

    assert math.isinf(hairpin_energy(base_i, base_j, seq, rna_energy_bundle))


def test_stack_energy_au_ua_matches_nn(rna_energy_bundle):
    """
    Build a context that yields NN key 'AU/UA' and verify ΔG matches the table at 37 °C.

    Expected
    --------
    - For seq="AUAU", (i,j,k,l)=(0,3,1,2), ΔG ≈ ΔG(NN["AU/UA"]) at 310.15 K
      within a small absolute tolerance.
    """
    # seq[0:4] = A U A U
    seq = "AUAU"
    base_i, base_j = 0, 3  # dimer_key -> left: seq[i]seq[i+1] = AU; right: seq[j]seq[j-1] = UA
    base_k, base_l = 1, 2  # typical stack indices (k=i+1, l=j-1)
    temp = 310.15

    expected = calculate_delta_g(rna_energy_bundle.NN["AU/UA"], temp)
    got = stack_energy(base_i, base_j, base_k, base_l, seq, rna_energy_bundle, temp_k=temp)

    assert math.isclose(got, expected, abs_tol=5e-3)


def test_stack_energy_invalid_geometry_returns_inf(rna_energy_bundle):
    """
    Non-stacked geometry (or indices out of range) should return +∞.

    Expected
    --------
    - If indices violate `0 <= i < k <= l < j < len(seq)`, the result is `+∞`.
    """
    seq = "AUAU"
    # Violates 0 <= i < k <= l < j < len(seq)
    assert math.isinf(stack_energy(0, 2, 1, 1, seq, rna_energy_bundle))


def test_internal_loop_energy_bulge_size_1_uses_bulge_anchor(rna_energy_bundle):
    """
    Bulge case: a=1, b=0 should use BULGE[1] baseline under the clamp policy.

    Expected
    --------
    - For a bulge of size 1, ΔG equals `calculate_delta_g(rna_energy_bundle.BULGE[1], temp)`.
    """
    # Choose indices so:
    #   a = k - i - 1 = 1
    #   b = j - l - 1 = 0
    # Use seq length 5 for safety: indices 0..4
    seq = "AUAUA"
    base_i, base_k, base_l, base_j = 0, 2, 3, 4
    temp = 310.15

    expected = calculate_delta_g(rna_energy_bundle.BULGE[1], temp)
    got = internal_loop_energy(base_i, base_j, base_k, base_l, seq, rna_energy_bundle, temp_k=temp)

    assert math.isclose(got, expected, rel_tol=1e-12)


def test_internal_loop_energy_internal_size_4_uses_internal_anchor(rna_energy_bundle):
    """
    Internal loop case: a=2, b=2 => size=4 should use INTERNAL[4].

    Expected
    --------
    - For size 4 internal loop, ΔG equals `calculate_delta_g(E.INTERNAL[4], temp)`.
    """
    # Need 0 < i < k <= l < j and size a+b = 4
    # Let i=0, k=3 => a=2; l=4, j=7 => b=2
    seq = "A" * 8
    base_i, base_k, base_l, base_j = 0, 3, 4, 7
    temp = 310.15

    expected = calculate_delta_g(rna_energy_bundle.INTERNAL[4], temp)
    got = internal_loop_energy(base_i, base_j, base_k, base_l, seq, rna_energy_bundle, temp_k=temp)

    assert math.isclose(got, expected, rel_tol=1e-12)


def test_internal_loop_energy_1x1_mismatch_falls_back_or_inf(rna_energy_bundle):
    """
    For a 1x1 internal loop (a=b=1), the adapter first tries INTERNAL_MM;
    if no motif key is present in the reduced table, it falls back to INTERNAL size 2.
    In the current bundle, INTERNAL has no anchor ≤ 2, so the result becomes +∞.

    Expected
    --------
    - With no available INTERNAL_MM motif and no INTERNAL anchor ≤ 2,
      the function returns `+∞`.
    """
    # Construct 1×1: a=1 (k=i+1), b=1 (j=l+1).
    seq = "AUGAUA"
    # i=0, k=2 => a=1; l=3, j=5 => b=1
    base_i, base_k, base_l, base_j = 0, 2, 3, 5
    temp = 310.15

    got = internal_loop_energy(base_i, base_j, base_k, base_l, seq, rna_energy_bundle, temp_k=temp)
    assert math.isinf(got)  # current reduced table yields no INTERNAL anchor ≤ 2


def test_multiloop_linear_energy_formula(rna_energy_bundle):
    """
    ΔG = a + b * branches + c * unpaired (+ d if unpaired == 0).
    With branches=2, unpaired=3 and MULTILOOP=(2.5, 0.1, 0.4, 2.0):
    ΔG = 2.5 + 0.2 + 1.2 = 3.9

    Expected
    --------
    - `multiloop_linear_energy(2, 3, E)` == 3.9 for the current MULTILOOP tuple.
    """
    multiloop_delta_g = multiloop_linear_energy(branches=2, unpaired_bases=3, energies=rna_energy_bundle)
    assert math.isclose(multiloop_delta_g, 3.9, rel_tol=1e-12)


def test_multiloop_linear_energy_zero_unpaired_adds_bonus(rna_energy_bundle):
    """
    When unpaired == 0, the 'd' bonus applies once.
    With MULTILOOP=(2.5, 0.1, 0.4, 2.0) and branches=1, unpaired=0:
    ΔG = 2.5 + 0.1*1 + 0.4*0 + 2.0 = 4.6

    Expected
    --------
    - `multiloop_linear_energy(1, 0, E)` == 4.6 for the current MULTILOOP tuple.
    """
    multiloop_delta_g = multiloop_linear_energy(branches=1, unpaired_bases=0, energies=rna_energy_bundle)
    assert math.isclose(multiloop_delta_g, 4.6, rel_tol=1e-12)
