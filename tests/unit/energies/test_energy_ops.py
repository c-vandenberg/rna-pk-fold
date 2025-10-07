import math
import pytest
from importlib.resources import files as ir_files
from types import SimpleNamespace

import rna_pk_fold
from rna_pk_fold.energies.energy_loader import SecondaryStructureEnergyLoader
from rna_pk_fold.energies.energy_ops import (
    hairpin_energy,
    stack_energy,
    internal_loop_energy,
    multiloop_linear_energy,
    dimer_key
)
from rna_pk_fold.utils.energy_utils import calculate_delta_g


@pytest.fixture(scope="module")
def yaml_path():
    # Use the same minimal YAML you validated earlier
    return ir_files(rna_pk_fold) / "data" / "xia1998_zucker_turner1999_min.yaml"


@pytest.fixture(scope="module")
def rna_energy_bundle(yaml_path):
    """
    Return the current RNA energy bundle used by the adapters.
    """
    return SecondaryStructureEnergyLoader().load("RNA", yaml_path=yaml_path)


def test_hairpin_energy_size_3_matches_table(rna_energy_bundle):
    """
    For a 3-nt hairpin (j - i - 1 = 3), ΔG should be ΔH - T*(ΔS/1000) from
    rna_energy_bundle.HAIRPIN[3].
    """
    seq = "AAAAA"  # i=0, j=4 => loop length = 3
    i, j = 0, 4
    T = 310.15

    expected = calculate_delta_g(rna_energy_bundle.HAIRPIN[3], T)
    got = hairpin_energy(i, j, seq, rna_energy_bundle, temp_k=T)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_hairpin_energy_below_min_size_returns_inf(rna_energy_bundle):
    seq = "AAAA"  # i=0, j=2 => loop length = 1
    assert math.isinf(hairpin_energy(0, 2, seq, rna_energy_bundle))


def test_stack_energy_cu_gg_matches_nn(rna_energy_bundle):
    # Use a 4-mer that forms two valid closing pairs: (0,3)=C-G and (1,2)=G-C
    seq = "CUGG"
    i, j, k, l = 0, 3, 1, 2
    T = 310.15

    # Build the exact Turner key your code uses
    key = dimer_key(seq, i, j)
    assert key in rna_energy_bundle.NN_STACK, f"Expected NN stack key {key!r} in table"

    energies_shim = SimpleNamespace(
        NN_STACK=rna_energy_bundle.NN_STACK,
        NN=rna_energy_bundle.NN_STACK,
        delta_g=rna_energy_bundle.delta_g,
    )

    expected = calculate_delta_g(rna_energy_bundle.NN_STACK[key], T)
    got = stack_energy(i, j, k, l, seq, energies_shim, temp_k=T)
    assert math.isclose(got, expected, abs_tol=5e-3)


def test_stack_energy_invalid_geometry_returns_inf(rna_energy_bundle):
    seq = "AUAU"
    assert math.isinf(stack_energy(0, 2, 1, 1, seq, rna_energy_bundle))


def test_internal_loop_energy_bulge_size_1_uses_bulge_anchor(rna_energy_bundle):
    """
    Bulge case: a=1, b=0 should use BULGE[1] baseline.
    """
    seq = "AUAUA"
    i, k, l, j = 0, 2, 3, 4
    T = 310.15

    expected = calculate_delta_g(rna_energy_bundle.BULGE[1], T)
    got = internal_loop_energy(i, j, k, l, seq, rna_energy_bundle, temp_k=T)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_internal_loop_energy_internal_size_4_uses_internal_anchor(rna_energy_bundle):
    """
    Internal loop: a=2, b=2 => size=4 should use INTERNAL[4].
    """
    seq = "A" * 8
    i, k, l, j = 0, 3, 4, 7
    T = 310.15

    expected = calculate_delta_g(rna_energy_bundle.INTERNAL[4], T)
    got = internal_loop_energy(i, j, k, l, seq, rna_energy_bundle, temp_k=T)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_internal_loop_energy_1x1_mismatch_falls_back_or_inf(rna_energy_bundle):
    """
    1x1 internal loop (a=b=1) → try INTERNAL_MISMATCH else fallback; with
    the minimal table, expect +∞.
    """
    seq = "AUGAUA"
    i, k, l, j = 0, 2, 3, 5
    T = 310.15

    got = internal_loop_energy(i, j, k, l, seq, rna_energy_bundle, temp_k=T)
    assert math.isinf(got)


def test_multiloop_linear_energy_formula(rna_energy_bundle):
    """
    ΔG = a + b * branches + c * unpaired (+ d if unpaired == 0).
    For MULTILOOP=(2.5, 0.1, 0.4, 2.0), branches=2, unpaired=3 → 3.9
    """
    multiloop_delta_g = multiloop_linear_energy(
        branches=2, unpaired_bases=3, energies=rna_energy_bundle
    )
    assert math.isclose(multiloop_delta_g, 3.9, rel_tol=1e-12)


def test_multiloop_linear_energy_zero_unpaired_adds_bonus(rna_energy_bundle):
    """
    unpaired==0 adds the 'd' bonus once; with (2.5,0.1,0.4,2.0), branches=1 → 4.6
    """
    multiloop_delta_g = multiloop_linear_energy(
        branches=1, unpaired_bases=0, energies=rna_energy_bundle
    )
    assert math.isclose(multiloop_delta_g, 4.6, rel_tol=1e-12)

