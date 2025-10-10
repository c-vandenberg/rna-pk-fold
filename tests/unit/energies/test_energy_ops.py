"""
Tests for the low-level energy calculation functions in `energy_ops`.

This module validates the correctness of individual energy calculation functions
(e.g., for hairpins, stacks, internal loops) by comparing their output against
expected values derived directly from the loaded thermodynamic parameter tables.
"""
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
    """
    A pytest fixture that provides the file path to the minimal test energy parameter YAML file.

    This fixture is scoped to the module level, so the path is determined only
    once per test file run.

    Returns
    -------
    str
        The absolute file path to the test YAML data file.
    """
    # Use the same minimal YAML you validated earlier
    return ir_files(rna_pk_fold) / "data" / "xia1998_zucker_turner1999_min.yaml"


@pytest.fixture(scope="module")
def rna_energy_bundle(yaml_path):
    """
    A pytest fixture that loads the Turner 1999 energy parameters for RNA.

    This fixture is scoped to the module, so the energy parameters are loaded
    only once, making the test suite more efficient.

    Parameters
    ----------
    yaml_path : str
        The path to the energy parameter file, provided by the `yaml_path` fixture.

    Returns
    -------
    SecondaryStructureEnergies
        An initialized `SecondaryStructureEnergies` object containing all the
        thermodynamic tables.
    """
    # Return the current RNA energy bundle used by the adapters.
    return SecondaryStructureEnergyLoader().load("RNA", yaml_path=yaml_path)


def test_hairpin_energy_size_3_matches_table(rna_energy_bundle):
    """
    Tests that the hairpin energy for a 3-nucleotide loop matches the table value.

    This test verifies that the `hairpin_energy` function correctly looks up the
    baseline (ΔH, ΔS) values for a loop of a given size and calculates the
    free energy (ΔG) at the specified temperature.

    Parameters
    ----------
    rna_energy_bundle : SecondaryStructureEnergies
        The pytest fixture providing the loaded energy parameters.
    """
    # Define a sequence and indices that form a 3-nucleotide hairpin loop.
    seq = "AAAAA"  # i=0, j=4 => loop length = 3
    i, j = 0, 4
    T = 310.15

    # Calculate the expected energy directly from the HAIRPIN table for size 3.
    expected = calculate_delta_g(rna_energy_bundle.HAIRPIN[3], T)
    # Calculate the energy using the function under test.
    got = hairpin_energy(i, j, seq, rna_energy_bundle, temp_k=T)
    # Assert that the calculated value is very close to the expected value.
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_hairpin_energy_below_min_size_returns_inf(rna_energy_bundle):
    """
    Tests that `hairpin_energy` returns infinity for loops smaller than the minimum allowed size.

    Parameters
    ----------
    rna_energy_bundle : SecondaryStructureEnergies
        The pytest fixture providing the loaded energy parameters.
    """
    # A closing pair (0, 2) creates a loop of length 1, which is smaller than the minimum of 3.
    seq = "AAAA"
    # The function should return infinity for this invalid geometry.
    assert math.isinf(hairpin_energy(0, 2, seq, rna_energy_bundle))


def test_stack_energy_cu_gg_matches_nn(rna_energy_bundle):
    """
    Tests that the stacking energy for a C-G pair on a G-C pair matches the nearest-neighbor table.

    Parameters
    ----------
    rna_energy_bundle : SecondaryStructureEnergies
        The pytest fixture providing the loaded energy parameters.
    """
    # Use a 4-mer that forms two valid closing pairs: (0,3)=C-G and (1,2)=G-C
    seq = "CUGG"
    i, j, k, l = 0, 3, 1, 2
    T = 310.15

    # Build the exact Turner key your code uses for the nearest-neighbor table.
    key = dimer_key(seq, i, j)
    assert key in rna_energy_bundle.NN_STACK, f"Expected NN stack key {key!r} in table"

    # Create a simplified object that mimics the energy bundle structure for the function call.
    energies_shim = SimpleNamespace(
        NN_STACK=rna_energy_bundle.NN_STACK,
        NN=rna_energy_bundle.NN_STACK,
        delta_g=rna_energy_bundle.delta_g,
    )

    # Calculate the expected energy directly from the table.
    expected = calculate_delta_g(rna_energy_bundle.NN_STACK[key], T)
    # Calculate the energy using the function under test.
    got = stack_energy(i, j, k, l, seq, energies_shim, temp_k=T)
    # Assert that the values are close.
    assert math.isclose(got, expected, abs_tol=5e-3)


def test_stack_energy_invalid_geometry_returns_inf(rna_energy_bundle):
    """
    Tests that `stack_energy` returns infinity for an invalid stacking geometry.

    Parameters
    ----------
    rna_energy_bundle : SecondaryStructureEnergies
        The pytest fixture providing the loaded energy parameters.
    """
    seq = "AUAU"
    # An invalid geometry where the indices are not correctly ordered (i < k <= l < j).
    assert math.isinf(stack_energy(0, 2, 1, 1, seq, rna_energy_bundle))


def test_internal_loop_energy_bulge_size_1_uses_bulge_anchor(rna_energy_bundle):
    """
    Tests that a 1-nucleotide bulge correctly uses the `BULGE[1]` table entry.

    The test creates a bulge of size 1 (a=1, b=0) and verifies that the
    calculated energy matches the baseline energy for a size-1 bulge from
    the parameter tables.

    Parameters
    ----------
    rna_energy_bundle : SecondaryStructureEnergies
        The pytest fixture providing the loaded energy parameters.
    """
    # Sequence with a 1-base bulge: A(U A U)A, where (0,4) and (2,3) are pairs, leaving 'U' at index 1 unpaired.
    seq = "AUAUA"
    i, k, l, j = 0, 2, 3, 4
    T = 310.15

    # The expected energy should be the baseline for a bulge of size 1.
    # Note: This test does not account for additional terminal mismatch penalties
    # that the `internal_loop_energy` function might add.
    expected = calculate_delta_g(rna_energy_bundle.BULGE[1], T)
    # Calculate the energy using the function under test.
    got = internal_loop_energy(i, j, k, l, seq, rna_energy_bundle, temp_k=T)
    # Assert that the values are close.
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_internal_loop_energy_internal_size_4_uses_internal_anchor(rna_energy_bundle):
    """
    Tests that a symmetric 2x2 internal loop correctly uses the `INTERNAL[4]` table entry.

    The test creates a 2x2 internal loop (a=2, b=2 => size=4) and verifies that
    the calculated energy matches the baseline for a size-4 internal loop.

    Parameters
    ----------
    rna_energy_bundle : SecondaryStructureEnergies
        The pytest fixture providing the loaded energy parameters.
    """
    # A sequence with a 2x2 internal loop: A(AA U AA)A, pairs are (0,7) and (3,4).
    seq = "A" * 8
    i, k, l, j = 0, 3, 4, 7
    T = 310.15

    # The loop has 2 unpaired bases on each side, for a total size of 4.
    expected = calculate_delta_g(rna_energy_bundle.INTERNAL[4], T)
    # Calculate the energy using the function under test.
    got = internal_loop_energy(i, j, k, l, seq, rna_energy_bundle, temp_k=T)
    # Assert that the values are close.
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_internal_loop_energy_1x1_mismatch_falls_back_or_inf(rna_energy_bundle):
    """
    Tests a 1x1 internal loop with no special mismatch parameters.

    In the minimal parameter set used for testing, there are no specific entries
    for 1x1 internal mismatches. The test verifies that the function correctly
    returns infinity in this case, as it cannot fall back to a general loop calculation.

    Parameters
    ----------
    rna_energy_bundle : SecondaryStructureEnergies
        The pytest fixture providing the loaded energy parameters.
    """
    # A(U G U)A, pairs (0,5) and (2,3), with a 1x1 internal loop of 'G'.
    seq = "AUGAUA"
    i, k, l, j = 0, 2, 3, 5
    T = 310.15

    # Calculate the energy using the function under test.
    got = internal_loop_energy(i, j, k, l, seq, rna_energy_bundle, temp_k=T)
    # Since the minimal table has no 1x1 mismatch entries, the lookup should fail, resulting in infinity.
    assert math.isinf(got)


def test_multiloop_linear_energy_formula(rna_energy_bundle):
    """
    Tests the linear multiloop energy model with non-zero unpaired bases.

    The model is `ΔG = a + b*branches + c*unpaired`. This test uses a specific
    example to verify the formula is applied correctly.

    Parameters
    ----------
    rna_energy_bundle : SecondaryStructureEnergies
        The pytest fixture providing the loaded energy parameters.
    """
    # The test YAML has MULTILOOP=(2.5, 0.1, 0.4, 2.0).
    # For 2 branches and 3 unpaired bases, expected energy is 2.5 + 0.1*2 + 0.4*3 = 3.9
    multiloop_delta_g = multiloop_linear_energy(
        branches=2, unpaired_bases=3, energies=rna_energy_bundle
    )
    assert math.isclose(multiloop_delta_g, 3.9, rel_tol=1e-12)


def test_multiloop_linear_energy_zero_unpaired_adds_bonus(rna_energy_bundle):
    """
    Tests the linear multiloop model with zero unpaired bases, which adds a bonus term.

    The model is `ΔG = a + b*branches + d` when unpaired bases are zero.

    Parameters
    ----------
    rna_energy_bundle : SecondaryStructureEnergies
        The pytest fixture providing the loaded energy parameters.
    """
    # The test YAML has MULTILOOP=(2.5, 0.1, 0.4, 2.0).
    # For 1 branch and 0 unpaired bases, expected energy is 2.5 + 0.1*1 + 2.0 = 4.6
    multiloop_delta_g = multiloop_linear_energy(
        branches=1, unpaired_bases=0, energies=rna_energy_bundle
    )
    assert math.isclose(multiloop_delta_g, 4.6, rel_tol=1e-12)

