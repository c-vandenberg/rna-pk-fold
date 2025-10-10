"""
Unit tests for core thermodynamic utility functions.

This module validates helper functions from `energy_utils` that perform
fundamental calculations for RNA energy models, including the calculation of
Gibbs free energy (ΔG) and the lookup and extrapolation of loop energies.
"""
import math

from rna_pk_fold.utils.energy_utils import (
    calculate_delta_g,
    lookup_loop_baseline_js,
    R_CAL,
)


# ------------------------------
# calculate_delta_g
# ------------------------------

def test_calculate_delta_g_none_returns_inf():
    """
    Tests the guard condition for missing input.
    If the enthalpy/entropy pair is `None`, the free energy is undefined, and
    the function should return +infinity to represent an impossible state.
    """
    assert math.isinf(calculate_delta_g(None, 310.15))


def test_calculate_delta_g_matches_formula():
    """
    Verifies the Gibbs free energy calculation against the standard formula.
    The test ensures that ΔG = ΔH - T * ΔS is computed correctly, including the
    necessary unit conversion for entropy.
    """
    # Define sample thermodynamic parameters: ΔH in kcal/mol, ΔS in cal/(K·mol).
    dh, ds = -7.7, -20.6
    # Define temperature in Kelvin (37 °C).
    T = 310.15
    # The standard formula requires consistent units, so ΔS is converted from cal to kcal.
    expected = dh - T * (ds / 1000.0)
    got = calculate_delta_g((dh, ds), T)
    assert math.isclose(got, expected, rel_tol=1e-12)


# ------------------------------
# lookup_loop_baseline_js
# ------------------------------
def test_lookup_loop_js_exact_hit_returns_table_entry():
    """
    Tests the case where the requested loop size is an exact key in the table.
    The function should return the corresponding (ΔH, ΔS) tuple directly.
    """
    table = {3: (1.3, -13.2), 6: (-2.9, -26.8), 10: (5.0, -4.8)}
    # Verify exact matches for different keys in the table.
    assert lookup_loop_baseline_js(table, 6) == table[6]
    assert lookup_loop_baseline_js(table, 3) == table[3]
    assert lookup_loop_baseline_js(table, 10) == table[10]


def test_lookup_loop_js_extrapolates_from_anchor_delta_s_only():
    """
    Tests the extrapolation logic for a loop size larger than any in the table.
    The function should use the largest available entry as an "anchor" and apply
    a logarithmic size correction only to the entropy (ΔS).
    """
    # The table's largest (and only) entry is for size 6.
    table = {6: (-2.9, -26.8)}
    size = 8  # The requested size is larger than the anchor.
    alpha = 1.75 # Default scaling factor for loop size correction.

    dh, ds = lookup_loop_baseline_js(table, size, alpha=alpha)
    dh_a, ds_a = table[6] # Anchor enthalpy and entropy.

    # Enthalpy (ΔH) should remain constant, equal to the anchor's ΔH.
    assert math.isclose(dh, dh_a, rel_tol=1e-12)

    # Entropy (ΔS) should be adjusted based on the size difference.
    # The formula is: ΔS(n) = ΔS(anchor) - α * R * ln(n / anchor_size).
    expected_ds = ds_a - alpha * R_CAL * math.log(size / 6)
    assert math.isclose(ds, expected_ds, rel_tol=1e-12)


def test_lookup_loop_js_custom_alpha_used_in_extrapolation():
    """
    Verifies that a custom `alpha` parameter is correctly used in the extrapolation.
    """
    table = {6: (-2.0, -20.0)}
    size = 12
    alpha = 2.5 # A non-default alpha value.

    dh, ds = lookup_loop_baseline_js(table, size, alpha=alpha)
    dh_a, ds_a = table[6]

    # Verify ΔH is unchanged.
    assert math.isclose(dh, dh_a, rel_tol=1e-12)
    # Verify ΔS is calculated using the custom alpha.
    expected_ds = ds_a - alpha * R_CAL * math.log(size / 6)
    assert math.isclose(ds, expected_ds, rel_tol=1e-12)


def test_lookup_loop_js_below_min_or_empty_returns_none():
    """
    Tests edge cases where lookup should fail and return `None`.
    This occurs if the energy table is empty or if the requested size is smaller
    than the smallest size available in the table (as extrapolation is not done
    for smaller loops).
    """
    # Case 1: The energy table is empty.
    assert lookup_loop_baseline_js({}, 5) is None
    # Case 2: The requested size (2) is smaller than the minimum key in the table (3).
    table = {3: (1.0, -10.0)}
    assert lookup_loop_baseline_js(table, 2) is None


def test_lookup_loop_js_larger_loops_are_less_stable_delta_g_increases():
    """
    Verifies the physical correctness of the extrapolation.
    For larger loops (size > anchor), the entropic penalty should increase,
    resulting in a less favorable (more positive) Gibbs free energy (ΔG),
    making the loop less stable.
    """
    table = {6: (-2.0, -20.0)}
    T = 310.15

    # Get the energy parameters for the anchor loop and a larger, extrapolated loop.
    dh_a, ds_a = table[6]
    dh_n, ds_n = lookup_loop_baseline_js(table, 12)  # size > anchor

    # Calculate the Gibbs free energy for both.
    g_a = calculate_delta_g((dh_a, ds_a), T)
    g_n = calculate_delta_g((dh_n, ds_n), T)

    # The larger loop should be less stable (have a higher ΔG).
    assert g_n > g_a
