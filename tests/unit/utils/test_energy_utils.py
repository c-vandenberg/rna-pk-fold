# tests/unit/utils/test_energy_utils.py
import math
import pytest

from rna_pk_fold.utils.energy_utils import (
    calculate_delta_g,
    lookup_loop_baseline_js,
    R_CAL,
)


# ------------------------------
# calculate_delta_g
# ------------------------------

def test_calculate_delta_g_none_returns_inf():
    assert math.isinf(calculate_delta_g(None, 310.15))


def test_calculate_delta_g_matches_formula():
    dh, ds = -7.7, -20.6    # kcal/mol, cal/(K·mol)
    T = 310.15              # 37 °C
    expected = dh - T * (ds / 1000.0)
    got = calculate_delta_g((dh, ds), T)
    assert math.isclose(got, expected, rel_tol=1e-12)


# ------------------------------
# lookup_loop_baseline_js
# ------------------------------
def test_lookup_loop_js_exact_hit_returns_table_entry():
    table = {3: (1.3, -13.2), 6: (-2.9, -26.8), 10: (5.0, -4.8)}
    assert lookup_loop_baseline_js(table, 6) == table[6]
    assert lookup_loop_baseline_js(table, 3) == table[3]
    assert lookup_loop_baseline_js(table, 10) == table[10]


def test_lookup_loop_js_extrapolates_from_anchor_delta_s_only():
    table = {6: (-2.9, -26.8)}
    size = 8
    alpha = 1.75

    dh, ds = lookup_loop_baseline_js(table, size, alpha=alpha)
    dh_a, ds_a = table[6]

    # ΔH(n) = ΔH(a)
    assert math.isclose(dh, dh_a, rel_tol=1e-12)

    # ΔS(n) = ΔS(a) − α · R_CAL · ln(n/a)
    expected_ds = ds_a - alpha * R_CAL * math.log(size / 6)
    assert math.isclose(ds, expected_ds, rel_tol=1e-12)


def test_lookup_loop_js_custom_alpha_used_in_extrapolation():
    table = {6: (-2.0, -20.0)}
    size = 12
    alpha = 2.5

    dh, ds = lookup_loop_baseline_js(table, size, alpha=alpha)
    dh_a, ds_a = table[6]
    assert math.isclose(dh, dh_a, rel_tol=1e-12)
    expected_ds = ds_a - alpha * R_CAL * math.log(size / 6)
    assert math.isclose(ds, expected_ds, rel_tol=1e-12)


def test_lookup_loop_js_below_min_or_empty_returns_none():
    assert lookup_loop_baseline_js({}, 5) is None
    table = {3: (1.0, -10.0)}
    assert lookup_loop_baseline_js(table, 2) is None


def test_lookup_loop_js_larger_loops_are_less_stable_delta_g_increases():
    """For alpha>0 and size>anchor, extrapolated ΔS decreases → ΔG increases."""
    table = {6: (-2.0, -20.0)}
    T = 310.15
    dh_a, ds_a = table[6]
    dh_n, ds_n = lookup_loop_baseline_js(table, 12)  # size > anchor

    g_a = calculate_delta_g((dh_a, ds_a), T)
    g_n = calculate_delta_g((dh_n, ds_n), T)
    assert g_n > g_a  # less stable (more positive) for larger loop
