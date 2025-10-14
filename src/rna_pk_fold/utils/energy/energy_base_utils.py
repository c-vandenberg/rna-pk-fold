from __future__ import annotations
from math import log
from typing import Mapping, Optional, Tuple

# Ideal Gas Constant in cal mol⁻¹ K⁻¹
# 8.3145112 J K-1 mol-1 = 1.9872159 cal mol⁻¹ K⁻¹
R_CAL = 1.98720425864083e-3


def calculate_delta_g(delta_h_delta_s: Optional[tuple[float, float]], temp_k: float) -> float:
    """
    Compute Gibbs free energy change, ΔG, from enthalpy/entropy at a temperature.

    Uses the thermodynamic relation `ΔG = ΔH − T * (ΔS / 1000)`
    Where:
        - ΔH is in kcal/mol
        - ΔS is in cal/(K·mol)
        - T is in Kelvin.

    The division by 1000 converts ΔS to kcal/(K·mol).

    Parameters
    ----------
    delta_h_delta_s : tuple[float, float] or None
        Two-tuple `(ΔH, ΔS)` with units `kcal/mol` and `cal/(K·mol)` respectively.
        If `None`, the value is considered unavailable and `+∞` is returned.
    temp_k : float
        Absolute temperature in Kelvin.

    Returns
    -------
    float
        Free energy change in `kcal/mol`. Returns `float('inf')` if
        `delta_h_delta_s` is `None`.
    """
    if delta_h_delta_s is None:
        return float("inf")
    delta_h, delta_s = delta_h_delta_s

    return delta_h - temp_k * (delta_s / 1000.0)


def lookup_loop_baseline_js(
    table: Mapping[int, Tuple[float, float]],
    size: int,
    *,
    alpha: float = 1.75,
) -> Optional[Tuple[float, float]]:
    """
    Fetch a loop baseline (ΔH, ΔS) for a given loop size, using Jacobson–Stockmayer
    (JS) extrapolation when `size` is not tabulated.

    The tables for hairpin/bulge/internal loops are typically given as tabulated
    values at selected sizes. This helper returns:
      - Returns the exact (ΔH, ΔS) if `size` is present;
      - Else anchor at the largest key `a` such that a <= size, apply the JS
        extrapolation to calculate (ΔH, ΔS):
                ΔH(n) = ΔH(a)
                ΔS(n) = ΔS(a) − α · R · ln(n/a)        (R in cal/(K·mol))
            * This yields ΔG(n) = ΔG(a) + α · R_kcal · T · ln(n/a) for any T, where
              R_kcal = R/1000, consistent with ΔG = ΔH − T·ΔS/1000.
      - Return `None` if `size` is smaller than the smallest key (invalid/too small),
        or if the table is empty.

    Parameters
    ----------
    table : Mapping[int, tuple[float, float]]
        Loop baseline table keyed by integer loop size (nt).
    size : int
        Requested loop size.
    alpha : float
        Jacobson–Stockmayer loop-entropy coefficient.

    Returns
    -------
    Optional[tuple[float, float]]
        The (ΔH, ΔS) pair or `None` if table or anchor is not valid.

    Notes
    -----
    - Centralizing this policy makes it easy to later replace clamping with a
      Jacobson–Stockmayer extrapolation without touching callers.
    """
    if not table:
        return None

    if size in table:
        return table[size]

    # Anchor = largest tabulated size <= requested size
    anchor = max((k for k in table.keys() if k <= size), default=None)
    if anchor is None:
        return None

    delta_h_a, delta_s_a = table[anchor]
    delta_h_n = delta_h_a
    delta_s_n = delta_s_a - alpha * R_CAL * log(size / anchor)

    return delta_h_n, delta_s_n