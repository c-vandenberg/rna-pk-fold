from __future__ import annotations
from typing import Mapping, Optional, Tuple


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


def lookup_loop_anchor(
    table: Mapping[int, Tuple[float, float]],
    size: int,
) -> Optional[Tuple[float, float]]:
    """
    Fetch a loop baseline (ΔH, ΔS) for a given loop size using an anchor+clamp policy.

    The tables for hairpin/bulge/internal loops are typically sparse (values at
    selected sizes). This helper returns:
      - The exact (ΔH, ΔS) if `size` is present;
      - otherwise the entry at the largest key `<= size` (clamp down);
      - `None` if `size` is smaller than the smallest key (invalid/too small),
        or if the table is empty.

    Parameters
    ----------
    table : Mapping[int, tuple[float, float]]
        Loop baseline table keyed by integer loop size (nt).
    size : int
        Requested loop size.

    Returns
    -------
    Optional[tuple[float, float]]
        The (ΔH, ΔS) pair or `None` if no valid anchor exists.

    Notes
    -----
    - Centralizing this policy makes it easy to later replace clamping with a
      Jacobson–Stockmayer extrapolation without touching callers.
    """
    if not table:
        return None
    if size in table:
        return table[size]
    # clamp to the largest key <= size
    keys = sorted(table.keys())
    if size < keys[0]:
        return None

    le = [k for k in keys if k <= size]

    return table[max(le)] if le else None