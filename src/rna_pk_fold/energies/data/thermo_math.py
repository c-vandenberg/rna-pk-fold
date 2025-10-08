from __future__ import annotations
import math


def delta_g(dh: float, ds: float, temp_k: float) -> float:
    """
    Compute Gibbs free energy change ΔG(T).

    Uses the thermodynamic relation:
    ``ΔG(T) = ΔH − T * (ΔS / 1000)``

    Parameters
    ----------
    dh : float
        Enthalpy change ΔH in kcal/mol.
    ds : float
        Entropy change ΔS in cal/(K·mol).
    temp_k : float
        Absolute temperature T in Kelvin.

    Returns
    -------
    float
        ΔG(T) in kcal/mol, rounded to 2 decimal places.
    """
    return round(float(dh) - float(temp_k) * (float(ds) / 1000.0), 2)


def resolve_dh_ds(*, dh: float | None, ds: float | None, dg: float | None, temp_k: float) -> tuple[float, float]:
    """
    Resolve (ΔH, ΔS) from any two of (ΔH, ΔS, ΔG(T)).

    Given exactly two provided values among ``dh``, ``ds``, and ``dg``, compute the
    missing one using:
        ``ΔG(T) = ΔH − T * (ΔS / 1000)``

    If all three are provided, the function returns the rounded ``(dh, ds)`` pair
    without checking internal consistency of ``dg``.

    Parameters
    ----------
    dh : float or None
        Enthalpy change ΔH in kcal/mol, or ``None`` if unknown.
    ds : float or None
        Entropy change ΔS in cal/(K·mol), or ``None`` if unknown.
    dg : float or None
        Gibbs free energy change ΔG(T) in kcal/mol at temperature ``temp_k``,
        or ``None`` if unknown.
    temp_k : float
        Absolute temperature T in Kelvin used for conversions.

    Returns
    -------
    tuple[float, float]
        A tuple ``(ΔH, ΔS)`` with units (kcal/mol, cal/(K·mol)), each rounded to
        2 decimal places.

    Raises
    ------
    ValueError
        If fewer than two of ``dh``, ``ds``, ``dg`` are provided.
    """
    present: int = sum(v is not None for v in (dh, ds, dg))
    if present < 2:
        raise ValueError("Insufficient thermo terms; need two of (dh, ds, dg).")

    if dh is not None and ds is not None:
        return round(float(dh), 2), round(float(ds), 2)

    if dh is not None and dg is not None:
        # ds = 1000 * (dh − dg) / T
        ds_calc: float = 1000.0 * (float(dh) - float(dg)) / float(temp_k)
        return round(float(dh), 2), round(ds_calc, 2)

    if ds is not None and dg is not None:
        # dh = dg + T * (ds / 1000)
        dh_calc: float = float(dg) + float(temp_k) * (float(ds) / 1000.0)
        return round(dh_calc, 2), round(float(ds), 2)

    # This line is unreachable due to the checks above, but keeps type checkers happy.
    raise ValueError("Unexpected input combination for (dh, ds, dg).")

