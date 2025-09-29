from __future__ import annotations
import math


def delta_g(dh: float, ds: float, temp_k: float) -> float:
    """
    ΔG(T) = ΔH − T * (ΔS / 1000), with ΔH in kcal/mol, ΔS in cal/(K·mol).
    """
    return round(float(dh) - float(temp_k) * (float(ds) / 1000.0), 2)


def resolve_dh_ds(*, dh: float | None, ds: float | None, dg: float | None, temp_k: float) -> tuple[float, float]:
    """
    Given exactly two of (dh, ds, dg), compute the missing one and return (dh, ds).
    Raises if fewer than two are provided.
    """
    present = sum(v is not None for v in (dh, ds, dg))
    if present < 2:
        raise ValueError("Insufficient thermo terms; need two of (dh, ds, dg).")

    if dh is not None and ds is not None:
        return round(float(dh), 2), round(float(ds), 2)

    if dh is not None and dg is not None:
        # ds = 1000 * (dh − dg) / T
        ds_calc = 1000.0 * (float(dh) - float(dg)) / float(temp_k)

        return round(float(dh), 2), round(ds_calc, 2)

    if ds is not None and dg is not None:
        # dh = dg + T * (ds / 1000)
        dh_calc = float(dg) + float(temp_k) * (float(ds) / 1000.0)

        return round(dh_calc, 2), round(float(ds), 2)
