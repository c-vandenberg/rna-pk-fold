import math
from typing import Tuple, Any, Optional

from rna_pk_fold.utils.dynamic_programming.matrix_utils import get_inner_matrix_energy

def is2_outer(seq: str, tables: Any, i: int, j: int, r: int, s: int) -> float:
    """
    Safely calculates the energy for an IS2 (Irreducible Surface of Order 2) outer bridge.

    This function acts as a safe wrapper to compute the energy contribution of the
    "bridge" part of an IS2 motif, which spans from an outer helix `(i, j)` to an
    inner helix `(r, s)`. It dynamically calls a function or uses a float value
    provided in the `tables` object.

    Parameters
    ----------
    seq : str
        The RNA sequence.
    tables : Any
        An object expected to have an `IS2_outer` attribute, which can be
        either a callable function `fn(seq, i, j, r, s)` or a float value.
    i, j : int
        The indices of the outer closing pair.
    r, s : int
        The indices of the inner closing pair.

    Returns
    -------
    float
        The calculated energy for the IS2 outer bridge in kcal/mol, or 0.0 if
        the energy function or value is not defined in the `tables` object.
    """
    # Check if a 'tables' object with the required attribute exists.
    if tables and hasattr(tables, "IS2_outer"):
        # Retrieve the attribute, which could be a function or a constant float.
        energy_calculator = tables.is2_outer
        # If it's a function, call it with the provided coordinates.
        if callable(energy_calculator):
            return energy_calculator(seq, i, j, r, s)
        # If it's not a function, treat it as a pre-calculated float value.
        else:
            return float(energy_calculator)

    # If the required attribute or tables object doesn't exist, return a neutral energy.
    return 0.0


def is2_outer_yhx(config: Any, seq: str, i: int, j: int, r: int, s: int) -> float:
    """
        Safely calculates the IS2 outer bridge energy in the YHX matrix context.

        This is a specialized version of the IS2 energy calculation tailored for the
        recursion rules of the YHX gap matrix. It safely retrieves the appropriate
        energy function from the configuration object.

        Parameters
        ----------
        config : Any
            The folding configuration object, expected to have a `tables` attribute.
        seq : str
            The RNA sequence.
        i, j : int
            The indices of the outer closing pair.
        r, s : int
            The indices of the inner closing pair.

        Returns
        -------
        float
            The calculated energy for the IS2 outer bridge in kcal/mol, or 0.0 if
            the energy function is not defined in the configuration.
        """
    # Safely get the 'tables' object from the main configuration.
    tables = getattr(config, "tables", None)
    if tables is None:
        return 0.0

    # Safely get the specific energy calculation function for the YHX context.
    energy_function = getattr(tables, "IS2_outer_yhx", None)
    if energy_function is None:
        return 0.0

    # Call the function and ensure the result is a float.
    return float(energy_function(seq, i, j, r, s))


def scan_is2_outer(
    state,
    cfg,
    seq: str,
    i: int, j: int, k: int, l: int,
    inner_matrix: str,
    bridge_kind: str,
    op
) -> Tuple[float, Optional[Tuple[int, int]], Any]:
    """
    Scan r in [i..k], s2 in [l..j] for an IS2 outer bridge and return (best_energy, (r,s2), op).
    No nested functions; matrix/bridge selection is by string dispatch.
    """
    best = math.inf
    best_bp: Optional[Tuple[int, int]] = None

    for r in range(i, k + 1):
        for s2 in range(l, j + 1):
            if r > s2:
                continue
            inner = get_inner_matrix_energy(state, inner_matrix, r, s2, k, l)
            if math.isfinite(inner):
                cand = bridge_energy(cfg, seq, bridge_kind, i, j, r, s2) + inner
                if cand < best:
                    best, best_bp = cand, (r, s2)

    return best, best_bp, op

def bridge_energy(cfg, seq: str, bridge_kind: str, i: int, j: int, r: int, s2: int) -> float:
    if bridge_kind == "yhx":
        return is2_outer_yhx(cfg, seq, i, j, r, s2)
    if bridge_kind == "default":
        return is2_outer(seq, cfg.tables, i, j, r, s2)

    raise ValueError(f"Unknown bridge_kind: {bridge_kind}")