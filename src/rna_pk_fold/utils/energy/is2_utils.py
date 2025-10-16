import math
from typing import Tuple, Any, Optional, Iterable

from rna_pk_fold.utils.dynamic_programming.matrix_utils import get_inner_matrix_energy


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _get_first_attr(obj: Any, candidate_names: Iterable[str]) -> Optional[Any]:
    """
    Return the first present attribute from candidate_names on obj, else None.
    """
    for name in candidate_names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def compute_is2_outer_bridge_energy(
    seq: str,
    tables: Any,
    i_index: int,
    j_index: int,
    r_index: int,
    s_index: int
) -> float:
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
    i_index, j_index : int
        The indices of the outer closing pair.
    r_index, s_index : int
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
        energy_calculator = tables.compute_is2_outer_bridge_energy
        # If it's a function, call it with the provided coordinates.
        if callable(energy_calculator):
            return energy_calculator(seq, i_index, j_index, r_index, s_index)
        # If it's not a function, treat it as a pre-calculated float value.
        else:
            return float(energy_calculator)

    # If the required attribute or tables object doesn't exist, return a neutral energy.
    return 0.0


def compute_is2_outer_bridge_energy_yhx(
    config: Any,
    seq: str,
    i_index: int,
    j_index: int,
    r_index: int,
    s_index: int
) -> float:
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
    i_index, j_index : int
        The indices of the outer closing pair.
    r_index, s_index : int
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
    return float(energy_function(seq, i_index, j_index, r_index, s_index))


# ---------------------------------------------------------------------
# Scanning / dispatch
# ---------------------------------------------------------------------
def compute_is2_bridge_energy(
    config: Any,
    seq: str,
    bridge_kind_name: str,
    i_index: int,
    j_index: int,
    r_index: int,
    s_index: int,
) -> float:
    """
    Dispatch to the appropriate IS2 bridge calculator for the requested kind.
    """
    if bridge_kind_name == "yhx":
        return compute_is2_outer_bridge_energy_yhx(
            config, seq, i_index, j_index, r_index, s_index
        )
    if bridge_kind_name == "default":
        tables = getattr(config, "tables", None)
        return compute_is2_outer_bridge_energy(
            seq, tables, i_index, j_index, r_index, s_index
        )
    raise ValueError(f"Unknown bridge_kind: {bridge_kind_name}")


def scan_is2_outer_bridge_candidates(
    fold_state: Any,
    config: Any,
    seq: str,
    i_index: int,
    j_index: int,
    k_index: int,
    l_index: int,
    inner_matrix_name: str,
    bridge_kind_name: str,
    backtrack_op: Any
) -> Tuple[float, Optional[Tuple[int, int]], Any]:
    """
    Scan r in [i..k], s in [l..j] for for the best IS2 outer-bridge placement.

    Returns:
        (best_energy, (r_best, s_best) or None, backtrack_op)
    """
    best_energy = math.inf
    best_coords: Optional[Tuple[int, int]] = None

    for r_index in range(i_index, k_index + 1):
        for s_index in range(l_index, j_index + 1):
            if r_index > s_index:
                continue

            inner_energy = get_inner_matrix_energy(
                fold_state, inner_matrix_name, r_index, s_index, k_index, l_index
            )
            if not math.isfinite(inner_energy):
                continue

            bridge = compute_is2_bridge_energy(
                config, seq, bridge_kind_name, i_index, j_index, r_index, s_index
            )
            candidate = bridge + inner_energy
            if candidate < best_energy:
                best_energy = candidate
                best_coords = (r_index, s_index)

    return best_energy, best_coords, backtrack_op
