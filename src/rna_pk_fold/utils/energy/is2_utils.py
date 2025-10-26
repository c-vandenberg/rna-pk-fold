import math
from typing import Tuple, Any, Optional, Iterable

from rna_pk_fold.utils.dynamic_programming.matrix_utils import get_gap_energy_for_named_matrix
from rna_pk_fold.utils.dynamic_programming.traceback_ops_utils import validate_is2_bridge_span
from rna_pk_fold.utils.dynamic_programming.dp_composition_utils import hole_is_pairable

Span = Tuple[int, int]

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _get_first_attr(obj: Any, candidate_names: Iterable[str]) -> Optional[Any]:
    """
    Return the first existing attribute of `obj` from a list of candidate names.

    Parameters
    ----------
    obj : Any
        Object to inspect.
    candidate_names : Iterable[str]
        Candidate attribute names to check, in priority order.

    Returns
    -------
    Any or None
        The value of the first attribute found, or ``None`` if none exist.
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
    if not tables:
        return 0.0

    # Safely get the specific energy calculation function for the YHX context.
    energy_calculator = getattr(tables, "IS2_outer", None)
    if energy_calculator is None:
        return 0.0

    # Call the function and ensure the result is a float.
    if callable(energy_calculator):
        return float(energy_calculator(seq, i_index, j_index, r_index, s_index))

    return float(energy_calculator)


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
    outer_i: int,
    outer_j: int,
    bridge_i: int,
    bridge_j: int,
) -> float:
    """
    Compute the IS2 outer-bridge energy for a given bridge kind.

    Parameters
    ----------
    config : Any
        Configuration object. When `bridge_kind == "default"`, its `tables`
        attribute is passed to the generic bridge energy routine.
    seq : str
        RNA sequence.
    bridge_kind_name : {"yhx", "default"}
        Selector for the bridge energy model.
    outer_i : int
        Left index of the outer span `[i..j]`.
    outer_j : int
        Right index of the outer span `[i..j]`.
    bridge_i : int
        Left index of the bridge span `[r..s]`.
    bridge_j : int
        Right index of the bridge span `[r..s]`.

    Returns
    -------
    float
        Bridge energy (kcal/mol).

    Raises
    ------
    ValueError
        If `bridge_kind` is not recognized.
    """
    if bridge_kind_name == "yhx":
        return compute_is2_outer_bridge_energy_yhx(
            config, seq, outer_i, outer_j, bridge_i, bridge_j
        )
    if bridge_kind_name == "default":
        tables = getattr(config, "tables", None)
        return compute_is2_outer_bridge_energy(
            seq, tables, outer_i, outer_j, bridge_i, bridge_j
        )
    raise ValueError(f"Unknown bridge_kind: {bridge_kind_name}")


def find_best_is2_outer_bridge(
    fold_state: Any,
    config: Any,
    seq: str,
    outer_i: int,
    outer_j: int,
    hole_k: int,
    hole_l: int,
    inner_matrix_name: str,
    bridge_kind_name: str,
    backtrack_op: Any
) -> Tuple[float, Optional[Span], Any]:
    """
    Search over candidate bridges and return the best IS2 outer-bridge placement.

    Enumerates ``r in [outer_i .. hole_k-1]`` and ``s in [hole_l+1 .. outer_j]``.
    Each candidate must:
      - satisfy strict progress (strict subspan within ``[outer_i..outer_j]``),
      - differ from the hole ``(hole_k, hole_l)``,
      - and use a **non-collapsed** hole (``hole_k + 1 < hole_l``).

    The objective is: ``inner_gap_energy(r, s, k, l) + bridge_cost(i, j, r, s)``.

    Parameters
    ----------
    fold_state : Any
        Folding state providing access to gap matrix energies via
        ``get_gap_energy_for_named_matrix(fold_state, inner_matrix_name, r, s, k, l)``.
    config : Any
        Configuration object passed to bridge energy calculators.
    seq : str
        RNA sequence.
    outer_i : int
        Left index of the outer span.
    outer_j : int
        Right index of the outer span.
    hole_k : int
        Left index of the hole span.
    hole_l : int
        Right index of the hole span.
    inner_matrix_name : str
        Name of the inner gap matrix to consult (e.g., ``"yhx"``, ``"zhx"``, ``"vhx"``, ``"whx"``).
    bridge_kind_name : {"yhx", "default"}
        Bridge energy model selector passed to :func:`compute_is2_bridge_cost`.
    backtrack_op : Any
        Backtracking opcode to return alongside the best candidate.

    Returns
    -------
    tuple
        ``(best_energy, best_bridge, backtrack_op)`` where:
        - ``best_energy`` : float
            Minimal total energy (kcal/mol), or ``math.inf`` if no feasible candidate exists.
        - ``best_bridge`` : tuple(int, int) or ``None``
            The best bridge coordinates ``(r, s)`` or ``None`` if none feasible.
        - ``backtrack_op`` : Any
            The same value that was passed in, for caller convenience.
    """
    best_energy = math.inf
    best_coords: Optional[Span] = None

    # Strict ranges (force the bridge to shrink something on at least one side)
    # r ∈ [i..k-1], s ∈ [l+1..j]
    for r_index in range(outer_i, hole_k):
        for s_index in range(hole_l + 1, outer_j + 1):
            # Ensure r ≤ s (ignore empty/inverted spans)
            if r_index > s_index:
                continue

            # Gate: must be a strict subspan of (i,j), not equal to (i,j), and not equal to (k,l).
            candidate_bridge = (r_index, s_index)
            if validate_is2_bridge_span(
                (outer_i, outer_j),
                (hole_k, hole_l),
                candidate_bridge,
                require_noncollapsed_hole=True,  # avoids k+1 >= l
            ) is None:
                continue

            inner_energy = get_gap_energy_for_named_matrix(
                fold_state, inner_matrix_name, r_index, s_index, hole_k, hole_l
            )
            if not math.isfinite(inner_energy):
                continue

            if bridge_kind_name == "yhx":
                bridge_cost = compute_is2_outer_bridge_energy_yhx(
                    config, seq, outer_i, outer_j, r_index, s_index
                )
            elif bridge_kind_name == "default":
                tables = getattr(config, "tables", None)
                bridge_cost = compute_is2_outer_bridge_energy(
                    seq, tables, outer_i, outer_j, r_index, s_index
                )
            else:
                bridge_cost = 0.0  # conservative fallback

            candidate = bridge_cost + inner_energy
            if candidate < best_energy:
                best_energy = candidate
                best_coords = candidate_bridge

    return best_energy, best_coords, backtrack_op


def validate_is2_progress(
    outer_i: int,
    outer_j: int,
    hole_k: int,
    hole_l: int,
    bridge_i: int,
    bridge_j: int,
    *,
    require_proper_hole: bool,
    can_pair_mask=None,
) -> tuple[bool, dict]:
    """
    Validate DP-time geometry/progress constraints for an IS2 candidate.

    The following guards are enforced:
      1. The bridge is a strict subspan of the outer interval (and not equal to it).
      2. The bridge differs from the hole coordinates.
      3. The hole is proper (`hole_k + 1 < hole_l`) if `require_proper_hole` is True.
      4. The hole pair is permitted by `can_pair_mask` if provided.

    Parameters
    ----------
    outer_i : int
        Left index of the outer span.
    outer_j : int
        Right index of the outer span.
    hole_k : int
        Left index of the hole span.
    hole_l : int
        Right index of the hole span.
    bridge_i : int
        Left index of the bridge span.
    bridge_j : int
        Right index of the bridge span.
    require_proper_hole : bool
        If True, require `hole_k + 1 < hole_l`.
    can_pair_mask : Any, optional
        Optional pairing mask used to validate that `(hole_k, hole_l)` is pairable.

    Returns
    -------
    (bool, dict)
        A tuple `(ok, flags)` where:
        - `ok` : bool
            `True` if all guards pass; else `False`.
        - `flags` : dict of str -> bool
            Individual guard outcomes with keys:
            `"strict_subspan"`, `"not_equal_hole"`, `"hole_proper"`, `"hole_pairable"`.
    """
    bridge_is_strict_subspan = (
            (outer_i <= bridge_i <= bridge_j <= outer_j) and ((bridge_i, bridge_j) != (outer_i, outer_j))
    )
    bridge_differs_from_hole = (bridge_i, bridge_j) != (hole_k, hole_l)
    hole_is_proper = (hole_k + 1) < hole_l if require_proper_hole else True
    is_hole_pairable = hole_is_pairable(can_pair_mask, hole_k, hole_l)

    ok = bridge_is_strict_subspan and bridge_differs_from_hole and hole_is_proper and hole_is_pairable
    return ok, {
        "strict_subspan": bridge_is_strict_subspan,
        "not_equal_hole": bridge_differs_from_hole,
        "hole_proper": hole_is_proper,
        "hole_pairable":  is_hole_pairable,
    }
