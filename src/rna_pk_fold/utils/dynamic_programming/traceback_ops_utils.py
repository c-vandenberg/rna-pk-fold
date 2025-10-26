from __future__ import annotations
import math
import logging
from typing import Set, Dict, Tuple, Callable, Any, Optional

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBacktrackOp
from rna_pk_fold.structures import Pair
from rna_pk_fold.folding.common_traceback import TraceResult
from rna_pk_fold.utils.sequences.indices_utils import canonical_pair
from rna_pk_fold.utils.dynamic_programming.back_pointer_utils import (get_whx_backpointer, get_yhx_backpointer,
                                                                      get_zhx_backpointer, get_vhx_backpointer)

Span = Tuple[int, int]

logger = logging.getLogger(__name__)


def add_canonical_pair_if_absent(
    pairs: Set[Pair],
    pair_to_layer: Dict[Tuple[int, int], int],
    i_index: int,
    j_index: int,
    layer_index: int = 0,
) -> None:
    """
    Safely adds a canonical base pair to a set and records its layer assignment.

    This function ensures that a pair is represented canonically (i < j) and
    is only added to the `pairs` set once. It then assigns the specified
    `layer` to that pair in the `pair_layer_map`.

    Parameters
    ----------
    pairs : Set[Pair]
        The set of unique `Pair` objects discovered so far. This set is
        modified in place.
    pair_to_layer : Dict[Tuple[int, int], int]
        A dictionary mapping canonical pair tuples `(i, j)` to their assigned
        dot-bracket layer. This dictionary is modified in place.
    i_index : int
        The 5' index of the base pair.
    j_index : int
        The 3' index of the base pair.
    layer_index : int, optional
        The dot-bracket layer to assign to this pair, by default 0.
    """
    i_canon, j_canon = canonical_pair(i_index, j_index)
    pair_obj = Pair(i_canon, j_canon)
    if pair_obj not in pairs:
        pairs.add(pair_obj)
        pair_to_layer[(i_canon, j_canon)] = layer_index


def merge_nested_region_pairs(
    seq: str,
    nested_state: Any,
    i_index: int,
    j_index: int,
    layer_index: int,
    collect_pairs: Callable[[str, Any, int, int], "TraceResult"],
    pairs: Set[Pair],
    pair_to_layer: Dict[Tuple[int, int], int],
) -> None:
    """
    Traces a purely nested substructure and merges its pairs into the main collection.

    This function is a bridge between the pseudoknot traceback engine and a
    standard nested traceback engine (like Zuker's). It calls the provided
    `collect_pairs_function` to resolve the structure of a given `[i, j]`
    interval and then adds the discovered pairs to the main `pairs` set,
    assigning them all to the specified `layer`.

    Parameters
    ----------
    seq : str
        The full RNA sequence.
    nested_state : Any
        The state object (containing matrices) for the nested folding engine.
    i_index, j_index : int
        The start and end indices of the interval to trace.
    layer_index : int
        The dot-bracket layer to assign to all pairs found in this interval.
    collect_pairs : Callable
        The traceback function for the nested algorithm (e.g., `traceback_nested_interval`).
    pairs : Set[Pair]
        The main set of pairs for the entire structure, which will be updated.
    pair_to_layer : Dict[Tuple[int, int], int]
        The main dictionary mapping pairs to layers, which will be updated.
    """
    # Fast path: degenerate intervals cannot contain pairs.
    if j_index <= i_index:
        print(f"\n[MERGE] Interval [{i_index},{j_index}] at layer={layer_index}")
        print("[MERGE] Found 0 nested pairs:")
        return

    print(f"\n[MERGE] Interval [{i_index},{j_index}] at layer={layer_index}", flush=True)
    trace_result = collect_pairs(seq, nested_state, i_index, j_index)
    print(f"[MERGE] Found {len(trace_result.pairs)} nested pairs:", flush=True)
    for pair in trace_result.pairs:
        print(f"  → ({pair.base_i},{pair.base_j})")
        add_canonical_pair_if_absent(
            pairs, pair_to_layer, pair.base_i, pair.base_j, layer_index
        )


# --- Layer-Safe Placement for Multilayer Dot-Bracket ---
def _do_pairs_cross(pair_a: tuple[int,int], pair_b: tuple[int,int]) -> bool:
    """
    Determine whether two base pairs geometrically cross (form a pseudoknot).

    A crossing occurs when the index intervals are interleaved:
    `a_i < b_i < a_j < b_j` or `b_i < a_i < b_j < a_j`.

    Parameters
    ----------
    pair_a : tuple[int, int]
        First base pair as `(i, j)` with `i < j`.
    pair_b : tuple[int, int]
        Second base pair as `(k, l)` with `k < l`.

    Returns
    -------
    bool
        `True` if the two pairs cross (are interleaved), `False` otherwise.
    """
    ai, aj = pair_a
    bi, bj = pair_b

    return (ai < bi < aj < bj) or (bi < ai < bj < aj)


def place_pair_in_first_non_crossing_layer(
    pairs: set,
    pair_to_layer: dict[tuple[int, int], int],
    i_index: int,
    j_index: int,
    starting_layer: int
) -> int:
    """
    Places a pair `(i, j)` on the lowest available layer without creating a crossing.

    This function is essential for rendering pseudoknots in multilayer dot-bracket
    notation. It starts checking from `start_layer` and increments the layer
    until it finds one where the new pair `(i, j)` does not cross any existing
    pairs already assigned to that layer.

    Parameters
    ----------
    pairs : set
        The main set of pairs for the entire structure, which will be updated.
    pair_to_layer : dict[tuple[int, int], int]
        The main dictionary mapping pairs to layers, which will be updated.
    i_index, j_index : int
        The indices of the new base pair to place.
    starting_layer : int
        The first layer to check for a valid placement.

    Returns
    -------
    int
        The layer on which the pair was successfully placed.
    """
    # Start checking from the suggested layer.
    current_layer = starting_layer
    while True:
        # Assume there is no conflict on the current layer.
        conflict_found = False
        # Check the new pair against all existing pairs on this layer.
        for (existing_i, existing_j), layer_idx in pair_to_layer.items():
            if layer_idx == current_layer and _do_pairs_cross(
                    (i_index, j_index), (existing_i, existing_j)
            ):
                # If a crossing is found, mark a conflict and stop checking this layer.
                conflict_found = True
                print(f"  Conflict with ({existing_i},{existing_j}) on L{current_layer}", flush=True)
                break

        # If no conflicts were found after checking all pairs on this layer...
        if not conflict_found:
            # ...place the new pair on this layer.
            add_canonical_pair_if_absent(pairs, pair_to_layer, i_index, j_index, current_layer)
            # Return the layer where the pair was placed.
            return current_layer

        # If there was a conflict, increment the layer and try again.
        current_layer += 1


def audit_layer_assignments(pair_to_layer: dict[tuple[int, int], int]) -> None:
    """
    Audits the final layer map to count and report intra-layer crossings.

    This is a debugging utility to verify the correctness of the layering
    algorithm. For a valid multilayer dot-bracket representation, the number
    of crossings within any single layer should be zero.

    Parameters
    ----------
    pair_to_layer : dict[tuple[int, int], int]
        The final dictionary mapping all pairs to their assigned layers.
    """
    # Group all pairs by their assigned layer.
    pairs_grouped_by_layer: Dict[int, list[tuple[int, int]]] = {}
    for (i_index, j_index), layer_idx in pair_to_layer.items():
        pairs_grouped_by_layer.setdefault(layer_idx, []).append((i_index, j_index))

    # Iterate through each layer and its list of pairs.
    for layer_idx, layer_pairs in sorted(pairs_grouped_by_layer.items()):
        # Count the number of crossings between all combinations of pairs within this layer.
        within_layer_crossings = sum(
            _do_pairs_cross(layer_pairs[a], layer_pairs[b])
            for a in range(len(layer_pairs))
            for b in range(a + 1, len(layer_pairs))
        )
        # Print a summary report for the layer.
        print(
            f"[L{layer_idx}] pairs={len(layer_pairs)} crossings_within_layer={within_layer_crossings}",
            flush=True,
        )


def is_strict_subspan(outer_span: Span, candidate_span: Span) -> bool:
    """
    Check whether a span lies strictly inside another span.

    A *strict subspan* means the candidate is fully contained within the outer
    span and is not equal to it.

    Parameters
    ----------
    outer_span : tuple[int, int]
        Inclusive indices ``(i, j)`` of the outer span, with ``i <= j``.
    candidate_span : tuple[int, int]
        Inclusive indices ``(a, b)`` of the candidate span, with ``a <= b``.

    Returns
    -------
    bool
        ``True`` if ``candidate_span`` satisfies ``i <= a <= b <= j`` and
        ``(a, b) != (i, j)``; ``False`` otherwise.

    Notes
    -----
    Indices are assumed to be 0-based and inclusive.
    """
    outer_start, outer_end = outer_span
    cand_start, cand_end = candidate_span
    return (outer_start <= cand_start <= cand_end <= outer_end) and (
            (cand_start, cand_end) != (outer_start, outer_end)
    )


def is_noncollapsed_hole(hole_span: Span) -> bool:
    """
    Determine whether a hole span is non-collapsed.

    In this context, a hole ``(k, l)`` is *non-collapsed* if there is at least
    one index strictly between ``k`` and ``l`` (i.e., ``k + 1 < l``).

    Parameters
    ----------
    hole_span : tuple[int, int]
        Inclusive indices ``(k, l)`` of the hole span, with ``k <= l``.

    Returns
    -------
    bool
        ``True`` if ``k + 1 < l``; ``False`` otherwise.

    Notes
    -----
    Indices are assumed to be 0-based and inclusive.
    """
    k, l = hole_span
    return (k + 1) < l


def validate_is2_bridge_span(
    outer_span: Span,
    hole_span: Span,
    bridge_span: Optional[Span],
    *,
    require_noncollapsed_hole: bool = False,
) -> Optional[Span]:
    """
    Validate an IS2 bridge span and return it if it guarantees progress.

    The bridge must exist, be a strict subspan of the outer span, differ from
    the hole span, and (optionally) the hole must be non-collapsed.

    Parameters
    ----------
    outer_span : tuple[int, int]
        Inclusive indices ``(i, j)`` of the outer span.
    hole_span : tuple[int, int]
        Inclusive indices ``(k, l)`` of the hole span.
    bridge_span : tuple[int, int] or None
        Inclusive indices ``(r, s)`` of the proposed bridge span. If ``None``,
        the validation fails.
    require_noncollapsed_hole : bool, optional
        If ``True``, additionally require that ``hole_span`` be non-collapsed
        (i.e., ``k + 1 < l``). Default is ``False``.

    Returns
    -------
    tuple[int, int] or None
        The validated ``bridge_span`` if all conditions are met; otherwise ``None``.

    Notes
    -----
    Validation conditions:

    * ``bridge_span`` is provided (not ``None``).
    * ``bridge_span`` is a strict subspan of ``outer_span``.
    * ``bridge_span`` is not equal to ``hole_span``.
    * If ``require_noncollapsed_hole`` is ``True``, then ``k + 1 < l`` for ``hole_span``.
    """
    if bridge_span is None:
        return None
    if require_noncollapsed_hole and not is_noncollapsed_hole(hole_span):
        return None
    if not is_strict_subspan(outer_span, bridge_span):
        return None
    if bridge_span == hole_span:
        return None
    return bridge_span


def select_pseudoknot_branch(
    fold_state: Any,
    side_label: str,
    outer_i: int,
    outer_j: int,
    hole_k: int,
    hole_l: int,
) -> Tuple[str, Tuple[int,int,int,int]]:
    """
    Select how to trace a pseudoknot branch: crossing (YHX), nested (WHX), or flattened.

    The decision prefers YHX when it has a backpointer and its energy is no worse
    than WHX; otherwise it falls back to WHX if available; otherwise the branch
    is flattened.

    Parameters
    ----------
    fold_state : Any
        Folding state providing accessors:
        ``get_yhx_backpointer(i, j, k, l)``, ``get_whx_backpointer(i, j, k, l)``,
        and energy queries via ``fold_state.yhx_matrix.get_energy(...)`` and
        ``fold_state.whx_matrix.get_energy(...)``.
    side_label : str
        Label for logging (e.g., ``"L"`` or ``"R"``).
    outer_i : int
        Left index of the outer span.
    outer_j : int
        Right index of the outer span.
    hole_k : int
        Left index of the hole span for this branch.
    hole_l : int
        Right index of the hole span for this branch.

    Returns
    -------
    tuple of (str, tuple of int)
        A pair ``(choice, indices)`` where:
        - ``choice`` is one of ``"YHX"``, ``"WHX"``, or ``"FLATTEN"``.
        - ``indices`` is the 4-tuple ``(outer_i, outer_j, hole_k, hole_l)``.

    Notes
    -----
    Ties are broken in favor of YHX when both have backpointers and YHX is not
    energetically worse than WHX (within a tiny epsilon).
    """
    yhx_backpointer = get_yhx_backpointer(fold_state, outer_i, outer_j, hole_k, hole_l)
    whx_backpointer = get_whx_backpointer(fold_state, outer_i, outer_j, hole_k, hole_l)

    yhx_energy = fold_state.yhx_matrix.get_energy(
        outer_i, outer_j, hole_k, hole_l
    ) if yhx_backpointer is not None else math.inf
    whx_energy = fold_state.whx_matrix.get_energy(
        outer_i, outer_j, hole_k, hole_l
    ) if whx_backpointer is not None else math.inf

    # Prefer YHX when it exists and is no worse than WHX
    if yhx_backpointer is not None and yhx_energy <= whx_energy + 1e-9:
        print(f"[WX CHOOSE-{side_label}] YHX (Ey={yhx_energy:.2f}, Ew={whx_energy:.2f})", flush=True)
        return "YHX", (outer_i, outer_j, hole_k, hole_l)
    if whx_backpointer is not None:
        print(f"[WX CHOOSE-{side_label}] WHX (Ey={yhx_energy:.2f}, Ew={whx_energy:.2f})", flush=True)
        return "WHX", (outer_i, outer_j, hole_k, hole_l)

    print(f"[WX CHOOSE-{side_label}] FLATTEN (no BP in YHX/WHX)", flush=True)
    return "FLATTEN", (outer_i, outer_j, hole_k, hole_l)
