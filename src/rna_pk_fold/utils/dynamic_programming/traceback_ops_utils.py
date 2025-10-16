from __future__ import annotations
import logging
from typing import Set, Dict, Tuple, Callable, Any

from rna_pk_fold.structures import Pair
from rna_pk_fold.folding.common_traceback import TraceResult
from rna_pk_fold.utils.sequences.indices_utils import canonical_pair

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
    print(f"\n[MERGE] Interval [{i_index},{j_index}] at layer={layer_index}")
    trace_result = collect_pairs(seq, nested_state, i_index, j_index)
    print(f"[MERGE] Found {len(trace_result.pairs)} nested pairs:")
    for pair in trace_result.pairs:
        print(f"  → ({pair.base_i},{pair.base_j})")
        add_canonical_pair_if_absent(
            pairs, pair_to_layer, pair.base_i, pair.base_j, layer_index
        )


# --- Layer-Safe Placement for Multilayer Dot-Bracket ---
def _pairs_cross(a_pair: tuple[int,int], b_pair: tuple[int,int]) -> bool:
    """A private helper to determine if two base pairs cross."""
    # A crossing (pseudoknot) occurs if the indices are interleaved: i < k < j < l.
    return (a_pair[0] < b_pair[0] < a_pair[1] < b_pair[1]) or (b_pair[0] < a_pair[0] < b_pair[1] < a_pair[1])


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
    print(f"[PLACE_PAIR] Attempting ({i_index},{j_index}) from layer {starting_layer}", flush=True)
    # Start checking from the suggested layer.
    current_layer = starting_layer
    while True:
        # Assume there is no conflict on the current layer.
        conflict_found = False
        # Check the new pair against all existing pairs on this layer.
        for (existing_i, existing_j), layer_idx in pair_to_layer.items():
            if layer_idx == current_layer and _pairs_cross(
                    (i_index, j_index), (existing_i, existing_j)
            ):
                # If a crossing is found, mark a conflict and stop checking this layer.
                has_conflict = True
                print(f"  Conflict with ({existing_i},{existing_j}) on L{current_layer}", flush=True)
                break

        # If no conflicts were found after checking all pairs on this layer...
        if not conflict_found:
            # ...place the new pair on this layer.
            add_canonical_pair_if_absent(pairs, pair_to_layer, i_index, j_index, current_layer)
            print(f"[PAIR] ({i_index},{j_index}) -> L{current_layer}", flush=True)
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
            _pairs_cross(layer_pairs[a], layer_pairs[b])
            for a in range(len(layer_pairs))
            for b in range(a + 1, len(layer_pairs))
        )
        # Print a summary report for the layer.
        print(
            f"[L{layer_idx}] pairs={len(layer_pairs)} crossings_within_layer={within_layer_crossings}",
            flush=True,
        )
