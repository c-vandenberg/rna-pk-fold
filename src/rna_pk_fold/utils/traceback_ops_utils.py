from __future__ import annotations
import logging
from typing import Set, Dict, Tuple, Callable, Any

from rna_pk_fold.structures import Pair
from rna_pk_fold.utils.indices_utils import canonical_pair

logger = logging.getLogger(__name__)


def add_pair_once(
    pairs: Set[Pair],
    pair_layer_map: Dict[Tuple[int, int], int],
    i: int,
    j: int,
    layer: int = 0,
) -> None:
    """
    Safely adds a base pair to a set and records its layer assignment.

    This function ensures that a pair is represented canonically (i < j) and
    is only added to the `pairs` set once. It then assigns the specified
    `layer` to that pair in the `pair_layer_map`.

    Parameters
    ----------
    pairs : Set[Pair]
        The set of unique `Pair` objects discovered so far. This set is
        modified in place.
    pair_layer_map : Dict[Tuple[int, int], int]
        A dictionary mapping canonical pair tuples `(i, j)` to their assigned
        dot-bracket layer. This dictionary is modified in place.
    i : int
        The 5' index of the base pair.
    j : int
        The 3' index of the base pair.
    layer : int, optional
        The dot-bracket layer to assign to this pair, by default 0.
    """
    i, j = canonical_pair(i, j)
    pr = Pair(i, j)
    if pr not in pairs:
        pairs.add(pr)
        pair_layer_map[(i, j)] = layer


def merge_nested_interval(
    seq: str,
    nested_state: Any,
    i: int, j: int,
    layer: int,
    collect_pairs_fn: Callable[[str, Any, int, int], "TraceResult"],
    pairs: Set[Pair],
    pair_layer_map: Dict[Tuple[int, int], int],
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
    i, j : int
        The start and end indices of the interval to trace.
    layer : int
        The dot-bracket layer to assign to all pairs found in this interval.
    collect_pairs_fn : Callable
        The traceback function for the nested algorithm (e.g., `traceback_nested_interval`).
    pairs : Set[Pair]
        The main set of pairs for the entire structure, which will be updated.
    pair_layer_map : Dict[Tuple[int, int], int]
        The main dictionary mapping pairs to layers, which will be updated.
    """
    print(f"\n[MERGE] Interval [{i},{j}] at layer={layer}")
    base = collect_pairs_fn(seq, nested_state, i, j)
    print(f"[MERGE] Found {len(base.pairs)} nested pairs:")
    for p in base.pairs:
        print(f"  → ({p.base_i},{p.base_j})")
        add_pair_once(pairs, pair_layer_map, p.base_i, p.base_j, layer)


# --- Layer-Safe Placement for Multilayer Dot-Bracket ---
def _crosses(a: tuple[int,int], b: tuple[int,int]) -> bool:
    """A private helper to determine if two base pairs cross."""
    # A crossing (pseudoknot) occurs if the indices are interleaved: i < k < j < l.
    return (a[0] < b[0] < a[1] < b[1]) or (b[0] < a[0] < b[1] < a[1])


def place_pair_non_crossing(
        pairs: set,
        pair_layer_map: dict[tuple[int, int], int],
        i: int,
        j: int,
        start_layer: int
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
    pair_layer_map : dict[tuple[int, int], int]
        The main dictionary mapping pairs to layers, which will be updated.
    i, j : int
        The indices of the new base pair to place.
    start_layer : int
        The first layer to check for a valid placement.

    Returns
    -------
    int
        The layer on which the pair was successfully placed.
    """
    # Start checking from the suggested layer.
    current_layer = start_layer
    while True:
        # Assume there is no conflict on the current layer.
        has_conflict = False
        # Check the new pair against all existing pairs on this layer.
        for (existing_i, existing_j), layer_index in pair_layer_map.items():
            if layer_index == current_layer and _crosses((i, j), (existing_i, existing_j)):
                # If a crossing is found, mark a conflict and stop checking this layer.
                has_conflict = True
                break

        # If no conflicts were found after checking all pairs on this layer...
        if not has_conflict:
            # ...place the new pair on this layer.
            add_pair_once(pairs, pair_layer_map, i, j, current_layer)
            print(f"[PAIR] ({i},{j}) -> L{current_layer}", flush=True)
            # Return the layer where the pair was placed.
            return current_layer

        # If there was a conflict, increment the layer and try again.
        current_layer += 1


def audit_layer_map(pair_layer_map: dict[tuple[int, int], int]) -> None:
    """
    Audits the final layer map to count and report intra-layer crossings.

    This is a debugging utility to verify the correctness of the layering
    algorithm. For a valid multilayer dot-bracket representation, the number
    of crossings within any single layer should be zero.

    Parameters
    ----------
    pair_layer_map : dict[tuple[int, int], int]
        The final dictionary mapping all pairs to their assigned layers.
    """
    # Group all pairs by their assigned layer.
    pairs_by_layer = {}
    for (i, j), layer_index in pair_layer_map.items():
        pairs_by_layer.setdefault(layer_index, []).append((i, j))

    # Iterate through each layer and its list of pairs.
    for layer_index, layer_pairs in sorted(pairs_by_layer.items()):
        # Count the number of crossings between all combinations of pairs within this layer.
        crossing_count = sum(
            _crosses(layer_pairs[idx1], layer_pairs[idx2])
            for idx1 in range(len(layer_pairs))
            for idx2 in range(idx1 + 1, len(layer_pairs))
        )
        # Print a summary report for the layer.
        print(f"[L{layer_index}] pairs={len(layer_pairs)} crossings_within_layer={crossing_count}", flush=True)
