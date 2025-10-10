from __future__ import annotations
import logging
from typing import Set, Dict, Tuple, Callable, Any

from rna_pk_fold.structures import Pair
from rna_pk_fold.utils.indices_utils import canonical_pair

logger = logging.getLogger(__name__)


def add_pair_once(
    pairs: Set[Pair],
    pair_layer: Dict[Tuple[int, int], int],
    i: int,
    j: int,
    layer: int = 0,
) -> None:
    i, j = canonical_pair(i, j)
    pr = Pair(i, j)
    if pr not in pairs:
        pairs.add(pr)
        pair_layer[(i, j)] = layer


def merge_nested_interval(
    seq: str,
    nested_state: Any,
    i: int, j: int,
    layer: int,
    collect_fn: Callable[[str, Any, int, int], "TraceResult"],
    pairs: Set[Pair],
    pair_layer: Dict[Tuple[int, int], int],
) -> None:
    print(f"\n[MERGE] Interval [{i},{j}] at layer={layer}")
    base = collect_fn(seq, nested_state, i, j)
    print(f"[MERGE] Found {len(base.pairs)} nested pairs:")
    for p in base.pairs:
        print(f"  → ({p.base_i},{p.base_j})")
        add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)


# --- layer-safe placement (avoids crossings within the same layer) ---
def _crosses(a: tuple[int,int], b: tuple[int,int]) -> bool:
    return (a[0] < b[0] < a[1] < b[1]) or (b[0] < a[0] < b[1] < a[1])


def place_pair_non_crossing(
    pairs: set,
    pair_layer: dict[tuple[int,int], int],
    i: int,
    j: int,
    start_layer: int
) -> int:
    """Place (i,j) at the lowest layer ≥ start_layer that doesn't create
    an intra-layer crossing. Returns the chosen layer."""
    layer = start_layer
    while True:
        conflict = False
        for (pi, pj), L in pair_layer.items():
            if L == layer and _crosses((i, j), (pi, pj)):
                conflict = True
                break
        if not conflict:
            # actually place it
            add_pair_once(pairs, pair_layer, i, j, layer)
            print(f"[PAIR] ({i},{j}) -> L{layer}", flush=True)
            return layer
        layer += 1


def audit_layer_map(pair_layer: dict[tuple[int, int], int]) -> None:
    by_layer = {}
    for (i, j), lay in pair_layer.items():
        by_layer.setdefault(lay, []).append((i, j))

    def crosses(a, b):
        return (a[0] < b[0] < a[1] < b[1]) or (b[0] < a[0] < b[1] < a[1])

    for lay, ps in sorted(by_layer.items()):
        c = sum(crosses(ps[u], ps[v]) for u in range(len(ps)) for v in range(u + 1, len(ps)))
        print(f"[L{lay}] pairs={len(ps)} crossings_within_layer={c}", flush=True)