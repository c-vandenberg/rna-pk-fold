from __future__ import annotations
from typing import Set, Dict, Tuple, Callable, Any
from rna_pk_fold.structures import Pair
from rna_pk_fold.utils.indices_utils import canonical_pair

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
    base = collect_fn(seq, nested_state, i, j)
    for p in base.pairs:
        add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)
