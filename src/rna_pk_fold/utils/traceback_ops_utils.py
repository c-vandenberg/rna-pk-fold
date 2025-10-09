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
