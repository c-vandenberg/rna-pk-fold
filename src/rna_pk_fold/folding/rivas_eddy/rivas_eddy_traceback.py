from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

from rna_pk_fold.folding import FoldState
from rna_pk_fold.folding.fold_state import RivasEddyState
from rna_pk_fold.folding.traceback import (
    TraceResult, Pair, BRACKETS, _pairs_to_multilayer_dotbracket,
    traceback_nested_interval
)
from rna_pk_fold.folding.rivas_eddy.rivas_eddy_recurrences import (
    RE_BP_COMPOSE_WX, RE_BP_COMPOSE_VX
)

@dataclass(frozen=True, slots=True)
class RETraceResult:
    pairs: List[Pair]
    dot_bracket: str

def traceback_re_with_pk(seq: str, nested: FoldState, re: RivasEddyState) -> RETraceResult:
    """
    Step-13 R&E traceback (minimal):
      - Start at wx(0,N-1). If composed, split into two sub-intervals and trace each via nested traceback.
      - Assign layers to show crossings (left piece layer 0, right piece layer 1).
      - If not composed, fall back to nested traceback on (0,N-1).
    """
    n = re.n
    i, j = 0, n - 1

    pairs: Set[Pair] = set()
    pair_layer: Dict[Tuple[int,int], int] = {}

    # Prefer wx composition; you can add a flag to choose vx if desired.
    bp = re.wx_back_ptr.get((i, j))

    if bp and bp[0] == RE_BP_COMPOSE_WX:
        r, k, l = bp[1]

        # Left piece: (i..r) on layer 0
        left = traceback_nested_interval(seq, nested, i, r)
        for p in left.pairs:
            pairs.add(p)
            pair_layer[(p.base_i, p.base_j)] = 0

        # Right piece: (k+1..j) on layer 1
        right = traceback_nested_interval(seq, nested, k + 1, j)
        for p in right.pairs:
            pairs.add(p)
            pair_layer[(p.base_i, p.base_j)] = 1

        ordered = sorted(pairs, key=lambda x: (x.base_i, x.base_j))
        dot = _pairs_to_multilayer_dotbracket(n, ordered, pair_layer)
        return RETraceResult(pairs=ordered, dot_bracket=dot)

    # Fallback: no PK composition at top → use nested on (0,N-1)
    base = traceback_nested_interval(seq, nested, i, j)
    return RETraceResult(pairs=base.pairs, dot_bracket=base.dot_bracket)
