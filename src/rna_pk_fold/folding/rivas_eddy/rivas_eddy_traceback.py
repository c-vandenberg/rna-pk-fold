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
    RE_BP_COMPOSE_WX, RE_BP_COMPOSE_VX, RE_BP_COMPOSE_WX_YHX
)

@dataclass(frozen=True, slots=True)
class RETraceResult:
    pairs: List[Pair]
    dot_bracket: str

def traceback_re_with_pk(seq: str, nested: FoldState, re: RivasEddyState) -> RETraceResult:
    n = re.n
    i, j = 0, n - 1

    pairs: Set[Pair] = set()
    pair_layer: Dict[Tuple[int,int], int] = {}

    bp = re.wx_back_ptr.get((i, j))

    # Case A: WHX+WHX composition (as before)
    if bp and bp[0] == RE_BP_COMPOSE_WX:
        r, k, l = bp[1]
        left = traceback_nested_interval(seq, nested, i, r)
        for p in left.pairs:
            pairs.add(p); pair_layer[(p.base_i, p.base_j)] = 0

        right = traceback_nested_interval(seq, nested, k + 1, j)
        for p in right.pairs:
            pairs.add(p); pair_layer[(p.base_i, p.base_j)] = 1

        ordered = sorted(pairs, key=lambda x: (x.base_i, x.base_j))
        dot = _pairs_to_multilayer_dotbracket(n, ordered, pair_layer)
        return RETraceResult(pairs=ordered, dot_bracket=dot)

    # Case B (NEW): YHX+YHX composition → same layering behavior
    if bp and bp[0] == RE_BP_COMPOSE_WX_YHX:  # NEW
        r, k, l = bp[1]
        left = traceback_nested_interval(seq, nested, i, r)
        for p in left.pairs:
            pairs.add(p); pair_layer[(p.base_i, p.base_j)] = 0

        right = traceback_nested_interval(seq, nested, k + 1, j)
        for p in right.pairs:
            pairs.add(p); pair_layer[(p.base_i, p.base_j)] = 1

        ordered = sorted(pairs, key=lambda x: (x.base_i, x.base_j))
        dot = _pairs_to_multilayer_dotbracket(n, ordered, pair_layer)
        return RETraceResult(pairs=ordered, dot_bracket=dot)

    # Fallback: no PK composition at the top → nested traceback
    base = traceback_nested_interval(seq, nested, i, j)
    return RETraceResult(pairs=base.pairs, dot_bracket=base.dot_bracket)
