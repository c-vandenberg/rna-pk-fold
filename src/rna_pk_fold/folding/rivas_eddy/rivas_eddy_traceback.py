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
    # Top-level compositions backpointer tags
    RE_BP_COMPOSE_WX, RE_BP_COMPOSE_WX_YHX,
    RE_BP_COMPOSE_WX_YHX_WHX, RE_BP_COMPOSE_WX_WHX_YHX,

    # WHX move backpointer tags
    RE_BP_WHX_SHRINK_LEFT, RE_BP_WHX_SHRINK_RIGHT, RE_BP_WHX_TRIM_LEFT,
    RE_BP_WHX_TRIM_RIGHT, RE_BP_WHX_COLLAPSE, RE_BP_WHX_SS_BOTH,
    RE_BP_WHX_SPLIT_LEFT_WHX_WX, RE_BP_WHX_SPLIT_RIGHT_WX_WHX,
    RE_BP_WHX_OVERLAP_SPLIT,

    # YHX move backpointer tags
    RE_BP_YHX_DANGLE_L, RE_BP_YHX_DANGLE_R, RE_BP_YHX_DANGLE_LR,
    RE_BP_YHX_SS_LEFT, RE_BP_YHX_SS_RIGHT, RE_BP_YHX_SS_BOTH,
    RE_BP_YHX_WRAP_WHX, RE_BP_YHX_WRAP_WHX_L, RE_BP_YHX_WRAP_WHX_R,
    RE_BP_YHX_WRAP_WHX_LR, RE_BP_YHX_SPLIT_LEFT_YHX_WX,
    RE_BP_YHX_SPLIT_RIGHT_WX_YHX,

    # ZHX move backpointer tags
    RE_BP_ZHX_FROM_VHX, RE_BP_ZHX_DANGLE_L, RE_BP_ZHX_DANGLE_R,
    RE_BP_ZHX_DANGLE_LR, RE_BP_ZHX_SS_LEFT, RE_BP_ZHX_SS_RIGHT,
    RE_BP_ZHX_SPLIT_LEFT_ZHX_WX, RE_BP_ZHX_SPLIT_RIGHT_ZHX_WX,
    RE_BP_ZHX_IS2_INNER_VHX,

    # VHX move backpointer tags
    RE_BP_VHX_DANGLE_L, RE_BP_VHX_DANGLE_R, RE_BP_VHX_DANGLE_LR,
    RE_BP_VHX_SS_LEFT, RE_BP_VHX_SS_RIGHT,
    RE_BP_VHX_SPLIT_LEFT_ZHX_WX, RE_BP_VHX_SPLIT_RIGHT_ZHX_WX,
    RE_BP_VHX_IS2_INNER_ZHX, RE_BP_VHX_WRAP_WHX, RE_BP_VHX_CLOSE_BOTH,
)


@dataclass(frozen=True, slots=True)
class RETraceResult:
    pairs: List[Pair]
    dot_bracket: str


def add_pair_once(
    pairs: Set[Pair],
    pair_layer: Dict[Tuple[int, int], int],
    i: int,
    j: int,
    layer: int,
) -> None:
    """
    Record a base pair (i,j) with a layer if it hasn't been recorded yet.
    Ensures i < j. Does NOT overwrite layer if the pair already exists.
    """
    if i > j:
        i, j = j, i
    pr = Pair(i, j)
    if pr not in pairs:
        pairs.add(pr)
        pair_layer[(i, j)] = layer


def traceback_re_with_pk(seq: str, nested: FoldState, re: RivasEddyState) -> RETraceResult:
    """
    Full(er) R&E traceback with layered dot-bracket:
      - Start at WX(0,N-1).
      - Handle WX → (WHX+WHX) and WX → (YHX+YHX) compositions with layer 0/1.
      - Descend WHX/YHX/ZHX/VHX using their backpointers.
      - For plain WX regions (or WHX collapses), run nested traceback on that subinterval.
    """
    n = re.n
    if n == 0:
        return RETraceResult(pairs=[], dot_bracket="")

    # Work stack frames:
    #   ("WX", i, j, layer)
    #   ("WHX", i, j, k, l, layer)
    #   ("YHX", i, j, k, l, layer)
    #   ("ZHX", i, j, k, l, layer)
    #   ("VHX", i, j, k, l, layer)
    Frame = Tuple  # for readability

    pairs: Set[Pair] = set()
    pair_layer: Dict[Tuple[int, int], int] = {}
    stack: List[Frame] = [("WX", 0, n - 1, 0)]

    while stack:
        frame = stack.pop()
        tag = frame[0]

        # ---------------- WX ----------------
        if tag == "WX":
            _, i, j, layer = frame
            bp = re.wx_back_ptr.get((i, j))

            if not bp:
                base = traceback_nested_interval(seq, nested, i, j)
                for p in base.pairs:
                    add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)
                continue

            op, payload = bp[0], bp[1]
            if op == RE_BP_COMPOSE_WX:
                r, k, l = payload
                stack.append(("WHX", i, r, k, l, layer))
                stack.append(("WHX", k + 1, j, l - 1, r + 1, layer + 1))
                continue

            if op == RE_BP_COMPOSE_WX_YHX:
                r, k, l = payload
                stack.append(("YHX", i, r, k, l, layer))
                stack.append(("YHX", k + 1, j, l - 1, r + 1, layer + 1))
                continue

            if op == RE_BP_COMPOSE_WX_YHX_WHX:
                r, k, l = payload
                # left layer = 0, right layer = 1 (consistent with others)
                stack.append(("YHX", i, r, k, l, layer))
                stack.append(("WHX", k + 1, j, l - 1, r + 1, layer + 1))
                continue

            if op == RE_BP_COMPOSE_WX_WHX_YHX:
                r, k, l = payload
                stack.append(("WHX", i, r, k, l, layer))
                stack.append(("YHX", k + 1, j, l - 1, r + 1, layer + 1))
                continue

            # Fallback: treat as nested
            base = traceback_nested_interval(seq, nested, i, j)
            for p in base.pairs:
                add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)
            continue

        # ---------------- WHX ----------------
        if tag == "WHX":
            _, i, j, k, l, layer = frame
            bp = re.whx_back_ptr.get(i, j, k, l)
            if not bp:
                base = traceback_nested_interval(seq, nested, i, j)
                for p in base.pairs:
                    add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)
                continue

            op, payload = bp[0], bp[1]
            if op in (RE_BP_WHX_SHRINK_LEFT, RE_BP_WHX_SHRINK_RIGHT):
                _, _, nk, nl = payload
                stack.append(("WHX", i, j, nk, nl, layer))

            elif op in (RE_BP_WHX_TRIM_LEFT, RE_BP_WHX_TRIM_RIGHT):
                ni, nj, nk, nl = payload
                stack.append(("WHX", ni, nj, nk, nl, layer))

            elif op == RE_BP_WHX_COLLAPSE:
                ni, nj = payload
                base = traceback_nested_interval(seq, nested, ni, nj)
                for p in base.pairs:
                    add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)

            elif op == RE_BP_WHX_SS_BOTH:
                ni, nj, nk, nl = payload
                stack.append(("WHX", ni, nj, nk, nl, layer))

            elif op == RE_BP_WHX_SPLIT_LEFT_WHX_WX:
                (r,) = payload
                stack.append(("WHX", i, r, k, l, layer))
                base = traceback_nested_interval(seq, nested, r + 1, j)
                for p in base.pairs:
                    add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)

            elif op == RE_BP_WHX_SPLIT_RIGHT_WX_WHX:
                (s2,) = payload
                base = traceback_nested_interval(seq, nested, i, s2)
                for p in base.pairs:
                    add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)
                stack.append(("WHX", s2 + 1, j, k, l, layer))

            elif op == RE_BP_WHX_OVERLAP_SPLIT:
                (r,) = payload
                stack.append(("WHX", i, r, k, l, layer))
                stack.append(("WHX", r + 1, j, k, l, layer))

            continue

        # ---------------- YHX ----------------
        if tag == "YHX":
            _, i, j, k, l, layer = frame
            # Ensure the inner (k,l) helix is recorded
            add_pair_once(pairs, pair_layer, k, l, layer)

            bp = re.yhx_back_ptr.get(i, j, k, l)
            if not bp:
                continue
            op, payload = bp[0], bp[1]

            if op in (RE_BP_YHX_DANGLE_L, RE_BP_YHX_DANGLE_R, RE_BP_YHX_DANGLE_LR,
                      RE_BP_YHX_SS_LEFT, RE_BP_YHX_SS_RIGHT, RE_BP_YHX_SS_BOTH):
                ni, nj, nk, nl = payload
                stack.append(("YHX", ni, nj, nk, nl, layer))

            elif op in (RE_BP_YHX_WRAP_WHX, RE_BP_YHX_WRAP_WHX_L,
                        RE_BP_YHX_WRAP_WHX_R, RE_BP_YHX_WRAP_WHX_LR):
                wi, wj, wk, wl = payload
                stack.append(("WHX", wi, wj, wk, wl, layer))

            elif op == RE_BP_YHX_SPLIT_LEFT_YHX_WX:
                (r,) = payload
                stack.append(("YHX", i, r, k, l, layer))
                base = traceback_nested_interval(seq, nested, r + 1, j)
                for p in base.pairs:
                    add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)

            elif op == RE_BP_YHX_SPLIT_RIGHT_WX_YHX:
                (s2,) = payload
                base = traceback_nested_interval(seq, nested, i, s2)
                for p in base.pairs:
                    add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)
                stack.append(("YHX", s2 + 1, j, k, l, layer))

            continue

        # ---------------- ZHX ----------------
        if tag == "ZHX":
            _, i, j, k, l, layer = frame
            # Ensure the outer (i,j) helix is recorded
            add_pair_once(pairs, pair_layer, i, j, layer)

            bp = re.zhx_back_ptr.get(i, j, k, l)
            if not bp:
                continue
            op, payload = bp[0], bp[1]

            if op in (RE_BP_ZHX_FROM_VHX, RE_BP_ZHX_DANGLE_L, RE_BP_ZHX_DANGLE_R,
                      RE_BP_ZHX_DANGLE_LR):
                ni, nj, nk, nl = payload
                stack.append(("VHX", ni, nj, nk, nl, layer))

            elif op in (RE_BP_ZHX_SS_LEFT, RE_BP_ZHX_SS_RIGHT):
                ni, nj, nk, nl = payload
                stack.append(("ZHX", ni, nj, nk, nl, layer))

            elif op == RE_BP_ZHX_SPLIT_LEFT_ZHX_WX:
                (r,) = payload
                stack.append(("ZHX", i, j, r, l, layer))
                base = traceback_nested_interval(seq, nested, r + 1, k)
                for p in base.pairs:
                    add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)

            elif op == RE_BP_ZHX_SPLIT_RIGHT_ZHX_WX:
                (s2,) = payload
                stack.append(("ZHX", i, j, k, s2, layer))
                base = traceback_nested_interval(seq, nested, l, s2 - 1)
                for p in base.pairs:
                    add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)

            elif op == RE_BP_ZHX_IS2_INNER_VHX:
                r, s2 = payload
                stack.append(("VHX", r, s2, k, l, layer))

            continue

        # ---------------- VHX ----------------
        if tag == "VHX":
            _, i, j, k, l, layer = frame
            # Ensure the inner (k,l) helix is recorded
            add_pair_once(pairs, pair_layer, k, l, layer)

            bp = re.vhx_back_ptr.get(i, j, k, l)
            if not bp:
                continue
            op, payload = bp[0], bp[1]

            if op in (RE_BP_VHX_DANGLE_L, RE_BP_VHX_DANGLE_R, RE_BP_VHX_DANGLE_LR):
                ni, nj, nk, nl = payload
                stack.append(("VHX", ni, nj, nk, nl, layer))

            elif op in (RE_BP_VHX_SS_LEFT, RE_BP_VHX_SS_RIGHT):
                ni, nj, nk, nl = payload
                stack.append(("ZHX", ni, nj, nk, nl, layer))

            elif op == RE_BP_VHX_SPLIT_LEFT_ZHX_WX:
                (r,) = payload
                stack.append(("ZHX", i, j, r, l, layer))
                base = traceback_nested_interval(seq, nested, r + 1, k)
                for p in base.pairs:
                    add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)

            elif op == RE_BP_VHX_SPLIT_RIGHT_ZHX_WX:
                (s2,) = payload
                stack.append(("ZHX", i, j, k, s2, layer))
                base = traceback_nested_interval(seq, nested, l, s2 - 1)
                for p in base.pairs:
                    add_pair_once(pairs, pair_layer, p.base_i, p.base_j, layer)

            elif op == RE_BP_VHX_IS2_INNER_ZHX:
                r, s2 = payload
                stack.append(("ZHX", r, s2, k, l, layer))

            elif op == RE_BP_VHX_WRAP_WHX:
                wi, wj = payload
                stack.append(("WHX", wi, wj, k, l, layer))

            elif op == RE_BP_VHX_CLOSE_BOTH:
                # Record both helices and dive into inner WHX if provided
                add_pair_once(pairs, pair_layer, i, j, layer)
                add_pair_once(pairs, pair_layer, k, l, layer)
                if isinstance(payload, tuple) and len(payload) == 4:
                    wi, wj, wk, wl = payload
                    stack.append(("WHX", wi, wj, wk, wl, layer))

            continue

        # Unknown tag → ignore and continue

    ordered_pairs = sorted(pairs, key=lambda pr: (pr.base_i, pr.base_j))
    dot = _pairs_to_multilayer_dotbracket(n, ordered_pairs, pair_layer)
    return RETraceResult(pairs=ordered_pairs, dot_bracket=dot)
