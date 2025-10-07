from __future__ import annotations
from typing import List, Set, Tuple

from rna_pk_fold.folding.common_traceback import TraceResult, pairs_to_dotbracket
from rna_pk_fold.folding.zucker import ZuckerFoldState, ZuckerBacktrackOp, ZuckerBackPointer
from rna_pk_fold.structures import Pair


def traceback_nested(seq: str, state: ZuckerFoldState) -> TraceResult:
    """Nested-only traceback: raises if a pseudoknot ZuckerBackPointer is encountered."""
    return _traceback_core(seq, state)


def traceback_nested_interval(seq: str, state: ZuckerFoldState, i: int, j: int) -> TraceResult:
    """
    Run the nested traceback but starting at W[i,j] instead of W[0,N-1].
    Produces a full-length dot-bracket (global coords; dots outside [i..j]).
    """
    return _traceback_core_with_seed(seq, state, seed_frames=[('W', i, j)])


def _traceback_core(seq: str, state: ZuckerFoldState) -> TraceResult:
    """
    Nested traceback:
      - start at W[0, N-1]
      - follow W/V/WM back-pointers
      - collect all (i, j) that represent base pairs
      - render plain dot-bracket (no layers)
    """
    n = len(seq)
    if n == 0:
        return TraceResult(pairs=[], dot_bracket="")

    pairs: Set[Pair] = set()
    stack: List[Tuple[str, int, int]] = [('W', 0, n - 1)]

    w_bp = state.w_back_ptr
    v_bp = state.v_back_ptr
    wm_bp = state.wm_back_ptr

    while stack:
        which, i, j = stack.pop()

        if i > j or i < 0 or j >= n:
            continue
        if which == 'W' and i == j:
            continue

        if which == 'W':
            bp: ZuckerBackPointer = w_bp.get(i, j)
            op = bp.operation

            if op is ZuckerBacktrackOp.UNPAIRED_LEFT:
                stack.append(('W', i + 1, j))

            elif op is ZuckerBacktrackOp.UNPAIRED_RIGHT:
                stack.append(('W', i, j - 1))

            elif op is ZuckerBacktrackOp.PAIR:
                pairs.add(Pair(i, j))
                stack.append(('V', i, j))

            elif op is ZuckerBacktrackOp.BIFURCATION:
                k = bp.split_k
                if k is not None:
                    stack.append(('W', i, k))
                    stack.append(('W', k + 1, j))

            # NONE/other => nothing to follow

        elif which == 'V':
            bp: ZuckerBackPointer = v_bp.get(i, j)
            op = bp.operation

            if op is ZuckerBacktrackOp.HAIRPIN:
                pairs.add(Pair(i, j))

            elif op is ZuckerBacktrackOp.STACK and bp.inner is not None:
                pairs.add(Pair(i, j))
                k, l = bp.inner
                stack.append(('V', k, l))

            elif op is ZuckerBacktrackOp.INTERNAL and bp.inner is not None:
                pairs.add(Pair(i, j))
                k, l = bp.inner
                stack.append(('V', k, l))

            elif op is ZuckerBacktrackOp.MULTI_ATTACH and bp.inner is not None:
                pairs.add(Pair(i, j))
                p, q = bp.inner
                # follow the helix content
                stack.append(('V', p, q))
                # remainder of the multiloop contents
                if q + 1 <= j:
                    stack.append(('WM', q + 1, j))

            else:
                # Defensive: record (i,j) if V was chosen but op is NONE/unknown
                pairs.add(Pair(i, j))

        elif which == 'WM':
            bp: ZuckerBackPointer = wm_bp.get(i, j)
            op = bp.operation

            if op is ZuckerBacktrackOp.UNPAIRED_LEFT:
                stack.append(('WM', i + 1, j))

            elif op is ZuckerBacktrackOp.UNPAIRED_RIGHT:
                stack.append(('WM', i, j - 1))

            elif op is ZuckerBacktrackOp.MULTI_ATTACH and bp.inner is not None:
                p, q = bp.inner
                pairs.add(Pair(p, q))
                stack.append(('V', p, q))
                if q + 1 <= j:
                    stack.append(('WM', q + 1, j))

            # NONE => base case / nothing more

    ordered = sorted(pairs, key=lambda pr: (pr.base_i, pr.base_j))
    return TraceResult(pairs=ordered, dot_bracket=pairs_to_dotbracket(n, ordered))


def _traceback_core_with_seed(
    seq: str,
    state: ZuckerFoldState,
    *,
    seed_frames: List[Tuple[str, int, int]],
) -> TraceResult:
    """Same as _traceback_core, but start from the provided (matrix, i, j) frames."""
    n = len(seq)
    if n == 0:
        return TraceResult(pairs=[], dot_bracket="")

    pairs: Set[Pair] = set()
    stack: List[Tuple[str, int, int]] = list(seed_frames)

    w_bp = state.w_back_ptr
    v_bp = state.v_back_ptr
    wm_bp = state.wm_back_ptr

    while stack:
        which, i, j = stack.pop()

        if i > j or i < 0 or j >= n:
            continue
        if which == 'W' and i == j:
            continue

        if which == 'W':
            bp: ZuckerBackPointer = w_bp.get(i, j)
            op = bp.operation

            if op is ZuckerBacktrackOp.UNPAIRED_LEFT:
                stack.append(('W', i + 1, j))

            elif op is ZuckerBacktrackOp.UNPAIRED_RIGHT:
                stack.append(('W', i, j - 1))

            elif op is ZuckerBacktrackOp.PAIR:
                pairs.add(Pair(i, j))
                stack.append(('V', i, j))

            elif op is ZuckerBacktrackOp.BIFURCATION:
                k = bp.split_k
                if k is not None:
                    stack.append(('W', i, k))
                    stack.append(('W', k + 1, j))

        elif which == 'V':
            bp: ZuckerBackPointer = v_bp.get(i, j)
            op = bp.operation

            if op is ZuckerBacktrackOp.HAIRPIN:
                pairs.add(Pair(i, j))

            elif op is ZuckerBacktrackOp.STACK and bp.inner is not None:
                pairs.add(Pair(i, j))
                k, l = bp.inner
                stack.append(('V', k, l))

            elif op is ZuckerBacktrackOp.INTERNAL and bp.inner is not None:
                pairs.add(Pair(i, j))
                k, l = bp.inner
                stack.append(('V', k, l))

            elif op is ZuckerBacktrackOp.MULTI_ATTACH and bp.inner is not None:
                pairs.add(Pair(i, j))
                p, q = bp.inner
                stack.append(('V', p, q))
                if q + 1 <= j:
                    stack.append(('WM', q + 1, j))

            else:
                pairs.add(Pair(i, j))

        elif which == 'WM':
            bp: ZuckerBackPointer = wm_bp.get(i, j)
            op = bp.operation

            if op is ZuckerBacktrackOp.UNPAIRED_LEFT:
                stack.append(('WM', i + 1, j))

            elif op is ZuckerBacktrackOp.UNPAIRED_RIGHT:
                stack.append(('WM', i, j - 1))

            elif op is ZuckerBacktrackOp.MULTI_ATTACH and bp.inner is not None:
                p, q = bp.inner
                pairs.add(Pair(p, q))
                stack.append(('V', p, q))
                if q + 1 <= j:
                    stack.append(('WM', q + 1, j))

    ordered = sorted(pairs, key=lambda pr: (pr.base_i, pr.base_j))

    return TraceResult(pairs=ordered, dot_bracket=pairs_to_dotbracket(n, ordered))

