from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict

from rna_pk_fold.folding import ZuckerFoldState, BacktrackOp, BackPointer
from rna_pk_fold.structures import Pair

BRACKETS: List[Tuple[str, str]] = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]


@dataclass(frozen=True, slots=True)
class TraceResult:
    """Result of traceback for nested structures."""
    pairs: List[Pair]        # List of Pair(base_i, base_j) with base_i < base_j
    dot_bracket: str         # E.g., '..((...))..'

def traceback_nested(seq: str, state: ZuckerFoldState) -> TraceResult:
    """Nested-only traceback: raises if a pseudoknot backpointer is encountered."""
    return _traceback_core(seq, state, allow_pk=False)

def traceback_with_pk(seq: str, state: ZuckerFoldState) -> TraceResult:
    """Traceback that supports the minimal H-type pseudoknot (two layers)."""
    return _traceback_core(seq, state, allow_pk=True)


def traceback_nested_interval(seq: str, state: ZuckerFoldState, i: int, j: int) -> TraceResult:
    """
    Run the nested traceback but starting at W[i,j] instead of W[0,N-1].
    Produces a full-length dot-bracket (global coords; dots outside [i..j]).
    """
    seed_frames: List[Tuple[str, int, int, int]] = [('W', i, j, 0)]
    return _traceback_core_with_seed(seq, state, seed_frames=seed_frames, allow_pk=False)


def pairs_to_dotbracket(dot_brac_len: int, pairs: List[Pair]) -> str:
    """
    Create a dot-bracket string of length `dot_brac_len` from a list of pairs.
    Assumes pairs are non-crossing (nested only) with i < j.
    """
    chars = ['.'] * dot_brac_len
    for pair in pairs:
        if 0 <= pair.base_i < dot_brac_len and 0 <= pair.base_j < dot_brac_len and pair.base_i < pair.base_j :
            chars[pair.base_i] = '('
            chars[pair.base_j ] = ')'
    return ''.join(chars)


def _traceback_core(seq: str, state: ZuckerFoldState, *, allow_pk: bool) -> TraceResult:
    """
    Shared Traceback:
      - start at W[0, N-1]
      - follow W/V/WM back-pointers
      - collect all (i, j) that represent base pairs
      - render dot-bracket; with layered brackets if allow_pk=True

    Assumes W, V, WM and their back-pointers have been filled.
    """
    seq_len = len(seq)
    if seq_len == 0:
        return TraceResult(pairs=[], dot_bracket="")

    pairs: Set[Pair] = set()

    # Worklist of frames to process (matrix_tag, i, j): Each frame says which
    # matrix to consult and which (i, j) coordinates to follow.
    frame_stack: List[Tuple[str, int, int, int]] = [('W', 0, seq_len - 1, 0)]

    w_back_ptr = state.w_back_ptr
    v_back_ptr = state.v_back_ptr
    wm_back_ptr = state.wm_back_ptr

    # Layer assignment per pair (only meaningful if allow_pk=True)
    pair_layer: Dict[Tuple[int, int], int] = {}

    while frame_stack:
        which, i, j, layer = frame_stack.pop()

        # Out-of-range or empty intervals are benign
        if i > j or i < 0 or j >= seq_len:
            continue

        if which == 'W' and i == j:
            # Base case W[i,i] = 0
            continue

        if which == 'W':
            back_ptr: BackPointer = w_back_ptr.get(i, j)
            op = back_ptr.operation

            if op is BacktrackOp.UNPAIRED_LEFT:
                frame_stack.append(('W', i + 1, j, layer))

            elif op is BacktrackOp.UNPAIRED_RIGHT:
                frame_stack.append(('W', i, j - 1, layer))

            elif op is BacktrackOp.PAIR:
                # W chose to use V[i,j]
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = layer

                frame_stack.append(('V', i, j, layer))

            elif op is BacktrackOp.BIFURCATION:
                k = back_ptr.split_k
                if k is not None:
                    frame_stack.append(('W', i, k, layer))
                    frame_stack.append(('W', k + 1, j, layer))

            elif op is BacktrackOp.PSEUDOKNOT_H:
                if not allow_pk:
                    raise AssertionError(
                        f"traceback_nested encountered a pseudoknot at W[{i},{j}]. "
                        f"Use traceback_with_pk or disable PK during DP."
                    )
                # Two crossing stems + four W segments
                if back_ptr.inner is None or back_ptr.inner_2 is None:
                    # Defensive: malformed backpointer
                    continue
                p1, q1 = back_ptr.inner  # e.g. (i, c)
                p2, q2 = back_ptr.inner_2  # e.g. (b, j)
                segs = tuple(back_ptr.segs or ())

                # Stem 1 on current layer
                pairs.add(Pair(p1, q1))
                pair_layer[(p1, q1)] = layer
                frame_stack.append(('V', p1, q1, layer))

                # Stem 2 on next layer
                pairs.add(Pair(p2, q2))
                pair_layer[(p2, q2)] = layer + 1
                frame_stack.append(('V', p2, q2, layer + 1))

                # Non-crossing regions on current layer
                for lo, hi in segs:
                    if lo <= hi:
                        frame_stack.append(('W', lo, hi, layer))

            # NONE/other => nothing to follow

        elif which == 'V':
            back_ptr: BackPointer = v_back_ptr.get(i, j)
            op = back_ptr.operation

            if op is BacktrackOp.HAIRPIN:
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = pair_layer.get((i, j), layer)

            elif op is BacktrackOp.STACK and back_ptr.inner is not None:
                # Base pair (i,j) stacked on (i+1, j-1)
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = pair_layer.get((i, j), layer)

                inn_i, inn_j = back_ptr.inner
                frame_stack.append(('V', inn_i, inn_j, layer))

            elif op is BacktrackOp.INTERNAL and back_ptr.inner is not None:
                # Internal/bulge with inner (k,l)
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = pair_layer.get((i, j), layer)
                k, l = back_ptr.inner
                frame_stack.append(('V', k, l, layer))

            elif op is BacktrackOp.MULTI_ATTACH and back_ptr.inner is not None:
                # Close a multiloop via a + WM[i+1, j-1]
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = pair_layer.get((i, j), layer)

                inn_i, inn_j = back_ptr.inner  # Typically (i+1, j-1)
                frame_stack.append(('WM', inn_i, inn_j, layer))

            else:
                # Defensive: record (i,j) if V was chosen but op is NONE/unknown
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = pair_layer.get((i, j), layer)

        elif which == 'WM':
            back_ptr: BackPointer = wm_back_ptr.get(i, j)
            op = back_ptr.operation

            if op is BacktrackOp.UNPAIRED_LEFT:
                frame_stack.append(('WM', i + 1, j, layer))

            elif op is BacktrackOp.UNPAIRED_RIGHT:
                frame_stack.append(('WM', i, j - 1, layer))

            elif op is BacktrackOp.MULTI_ATTACH and back_ptr.inner is not None:
                # Attach a helix at (p,q) then continue with WM[q+1, j]
                p, q = back_ptr.inner
                pairs.add(Pair(p, q))
                if allow_pk:
                    pair_layer[(p, q)] = pair_layer.get((p, q), layer)

                # Follow the helix content via V
                frame_stack.append(('V', p, q, layer))
                # and the remainder (if any) of the multiloop contents
                if q + 1 <= j:
                    frame_stack.append(('WM', q + 1, j, layer))

            # NONE => base case or nothing more

    # Sort pairs for deterministic output
    ordered_pairs = sorted(pairs, key=lambda base_pairs: (base_pairs.base_i, base_pairs.base_j))

    if not allow_pk:
        dot_brac = pairs_to_dotbracket(seq_len, ordered_pairs)
    else:
        dot_brac = _pairs_to_multilayer_dotbracket(seq_len, ordered_pairs, pair_layer)

    return TraceResult(pairs=ordered_pairs, dot_bracket=dot_brac)


def _traceback_core_with_seed(
    seq: str,
    state: ZuckerFoldState,
    *,
    seed_frames: List[Tuple[str, int, int, int]],
    allow_pk: bool,
) -> TraceResult:
    """
    Same as _traceback_core, but the initial frame stack is provided by caller.
    This lets us start at an arbitrary sub-interval (i,j).
    """
    seq_len = len(seq)
    if seq_len == 0:
        return TraceResult(pairs=[], dot_bracket="")

    pairs: Set[Pair] = set()

    # Use caller-provided seed instead of ('W', 0, N-1, 0)
    frame_stack: List[Tuple[str, int, int, int]] = list(seed_frames)

    w_back_ptr = state.w_back_ptr
    v_back_ptr = state.v_back_ptr
    wm_back_ptr = state.wm_back_ptr

    # Layer assignment per pair (only meaningful if allow_pk=True)
    pair_layer: Dict[Tuple[int, int], int] = {}

    while frame_stack:
        which, i, j, layer = frame_stack.pop()

        # Out-of-range or empty intervals are benign
        if i > j or i < 0 or j >= seq_len:
            continue

        if which == 'W' and i == j:
            # Base case W[i,i] = 0
            continue

        if which == 'W':
            back_ptr: BackPointer = w_back_ptr.get(i, j)
            op = back_ptr.operation

            if op is BacktrackOp.UNPAIRED_LEFT:
                frame_stack.append(('W', i + 1, j, layer))

            elif op is BacktrackOp.UNPAIRED_RIGHT:
                frame_stack.append(('W', i, j - 1, layer))

            elif op is BacktrackOp.PAIR:
                # W chose to use V[i,j]
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = layer

                frame_stack.append(('V', i, j, layer))

            elif op is BacktrackOp.BIFURCATION:
                k = back_ptr.split_k
                if k is not None:
                    frame_stack.append(('W', i, k, layer))
                    frame_stack.append(('W', k + 1, j, layer))

            elif op is BacktrackOp.PSEUDOKNOT_H:
                if not allow_pk:
                    raise AssertionError(
                        f"traceback_nested encountered a pseudoknot at W[{i},{j}]. "
                        f"Use traceback_with_pk or disable PK during DP."
                    )
                # Two crossing stems + four W segments
                if back_ptr.inner is None or back_ptr.inner_2 is None:
                    # Defensive: malformed backpointer
                    continue
                p1, q1 = back_ptr.inner  # e.g. (i, c)
                p2, q2 = back_ptr.inner_2  # e.g. (b, j)
                segs = tuple(back_ptr.segs or ())

                # Stem 1 on current layer
                pairs.add(Pair(p1, q1))
                pair_layer[(p1, q1)] = layer
                frame_stack.append(('V', p1, q1, layer))

                # Stem 2 on next layer
                pairs.add(Pair(p2, q2))
                pair_layer[(p2, q2)] = layer + 1
                frame_stack.append(('V', p2, q2, layer + 1))

                # Non-crossing regions on current layer
                for lo, hi in segs:
                    if lo <= hi:
                        frame_stack.append(('W', lo, hi, layer))

            # NONE/other => nothing to follow

        elif which == 'V':
            back_ptr: BackPointer = v_back_ptr.get(i, j)
            op = back_ptr.operation

            if op is BacktrackOp.HAIRPIN:
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = pair_layer.get((i, j), layer)

            elif op is BacktrackOp.STACK and back_ptr.inner is not None:
                # Base pair (i,j) stacked on (i+1, j-1)
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = pair_layer.get((i, j), layer)

                inn_i, inn_j = back_ptr.inner
                frame_stack.append(('V', inn_i, inn_j, layer))

            elif op is BacktrackOp.INTERNAL and back_ptr.inner is not None:
                # Internal/bulge with inner (k,l)
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = pair_layer.get((i, j), layer)
                k, l = back_ptr.inner
                frame_stack.append(('V', k, l, layer))

            elif op is BacktrackOp.MULTI_ATTACH and back_ptr.inner is not None:
                # Close a multiloop via a + WM[i+1, j-1]
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = pair_layer.get((i, j), layer)

                inn_i, inn_j = back_ptr.inner  # Typically (i+1, j-1)
                frame_stack.append(('WM', inn_i, inn_j, layer))

            else:
                # Defensive: record (i,j) if V was chosen but op is NONE/unknown
                pairs.add(Pair(i, j))
                if allow_pk:
                    pair_layer[(i, j)] = pair_layer.get((i, j), layer)

        elif which == 'WM':
            back_ptr: BackPointer = wm_back_ptr.get(i, j)
            op = back_ptr.operation

            if op is BacktrackOp.UNPAIRED_LEFT:
                frame_stack.append(('WM', i + 1, j, layer))

            elif op is BacktrackOp.UNPAIRED_RIGHT:
                frame_stack.append(('WM', i, j - 1, layer))

            elif op is BacktrackOp.MULTI_ATTACH and back_ptr.inner is not None:
                # Attach a helix at (p,q) then continue with WM[q+1, j]
                p, q = back_ptr.inner
                pairs.add(Pair(p, q))
                if allow_pk:
                    pair_layer[(p, q)] = pair_layer.get((p, q), layer)

                # Follow the helix content via V
                frame_stack.append(('V', p, q, layer))
                # and the remainder (if any) of the multiloop contents
                if q + 1 <= j:
                    frame_stack.append(('WM', q + 1, j, layer))

            # NONE => base case or nothing more

    # Sort pairs for deterministic output
    ordered_pairs = sorted(pairs, key=lambda base_pairs: (base_pairs.base_i, base_pairs.base_j))

    if not allow_pk:
        dot_brac = pairs_to_dotbracket(seq_len, ordered_pairs)
    else:
        dot_brac = _pairs_to_multilayer_dotbracket(seq_len, ordered_pairs, pair_layer)

    return TraceResult(pairs=ordered_pairs, dot_bracket=dot_brac)


def _pairs_to_multilayer_dotbracket(
    seq_len: int,
    pairs: List[Pair],
    pair_layer: Dict[Tuple[int, int], int],
) -> str:
    """Render layered dot-bracket using BRACKETS by layer; unpaired are '.'."""
    chars = ['.'] * seq_len
    for pair in pairs:
        i, j = pair.base_i, pair.base_j
        layer = pair_layer.get((i, j), 0)
        brac_open, brac_close = BRACKETS[layer % len(BRACKETS)]
        chars[i] = brac_open
        chars[j] = brac_close

    return ''.join(chars)

def dotbracket_to_pairs(db: str) -> Set[Tuple[int, int]]:
    """
    Convert dot-bracket to a set of 0-based base-pair tuples (i, j) with i<j.
    Supports only '(' and ')'.
    """
    stack: List[int] = []
    pairs: Set[Tuple[int, int]] = set()
    for idx, ch in enumerate(db):
        if ch == '(':
            stack.append(idx)
        elif ch == ')':
            if not stack:
                # Unbalanced, ignore to keep test robust
                continue
            i = stack.pop()
            pairs.add((i, idx))
    return pairs
