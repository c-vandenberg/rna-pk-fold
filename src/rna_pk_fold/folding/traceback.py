from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple

from rna_pk_fold.folding import FoldState, BacktrackOp, BackPointer
from rna_pk_fold.structures import Pair


@dataclass(frozen=True, slots=True)
class TraceResult:
    """Result of traceback for nested structures."""
    pairs: List[Pair]        # List of Pair(base_i, base_j) with base_i < base_j
    dot_bracket: str         # E.g., '..((...))..'


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


def traceback_nested(seq: str, state: FoldState) -> TraceResult:
    """
    Nested-only traceback:
      - start at W[0, N-1]
      - follow W/V/WM back-pointers
      - collect all (i, j) that represent base pairs
      - emit simple dot-bracket with '(' and ')' (no pseudoknots)

    This function assumes DP tables have been filled by your
    SecondaryStructureFoldingEngine (W, V, WM and their back-pointers).
    """
    seq_len = len(seq)
    if seq_len == 0:
        return TraceResult(pairs=[], dot_bracket="")

    pairs: Set[Pair] = set()

    # Worklist of frames to process (matrix_tag, i, j): Each frame says which
    # matrix to consult and which (i, j) coordinates to follow.
    frame_stack: List[Tuple[str, int, int]] = [('W', 0, seq_len - 1)]

    w_back_ptr = state.w_back_ptr
    v_back_ptr = state.v_back_ptr
    wm_back_ptr = state.wm_back_ptr

    while frame_stack:
        which, i, j = frame_stack.pop()

        # Out-of-range or empty intervals are benign
        if i >= j or i < 0 or j >= seq_len:
            # Note: i==j is valid in WM (base case NONE), and in W we treat it as 0 with NONE.
            if which == 'W' and i == j:
                pass
            continue

        if which == 'W':
            back_ptr: BackPointer = w_back_ptr.get(i, j)
            op = back_ptr.operation

            if op is BacktrackOp.UNPAIRED_LEFT:
                frame_stack.append(('W', i + 1, j))

            elif op is BacktrackOp.UNPAIRED_RIGHT:
                frame_stack.append(('W', i, j - 1))

            elif op is BacktrackOp.PAIR:
                # W chose to use V[i,j]
                pairs.add(Pair(i, j))
                frame_stack.append(('V', i, j))

            elif op is BacktrackOp.BIFURCATION:
                k = back_ptr.split_k
                if k is not None:
                    frame_stack.append(('W', i, k))
                    frame_stack.append(('W', k + 1, j))

            # NONE/other => nothing to follow

        elif which == 'V':
            back_ptr: BackPointer = v_back_ptr.get(i, j)
            op = back_ptr.operation

            if op is BacktrackOp.HAIRPIN:
                pairs.add(Pair(i, j))
                # Done with this branch

            elif op is BacktrackOp.STACK and back_ptr.inner is not None:
                # Base pair (i,j) stacked on (i+1, j-1)
                pairs.add(Pair(i, j))
                inn_i, inn_j = back_ptr.inner
                frame_stack.append(('V', inn_i, inn_j))

            elif op is BacktrackOp.INTERNAL and back_ptr.inner is not None:
                # Internal/bulge with inner (k,l)
                pairs.add(Pair(i, j))
                k, l = back_ptr.inner
                frame_stack.append(('V', k, l))

            elif op is BacktrackOp.MULTI_ATTACH and back_ptr.inner is not None:
                # Close a multiloop via a + WM[i+1, j-1]
                pairs.add(Pair(i, j))
                inn_i, inn_j = back_ptr.inner  # Typically (i+1, j-1)
                frame_stack.append(('WM', inn_i, inn_j))

            else:
                # If V was selected but op is NONE/unknown, at minimum record (i,j)
                pairs.add(Pair(i, j))

        elif which == 'WM':
            back_ptr: BackPointer = wm_back_ptr.get(i, j)
            op = back_ptr.operation

            if op is BacktrackOp.UNPAIRED_LEFT:
                frame_stack.append(('WM', i + 1, j))

            elif op is BacktrackOp.UNPAIRED_RIGHT:
                frame_stack.append(('WM', i, j - 1))

            elif op is BacktrackOp.MULTI_ATTACH and back_ptr.inner is not None:
                # Attach a helix at (p,q) then continue with WM[q+1, j]
                p, q = back_ptr.inner
                pairs.add(Pair(p, q))
                # Follow the helix content via V
                frame_stack.append(('V', p, q))
                # and the remainder (if any) of the multiloop contents
                if q + 1 <= j:
                    frame_stack.append(('WM', q + 1, j))

            # NONE => base case or nothing more

    # Sort pairs for deterministic output
    ordered_pairs = sorted(pairs, key=lambda base_pairs: (base_pairs.base_i, base_pairs.base_j))
    dot_brac = pairs_to_dotbracket(seq_len, ordered_pairs)

    return TraceResult(pairs=ordered_pairs, dot_bracket=dot_brac)
