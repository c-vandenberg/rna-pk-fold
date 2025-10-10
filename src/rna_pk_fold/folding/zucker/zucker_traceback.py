from __future__ import annotations
import math
from typing import List, Set, Tuple

from rna_pk_fold.folding.common_traceback import TraceResult, pairs_to_dotbracket
from rna_pk_fold.folding.zucker import ZuckerFoldState, ZuckerBacktrackOp, ZuckerBackPointer
from rna_pk_fold.structures import Pair


def traceback_nested(seq: str, state: ZuckerFoldState) -> TraceResult:
    """
    Reconstructs the optimal nested structure for an entire RNA sequence.

    This function serves as the primary entry point for a standard Zuker-style
    traceback. It initiates the core traceback process starting from the
    top-level matrix cell `W[0, N-1]`.

    Parameters
    ----------
    seq : str
        The RNA sequence that was folded.
    state : ZuckerFoldState
        The state object containing the filled dynamic programming matrices
        and backpointers from the Zuker algorithm.

    Returns
    -------
    TraceResult
        A data object containing the final list of base pairs and the
        corresponding dot-bracket string representation.
    """
    return _traceback_core(seq, state)


def traceback_nested_interval(seq: str, state: ZuckerFoldState, i: int, j: int) -> TraceResult:
    """
    Reconstructs the optimal nested structure for a specific subsequence `[i, j]`.

    This function is a specialized entry point used to trace only a part of
    the full DP matrix. It's primarily used when a larger, pseudoknot-aware
    algorithm needs to resolve a purely nested substructure.

    Parameters
    ----------
    seq : str
        The full RNA sequence.
    state : ZuckerFoldState
        The state object containing the filled DP matrices.
    i : int
        The 5' start index of the interval to trace.
    j : int
        The 3' end index of the interval to trace.

    Returns
    -------
    TraceResult
        A data object containing the base pairs found within the interval
        and a full-length dot-bracket string.
    """
    return _traceback_core_with_seed(seq, state, seed_frames=[('W', i, j)])


def _traceback_core(seq: str, state: ZuckerFoldState) -> TraceResult:
    """
    Core stack-based traceback state machine for Zuker algorithm.

    This function reconstructs the optimal secondary structure by following the
    chain of backpointers. It uses a stack to manage subproblems, popping a frame,
    interpreting its backpointer, and pushing the corresponding smaller subproblem(s)
    back onto the stack. Base pairs are collected when terminal operations (like HAIRPIN)
    or pair-forming operations (like PAIR) are encountered.

    Parameters
    ----------
    seq : str
        The RNA sequence.
    state : ZuckerFoldState
        The state object containing the filled DP matrices and backpointers.

    Returns
    -------
    TraceResult
        The final collection of pairs and the rendered dot-bracket string.
    """
    seq_len = len(seq)
    if seq_len == 0:
        return TraceResult(pairs=[], dot_bracket="")

    # --- State Initialization ---
    # `pairs` will store the final set of (i, j) base pairs.
    pairs: Set[Pair] = set()

    # The `stack` holds "frames" representing subproblems to be solved.
    stack: List[Tuple[str, int, int]] = [('W', 0, seq_len - 1)]

    # Get local references to the backpointer matrices for convenience.
    w_bp = state.w_back_ptr
    v_bp = state.v_back_ptr
    wm_bp = state.wm_back_ptr

    # --- Main Traceback Loop ---
    # Process frames from the stack until all subproblems are resolved.
    while stack:
        # Pop the next subproblem (frame) to work on.
        which, i, j = stack.pop()

        # --- Sanity Checks ---
        # Skip invalid intervals or trivial single-base 'W' frames.
        if i > j or i < 0 or j >= seq_len:
            continue

        if which == 'W' and i == j:
            continue

        # --- 'W' Matrix Traceback ---
        # A 'W' frame represents the general problem for an interval [i, j].
        if which == 'W':
            # Retrieve the backpointer for this cell.
            bp: ZuckerBackPointer = w_bp.get(i, j)
            op = bp.operation

            # Rule: Base 'i' was left unpaired. Recurse on the smaller interval W[i+1, j].
            if op is ZuckerBacktrackOp.UNPAIRED_LEFT:
                stack.append(('W', i + 1, j))

            # Rule: Base 'j' was left unpaired. Recurse on the smaller interval W[i, j-1].
            elif op is ZuckerBacktrackOp.UNPAIRED_RIGHT:
                stack.append(('W', i, j - 1))

            # Rule: Bases 'i' and 'j' formed a pair. Add the pair and continue traceback from V[i, j].
            elif op is ZuckerBacktrackOp.PAIR:
                pairs.add(Pair(i, j))
                stack.append(('V', i, j))

            # Rule: The structure was a bifurcation. Recurse on the two independent subproblems.
            elif op is ZuckerBacktrackOp.BIFURCATION:
                k = bp.split_k
                if k is not None:
                    stack.append(('W', i, k))
                    stack.append(('W', k + 1, j))

            # If the operation is NONE or unknown, this path terminates.

        # A 'V' frame represents a subproblem enclosed by a pair (i, j).
        elif which == 'V':
            bp: ZuckerBackPointer = v_bp.get(i, j)
            op = bp.operation

            # Rule: (i,j) closed a hairpin. This is a terminal rule, so we just record the pair.
            if op is ZuckerBacktrackOp.HAIRPIN:
                pairs.add(Pair(i, j))

            # Rule: (i,j) stacked on an inner pair. Record the (i,j) pair and recurse on the inner V subproblem.
            elif op is ZuckerBacktrackOp.STACK and bp.inner is not None:
                pairs.add(Pair(i, j))
                k, l = bp.inner
                stack.append(('V', k, l))

            # Rule: (i,j) closed an internal loop. Record the (i,j) pair and recurse on the inner V subproblem.
            elif op is ZuckerBacktrackOp.INTERNAL and bp.inner is not None:
                pairs.add(Pair(i, j))
                k, l = bp.inner
                stack.append(('V', k, l))

            # Rule: (i,j) closed a multiloop. Record the (i,j) pair and recurse on the interior WM subproblem.
            elif op is ZuckerBacktrackOp.MULTI_ATTACH and bp.inner is not None:
                pairs.add(Pair(i, j))
                p, q = bp.inner

                # Follow the helix content
                stack.append(('WM', p, q))

                # Remainder of the multiloop contents
                if q + 1 <= j:
                    stack.append(('WM', q + 1, j))

            # Defensive fallback: if V was chosen but the operation is unknown, still record the pair.
            else:
                pairs.add(Pair(i, j))

        # --- 'WM' Matrix Traceback ---
        # A 'WM' frame represents a subproblem inside a multiloop.
        elif which == 'WM':
            bp: ZuckerBackPointer = wm_bp.get(i, j)
            op = bp.operation

            # Rule: Base 'i' was unpaired. Recurse on the smaller WM subproblem.
            if op is ZuckerBacktrackOp.UNPAIRED_LEFT:
                stack.append(('WM', i + 1, j))

            # Rule: Base 'j' was unpaired. Recurse on the smaller WM subproblem.
            elif op is ZuckerBacktrackOp.UNPAIRED_RIGHT:
                stack.append(('WM', i, j - 1))

            # Rule: A helix (p,q) branched off.
            elif op is ZuckerBacktrackOp.MULTI_ATTACH and bp.inner is not None:
                p, q = bp.inner
                # Ensure the branch is a valid helix before adding it.
                if math.isfinite(state.v_matrix.get(p, q)):
                    pairs.add(Pair(p, q))
                    # Recurse on the helix interior (V) and the rest of the multiloop (WM).
                    stack.append(('V', p, q))
                    if q + 1 <= j:
                        stack.append(('WM', q + 1, j))
                else:
                    # If the pair was invalid, just continue with the rest of the multiloop.
                    if q + 1 <= j:
                        stack.append(('WM', q + 1, j))

            # If the operation is NONE, it's a base case for WM (single base), so terminate this path.

    # --- Finalization ---
    # Sort the collected pairs by their 5' index for a canonical representation and convert
    # the final set of pairs into a standard dot-bracket string.
    ordered = sorted(pairs, key=lambda pr: (pr.base_i, pr.base_j))

    return TraceResult(pairs=ordered, dot_bracket=pairs_to_dotbracket(seq_len, ordered))


def _traceback_core_with_seed(
    seq: str,
    state: ZuckerFoldState,
    *,
    seed_frames: List[Tuple[str, int, int]],
) -> TraceResult:
    """
    Same as _traceback_core, but start from the provided (matrix, i, j) frames.
    """
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
                stack.append(('WM', p, q))

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
                if math.isfinite(state.v_matrix.get(p, q)):
                    pairs.add(Pair(p, q))
                    stack.append(('V', p, q))
                    if q + 1 <= j:
                        stack.append(('WM', q + 1, j))
                else:
                    # Skip invalid pair, continue with rest of multiloop
                    if q + 1 <= j:
                        stack.append(('WM', q + 1, j))

    ordered = sorted(pairs, key=lambda pr: (pr.base_i, pr.base_j))

    return TraceResult(pairs=ordered, dot_bracket=pairs_to_dotbracket(n, ordered))

