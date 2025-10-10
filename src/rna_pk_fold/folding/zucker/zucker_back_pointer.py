from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Sequence

__all__ = ["ZuckerBacktrackOp", "ZuckerBackPointer"]

Interval = Tuple[int, int]


class ZuckerBacktrackOp(Enum):
    """
    Defines all possible backtrack operations for the Zuker algorithm.

    This enumeration provides a set of unique identifiers for each recursion
    rule used in the Zuker-style dynamic programming algorithm for nested RNA
    secondary structure prediction. Each member represents a specific way an
    optimal energy for a subsequence could have been calculated. Storing one
    of these members in a backpointer allows the traceback process to correctly
    reconstruct the sequence of decisions that led to the final optimal structure.

    NONE            : Not set yet.
    HAIRPIN         : V[i,j] formed a hairpin closed by (i,j).
    STACK           : V[i,j] formed by stacking on V[i+1,j-1].
    INTERNAL        : V[i,j] formed an internal/bulge loop with inner pair (k,l).
    PAIR            : W[i,j] chose the paired state V[i,j].
    BIFURCATION     : W[i,j] split into W[i,k] + W[k+1,j].
    MULTI_ATTACH    : V/W case where a helix attaches into a multiloop context.
    UNPAIRED_LEFT   : W[i,j] best by leaving i unpaired (use W[i+1,j]).
    UNPAIRED_RIGHT  : W[i,j] best by leaving j unpaired (use W[i,j-1]).
    PSEUDOKNOT      : Placeholder for future pseudoknot logic.
    """
    NONE = auto()
    HAIRPIN = auto()
    STACK = auto()
    INTERNAL = auto()
    PAIR = auto()
    BIFURCATION = auto()
    MULTI_ATTACH = auto()
    UNPAIRED_LEFT = auto()
    UNPAIRED_RIGHT = auto()


@dataclass(frozen=True, slots=True)
class ZuckerBackPointer:
    """
    Stores the information needed to backtrack a single step in the Zuker DP matrix.

    This immutable and memory-efficient data structure represents a single
    decision point in the dynamic programming process. It captures which
    recursion rule (`operation`) was chosen as optimal for a given matrix cell,
    along with any coordinates (`split_k`, `inner`) needed to recursively call
    the traceback on the corresponding subproblems.

    Attributes
    ----------
    operation : ZuckerBacktrackOp
        The specific DP recursion rule that was chosen as optimal for this cell.
    split_k : Optional[int]
        The index `k` used in a `BIFURCATION` rule, where the problem on
        `[i,j]` was split into `[i,k]` and `[k+1,j]`.
    inner : Optional[Tuple[int, int]]
        The coordinates `(k,l)` of the inner base pair for an `INTERNAL` loop,
        or `(i+1, j-1)` for a `STACK` operation.
    inner_2 : Optional[Tuple[int, int]]
        Coordinates for a second inner helix, used for more complex motifs like
        H-type pseudoknots (not part of the standard Zuker algorithm).
    segs : Optional[Sequence[Interval]]
        A sequence of sub-intervals to recurse on, typically for multiloops or
        pseudoknot decompositions.
    layer : Optional[int]
        A hint for rendering multilayer dot-bracket strings, where 0 corresponds
        to '()', 1 to '[]', etc.
    note : Optional[str]
        A free-form string for storing debugging information or metadata about
        the specific rule variant used (e.g., "triloop", "tetraloop").
    """
    operation: ZuckerBacktrackOp = ZuckerBacktrackOp.NONE
    split_k: Optional[int] = None
    inner: Optional[Tuple[int, int]] = None
    inner_2: Optional[Tuple[int, int]] = None
    segs: Optional[Sequence[Interval]] = None
    layer: Optional[int] = None
    note: Optional[str] = None
