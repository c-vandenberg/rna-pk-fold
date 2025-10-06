from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Sequence

__all__ = ["ZuckerBacktrackOp", "ZuckerBackPointer"]

Interval = Tuple[int, int]


class ZuckerBacktrackOp(Enum):
    """
    Operation chosen at a matrix cell. This is to be used during traceback to
    track what recurrence case produced the optimal value at a given cell.

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
    PSEUDOKNOT_H = auto()


@dataclass(frozen=True, slots=True)
class ZuckerBackPointer:
    """
    Back-pointer describing how a matrix cell's value was derived.

    Parameters
    ----------
    operation : ZuckerBacktrackOp
                The recurrence operation selected for this cell.
    split_k   : Optional[int]
                For bifurcations W[i,j] -> W[i,k] + W[k+1,j], record k.
    inner     : Optional[Tuple[int, int]]
                For internal/stack cases, the inner paired indices, e.g., (i+1, j-1)
                for STACK, or (k, l) for INTERNAL loops.
    inner_2   : the second crossing stem (k,l) (in addition to `inner`)
    segs      : a tuple/list of up to four W-subintervals to recurse on,
                e.g. ((i+1,a-1), (a,b-1), (c+1,d-1), (d,j-1))
    layer     : optional bracket layer hint for rendering (0=(), 1=[], ...)
    note      : Optional[str]
                Free-form metadata (e.g., “tri-tetra hairpin”, “multi enter”).
    """
    operation: ZuckerBacktrackOp = ZuckerBacktrackOp.NONE

    # Generic fields
    split_k: Optional[int] = None
    inner: Optional[Tuple[int, int]] = None

    # Pseudoknot (H-type) extras
    inner_2: Optional[Tuple[int, int]] = None
    segs: Optional[Sequence[Interval]] = None
    layer: Optional[int] = None

    # Free-form metadata (e.g., “tri-tetra hairpin”, “attach-helix”, “H-type”)
    note: Optional[str] = None
