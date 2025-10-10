from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any
try:
    # Use the standard library StrEnum if available (Python 3.11+).
    from enum import StrEnum
except Exception:
    # Provide a fallback for older Python versions.
    class StrEnum(str, Enum): pass

Interval = Tuple[int, int]


class _AutoName(StrEnum):
    """
    An Enum helper that automatically uses the member name as its string value.
    """
    def _generate_next_value_(name, start, count, last_values):
        """
        This method is called by the Enum machinery to get the value for 'auto()'.
        Simply returns the member's name as a string.

        Parameters
        ----------
        start : Any
        count : int
        last_values : Any
        """
        return name


class EddyRivasBacktrackOp(_AutoName):
    """
    Defines all possible backtrack operations for the Eddy & Rivas algorithm.

    This enumeration provides a comprehensive, serializable list of every
    dynamic programming recursion rule used in the folding process. Each member
    represents a specific transition from a larger problem to one or more
    smaller subproblems. During the DP fill, the chosen operation is stored
    in a backpointer, allowing for the reconstruction of the optimal RNA
    secondary structure via a traceback procedure.

    The naming convention is `RE_{MATRIX}_{OPERATION}` where `RE` stands for
    Rivas-Eddy, `{MATRIX}` is the DP matrix being calculated (e.g., WX, VHX),
    and `{OPERATION}` describes the specific recursion rule.
    """
    # ----------------------------------------------------------------------
    ## WX / VX: Composition, Overlap, and Final Selection
    # These operations represent the O(N^6) step where gapped fragments are
    # combined to form pseudoknots, or the final choice is made.
    # ----------------------------------------------------------------------
    RE_PK_COMPOSE_WX = auto()              # WX composition: Forms a pseudoknot from two WHX subproblems.
    RE_PK_COMPOSE_VX = auto()              # VX composition: Forms a pseudoknot inside a closing pair from two ZHX subproblems.
    RE_PK_COMPOSE_WX_YHX = auto()          # WX composition: Forms a pseudoknot from two YHX subproblems.
    RE_PK_COMPOSE_WX_YHX_WHX = auto()      # WX composition: Forms a pseudoknot from a YHX (left) and WHX (right) subproblem.
    RE_PK_COMPOSE_WX_WHX_YHX = auto()      # WX composition: Forms a pseudoknot from a WHX (left) and YHX (right) subproblem.
    RE_PK_COMPOSE_WX_YHX_OVERLAP = auto()  # WX composition: Forms an overlapping pseudoknot from two YHX subproblems sharing a hole.
    RE_WX_SELECT_UNCHARGED = auto()        # WX finalization: The optimal structure was nested (uncharged), not pseudoknotted.
    RE_VX_SELECT_UNCHARGED = auto()        # VX finalization: The optimal structure within a pair was nested (uncharged).
    RE_PK_COMPOSE_WX_DRIFT = auto()        # WX composition: An experimental variant allowing hole positions to shift.
    RE_PK_COMPOSE_VX_DRIFT = auto()        # VX composition: An experimental variant allowing hole positions to shift.

    # ----------------------------------------------------------------------
    # IS2: Irreducible Surface of Order 2
    # These operations represent forming a loop closed by two base pairs,
    # one of which is part of a gapped subproblem.
    # ----------------------------------------------------------------------
    RE_YHX_IS2_INNER_WHX = auto()  # YHX calculation: Forms an IS2 loop around an inner WHX subproblem.
    RE_WHX_IS2_INNER_YHX = auto()  # WHX calculation: Forms an IS2 loop around an inner YHX subproblem.
    RE_VHX_IS2_INNER_ZHX = auto()  # VHX calculation: Forms an IS2 loop around an inner ZHX subproblem.
    RE_ZHX_IS2_INNER_VHX = auto()  # ZHX calculation: Forms an IS2 loop around an inner VHX subproblem.

    # ----------------------------------------------------------------------
    # WHX: The most general gap matrix (undetermined pairs at all ends).
    # Operations involve adding unpaired bases or splitting the problem.
    # ----------------------------------------------------------------------
    RE_WHX_SHRINK_LEFT = auto()         # WHX(i,j:k,l) -> Adds an unpaired base at k, recursing on WHX(i,j:k+1,l).
    RE_WHX_SHRINK_RIGHT = auto()        # WHX(i,j:k,l) -> Adds an unpaired base at l, recursing on WHX(i,j:k,l-1).
    RE_WHX_TRIM_LEFT = auto()           # WHX(i,j:k,l) -> Adds an unpaired base at i, recursing on WHX(i+1,j:k,l).
    RE_WHX_TRIM_RIGHT = auto()          # WHX(i,j:k,l) -> Adds an unpaired base at j, recursing on WHX(i,j-1:k,l).
    RE_WHX_COLLAPSE = auto()            # WHX(i,j:k,l) -> The hole collapses, transitioning to a nested WXU(i,j) structure.
    RE_WHX_SS_BOTH = auto()             # WHX(i,j:k,l) -> Adds unpaired bases at i and j, recursing on WHX(i+1,j-1:k,l).
    RE_WHX_SPLIT_LEFT_WHX_WX = auto()   # WHX bifurcation: Splits into a gapped WHX(i,r:k,l) and a nested WX(r+1,j).
    RE_WHX_SPLIT_RIGHT_WX_WHX = auto()  # WHX bifurcation: Splits into a nested WX(i,s) and a gapped WHX(s+1,j:k,l).
    RE_WHX_OVERLAP_SPLIT = auto()       # WHX overlap: Joins two WHX subproblems that share the same hole (k,l).

    # ----------------------------------------------------------------------
    # VHX: Outer span (i,j) and inner hole (k,l) are both paired.
    # Recursions build out the structure between the two helices.
    # ----------------------------------------------------------------------
    RE_VHX_DANGLE_L = auto()            # VHX(i,j:k,l) -> Adds a 5' dangle to the (k,l) pair, from VHX(i,j:k+1,l).
    RE_VHX_DANGLE_R = auto()            # VHX(i,j:k,l) -> Adds a 3' dangle to the (k,l) pair, from VHX(i,j:k,l-1).
    RE_VHX_DANGLE_LR = auto()           # VHX(i,j:k,l) -> Adds dangles on both sides of (k,l), from VHX(i,j:k+1,l-1).
    RE_VHX_SS_LEFT = auto()             # VHX(i,j:k,l) -> Adds an unpaired base in the hole, from a ZHX subproblem.
    RE_VHX_SS_RIGHT = auto()            # VHX(i,j:k,l) -> Same as SS_LEFT, used for tie-breaking during DP fill.
    RE_VHX_SPLIT_LEFT_ZHX_WX = auto()   # VHX bifurcation: Splits region into ZHX(i,j:r,l) and a nested WX(r+1,k).
    RE_VHX_SPLIT_RIGHT_ZHX_WX = auto()  # VHX bifurcation: Splits region into ZHX(i,j:k,s) and a nested WX(l,s-1).
    RE_VHX_WRAP_WHX = auto()            # VHX multiloop: Forms a multiloop around a WHX(i+1,j-1:k,l) subproblem.
    RE_VHX_CLOSE_BOTH = auto()          # VHX multiloop: Closes a multiloop around a smaller WHX(i+1,j-1:k-1,l+1).

    # ----------------------------------------------------------------------
    # ZHX: Outer span (i,j) is paired, inner hole (k,l) is undetermined.
    # Recursions define the structure around the hole.
    # ----------------------------------------------------------------------
    RE_ZHX_FROM_VHX = auto()            # ZHX(i,j:k,l) -> Forms a pair at (k,l), transitioning from a VHX(i,j:k,l) subproblem.
    RE_ZHX_DANGLE_LR = auto()           # ZHX(i,j:k,l) -> Forms dangles around a new (k,l) pair, from VHX(i,j:k-1,l+1).
    RE_ZHX_DANGLE_L = auto()            # ZHX(i,j:k,l) -> Forms a 5' dangle on (k,l), from VHX(i,j:k,l+1).
    RE_ZHX_DANGLE_R = auto()            # ZHX(i,j:k,l) -> Forms a 3' dangle on (k,l), from VHX(i,j:k-1,l).
    RE_ZHX_SS_LEFT = auto()             # ZHX(i,j:k,l) -> Adds an unpaired base at k-1, from ZHX(i,j:k-1,l).
    RE_ZHX_SS_RIGHT = auto()            # ZHX(i,j:k,l) -> Adds an unpaired base at l+1, from ZHX(i,j:k,l+1).
    RE_ZHX_SPLIT_LEFT_ZHX_WX = auto()   # ZHX bifurcation: Splits into ZHX(i,j:r,l) and a nested WX(r+1,k).
    RE_ZHX_SPLIT_RIGHT_ZHX_WX = auto()  # ZHX bifurcation: Splits into ZHX(i,j:k,s) and a nested WX(l,s-1).

    # ----------------------------------------------------------------------
    # YHX: Inner hole (k,l) is paired, outer span (i,j) is undetermined.
    # Symmetric to ZHX, defines structure outside the inner helix.
    # ----------------------------------------------------------------------
    RE_YHX_DANGLE_L = auto()            # YHX(i,j:k,l) -> Forms a 5' dangle on (i,j), from VHX(i+1,j:k,l).
    RE_YHX_DANGLE_R = auto()            # YHX(i,j:k,l) -> Forms a 3' dangle on (i,j), from VHX(i,j-1:k,l).
    RE_YHX_DANGLE_LR = auto()           # YHX(i,j:k,l) -> Forms dangles on both sides of (i,j), from VHX(i+1,j-1:k,l).
    RE_YHX_SS_LEFT = auto()             # YHX(i,j:k,l) -> Adds an unpaired base at i, from YHX(i+1,j:k,l).
    RE_YHX_SS_RIGHT = auto()            # YHX(i,j:k,l) -> Adds an unpaired base at j, from YHX(i,j-1:k,l).
    RE_YHX_SS_BOTH = auto()             # YHX(i,j:k,l) -> Adds unpaired bases at i and j, from YHX(i+1,j-1:k,l).
    RE_YHX_SPLIT_LEFT_YHX_WX = auto()   # YHX bifurcation: Splits into YHX(i,r:k,l) and a nested WX(r+1,j).
    RE_YHX_SPLIT_RIGHT_WX_YHX = auto()  # YHX bifurcation: Splits into a nested WX(i,s) and YHX(s+1,j:k,l).
    RE_YHX_WRAP_WHX = auto()            # YHX multiloop: Forms a multiloop around WHX(i,j:k-1,l+1).
    RE_YHX_WRAP_WHX_L = auto()          # YHX multiloop with 5' outer dangle.
    RE_YHX_WRAP_WHX_R = auto()          # YHX multiloop with 3' outer dangle.
    RE_YHX_WRAP_WHX_LR = auto()         # YHX multiloop with dangles on both outer sides.


@dataclass(frozen=True, slots=True)
class EddyRivasBackPointer:
    """
    Stores the information needed to backtrack a single step in the DP matrix.

    This immutable and memory-efficient object represents a single node in the
    backtrack path. It records the specific dynamic programming rule (`op`) used
    to calculate an optimal energy, along with the coordinates of the
    subproblems that were combined. The traceback algorithm follows these
    pointers from the final state `WX(0, N-1)` to reconstruct the full secondary
    structure.

    Attributes
    ----------
    op : EddyRivasBacktrackOp
        The specific DP recursion rule that was chosen as optimal.
    outer : Optional[Interval]
        The `(i, j)` coordinates of the outer span of the current problem.
    hole : Optional[Interval]
        The `(k, l)` coordinates of the inner hole for gap matrix operations.
    hole_left : Optional[Tuple[int, int]]
        The coordinates of the left sub-hole created during composition.
    hole_right : Optional[Tuple[int, int]]
        The coordinates of the right sub-hole created during composition.
    split : Optional[int]
        The split index `r` or `s` used in a bifurcation or composition rule.
    split2 : Optional[int]
        A second split index, used in more complex (rare) rules.
    bridge : Optional[Interval]
        The `(r, s)` coordinates of the inner structure in an IS2 motif rule.
    drift : Optional[int]
        The distance `d` of a hole-drift operation, if used.
    charged : Optional[bool]
        Indicates if the chosen path involved a pseudoknotted ("charged") subproblem.
    note : Optional[str]
        Free-form text for debugging or additional metadata.
    args : Tuple[Any, ...]
        A tuple payload primarily used for simplified verification in unit tests.
    """
    op: EddyRivasBacktrackOp
    outer: Optional[Interval] = None
    hole: Optional[Interval] = None
    hole_left: Optional[Tuple[int, int]] = None
    hole_right: Optional[Tuple[int, int]] = None
    split: Optional[int] = None
    split2: Optional[int] = None
    bridge: Optional[Interval] = None
    drift: Optional[int] = None
    charged: Optional[bool] = None
    note: Optional[str] = None

    # A generic tuple to hold arguments for simplified validation in unit tests.
    args: Tuple[Any, ...] = field(default_factory=tuple)

    # --- Serialization Helpers ---
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the backpointer to a JSON-serializable dictionary.

        This is useful for logging, debugging, or saving the backtrack path to a file.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the backpointer's fields.
        """
        return {
            "op": self.op.value,
            "outer": self.outer,
            "hole": self.hole,
            "split": self.split,
            "bridge": self.bridge,
            "drift": self.drift,
            "charged": self.charged,
            "meta": self.note,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EddyRivasBackPointer":
        """
        Creates an EddyRivasBackPointer instance from a dictionary.

        This is the counterpart to `to_dict`, allowing for deserialization.
        It will fail with a KeyError if the 'op' value is not a valid
        `EddyRivasBacktrackOp` member name.

        Parameters
        ----------
        d : Dict[str, Any]
            A dictionary containing the backpointer's data.

        Returns
        -------
        EddyRivasBackPointer
        """
        op = EddyRivasBacktrackOp(d["op"])
        return EddyRivasBackPointer(
            op=op,
            outer=tuple(d["outer"]) if d.get("outer") else None,
            hole=tuple(d["hole"]) if d.get("hole") else None,
            split=d.get("split"),
            bridge=tuple(d["bridge"]) if d.get("bridge") else None,
            drift=d.get("drift"),
            charged=d.get("charged"),
            note=d.get("note"),
        )

    # --- Factory Methods ---
    # Prove a clean API for creating specific, commonly used backpointers within the main DP loops,
    # reducing boilerplate.
    @classmethod
    def compose_vx(cls, r: int, k: int, l: int) -> "EddyRivasBackPointer":
        """Creates a backpointer for a standard VX composition."""
        return cls(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX,
                   split=r, hole=(k, l), args=(r, k, l))

    @classmethod
    def compose_vx_drift(cls, r: int, k: int, l: int, d: int) -> "EddyRivasBackPointer":
        """Creates a backpointer for a VX composition with hole drift."""
        return cls(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT,
                   split=r, hole=(k, l), drift=d, args=(r, k, l, d))

    @classmethod
    def vx_select_uncharged(cls) -> "EddyRivasBackPointer":
        """Creates a backpointer for when the nested VX path is chosen."""
        return cls(op=EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED, args=())

    @classmethod
    def wx_select_uncharged(cls) -> "EddyRivasBackPointer":
        """Creates a backpointer for when the nested WX path is chosen."""
        return cls(op=EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED, args=())

    @classmethod
    def whx_shrink_left(cls, i: int, j: int, k1: int, l: int) -> "EddyRivasBackPointer":
        """Creates a backpointer for adding an unpaired base to the left of a WHX hole."""
        return cls(op=EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT,
                   outer=(i, j), hole=(k1, l), args=(i, j, k1, l))

    @classmethod
    def whx_split_left_whx_wx(cls, r: int) -> "EddyRivasBackPointer":
        """Creates a backpointer for a WHX bifurcation into WHX + WX."""
        return cls(op=EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX,
                   split=r, args=(r,))