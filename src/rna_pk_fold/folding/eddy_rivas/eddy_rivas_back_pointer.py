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


class _AutoNameStr(StrEnum):
    """
    An Enum helper that automatically uses the member name as its string value.
    """
    def _generate_next_value_(member_name, start_value, member_index, previous_values):
        """
        Compute the value for an `Enum.auto()` member.

        Returns the enum member's name so that each member's value equals
        its identifier (e.g., `FOO.value == "FOO"`).

        Parameters
        ----------
        member_name : str
            The member name being created.
        start_value : int
            Starting value for the first enumeration (unused here).
        member_index : int
            0-based index of the member within the enumeration (unused here).
        previous_values : list of Any
            Values already assigned in this enumeration (unused here).

        Returns
        -------
        str
            The member name, used as the enum value.
        """
        return member_name


class EddyRivasBacktrackOp(_AutoNameStr):
    """
    Defines all possible backtrack operations for the Eddy & Rivas algorithm.

    This enumeration provides a comprehensive, serializable list of every
    dynamic programming recursion rule used in the folding process. Each member
    represents a specific transition from a larger problem to one or more
    smaller sub-problems. During the DP fill, the chosen operation is stored
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
    RE_PK_COMPOSE_WX = auto()              # WX composition: Forms a pseudoknot from two WHX sub-problems.
    RE_PK_COMPOSE_VX = auto()              # VX composition: Forms a pseudoknot inside a closing pair from two ZHX sub-problems.
    RE_PK_COMPOSE_WX_YHX = auto()          # WX composition: Forms a pseudoknot from two YHX sub-problems.
    RE_PK_COMPOSE_WX_YHX_WHX = auto()      # WX composition: Forms a pseudoknot from a YHX (left) and WHX (right) sub-problem.
    RE_PK_COMPOSE_WX_WHX_YHX = auto()      # WX composition: Forms a pseudoknot from a WHX (left) and YHX (right) sub-problem.
    RE_PK_COMPOSE_WX_YHX_OVERLAP = auto()  # WX composition: Forms an overlapping pseudoknot from two YHX sub-problems sharing a hole.
    RE_WX_SELECT_UNCHARGED = auto()        # WX finalization: The optimal structure was nested (uncharged), not pseudoknotted.
    RE_VX_SELECT_UNCHARGED = auto()        # VX finalization: The optimal structure within a pair was nested (uncharged).
    RE_PK_COMPOSE_WX_DRIFT = auto()        # WX composition: An experimental variant allowing hole positions to shift.
    RE_PK_COMPOSE_VX_DRIFT = auto()        # VX composition: An experimental variant allowing hole positions to shift.

    # ----------------------------------------------------------------------
    # IS2: Irreducible Surface of Order 2
    # These operations represent forming a loop closed by two base pairs,
    # one of which is part of a gapped sub-problem.
    # ----------------------------------------------------------------------
    RE_YHX_IS2_INNER_WHX = auto()  # YHX calculation: Forms an IS2 loop around an inner WHX sub-problem.
    RE_WHX_IS2_INNER_YHX = auto()  # WHX calculation: Forms an IS2 loop around an inner YHX sub-problem.
    RE_VHX_IS2_INNER_ZHX = auto()  # VHX calculation: Forms an IS2 loop around an inner ZHX sub-problem.
    RE_ZHX_IS2_INNER_VHX = auto()  # ZHX calculation: Forms an IS2 loop around an inner VHX sub-problem.

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
    RE_WHX_OVERLAP_SPLIT = auto()       # WHX overlap: Joins two WHX sub-problems that share the same hole (k,l).

    # ----------------------------------------------------------------------
    # VHX: Outer span (i,j) and inner hole (k,l) are both paired.
    # Recursions build out the structure between the two helices.
    # ----------------------------------------------------------------------
    RE_VHX_DANGLE_L = auto()            # VHX(i,j:k,l) -> Adds a 5' dangle to the (k,l) pair, from VHX(i,j:k+1,l).
    RE_VHX_DANGLE_R = auto()            # VHX(i,j:k,l) -> Adds a 3' dangle to the (k,l) pair, from VHX(i,j:k,l-1).
    RE_VHX_DANGLE_LR = auto()           # VHX(i,j:k,l) -> Adds dangles on both sides of (k,l), from VHX(i,j:k+1,l-1).
    RE_VHX_SS_LEFT = auto()             # VHX(i,j:k,l) -> Adds an unpaired base in the hole, from a ZHX sub-problem.
    RE_VHX_SS_RIGHT = auto()            # VHX(i,j:k,l) -> Same as SS_LEFT, used for tie-breaking during DP fill.
    RE_VHX_SPLIT_LEFT_ZHX_WX = auto()   # VHX bifurcation: Splits region into ZHX(i,j:r,l) and a nested WX(r+1,k).
    RE_VHX_SPLIT_RIGHT_ZHX_WX = auto()  # VHX bifurcation: Splits region into ZHX(i,j:k,s) and a nested WX(l,s-1).
    RE_VHX_WRAP_WHX = auto()            # VHX multiloop: Forms a multiloop around a WHX(i+1,j-1:k,l) sub-problem.
    RE_VHX_CLOSE_BOTH = auto()          # VHX multiloop: Closes a multiloop around a smaller WHX(i+1,j-1:k-1,l+1).

    # ----------------------------------------------------------------------
    # ZHX: Outer span (i,j) is paired, inner hole (k,l) is undetermined.
    # Recursions define the structure around the hole.
    # ----------------------------------------------------------------------
    RE_ZHX_FROM_VHX = auto()            # ZHX(i,j:k,l) -> Forms a pair at (k,l), transitioning from a VHX(i,j:k,l) sub-problem.
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
    sub-problems that were combined. The traceback algorithm follows these
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
        Indicates if the chosen path involved a pseudoknotted ("charged") sub-problem.
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

    # ---------------------------------------------------------------------
    # Serialization Helpers
    # ---------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the backpointer to a plain dictionary.

        Produces a JSON-serializable view of the most relevant fields for
        logging, debugging, or persistence.

        Returns
        -------
        Dict[str, Any]
            A dictionary with keys `op`, `outer`, `hole`, `split`,
            `bridge`, `drift`, `charged`, and `meta`.
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
        Deserialize a backpointer from a dictionary.

        This is the inverse of `to_dict`. The `op` string must match a
        member of `EddyRivasBacktrackOp`.

        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary produced by `to_dict` (or equivalent), containing
            at least the key `"op"` and optionally `"outer"`, `"hole"`,
            `"split"`, `"bridge"`, `"drift"`, `"charged"`, and `"note"`.

        Returns
        -------
        EddyRivasBackPointer
            A new backpointer instance built from the provided mapping.

        Raises
        ------
        KeyError
            If `"op"` is missing or not a valid `EddyRivasBacktrackOp`.
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

    # ---------------------------------------------------------------------
    # Factory methods
    # ---------------------------------------------------------------------
    @classmethod
    def create_vx_composition_backpointer(
        cls,
        split_index:int,
        hole_left_index: int,
        hole_right_index: int
    ) -> "EddyRivasBackPointer":
        """
        Construct a backpointer for a standard VX composition.

        Represents the O(N^6) composition that forms a pseudoknot enclosed by
        a closing pair `(i, j)` by splitting at index `r` with inner hole
        endpoints `(k, l)`. (The surrounding `(i, j)` span is carried in
        the consumer's context.)

        Parameters
        ----------
        split_index : int
            Split index within `[k, l-1]`.
        hole_left_index : int
            5' endpoint of the inner hole.
        hole_right_index : int
            3' endpoint of the inner hole.

        Returns
        -------
        EddyRivasBackPointer
            Factory backpointer with operation
            `EddyRivasBacktrackOp.RE_PK_COMPOSE_VX`.
        """
        return cls(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX, split=split_index,
                   hole=(hole_left_index, hole_right_index), args=(split_index, hole_left_index, hole_right_index))

    @classmethod
    def create_vx_composition_with_drift_backpointer(
        cls,
        split_index: int,
        hole_left_index: int,
        hole_right_index: int,
        drift_distance: int
    ) -> "EddyRivasBackPointer":
        """
        Construct a backpointer for a VX composition with hole drift.

        This experimental variant allows the inner hole to shift by `d`
        nucleotides during composition.

        Parameters
        ----------
        split_index : int
            Split index within `[k, l-1]`.
        hole_left_index : int
            5' endpoint of the inner hole (pre-drift).
        hole_right_index : int
            3' endpoint of the inner hole (pre-drift).
        drift_distance : int
            Drift distance (positive values shift toward 3').

        Returns
        -------
        EddyRivasBackPointer
            Factory backpointer with operation
            `EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT`.
        """
        return cls(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT, split=split_index,
                   hole=(hole_left_index, hole_right_index), drift=drift_distance,
                   args=(split_index, hole_left_index, hole_right_index, drift_distance))

    @classmethod
    def select_uncharged_vx_backpointer(cls) -> "EddyRivasBackPointer":
        """
        Construct a backpointer for selecting the nested VX path.

        Used when the best energy inside a closing pair is achieved without
        introducing a pseudoknot.

        Returns
        -------
        EddyRivasBackPointer
            Factory backpointer with operation
            `EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED`.
        """
        return cls(op=EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED, args=())

    @classmethod
    def select_uncharged_wx_backpointer(cls) -> "EddyRivasBackPointer":
        """
        Construct a backpointer for selecting the nested WX path.

        Used when the best energy for a span `(i, j)` is achieved without
        introducing a pseudoknot.

        Returns
        -------
        EddyRivasBackPointer
            Factory backpointer with operation
            `EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED`.
        """
        return cls(op=EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED, args=())

    @classmethod
    def create_whx_shrink_left_backpointer(
        cls,
        outer_start_index: int,
        outer_end_index: int,
        hole_start_index: int,
        hole_end_index: int
    ) -> "EddyRivasBackPointer":
        """
        Construct a backpointer for shrinking the WHX hole from the left.

        Models the recursion `WHX(i, j : k1, l) -> WHX(i, j : k1+1, l)` by
        adding a single-stranded nucleotide at position `k1`.

        Parameters
        ----------
        outer_start_index : int
            5' index of the outer span.
        outer_end_index : int
            3' index of the outer span.
        hole_start_index : int
            Current 5' endpoint of the hole being advanced.
        hole_end_index : int
            3' endpoint of the hole (unchanged in this step).

        Returns
        -------
        EddyRivasBackPointer
            Factory backpointer with operation
            `EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT`.
        """
        return cls(op=EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT, outer=(outer_start_index, outer_end_index),
                   hole=(hole_start_index, hole_end_index),
                   args=(outer_start_index, outer_end_index, hole_start_index, hole_end_index))

    @classmethod
    def create_whx_left_split_whx_plus_wx_backpointer(cls, split_index: int) -> "EddyRivasBackPointer":
        """
        Construct a backpointer for the WHX left split into `WHX + WX`.

        Represents the bifurcation
        `WHX(i, j : k, l) -> WHX(i, r : k, l) + WX(r+1, j)`.

        Parameters
        ----------
        split_index : int
            Split index in the outer span satisfying `i ≤ r < j`.

        Returns
        -------
        EddyRivasBackPointer
            Factory backpointer with operation
            `EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX`.
        """
        return cls(op=EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX,
                   split=split_index, args=(split_index,))