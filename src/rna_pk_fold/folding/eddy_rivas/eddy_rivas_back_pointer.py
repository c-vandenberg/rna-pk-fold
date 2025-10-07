from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Sequence, Any
try:
    from enum import StrEnum
except Exception:
    class StrEnum(str, Enum): pass

Interval = Tuple[int, int]


class _AutoName(StrEnum):
    """auto() -> member name as string (stable, readable, serializable)."""
    def _generate_next_value_(name, start, count, last_values):
        return name


class EddyRivasBacktrackOp(_AutoName):
    # ----------------------------------------------------------------------
    # WX / VX: Composition, Overlap, Selection, Drift
    # ----------------------------------------------------------------------
    RE_PK_COMPOSE_WX = auto()              # WX: Gw + [left WHX] + [right WHX] (+ caps/mix variants)
    RE_PK_COMPOSE_VX = auto()              # VX: Gw + [left ZHX] + [right ZHX] (+ coax/mismatch/bonus)
    RE_PK_COMPOSE_WX_YHX = auto()          # WX: Gw + YHX(i,r:k,l) + YHX(k+1,j:l-1,r+1)
    RE_PK_COMPOSE_WX_YHX_WHX = auto()      # WX: Gw + YHX(i,r:k,l) + WHX(k+1,j:l-1,r+1)
    RE_PK_COMPOSE_WX_WHX_YHX = auto()      # WX: Gw + WHX(i,r:k,l) + YHX(k+1,j:l-1,r+1)
    RE_PK_COMPOSE_WX_YHX_OVERLAP = auto()  # WX: Gwh_wx + YHX(i,r:k,l) + YHX(r+1,j:k,l) (same-hole overlap)
    RE_WX_SELECT_UNCHARGED = auto()        # WX Publish: min(WX_un, WX_ch) → chose uncharged
    RE_VX_SELECT_UNCHARGED = auto()        # VX Publish: min(VX_un, VX_ch) → chose uncharged
    RE_PK_COMPOSE_WX_DRIFT = auto()        # WX: join-drift variant (charged path improves via drift)
    RE_PK_COMPOSE_VX_DRIFT = auto()        # VX: join-drift variant (charged path improves via drift)

    # ----------------------------------------------------------------------
    # IS2 bridge hooks (outer-context symmetric terms)
    # ----------------------------------------------------------------------
    RE_YHX_IS2_INNER_WHX = auto()  # YHX: IS2_outer_yhx(i,j,r,s) + WHX(r,s:k,l)
    RE_WHX_IS2_INNER_YHX = auto()  # WHX: IS2_outer_yhx(i,j,r,s) + YHX(r,s:k,l)
    RE_VHX_IS2_INNER_ZHX = auto()  # VHX: IS2_outer(i,j,r,s) + ZHX(r,s:k,l)
    RE_ZHX_IS2_INNER_VHX = auto()  # ZHX: IS2_outer(i,j,r,s) + VHX(r,s:k,l)

    # ----------------------------------------------------------------------
    # WHX: Hole DP (outer [i,j], hole [k,l])
    # ----------------------------------------------------------------------
    RE_WHX_SHRINK_LEFT = auto()             # WHX(i,j:k,l) ← WHX(i,j:k+1,l) + q_ss
    RE_WHX_SHRINK_RIGHT = auto()            # WHX(i,j:k,l) ← WHX(i,j:k,l-1) + q_ss
    RE_WHX_TRIM_LEFT = auto()               # WHX(i,j:k,l) ← WHX(i+1,j:k,l) + q_ss
    RE_WHX_TRIM_RIGHT = auto()              # WHX(i,j:k,l) ← WHX(i,j-1:k,l) + q_ss
    RE_WHX_COLLAPSE = auto()                # WHX(i,j:k,l) ← WXU(i,j)      (Collapse identity)
    RE_WHX_SS_BOTH = auto()             # WHX(i,j:k,l) ← WHX(i+1,j-1:k,l) + 2*q_ss
    RE_WHX_SPLIT_LEFT_WHX_WX = auto()   # WHX(i,j:k,l) ← WHX(i,r:k,l) + WX*(r+1,j)
    RE_WHX_SPLIT_RIGHT_WX_WHX = auto()  # WHX(i,j:k,l) ← WX*(i,s) + WHX(s+1,j:k,l)
    RE_WHX_OVERLAP_SPLIT = auto()       # WHX(i,j:k,l) ← Gwh_whx + WHX(i,r:k,l) + WHX(r+1,j:k,l)

    # ----------------------------------------------------------------------
    # VHX: Outer paired & hole paired; uses ZHX and WHX for closures/dangles
    # ----------------------------------------------------------------------
    RE_VHX_DANGLE_L = auto()            # VHX(i,j:k,l) ← P~_hole + L~ + VHX(i,j:k+1,l)
    RE_VHX_DANGLE_R = auto()            # VHX(i,j:k,l) ← P~_hole + R~ + VHX(i,j:k,l-1)
    RE_VHX_DANGLE_LR = auto()           # VHX(i,j:k,l) ← P~_hole + L~ + R~ + VHX(i,j:k+1,l-1)
    RE_VHX_SS_LEFT = auto()             # VHX(i,j:k,l) ← Q~_hole + ZHX*(i,j:k,l)   (Prefer LEFT on strict better)
    RE_VHX_SS_RIGHT = auto()            # VHX(i,j:k,l) ← Q~_hole + ZHX*(i,j:k,l)   (Flip to RIGHT on tie)
    RE_VHX_SPLIT_LEFT_ZHX_WX = auto()   # VHX(i,j:k,l) ← ZHX*(i,j:r,l) + WX*(r+1,k)
    RE_VHX_SPLIT_RIGHT_ZHX_WX = auto()  # VHX(i,j:k,l) ← ZHX*(i,j:k,s) + WX*(l, s-1)
    RE_VHX_WRAP_WHX = auto()            # VHX(i,j:k,l) ← P~_hole + M~_vhx + WHX*(i+1,j-1:k,l)
    RE_VHX_CLOSE_BOTH = auto()          # VHX(i,j:k,l) ← 2·P~_hole + M~_vhx + WHX*(i+1,j-1:k-1,l+1)

    # ----------------------------------------------------------------------
    # ZHX: Hole DP fed by VHX; hole SS/dangles/splits
    # ----------------------------------------------------------------------
    RE_ZHX_FROM_VHX = auto()            # ZHX(i,j:k,l) ← P~_hole + VHX(i,j:k,l)
    RE_ZHX_DANGLE_LR = auto()           # ZHX(i,j:k,l) ← Lh+Rh+P~_hole + VHX(i,j:k-1,l+1)
    RE_ZHX_DANGLE_L = auto()            # ZHX(i,j:k,l) ← Lh+P~_hole + VHX(i,j:k,l+1)
    RE_ZHX_DANGLE_R = auto()            # ZHX(i,j:k,l) ← Rh+P~_hole + VHX(i,j:k-1,l)
    RE_ZHX_SS_LEFT = auto()             # ZHX(i,j:k,l) ← Q~_hole + ZHX(i,j:k-1,l)
    RE_ZHX_SS_RIGHT = auto()            # ZHX(i,j:k,l) ← Q~_hole + ZHX(i,j:k,l+1)
    RE_ZHX_SPLIT_LEFT_ZHX_WX = auto()   # ZHX(i,j:k,l) ← ZHX(i,j:r,l) + WX*(r+1,k)
    RE_ZHX_SPLIT_RIGHT_ZHX_WX = auto()  # ZHX(i,j:k,l) ← ZHX(i,j:k,s) + WX*(l, s-1)

    # ----------------------------------------------------------------------
    # YHX: Outer DP with hole paired; outer dangles / outer SS / wrap via WHX / splits
    # ----------------------------------------------------------------------
    RE_YHX_DANGLE_L = auto()            # YHX(i,j:k,l) ← Lo + P~_out + VHX(i+1, j:k,l)
    RE_YHX_DANGLE_R = auto()            # YHX(i,j:k,l) ← Ro + P~_out + VHX(i, j-1:k,l)
    RE_YHX_DANGLE_LR = auto()           # YHX(i,j:k,l) ← Lo+Ro + P~_out + VHX(i+1, j-1:k,l)
    RE_YHX_SS_LEFT = auto()             # YHX(i,j:k,l) ← Q~_out + YHX(i+1, j:k,l)
    RE_YHX_SS_RIGHT = auto()            # YHX(i,j:k,l) ← Q~_out + YHX(i, j-1:k,l)  (flip to RIGHT on tie)
    RE_YHX_SS_BOTH = auto()             # YHX(i,j:k,l) ← 2·Q~_out + YHX(i+1, j-1:k,l)
    RE_YHX_SPLIT_LEFT_YHX_WX = auto()   # YHX(i,j:k,l) ← YHX(i, r:k,l) + WX*(r+1, j)
    RE_YHX_SPLIT_RIGHT_WX_YHX = auto()  # YHX(i,j:k,l) ← WX*(i, s) + YHX(s+1, j:k,l)
    RE_YHX_WRAP_WHX = auto()            # YHX(i,j:k,l) ← P~_out + M~_yhx + M~_whx + WHX(i, j:k-1, l+1)
    RE_YHX_WRAP_WHX_L = auto()          # YHX(i,j:k,l) ← Lo + P~_out + M~_yhx + M~_whx + WHX(i+1, j:k-1, l+1)
    RE_YHX_WRAP_WHX_R = auto()          # YHX(i,j:k,l) ← Ro + P~_out + M~_yhx + M~_whx + WHX(i, j-1:k-1, l+1)
    RE_YHX_WRAP_WHX_LR = auto()         # YHX(i,j:k,l) ← Lo+Ro + P~_out + M~_yhx + M~_whx + WHX(i+1, j-1:k-1, l+1)


@dataclass(frozen=True, slots=True)
class EddyRivasBackPointer:
    """
    Structured Rivas–Eddy back pointer.

    Fields are optional and only populated when they make sense for the operation:
        - outer: `(i, j)`
        - hole : `(k, l)` for 4-index gap DPs, when relevant
        - split: `r` (or `s`) when a single split index is used
        - split2: A second split index (rare)
        - bridge: `(r, s)` for ĨS2 outer bridge hooks
        - drift: Join-drift distance (d) if used
        - charged: Whether this was a charged composition (wx/vx)
        - note: Free-form metadata
        - args: Payload used by unit tests (tuple of ints)
    """
    op: EddyRivasBacktrackOp
    outer: Optional[Interval] = None
    hole: Optional[Interval] = None
    split: Optional[int] = None
    split2: Optional[int] = None
    bridge: Optional[Interval] = None
    drift: Optional[int] = None
    charged: Optional[bool] = None
    note: Optional[str] = None

    # Unit test payload
    args: Tuple[Any, ...] = field(default_factory=tuple)

    # --- Serialization helpers (nice for logs/debugging) ---
    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op.value,
            "outer": self.outer,
            "hole": self.hole,
            "split": self.split,
            "bridge": self.bridge,
            "drift": self.drift,
            "charged": self.charged,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EddyRivasBackPointer":
        # Loose loader; unknown strings will raise KeyError (good fail-fast)
        op = EddyRivasBacktrackOp(d["op"])
        return EddyRivasBackPointer(
            op=op,
            outer=tuple(d["outer"]) if d.get("outer") else None,
            hole=tuple(d["hole"]) if d.get("hole") else None,
            split=d.get("split"),
            bridge=tuple(d["bridge"]) if d.get("bridge") else None,
            drift=d.get("drift"),
            charged=d.get("charged"),
            meta=d.get("meta"),
        )

    @classmethod
    def compose_vx(cls, r: int, k: int, l: int) -> "EddyRivasBackPointer":
        return cls(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX,
                   split=r, hole=(k, l), args=(r, k, l))

    @classmethod
    def compose_vx_drift(cls, r: int, k: int, l: int, d: int) -> "EddyRivasBackPointer":
        return cls(op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT,
                   split=r, hole=(k, l), drift=d, args=(r, k, l, d))

    @classmethod
    def vx_select_uncharged(cls) -> "EddyRivasBackPointer":
        return cls(op=EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED, args=())

    @classmethod
    def wx_select_uncharged(cls) -> "EddyRivasBackPointer":
        return cls(op=EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED, args=())

    @classmethod
    def whx_shrink_left(cls, i: int, j: int, k1: int, l: int) -> "EddyRivasBackPointer":
        return cls(op=EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT,
                   outer=(i, j), hole=(k1, l), args=(i, j, k1, l))

    @classmethod
    def whx_split_left_whx_wx(cls, r: int) -> "EddyRivasBackPointer":
        return cls(op=EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX,
                   split=r, args=(r,))