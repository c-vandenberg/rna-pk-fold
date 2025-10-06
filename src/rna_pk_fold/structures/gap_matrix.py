from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any
import math

Pair = Tuple[int, int]
Hole = Tuple[int, int]
Outer = Tuple[int, int]

INF = math.inf


@dataclass(slots=True)
class SparseGapMatrix:
    """
    Sparse 4D one-hole matrix: whx/vhx/zhx/yhx.
    Stored as: store[(i,j)][(k,l)] = float
    Default is +inf unless special "collapse identities" apply.
    """
    n: int
    data: Dict[Outer, Dict[Hole, float]] = field(default_factory=dict)

    def get(self, i: int, j: int, k: int, l: int) -> float:
        # Enforce valid triangular bounds
        if i < 0 or j >= self.n or k < i or l > j or i > j or k > l:
            return INF

        # Collapse identities when the hole vanishes (k = l - 1). Caller must
        # implement via overrides or by passing wx/vx for fallback.
        row = self.data.get((i, j))
        if row is None:
            return INF
        return row.get((k, l), INF)

    def set(self, i: int, j: int, k: int, l: int, value: float) -> None:
        row = self.data.setdefault((i, j), {})
        row[(k, l)] = value

    def row(self, i: int, j: int) -> Dict[Hole, float]:
        return self.data.setdefault((i, j), {})


@dataclass(slots=True)
class SparseGapBackptr:
    """
    Backpointers for the 4D gap matrices; mirrors SparseGapMatrix layout.
    """
    n: int
    data: Dict[Outer, Dict[Hole, object]] = field(default_factory=dict)

    def get(self, i: int, j: int, k: int, l: int):
        return self.data.get((i, j), {}).get((k, l))

    def set(self, i: int, j: int, k: int, l: int, bp) -> None:
        self.data.setdefault((i, j), {})[(k, l)] = bp