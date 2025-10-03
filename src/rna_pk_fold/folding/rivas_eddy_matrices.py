from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple
import math

Pair = Tuple[int, int]
Hole = Tuple[int, int]
Outer = Tuple[int, int]

INF = math.inf

@dataclass(slots=True)
class TriMatrix:
    """
    Triangular N x N float matrix with +inf default.
    Only (i <= j) are meaningful. Use get/set.
    """
    n: int
    data: Dict[Outer, float] = field(default_factory=dict)

    def get(self, i: int, j: int) -> float:
        if i > j:
            return 0.0 if i == j + 1 else INF  # empty segment convenience
        if i < 0 or j >= self.n:
            return INF
        return self.data.get((i, j), INF)

    def set(self, i: int, j: int, value: float) -> None:
        self.data[(i, j)] = value


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


def get_whx_with_collapse(whx: SparseGapMatrix, wx: TriMatrix,
                          i: int, j: int, k: int, l: int) -> float:
    # identity: whx(i,j:k,k+1) == wx(i,j)
    if k + 1 == l:
        return wx.get(i, j)
    return whx.get(i, j, k, l)

def get_zhx_with_collapse(zhx: SparseGapMatrix, vx: TriMatrix,
                          i: int, j: int, k: int, l: int) -> float:
    # identity: zhx(i,j:k,k+1) == vx(i,j)
    if k + 1 == l:
        return vx.get(i, j)
    return zhx.get(i, j, k, l)
