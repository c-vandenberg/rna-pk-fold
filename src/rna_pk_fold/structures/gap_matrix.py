from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterator, MutableMapping, Optional, Union
import math

import numpy as np

Pair = Tuple[int, int]
Hole = Tuple[int, int]
Outer = Tuple[int, int]

INF = np.inf


class _DenseRowProxy(MutableMapping[Hole, float]):
    """
    Read/write mapping view over a dense (i,j) plane.
    Only (k,l) with i <= k <= l <= j are addressable; iteration
    yields finite entries (k < l) by default.
    """
    def __init__(self, parent: 'SparseGapMatrix', i: int, j: int):
        self._p = parent
        self._i = i
        self._j = j

    def __getitem__(self, key: Hole) -> float:
        k, l = key
        return self._p.get(self._i, self._j, k, l)

    def __setitem__(self, key: Hole, value: float) -> None:
        k, l = key
        self._p.set(self._i, self._j, k, l, value)

    def __delitem__(self, key: Hole) -> None:
        k, l = key
        # "Deleting" sets to +inf
        self._p.set(self._i, self._j, k, l, INF)

    def __iter__(self) -> Iterator[Hole]:
        plane = self._p._planes.get((self._i, self._j))
        if plane is None:
            return

        L = plane.shape[0]
        base = self._i
        for dk in range(L):
            for dl in range(dk + 1, L):
                if math.isfinite(plane[dk, dl]):
                    yield base + dk, base + dl

    def __len__(self) -> int:
        plane = self._p.planes.get((self._i, self._j))
        if plane is None:
            return 0
        L = plane.shape[0]
        cnt = 0
        for dk in range(L):
            for dl in range(dk + 1, L):
                if math.isfinite(plane[dk, dl]):
                    cnt += 1
        return cnt

    # Keep common dict-ish conveniences used by older code
    def get(self, key: Hole, default=None):
        v = self.__getitem__(key)
        return v if math.isfinite(v) else default

    def setdefault(self, key: Hole, default: float):
        v = self.__getitem__(key)
        if not math.isfinite(v):
            self.__setitem__(key, default)
            return default
        return v

    def items(self):
        for k in self.__iter__():
            yield k, self.__getitem__(k)


@dataclass(slots=True)
class SparseGapMatrix:
    """
    Sparse 4D one-hole matrix: whx/vhx/zhx/yhx.
    Stored as: store[(i,j)][(k,l)] = float
    Default is +inf unless special "collapse identities" apply.
    """
    n: int
    data: Dict[Outer, Dict[Hole, float]] = field(default_factory=dict)

    _dense_enabled: bool = False
    _planes: Dict[Outer, np.ndarray] = field(default_factory=dict)

    def enable_dense(self, drop_sparse: bool = False) -> None:
        """
        Enable dense per-(i,j) planes. If drop_sparse=True, clears the
        sparse store to free memory (safe to do if you enable before writing).
        """
        self._dense_enabled = True
        if drop_sparse:
            self.data.clear()

    def _ensure_plane(self, i: int, j: int) -> np.ndarray:
        key = (i, j)
        p = self._planes.get(key)
        if p is None:
            L = j - i + 1
            # Full L×L plane; diagonal defaults to +inf (we rarely write it),
            # valid "hole" entries are off-diagonal (k < l).
            p = np.full((L, L), INF, dtype=np.float64, order='C')
            self._planes[key] = p
        return p

    def get(self, i: int, j: int, k: int, l: int) -> float:
        if i < 0 or j >= self.n or i > j or k < i or l > j or k > l:
            return INF

        if not self._dense_enabled:
            row = self.data.get((i, j))
            if row is None:
                return INF
            return row.get((k, l), INF)

        plane = self._planes.get((i, j))
        if plane is None:
            return INF
        return float(plane[k - i, l - i])

    def set(self, i: int, j: int, k: int, l: int, value: float) -> None:
        if not (0 <= i <= j < self.n and i <= k <= l <= j):
            return

        if not self._dense_enabled:
            self.data.setdefault((i, j), {})[(k, l)] = value
            return

        plane = self._ensure_plane(i, j)
        plane[k - i, l - i] = value

    def row(self, i: int, j: int) -> Union[Dict[Hole, float], _DenseRowProxy] :
        """
        Return a mapping-like object for (i,j).
        - Sparse mode: the actual dict (backward compatible).
        - Dense mode : a proxy that reads/writes the dense plane.
        """
        if not self._dense_enabled:
            return self.data.setdefault((i, j), {})

        return _DenseRowProxy(self, i, j)

    @property
    def planes(self):
        return self._planes


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