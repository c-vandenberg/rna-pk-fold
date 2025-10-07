from __future__ import annotations
from typing import Optional, Tuple

def safe_base(seq: str, idx: int) -> Optional[str]:
    return seq[idx] if 0 <= idx < len(seq) else None

def canonical_pair(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i <= j else (j, i)

def interval_ok(i: int, j: int, n: int) -> bool:
    return 0 <= i <= j < n

def split_default(i: int, j: int, fallback: int | None = None) -> int:
    """A tiny convenience for WHX/YHX/ZHX/VHX split defaults."""
    return ((i + j) // 2) if fallback is None else fallback
