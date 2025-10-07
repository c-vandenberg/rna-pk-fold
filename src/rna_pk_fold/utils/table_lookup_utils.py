from __future__ import annotations
from typing import Dict, Tuple, Optional


def table_lookup(tbl: Dict[Tuple[str, str], float],
                  x: Optional[str],
                  y: Optional[str],
                  default: float,
                  none_value: float = 0.0) -> float:
    if x is None or y is None:
        return none_value
    return tbl.get((x, y), default)


def clamp_non_favorable(e: float) -> float:
    """Return e if stabilizing (≤0), else 0.0."""
    return e if e <= 0.0 else 0.0