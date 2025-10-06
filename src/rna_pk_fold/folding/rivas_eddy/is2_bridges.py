from typing import Any

def IS2_outer(seq: str, tables: Any, i: int, j: int, r: int, s: int) -> float:
    if tables and hasattr(tables, "IS2_outer"):
        fn = tables.IS2_outer
        return fn(seq, i, j, r, s) if callable(fn) else float(fn)
    return 0.0

def IS2_outer_yhx(cfg: Any, seq: str, i: int, j: int, r: int, s: int) -> float:
    t = getattr(cfg, "tables", None)
    if t is None:
        return 0.0
    fn = getattr(t, "IS2_outer_yhx", None)
    if fn is None:
        return 0.0
    return float(fn(seq, i, j, r, s))