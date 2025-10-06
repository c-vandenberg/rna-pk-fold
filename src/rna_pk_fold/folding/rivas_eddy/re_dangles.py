from typing import Optional, Tuple, Dict
from rna_pk_fold.folding.rivas_eddy.re_recurrences import RERECosts

def _safe_base(seq: str, idx: int) -> Optional[str]:
    return seq[idx] if 0 <= idx < len(seq) else None

def _table_lookup(tbl: Dict[Tuple[str, str], float], x: Optional[str], y: Optional[str], default: float) -> float:
    if x is None or y is None:
        return 0.0
    return tbl.get((x, y), default)

def _dangle_hole_L(seq: str, k: int, costs: "RERECosts") -> float:
    return _table_lookup(costs.dangle_hole_L, _safe_base(seq, k - 1), _safe_base(seq, k), costs.L_tilde)

def _dangle_hole_R(seq: str, l: int, costs: "RERECosts") -> float:
    return _table_lookup(costs.dangle_hole_R, _safe_base(seq, l), _safe_base(seq, l + 1), costs.R_tilde)

def _dangle_outer_L(seq: str, i: int, costs: "RERECosts") -> float:
    return _table_lookup(costs.dangle_outer_L, _safe_base(seq, i), _safe_base(seq, i + 1), costs.L_tilde)

def _dangle_outer_R(seq: str, j: int, costs: "RERECosts") -> float:
    return _table_lookup(costs.dangle_outer_R, _safe_base(seq, j - 1), _safe_base(seq, j), costs.R_tilde)