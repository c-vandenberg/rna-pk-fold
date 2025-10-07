from __future__ import annotations
from typing import Optional, Tuple, Dict, Protocol
from rna_pk_fold.utils.nucleotide_utils import pair_str

# --- Light protocols so we can stay duck-typed and avoid import cycles -----
class PKCoaxConfig(Protocol):
    enable_coax: bool
    enable_coax_mismatch: bool
    enable_coax_variants: bool

class PKEnergyCosts(Protocol):
    # dangles
    dangle_hole_L: Dict[Tuple[str, str], float]
    dangle_hole_R: Dict[Tuple[str, str], float]
    dangle_outer_L: Dict[Tuple[str, str], float]
    dangle_outer_R: Dict[Tuple[str, str], float]
    L_tilde: float
    R_tilde: float

    # coax
    coax_pairs: Dict[Tuple[str, str], float]
    coax_min_helix_len: int
    mismatch_coax_scale: float
    mismatch_coax_bonus: float
    coax_scale_oo: float
    coax_scale_oi: float
    coax_scale_io: float
    coax_scale: float
    coax_bonus: float

    # short-hole caps
    short_hole_caps: Dict[int, float]


# --------------------- DANGLES (hole & outer) -------------------------------
def _safe_base(seq: str, idx: int) -> Optional[str]:
    return seq[idx] if 0 <= idx < len(seq) else None

def _table_lookup(tbl: Dict[Tuple[str, str], float],
                  x: Optional[str],
                  y: Optional[str],
                  default: float) -> float:
    if x is None or y is None:
        return 0.0
    return tbl.get((x, y), default)

def dangle_hole_L(seq: str, k: int, costs: PKEnergyCosts) -> float:
    # uses (k-1, k)
    return _table_lookup(costs.dangle_hole_L, _safe_base(seq, k - 1), _safe_base(seq, k), costs.L_tilde)

def dangle_hole_R(seq: str, l: int, costs: PKEnergyCosts) -> float:
    # uses (l, l+1)
    return _table_lookup(costs.dangle_hole_R, _safe_base(seq, l), _safe_base(seq, l + 1), costs.R_tilde)

def dangle_outer_L(seq: str, i: int, costs: PKEnergyCosts) -> float:
    # uses (i, i+1)
    return _table_lookup(costs.dangle_outer_L, _safe_base(seq, i), _safe_base(seq, i + 1), costs.L_tilde)

def dangle_outer_R(seq: str, j: int, costs: PKEnergyCosts) -> float:
    # uses (j-1, j)
    return _table_lookup(costs.dangle_outer_R, _safe_base(seq, j - 1), _safe_base(seq, j), costs.R_tilde)


# --------------------- COAX PACKING ----------------------------------------

def _pair_key(seq: str, a: int, b: int) -> Optional[str]:
    return pair_str(seq, a, b) if (0 <= a < len(seq) and 0 <= b < len(seq)) else None

def coax_energy_for_join(seq: str,
                         left_pair: Tuple[int, int],
                         right_pair: Tuple[int, int],
                         costs: PKEnergyCosts) -> float:
    lp = _pair_key(seq, *left_pair)
    rp = _pair_key(seq, *right_pair)
    if lp is None or rp is None:
        return 0.0
    # symmetric lookup
    return costs.coax_pairs.get((lp, rp), costs.coax_pairs.get((rp, lp), 0.0))


def coax_pack(seq: str,
              i: int, j: int, r: int, k: int, l: int,
              cfg: PKCoaxConfig,
              costs: PKEnergyCosts,
              adjacent: bool) -> Tuple[float, float]:
    """
    Return (coax_total, coax_bonus_term) for a seam between
    left helix (i..r) and right helix (k+1..j) with optional variants.

    - Respects enable_coax / enable_coax_mismatch / enable_coax_variants gates.
    - Applies min-helix-length filters.
    - Clamps any positive coax energies to 0 (destabilizing inputs are disallowed).
    """
    left_len  = (r - i + 1)
    right_len = (j - (k + 1) + 1)
    if left_len < costs.coax_min_helix_len or right_len < costs.coax_min_helix_len:
        return 0.0, 0.0

    mismatch = bool(cfg.enable_coax_mismatch) and (abs(k - r) == 1)
    seam_ok  = bool(cfg.enable_coax) and (adjacent or mismatch)
    if not seam_ok:
        return 0.0, 0.0

    total = 0.0
    # outer↔outer seam
    e = coax_energy_for_join(seq, (i, r), (k + 1, j), costs)
    if mismatch:
        e = e * costs.mismatch_coax_scale + costs.mismatch_coax_bonus
    if e > 0.0:  # clamp to non-positive
        e = 0.0
    total += costs.coax_scale_oo * e

    # optional variants
    if getattr(cfg, "enable_coax_variants", False):
        e_oi = coax_energy_for_join(seq, (i, r), (k, l), costs)
        if e_oi > 0.0: e_oi = 0.0
        total += costs.coax_scale_oi * e_oi

        e_io = coax_energy_for_join(seq, (k, l), (k + 1, j), costs)
        if e_io > 0.0: e_io = 0.0
        total += costs.coax_scale_io * e_io

    return total, costs.coax_bonus


# --------------------- SHORT-HOLE PENALTY -----------------------------------

def short_hole_penalty(costs: PKEnergyCosts, k: int, l: int) -> float:
    """
    Cap/penalize very narrow hole widths once per seam.
    """
    h = l - k - 1
    return (costs.short_hole_caps or {}).get(h, 0.0)
