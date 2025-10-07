from __future__ import annotations
from typing import Tuple, Mapping

from rna_pk_fold.energies.energy_types import PseudoknotEnergies
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences import EddyRivasFoldingConfig
from rna_pk_fold.utils.indices_utils import safe_base
from rna_pk_fold.utils.table_lookup_utils import table_lookup
from rna_pk_fold.utils.energy_pk_utils import coax_energy_for_join

# --------------------- DANGLES ---------------------------------------------

def dangle_hole_left(seq: str, k: int, costs: PseudoknotEnergies) -> float:
    # uses (k-1, k)
    return table_lookup(costs.dangle_hole_left or {}, safe_base(seq, k - 1), safe_base(seq, k), costs.L_tilde)

def dangle_hole_right(seq: str, l: int, costs: PseudoknotEnergies) -> float:
    # uses (l, l+1)
    return table_lookup(costs.dangle_hole_right or {}, safe_base(seq, l), safe_base(seq, l + 1), costs.R_tilde)

def dangle_outer_left(seq: str, i: int, costs: PseudoknotEnergies) -> float:
    # uses (i, i+1)
    return table_lookup(costs.dangle_outer_left or {}, safe_base(seq, i), safe_base(seq, i + 1), costs.L_tilde)

def dangle_outer_right(seq: str, j: int, costs: PseudoknotEnergies) -> float:
    # uses (j-1, j)
    return table_lookup(costs.dangle_outer_right or {}, safe_base(seq, j - 1), safe_base(seq, j), costs.R_tilde)


# --------------------- COAX PACKING ----------------------------------------

def coax_pack(seq: str,
              i: int, j: int, r: int, k: int, l: int,
              cfg: EddyRivasFoldingConfig,
              costs: PseudoknotEnergies,
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

    pairs_tbl: Mapping[Tuple[str, str], float] = costs.coax_pairs or {}

    total = 0.0
    # outer↔outer seam
    e = coax_energy_for_join(seq, (i, r), (k + 1, j), pairs_tbl)
    if mismatch:
        e = e * costs.mismatch_coax_scale + costs.mismatch_coax_bonus
    if e > 0.0:  # clamp to non-positive
        e = 0.0
    total += costs.coax_scale_oo * e

    # optional variants
    if getattr(cfg, "enable_coax_variants", False):
        e_oi = coax_energy_for_join(seq, (i, r), (k, l), pairs_tbl)
        if e_oi > 0.0: e_oi = 0.0
        total += costs.coax_scale_oi * e_oi

        e_io = coax_energy_for_join(seq, (k, l), (k + 1, j), pairs_tbl)
        if e_io > 0.0: e_io = 0.0
        total += costs.coax_scale_io * e_io

    return total, costs.coax_bonus

# --------------------- SHORT-HOLE CAP --------------------------------------

def short_hole_penalty(costs: PseudoknotEnergies, k: int, l: int) -> float:
    """Cap/penalize very narrow hole widths once per seam."""
    h = l - k - 1
    return (costs.short_hole_caps or {}).get(h, 0.0)
