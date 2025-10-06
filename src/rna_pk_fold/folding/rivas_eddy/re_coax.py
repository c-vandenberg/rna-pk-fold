from typing import Tuple
from rna_pk_fold.utils.nucleotide_utils import pair_str

def _pair_key(seq: str, a: int, b: int):
    return pair_str(seq, a, b) if (0 <= a < len(seq) and 0 <= b < len(seq)) else None

def _coax_energy_for_join(seq: str, left_pair: Tuple[int, int], right_pair: Tuple[int, int], costs) -> float:
    lp = _pair_key(seq, *left_pair)
    rp = _pair_key(seq, *right_pair)
    if lp is None or rp is None:
        return 0.0
    return costs.coax_pairs.get((lp, rp), costs.coax_pairs.get((rp, lp), 0.0))

def _coax_pack(seq: str, i: int, j: int, r: int, k: int, l: int, cfg, costs, adjacent: bool):
    left_len  = (r - i + 1)
    right_len = (j - (k + 1) + 1)
    if left_len < costs.coax_min_helix_len or right_len < costs.coax_min_helix_len:
        return 0.0, 0.0

    mismatch = cfg.enable_coax_mismatch and (abs(k - r) == 1)
    seam_ok  = cfg.enable_coax and (adjacent or mismatch)
    if not seam_ok:
        return 0.0, 0.0

    total = 0.0
    e = _coax_energy_for_join(seq, (i, r), (k + 1, j), costs)
    if mismatch:
        e = e * costs.mismatch_coax_scale + costs.mismatch_coax_bonus
    if e > 0.0:
        e = 0.0
    total += costs.coax_scale_oo * e

    if cfg.enable_coax_variants:
        e_oi = _coax_energy_for_join(seq, (i, r), (k, l), costs)
        if e_oi > 0.0: e_oi = 0.0
        total += costs.coax_scale_oi * e_oi

        e_io = _coax_energy_for_join(seq, (k, l), (k + 1, j), costs)
        if e_io > 0.0: e_io = 0.0
        total += costs.coax_scale_io * e_io

    return total, costs.coax_bonus