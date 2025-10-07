from __future__ import annotations
from typing import Optional, Tuple, Mapping
from rna_pk_fold.utils.nucleotide_utils import pair_str

def coax_pair_key(seq: str, a: int, b: int) -> Optional[str]:
    # Safe, returns None if out-of-range or non-canonical handled upstream
    n = len(seq)
    if 0 <= a < n and 0 <= b < n:
        return pair_str(seq, a, b)
    return None

def coax_energy_for_join(
    seq: str,
    left_pair: Tuple[int, int],
    right_pair: Tuple[int, int],
    pairs_tbl: Mapping[Tuple[str, str], float],
) -> float:
    lp = coax_pair_key(seq, *left_pair)
    rp = coax_pair_key(seq, *right_pair)
    if lp is None or rp is None:
        return 0.0

    return pairs_tbl.get((lp, rp), pairs_tbl.get((rp, lp), 0.0))
