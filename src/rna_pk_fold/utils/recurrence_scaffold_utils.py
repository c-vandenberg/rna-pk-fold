from typing import Iterable, Iterator, Tuple

from rna_pk_fold.utils.iter_utils import iter_holes_pairable


def hole_passes_filters(i:int, j:int, k:int, l:int, cfg, vxu_matrix) -> bool:
    hole_w = (l - k - 1)
    if cfg.min_hole_width and hole_w < cfg.min_hole_width:
        return False
    if cfg.max_hole_width and hole_w > cfg.max_hole_width:
        return False
    if cfg.beam_v_threshold != 0.0:
        if vxu_matrix.get(k, l) > cfg.beam_v_threshold:
            return False
    return True


def iter_valid_holes(i:int, j:int, can_pair_mask, cfg, vxu_matrix) -> Iterator[Tuple[int,int]]:
    for (k, l) in iter_holes_pairable(i, j, can_pair_mask):
        if can_pair_mask and not can_pair_mask[k][l]:
            continue
        if hole_passes_filters(i, j, k, l, cfg, vxu_matrix):
            yield k, l
