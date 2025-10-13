import math

import numpy as np

from rna_pk_fold.utils.matrix_utils import whx_collapse_with, zhx_collapse_with
from rna_pk_fold.energies.energy_pk_ops import coax_pack


def enforce_outer_min_lengths(i, j, r, cfg) -> bool:
    return (r - i) >= cfg.min_outer_left and (j - (r + 1)) >= cfg.min_outer_right


def fill_whx_vectors(state, i, j, k, l, can_pair_mask, cfg):
    base_l = l - k
    l_u = np.full(base_l, np.inf); r_u = np.full(base_l, np.inf)
    l_c = np.full(base_l, np.inf); r_c = np.full(base_l, np.inf)
    left_y = np.full(base_l, np.inf); right_y = np.full(base_l, np.inf)
    left_y_is_ch = np.zeros(base_l, np.uint8); right_y_is_ch = np.zeros(base_l, np.uint8)

    for t in range(base_l):
        r = k + t
        if cfg.strict_complement_order and not (i < k <= r < l <= j):
            continue
        if not enforce_outer_min_lengths(i, j, r, cfg):
            continue

        l_u[t] = whx_collapse_with(state, i, r, k, r, charged=False, can_pair_mask=can_pair_mask)
        r_u[t] = whx_collapse_with(state, r + 1, j, r + 1, l, charged=False, can_pair_mask=can_pair_mask)
        l_c[t] = whx_collapse_with(state, i, r, k, r, charged=True,  can_pair_mask=can_pair_mask)
        r_c[t] = whx_collapse_with(state, r + 1, j, r + 1, l, charged=True,  can_pair_mask=can_pair_mask)

        if can_pair_mask[k][r]:
            ly = state.yhx_matrix.get(i, r, k, r)
            if math.isfinite(ly):
                left_y[t] = ly
                bp_ly = state.yhx_back_ptr.get(i, r, k, r)
                if bp_ly is not None and getattr(bp_ly, "charged", False):
                    left_y_is_ch[t] = 1

        if can_pair_mask[r+1][l]:
            ry = state.yhx_matrix.get(r + 1, j, r + 1, l)
            if math.isfinite(ry):
                right_y[t] = ry
                bp_ry = state.yhx_back_ptr.get(r + 1, j, r + 1, l)
                if bp_ry is not None and getattr(bp_ry, "charged", False):
                    right_y_is_ch[t] = 1

    return l_u, r_u, l_c, r_c, left_y, right_y, left_y_is_ch, right_y_is_ch


def fill_zhx_vectors_and_coax(state, i, j, k, l, seq, cfg, can_pair_mask):
    base_l = l - k
    l_u = np.full(base_l, np.inf); r_u = np.full(base_l, np.inf)
    l_c = np.full(base_l, np.inf); r_c = np.full(base_l, np.inf)
    cx_tot = np.zeros(base_l); cx_bonus = np.zeros(base_l)

    for t in range(base_l):
        r = k + t
        if cfg.strict_complement_order and not (i < k <= r < l <= j):
            continue
        if not enforce_outer_min_lengths(i, j, r, cfg):
            continue

        l_u[t] = zhx_collapse_with(state, i, r, k, r, charged=False, can_pair_mask=can_pair_mask)
        r_u[t] = zhx_collapse_with(state, r + 1, j, r + 1, l, charged=False, can_pair_mask=can_pair_mask)
        l_c[t] = zhx_collapse_with(state, i, r, k, r, charged=True,  can_pair_mask=can_pair_mask)
        r_c[t] = zhx_collapse_with(state, r + 1, j, r + 1, l, charged=True,  can_pair_mask=can_pair_mask)

        adjacent = (r == k)
        cx_tot[t], cx_bonus[t] = coax_pack(seq, i, j, r, k, l, cfg, cfg.costs, adjacent)

    return l_u, r_u, l_c, r_c, cx_tot, cx_bonus
