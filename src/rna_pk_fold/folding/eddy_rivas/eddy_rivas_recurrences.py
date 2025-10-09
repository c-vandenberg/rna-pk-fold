from __future__ import annotations
import math
import json
import time
import logging
from dataclasses import dataclass, fields
from typing import Tuple, Dict, Optional, Any, Callable

import numpy as np
from tqdm import tqdm

from rna_pk_fold.energies.energy_types import PseudoknotEnergies
from rna_pk_fold.folding.zucker.zucker_fold_state import ZuckerFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer, EddyRivasBacktrackOp
from rna_pk_fold.utils.is2_utils import IS2_outer, IS2_outer_yhx
from rna_pk_fold.utils.iter_utils import (iter_spans, iter_holes, iter_complementary_tuples, iter_inner_holes,
                                          iter_holes_pairable, iter_complementary_tuples_pairable_kl)
from rna_pk_fold.utils.matrix_utils import (clear_matrix_caches, get_whx_with_collapse, get_zhx_with_collapse, wxI,
                                                             whx_collapse_with, zhx_collapse_with)
from rna_pk_fold.energies.energy_pk_ops import (dangle_hole_left, dangle_hole_right, dangle_outer_left,
                                                dangle_outer_right, coax_pack, short_hole_penalty)
from rna_pk_fold.folding.eddy_rivas.numba_kernels import (
    compose_wx_best_over_r_arrays,
    compose_vx_best_over_r,
    best_sum,
    best_sum_with_penalty,
)

logger = logging.getLogger(__name__)


def take_best(
    best: float,
    best_bp: Optional[EddyRivasBackPointer],
    cand: float,
    mk_bp: Callable[[], EddyRivasBackPointer],
) -> Tuple[float, Optional[EddyRivasBackPointer]]:
    if cand < best:
        return cand, mk_bp()
    return best, best_bp


def make_bp(i: int, j: int, k: int, l: int) -> Callable[..., EddyRivasBackPointer]:
    def BP(op: EddyRivasBacktrackOp, **kw) -> EddyRivasBackPointer:
        return EddyRivasBackPointer(
            op=op,
            outer=(i, j),
            hole=(k, l),
            **kw
        )
    return BP


@dataclass(slots=True)
class EddyRivasFoldingConfig:
    enable_coax: bool = False # keep off initially
    enable_wx_overlap: bool = False # turn on WX same-hole overlap terms
    enable_coax_variants: bool = False  # NEW: add extra coax topologies in VX composition
    enable_coax_mismatch: bool = False  # allow |k-r|==1 seam as "mismatch coax"
    enable_join_drift: bool = False  # enable slight hole drift at join
    drift_radius: int = 0  # how far to drift (0 = off)
    enable_is2: bool = True
    pk_penalty_gw: float = 1.0 # Gw: pseudoknot introduction penalty (kcal/mol)
    max_hole_width: int = 0
    min_hole_width: int = 0  # 0 = identical behavior; 1+ prunes zero/narrow holes
    min_outer_left: int = 0  # minimal length of [i..r]
    min_outer_right: int = 0  # minimal length of [r+1..j]
    beam_k: int = 0                 # 0 = disabled, else keep at most K (k,l) per (i,j)
    beam_v_threshold: float = 0.0  # keep (k,l) only if nested V[k][l] <= this (e.g. -0.1)
    strict_complement_order: bool = True  # enforce i<k<=r<l<=j
    costs: Optional[PseudoknotEnergies] = None
    tables: object = None
    verbose: bool = False

# -----------------------
# Engine
# -----------------------
class EddyRivasFoldingEngine:
    def __init__(self, config: EddyRivasFoldingConfig):
        self.cfg = config
        self.timings = {}

    @staticmethod
    def _build_can_pair_mask(seq: str) -> list[list[bool]]:
        from rna_pk_fold.rules.constraints import can_pair
        n = len(seq)
        mask = [[False] * n for _ in range(n)]
        for k in range(n):
            bk = seq[k]
            for l in range(k + 1, n):
                mask[k][l] = can_pair(bk, seq[l])
        return mask

    def fill_with_costs(self, seq: str, nested: ZuckerFoldState, re: EddyRivasFoldState) -> None:
        total_start = time.perf_counter()

        n = re.n
        # Log algorithm start
        logger.info("=" * 60)
        logger.info(f"Eddy-Rivas DP for sequence length N={n}")
        logger.info(f"Expected complexity:")
        logger.info(f"  Gap matrices: O(N⁴) ≈ {n ** 4:,} operations")
        logger.info(f"  Compositions: O(N⁶) ≈ {n ** 6:,} operations")
        logger.info("=" * 60)

        clear_matrix_caches()

        q = self.cfg.costs.q_ss
        Gw = self.cfg.pk_penalty_gw
        Gwh = getattr(self.cfg.costs, "Gwh", 0.0)
        Gwi = self.cfg.costs.Gwi
        Gwh_wx = getattr(self.cfg.costs, "Gwh_wx", 0.0)
        Gwh_whx = getattr(self.cfg.costs, "Gwh_whx", 0.0)
        tables = getattr(self.cfg, "tables", None)
        g = self.cfg.costs.coax_scale

        # tilde scalars (names preserved)
        P_out = getattr(tables, "P_tilde_out", getattr(self.cfg.costs, "P_tilde_out", 1.0))
        P_hole = getattr(tables, "P_tilde_hole", getattr(self.cfg.costs, "P_tilde_hole", 1.0))
        L_ = getattr(tables, "L_tilde", 0.0)
        R_ = getattr(tables, "R_tilde", 0.0)
        Q_out = getattr(tables, "Q_tilde_out", getattr(self.cfg.costs, "Q_tilde_out", 0.0))
        Q_hole = getattr(tables, "Q_tilde_hole", getattr(self.cfg.costs, "Q_tilde_hole", 0.0))
        M_yhx = getattr(tables, "M_tilde_yhx", getattr(self.cfg.costs, "M_tilde_yhx", 0.0))
        M_vhx = getattr(tables, "M_tilde_vhx", getattr(self.cfg.costs, "M_tilde_vhx", 0.0))
        M_whx = getattr(tables, "M_tilde_whx", getattr(self.cfg.costs, "M_tilde_whx", 0.0))

        # Seeding
        seed_start = time.perf_counter()
        self._seed_from_nested(nested, re)
        re.wxu_matrix.enable_dense()
        re.wxc_matrix.enable_dense()
        re.vxu_matrix.enable_dense()
        re.vxc_matrix.enable_dense()
        can_pair_mask = self._build_can_pair_mask(seq)
        self.timings['seed'] = time.perf_counter() - seed_start
        logger.info(f"Seeding completed in {self.timings['seed']:.2f}s")

        # WHX
        logger.info("Filling WHX matrix...")
        whx_start = time.perf_counter()
        self._dp_whx(seq, re, q, Gwh_whx, can_pair_mask)
        self.timings['whx'] = time.perf_counter() - whx_start
        logger.info(f"WHX filled in {self.timings['whx']:.2f}s")

        # VHX
        logger.info("Filling VHX matrix...")
        vhx_start = time.perf_counter()
        self._dp_vhx(seq, re, q, Gwi, P_hole, L_, R_, Q_hole, M_vhx, M_whx, can_pair_mask)
        self.timings['vhx'] = time.perf_counter() - vhx_start
        logger.info(f"VHX filled in {self.timings['vhx']:.2f}s")

        # ZHX
        logger.info("Filling ZHX matrix...")
        zhx_start = time.perf_counter()
        self._dp_zhx(seq, re, q, Gwi, P_hole, Q_hole, can_pair_mask)
        self.timings['zhx'] = time.perf_counter() - zhx_start
        logger.info(f"ZHX filled in {self.timings['zhx']:.2f}s")

        # YHX
        logger.info("Filling YHX matrix...")
        yhx_start = time.perf_counter()
        self._dp_yhx(seq, re, q, Gwi, P_out, Q_out, M_yhx, M_whx, can_pair_mask)
        self.timings['yhx'] = time.perf_counter() - yhx_start
        logger.info(f"YHX filled in {self.timings['yhx']:.2f}s")

        # WX Composition
        logger.info("Composing WX matrix...")
        wx_start = time.perf_counter()
        self._compose_wx(seq, re, Gw, Gwh_wx, can_pair_mask)
        self._publish_wx(re)
        self.timings['wx_compose'] = time.perf_counter() - wx_start
        logger.info(f"WX composed in {self.timings['wx_compose']:.2f}s")

        # VX Composition
        logger.info("Composing VX matrix...")
        vx_start = time.perf_counter()
        self._compose_vx(seq, re, Gw, g, can_pair_mask)
        self._publish_vx(re)
        self.timings['vx_compose'] = time.perf_counter() - vx_start
        logger.info(f"VX composed in {self.timings['vx_compose']:.2f}s")

        # Total timing
        self.timings['total'] = time.perf_counter() - total_start

        final_energy = re.wx_matrix.get(0, n - 1)
        logger.info("=" * 60)
        logger.info(f"Eddy-Rivas DP completed in {self.timings['total']:.2f}s")
        logger.info(f"Final WX[0,{n - 1}] = {final_energy:.3f} kcal/mol")
        logger.info("")
        logger.info("Timing breakdown:")
        logger.info(
            f"  Seeding:        {self.timings['seed']:7.2f}s ({self.timings['seed'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  WHX fill:       {self.timings['whx']:7.2f}s ({self.timings['whx'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  VHX fill:       {self.timings['vhx']:7.2f}s ({self.timings['vhx'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  ZHX fill:       {self.timings['zhx']:7.2f}s ({self.timings['zhx'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  YHX fill:       {self.timings['yhx']:7.2f}s ({self.timings['yhx'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  WX composition: {self.timings['wx_compose']:7.2f}s ({self.timings['wx_compose'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  VX composition: {self.timings['vx_compose']:7.2f}s ({self.timings['vx_compose'] / self.timings['total'] * 100:5.1f}%)")
        gap_total = self.timings['whx'] + self.timings['vhx'] + self.timings['zhx'] + self.timings['yhx']
        comp_total = self.timings['wx_compose'] + self.timings['vx_compose']
        logger.info(f"  Gap matrices:   {gap_total:7.2f}s ({gap_total / self.timings['total'] * 100:5.1f}%)")
        logger.info(f"  Compositions:   {comp_total:7.2f}s ({comp_total / self.timings['total'] * 100:5.1f}%)")
        logger.info("=" * 60)

    # --------- Seeding ---------
    @staticmethod
    def _seed_from_nested(nested: ZuckerFoldState, re: EddyRivasFoldState) -> None:
        n = re.n
        for i, j in iter_spans(n):
            base_w = nested.w_matrix.get(i, j)
            base_v = nested.v_matrix.get(i, j)

            re.wxu_matrix.set(i, j, base_w)
            re.vxu_matrix.set(i, j, base_v)

            if i != j:
                re.wxc_matrix.set(i, j, math.inf)
                re.vxc_matrix.set(i, j, math.inf)

            re.wx_matrix.set(i, j, base_w)
            re.vx_matrix.set(i, j, base_v)

            if hasattr(re, "wxi_matrix") and re.wxi_matrix is not None:
                re.wxi_matrix.set(i, j, base_w)

    # --------- WHX ---------
    def _dp_whx(self, seq: str, re: EddyRivasFoldState, q: float, Gwh_whx: float, can_pair_mask) -> None:
        spans = list(iter_spans(re.n))
        for i, j in tqdm(spans, desc="WHX", leave=False):
            for k, l in iter_holes_pairable(i, j, can_pair_mask):
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                if self.cfg.beam_v_threshold != 0.0:
                    if re.vxu_matrix.get(k, l) > self.cfg.beam_v_threshold:
                        continue

                best = math.inf
                best_bp: Optional[EddyRivasBackPointer] = None
                BP = make_bp(i, j, k, l)

                # 1) shrink-left: (k+1,l) + q
                v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k + 1, l)
                cand = v + q
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT, outer=(i, j), hole=(k, l))

                # 2) shrink-right: (k,l-1) + q
                v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k, l - 1)
                cand = v + q
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SHRINK_RIGHT, outer=(i, j), hole=(k, l))

                # 3) trim outer-left: (i+1,j:k,l) + q
                v = re.whx_matrix.get(i + 1, j, k, l)
                cand = v + q
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_TRIM_LEFT, outer=(i, j), hole=(k, l))

                # 4) trim outer-right: (i,j-1:k,l) + q
                v = re.whx_matrix.get(i, j - 1, k, l)
                cand = v + q
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_TRIM_RIGHT, outer=(i, j), hole=(k, l))

                # 5) direct collapse
                v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k, l)
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE, outer=(i, j), hole=(k, l))

                # SS both sides
                v = re.whx_matrix.get(i + 1, j - 1, k, l)
                if math.isfinite(v):
                    cand = v + 2.0 * q
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SS_BOTH, outer=(i, j), hole=(k, l))

                # non-nested outer splits with WX
                Lr = max(0, j - i)
                if Lr > 0:
                    left_vec = np.full(Lr, np.inf, dtype=np.float64)
                    right_vec = np.full(Lr, np.inf, dtype=np.float64)
                    for t in range(Lr):
                        r = i + t
                        lv = re.whx_matrix.get(i, r, k, l)
                        rv = wxI(re, r + 1, j)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv

                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        r_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX,
                            outer=(i, j), hole=(k, l), split=r_star
                        )

                Ls = max(0, j - i)
                if Ls > 0:
                    left_vec = np.full(Ls, np.inf, dtype=np.float64)
                    right_vec = np.full(Ls, np.inf, dtype=np.float64)
                    for t in range(Ls):
                        s2 = i + t
                        lv = wxI(re, i, s2)
                        rv = re.whx_matrix.get(s2 + 1, j, k, l)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv

                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        s2_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_WHX_SPLIT_RIGHT_WX_WHX,
                            outer=(i, j), hole=(k, l), split=s2_star
                        )

                if Gwh_whx != 0.0:
                    Lr = max(0, j - i)
                    if Lr > 0:
                        left_vec = np.full(Lr, np.inf, dtype=np.float64)
                        right_vec = np.full(Lr, np.inf, dtype=np.float64)
                        for t in range(Lr):
                            r = i + t
                            lv = re.whx_matrix.get(i, r, k, l)
                            rv = re.whx_matrix.get(r + 1, j, k, l)
                            if math.isfinite(lv): left_vec[t] = lv
                            if math.isfinite(rv): right_vec[t] = rv

                        cand_overlap, t_star = best_sum_with_penalty(left_vec, right_vec, float(Gwh_whx))
                        if t_star >= 0 and cand_overlap < best:
                            r_star = i + t_star
                            best = cand_overlap
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_WHX_OVERLAP_SPLIT,
                                outer=(i, j), hole=(k, l), split=r_star
                            )

                # IS2 (outer_yhx) + yhx(r2,s2:k,l)
                if self.cfg.enable_is2:
                    for r2 in range(i, k + 1):
                        for s2 in range(l, j + 1):
                            if r2 <= k and l <= s2 and r2 <= s2:
                                inner_y = re.yhx_matrix.get(r2, s2, k, l)
                                if math.isfinite(inner_y):
                                    bridge = IS2_outer_yhx(self.cfg, seq, i, j, r2, s2)
                                    cand = bridge + inner_y
                                    if cand < best:
                                        best = cand
                                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_IS2_INNER_YHX,
                                                                       outer=(i, j), hole=(k, l), bridge=(r2, s2))

                re.whx_matrix.set(i, j, k, l, best)
                re.whx_back_ptr.set(i, j, k, l, best_bp)

    # --------- VHX ---------
    def _dp_vhx(
        self,
        seq: str,
        re: EddyRivasFoldState,
        q: float,
        Gwi: float,
        P_hole: float,
        L_: float,
        R_: float,
        Q_hole: float,
        M_vhx: float,
        M_whx: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        VHX: inner-helix context (k,l) inside outer (i,j).
        Pairability filter on (k,l) + hole width guards.
        """
        spans = list(iter_spans(re.n))
        for i, j in tqdm(spans, desc="VHX", leave=False):
            for k, l in iter_holes_pairable(i, j, can_pair_mask):
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                if self.cfg.beam_v_threshold != 0.0:
                    if re.vxu_matrix.get(k, l) > self.cfg.beam_v_threshold:
                        continue

                best = re.vhx_matrix.get(i, j, k, l)
                best_bp: Optional[EddyRivasBackPointer] = None

                # DANGLES
                v = re.vhx_matrix.get(i, j, k + 1, l)
                cand = P_hole + L_ + v
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_DANGLE_L,
                        outer=(i, j), hole=(k, l)
                    )

                v = re.vhx_matrix.get(i, j, k, l - 1)
                cand = P_hole + R_ + v
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_DANGLE_R,
                        outer=(i, j), hole=(k, l)
                    )

                v = re.vhx_matrix.get(i, j, k + 1, l - 1)
                cand = P_hole + L_ + R_ + v
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_DANGLE_LR,
                        outer=(i, j), hole=(k, l)
                    )

                # SS from ZHX
                v_zhx = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, k, l)
                cand = Q_hole + v_zhx
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_SS_LEFT,
                        outer=(i, j), hole=(k, l)
                    )
                elif (cand == best and isinstance(best_bp, EddyRivasBackPointer) and
                      best_bp.op in (EddyRivasBacktrackOp.RE_VHX_SS_LEFT,
                                     EddyRivasBacktrackOp.RE_VHX_SS_RIGHT)):
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_SS_RIGHT,
                        outer=(i, j), hole=(k, l)
                    )

                # SPLIT LEFT
                Lr = max(0, k - i)
                if Lr > 0:
                    left_vec = np.full(Lr, np.inf, dtype=np.float64)
                    right_vec = np.full(Lr, np.inf, dtype=np.float64)
                    for t in range(Lr):
                        r = i + t
                        lv = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, r, l)
                        rv = wxI(re, r + 1, k)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv

                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        r_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_SPLIT_LEFT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=r_star
                        )

                # SPLIT RIGHT
                Ls = max(0, j - (l + 1) + 1)  # = j - l
                if Ls > 0:
                    left_vec = np.full(Ls, np.inf, dtype=np.float64)
                    right_vec = np.full(Ls, np.inf, dtype=np.float64)
                    for t in range(Ls):
                        s2 = (l + 1) + t
                        lv = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, k, s2)
                        rv = wxI(re, l, s2 - 1)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv

                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        s2_star = (l + 1) + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_SPLIT_RIGHT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=s2_star
                        )

                # IS2 + zhx(r,s2:k,l)  (optional, expensive)
                if self.cfg.enable_is2:
                    for r in range(i, k + 1):
                        for s2 in range(l, j + 1):
                            if r <= k and l <= s2 and r <= s2:
                                inner = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, r, s2, k, l)
                                cand = IS2_outer(seq, self.cfg.tables, i, j, r, s2) + inner
                                if cand < best:
                                    best = cand
                                    best_bp = EddyRivasBackPointer(
                                        op=EddyRivasBacktrackOp.RE_VHX_IS2_INNER_ZHX,
                                        outer=(i, j), hole=(k, l), bridge=(r, s2)
                                    )

                # CLOSE_BOTH
                close = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i + 1, j - 1, k - 1, l + 1)
                if math.isfinite(close):
                    cand = 2.0 * P_hole + M_vhx + close + Gwi + M_whx
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_CLOSE_BOTH,
                            outer=(i, j), hole=(k, l)
                        )

                # WRAP via WHX
                wrap = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i + 1, j - 1, k, l)
                cand = P_hole + M_vhx + wrap + Gwi + M_whx
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_WRAP_WHX,
                        outer=(i, j), hole=(k, l)
                    )

                re.vhx_matrix.set(i, j, k, l, best)
                re.vhx_back_ptr.set(i, j, k, l, best_bp)

    # --------- ZHX ---------
    def _dp_zhx(
        self,
        seq: str,
        re: EddyRivasFoldState,
        q: float,
        Gwi: float,
        P_hole: float,
        Q_hole: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        ZHX: inner-helix-anchored context for the outer stem (i,j) around hole (k,l).
        Pairability filter on (k,l) + hole width guards.
        """
        spans = list(iter_spans(re.n))
        for i, j in tqdm(spans, desc="ZHX", leave=False):
            for k, l in iter_holes_pairable(i, j, can_pair_mask):
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                if self.cfg.beam_v_threshold != 0.0:
                    if re.vxu_matrix.get(k, l) > self.cfg.beam_v_threshold:
                        continue

                best = math.inf
                best_bp: Optional[EddyRivasBackPointer] = None

                # FROM_VHX
                v = re.vhx_matrix.get(i, j, k, l)
                if math.isfinite(v):
                    cand = P_hole + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_FROM_VHX, outer=(i, j),
                                                       hole=(k, l))

                # DANGLE_LR from VHX
                v = re.vhx_matrix.get(i, j, k - 1, l + 1)
                if math.isfinite(v):
                    Lh = dangle_hole_left(seq, k, self.cfg.costs)
                    Rh = dangle_hole_right(seq, l, self.cfg.costs)
                    cand = Lh + Rh + P_hole + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_DANGLE_LR, outer=(i, j),
                                                       hole=(k, l))

                # DANGLE_R from VHX
                v = re.vhx_matrix.get(i, j, k - 1, l)
                if math.isfinite(v):
                    Rh = dangle_hole_right(seq, l - 1, self.cfg.costs)
                    cand = Rh + P_hole + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_DANGLE_R, outer=(i, j), hole=(k, l))

                # DANGLE_L from VHX
                v = re.vhx_matrix.get(i, j, k, l + 1)
                if math.isfinite(v):
                    Lh = dangle_hole_left(seq, k + 1, self.cfg.costs)
                    cand = Lh + P_hole + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_DANGLE_L, outer=(i, j),
                                                       hole=(k, l))

                # SS_LEFT
                v = re.zhx_matrix.get(i, j, k - 1, l)
                if math.isfinite(v):
                    cand = Q_hole + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_SS_LEFT, outer=(i, j),
                                                       hole=(k, l))

                # SS_RIGHT (flip on tie)
                v = re.zhx_matrix.get(i, j, k, l + 1)
                if math.isfinite(v):
                    cand = Q_hole + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT,
                            outer=(i, j), hole=(k, l)
                        )
                    elif (cand == best and isinstance(best_bp, EddyRivasBackPointer) and
                          best_bp.op in (EddyRivasBacktrackOp.RE_ZHX_SS_LEFT,
                                         EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT)):
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT,
                            outer=(i, j), hole=(k, l)
                        )

                # SPLITS
                Lr = max(0, k - i)
                if Lr > 0:
                    left_vec = np.full(Lr, np.inf, dtype=np.float64)
                    right_vec = np.full(Lr, np.inf, dtype=np.float64)
                    for t in range(Lr):
                        r = i + t
                        lv = re.zhx_matrix.get(i, j, r, l)
                        rv = wxI(re, r + 1, k)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv
                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        r_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SPLIT_LEFT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=r_star
                        )

                Ls = max(0, j - l)
                if Ls > 0:
                    left_vec = np.full(Ls, np.inf, dtype=np.float64)
                    right_vec = np.full(Ls, np.inf, dtype=np.float64)
                    for t in range(Ls):
                        s2 = (l + 1) + t
                        lv = re.zhx_matrix.get(i, j, k, s2)
                        rv = wxI(re, l, s2 - 1)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv
                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        s2_star = (l + 1) + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SPLIT_RIGHT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=s2_star
                        )

                # IS2 + vhx(r,s2:k,l)  (optional, expensive)
                if self.cfg.enable_is2:
                    for r in range(i, k + 1):
                        for s2 in range(l, j + 1):
                            if r <= s2:
                                inner = re.vhx_matrix.get(r, s2, k, l)
                                if math.isfinite(inner):
                                    bridge = IS2_outer(seq, self.cfg.tables, i, j, r, s2)
                                    cand = bridge + inner
                                    if cand < best:
                                        best = cand
                                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_IS2_INNER_VHX,
                                                                       outer=(i, j), hole=(k, l), bridge=(r, s2))

                re.zhx_matrix.set(i, j, k, l, best)
                re.zhx_back_ptr.set(i, j, k, l, best_bp)

    # --------- YHX ---------
    def _dp_yhx(
        self,
        seq: str,
        re: EddyRivasFoldState,
        q: float,
        Gwi: float,
        P_out: float,
        Q_out: float,
        M_yhx: float,
        M_whx: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        YHX: outer helix context around an inner (k,l) helix.
        Now iterates only pairable (k,l) and respects min/max hole width.
        """
        for i, j in iter_spans(re.n):
            for k, l in iter_holes_pairable(i, j, can_pair_mask):
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                if self.cfg.beam_v_threshold != 0.0:
                    if re.vxu_matrix.get(k, l) > self.cfg.beam_v_threshold:
                        continue

                best = math.inf
                best_bp: Optional[EddyRivasBackPointer] = None

                # Outer dangle L
                v = re.vhx_matrix.get(i + 1, j, k, l)
                if math.isfinite(v):
                    Lo = dangle_outer_left(seq, i, self.cfg.costs)
                    cand = Lo + P_out + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_DANGLE_L, outer=(i, j), hole=(k, l))

                # Outer dangle R
                v = re.vhx_matrix.get(i, j - 1, k, l)
                if math.isfinite(v):
                    Ro = dangle_outer_right(seq, j, self.cfg.costs)
                    cand = Ro + P_out + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_DANGLE_R,
                                                       outer=(i, j), hole=(k, l))

                # Outer dangle LR
                v = re.vhx_matrix.get(i + 1, j - 1, k, l)
                if math.isfinite(v):
                    Lo = dangle_outer_left(seq, i, self.cfg.costs)
                    Ro = dangle_outer_right(seq, j, self.cfg.costs)
                    cand = Lo + Ro + P_out + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_DANGLE_LR, outer=(i, j),
                                                       hole=(k, l))

                # SS trims: Left
                v = re.yhx_matrix.get(i + 1, j, k, l)
                if math.isfinite(v):
                    cand = Q_out + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_SS_LEFT, outer=(i, j),
                                                       hole=(k, l))

                # SS trims: Right (flip on tie)
                v = re.yhx_matrix.get(i, j - 1, k, l)
                if math.isfinite(v):
                    cand = Q_out + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_YHX_SS_RIGHT,
                            outer=(i, j), hole=(k, l)
                        )
                    elif (cand == best and isinstance(best_bp, EddyRivasBackPointer) and
                          best_bp.op in (EddyRivasBacktrackOp.RE_YHX_SS_LEFT,
                                         EddyRivasBacktrackOp.RE_YHX_SS_RIGHT)):
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_YHX_SS_RIGHT,
                            outer=(i, j), hole=(k, l)
                        )

                # SS both sides
                v = re.yhx_matrix.get(i + 1, j - 1, k, l)
                if math.isfinite(v):
                    cand = 2.0 * Q_out + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_SS_BOTH, outer=(i, j),
                                                       hole=(k, l))

                # Wrap via WHX(i,j:k-1,l+1)
                v = re.whx_matrix.get(i, j, k - 1, l + 1)
                if math.isfinite(v):
                    cand = P_out + M_yhx + M_whx + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX, outer=(i, j), hole=(k, l))

                # Wrap + outer dangles
                v = re.whx_matrix.get(i + 1, j, k - 1, l + 1)
                if math.isfinite(v):
                    Lo = dangle_outer_left(seq, i, self.cfg.costs)
                    cand = Lo + P_out + M_yhx + M_whx + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_L, outer=(i, j), hole=(k, l))

                v = re.whx_matrix.get(i, j - 1, k - 1, l + 1)
                if math.isfinite(v):
                    Ro = dangle_outer_right(seq, j, self.cfg.costs)
                    cand = Ro + P_out + M_yhx + M_whx + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_R, outer=(i, j), hole=(k, l))

                v = re.whx_matrix.get(i + 1, j - 1, k - 1, l + 1)
                if math.isfinite(v):
                    Lo = dangle_outer_left(seq, i, self.cfg.costs)
                    Ro = dangle_outer_right(seq, j, self.cfg.costs)
                    cand = Lo + Ro + P_out + M_yhx + M_whx + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR, outer=(i, j), hole=(k, l))

                # Outer splits with WX
                Lr = max(0, j - i)
                if Lr > 0:
                    left_vec = np.full(Lr, np.inf, dtype=np.float64)
                    right_vec = np.full(Lr, np.inf, dtype=np.float64)
                    for t in range(Lr):
                        r = i + t
                        lv = re.yhx_matrix.get(i, r, k, l)
                        rv = wxI(re, r + 1, j)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv
                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        r_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_YHX_SPLIT_LEFT_YHX_WX,
                            outer=(i, j), hole=(k, l), split=r_star
                        )

                Ls = max(0, j - i)
                if Ls > 0:
                    left_vec = np.full(Ls, np.inf, dtype=np.float64)
                    right_vec = np.full(Ls, np.inf, dtype=np.float64)
                    for t in range(Ls):
                        s2 = i + t
                        lv = wxI(re, i, s2)
                        rv = re.yhx_matrix.get(s2 + 1, j, k, l)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv
                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        s2_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_YHX_SPLIT_RIGHT_WX_YHX,
                            outer=(i, j), hole=(k, l), split=s2_star
                        )

                # IS2 (optional, expensive)
                if self.cfg.enable_is2:
                    for r2 in range(i, k + 1):
                        for s2 in range(l, j + 1):
                            if r2 <= s2:
                                inner_w = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, r2, s2, k, l)
                                if math.isfinite(inner_w):
                                    bridge = IS2_outer_yhx(self.cfg, seq, i, j, r2, s2)
                                    cand = bridge + inner_w
                                    if cand < best:
                                        best = cand
                                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_IS2_INNER_WHX,
                                                                       outer=(i, j), hole=(k, l), bridge=(r2, s2))


                re.yhx_matrix.set(i, j, k, l, best)
                re.yhx_back_ptr.set(i, j, k, l, best_bp)

    # --------- WX Composition & Publish ---------
    def _compose_wx(
        self,
        seq: str,
        re: EddyRivasFoldState,
        Gw: float,
        Gwh_wx: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        Compose WX from uncharged (WXU) + charged candidates built around pairable (k,l).
        Numba-accelerated: per-(k,l) we precompute vector components over r and let
        the kernel minimize across r in one go.
        """
        spans = list(iter_spans(re.n))
        for i, j in tqdm(spans, desc="WX Compose", leave=False):
            best_c = re.wxc_matrix.get(i, j)
            best_bp: Optional[EddyRivasBackPointer] = None
            wxu_baseline = re.wxu_matrix.get(i, j)

            # iterate pairable holes only
            for (k, l) in iter_holes_pairable(i, j, can_pair_mask):
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                # Beam pruning on inner helix
                if self.cfg.beam_v_threshold != 0.0:
                    v_inner = re.vxu_matrix.get(k, l)
                    if v_inner > self.cfg.beam_v_threshold:
                        continue

                # Precompute r-range vectors: r ∈ [k, l-1]
                L = l - k
                Lu = np.full(L, np.inf, dtype=np.float64)
                Ru = np.full(L, np.inf, dtype=np.float64)
                Lc = np.full(L, np.inf, dtype=np.float64)
                Rc = np.full(L, np.inf, dtype=np.float64)
                left_y = np.full(L, np.inf, dtype=np.float64)
                right_y = np.full(L, np.inf, dtype=np.float64)

                for t in range(L):
                    r = k + t
                    if self.cfg.strict_complement_order:
                        if not (i < k <= r < l <= j):
                            continue
                    # min outer lengths mask
                    if (r - i) < self.cfg.min_outer_left or (j - (r + 1)) < self.cfg.min_outer_right:
                        continue

                    # collapse terms (uncharged/charged)
                    Lu[t] = whx_collapse_with(re, i, r, k, r, charged=False)
                    Ru[t] = whx_collapse_with(re, k + 1, j, r + 1, l, charged=False)
                    Lc[t] = whx_collapse_with(re, i, r, k, r, charged=True)
                    Rc[t] = whx_collapse_with(re, k + 1, j, r + 1, l, charged=True)

                    # yhx pairings
                    ly = re.yhx_matrix.get(i, r, k, r)
                    ry = re.yhx_matrix.get(k + 1, j, r + 1, l - 1)  # note (l-1) as in your code
                    if math.isfinite(ly): left_y[t] = ly
                    if math.isfinite(ry): right_y[t] = ry

                cap_pen = short_hole_penalty(self.cfg.costs, k, l)

                # Kernel: get best over r for this (k,l)
                cand, t_star, case_id = compose_wx_best_over_r_arrays(
                    Lu, Ru, Lc, Rc, left_y, right_y, float(Gw), float(cap_pen)
                )

                if cand < best_c:
                    best_c = cand
                    r_star = k + t_star  # Convert array index to absolute position

                    hole_left = None
                    hole_right = None

                    if case_id in (0, 1, 2, 3):
                        # WHX + WHX
                        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX
                        hole_left = (k, r_star)
                        hole_right = (r_star + 1, l)  # WHX uses l

                    elif case_id == 4:
                        # YHX + YHX
                        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX
                        hole_left = (k, r_star)
                        hole_right = (r_star + 1, l - 1)  # YHX uses l-1

                    elif case_id in (5, 6):
                        # YHX + WHX (both cases use same holes!)
                        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX
                        hole_left = (k, r_star)
                        hole_right = (r_star + 1, l)  # WHX uses l

                    else:  # case_id in (7, 8)
                        # WHX + YHX
                        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX
                        hole_left = (k, r_star)
                        hole_right = (r_star + 1, l - 1)  # YHX uses l-1

                    best_bp = EddyRivasBackPointer(
                        op=op,
                        outer=(i, j),
                        hole=(k, l),
                        hole_left=hole_left,
                        hole_right=hole_right,
                        split=r_star,
                        charged=True
                    )

            # optional overlap path (unchanged)
            if self.cfg.enable_wx_overlap and Gwh_wx != 0.0:
                for (k2, l2) in iter_inner_holes(i, j, min_hole=self.cfg.min_hole_width):
                    for r2 in range(i, j):
                        left_y = re.yhx_matrix.get(i, r2, k2, l2)
                        right_y = re.yhx_matrix.get(r2 + 1, j, k2, l2)
                        if math.isfinite(left_y) and math.isfinite(right_y):
                            cand_overlap = (
                                    Gwh_wx + left_y + right_y +
                                    short_hole_penalty(self.cfg.costs, k2, l2)
                            )
                            if cand_overlap < best_c:
                                best_c = cand_overlap
                                best_bp = EddyRivasBackPointer(
                                    op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_OVERLAP,
                                    outer=(i, j), hole=(k2, l2), split=r2, charged=True
                                )

            # (optional) summary print you already had
            if i == 0 and j == re.n - 1:
                print(f"\n=== Final WX Composition Summary [0,{j}] ===")
                print(f"WXU (nested baseline): {wxu_baseline:.3f}")
                print(f"Best WXC found: {best_c:.3f}")
                if best_bp:
                    print(f"Winning op: {best_bp.op}")
                    print(f"Winning (r,k,l): ({best_bp.split}, {best_bp.hole})")
                else:
                    print("No winning PK candidate found")
                print(f"Improvement: {wxu_baseline - best_c:.3f} kcal/mol")

            re.wxc_matrix.set(i, j, best_c)
            if best_bp is not None:
                re.wx_back_ptr.set(i, j, best_bp)

    def _publish_wx(self, re: EddyRivasFoldState) -> None:
        for i, j in iter_spans(re.n):
            wxu = re.wxu_matrix.get(i, j)
            wxc = re.wxc_matrix.get(i, j)
            if self.cfg.enable_wx_overlap and not math.isfinite(wxc):
                re.wxc_matrix.set(i, j, wxu)
                wxc = wxu
            if wxu <= wxc:
                re.wx_matrix.set(i, j, wxu)
                re.wx_back_ptr.set(i, j, EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED))
            else:
                re.wx_matrix.set(i, j, wxc)

    # --------- VX Composition & Publish ---------
    def _compose_vx(
        self,
        seq: str,
        re: EddyRivasFoldState,
        Gw: float,
        g: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        Compose VX from uncharged (VXU) + charged candidates around pairable (k,l).
        Numba-accelerated: precompute vector components over r and minimize in kernel.
        """
        spans = list(iter_spans(re.n))
        for i, j in tqdm(spans, desc="VX Compose", leave=False):
            best_c = re.vxc_matrix.get(i, j)
            best_bp: Optional[EddyRivasBackPointer] = None

            for (k, l) in iter_holes_pairable(i, j, can_pair_mask):
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                L = l - k
                Lu = np.full(L, np.inf, dtype=np.float64)
                Ru = np.full(L, np.inf, dtype=np.float64)
                Lc = np.full(L, np.inf, dtype=np.float64)
                Rc = np.full(L, np.inf, dtype=np.float64)
                coax_total = np.zeros(L, dtype=np.float64)
                coax_bonus = np.zeros(L, dtype=np.float64)

                for t in range(L):
                    r = k + t
                    if self.cfg.strict_complement_order:
                        if not (i < k <= r < l <= j):
                            continue
                    if (r - i) < self.cfg.min_outer_left or (j - (r + 1)) < self.cfg.min_outer_right:
                        continue

                    # collapse terms for VX composition
                    Lu[t] = zhx_collapse_with(re, i, r, k, r, charged=False)
                    Ru[t] = zhx_collapse_with(re, k + 1, j, r + 1, l, charged=False)
                    Lc[t] = zhx_collapse_with(re, i, r, k, r, charged=True)
                    Rc[t] = zhx_collapse_with(re, k + 1, j, r + 1, l, charged=True)

                    # coax terms for this r
                    adjacent = (r == k)
                    cx_total, cx_bonus = coax_pack(
                        seq, i, j, r, k, l, self.cfg, self.cfg.costs, adjacent
                    )
                    coax_total[t] = cx_total
                    coax_bonus[t] = cx_bonus

                cap_pen = short_hole_penalty(self.cfg.costs, k, l)

                cand, t_star, base_case = compose_vx_best_over_r(
                    Lu, Ru, Lc, Rc, coax_total, coax_bonus,
                    float(Gw), float(cap_pen), float(g)
                )

                if cand < best_c:
                    best_c = cand
                    r_star = k + t_star
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX,
                        outer=(i, j), hole=(k, l), split=r_star, charged=True
                    )

            re.vxc_matrix.set(i, j, best_c)
            if best_bp is not None:
                re.vx_back_ptr.set(i, j, best_bp)

    @staticmethod
    def _publish_vx(re: EddyRivasFoldState) -> None:
        for i, j in iter_spans(re.n):
            vxu = re.vxu_matrix.get(i, j)
            vxc = re.vxc_matrix.get(i, j)
            if vxu <= vxc:
                re.vx_matrix.set(i, j, vxu)
                re.vx_back_ptr.set(i, j, EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED))
            else:
                re.vx_matrix.set(i, j, vxc)


def load_costs_json(path: str) -> PseudoknotEnergies:
    with open(path, "r") as fh:
        d = json.load(fh)
    return costs_from_dict(d)

def save_costs_json(path: str, costs: PseudoknotEnergies) -> None:
    with open(path, "w") as fh:
        json.dump(costs_to_dict(costs), fh, indent=2, sort_keys=True)

def costs_from_dict(d: Dict) -> PseudoknotEnergies:
    """Create RERECosts from a flat dict; keys not present use dataclass defaults."""
    field_names = {f.name for f in fields(PseudoknotEnergies)}
    kwargs = {k: v for k, v in d.items() if k in field_names}
    return PseudoknotEnergies(**kwargs)

def costs_to_dict(costs: PseudoknotEnergies) -> Dict:
    """Round-trip exporter useful for saving tuned params."""
    return {f.name: getattr(costs, f.name) for f in fields(PseudoknotEnergies)}

def costs_from_vienna_like(tbl: Dict[str, Any]) -> PseudoknotEnergies:
    """
    Map a Vienna-like dict to RERECosts.
    Expected keys (suggested, adapt to your source):
      - 'q_ss', 'Gw', 'Gwi', 'Gwh', 'coax_scale', 'coax_bonus',
      - 'coax_pairs': { "GC|CG": -0.5, "AU|UA": -0.3, ... },
      - 'dangle_outer_left/R', 'dangle_hole_left/R': { "GA": -0.1, ... },
      - 'short_hole_caps': { "1": 2.0, "2": 1.0 },
      - optional: 'mismatch_coax_scale', 'mismatch_coax_bonus',
                  'coax_min_helix_len', 'coax_scale_oo/oi/io',
                  'P_tilde_out', 'P_tilde_hole', 'Q_tilde_out', 'Q_tilde_hole',
                  'M_tilde_yhx', 'M_tilde_vhx', 'M_tilde_whx'.
    """
    d = {}

    # simple scalars
    for k in [
        "q_ss","Gwh","Gwi","coax_scale","coax_bonus",
        "mismatch_coax_scale","mismatch_coax_bonus",
        "coax_min_helix_len","coax_scale_oo","coax_scale_oi","coax_scale_io",
        "P_tilde_out","P_tilde_hole","Q_tilde_out","Q_tilde_hole",
        "L_tilde", "R_tilde", "M_tilde_yhx","M_tilde_vhx","M_tilde_whx"
    ]:
        if k in tbl: d[k] = tbl[k]

    # coax pairs: accept "XY|UV" keys
    coax_pairs = {}
    for key, val in tbl.get("coax_pairs", {}).items():
        left,right = key.split("|")
        coax_pairs[(left, right)] = float(val)
    d["coax_pairs"] = coax_pairs

    # dangles: accept bigrams like "GA": value
    for name in ["dangle_outer_left","dangle_outer_right","dangle_hole_left","dangle_hole_right"]:
        m = {}
        for bigram, val in tbl.get(name, {}).items():
            if len(bigram) == 2:
                m[(bigram[0], bigram[1])] = float(val)
        d[name] = m

    # short-hole caps
    caps = {}
    for h, val in tbl.get("short_hole_caps", {}).items():
        caps[int(h)] = float(val)
    d["short_hole_caps"] = caps

    return costs_from_dict(d)

def quick_energy_harness(seq: str, cfg: EddyRivasFoldingConfig, nested: ZuckerFoldState, re: EddyRivasFoldState) -> Dict[str, float]:
    """
    Run fill_with_costs and report a few sentinel energies for regression:
    """
    eng = EddyRivasFoldingEngine(cfg)
    eng.fill_with_costs(seq, nested, re)
    out = {
        "W(0,n-1)": re.wx_matrix.get(0, re.n - 1),
        "V(0,n-1)": re.vx_matrix.get(0, re.n - 1),
    }
    # Add any other coordinates you want to track here.
    return out

