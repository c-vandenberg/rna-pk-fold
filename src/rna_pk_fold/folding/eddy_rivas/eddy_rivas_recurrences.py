from __future__ import annotations
import math
import json
from dataclasses import dataclass, field, fields
from typing import Tuple, Dict, Optional, Any, Callable

from rna_pk_fold.energies.energy_types import PseudoknotEnergies
from rna_pk_fold.folding.zucker.zucker_fold_state import ZuckerFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer, EddyRivasBacktrackOp
from rna_pk_fold.folding.eddy_rivas.is2_bridges import IS2_outer, IS2_outer_yhx
from rna_pk_fold.folding.eddy_rivas.iterators import (iter_spans, iter_holes, iter_complementary_tuples,
                                                      iter_inner_holes)
from rna_pk_fold.folding.eddy_rivas.matrix_accessors import (get_whx_with_collapse, get_zhx_with_collapse, wxI,
                                                             whx_collapse_with, zhx_collapse_with)
from rna_pk_fold.energies.energy_pk_ops import (dangle_hole_left, dangle_hole_right, dangle_outer_left,
                                                dangle_outer_right, coax_pack, short_hole_penalty)


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
    pk_penalty_gw: float = 1.0 # Gw: pseudoknot introduction penalty (kcal/mol)
    min_hole_width: int = 0  # 0 = identical behavior; 1+ prunes zero/narrow holes
    min_outer_left: int = 0  # minimal length of [i..r]
    min_outer_right: int = 0  # minimal length of [r+1..j]
    strict_complement_order: bool = True  # enforce i<k<=r<l<=j
    costs: Optional[PseudoknotEnergies] = None
    tables: object = None


# -----------------------
# Engine
# -----------------------
class EddyRivasFoldingEngine:
    def __init__(self, config: EddyRivasFoldingConfig):
        self.cfg = config

    def fill_with_costs(self, seq: str, nested: ZuckerFoldState, re: EddyRivasFoldState) -> None:
        n = re.n
        q = self.cfg.costs.q_ss
        Gw = self.cfg.pk_penalty_gw
        Gwh = getattr(self.cfg.costs, "Gwh", 0.0)
        Gwi = self.cfg.costs.Gwi
        Gwh_wx = (self.cfg.costs.Gwh_wx if self.cfg.costs.Gwh_wx != 0.0 else self.cfg.costs.Gwh)
        Gwh_whx = (self.cfg.costs.Gwh_whx if self.cfg.costs.Gwh_whx != 0.0 else self.cfg.costs.Gwh)
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

        self._seed_from_nested(nested, re)

        self._dp_whx(seq, re, q, Gwh_whx)
        self._dp_vhx(seq, re, q, Gwi, P_hole, L_, R_, Q_hole, M_vhx, M_whx)
        self._dp_zhx(seq, re, q, Gwi, P_hole, Q_hole)
        self._dp_yhx(seq, re, q, Gwi, P_out, Q_out, M_yhx, M_whx)

        self._compose_wx(seq, re, Gw, Gwh_wx)
        self._publish_wx(re)

        self._compose_vx(seq, re, Gw, g)
        self._publish_vx(re)

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
    def _dp_whx(self, seq: str, re: EddyRivasFoldState, q: float, Gwh_whx: float) -> None:
        for i, j in iter_spans(re.n):
            for k, l in iter_holes(i, j):
                best = math.inf
                best_bp: Optional[EddyRivasBackPointer] = None
                BP = make_bp(i, j, k, l)

                # 1) shrink-left: (k+1,l) + q
                v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k + 1, l)
                cand = v + q
                best, best_bp = take_best(
                    best, best_bp, cand,
                    lambda: EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT,
                        outer=(i, j),
                        hole=(k, l)
                    )
                )

                # 2) shrink-right: (k,l-1) + q
                v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k, l - 1)
                cand = v + q
                best, best_bp = take_best(
                    best, best_bp, cand,
                    lambda: EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_WHX_SHRINK_RIGHT,
                        outer=(i, j),
                        hole=(k, l)
                    )
                )

                # 3) trim outer-left: (i+1,j:k,l) + q
                v = re.whx_matrix.get(i + 1, j, k, l)
                cand = v + q
                best, best_bp = take_best(
                    best, best_bp, cand,
                    lambda: EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_WHX_TRIM_LEFT,
                        outer=(i, j),
                        hole=(k, l)
                    )
                )

                # 4) trim outer-right: (i,j-1:k,l) + q
                v = re.whx_matrix.get(i, j - 1, k, l)
                cand = v + q
                best, best_bp = take_best(
                    best, best_bp, cand,
                    lambda: EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_WHX_TRIM_RIGHT,
                        outer=(i, j),
                        hole=(k, l)
                    )
                )

                # 5) direct collapse
                v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k, l)
                best, best_bp = take_best(
                    best, best_bp, v,
                    lambda: EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE,
                        outer=(i, j),
                        hole=(k, l)
                    )
                )

                # SS both sides
                v = re.whx_matrix.get(i + 1, j - 1, k, l)
                if math.isfinite(v):
                    cand = v + 2.0 * q
                    best, best_bp = take_best(
                        best, best_bp, cand,
                        lambda: EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_WHX_SS_BOTH,
                            outer=(i, j),
                            hole=(k, l)
                        )
                    )

                # non-nested outer splits with WX
                for r in range(i, j):
                    left = re.whx_matrix.get(i, r, k, l)
                    right = wxI(re, r + 1, j)
                    if math.isfinite(left) and math.isfinite(right):
                        cand = left + right
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX,
                                outer=(i, j),
                                hole=(k, l),
                                split=r
                            )
                        )

                for s2 in range(i, j):
                    left = wxI(re, i, s2)
                    right = re.whx_matrix.get(s2 + 1, j, k, l)
                    if math.isfinite(left) and math.isfinite(right):
                        cand = left + right
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_WHX_SPLIT_RIGHT_WX_WHX,
                                outer=(i, j),
                                hole=(k, l),
                                split=s2
                            )
                        )

                # overlapping-PK split with penalty
                if Gwh_whx != 0.0:
                    for r in range(i, j):
                        left = re.whx_matrix.get(i, r, k, l)
                        right = re.whx_matrix.get(r + 1, j, k, l)
                        if math.isfinite(left) and math.isfinite(right):
                            cand = Gwh_whx + left + right
                            best, best_bp = take_best(
                                best, best_bp, cand,
                                lambda: EddyRivasBackPointer(
                                    op=EddyRivasBacktrackOp.RE_WHX_OVERLAP_SPLIT,
                                    outer=(i, j),
                                    hole=(k, l),
                                    split=r
                                )
                            )

                # IS2 (outer_yhx) + yhx(r2,s2:k,l)
                for r2 in range(i, k + 1):
                    for s2 in range(l, j + 1):
                        if r2 <= k and l <= s2 and r2 <= s2:
                            inner_y = re.yhx_matrix.get(r2, s2, k, l)
                            if math.isfinite(inner_y):
                                bridge = IS2_outer_yhx(self.cfg, seq, i, j, r2, s2)
                                cand = bridge + inner_y
                                best, best_bp = take_best(
                                    best, best_bp, cand,
                                    lambda: EddyRivasBackPointer(
                                        op=EddyRivasBacktrackOp.RE_WHX_IS2_INNER_YHX,
                                        outer=(i, j),
                                        hole=(k, l),
                                        bridge=(r2, s2)
                                    )
                                )

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
    ) -> None:
        for i, j in iter_spans(re.n):
            max_h = max(0, j - i - 1)
            for h in range(1, max_h + 1):
                for k in range(i, j - h):
                    l = k + h + 1
                    best = re.vhx_matrix.get(i, j, k, l)
                    best_bp: Optional[EddyRivasBackPointer] = None

                    # DANGLES
                    v = re.vhx_matrix.get(i, j, k + 1, l)
                    cand = P_hole + L_ + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_DANGLE_L,
                            outer=(i, j),
                            hole=(k, l)
                        )

                    v = re.vhx_matrix.get(i, j, k, l - 1)
                    cand = P_hole + R_ + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_DANGLE_R,
                            outer=(i, j),
                            hole=(k, l)
                        )

                    v = re.vhx_matrix.get(i, j, k + 1, l - 1)
                    cand = P_hole + L_ + R_ + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_DANGLE_LR,
                            outer=(i, j),
                            hole=(k, l)
                        )

                    # SS from ZHX
                    v_zhx = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, k, l)
                    cand = Q_hole + v_zhx
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_SS_LEFT,
                            outer=(i, j),
                            hole=(k, l)
                        )
                    elif (cand == best and isinstance(best_bp, EddyRivasBackPointer) and
                          best_bp.op in (EddyRivasBacktrackOp.RE_VHX_SS_LEFT,
                                         EddyRivasBacktrackOp.RE_VHX_SS_RIGHT)):
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_SS_RIGHT,
                            outer=(i, j),
                            hole=(k, l)
                        )

                    # SPLIT LEFT
                    for r in range(i, k):
                        left = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, r, l)
                        right = wxI(re, r + 1, k)
                        cand = left + right
                        if cand < best:
                            best = cand
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_VHX_SPLIT_LEFT_ZHX_WX,
                                outer=(i, j),
                                hole=(k, l),
                                split=r
                            )

                    # SPLIT RIGHT
                    for s2 in range(l + 1, j + 1):
                        left = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, k, s2)
                        right = wxI(re, l, s2 - 1)
                        cand = left + right
                        if cand < best:
                            best = cand
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_VHX_SPLIT_RIGHT_ZHX_WX,
                                outer=(i, j),
                                hole=(k, l),
                                split=s2
                            )

                    # IS2 + zhx(r,s2:k,l)
                    for r in range(i, k + 1):
                        for s2 in range(l, j + 1):
                            if r <= k and l <= s2 and r <= s2:
                                inner = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, r, s2, k, l)
                                cand = IS2_outer(seq, self.cfg.tables, i, j, r, s2) + inner
                                if cand < best:
                                    best = cand
                                    best_bp = EddyRivasBackPointer(
                                        op=EddyRivasBacktrackOp.RE_VHX_IS2_INNER_ZHX,
                                        outer=(i, j),
                                        hole=(k, l),
                                        bridge=(r, s2)
                                    )

                    # CLOSE_BOTH
                    close = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i + 1, j - 1, k - 1, l + 1)
                    if math.isfinite(close):
                        cand = 2.0 * P_hole + M_vhx + close + Gwi + M_whx
                        if cand < best:
                            best = cand
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_VHX_CLOSE_BOTH,
                                outer=(i, j),
                                hole=(k, l)
                            )

                    # WRAP via WHX
                    wrap = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i + 1, j - 1, k, l)
                    cand = P_hole + M_vhx + wrap + Gwi + M_whx
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_WRAP_WHX,
                            outer=(i, j),
                            hole=(k, l)
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
    ) -> None:
        for i, j in iter_spans(re.n):
            for k, l in iter_holes(i, j):
                best = math.inf
                best_bp: Optional[EddyRivasBackPointer] = None

                # FROM_VHX
                v = re.vhx_matrix.get(i, j, k, l)
                if math.isfinite(v):
                    cand = P_hole + v + Gwi
                    best, best_bp = take_best(
                        best, best_bp, cand,
                        lambda: EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_FROM_VHX,
                            outer=(i, j),
                            hole=(k, l)
                        )
                    )

                # DANGLE_LR from VHX
                v = re.vhx_matrix.get(i, j, k - 1, l + 1)
                if math.isfinite(v):
                    Lh = dangle_hole_left(seq, k, self.cfg.costs)
                    Rh = dangle_hole_right(seq, l, self.cfg.costs)
                    cand = Lh + Rh + P_hole + v + Gwi
                    best, best_bp = take_best(
                        best, best_bp, cand,
                        lambda: EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_DANGLE_LR,
                            outer=(i, j),
                            hole=(k, l)
                        )
                    )

                # DANGLE_R from VHX
                v = re.vhx_matrix.get(i, j, k - 1, l)
                if math.isfinite(v):
                    Rh = dangle_hole_right(seq, l - 1, self.cfg.costs)
                    cand = Rh + P_hole + v + Gwi
                    best, best_bp = take_best(
                        best, best_bp, cand,
                        lambda: EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_DANGLE_R,
                            outer=(i, j),
                            hole=(k, l)
                        )
                    )

                # DANGLE_L from VHX
                v = re.vhx_matrix.get(i, j, k, l + 1)
                if math.isfinite(v):
                    Lh = dangle_hole_left(seq, k + 1, self.cfg.costs)
                    cand = Lh + P_hole + v + Gwi
                    best, best_bp = take_best(
                        best, best_bp, cand,
                        lambda: EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_DANGLE_L,
                            outer=(i, j),
                            hole=(k, l)
                        )
                    )

                # SS_LEFT
                v = re.zhx_matrix.get(i, j, k - 1, l)
                if math.isfinite(v):
                    cand = Q_hole + v
                    best, best_bp = take_best(
                        best, best_bp, cand,
                        lambda: EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SS_LEFT,
                            outer=(i, j),
                            hole=(k, l)
                        )
                    )

                # SS_RIGHT (flip on tie)
                v = re.zhx_matrix.get(i, j, k, l + 1)
                if math.isfinite(v):
                    cand = Q_hole + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT,
                            outer=(i, j),
                            hole=(k, l)
                        )
                    elif (cand == best and isinstance(best_bp, EddyRivasBackPointer) and
                          best_bp.op in (EddyRivasBacktrackOp.RE_ZHX_SS_LEFT,
                                         EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT)):
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT,
                            outer=(i, j),
                            hole=(k, l)
                        )

                # SPLITS
                for r in range(i, k):
                    left = re.zhx_matrix.get(i, j, r, l)
                    right = wxI(re, r + 1, k)
                    if math.isfinite(left) and math.isfinite(right):
                        cand = left + right
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_ZHX_SPLIT_LEFT_ZHX_WX,
                                outer=(i, j),
                                hole=(k, l),
                                split=r
                            )
                        )

                for s2 in range(l + 1, j + 1):
                    left = re.zhx_matrix.get(i, j, k, s2)
                    right = wxI(re, l, s2 - 1)
                    if math.isfinite(left) and math.isfinite(right):
                        cand = left + right
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_ZHX_SPLIT_RIGHT_ZHX_WX,
                                outer=(i, j),
                                hole=(k, l),
                                split=s2
                            )
                        )

                # IS2 + vhx(r,s2:k,l)
                for r in range(i, k + 1):
                    for s2 in range(l, j + 1):
                        if r <= s2:
                            inner = re.vhx_matrix.get(r, s2, k, l)
                            if math.isfinite(inner):
                                bridge = IS2_outer(seq, self.cfg.tables, i, j, r, s2)
                                cand = bridge + inner
                                best, best_bp = take_best(
                                    best, best_bp, cand,
                                    lambda: EddyRivasBackPointer(
                                        op=EddyRivasBacktrackOp.RE_ZHX_IS2_INNER_VHX,
                                        outer=(i, j),
                                        hole=(k, l),
                                        bridge=(r, s2)
                                    )
                                )

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
    ) -> None:
        for i, j in iter_spans(re.n):
            max_h = max(0, j - i - 1)
            for h in range(1, max_h + 1):
                for k in range(i, j - h):
                    l = k + h + 1
                    best = math.inf
                    best_bp: Optional[EddyRivasBackPointer] = None

                    # Outer dangle L
                    v = re.vhx_matrix.get(i + 1, j, k, l)
                    if math.isfinite(v):
                        Lo = dangle_outer_left(seq, i, self.cfg.costs)
                        cand = Lo + P_out + v + Gwi
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_DANGLE_L,
                                outer=(i, j),
                                hole=(k, l)
                            )
                        )

                    # Outer dangle R
                    v = re.vhx_matrix.get(i, j - 1, k, l)
                    if math.isfinite(v):
                        Ro = dangle_outer_right(seq, j, self.cfg.costs)
                        cand = Ro + P_out + v + Gwi
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_DANGLE_R,
                                outer=(i, j),
                                hole=(k, l)
                            )
                        )

                    # Outer dangle LR
                    v = re.vhx_matrix.get(i + 1, j - 1, k, l)
                    if math.isfinite(v):
                        Lo = dangle_outer_left(seq, i, self.cfg.costs)
                        Ro = dangle_outer_right(seq, j, self.cfg.costs)
                        cand = Lo + Ro + P_out + v + Gwi
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_DANGLE_LR,
                                outer=(i, j),
                                hole=(k, l)
                            )
                        )

                    # SS trims: Left
                    v = re.yhx_matrix.get(i + 1, j, k, l)
                    if math.isfinite(v):
                        cand = Q_out + v
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_SS_LEFT,
                                outer=(i, j),
                                hole=(k, l)
                            )
                        )

                    # SS trims: Right (flip on tie)
                    v = re.yhx_matrix.get(i, j - 1, k, l)
                    if math.isfinite(v):
                        cand = Q_out + v
                        if cand < best:
                            best = cand
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_SS_RIGHT,
                                outer=(i, j),
                                hole=(k, l)
                            )
                        elif (cand == best and isinstance(best_bp, EddyRivasBackPointer) and
                              best_bp.op in (EddyRivasBacktrackOp.RE_YHX_SS_LEFT,
                                             EddyRivasBacktrackOp.RE_YHX_SS_RIGHT)):
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_SS_RIGHT,
                                outer=(i, j),
                                hole=(k, l)
                            )

                    # SS both sides
                    v = re.yhx_matrix.get(i + 1, j - 1, k, l)
                    if math.isfinite(v):
                        cand = 2.0 * Q_out + v
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_SS_BOTH,
                                outer=(i, j),
                                hole=(k, l)
                            )
                        )

                    # Wrap via WHX(i,j:k-1,l+1)
                    v = re.whx_matrix.get(i, j, k - 1, l + 1)
                    if math.isfinite(v):
                        cand = P_out + M_yhx + M_whx + v + Gwi
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX,
                                outer=(i, j),
                                hole=(k, l)
                            )
                        )

                    # Wrap + outer dangles
                    v = re.whx_matrix.get(i + 1, j, k - 1, l + 1)
                    if math.isfinite(v):
                        Lo = dangle_outer_left(seq, i, self.cfg.costs)
                        cand = Lo + P_out + M_yhx + M_whx + v + Gwi
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_L,
                                outer=(i, j),
                                hole=(k, l)
                            )
                        )

                    v = re.whx_matrix.get(i, j - 1, k - 1, l + 1)
                    if math.isfinite(v):
                        Ro = dangle_outer_right(seq, j, self.cfg.costs)
                        cand = Ro + P_out + M_yhx + M_whx + v + Gwi
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_R,
                                outer=(i, j),
                                hole=(k, l)
                            )
                        )

                    v = re.whx_matrix.get(i + 1, j - 1, k - 1, l + 1)
                    if math.isfinite(v):
                        Lo = dangle_outer_left(seq, i, self.cfg.costs)
                        Ro = dangle_outer_right(seq, j, self.cfg.costs)
                        cand = Lo + Ro + P_out + M_yhx + M_whx + v + Gwi
                        best, best_bp = take_best(
                            best, best_bp, cand,
                            lambda: EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR,
                                outer=(i, j),
                                hole=(k, l)
                            )
                        )

                    # Outer splits with WX
                    for r in range(i, j):
                        left = re.yhx_matrix.get(i, r, k, l)
                        right = wxI(re, r + 1, j)
                        if math.isfinite(left) and math.isfinite(right):
                            cand = left + right
                            best, best_bp = take_best(
                                best, best_bp, cand,
                                lambda: EddyRivasBackPointer(
                                    op=EddyRivasBacktrackOp.RE_YHX_SPLIT_LEFT_YHX_WX,
                                    outer=(i, j),
                                    hole=(k, l),
                                    split=r
                                )
                            )

                    for s2 in range(i, j):
                        left = wxI(re, i, s2)
                        right = re.yhx_matrix.get(s2 + 1, j, k, l)
                        if math.isfinite(left) and math.isfinite(right):
                            cand = left + right
                            best, best_bp = take_best(
                                best, best_bp, cand,
                                lambda: EddyRivasBackPointer(
                                    op=EddyRivasBacktrackOp.RE_YHX_SPLIT_RIGHT_WX_YHX,
                                    outer=(i, j),
                                    hole=(k, l),
                                    split=s2
                                )
                            )

                    # IS2 for YHX + WHX(r2,s2:k,l)
                    for r2 in range(i, k + 1):
                        for s2 in range(l, j + 1):
                            if r2 <= s2:
                                inner_w = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, r2, s2, k, l)
                                if math.isfinite(inner_w):
                                    bridge = IS2_outer_yhx(self.cfg, seq, i, j, r2, s2)
                                    cand = bridge + inner_w
                                    best, best_bp = take_best(
                                        best, best_bp, cand,
                                        lambda: EddyRivasBackPointer(
                                            op=EddyRivasBacktrackOp.RE_YHX_IS2_INNER_WHX,
                                            outer=(i, j),
                                            hole=(k, l),
                                            bridge=(r2, s2)
                                        )
                                    )

                    re.yhx_matrix.set(i, j, k, l, best)
                    re.yhx_back_ptr.set(i, j, k, l, best_bp)

    # --------- WX Composition & Publish ---------
    def _compose_wx(self, seq: str, re: EddyRivasFoldState, Gw: float, Gwh_wx: float) -> None:
        for i, j in iter_spans(re.n):
            best_c = re.wxc_matrix.get(i, j)
            best_bp: Optional[EddyRivasBackPointer] = None

            for (r, k, l) in iter_complementary_tuples(i, j):
                if self.cfg.strict_complement_order and not (i < k <= r < l <= j):
                    continue
                if (l - k - 1) < self.cfg.min_hole_width:
                    continue
                if (r - i) < self.cfg.min_outer_left or (j - (r + 1)) < self.cfg.min_outer_right:
                    continue

                L_u = whx_collapse_with(re, i, r, k, l, charged=False)
                R_u = whx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=False)
                L_c = whx_collapse_with(re, i, r, k, l, charged=True)
                R_c = whx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=True)

                cap_pen = short_hole_penalty(self.cfg.costs, k, l)

                cand_first = Gw + L_u + R_u + cap_pen
                cand_Lc = L_c + R_u + cap_pen
                cand_Rc = L_u + R_c + cap_pen
                cand_both = L_c + R_c + cap_pen

                cand = min(cand_first, cand_Lc, cand_Rc, cand_both)
                if cand < best_c:
                    best_c = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX,
                        outer=(i, j),
                        hole=(k, l),
                        split=r,
                        charged=True
                    )

                # yhx + yhx
                left_y = re.yhx_matrix.get(i, r, k, l)
                right_y = re.yhx_matrix.get(k + 1, j, l - 1, r + 1)
                if math.isfinite(left_y) and math.isfinite(right_y):
                    cand_y = Gw + left_y + right_y + cap_pen
                    if cand_y < best_c:
                        best_c = cand_y
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX,
                            outer=(i, j),
                            hole=(k, l),
                            split=r,
                            charged=True
                        )

                # mix: yhx + whx
                left_y = re.yhx_matrix.get(i, r, k, l)
                if math.isfinite(left_y):
                    R_u2 = whx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=False)
                    R_c2 = whx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=True)
                    if math.isfinite(R_u2):
                        cand2 = Gw + left_y + R_u2 + cap_pen
                        if cand2 < best_c:
                            best_c = cand2
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX,
                                outer=(i, j),
                                hole=(k, l),
                                split=r,
                                charged=True
                            )
                    if math.isfinite(R_c2):
                        cand2 = left_y + R_c2 + cap_pen
                        if cand2 < best_c:
                            best_c = cand2
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX,
                                outer=(i, j),
                                hole=(k, l),
                                split=r,
                                charged=True
                            )

                # mix: whx + yhx
                right_y = re.yhx_matrix.get(k + 1, j, l - 1, r + 1)
                if math.isfinite(right_y):
                    L_u2 = whx_collapse_with(re, i, r, k, l, charged=False)
                    L_c2 = whx_collapse_with(re, i, r, k, l, charged=True)
                    if math.isfinite(L_u2):
                        cand2 = Gw + right_y + L_u2 + cap_pen
                        if cand2 < best_c:
                            best_c = cand2
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX,
                                outer=(i, j),
                                hole=(k, l),
                                split=r,
                                charged=True
                            )
                    if math.isfinite(L_c2):
                        cand2 = L_c2 + right_y + cap_pen
                        if cand2 < best_c:
                            best_c = cand2
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX,
                                outer=(i, j),
                                hole=(k, l),
                                split=r,
                                charged=True
                            )

                # optional overlap via YHX+YHX
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
                                        outer=(i, j),
                                        hole=(k2, l2),
                                        split=r2,
                                        charged=True
                                    )

                # optional drift at the join
                if self.cfg.enable_join_drift and self.cfg.drift_radius > 0:
                    for d in range(1, self.cfg.drift_radius + 1):
                        kR = (l - 1) - d
                        lR = (r + 1) + d
                        if not (kR < lR):
                            continue
                        if (lR - kR - 1) < self.cfg.min_hole_width:
                            continue

                        R_u_d = whx_collapse_with(re, k + 1, j, kR, lR, charged=False)
                        R_c_d = whx_collapse_with(re, k + 1, j, kR, lR, charged=True)
                        drift_pen = d * (self.cfg.costs.join_drift_penalty or self.cfg.costs.q_ss)
                        if math.isfinite(R_u_d) or math.isfinite(R_c_d):
                            cand_first_d = Gw + whx_collapse_with(re, i, r, k, l, False) + (
                                R_u_d if math.isfinite(R_u_d) else math.inf) + cap_pen + drift_pen
                            cand_Lc_d = whx_collapse_with(re, i, r, k, l, True) + (
                                R_u_d if math.isfinite(R_u_d) else math.inf) + cap_pen + drift_pen
                            cand_Rc_d = whx_collapse_with(re, i, r, k, l, False) + (
                                R_c_d if math.isfinite(R_c_d) else math.inf) + cap_pen + drift_pen
                            cand_both_d = whx_collapse_with(re, i, r, k, l, True) + (
                                R_c_d if math.isfinite(R_c_d) else math.inf) + cap_pen + drift_pen

                            cand_d = min(cand_first_d, cand_Lc_d, cand_Rc_d, cand_both_d)
                            if cand_d < best_c:
                                best_c = cand_d
                                best_bp = EddyRivasBackPointer(
                                    op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_DRIFT,
                                    outer=(i, j),
                                    hole=(k, l),
                                    split=r,
                                    drift=d,
                                    charged=True
                                )

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

    def _compose_vx(self, seq: str, re: EddyRivasFoldState, Gw: float, g: float) -> None:
        for i, j in iter_spans(re.n):
            best_c = re.vxc_matrix.get(i, j)
            best_bp: Optional[EddyRivasBackPointer] = None

            for (r, k, l) in iter_complementary_tuples(i, j):
                if self.cfg.strict_complement_order and not (i < k <= r < l <= j):
                    continue
                if (l - k - 1) < self.cfg.min_hole_width:
                    continue
                if (r - i) < self.cfg.min_outer_left or (j - (r + 1)) < self.cfg.min_outer_right:
                    continue

                L_u = zhx_collapse_with(re, i, r, k, l, charged=False)
                R_u = zhx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=False)
                L_c = zhx_collapse_with(re, i, r, k, l, charged=True)
                R_c = zhx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=True)

                adjacent = (r == k)
                cap_pen = short_hole_penalty(self.cfg.costs, k, l)

                coax_total, coax_bonus_term = coax_pack(
                    seq, i, j, r, k, l, self.cfg, self.cfg.costs, adjacent
                )

                cand_first = Gw + L_u + R_u + cap_pen
                cand_Lc = L_c + R_u + cap_pen
                cand_Rc = L_u + R_c + cap_pen
                cand_both = L_c + R_c + cap_pen

                cand = min(cand_first, cand_Lc, cand_Rc, cand_both) + g * coax_total + coax_bonus_term
                if cand < best_c:
                    best_c = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX,
                        outer=(i, j),
                        hole=(k, l),
                        split=r,
                        charged=True
                    )

                # Optional drift (VX/ZHX)
                if self.cfg.enable_join_drift and self.cfg.drift_radius > 0:
                    for d in range(1, self.cfg.drift_radius + 1):
                        kR = (l - 1) - d
                        lR = (r + 1) + d
                        iR = k + 1

                        if not (iR <= kR < lR <= j):
                            continue
                        if (lR - kR - 1) < self.cfg.min_hole_width:
                            continue

                        R_u_d = zhx_collapse_with(re, iR, j, kR, lR, charged=False)
                        R_c_d = zhx_collapse_with(re, iR, j, kR, lR, charged=True)
                        if not (math.isfinite(R_u_d) or math.isfinite(R_c_d)):
                            continue

                        L_u_base = zhx_collapse_with(re, i, r, k, l, charged=False)
                        L_c_base = zhx_collapse_with(re, i, r, k, l, charged=True)

                        drift_pen = d * (self.cfg.costs.join_drift_penalty or self.cfg.costs.q_ss)

                        cand_first_d = (Gw + L_u_base + (R_u_d if math.isfinite(R_u_d) else math.inf) + drift_pen
                                        + cap_pen)
                        cand_Lc_d = (L_c_base + (R_u_d if math.isfinite(R_u_d) else math.inf) + drift_pen
                                     + cap_pen)
                        cand_Rc_d = (L_u_base + (R_c_d if math.isfinite(R_c_d) else math.inf) + drift_pen
                                     + cap_pen)
                        cand_both_d = (L_c_base + (R_c_d if math.isfinite(R_c_d) else math.inf) + drift_pen
                                       + cap_pen)

                        cand_d = min(cand_first_d, cand_Lc_d, cand_Rc_d, cand_both_d)
                        if cand_d < best_c:
                            best_c = cand_d
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT,
                                outer=(i, j),
                                hole=(k, l),
                                split=r,
                                drift=d,
                                charged=True
                            )

            # drift-only fallback
            if self.cfg.enable_join_drift and self.cfg.drift_radius > 0:
                vxu_ij = re.vxu_matrix.get(i, j)
                improved_bp: Optional[EddyRivasBackPointer] = None
                improved_val = best_c

                for (r, k, l) in iter_complementary_tuples(i, j):
                    for d in range(1, self.cfg.drift_radius + 1):
                        r2 = r + d
                        if i < k <= r2 < l <= j:
                            drift_pen = d * (self.cfg.costs.join_drift_penalty or self.cfg.costs.q_ss)
                            cand = drift_pen
                            if cand < improved_val:
                                improved_val = cand
                                improved_bp = EddyRivasBackPointer(
                                    op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX_DRIFT,
                                    outer=(i, j),
                                    hole=(k, l),
                                    split=r,
                                    drift=d,
                                    charged=True
                                )

                if improved_bp is not None and improved_val < best_c:
                    best_c = improved_val
                    best_bp = improved_bp

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
        "M_tilde_yhx","M_tilde_vhx","M_tilde_whx"
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
