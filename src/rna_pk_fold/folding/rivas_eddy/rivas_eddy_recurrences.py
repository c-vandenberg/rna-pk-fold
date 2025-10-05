from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Iterator, Tuple, Dict, Optional

from rna_pk_fold.folding.fold_state import FoldState, RivasEddyState
from rna_pk_fold.folding.rivas_eddy.rivas_eddy_matrices import (
    get_whx_with_collapse,
    get_zhx_with_collapse
)
from rna_pk_fold.utils.nucleotide_utils import dimer_key

RE_BP_SHRINK_LEFT  = "RE_SHRINK_LEFT"
RE_BP_SHRINK_RIGHT = "RE_SHRINK_RIGHT"
RE_BP_TRIM_LEFT    = "RE_TRIM_LEFT"
RE_BP_TRIM_RIGHT   = "RE_TRIM_RIGHT"
RE_BP_COLLAPSE     = "RE_COLLAPSE"
RE_BP_COMPOSE_WX   = "RE_PK_COMPOSE_WX"
RE_BP_COMPOSE_VX   = "RE_PK_COMPOSE_VX"
RE_BP_COMPOSE_WX_YHX = "RE_PK_COMPOSE_WX_YHX"
RE_BP_VHX_CLOSE_BOTH = "RE_VHX_CLOSE_BOTH"      # 2*P~ + M~ + whx(i+1,j-1:k-1,l+1)
RE_BP_ZHX_FROM_VHX   = "RE_ZHX_FROM_VHX"        # P~ + vhx(i,j:k,l)
RE_BP_ZHX_DANGLE_LR  = "RE_ZHX_DANGLE_LR"       # L~+R~+P~ + vhx(i,j:k-1,l+1)
RE_BP_ZHX_DANGLE_L   = "RE_ZHX_DANGLE_L"        # L~+P~     + vhx(i,j:k,l+1)
RE_BP_ZHX_DANGLE_R   = "RE_ZHX_DANGLE_R"        # R~+P~     + vhx(i,j:k-1,l)
RE_BP_ZHX_SS_LEFT    = "RE_ZHX_SS_LEFT"         # Q~ + zhx(i,j:k-1,l)
RE_BP_ZHX_SS_RIGHT   = "RE_ZHX_SS_RIGHT"        # Q~ + zhx(i,j:k,l+1)
RE_BP_YHX_DANGLE_L   = "RE_YHX_DANGLE_L"        # L~+P~ + vhx(i+1,j:k,l)
RE_BP_YHX_DANGLE_R   = "RE_YHX_DANGLE_R"        # R~+P~ + vhx(i, j-1:k,l)
RE_BP_YHX_SS_LEFT    = "RE_YHX_SS_LEFT"         # Q~ + yhx(i+1,j:k,l)
RE_BP_YHX_SS_RIGHT   = "RE_YHX_SS_RIGHT"        # Q~ + yhx(i, j-1:k,l)
RE_BP_YHX_WRAP_WHX   = "RE_YHX_WRAP_WHX"        # P~ + M~ + whx(i,j:k-1,l+1)"


# ---------- Table/evaluator helpers (DANGLES + COAX) ----------
def _safe_base(seq: str, idx: int) -> Optional[str]:
    return seq[idx] if 0 <= idx < len(seq) else None

def _pair_key(seq: str, a: int, b: int) -> Optional[str]:
    ba, bb = _safe_base(seq, a), _safe_base(seq, b)
    if ba is None or bb is None:
        return None
    return dimer_key(ba, bb)  # e.g. "GC", "AU", "GU", ...

def _table_lookup(tbl: Dict[Tuple[str, str], float], x: Optional[str], y: Optional[str], default: float) -> float:
    if x is None or y is None:
        return 0.0  # no dangle if missing context
    return tbl.get((x, y), default)

def _dangle_hole_L(seq: str, k: int, costs: "RERECosts") -> float:
    # left hole dangle uses (base at k-1, base at k)
    return _table_lookup(costs.dangle_hole_L, _safe_base(seq, k - 1), _safe_base(seq, k), costs.L_tilde)

def _dangle_hole_R(seq: str, l: int, costs: "RERECosts") -> float:
    # right hole dangle uses (base at l, base at l+1)
    return _table_lookup(costs.dangle_hole_R, _safe_base(seq, l), _safe_base(seq, l + 1), costs.R_tilde)

def _dangle_outer_L(seq: str, i: int, costs: "RERECosts") -> float:
    # outer-left dangle uses (base at i, base at i+1)
    return _table_lookup(costs.dangle_outer_L, _safe_base(seq, i), _safe_base(seq, i + 1), costs.L_tilde)

def _dangle_outer_R(seq: str, j: int, costs: "RERECosts") -> float:
    # outer-right dangle uses (base at j-1, base at j)
    return _table_lookup(costs.dangle_outer_R, _safe_base(seq, j - 1), _safe_base(seq, j), costs.R_tilde)

def _coax_energy_for_join(seq: str, left_pair: Tuple[int, int], right_pair: Tuple[int, int], costs: "RERECosts") -> float:
    # Join two adjacent helices; energy is looked up by their pair-types.
    lp = _pair_key(seq, *left_pair)
    rp = _pair_key(seq, *right_pair)
    if lp is None or rp is None:
        return 0.0
    return costs.coax_pairs.get((lp, rp), 0.0)  # table value, typically negative if stabilizing


@dataclass(slots=True)
class RERECosts:
    # Per-step single-strand “gap” cost (move a boundary by 1, or trim outer by 1)

    # Tilde params fallback (used if tables don’t have an entry)
    q_ss: float = 0.2 # Per-step single-strand “gap” cost (move a boundary by 1, or trim outer by 1)
    P_tilde: float = 1.0  # pair score used in gap recurrences (acts like P~)
    Q_tilde: float = 0.2  # single-stranded nucleotide in gap (acts like Q~)
    L_tilde: float = 0.0  # 5′ dangle on gap edge (acts like L~)
    R_tilde: float = 0.0  # 3′ dangle on gap edge (acts like R~)
    M_tilde: float = 0.0  # multiloop-ish term inside vhx/yhx closures (acts like M~)

    # Dangle/coax tables (keyed on local bases or pair types)
    dangle_hole_L: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (k-1, k)
    dangle_hole_R: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (l, l+1)
    dangle_outer_L: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (i, i+1)
    dangle_outer_R: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (j-1, j)
    coax_pairs: Dict[Tuple[str, str], float] = field(default_factory=dict)  # ((pairTypeA),(pairTypeB))

    # Optional bonuses/penalties
    coax_bonus: float = 0.0  # used in your vx path; leave 0.0 unless you want it
    Gwh: float = 0.0  # penalty for overlapping PKs (not used yet)


@dataclass(slots=True)
class REREConfig:
    enable_coax: bool = False        # keep off initially
    pk_penalty_gw: float = 1.0       # Gw: pseudoknot introduction penalty (kcal/mol)
    costs: RERECosts = RERECosts()


class RivasEddyEngine:
    """
    Minimal R&E filler:
      - seeds wx/vx from nested W/V,
      - makes whx finite via zero-cost hole-shrink recurrences,
      - adds a two-gap (whx+whx) composition term to wx.
    """
    def __init__(self, config: REREConfig):
        self.cfg = config

    def fill_minimal(self, seq: str, nested: FoldState, re: RivasEddyState) -> None:
        n = re.n

        # --- 0. Seed non-gap from nested ---
        for s in range(0, n):
            for i in range(0, n - s):
                j = i + s
                # Baselines copied from your nested DP (use what you already computed)
                w_base = nested.w_matrix.get(i, j)
                v_base = nested.v_matrix.get(i, j)
                re.wx_matrix.set(i, j, w_base)
                re.vx_matrix.set(i, j, v_base)

        # --- 1.1 `whx` zero-cost hole-shrink (collapse to wx)) ---
        # Order: outer span s increasing; for each (i,j) enumerate holes by increasing width h.
        for s in range(0, n):
            for i in range(0, n - s):
                j = i + s
                # hole width h = l - k - 1  (interior length)
                max_h = max(0, j - i - 1)
                for h in range(1, max_h + 1):                 # only non-collapsed
                    for k in range(i, j - h):
                        l = k + h + 1
                        # whx(i,j:k,l) can be reached from:
                        #  - shrink left hole boundary (k+1,l),
                        #  - shrink right hole boundary (k,l-1),
                        #  - trim outer left (i+1,j),
                        #  - trim outer right (i,j-1),
                        # or collapse identity if h==0 (handled by accessor).
                        candidates = []

                        # collapse not applicable here (h>=1), but neighbors might be finite
                        # NOTE: all moves are zero-cost for Step 12 minimal slice
                        # shrink hole (use collapse-aware accessor)
                        val = get_whx_with_collapse(re.whx_matrix, re.wx_matrix, i, j, k + 1, l)
                        if math.isfinite(val):
                            candidates.append(val)
                        val = get_whx_with_collapse(re.whx_matrix, re.wx_matrix, i, j, k, l - 1)
                        if math.isfinite(val):
                            candidates.append(val)
                        # trim outer (smaller span)
                        val = re.whx_matrix.get(i + 1, j, k, l)
                        if math.isfinite(val):
                            candidates.append(val)
                        val = re.whx_matrix.get(i, j - 1, k, l)
                        if math.isfinite(val):
                            candidates.append(val)

                        # Always allow fallback via collapse by jumping to accessor
                        # (gives a finite anchor path even if neighbors are +inf):
                        candidates.append(get_whx_with_collapse(re.whx_matrix, re.wx_matrix, i, j, k, l))

                        best = min(candidates) if candidates else math.inf
                        re.whx_matrix.set(i, j, k, l, best)

        # --- 1.2 `zhx` zero-cost hole-shrink (collapse to vx) ---
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                max_h = max(0, j - i - 1)
                for h in range(1, max_h + 1):
                    for k in range(i, j - h):
                        l = k + h + 1
                        candidates = []
                        # shrink hole (collapse-aware)
                        v = get_zhx_with_collapse(re.zhx_matrix, re.vx_matrix, i, j, k + 1, l)
                        if math.isfinite(v): candidates.append(v)
                        v = get_zhx_with_collapse(re.zhx_matrix, re.vx_matrix, i, j, k, l - 1)
                        if math.isfinite(v): candidates.append(v)
                        # trim outer (smaller span)
                        v = re.zhx_matrix.get(i + 1, j, k, l)
                        if math.isfinite(v): candidates.append(v)
                        v = re.zhx_matrix.get(i, j - 1, k, l)
                        if math.isfinite(v): candidates.append(v)
                        # direct collapse allowed too
                        candidates.append(get_zhx_with_collapse(re.zhx_matrix, re.vx_matrix, i, j, k, l))
                        best = min(candidates) if candidates else math.inf
                        re.zhx_matrix.set(i, j, k, l, best)

        # --- 2.1 Add two-gap composition candidate to wx(i,j) ---
        Gw = self.cfg.pk_penalty_gw
        for s in range(0, n):
            for i in range(0, n - s):
                j = i + s
                best = re.wx_matrix.get(i, j)   # baseline from nested copy
                best_bp = None

                # Enumerate a small set of complementary-hole tuples.
                # This iterator yields (r,k,l) where both whx pieces are "well-formed"
                # for left:  whx(i, r : k, l)
                # for right: whx(k+1, j : l-1, r+1)
                # If an index combo is invalid, the whx get() simply returns +inf.
                for (r, k, l) in _iter_complementary_tuples(i, j):
                    left  = _whx_collapse_first(re, i, r, k, l)
                    right = _whx_collapse_first(re, k + 1, j, l - 1, r + 1)
                    cand = Gw + left + right
                    if cand < best:
                        best = cand
                        best_bp = ("RE_PK_COMPOSE_WX", (i, r, k, l))

                # Keep the winner
                re.wx_matrix.set(i, j, best)
                if best_bp is not None:
                    re.wx_back_ptr[(i, j)] = best_bp

        # --- 2.2 Add two-gap composition into vx (via zhx) ---
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                best = re.vx_matrix.get(i, j)
                best_bp = None

                for (r, k, l) in _iter_complementary_tuples(i, j):
                    left = _zhx_collapse_first(re, i, r, k, l)
                    right = _zhx_collapse_first(re, k + 1, j, l - 1, r + 1)
                    cand = Gw + left + right
                    if cand < best:
                        best = cand
                        best_bp = ("RE_PK_COMPOSE_VX", (r, k, l))

                re.vx_matrix.set(i, j, best)
                if best_bp is not None:
                    re.vx_back_ptr[(i, j)] = best_bp

        # --- 3) Final relax for zero-cost model (keep WHX/ ZHX aligned to updated WX/VX, `WHX(i,j:k,l) == WX(i,j)`) ---
        for s in range(0, n):
            for i in range(0, n - s):
                j = i + s
                # current outer value (after PK composition)
                w_ij = re.wx_matrix.get(i, j)
                v_ij = re.vx_matrix.get(i, j)
                max_h = max(0, j - i - 1)
                for h in range(1, max_h + 1):  # only non-collapsed holes
                    for k in range(i, j - h):
                        l = k + h + 1
                        # zero-cost shrink => hole reduces to collapse => equals WX(i,j)
                        re.whx_matrix.set(i, j, k, l, w_ij)
                        re.zhx_matrix.set(i, j, k, l, v_ij)

    def fill_with_costs(self, seq: str, nested: FoldState, re: RivasEddyState) -> None:
        """
        Step 12.5:
          - seed wx/vx from nested,
          - whx/zhx DP with single-strand per-step cost q_ss,
          - two-gap composition into wx/vx with Gw (and optional coax),
          - final relax removed (since moves are not zero-cost anymore).
        """
        n = re.n
        q = self.cfg.costs.q_ss
        Gw = self.cfg.pk_penalty_gw
        P_ = self.cfg.costs.P_tilde
        Q_ = self.cfg.costs.Q_tilde
        L_ = self.cfg.costs.L_tilde
        R_ = self.cfg.costs.R_tilde
        M_ = self.cfg.costs.M_tilde
        coax = self.cfg.costs.coax_bonus if self.cfg.enable_coax else 0.0

        # --- 0. Seed non-gap from nested ---
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                re.wx_matrix.set(i, j, nested.w_matrix.get(i, j))
                re.vx_matrix.set(i, j, nested.v_matrix.get(i, j))

        # --- 1.1 whx DP with `q_ss` costs ---
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                max_h = max(0, j - i - 1)
                for h in range(1, max_h + 1):
                    for k in range(i, j - h):
                        l = k + h + 1
                        best = math.inf
                        best_bp = None

                        # 1) shrink-left: (k+1,l) + q
                        v = get_whx_with_collapse(re.whx_matrix, re.wx_matrix, i, j, k + 1, l)
                        cand = v + q
                        if cand < best:
                            best = cand
                            best_bp = (RE_BP_SHRINK_LEFT, (i, j, k + 1, l))

                        # 2) shrink-right: (k,l-1) + q
                        v = get_whx_with_collapse(re.whx_matrix, re.wx_matrix, i, j, k, l - 1)
                        cand = v + q
                        if cand < best:
                            best = cand
                            best_bp = (RE_BP_SHRINK_RIGHT, (i, j, k, l - 1))

                        # 3) trim outer-left: (i+1,j:k,l) + q
                        v = re.whx_matrix.get(i + 1, j, k, l)
                        cand = v + q
                        if cand < best:
                            best = cand
                            best_bp = (RE_BP_TRIM_LEFT, (i + 1, j, k, l))

                        # 4) trim outer-right: (i,j-1:k,l) + q
                        v = re.whx_matrix.get(i, j - 1, k, l)
                        cand = v + q
                        if cand < best:
                            best = cand
                            best_bp = (RE_BP_TRIM_RIGHT, (i, j - 1, k, l))

                        # 5) direct collapse (if h==0 via accessor, but here h>=1): allow as candidate anyway
                        v = get_whx_with_collapse(re.whx_matrix, re.wx_matrix, i, j, k, l)
                        if v < best:
                            best = v
                            best_bp = (RE_BP_COLLAPSE, (i, j))

                        re.whx_matrix.set(i, j, k, l, best)
                        re.whx_back_ptr.set(i, j, k, l, best_bp)

        # --- 1.2 VHX: `vhx` DP with paired outer and paired hole. Close both sides -------------------
        # ---     via WHX(i+1,j-1 : k-1,l+1) -------------------
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                max_h = max(0, j - i - 1)
                for h in range(1, max_h + 1):
                    for k in range(i, j - h):
                        l = k + h + 1
                        best = math.inf
                        best_bp = None

                        inner = re.whx_matrix.get(i + 1, j - 1, k - 1, l + 1)
                        if math.isfinite(inner):
                            cand = 2.0 * P_ + M_ + inner
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_VHX_CLOSE_BOTH, (i + 1, j - 1, k - 1, l + 1))

                        re.vhx_matrix.set(i, j, k, l, best)
                        re.vhx_back_ptr.set(i, j, k, l, best_bp)

        # --- 1.3 ZHX: `zhx` DP with q_ss costs. Implements paired & dangle variants via vhx matrix, and  ---
        # --- single-strand hole shrink ---
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                max_h = max(0, j - i - 1)
                for h in range(1, max_h + 1):
                    for k in range(i, j - h):
                        l = k + h + 1
                        best = math.inf
                        best_bp = None

                        # paired: P~ + vhx(i,j:k,l)
                        v = re.vhx_matrix.get(i, j, k, l)
                        if math.isfinite(v):
                            cand = P_ + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_FROM_VHX, (i, j, k, l))

                        # hole dangles (use tables)
                        v = re.vhx_matrix.get(i, j, k - 1, l + 1)
                        if math.isfinite(v):
                            Lh = _dangle_hole_L(seq, k, self.cfg.costs)
                            Rh = _dangle_hole_R(seq, l, self.cfg.costs)
                            cand = Lh + Rh + P_ + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_DANGLE_LR, (i, j, k - 1, l + 1))

                        v = re.vhx_matrix.get(i, j, k - 1, l)
                        if math.isfinite(v):
                            Rh = _dangle_hole_R(seq, l - 1, self.cfg.costs)  # hole-right adjacent at l
                            cand = Rh + P_ + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_DANGLE_R, (i, j, k - 1, l))

                        v = re.vhx_matrix.get(i, j, k, l + 1)
                        if math.isfinite(v):
                            Lh = _dangle_hole_L(seq, k + 1, self.cfg.costs)
                            cand = Lh + P_ + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_DANGLE_L, (i, j, k, l + 1))

                        # single-stranded hole shrink
                        v = re.zhx_matrix.get(i, j, k - 1, l)
                        if math.isfinite(v):
                            cand = Q_ + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_SS_LEFT, (i, j, k - 1, l))
                        v = re.zhx_matrix.get(i, j, k, l + 1)
                        if math.isfinite(v):
                            cand = Q_ + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_SS_RIGHT, (i, j, k, l + 1))

                        re.zhx_matrix.set(i, j, k, l, best)
                        re.zhx_back_ptr.set(i, j, k, l, best_bp)

        # --- 1.4 YHX: `yhx` DP (k,l paired; i,j undetermined). Implements dangles/singles on -------------------
        # --- outer, and wrap-around via whx(i,j:k-1,l+1) -------------------
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                max_h = max(0, j - i - 1)
                for h in range(1, max_h + 1):
                    for k in range(i, j - h):
                        l = k + h + 1
                        best = math.inf
                        best_bp = None

                        # outer dangles (tables)
                        v = re.vhx_matrix.get(i + 1, j, k, l)
                        if math.isfinite(v):
                            Lo = _dangle_outer_L(seq, i, self.cfg.costs)
                            cand = Lo + P_ + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_DANGLE_L, (i + 1, j, k, l))

                        v = re.vhx_matrix.get(i, j - 1, k, l)
                        if math.isfinite(v):
                            Ro = _dangle_outer_R(seq, j, self.cfg.costs)
                            cand = Ro + P_ + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_DANGLE_R, (i, j - 1, k, l))

                        # single-strand on outer
                        v = re.yhx_matrix.get(i + 1, j, k, l)
                        if math.isfinite(v):
                            cand = Q_ + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_SS_LEFT, (i + 1, j, k, l))
                        v = re.yhx_matrix.get(i, j - 1, k, l)
                        if math.isfinite(v):
                            cand = Q_ + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_SS_RIGHT, (i, j - 1, k, l))

                        # wrap via WHX(i,j:k-1,l+1)
                        v = re.whx_matrix.get(i, j, k - 1, l + 1)
                        if math.isfinite(v):
                            cand = P_ + M_ + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_WRAP_WHX, (i, j, k - 1, l + 1))

                        re.yhx_matrix.set(i, j, k, l, best)
                        re.yhx_back_ptr.set(i, j, k, l, best_bp)

        # --- 2.1 Composition into WX: (a) WHX+WHX  (b) NEW: YHX+YHX ---
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                best = re.wx_matrix.get(i, j)
                best_bp = None

                for (r, k, l) in _iter_complementary_tuples(i, j):
                    # (a) whx + whx (as before)
                    left_w = _whx_collapse_first(re, i, r, k, l)
                    right_w = _whx_collapse_first(re, k + 1, j, l - 1, r + 1)
                    cand_w = Gw + left_w + right_w
                    if cand_w < best:
                        best = cand_w
                        best_bp = (RE_BP_COMPOSE_WX, (r, k, l))

                    # (b) NEW: yhx + yhx (non-nested branch)
                    left_y = re.yhx_matrix.get(i, r, k, l)
                    right_y = re.yhx_matrix.get(k + 1, j, l - 1, r + 1)
                    if math.isfinite(left_y) and math.isfinite(right_y):
                        cand_y = Gw + left_y + right_y
                        if cand_y < best:
                            best = cand_y
                            best_bp = (RE_BP_COMPOSE_WX_YHX, (r, k, l))

                re.wx_matrix.set(i, j, best)
                if best_bp is not None:
                    re.wx_back_ptr[(i, j)] = best_bp

        # --- 2.2 two-gap composition into vx (zhx) ---
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                best = re.vx_matrix.get(i, j)
                best_bp = None

                for (r, k, l) in _iter_complementary_tuples(i, j):
                    left = _zhx_collapse_first(re, i, r, k, l)
                    right = _zhx_collapse_first(re, k + 1, j, l - 1, r + 1)
                    # NEW: coax from pair types at the join (i,r) and (k+1,j)
                    coax_e = _coax_energy_for_join(seq, (i, r), (k + 1, j), self.cfg.costs)
                    cand = Gw + left + right + coax_e
                    if cand < best:
                        best = cand
                        best_bp = (RE_BP_COMPOSE_VX, (r, k, l))

                re.vx_matrix.set(i, j, best)
                if best_bp is not None:
                    re.vx_back_ptr[(i, j)] = best_bp

def _whx_collapse_first(re: RivasEddyState, i: int, j: int, k: int, l: int) -> float:
    """
    Safe accessor for whx(i,j:k,l): try collapse identity first (finite),
    then stored value (which may be +inf if not set).
    """
    v = get_whx_with_collapse(re.whx_matrix, re.wx_matrix, i, j, k, l)
    if math.isfinite(v):
        return v
    return re.whx_matrix.get(i, j, k, l)

def _zhx_collapse_first(re: RivasEddyState, i: int, j: int, k: int, l: int) -> float:
    v = get_zhx_with_collapse(re.zhx_matrix, re.vx_matrix, i, j, k, l)
    if math.isfinite(v): return v
    return re.zhx_matrix.get(i, j, k, l)

def _iter_complementary_tuples(i: int, j: int) -> Iterator[Tuple[int, int, int]]:
    """
    Very conservative enumeration of (r,k,l) for the two-gap composition.
    We keep r strictly inside (i..j), and choose k<l with some spacing.
    Many combos will be filtered by +inf lookups; that's fine for Step 12 minimal.
    """
    # i < k < r < l <= j  (keeps non-degenerate subspans)
    for r in range(i + 1, j):      # r is a "connector" split inside [i..j]
        for k in range(i, r + 1):  # hole start (left/before r)
            for l in range(r + 1, j + 1):  # hole end (right/after r)
                # Basic sanity: ensure each index stays in outer bounds
                # The whx accessors will return +inf for any illegal shapes.
                yield (r, k, l)
