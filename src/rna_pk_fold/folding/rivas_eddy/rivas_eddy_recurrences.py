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

# ======================================================================

# ======================================================================
RE_BP_COMPOSE_WX   = "RE_PK_COMPOSE_WX"
RE_BP_COMPOSE_VX   = "RE_PK_COMPOSE_VX"
RE_BP_COMPOSE_WX_YHX = "RE_PK_COMPOSE_WX_YHX"
RE_BP_COMPOSE_WX_YHX_WHX = "RE_PK_COMPOSE_WX_YHX_WHX"  # yhx + whx
RE_BP_COMPOSE_WX_WHX_YHX = "RE_PK_COMPOSE_WX_WHX_YHX"  # whx + yhx
RE_BP_COMPOSE_WX_YHX_OVERLAP = "RE_PK_COMPOSE_WX_YHX_OVERLAP"  # yhx(i,r:k,l) + yhx(r+1,j:k,l)
RE_BP_WX_SELECT_UNCHARGED = "RE_WX_SELECT_UNCHARGED"
RE_BP_VX_SELECT_UNCHARGED = "RE_VX_SELECT_UNCHARGED"

# ======================================================================

# ======================================================================
RE_BP_WHX_SHRINK_LEFT  = "RE_SHRINK_LEFT"
RE_BP_WHX_SHRINK_RIGHT = "RE_SHRINK_RIGHT"
RE_BP_WHX_TRIM_LEFT    = "RE_TRIM_LEFT"
RE_BP_WHX_TRIM_RIGHT   = "RE_TRIM_RIGHT"
RE_BP_WHX_COLLAPSE     = "RE_COLLAPSE"
RE_BP_WHX_SS_BOTH              = "RE_WHX_SS_BOTH"              # 2*q_ss + whx(i+1, j-1 : k, l)
RE_BP_WHX_SPLIT_LEFT_WHX_WX    = "RE_WHX_SPLIT_LEFT_WHX_WX"    # whx(i, r : k, l) + wx(r+1, j)
RE_BP_WHX_SPLIT_RIGHT_WX_WHX   = "RE_WHX_SPLIT_RIGHT_WX_WHX"   # wx(i, s) + whx(s+1, j : k, l)
RE_BP_WHX_OVERLAP_SPLIT        = "RE_WHX_OVERLAP_SPLIT"        # Gwh + whx(i, r : k, l) + whx(r+1, j : k, l)

# ======================================================================
# VHX backpointer tags
# ======================================================================
RE_BP_VHX_DANGLE_L        = "RE_VHX_DANGLE_L"        # P~+L~ + vhx(i,j:k+1,l)
RE_BP_VHX_DANGLE_R        = "RE_VHX_DANGLE_R"        # P~+R~ + vhx(i,j:k,l-1)
RE_BP_VHX_DANGLE_LR       = "RE_VHX_DANGLE_LR"       # P~+L~+R~ + vhx(i,j:k+1,l-1)
RE_BP_VHX_SS_LEFT         = "RE_VHX_SS_LEFT"         # Q~ + zhx(i,j:k,l)   (left-labelled)
RE_BP_VHX_SS_RIGHT        = "RE_VHX_SS_RIGHT"        # Q~ + zhx(i,j:k,l)   (right-labelled)
RE_BP_VHX_SPLIT_LEFT_ZHX_WX  = "RE_VHX_SPLIT_LEFT_ZHX_WX"   # zhx(i,j:r,l) + wx(r+1,k)
RE_BP_VHX_SPLIT_RIGHT_ZHX_WX = "RE_VHX_SPLIT_RIGHT_ZHX_WX"  # zhx(i,j:k,s) + wx(l,s-1)
RE_BP_VHX_IS2_INNER_ZHX      = "RE_VHX_IS2_INNER_ZHX"       # ĨS2(i,j:r,s) + zhx(r,s:k,l)
RE_BP_VHX_WRAP_WHX           = "RE_VHX_WRAP_WHX"            # P~+M~ + whx(i+1,j-1:k,l)

# ======================================================================

# ======================================================================
RE_BP_VHX_CLOSE_BOTH = "RE_VHX_CLOSE_BOTH"      # 2*P~ + M~ + whx(i+1,j-1:k-1,l+1)
RE_BP_ZHX_FROM_VHX   = "RE_ZHX_FROM_VHX"        # P~ + vhx(i,j:k,l)
RE_BP_ZHX_DANGLE_LR  = "RE_ZHX_DANGLE_LR"       # L~+R~+P~ + vhx(i,j:k-1,l+1)
RE_BP_ZHX_DANGLE_L   = "RE_ZHX_DANGLE_L"        # L~+P~     + vhx(i,j:k,l+1)
RE_BP_ZHX_DANGLE_R   = "RE_ZHX_DANGLE_R"        # R~+P~     + vhx(i,j:k-1,l)
RE_BP_ZHX_SS_LEFT    = "RE_ZHX_SS_LEFT"         # Q~ + zhx(i,j:k-1,l)
RE_BP_ZHX_SS_RIGHT   = "RE_ZHX_SS_RIGHT"        # Q~ + zhx(i,j:k,l+1)
RE_BP_ZHX_SPLIT_LEFT_ZHX_WX   = "RE_ZHX_SPLIT_LEFT_ZHX_WX"    # zhx(i,j:r,l) + wx(r+1,k)
RE_BP_ZHX_SPLIT_RIGHT_ZHX_WX  = "RE_ZHX_SPLIT_RIGHT_ZHX_WX"   # zhx(i,j:k,s) + wx(l, s-1)
RE_BP_ZHX_IS2_INNER_VHX       = "RE_ZHX_IS2_INNER_VHX"        # ĨS2(i,j:r,s) + vhx(r,s:k,l)

# ======================================================================

# ======================================================================
RE_BP_YHX_DANGLE_L   = "RE_YHX_DANGLE_L"        # L~+P~ + vhx(i+1,j:k,l)
RE_BP_YHX_DANGLE_R   = "RE_YHX_DANGLE_R"        # R~+P~ + vhx(i, j-1:k,l)
RE_BP_YHX_SS_LEFT    = "RE_YHX_SS_LEFT"         # Q~ + yhx(i+1,j:k,l)
RE_BP_YHX_SS_RIGHT   = "RE_YHX_SS_RIGHT"        # Q~ + yhx(i, j-1:k,l)
RE_BP_YHX_WRAP_WHX   = "RE_YHX_WRAP_WHX"        # P~ + M~ + whx(i,j:k-1,l+1)"
RE_BP_YHX_DANGLE_LR          = "RE_YHX_DANGLE_LR"          # L~+R~+P~ + vhx(i+1, j-1:k,l)
RE_BP_YHX_SS_BOTH            = "RE_YHX_SS_BOTH"            # 2·Q~ + yhx(i+1, j-1:k,l)
RE_BP_YHX_SPLIT_LEFT_YHX_WX  = "RE_YHX_SPLIT_LEFT_YHX_WX"  # yhx(i,r:k,l) + wx(r+1, j)
RE_BP_YHX_SPLIT_RIGHT_WX_YHX = "RE_YHX_SPLIT_RIGHT_WX_YHX" # wx(i, s) + yhx(s+1, j:k,l)
RE_BP_YHX_WRAP_WHX_L         = "RE_YHX_WRAP_WHX_L"         # L~+P~+M~ + whx(i+1, j:k-1, l+1)
RE_BP_YHX_WRAP_WHX_R         = "RE_YHX_WRAP_WHX_R"         # R~+P~+M~ + whx(i, j-1:k-1, l+1)
RE_BP_YHX_WRAP_WHX_LR        = "RE_YHX_WRAP_WHX_LR"        # L~+R~+P~+M~ + whx(i+1, j-1:k-1, l+1)


# ---------- Table/evaluator helpers (DANGLES + COAX) ----------
def wxI(re: RivasEddyState, i: int, j: int) -> float:
    """
    Inner-W accessor: use dedicated wxi_matrix if present; otherwise fall back to wx.
    """
    mat = getattr(re, "wxi_matrix", None)
    return mat.get(i, j) if mat is not None else re.wx_matrix.get(i, j)


def IS2_outer(seq: str, tables, i: int, j: int, r: int, s: int) -> float:
    """
    Hook for the ĨS₂ outer-context term used by VHX (and later WHX).
    If cfg.tables.IS2_outer is provided, call it; otherwise 0.0.
    """
    if tables and hasattr(tables, "IS2_outer"):
        fn = tables.IS2_outer
        return fn(seq, i, j, r, s) if callable(fn) else float(fn)
    return 0.0

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
    P_tilde_out: float = 1.0  # outer pairing weight (YHX-side)
    P_tilde_hole: float = 1.0  # hole pairing weight (VHX/ZHX-side)
    Q_tilde_out: float = 0.2  # outer single-strand (YHX SS trims)
    Q_tilde_hole: float = 0.2  # hole single-strand (ZHX SS; VHX→ZHX SS)
    L_tilde: float = 0.0  # 5′ dangle on gap edge (acts like L~)
    R_tilde: float = 0.0  # 3′ dangle on gap edge (acts like R~)
    M_tilde: float = 0.0  # multiloop-ish term inside vhx/yhx closures (acts like M~)
    M_tilde_yhx: float = 0.0  # M~ inside YHX wrap
    M_tilde_vhx: float = 0.0  # M~ inside VHX wrap
    M_tilde_whx: float = 0.0  # (optional hook) M~ inside WHX contexts

    # Dangle/coax tables (keyed on local bases or pair types)
    dangle_hole_L: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (k-1, k)
    dangle_hole_R: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (l, l+1)
    dangle_outer_L: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (i, i+1)
    dangle_outer_R: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (j-1, j)
    coax_pairs: Dict[Tuple[str, str], float] = field(default_factory=dict)  # ((pairTypeA),(pairTypeB))

    # Optional bonuses/penalties
    coax_bonus: float = 0.0  # used in your vx path; leave 0.0 unless you want it
    Gwh: float = 0.0  # penalty for overlapping PKs (not used yet)
    Gwi: float = 0.0 # inner-gap/inner-PK entry penalty (Ĝ_wI)
    Gwh_wx: float = 0.0  # used by WX-level YHX+YHX “same-hole overlap”
    Gwh_whx: float = 0.0  # used by WHX-level overlap-split
    coax_scale: float = 1.0

@dataclass(slots=True)
class REREConfig:
    enable_coax: bool = False # keep off initially
    enable_wx_overlap: bool = False # turn on WX same-hole overlap terms
    enable_coax_variants: bool = False  # NEW: add extra coax topologies in VX composition
    pk_penalty_gw: float = 1.0 # Gw: pseudoknot introduction penalty (kcal/mol)
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
        Gwh = getattr(self.cfg.costs, "Gwh", 0.0)
        Gwi = self.cfg.costs.Gwi
        Gwh_wx = (self.cfg.costs.Gwh_wx if self.cfg.costs.Gwh_wx != 0.0 else self.cfg.costs.Gwh)
        Gwh_whx = (self.cfg.costs.Gwh_whx if self.cfg.costs.Gwh_whx != 0.0 else self.cfg.costs.Gwh)
        tables = getattr(self.cfg, "tables", None)
        coax = self.cfg.costs.coax_bonus if self.cfg.enable_coax else 0.0
        g = self.cfg.costs.coax_scale

        # tilde scalars (fallback 0)
        P_out = getattr(tables, "P_tilde_out", getattr(self.cfg.costs, "P_tilde_out", 1.0))
        P_hole = getattr(tables, "P_tilde_hole", getattr(self.cfg.costs, "P_tilde_hole", 1.0))
        L_ = getattr(tables, "L_tilde", 0.0)
        R_ = getattr(tables, "R_tilde", 0.0)
        Q_out = getattr(tables, "Q_tilde_out", getattr(self.cfg.costs, "Q_tilde_out", 0.0))
        Q_hole = getattr(tables, "Q_tilde_hole", getattr(self.cfg.costs, "Q_tilde_hole", 0.0))
        M_yhx = getattr(tables, "M_tilde_yhx", getattr(self.cfg.costs, "M_tilde_yhx", 0.0))
        M_vhx = getattr(tables, "M_tilde_vhx", getattr(self.cfg.costs, "M_tilde_vhx", 0.0))
        M_whx = getattr(tables, "M_tilde_whx", getattr(self.cfg.costs, "M_tilde_whx", 0.0))

        # --- 0. Seed non-gap from nested ---
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                base_w = nested.w_matrix.get(i, j)
                base_v = nested.v_matrix.get(i, j)

                # Public views will become min(wxu, wxc) and min(vxu, vxc) later.
                re.wxu_matrix.set(i, j, base_w)  # uncharged baseline
                re.vxu_matrix.set(i, j, base_v)

                # charged start as +inf (except i==j already set in make_re_fold_state)
                if i != j:
                    re.wxc_matrix.set(i, j, math.inf)
                    re.vxc_matrix.set(i, j, math.inf)

                # keep existing copies in wx/vx for now; rewrite at the very end
                re.wx_matrix.set(i, j, base_w)
                re.vx_matrix.set(i, j, base_v)

                # wxi stays nested only (already seeded earlier step)
                if hasattr(re, "wxi_matrix") and re.wxi_matrix is not None:
                    re.wxi_matrix.set(i, j, base_w)

        # --- 1.1 WHX Recurrence: whx DP with `q_ss` costs ---
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
                        v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k + 1, l)
                        cand = v + q
                        if cand < best:
                            best = cand
                            best_bp = (RE_BP_WHX_SHRINK_LEFT, (i, j, k + 1, l))

                        # 2) shrink-right: (k,l-1) + q
                        v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k, l - 1)
                        cand = v + q
                        if cand < best:
                            best = cand
                            best_bp = (RE_BP_WHX_SHRINK_RIGHT, (i, j, k, l - 1))

                        # 3) trim outer-left: (i+1,j:k,l) + q
                        v = re.whx_matrix.get(i + 1, j, k, l)
                        cand = v + q
                        if cand < best:
                            best = cand
                            best_bp = (RE_BP_WHX_TRIM_LEFT, (i + 1, j, k, l))

                        # 4) trim outer-right: (i,j-1:k,l) + q
                        v = re.whx_matrix.get(i, j - 1, k, l)
                        cand = v + q
                        if cand < best:
                            best = cand
                            best_bp = (RE_BP_WHX_TRIM_RIGHT, (i, j - 1, k, l))

                        # 5) direct collapse (if h==0 via accessor, but here h>=1): allow as candidate anyway
                        v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k, l)
                        if v < best:
                            best = v
                            best_bp = (RE_BP_WHX_COLLAPSE, (i, j))

                        v = re.whx_matrix.get(i + 1, j - 1, k, l)
                        if math.isfinite(v):
                            cand = v + 2.0 * q  # keep using q_ss to preserve earlier tests
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_WHX_SS_BOTH, (i + 1, j - 1, k, l))

                        # --- NEW: non-nested outer splits with WX ---
                        # Left split: whx(i, r : k, l) + wx(r+1, j)
                        for r in range(i, j):
                            left = re.whx_matrix.get(i, r, k, l)
                            right = wxI(re, r + 1, j)
                            if math.isfinite(left) and math.isfinite(right):
                                cand = left + right
                                if cand < best:
                                    best = cand
                                    best_bp = (RE_BP_WHX_SPLIT_LEFT_WHX_WX, (r,))

                        # Right split: wx(i, s) + whx(s+1, j : k, l)
                        for s2 in range(i, j):
                            left = wxI(re, i, s2)
                            right = re.whx_matrix.get(s2 + 1, j, k, l)
                            if math.isfinite(left) and math.isfinite(right):
                                cand = left + right
                                if cand < best:
                                    best = cand
                                    best_bp = (RE_BP_WHX_SPLIT_RIGHT_WX_WHX, (s2,))

                        # --- NEW: overlapping-PK split into WHX + WHX with penalty Gwh_whx ---
                        if Gwh_whx != 0.0:  # skip loop if it's 0 for speed
                            for r in range(i, j):
                                left = re.whx_matrix.get(i, r, k, l)
                                right = re.whx_matrix.get(r + 1, j, k, l)
                                if math.isfinite(left) and math.isfinite(right):
                                    cand = Gwh_whx + left + right
                                    if cand < best:
                                        best = cand
                                        best_bp = (RE_BP_WHX_OVERLAP_SPLIT, (r,))

                        re.whx_matrix.set(i, j, k, l, best)
                        re.whx_back_ptr.set(i, j, k, l, best_bp)

        # --- 1.2 VHX Recurrence: `vhx` DP with paired outer and paired hole. Close both sides -------------------
        # ---     via WHX(i+1,j-1 : k-1,l+1) -------------------
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                max_h = max(0, j - i - 1)
                for h in range(1, max_h + 1):
                    for k in range(i, j - h):
                        l = k + h + 1
                        best = re.vhx_matrix.get(i, j, k, l)
                        best_bp = None

                        # --- DANGLES (shrink left/right/both) ---
                        v = re.vhx_matrix.get(i, j, k + 1, l)
                        cand = P_hole + L_ + v
                        if cand < best:
                            best, best_bp = cand, (RE_BP_VHX_DANGLE_L, (i, j, k + 1, l))

                        v = re.vhx_matrix.get(i, j, k, l - 1)
                        cand = P_hole + R_ + v
                        if cand < best:
                            best, best_bp = cand, (RE_BP_VHX_DANGLE_R, (i, j, k, l - 1))

                        v = re.vhx_matrix.get(i, j, k + 1, l - 1)
                        cand = P_hole + L_ + R_ + v
                        if cand < best:
                            best, best_bp = cand, (RE_BP_VHX_DANGLE_LR, (i, j, k + 1, l - 1))

                        # --- SINGLE-STRAND from ZHX (label left/right)
                        v_zhx = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, k, l)
                        cand = Q_hole + v_zhx
                        if cand < best:
                            best, best_bp = cand, (RE_BP_VHX_SS_LEFT, (i, j, k, l))

                        # Same energy; alternate label for symmetry
                        elif cand == best:
                            best_bp = (RE_BP_VHX_SS_RIGHT, (i, j, k, l))

                        # --- SPLIT on the LEFT: r in [i..k-1]  →  zhx(i,j:r,l) + wx(r+1,k)
                        for r in range(i, k):
                            left = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, r, l)
                            right = wxI(re, r + 1, k)
                            cand = left + right
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_VHX_SPLIT_LEFT_ZHX_WX, (r,))

                        # --- SPLIT on the RIGHT: s in [l+1..j] →  zhx(i,j:k,s) + wx(l, s-1)
                        for s2 in range(l + 1, j + 1):
                            left = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, k, s2)
                            right = wxI(re, l, s2 - 1)
                            cand = left + right
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_VHX_SPLIT_RIGHT_ZHX_WX, (s2,))

                        # --- INTERIOR (ĨS₂ + zhx(r,s:k,l)) over r,s covering hole ---
                        for r in range(i, k + 1):
                            for s2 in range(l, j + 1):
                                if r <= k and l <= s2 and r <= s2:
                                    inner = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, r, s2, k, l)
                                    cand = IS2_outer(seq, tables, i, j, r, s2) + inner
                                    if cand < best:
                                        best = cand
                                        best_bp = (RE_BP_VHX_IS2_INNER_ZHX, (r, s2))

                        # --- CLOSE_BOTH: pair both outer (i,j) and hole (k,l) ends in one step
                        close = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i + 1, j - 1, k - 1, l + 1)
                        if math.isfinite(close):
                            cand = 2.0 * P_hole + M_vhx + close + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_VHX_CLOSE_BOTH, (i + 1, j - 1, k - 1, l + 1))

                        # --- WRAP via WHX (P̃+M̃ + whx(i+1,j-1:k,l)) ---
                        wrap = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i + 1, j - 1, k, l)
                        cand = P_hole + M_vhx + wrap + Gwi
                        if cand < best:
                            best = cand
                            best_bp = (RE_BP_VHX_WRAP_WHX, (i + 1, j - 1))

                        re.vhx_matrix.set(i, j, k, l, best)
                        re.vhx_back_ptr.set(i, j, k, l, best_bp)

        # --- 1.3 ZHX Recurrence: `zhx` DP with q_ss costs. Implements paired & dangle variants via vhx matrix, and  ---
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

                        # FROM_VHX
                        v = re.vhx_matrix.get(i, j, k, l)
                        if math.isfinite(v):
                            cand = P_hole + v + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_FROM_VHX, (i, j, k, l))

                        # DANGLE_LR from VHX
                        v = re.vhx_matrix.get(i, j, k - 1, l + 1)
                        if math.isfinite(v):
                            Lh = _dangle_hole_L(seq, k, self.cfg.costs)
                            Rh = _dangle_hole_R(seq, l, self.cfg.costs)
                            cand = Lh + Rh + P_hole + v + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_DANGLE_LR, (i, j, k - 1, l + 1))

                        # DANGLE_R from VHX
                        v = re.vhx_matrix.get(i, j, k - 1, l)
                        if math.isfinite(v):
                            Rh = _dangle_hole_R(seq, l - 1, self.cfg.costs)
                            cand = Rh + P_hole + v + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_DANGLE_R, (i, j, k - 1, l))

                        # DANGLE_L from VHX
                        v = re.vhx_matrix.get(i, j, k, l + 1)
                        if math.isfinite(v):
                            Lh = _dangle_hole_L(seq, k + 1, self.cfg.costs)
                            cand = Lh + P_hole + v + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_DANGLE_L, (i, j, k, l + 1))

                        # SS_LEFT
                        v = re.zhx_matrix.get(i, j, k - 1, l)
                        if math.isfinite(v):
                            cand = Q_hole + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_SS_LEFT, (i, j, k - 1, l))

                        # SS_RIGHT
                        v = re.zhx_matrix.get(i, j, k, l + 1)
                        if math.isfinite(v):
                            cand = Q_hole + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_ZHX_SS_RIGHT, (i, j, k, l + 1))

                        for r in range(i, k):
                            left = re.zhx_matrix.get(i, j, r, l)
                            right = wxI(re, r + 1, k)
                            if math.isfinite(left) and math.isfinite(right):
                                cand = left + right
                                if cand < best:
                                    best = cand
                                    best_bp = (RE_BP_ZHX_SPLIT_LEFT_ZHX_WX, (r,))

                        for s2 in range(l + 1, j + 1):
                            left = re.zhx_matrix.get(i, j, k, s2)
                            right = wxI(re, l, s2 - 1)
                            if math.isfinite(left) and math.isfinite(right):
                                cand = left + right
                                if cand < best:
                                    best = cand
                                    best_bp = (RE_BP_ZHX_SPLIT_RIGHT_ZHX_WX, (s2,))

                        for r in range(i, k + 1):
                            for s2 in range(l, j + 1):
                                if r <= s2:
                                    inner = re.vhx_matrix.get(r, s2, k, l)
                                    if math.isfinite(inner):
                                        bridge = IS2_outer(seq, tables, i, j, r, s2)
                                        cand = bridge + inner
                                        if cand < best:
                                            best = cand
                                            best_bp = (RE_BP_ZHX_IS2_INNER_VHX, (r, s2))

                        # write back
                        re.zhx_matrix.set(i, j, k, l, best)
                        re.zhx_back_ptr.set(i, j, k, l, best_bp)

        # --- 1.4 YHX Recurrence: `yhx` DP (k,l paired; i,j undetermined). Implements dangles/singles on -------------------
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

                        # Outer dangle L
                        v = re.vhx_matrix.get(i + 1, j, k, l)
                        if math.isfinite(v):
                            Lo = _dangle_outer_L(seq, i, self.cfg.costs)
                            cand = Lo + P_out + v + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_DANGLE_L, (i + 1, j, k, l))

                        # Outer dangle R
                        v = re.vhx_matrix.get(i, j - 1, k, l)
                        if math.isfinite(v):
                            Ro = _dangle_outer_R(seq, j, self.cfg.costs)
                            cand = Ro + P_out + v + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_DANGLE_R, (i, j - 1, k, l))

                        # Outer dangle LR (both sides)
                        v = re.vhx_matrix.get(i + 1, j - 1, k, l)
                        if math.isfinite(v):
                            Lo = _dangle_outer_L(seq, i, self.cfg.costs)
                            Ro = _dangle_outer_R(seq, j, self.cfg.costs)
                            cand = Lo + Ro + P_out + v + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_DANGLE_LR, (i + 1, j - 1, k, l))

                        # Single-strand outer trims: Left
                        v = re.yhx_matrix.get(i + 1, j, k, l)
                        if math.isfinite(v):
                            cand = Q_out + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_SS_LEFT, (i + 1, j, k, l))

                        # Single-strand outer trims: Right
                        v = re.yhx_matrix.get(i, j - 1, k, l)
                        if math.isfinite(v):
                            cand = Q_out + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_SS_RIGHT, (i, j - 1, k, l))

                        # Single-strand both sides (shortcut; equivalent to two trims)
                        v = re.yhx_matrix.get(i + 1, j - 1, k, l)
                        if math.isfinite(v):
                            cand = 2.0 * Q_out + v
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_SS_BOTH, (i + 1, j - 1, k, l))

                        # Wrap via WHX(i,j:k-1,l+1)
                        v = re.whx_matrix.get(i, j, k - 1, l + 1)
                        if math.isfinite(v):
                            cand = P_out + M_yhx + v + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_WRAP_WHX, (i, j, k - 1, l + 1))

                        # Wrap + outer dangles L / R / LR
                        v = re.whx_matrix.get(i + 1, j, k - 1, l + 1)
                        if math.isfinite(v):
                            Lo = _dangle_outer_L(seq, i, self.cfg.costs)
                            cand = Lo + P_out + M_yhx + v + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_WRAP_WHX_L, (i + 1, j, k - 1, l + 1))

                        v = re.whx_matrix.get(i, j - 1, k - 1, l + 1)
                        if math.isfinite(v):
                            Ro = _dangle_outer_R(seq, j, self.cfg.costs)
                            cand = Ro + P_out + M_yhx + v + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_WRAP_WHX_R, (i, j - 1, k - 1, l + 1))

                        v = re.whx_matrix.get(i + 1, j - 1, k - 1, l + 1)
                        if math.isfinite(v):
                            Lo = _dangle_outer_L(seq, i, self.cfg.costs)
                            Ro = _dangle_outer_R(seq, j, self.cfg.costs)
                            cand = Lo + Ro + P_out + M_yhx + v + Gwi
                            if cand < best:
                                best = cand
                                best_bp = (RE_BP_YHX_WRAP_WHX_LR, (i + 1, j - 1, k - 1, l + 1))

                        # Non-nested OUTER splits with WX
                        #   Left split:  yhx(i,r:k,l) + wx(r+1, j)
                        for r in range(i, j):
                            left = re.yhx_matrix.get(i, r, k, l)
                            right = wxI(re, r + 1, j)
                            if math.isfinite(left) and math.isfinite(right):
                                cand = left + right
                                if cand < best:
                                    best = cand
                                    best_bp = (RE_BP_YHX_SPLIT_LEFT_YHX_WX, (r,))

                        #   Right split: wx(i, s) + yhx(s+1, j:k,l)
                        for s2 in range(i, j):
                            left = wxI(re, i, s2)
                            right = re.yhx_matrix.get(s2 + 1, j, k, l)
                            if math.isfinite(left) and math.isfinite(right):
                                cand = left + right
                                if cand < best:
                                    best = cand
                                    best_bp = (RE_BP_YHX_SPLIT_RIGHT_WX_YHX, (s2,))

                        re.yhx_matrix.set(i, j, k, l, best)
                        re.yhx_back_ptr.set(i, j, k, l, best_bp)

        # --- 2.1 Composition into WX: (a) WHX+WHX  (b) YHX+YHX ---
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                best_c = re.wxc_matrix.get(i, j)
                best_bp = None

                for (r, k, l) in _iter_complementary_tuples(i, j):
                    # gather both flavors via collapse (charged only differs if hole degenerates)
                    L_u = _whx_collapse_with(re, i, r, k, l, charged=False)
                    R_u = _whx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=False)
                    L_c = _whx_collapse_with(re, i, r, k, l, charged=True)
                    R_c = _whx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=True)
                    # introduce charge ONCE
                    cand_first = Gw + L_u + R_u

                    # propagate charge WITHOUT re-charging if either side is already charged
                    cand_Lc = L_c + R_u
                    cand_Rc = L_u + R_c
                    cand_both = L_c + R_c

                    # choose the best way to be 'charged'
                    cand = min(cand_first, cand_Lc, cand_Rc, cand_both)
                    if cand < best_c:
                        best_c = cand
                        best_bp = (RE_BP_COMPOSE_WX, (r, k, l))

                    # (b) yhx + yhx (non-nested branch)
                    left_y = re.yhx_matrix.get(i, r, k, l)
                    right_y = re.yhx_matrix.get(k + 1, j, l - 1, r + 1)
                    if math.isfinite(left_y) and math.isfinite(right_y):
                        cand_y = Gw + left_y + right_y
                        if cand_y < best_c:
                            best_c = cand_y
                            best_bp = (RE_BP_COMPOSE_WX_YHX, (r, k, l))

                    # (c) MIXED: yhx (left) + whx (right)
                    left_y = re.yhx_matrix.get(i, r, k, l)
                    if math.isfinite(left_y):
                        R_u = _whx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=False)
                        R_c = _whx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=True)
                        if math.isfinite(R_u):
                            cand = Gw + left_y + R_u  # introduce charge
                            if cand < best_c:
                                best_c = cand
                                best_bp = (RE_BP_COMPOSE_WX_YHX_WHX, (r, k, l))
                        if math.isfinite(R_c):
                            cand = left_y + R_c  # carry charge, no +Gw
                            if cand < best_c:
                                best_c = cand
                                best_bp = (RE_BP_COMPOSE_WX_YHX_WHX, (r, k, l))

                    # (d) MIXED: whx (left) + yhx (right)
                    right_y = re.yhx_matrix.get(k + 1, j, l - 1, r + 1)
                    if math.isfinite(right_y):
                        L_u = _whx_collapse_with(re, i, r, k, l, charged=False)
                        L_c = _whx_collapse_with(re, i, r, k, l, charged=True)
                        if math.isfinite(L_u):
                            cand = Gw + L_u + right_y  # introduce charge
                            if cand < best_c:
                                best_c = cand
                                best_bp = (RE_BP_COMPOSE_WX_WHX_YHX, (r, k, l))
                        if math.isfinite(L_c):
                            cand = L_c + right_y  # carry charge, no +Gw
                            if cand < best_c:
                                best_c = cand
                                best_bp = (RE_BP_COMPOSE_WX_WHX_YHX, (r, k, l))

                    # (e) OPTIONAL: same-hole overlap via YHX+YHX with penalty Gwh
                    if self.cfg.enable_wx_overlap and Gwh_wx != 0.0:
                        # enumerate all inner holes (k,l) within (i,j)
                        for (k2, l2) in _iter_inner_holes(i, j):
                            # split the outer interval at r; both subproblems share the same (k2,l2)
                            for r2 in range(i, j):
                                left_y = re.yhx_matrix.get(i, r2, k2, l2)
                                right_y = re.yhx_matrix.get(r2 + 1, j, k2, l2)
                                if math.isfinite(left_y) and math.isfinite(right_y):
                                    cand_overlap = Gwh_wx + left_y + right_y
                                    if cand_overlap < best_c:
                                        best_c = cand_overlap
                                        best_bp = (RE_BP_COMPOSE_WX_YHX_OVERLAP, (r2, k2, l2))

                re.wxc_matrix.set(i, j, best_c)
                if best_bp is not None:
                    re.wx_back_ptr[(i, j)] = best_bp

        # Publish final WX as min(uncharged, charged) with selection backpointer
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                wxu = re.wxu_matrix.get(i, j)
                wxc = re.wxc_matrix.get(i, j)
                if wxu <= wxc:
                    re.wx_matrix.set(i, j, wxu)
                    # prefer neutral path; override any charged bp
                    re.wx_back_ptr[(i, j)] = (RE_BP_WX_SELECT_UNCHARGED, ())
                else:
                    re.wx_matrix.set(i, j, wxc)

        # --- 2.2 Composition into VX:  (zhx) ---
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                best_c = re.vxc_matrix.get(i, j)
                best_bp = None

                for (r, k, l) in _iter_complementary_tuples(i, j):
                    L_u = _zhx_collapse_with(re, i, r, k, l, charged=False)
                    R_u = _zhx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=False)
                    L_c = _zhx_collapse_with(re, i, r, k, l, charged=True)
                    R_c = _zhx_collapse_with(re, k + 1, j, l - 1, r + 1, charged=True)

                    coax_e = _coax_energy_for_join(seq, (i, r), (k + 1, j), self.cfg.costs)
                    if self.cfg.enable_coax_variants:
                        coax_e += _coax_energy_for_join(seq, (i, r), (k, l), self.cfg.costs)
                        coax_e += _coax_energy_for_join(seq, (k, l), (k + 1, j), self.cfg.costs)

                    cand_first = Gw + L_u + R_u
                    cand_Lc = L_c + R_u
                    cand_Rc = L_u + R_c
                    cand_both = L_c + R_c

                    cand = min(cand_first, cand_Lc, cand_Rc, cand_both) + g * coax_e + (
                        coax if self.cfg.enable_coax else 0.0)
                    if cand < best_c:
                        best_c = cand
                        best_bp = (RE_BP_COMPOSE_VX, (r, k, l))

                re.vxc_matrix.set(i, j, best_c)
                if best_bp is not None:
                    re.vx_back_ptr[(i, j)] = best_bp

        # Publish final VX as min(uncharged, charged) with selection backpointer
        for s in range(n):
            for i in range(0, n - s):
                j = i + s
                vxu = re.vxu_matrix.get(i, j)
                vxc = re.vxc_matrix.get(i, j)
                if vxu <= vxc:
                    re.vx_matrix.set(i, j, vxu)
                    re.vx_back_ptr[(i, j)] = (RE_BP_VX_SELECT_UNCHARGED, ())
                else:
                    re.vx_matrix.set(i, j, vxc)
                    # keep the charged path’s detailed BP

def _whx_collapse_first(re: RivasEddyState, i: int, j: int, k: int, l: int) -> float:
    """
    Safe accessor for whx(i,j:k,l): try collapse identity first (finite),
    then stored value (which may be +inf if not set).
    """
    v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k, l)
    if math.isfinite(v):
        return v
    return re.whx_matrix.get(i, j, k, l)

def _zhx_collapse_first(re: RivasEddyState, i: int, j: int, k: int, l: int) -> float:
    v = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, k, l)
    if math.isfinite(v): return v
    return re.zhx_matrix.get(i, j, k, l)


def _whx_collapse_with(re: RivasEddyState, i, j, k, l, charged: bool) -> float:
    wx = re.wxc_matrix if charged else re.wxu_matrix
    v = get_whx_with_collapse(re.whx_matrix, wx, i, j, k, l)
    return v if math.isfinite(v) else re.whx_matrix.get(i, j, k, l)


def _zhx_collapse_with(re: RivasEddyState, i, j, k, l, charged: bool) -> float:
    vx = re.vxc_matrix if charged else re.vxu_matrix
    v = get_zhx_with_collapse(re.zhx_matrix, vx, i, j, k, l)
    return v if math.isfinite(v) else re.zhx_matrix.get(i, j, k, l)


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
                yield r, k, l

def _iter_inner_holes(i: int, j: int) -> Iterator[Tuple[int, int]]:
    """Yield all (k,l) with i <= k < l <= j-1 and at least one base inside (i..j)."""
    if j - i <= 1:
        return
    for k in range(i, j):
        for l in range(k + 1, j + 1):
            yield k, l
