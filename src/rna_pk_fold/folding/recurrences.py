from __future__ import annotations
from dataclasses import dataclass
import math

from rna_pk_fold.folding import FoldState, BackPointer, BacktrackOp
from rna_pk_fold.energies import SecondaryStructureEnergyModelProtocol
from rna_pk_fold.rules import can_pair, MIN_HAIRPIN_UNPAIRED


@dataclass(slots=True)
class RecurrenceConfig:
    temp_k: float = 310.15


@dataclass(slots=True)
class SecondaryStructureFoldingEngine:
    energy_model: SecondaryStructureEnergyModelProtocol
    config: RecurrenceConfig

    def fill_all_matrices(self, seq: str, state: FoldState) -> None:
        """
        Fill WM and V bottom-up by span. For each span d:
          1) compute WM[i][j] (uses V on smaller spans)
          2) compute V[i][j]   (can use WM[i+1][j-1])
        """
        n = len(seq)
        a, b, c, d_bonus = self.energy_model.params.MULTILOOP

        # d = 0..n-1 for WM; V is defined for d >= 1
        for d in range(0, n):
            for i in range(0, n - d):
                j = i + d

                # ---------- 1. Matrix WM: Multiloop Accumulator (content between i..j) ----------
                self._fill_wm_cell(seq, i, j, state, b, c)

                # ---------- 2. Matrix V: Paired Spans (only if length >= 2) ----------
                if d >= 1:
                    self._fill_v_cell(seq, i, j, state, a)

                # ---------- 3. Matrix W: ----------
                self._fill_w_cell(i, j, state)

    @staticmethod
    def _fill_wm_cell(seq: str, i: int, j: int,
                      state: FoldState, branch_cost_b: float, unpaired_cost_c: float) -> None:
        wm_matrix = state.wm_matrix
        wm_back_ptr = state.wm_back_ptr
        v_matrix = state.v_matrix

        # Base case already initialized in make_fold_state: WM[i][i] = 0.0
        if i == j:
            wm_back_ptr.set(i, j, BackPointer(operation=BacktrackOp.NONE))
            return

        best = math.inf
        best_bp = BackPointer()

        # Option 1: Leave left base unpaired
        cand_left = wm_matrix.get(i + 1, j) + unpaired_cost_c
        if cand_left < best:
            best = cand_left
            best_bp = BackPointer(operation=BacktrackOp.UNPAIRED_LEFT)

        # Option 2: Leave right base unpaired
        cand_right = wm_matrix.get(i, j - 1) + unpaired_cost_c
        if cand_right < best:
            best = cand_right
            best_bp = BackPointer(operation=BacktrackOp.UNPAIRED_RIGHT)

        # Option 3: attach a helix that starts at i and pairs with k (i < k <= j)
        for k in range(i + 1, j + 1):
            if not can_pair(seq[i], seq[k]):
                continue

            v_ik = v_matrix.get(i, k)
            if math.isinf(v_ik):
                continue

            tail = 0.0 if k + 1 > j else wm_matrix.get(k + 1, j)
            cand = branch_cost_b + v_ik + tail
            if cand < best:
                best = cand
                best_bp = BackPointer(operation=BacktrackOp.MULTI_ATTACH, inner=(i, k), split_k=k, note="attach-helix")

        wm_matrix.set(i, j, best)
        wm_back_ptr.set(i, j, best_bp)

    def _fill_v_cell(self, seq: str, i: int, j: int, state: FoldState, multi_close_a: float) -> None:
        v_matrix = state.v_matrix
        v_back_ptr = state.v_back_ptr

        # If endpoints can't pair, V is +inf
        if not can_pair(seq[i], seq[j]):
            v_matrix.set(i, j, math.inf)
            v_back_ptr.set(i, j, BackPointer())
            return

        best = math.inf
        best_bp = BackPointer()

        # Case 1: hairpin
        delta_g_hp = self.energy_model.hairpin(base_i=i, base_j=j, seq=seq, temp_k=self.config.temp_k)
        if delta_g_hp < best:
            best = delta_g_hp
            best_bp = BackPointer(operation=BacktrackOp.HAIRPIN)

        # Case 2.1: stack (i+1, j-1)
        if i + 1 <= j - 1 and can_pair(seq[i + 1], seq[j - 1]):
            delta_g_stk = self.energy_model.stack(base_i=i, base_j=j, base_k=i + 1, base_l=j - 1,
                                            seq=seq, temp_k=self.config.temp_k)
            if delta_g_stk != math.inf:
                cand = delta_g_stk + v_matrix.get(i + 1, j - 1)
                if cand < best:
                    best = cand
                    best_bp = BackPointer(operation=BacktrackOp.STACK, inner=(i + 1, j - 1))

        # Case 2.2: Internal/bulge loops via all inner pairs (k,l)
        for k in range(i + 1, j):
            for l in range(k + 1, j):
                if not can_pair(seq[k], seq[l]):
                    continue
                delta_g_intl = self.energy_model.internal(base_i=i, base_j=j, base_k=k, base_l=l,
                                                   seq=seq, temp_k=self.config.temp_k)
                if delta_g_intl == math.inf:
                    continue
                cand = delta_g_intl + v_matrix.get(k, l)
                if cand < best:
                    best = cand
                    best_bp = BackPointer(operation=BacktrackOp.INTERNAL, inner=(k, l))

        # Case 3: Close a multiloop: a + WM[i+1][j-1]
        if j - i - 1 >= MIN_HAIRPIN_UNPAIRED:  # only meaningful if there is room inside
            wm_inside = state.wm_matrix.get(i + 1, j - 1)
            cand = multi_close_a + wm_inside
            if cand < best:
                best = cand
                best_bp = BackPointer(operation=BacktrackOp.MULTI_ATTACH, inner=(i + 1, j - 1), note="close-ml")

        v_matrix.set(i, j, best)
        v_back_ptr.set(i, j, best_bp)

    @staticmethod
    def _fill_w_cell(i: int, j: int, state: FoldState) -> None:
        """
        Fill W[i][j] = min(
            W[i+1][j],         # leave i unpaired
            W[i][j-1],         # leave j unpaired
            V[i][j],           # pair i..j
            min_k W[i][k] + W[k+1][j]   # bifurcation
        )
        """
        w_matrix = state.w_matrix
        v_matrix = state.v_matrix
        w_back_ptr = state.w_back_ptr

        # Base case: single cell (i==j) -> nothing to pair; cost 0 by convention
        if i == j:
            w_matrix.set(i, j, 0.0)
            w_back_ptr.set(i, j, BackPointer(operation=BacktrackOp.NONE))
            return

        best = math.inf
        best_back_ptr = BackPointer()

        # Case 1: leave i unpaired
        cand = w_matrix.get(i + 1, j)
        if cand < best:
            best = cand
            best_back_ptr = BackPointer(operation=BacktrackOp.UNPAIRED_LEFT)

        # Case 2: leave j unpaired
        cand = w_matrix.get(i, j - 1)
        if cand < best:
            best = cand
            best_back_ptr = BackPointer(operation=BacktrackOp.UNPAIRED_RIGHT)

        # Case 3: take V[i][j] (only meaningful when span >= 2)
        v_ij = v_matrix.get(i, j)
        if v_ij < best:
            best = v_ij
            best_back_ptr = BackPointer(operation=BacktrackOp.PAIR)

        # Case 4: bifurcation
        for k in range(i, j):
            left = w_matrix.get(i, k)
            right = w_matrix.get(k + 1, j)
            cand = left + right
            if cand < best:
                best = cand
                best_back_ptr = BackPointer(operation=BacktrackOp.BIFURCATION, split_k=k)

        w_matrix.set(i, j, best)
        w_back_ptr.set(i, j, best_back_ptr)
