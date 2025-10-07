from __future__ import annotations
from dataclasses import dataclass
import math

from rna_pk_fold.folding.zucker import ZuckerFoldState, ZuckerBacktrackOp, ZuckerBackPointer
from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModelProtocol
from rna_pk_fold.rules import can_pair, MIN_HAIRPIN_UNPAIRED
from rna_pk_fold.energies.energy_ops import best_multiloop_end_bonus


@dataclass(slots=True)
class ZuckerFoldingConfig:
    temp_k: float = 310.15
    enable_pk_h: bool = False
    pk_h_penalty: float = 1.0


@dataclass(slots=True)
class ZuckerFoldingEngine:
    energy_model: SecondaryStructureEnergyModelProtocol
    config: ZuckerFoldingConfig

    def fill_all_matrices(self, seq: str, state: ZuckerFoldState) -> None:
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

    def _fill_wm_cell(
        self,
        seq: str,
        i: int,
        j: int,
        state: ZuckerFoldState,
        branch_cost_b: float,
        unpaired_cost_c: float
    ) -> None:
        wm_matrix = state.wm_matrix
        wm_back_ptr = state.wm_back_ptr
        v_matrix = state.v_matrix

        # Base case already initialized in make_fold_state: WM[i][i] = 0.0
        if i == j:
            wm_back_ptr.set(i, j, ZuckerBackPointer(operation=ZuckerBacktrackOp.NONE))
            return

        best_energy = math.inf
        best_rank = math.inf
        best_back_ptr = ZuckerBackPointer()

        # Option 1: Leave Left Base Unpaired. (Rank 1).
        cand_energy = wm_matrix.get(i + 1, j) + unpaired_cost_c
        cand_rank = 1
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.UNPAIRED_LEFT)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Option 2: Leave Right Base Unpaired. (Rank 1).
        cand_energy = wm_matrix.get(i, j - 1) + unpaired_cost_c
        cand_rank = 1
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.UNPAIRED_RIGHT)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Option 3: Attach a helix that starts at i and pairs with k (i < k <= j). Rank 0.
        for k in range(i + 1, j + 1):
            if not can_pair(seq[i], seq[k]):
                continue

            v_ik = v_matrix.get(i, k)
            if math.isinf(v_ik):
                continue
            end_bonus = 0.0
            if self.energy_model.params.MULTI_MISMATCH is not None:
                end_bonus = best_multiloop_end_bonus(i, k, seq, self.energy_model.params, self.config.temp_k)

            tail = 0.0 if k + 1 > j else wm_matrix.get(k + 1, j)
            cand_energy = branch_cost_b + v_ik + end_bonus + tail
            cand_rank = 0
            cand_back_ptr = ZuckerBackPointer(
                operation=ZuckerBacktrackOp.MULTI_ATTACH, inner=(i, k), split_k=k, note="attach-helix"
            )
            best_energy, best_rank, best_back_ptr = self._compare_candidates(
                cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
            )

        wm_matrix.set(i, j, best_energy)
        wm_back_ptr.set(i, j, best_back_ptr)

    def _fill_v_cell(self, seq: str, i: int, j: int, state: ZuckerFoldState, multi_close_a: float) -> None:
        v_matrix = state.v_matrix
        v_back_ptr = state.v_back_ptr

        # If endpoints can't pair, V is +inf
        if not can_pair(seq[i], seq[j]):
            v_matrix.set(i, j, math.inf)
            v_back_ptr.set(i, j, ZuckerBackPointer())
            return

        best_energy = math.inf
        best_rank = math.inf
        best_back_ptr = ZuckerBackPointer()

        # Case 1: Hairpin. (Rank 3)
        delta_g_hp = self.energy_model.hairpin(base_i=i, base_j=j, seq=seq, temp_k=self.config.temp_k)
        cand_energy = delta_g_hp
        cand_rank = 3
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.HAIRPIN)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Case 2.1: Stack On (i+1, j-1). (Rank 0)
        if i + 1 <= j - 1 and can_pair(seq[i + 1], seq[j - 1]):
            delta_g_stk = self.energy_model.stack(
                base_i=i, base_j=j, base_k=i + 1, base_l=j - 1, seq=seq, temp_k=self.config.temp_k
            )
            if math.isfinite(delta_g_stk):
                inner = v_matrix.get(i + 1, j - 1)
                cand_energy = delta_g_stk + inner
                cand_rank = 0
                cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.STACK, inner=(i + 1, j - 1))
                best_energy, best_rank, best_back_ptr = self._compare_candidates(
                    cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
                )

        # Case 2.2: Internal/Bulge Loops Via All Inner Pairs (k,l). (Rank 1).
        for k in range(i + 1, j):
            for l in range(k + 1, j):
                if not can_pair(seq[k], seq[l]):
                    continue
                delta_g_intl = self.energy_model.internal(
                    base_i=i, base_j=j, base_k=k, base_l=l, seq=seq, temp_k=self.config.temp_k
                )
                if not math.isfinite(delta_g_intl):
                    continue
                cand_energy = delta_g_intl + v_matrix.get(k, l)
                cand_rank = 1
                cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.INTERNAL, inner=(k, l))
                best_energy, best_rank, best_back_ptr = self._compare_candidates(
                    cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
                )

        # Case 3: Close a Multiloop. (Rank 2).
        if j - i - 1 >= MIN_HAIRPIN_UNPAIRED:
            wm_inside = state.wm_matrix.get(i + 1, j - 1)
            cand_energy = multi_close_a + wm_inside
            cand_rank = 2
            cand_back_ptr = ZuckerBackPointer(
                operation=ZuckerBacktrackOp.MULTI_ATTACH, inner=(i + 1, j - 1), note="close-multiloop"
            )
            best_energy, best_rank, best_back_ptr = self._compare_candidates(
                cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
            )

        v_matrix.set(i, j, best_energy)
        v_back_ptr.set(i, j, best_back_ptr)

    def _fill_w_cell(self, i: int, j: int, state: ZuckerFoldState) -> None:
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
            w_back_ptr.set(i, j, ZuckerBackPointer(operation=ZuckerBacktrackOp.NONE))
            return

        best_energy = math.inf
        best_rank = math.inf
        best_back_ptr = ZuckerBackPointer()

        # Case 1: Leave `i` Unpaired. (Rank 2).
        cand_energy = w_matrix.get(i + 1, j)
        cand_rank = 2
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.UNPAIRED_LEFT)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Case 2: Leave `j` Unpaired. (Rank 2).
        cand_energy = w_matrix.get(i, j - 1)
        cand_rank = 2
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.UNPAIRED_RIGHT)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Case 3: Take V[i][j]. (Rank 0).
        cand_energy = v_matrix.get(i, j)
        cand_rank = 0
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.PAIR)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Case 4: Bifurcation. (Rank 1).
        for k in range(i, j):
            cand_energy = w_matrix.get(i, k) + w_matrix.get(k + 1, j)
            cand_rank = 1
            cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.BIFURCATION, split_k=k)
            best_energy, best_rank, best_back_ptr = self._compare_candidates(
                cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
            )

        w_matrix.set(i, j, best_energy)
        w_back_ptr.set(i, j, best_back_ptr)

    @staticmethod
    def _compare_candidates(
        cand_energy: float,
        cand_rank: float,
        cand_back_ptr: ZuckerBackPointer,
        best_energy: float,
        best_rank: float,
        best_back_ptr: ZuckerBackPointer,
    )-> tuple[float, float, ZuckerBackPointer]:
        """
        Pick the candidate if it has lower energy, or same energy but lower rank.
        Returns the (energy, rank, ZuckerBackPointer) triple to keep as 'best'.
        """
        if (cand_energy < best_energy) or (cand_energy == best_energy and cand_rank < best_rank):
            return cand_energy, cand_rank, cand_back_ptr

        return best_energy, best_rank, best_back_ptr
