from __future__ import annotations
from dataclasses import dataclass

from rna_pk_fold.folding import FoldState, BackPointer, BacktrackOp
from rna_pk_fold.energies import SecondaryStructureEnergyModelProtocol
from rna_pk_fold.rules import can_pair, MIN_HAIRPIN_UNPAIRED


@dataclass(slots=True)
class RecurrenceConfig:
    temp_k: float = 310.15
    # Enable/disable placeholder multiloop in V while WM isn’t implemented:
    enable_multiloop_placeholder: bool = True


@dataclass(slots=True)
class SecondaryStructureFoldingEngine:
    energy_model: SecondaryStructureEnergyModelProtocol
    config: RecurrenceConfig

    def fill_matrix_v(self, seq: str, state: FoldState) -> None:
        seq_len = len(seq)
        v_matrix = state.v_matrix
        v_back_ptr = state.v_back_ptr
        w_matrix = state.w_matrix  # used only by the placeholder

        for d in range(1, seq_len):
            for i in range(0, seq_len - d):
                j = i + d

                best = float("inf")
                best_bp = BackPointer()

                if not can_pair(seq[i], seq[j]):
                    v_matrix.set(i, j, float("inf"))
                    v_back_ptr.set(i, j, best_bp)
                    continue

                # Hairpin
                g_hp = self.energy_model.hairpin(base_i=i, base_j=j, seq=seq, temp_k=self.config.temp_k)
                if g_hp < best:
                    best = g_hp
                    best_bp = BackPointer(operation=BacktrackOp.HAIRPIN)

                # Stack (i+1, j-1)
                if i + 1 <= j - 1 and can_pair(seq[i + 1], seq[j - 1]):
                    g_stk = self.energy_model.stack(base_i=i, base_j=j, base_k=i + 1, base_l=j - 1, seq=seq,
                                                temp_k=self.config.temp_k)
                    if g_stk != float("inf"):
                        cand = g_stk + v_matrix.get(i + 1, j - 1)
                        if cand < best:
                            best = cand
                            best_bp = BackPointer(operation=BacktrackOp.STACK, inner=(i + 1, j - 1))

                # Internal / Bulge loops
                for k in range(i + 1, j):
                    for l in range(k + 1, j):
                        if not can_pair(seq[k], seq[l]):
                            continue
                        g_int = self.energy_model.internal(base_i=i, base_j=j, base_k=k, base_l=l, seq=seq,
                                                       temp_k=self.config.temp_k)
                        if g_int == float("inf"):
                            continue
                        cand = g_int + v_matrix.get(k, l)
                        if cand < best:
                            best = cand
                            best_bp = BackPointer(operation=BacktrackOp.INTERNAL, inner=(k, l))

                # Multiloop placeholder (optional)
                if self.config.enable_multiloop_placeholder and j - i - 1 >= MIN_HAIRPIN_UNPAIRED:
                    penalty = self.energy_model.multiloop(branches=2, unpaired_bases=0)
                    for m in range(i + 1, j - 1):
                        cand = w_matrix.get(i + 1, m) + w_matrix.get(m + 1, j - 1) + penalty
                        if cand < best:
                            best = cand
                            best_bp = BackPointer(operation=BacktrackOp.MULTI_ATTACH, split_k=m)

                v_matrix.set(i, j, best)
                v_back_ptr.set(i, j, best_bp)
