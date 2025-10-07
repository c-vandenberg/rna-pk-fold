from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Tuple

from rna_pk_fold.energies.energy_types import SecondaryStructureEnergies, PseudoknotEnergies
from rna_pk_fold.energies.energy_ops import (
    hairpin_energy, stack_energy, internal_loop_energy, multiloop_linear_energy,
)
from rna_pk_fold.energies.energy_pk_ops import (
    PKCoaxConfig, PKEnergyCosts,
    dangle_hole_left as pk_dangle_hole_left_fn,
    dangle_hole_right as pk_dangle_hole_right_fn,
    dangle_outer_left as pk_dangle_outer_left_fn,
    dangle_outer_right as pk_dangle_outer_right_fn,
    coax_pack as pk_coax_pack_fn,
    short_hole_penalty as pk_short_hole_penalty_fn,
)


class SecondaryStructureEnergyModelProtocol(Protocol):
    params: SecondaryStructureEnergies
    temp_k: float

    # --- Core (Zucker) Secondary Structure Operations ---
    def hairpin(self, base_i: int, base_j: int, seq: str, *, temp_k: Optional[float] = None) -> float: ...
    def stack(self, base_i: int, base_j: int, base_k: int, base_l: int, seq: str, *,
              temp_k: Optional[float] = None) -> float: ...
    def internal(self, base_i: int, base_j: int, base_k: int, base_l: int, seq:str, *,
                 temp_k: Optional[float] = None) -> float: ...
    def multiloop(self, branches:int, unpaired_bases: int) -> float: ...

    # --- Pseudoknot (Eddy & Rivas) Secondary Structure Operations ---
    # Dangles (hole & outer)
    def pk_dangle_hole_left(self, k: int, seq: str, costs: PKEnergyCosts) -> float: ...
    def pk_dangle_hole_right(self, l: int, seq: str, costs: PKEnergyCosts) -> float: ...
    def pk_dangle_outer_left(self, i: int, seq: str, costs: PKEnergyCosts) -> float: ...
    def pk_dangle_outer_right(self, j: int, seq: str, costs: PKEnergyCosts) -> float: ...

    # Coax seam packing (returns (coax_total, coax_bonus_term))
    def pk_coax_pack(self, i: int, j: int, r: int, k: int, l: int, seq: str,
                     cfg: PKCoaxConfig, costs: PKEnergyCosts, adjacent: bool) -> Tuple[float, float]: ...

    # Short-hole penalty (once per seam)
    def pk_short_hole_penalty(self, k: int, l: int, costs: PKEnergyCosts) -> float: ...


@dataclass(frozen=True, slots=True)
class SecondaryStructureEnergyModel:
    """Thin strategy wrapper around energy adapters."""
    params: SecondaryStructureEnergies
    temp_k: float = 310.15  # 37 °C

    def _temp(self, temp_k: Optional[float]) -> float:
        return self.temp_k if temp_k is None else temp_k

    def hairpin(self, base_i: int, base_j: int, seq: str, *, temp_k: Optional[float] = None) -> float:
        return hairpin_energy(base_i, base_j, seq, self.params, self._temp(temp_k))

    def stack(self, base_i: int, base_j: int, base_k: int, base_l: int, seq: str, *,
              temp_k: Optional[float] = None) -> float:
        return stack_energy(base_i, base_j, base_k, base_l, seq, self.params, self._temp(temp_k))

    def internal(self, base_i: int, base_j: int, base_k: int, base_l: int, seq: str, *,
                 temp_k: Optional[float] = None) -> float:
        return internal_loop_energy(base_i, base_j, base_k, base_l, seq, self.params, self._temp(temp_k))

    def multiloop(self, branches: int, unpaired_bases: int) -> float:
        return multiloop_linear_energy(branches, unpaired_bases, self.params)

    def pk_dangle_hole_left(self, k: int, seq: str, costs: Optional[PKEnergyCosts] = None) -> float:
        return pk_dangle_hole_left_fn(seq, k, self._pk_costs(costs))

    def pk_dangle_hole_right(self, l: int, seq: str, costs: Optional[PKEnergyCosts] = None) -> float:
        return pk_dangle_hole_right_fn(seq, l, self._pk_costs(costs))

    def pk_dangle_outer_left(self, i: int, seq: str, costs: Optional[PKEnergyCosts] = None) -> float:
        return pk_dangle_outer_left_fn(seq, i, self._pk_costs(costs))

    def pk_dangle_outer_right(self, j: int, seq: str, costs: Optional[PKEnergyCosts] = None) -> float:
        return pk_dangle_outer_right_fn(seq, j, self._pk_costs(costs))

    def pk_coax_pack(self, i: int, j: int, r: int, k: int, l: int, seq: str,
                     cfg: PKCoaxConfig, costs: Optional[PKEnergyCosts] = None, adjacent: bool = False) -> Tuple[float, float]:
        return pk_coax_pack_fn(seq, i, j, r, k, l, cfg, self._pk_costs(costs), adjacent)

    def pk_short_hole_penalty(self, k: int, l: int, costs: Optional[PKEnergyCosts] = None) -> float:
        return pk_short_hole_penalty_fn(self._pk_costs(costs), k, l)

    def _pk_costs(self, costs: Optional[PKEnergyCosts]) -> PKEnergyCosts:
        if costs is not None:
            return costs
        pk: Optional[PseudoknotEnergies] = self.params.PSEUDOKNOT
        if pk is None:
            raise ValueError("Pseudoknot parameters are not loaded in self.params.PSEUDOKNOT.")
        return pk