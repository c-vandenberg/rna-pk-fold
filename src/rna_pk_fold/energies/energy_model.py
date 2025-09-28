from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional

from rna_pk_fold.energies.types import SecondaryStructureEnergies
from rna_pk_fold.energies.energy_ops import (
    hairpin_energy, stack_energy, internal_loop_energy, multiloop_linear_energy,
)

class SecondaryStructureEnergyModelProtocol(Protocol):
    params: SecondaryStructureEnergies
    temp_k: float

    def hairpin(self, base_i: int, base_j: int, seq: str, *, temp_k: Optional[float] = None) -> float: ...
    def stack(self, base_i: int, base_j: int, base_k: int, base_l: int, seq: str, *,
              temp_k: Optional[float] = None) -> float: ...
    def internal(self, base_i: int, base_j: int, base_k: int, base_l: int, seq:str, *,
                 temp_k: Optional[float] = None) -> float: ...
    def multiloop(self, branches:int, unpaired_bases: int) -> float: ...


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
