from __future__ import annotations
from typing import Protocol

from rna_pk_fold.energies.types import SecondaryStructureEnergies

# --- Protocols For Structural Contracts & Dependency Inversion
class HairpinFn(Protocol):
    def __call__(
        self,
        base_i: int,
        base_j: int,
        seq: str,
        energies: SecondaryStructureEnergies,
        temp_k: float
    ) -> float: ...


class StackFn(Protocol):
    def __call__(
        self,
        base_i: int,
        base_j: int,
        base_k: int,
        base_l: int,
        seq: str,
        energies: SecondaryStructureEnergies,
        temp_k: float
    ) -> float: ...


class InternalFn(Protocol):
    def __call__(
        self,
        base_i: int,
        base_j: int,
        base_k: int,
        base_l: int,
        seq: str,
        energies: SecondaryStructureEnergies,
        temp_k: float
    ) -> float: ...


class MultiloopFn(Protocol):
    def __call__(
        self,
        branches: int,
        unpaired: int,
        energies: SecondaryStructureEnergies
    ) -> float: ...
