from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Dict, Optional, Tuple, List

Cache = List[List[float]]

BasePairMap = Mapping[str, str]
MultiLoopCoeffs = Tuple[float, float, float, float]
PairEnergies= Dict[str, Tuple[float, float]]
LoopEnergies = Dict[int, Tuple[float, float]]

__all__ = [
    "Cache",
    "BasePairMap",
    "MultiLoopCoeffs",
    "PairEnergies",
    "LoopEnergies",
    "SecondaryStructureEnergies",
]


@dataclass(frozen=True, slots=True)
class SecondaryStructureEnergies:
    """
    Immutable container for all secondary structure thermodynamic
    energy tables the DP layer will query.

    Energies are (ΔH [kcal/mol], ΔS [cal/(K·mol)]).

    Parameters
    ----------
    BULGE : LoopEnergies
        Bulge loop baseline by total loop length (nt).
    COMPLEMENT : BasePairMap
        Map of canonical complements for given nucleic acid class.
    DANGLES : PairEnergies
        Dangling-end contributions where one side has a single unpaired
        nucleotide adjacent to a closing pair.
    HAIRPIN : LoopEnergies
        Hairpin loop baseline as a function of loop length (nt).
    MULTILOOP : MultiLoopCoeffs
        Linear multibranch model coefficients `(a, b, c, d)` used for
        multiloops. Typically, `a + b * (#branches) + c * (#unpaired)`,
        with `d` optionally used when there are zero unpaired nucleotides
        enclosed.
    INTERNAL : LoopEnergies
        Internal loop baseline by total loop length (nt), excluding
        1×1 which are handled via `internal_mm`.
    INTERNAL_MM : PairEnergies
        Internal mismatch nearest-neighbor terms for small internal loops
        (1×1 mismatches).
    NN : PairEnergies
        Minimal nearest-neighbor stack table using `"ij/kl"` keys
        (left 5'→3' dimer `"ij"`, right 3'→5' dimer `"kl"`). These capture
        stacking energetics of adjacent base pairs
        (e.g., `"AU/UA"`, `"GC/CG"`).
    TERMINAL_MM : PairEnergies
        Terminal mismatch penalties/bonuses applied at helix ends.
    SPECIAL_HAIRPINS : PairEnergies, optional
        Sequence-specific hairpin entries

    Notes
    -----
    - The abbreviation "nt" is often used to describe the size or length
      of a DNA or RNA molecule. For example, a 500-nt RNA molecule contains
      500 nucleotides
    """
    BULGE: LoopEnergies
    COMPLEMENT: BasePairMap
    DANGLES: PairEnergies
    HAIRPIN: LoopEnergies
    MULTILOOP: MultiLoopCoeffs
    INTERNAL: LoopEnergies
    INTERNAL_MM: PairEnergies
    NN: PairEnergies
    TERMINAL_MM: PairEnergies
    SPECIAL_HAIRPINS: Optional[PairEnergies] = None

    @staticmethod
    def delta_g(dh: float, ds: float, temp_kelvin: float) -> float:
        return dh - temp_kelvin * (ds / 1000.0)
