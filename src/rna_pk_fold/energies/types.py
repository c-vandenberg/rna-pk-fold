from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Dict, Optional, Tuple, List

Cache = List[List[float]]

BasePairMap = Mapping[str, str]
MultiLoopCoeffs = Tuple[float, float, float, float]
PairEnergies = Dict[str, Tuple[float, float]]
LoopEnergies = Dict[int, Tuple[float, float]]
InternalMismatchEnergies = dict[str, dict[str, dict[str, tuple[float, float]]]]
TerminalMismatchEnergies = dict[str, dict[str, dict[str, tuple[float, float]]]]


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
    COMPLEMENT_BASES : BasePairMap
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
    NN_STACK : PairEnergies
        Minimal nearest-neighbor stack table using `"XY/ZW"` keys
        (left 5'→3' dimer `"XY"`, right 3'→5' dimer `"ZW"`). These capture
        stacking energetics of adjacent base pairs
        (e.g., `"AU/UA"`, `"GC/CG"`).
    INTERNAL_MISMATCH : InternalMismatchEnergies
        Internal mismatch nearest-neighbor terms for small internal loops
        (1×1 mismatches).
    TERMINAL_MISMATCH : TerminalMismatchEnergies
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
    COMPLEMENT_BASES: BasePairMap
    DANGLES: PairEnergies
    HAIRPIN: LoopEnergies
    MULTILOOP: MultiLoopCoeffs
    INTERNAL: LoopEnergies
    NN_STACK: PairEnergies
    INTERNAL_MISMATCH: InternalMismatchEnergies
    TERMINAL_MISMATCH: TerminalMismatchEnergies
    SPECIAL_HAIRPINS: Optional[PairEnergies] = None

    @staticmethod
    def delta_g(dh: float, ds: float, temp_kelvin: float) -> float:
        return dh - temp_kelvin * (ds / 1000.0)
