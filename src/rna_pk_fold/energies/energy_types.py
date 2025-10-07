from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Dict, Optional, Tuple, List

Cache = List[List[float]]

BasePairMap = Mapping[str, str]
MultiLoopCoeffs = Tuple[float, float, float, float]
PairEnergies = Dict[str, Tuple[float, float]]
LoopEnergies = Dict[int, Tuple[float, float]]

BigramBaseEnergyMap = Dict[Tuple[str, str], float]  # e.g., ("G","A") for adjacent nts
PairType = Tuple[str, str] # e.g., ("GC","CG") for coax pair types


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
    INTERNAL_MISMATCH : PairEnergies
        Internal mismatch nearest-neighbor terms for small internal loops
        (1×1 mismatches).
    TERMINAL_MISMATCH : PairEnergies
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
    INTERNAL_MISMATCH: PairEnergies
    TERMINAL_MISMATCH: PairEnergies
    HAIRPIN_MISMATCH: Optional[PairEnergies] = None
    MULTI_MISMATCH: Optional[PairEnergies] = None
    SPECIAL_HAIRPINS: Optional[PairEnergies] = None
    PSEUDOKNOT: Optional[PseudoknotEnergies] = None

    @staticmethod
    def delta_g(delta_h: float, delta_s: float, temp_k: float) -> float:
        if delta_h is None or delta_s is None:
            return float("inf")

        return delta_h - temp_k * (delta_s / 1000.0)


@dataclass(frozen=True, slots=True)
class PseudoknotEnergies:
    """
    Immutable container of scalar/tables used by the Rivas–Eddy recurrences.
    These mirror the fields your RE costs/config expect, but live in the
    'energies' layer so they can be loaded/validated like the Zuker params.
    """
    # Gap + tilde scalars
    q_ss: float
    P_tilde_out: float
    P_tilde_hole: float
    Q_tilde_out: float
    Q_tilde_hole: float
    L_tilde: float
    R_tilde: float
    M_tilde_yhx: float
    M_tilde_vhx: float
    M_tilde_whx: float

    # dangle/coax tables (simple bigram maps; keys like "AU", "CG", etc.)
    dangle_hole_left: Optional[BigramBaseEnergyMap] = None # (k-1, k)
    dangle_hole_right: Optional[BigramBaseEnergyMap] = None # (l, l+1)
    dangle_outer_left: Optional[BigramBaseEnergyMap] = None # (i, i+1)
    dangle_outer_right: Optional[BigramBaseEnergyMap] = None # (j-1, j)

    # coax pairs keyed as "XY|ZW" (string pair keys) -> ΔG
    coax_pairs: Optional[BigramBaseEnergyMap] = None        # ("GC","CG") → ΔG

    # Coax controls
    coax_bonus: float = 0.0
    coax_scale_oo: float = 1.0
    coax_scale_oi: float = 1.0
    coax_scale_io: float = 1.0
    coax_min_helix_len: int = 1
    coax_scale: float = 1.0

    mismatch_coax_scale: float = 0.5
    mismatch_coax_bonus: float = 0.0

    # Join drift & short-hole caps
    join_drift_penalty: float = 0.0
    short_hole_caps: Optional[Dict[int, float]] = None

    # Overlap/inner penalties
    Gwh: float = 0.0
    Gwi: float = 0.0
    Gwh_wx: float = 0.0
    Gwh_whx: float = 0.0

    # Pseudoknot introduction penalty used by WX/VX composition
    pk_penalty_gw: float = 1.0
