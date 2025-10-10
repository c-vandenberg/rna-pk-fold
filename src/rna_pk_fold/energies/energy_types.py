from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Dict, Optional, Tuple, List

# A mapping from a base to its canonical complement, e.g., {"A": "U", "C": "G"}.
BasePairMap = Mapping[str, str]

# A tuple of the four linear model coefficients (a, b, c, d) for multiloop energy.
MultiLoopCoeffs = Tuple[float, float, float, float]

# A dictionary mapping a key (e.g., a stacking dimer "AU/UA") to its (ΔH, ΔS) values.
PairEnergies = Dict[str, Tuple[float, float]]

# A dictionary mapping a loop length (integer) to its (ΔH, ΔS) values.
LoopEnergies = Dict[int, Tuple[float, float]]

# A dictionary mapping a bigram of bases to a float energy value.
BigramBaseEnergyMap = Dict[Tuple[str, str], float]

# A tuple representing a type of coaxial stack, e.g., ("GC", "CG").
PairType = Tuple[str, str]


@dataclass(frozen=True, slots=True)
class SecondaryStructureEnergies:
    """
    Immutable container for all nested secondary structure thermodynamic parameters.

    This data class aggregates all the energy tables and coefficients required
    by a Zuker-style dynamic programming algorithm. The parameters are typically
    loaded from a YAML file and represent enthalpy (ΔH) and entropy (ΔS) values
    for various structural motifs.

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
        """
        Calculates the free energy (ΔG) from enthalpy (ΔH) and entropy (ΔS).

        This static method applies the Gibbs free energy equation:
        ΔG = ΔH - T * ΔS.

        Parameters
        ----------
        delta_h : float
            The enthalpy change in kcal/mol.
        delta_s : float
            The entropy change in cal/(K·mol).
        temp_k : float
            The absolute temperature in Kelvin.

        Returns
        -------
        float
            The calculated free energy change (ΔG) in kcal/mol. Returns
            positive infinity if either ΔH or ΔS is None.
        """
        if delta_h is None or delta_s is None:
            return float("inf")

        return delta_h - temp_k * (delta_s / 1000.0)


@dataclass(frozen=True, slots=True)
class PseudoknotEnergies:
    """
    Immutable container for all parameters used by the Eddy-Rivas recurrences.

    This data class aggregates all the specialized energy penalties, bonuses,
    and scaling factors required by the pseudoknot-aware folding algorithm.
    These parameters are typically defined in a dedicated 'pseudoknot' section
    of the main energy YAML file.

    Attributes
    ----------
    q_ss : float
        Penalty for an unpaired single-stranded base in a pseudoknot context.
    p_tilde_out, p_tilde_hole : float
        Penalties for base pairs in the outer span or adjacent to the hole.
    q_tilde_out, q_tilde_hole : float
        Penalties for unpaired bases in the outer span or within the hole.
    l_tilde, r_tilde : float
        Default energy values for 5' and 3' dangles.
    m_tilde_yhx, m_tilde_vhx, m_tilde_whx : float
        Multiloop initiation penalties for different gap matrix contexts.
    dangle_hole_left, dangle_hole_right : Optional[BigramBaseEnergyMap]
        Sequence-dependent energies for dangles inside the pseudoknot hole.
    dangle_outer_left, dangle_outer_right : Optional[BigramBaseEnergyMap]
        Sequence-dependent energies for dangles on the outer span.
    coax_pairs : Optional[BigramBaseEnergyMap]
        Energies for coaxial stacking between different types of base pairs.
    coax_bonus, coax_scale, coax_min_helix_len : float, float, int
        Parameters controlling the coaxial stacking model.
    mismatch_coax_scale, mismatch_coax_bonus : float
        Parameters for coaxial stacking over a one-base mismatch.
    join_drift_penalty : float
        Penalty for an experimental feature allowing hole positions to shift.
    short_hole_caps : Optional[Dict[int, float]]
        Penalties for sterically unfavorable short linkers between helices.
    g_wh, g_wi, g_wh_wx, g_wh_whx : float
        Penalties for initiating various types of overlapping or internal pseudoknots.
    pk_penalty_gw : float
        The main penalty for introducing a new pseudoknot.
    """
    # --- Scalar Penalties for Pseudoknot Contexts (Tilde Parameters) ---
    q_ss: float
    p_tilde_out: float
    p_tilde_hole: float
    q_tilde_out: float
    q_tilde_hole: float
    l_tilde: float
    r_tilde: float
    m_tilde_yhx: float
    m_tilde_vhx: float
    m_tilde_whx: float

    # --- Sequence-Dependent Dangle and Coaxial Stacking Tables ---
    dangle_hole_left: Optional[BigramBaseEnergyMap] = None
    dangle_hole_right: Optional[BigramBaseEnergyMap] = None
    dangle_outer_left: Optional[BigramBaseEnergyMap] = None
    dangle_outer_right: Optional[BigramBaseEnergyMap] = None
    coax_pairs: Optional[BigramBaseEnergyMap] = None

    # --- Coaxial Stacking Control Parameters ---
    coax_bonus: float = 0.0
    coax_scale_oo: float = 1.0  # Outer-Outer seam scaling
    coax_scale_oi: float = 1.0  # Outer-Inner seam scaling
    coax_scale_io: float = 1.0  # Inner-Outer seam scaling
    coax_min_helix_len: int = 1
    coax_scale: float = 1.0

    # --- Mismatched Coaxial Stacking Parameters ---
    mismatch_coax_scale: float = 0.5
    mismatch_coax_bonus: float = 0.0

    # --- Penalties for Specific Geometries ---
    join_drift_penalty: float = 0.0
    short_hole_caps: Optional[Dict[int, float]] = None

    # --- Global Composition and Overlap Penalties (G-values) ---
    g_wh: float = 0.0
    g_wi: float = 0.0
    g_wh_wx: float = 0.0
    g_wh_whx: float = 0.0

    # The main penalty for introducing a pseudoknot structure.
    pk_penalty_gw: float = 1.0
