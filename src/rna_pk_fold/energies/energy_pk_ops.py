from __future__ import annotations
from typing import Tuple, Mapping, Protocol, Optional

from rna_pk_fold.energies.energy_types import PseudoknotEnergies
from rna_pk_fold.utils.indices_utils import safe_base
from rna_pk_fold.utils.table_lookup_utils import table_lookup
from rna_pk_fold.utils.energy_pk_utils import coax_energy_for_join


class CoaxConfigLike(Protocol):
    """
    Defines a minimal interface for configuration objects used by coaxial stacking calculations.

    This protocol is used for type hinting to avoid circular dependencies while
    ensuring that the configuration object passed to `coax_pack` has the
    necessary attributes.

    Attributes
    ----------
    enable_coax : bool
        Flag to enable or disable coaxial stacking calculations.
    enable_coax_mismatch : bool
        Flag to enable coaxial stacking for helices separated by a one-base mismatch.
    enable_coax_variants : Optional[bool]
        Flag to enable additional, experimental coaxial stacking geometries.
    """
    enable_coax: bool
    enable_coax_mismatch: bool
    enable_coax_variants: Optional[bool]

# --------------------- DANGLE ENERGIES ---------------------------------------------
def dangle_hole_left(seq: str, k: int, costs: PseudoknotEnergies) -> float:
    """
    Calculates the energy of a 5' dangle inside a pseudoknot hole.

    This corresponds to the energy contribution of the base at `k-1` dangling
    off the 5' side of the inner helix, which starts at `k`.

    Parameters
    ----------
    seq : str
        The RNA sequence.
    k : int
        The 5' index of the inner helix.
    costs : PseudoknotEnergies
        The data object containing pseudoknot energy parameters.

    Returns
    -------
    float
        The dangling end energy contribution in kcal/mol.
    """
    # The dangle involves the base just before the helix (k-1) and the first base of the helix (k).
    return table_lookup(costs.dangle_hole_left or {}, safe_base(seq, k - 1), safe_base(seq, k), costs.l_tilde)

def dangle_hole_right(seq: str, l: int, costs: PseudoknotEnergies) -> float:
    """
    Calculates the energy of a 3' dangle inside a pseudoknot hole.

    This corresponds to the energy contribution of the base at `l+1` dangling
    off the 3' side of the inner helix, which ends at `l`.

    Parameters
    ----------
    seq : str
        The RNA sequence.
    l : int
        The 3' index of the inner helix.
    costs : PseudoknotEnergies
        The data object containing pseudoknot energy parameters.

    Returns
    -------
    float
        The dangling end energy contribution in kcal/mol.
    """
    # The dangle involves the last base of the helix (l) and the base just after it (l+1).
    return table_lookup(costs.dangle_hole_right or {}, safe_base(seq, l), safe_base(seq, l + 1), costs.r_tilde)

def dangle_outer_left(seq: str, i: int, costs: PseudoknotEnergies) -> float:
    """
    Calculates the energy of a 5' dangle on the outer span of a pseudoknot.

    This corresponds to the energy contribution of the base at `i` dangling off
    the 5' side of the outer helix, which effectively starts at `i+1`.

    Parameters
    ----------
    seq : str
        The RNA sequence.
    i : int
        The index of the dangling base.
    costs : PseudoknotEnergies
        The data object containing pseudoknot energy parameters.

    Returns
    -------
    float
        The dangling end energy contribution in kcal/mol.
    """
    # The dangle involves the dangling base (i) and the first base of the helix (i+1).
    return table_lookup(costs.dangle_outer_left or {}, safe_base(seq, i), safe_base(seq, i + 1), costs.l_tilde)

def dangle_outer_right(seq: str, j: int, costs: PseudoknotEnergies) -> float:
    """
    Calculates the energy of a 3' dangle on the outer span of a pseudoknot.

    This corresponds to the energy contribution of the base at `j` dangling off
    the 3' side of the outer helix, which effectively ends at `j-1`.

    Parameters
    ----------
    seq : str
        The RNA sequence.
    j : int
        The index of the dangling base.
    costs : PseudoknotEnergies
        The data object containing pseudoknot energy parameters.

    Returns
    -------
    float
        The dangling end energy contribution in kcal/mol.
    """
    # The dangle involves the last base of the helix (j-1) and the dangling base (j).
    return table_lookup(costs.dangle_outer_right or {}, safe_base(seq, j - 1), safe_base(seq, j), costs.r_tilde)


# --------------------- COAX PACKING ----------------------------------------
def coax_pack(seq: str, i: int, j: int, r: int, k: int, l: int, cfg: CoaxConfigLike,
              costs: PseudoknotEnergies, adjacent: bool) -> Tuple[float, float]:
    """
    Calculates the total coaxial stacking energy for two helices at a pseudoknot seam.

    This function computes the stabilizing energy gained when two helices,
    a left helix `(i, r)` and a right helix `(k+1, j)`, are positioned next to
    each other. It handles both flush (adjacent) and mismatched stacking.

    Parameters
    ----------
    seq : str
        The RNA sequence.
    i, j : int
        The outer indices of the entire construct.
    r : int
        The 3' index of the left helix.
    k, l : int
        The indices of the inner hole.
    cfg : CoaxConfigLike
        A configuration object with flags to control coaxial stacking behavior.
    costs : PseudoknotEnergies
        The data object containing pseudoknot energy parameters.
    adjacent : bool
        True if the two helices are perfectly adjacent (flush).

    Returns
    -------
    Tuple[float, float]
        A tuple `(total_coax_energy, coax_bonus)`, where the first element is the
        sum of all stacking contributions and the second is a fixed bonus term.
    """
    # --- 1. Validate Helix Lengths ---
    # Check if both helices meet the minimum length requirement for coaxial stacking.
    left_helix_len = (r - i + 1)
    right_helix_len = (j - (k + 1) + 1)
    if left_helix_len < costs.coax_min_helix_len or right_helix_len < costs.coax_min_helix_len:
        return 0.0, 0.0

    # --- 2. Validate Seam Geometry ---
    # Check if stacking is a one-base mismatch, if enabled.
    is_mismatch = bool(cfg.enable_coax_mismatch) and (abs(k - r) == 1)
    # Stacking is allowed only if it's enabled and the helices are flush or a valid mismatch.
    is_seam_ok = bool(cfg.enable_coax) and (adjacent or is_mismatch)
    if not is_seam_ok:
        return 0.0, 0.0

    # Get the table of pair-dependent coaxial stacking energies.
    pairs_table: Mapping[Tuple[str, str], float] = costs.coax_pairs or {}

    total_energy = 0.0
    # --- 3. Calculate Stacking Energy for Each Geometry ---
    # Calculate the primary outer-outer seam stacking energy.
    energy = coax_energy_for_join(seq, (i, r), (k + 1, j), pairs_table)

    # If it's a mismatch, apply scaling and bonus penalties/bonuses.
    if is_mismatch:
        energy = energy * costs.mismatch_coax_scale + costs.mismatch_coax_bonus
    # Clamp the energy to be non-positive (stacking can only be stabilizing or neutral).
    if energy > 0.0:
        energy = 0.0
    # Add the scaled energy to the total.
    total_energy += costs.coax_scale_oo * energy

    # If enabled, calculate energies for optional, alternative stacking geometries.
    if getattr(cfg, "enable_coax_variants", False):
        # Outer-inner stacking.
        energy_oi = coax_energy_for_join(seq, (i, r), (k, l), pairs_table)
        if energy_oi > 0.0: energy_oi = 0.0
        total_energy += costs.coax_scale_oi * energy_oi
        # Inner-outer stacking.
        energy_io = coax_energy_for_join(seq, (k, l), (k + 1, j), pairs_table)
        if energy_oi > 0.0: energy_io = 0.0
        total_energy += costs.coax_scale_io * energy_io

    # Return the total calculated energy and the fixed bonus term.
    return total_energy, costs.coax_bonus

# --------------------- SHORT-HOLE CAP --------------------------------------
def short_hole_penalty(costs: PseudoknotEnergies, k: int, l: int) -> float:
    """
    Retrieves the energy penalty for a short loop (hole) between pseudoknot helices.

    This function looks up a pre-defined penalty based on the width of the hole
    `(l - k - 1)`. It is applied once per pseudoknot seam to destabilize
    conformations with very short, sterically unfavorable linkers.

    Parameters
    ----------
    costs : PseudoknotEnergies
        The data object containing pseudoknot energy parameters.
    k : int
        The 5' index of the inner hole.
    l : int
        The 3' index of the inner hole.

    Returns
    -------
    float
        The penalty energy in kcal/mol, or 0.0 if no penalty is defined for
        the given hole width.
    """
    h = l - k - 1
    return (costs.short_hole_caps or {}).get(h, 0.0)
