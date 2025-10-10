from __future__ import annotations
import math

from rna_pk_fold.energies.energy_types import SecondaryStructureEnergies
from rna_pk_fold.utils import calculate_delta_g, lookup_loop_baseline_js, normalize_base, dimer_key
from rna_pk_fold.rules.constraints import MIN_HAIRPIN_UNPAIRED
from rna_pk_fold.utils.nucleotide_utils import pair_str, dangle3_key, dangle5_key

DEFAULT_T_K = 310.15  # 37 °C in Kelvin


def hairpin_energy(
    base_i: int,
    base_j: int,
    seq: str,
    energies: SecondaryStructureEnergies,
    temp_k: float = DEFAULT_T_K
) -> float:
    """
    Calculates the free energy (ΔG) of a hairpin loop.

    This function computes the total free energy for a hairpin loop closed by
    the base pair `(base_i, base_j)`. The calculation includes a length-dependent
    baseline energy, a terminal mismatch penalty for the closing pair, and
    bonuses for special stable hairpins (e.g., tetraloops).

    Parameters
    ----------
    base_i : int
        The 0-based 5' index of the closing base pair.
    base_j : int
        The 0-based 3' index of the closing base pair.
    seq : str
        The RNA sequence.
    energies : SecondaryStructureEnergies
        An object containing the thermodynamic parameter tables.
    temp_k : float, optional
        The temperature in Kelvin for the free energy calculation, by default
        `DEFAULT_TEMP_K`.

    Returns
    -------
    float
        The calculated free energy (ΔG) of the hairpin loop in kcal/mol.
        Returns positive infinity if the loop geometry is invalid.
    """
    # --- 1. Validate Geometry ---
    # Ensure indices are within bounds and the loop is large enough to form.
    if base_i < 0 or base_j >= len(seq) or base_i >= base_j:
        return float("inf")
    hairpin_len = base_j - base_i - 1
    if hairpin_len < MIN_HAIRPIN_UNPAIRED:
        return float("inf")

    # --- 2. Baseline Energy ---
    # Look up the length-dependent baseline ΔH and ΔS values.
    base_hp_dh_ds = lookup_loop_baseline_js(energies.HAIRPIN, hairpin_len)
    if base_hp_dh_ds is None:
        return float("inf")
    # Convert (ΔH, ΔS) to ΔG at the specified temperature.
    delta_g = calculate_delta_g(base_hp_dh_ds, temp_k)

    # --- 3. Terminal Mismatch Penalty ---
    # Get the bases of the closing pair and their immediate neighbors inside the loop.
    base_x = normalize_base(seq[base_i])
    base_y = normalize_base(seq[base_j])
    left_neighbor = normalize_base(seq[base_i + 1])
    right_neighbor = normalize_base(seq[base_j - 1])

    # Construct the key for the mismatch lookup table (e.g., "CA/GU").
    mismatch_key = f"{left_neighbor}{base_x}/{base_y}{right_neighbor}"

    # Preferentially use hairpin-specific mismatch parameters if available.
    hairpin_mismatch_dh_ds = energies.HAIRPIN_MISMATCH.get(mismatch_key)
    if hairpin_mismatch_dh_ds is not None:
        delta_g += calculate_delta_g(hairpin_mismatch_dh_ds, temp_k)
    else:
        # Otherwise, fall back to the general terminal mismatch parameters.
        terminal_mismatch_dh_ds = energies.TERMINAL_MISMATCH.get(mismatch_key)
        if terminal_mismatch_dh_ds is not None:
            delta_g += calculate_delta_g(terminal_mismatch_dh_ds, temp_k)

    # --- 4. Terminal AU/GU Penalty ---
    # Add a destabilizing penalty for weaker AU or GU closing pairs.
    delta_g += _terminal_au_penalty(base_x, base_y)

    return delta_g


def stack_energy(
    base_i: int,
    base_j: int,
    base_k: int,
    base_l: int,
    seq: str,
    energies: SecondaryStructureEnergies,
    temp_k: float = DEFAULT_T_K
) -> float:
    """
    Calculates the stacking free energy (ΔG) between two adjacent base pairs.

    This function looks up the nearest-neighbor thermodynamic parameters for
    the stacking of an outer pair `(base_i, base_j)` on an inner pair `(base_k, base_l)`.

    Parameters
    ----------
    base_i : int
        The 5' index of the outer base pair.
    base_j : int
        The 3' index of the outer base pair.
    base_k : int
        The 5' index of the inner base pair (typically `base_i + 1`).
    base_l : int
        The 3' index of the inner base pair (typically `base_j - 1`).
    seq : str
        The RNA sequence.
    energies : SecondaryStructureEnergies
        An object containing the thermodynamic parameter tables.
    temp_k : float, optional
        The temperature in Kelvin for the free energy calculation, by default
        `DEFAULT_TEMP_K`.

    Returns
    -------
    float
        The calculated stacking energy (ΔG) in kcal/mol. Returns positive
        infinity if the geometry is invalid or parameters are not found.
    """
    # --- 1. Validate Geometry ---
    # Ensure the indices represent a valid stacked pair geometry.
    if not (0 <= base_i < base_k <= base_l < base_j < len(seq)):
        return float("inf")

    # --- 2. Look up Parameters ---
    # Construct the key for the nearest-neighbor stacking table (e.g., "GA/UC").
    key = dimer_key(seq, base_i, base_j)
    if key is None:
        return float("inf")
    # Retrieve the (ΔH, ΔS) tuple for this stacking interaction.
    dh_ds = energies.NN_STACK.get(key)

    # --- 3. Calculate ΔG ---
    # Convert (ΔH, ΔS) to free energy at the specified temperature.
    return calculate_delta_g(dh_ds, temp_k)


def internal_loop_energy(
    base_i: int,
    base_j: int,
    base_k: int,
    base_l: int,
    seq: str,
    energies: SecondaryStructureEnergies,
    temp_k: float = DEFAULT_T_K
) -> float:
    """
    Calculates the free energy (ΔG) of an internal loop or bulge.

    This function determines whether the loop is a bulge (unpaired bases on one
    side) or an internal loop (unpaired bases on both sides) and calculates
    the corresponding free energy.

    Parameters
    ----------
    base_i : int
        The 5' index of the outer closing pair.
    base_j : int
        The 3' index of the outer closing pair.
    base_k : int
        The 5' index of the inner closing pair.
    base_l : int
        The 3' index of the inner closing pair.
    seq : str
        The RNA sequence.
    energies : SecondaryStructureEnergies
        An object containing the thermodynamic parameter tables.
    temp_k : float, optional
        The temperature in Kelvin for the free energy calculation, by default
        `DEFAULT_TEMP_K`.

    Returns
    -------
    float
        The calculated loop energy (ΔG) in kcal/mol. Returns positive infinity
        if the geometry is invalid.
    """
    # --- 1. Validate Geometry and Calculate Loop Sizes ---
    if not (0 <= base_i < base_k <= base_l < base_j < len(seq)):
        return float("inf")

    # The number of unpaired bases on the 5' strand of the loop.
    unpaired_len_5 = base_k - base_i - 1
    # The number of unpaired bases on the 3' strand of the loop.
    unpaired_len_3 = base_j - base_l - 1

    # --- 2. Bulge Loop Case ---
    # A bulge occurs if one side has unpaired bases and the other has none.
    if (unpaired_len_5 == 0) != (unpaired_len_3 == 0):
        bulge_size = unpaired_len_5 + unpaired_len_3
        # Look up the length-dependent baseline energy for the bulge.
        base_dh_ds = lookup_loop_baseline_js(energies.BULGE, bulge_size)
        delta_g = calculate_delta_g(base_dh_ds, temp_k)

        # Apply terminal AU/GU penalties to both the outer and inner closing pairs.
        delta_g += _terminal_au_penalty(normalize_base(seq[base_i]), normalize_base(seq[base_j]))
        delta_g += _terminal_au_penalty(normalize_base(seq[base_k]), normalize_base(seq[base_l]))
        return delta_g

    # --- 3. Internal Loop Case ---
    # An internal loop has unpaired bases on both sides.
    if unpaired_len_5 > 0 and unpaired_len_3 > 0:
        loop_size = unpaired_len_5 + unpaired_len_3

        # For a 1x1 internal loop, there are special mismatch parameters.
        if unpaired_len_5 == 1 and unpaired_len_3 == 1:
            try:
                # Construct the key for the 1x1 internal mismatch table.
                left_motif = normalize_base(seq[base_i + 1]) + normalize_base(seq[base_k - 1])
                right_motif = normalize_base(seq[base_j - 1]) + normalize_base(seq[base_l + 1])
                key = f"{left_motif}/{right_motif}"
                # If special parameters exist, use them directly.
                if key in energies.INTERNAL_MISMATCH:
                    return calculate_delta_g(energies.INTERNAL_MISMATCH[key], temp_k)
            except IndexError:
                # Fall through to the general internal loop calculation if indices are out of bounds.
                pass

        # For all other internal loops, use the general length-dependent baseline.
        base_dh_ds = lookup_loop_baseline_js(energies.INTERNAL, loop_size)
        delta_g = calculate_delta_g(base_dh_ds, temp_k)

        # Apply terminal AU/GU penalties to both closing pairs.
        delta_g += _terminal_au_penalty(normalize_base(seq[base_i]), normalize_base(seq[base_j]))
        delta_g += _terminal_au_penalty(normalize_base(seq[base_k]), normalize_base(seq[base_l]))
        return delta_g

    # If the geometry doesn't match a bulge or internal loop (e.g., a simple stack), return infinity.
    return float("inf")


def multiloop_linear_energy(
    branches: int,
    unpaired_bases: int,
    energies: SecondaryStructureEnergies
) -> float:
    """
    Calculates the free energy (ΔG) of a multiloop using a linear model.

    The energy is a function of a fixed penalty `a`, a penalty per branching
    helix `b`, and a penalty per unpaired nucleotide `c`.

    Parameters
    ----------
    branches : int
        The number of helices branching from the multiloop.
    unpaired_bases : int
        The number of unpaired nucleotides inside the multiloop.
    energies : SecondaryStructureEnergies
        An object containing the thermodynamic parameter tables.

    Returns
    -------
    float
        The calculated multiloop energy (ΔG) in kcal/mol.
    """
    coeff_a, coeff_b, coeff_c, coeff_d = energies.MULTILOOP

    # The 'd' coefficient is a special bonus applied only if there are zero unpaired bases.
    bonus = coeff_d if unpaired_bases == 0 else 0.0

    # Apply the linear model: ΔG = a + b*branches + c*unpaired + bonus.
    return coeff_a + coeff_b * branches + coeff_c * unpaired_bases + bonus


def exterior_end_bonus(
    seq: str,
    base_i: int,
    base_j:  int,
    energies: SecondaryStructureEnergies,
    temp_k: float
) -> float:
    """
    Calculates the most favorable end bonus for an exterior helix.

    This function models the behavior of dangling ends or terminal mismatches
    on a helix that is part of the exterior loop. It computes the energy for
    a 5' dangle, a 3' dangle, a combined 5'+3' dangle, and a terminal mismatch,
    and returns the most stabilizing (most negative) of these options.

    Parameters
    ----------
    seq : str
        The RNA sequence.
    base_i : int
        The 5' index of the exterior helix's closing pair.
    base_j : int
        The 3' index of the exterior helix's closing pair.
    energies : SecondaryStructureEnergies
        An object containing the thermodynamic parameter tables.
    temp_k : float
        The temperature in Kelvin.

    Returns
    -------
    float
        The most stabilizing bonus energy (a negative value or 0.0) in kcal/mol.
    """
    seq_len = len(seq)
    # Get the closing pair as a string (e.g., "GC").
    xy_pair = pair_str(seq, base_i, base_j)
    # Get the neighboring bases in the exterior loop.
    left_base = normalize_base(seq[base_i - 1]) if base_i > 0 else "N"
    right_base = normalize_base(seq[base_j + 1]) if base_j < seq_len - 1 else "N"

    # --- Calculate Energy for Each Possible End Configuration ---
    # 1. Terminal mismatch energy.
    mismatch_key = f"{left_base}{xy_pair[0]}/{xy_pair[1]}{right_base}"
    delta_g_mismatch = calculate_delta_g((energies.TERMINAL_MISMATCH or {}).get(mismatch_key), temp_k)

    # 2. 5' and 3' dangle energies.
    delta_g_dangle5 = calculate_delta_g(energies.DANGLES.get(dangle5_key(left_base, xy_pair)), temp_k)
    delta_g_dangle3 = calculate_delta_g(energies.DANGLES.get(dangle3_key(xy_pair, right_base)), temp_k)

    # --- Determine the Best Option ---
    # Find the minimum (most stabilizing) energy among all options, including doing nothing (0.0).
    best_bonus = min(delta_g_mismatch, delta_g_dangle5 + delta_g_dangle3, delta_g_dangle5, delta_g_dangle3, 0.0)

    return best_bonus if best_bonus != float("inf") else 0.0


def multiloop_close_bonus(
    seq: str,
    base_i: int,
    base_j: int,
    energies: SecondaryStructureEnergies,
    temp_k: float
) -> float:
    """
    Calculates the terminal mismatch bonus for a pair closing a multiloop.

    Parameters
    ----------
    seq : str
        The RNA sequence.
    base_i : int
        The 5' index of the multiloop's closing pair.
    base_j : int
        The 3' index of the multiloop's closing pair.
    energies : SecondaryStructureEnergies
        An object containing the thermodynamic parameter tables.
    temp_k : float
        The temperature in Kelvin.

    Returns
    -------
    float
        The terminal mismatch energy bonus in kcal/mol, or 0.0 if not applicable.
    """
    # The bonus is not applicable if there are no internal bases or no mismatch parameters.
    if base_i + 1 >= base_j or not energies.MULTI_MISMATCH:
        return 0.0

    # Get the closing pair and its adjacent internal bases.
    xy_pair = pair_str(seq, base_i, base_j)
    left_base = normalize_base(seq[base_i + 1])
    right_base = normalize_base(seq[base_j - 1])

    # Construct the key and look up the energy.
    mismatch_key = f"{left_base}{xy_pair[0]}/{xy_pair[1]}{right_base}"
    delta_g = calculate_delta_g(energies.MULTI_MISMATCH.get(mismatch_key), temp_k)

    return 0.0 if delta_g == float("inf") else delta_g


def best_multiloop_end_bonus(
    base_i: int,
    base_k: int,
    seq: str,
    energies: SecondaryStructureEnergies,
    temp_k: float
) -> float:
    """
    Calculates the most favorable end bonus for a helix branching into a multiloop.

    This function considers a two-sided terminal mismatch, a 5' dangle, a 3'
    dangle, and a combined 5'+3' dangle, returning the most stabilizing
    (most negative) energy among these options.

    Parameters
    ----------
    base_i : int
        The 5' index of the branching helix's closing pair.
    base_k : int
        The 3' index of the branching helix's closing pair.
    seq : str
        The RNA sequence.
    energies : SecondaryStructureEnergies
        An object containing the thermodynamic parameter tables.
    temp_k : float
        The temperature in Kelvin.

    Returns
    -------
    float
        The most stabilizing bonus energy in kcal/mol, or 0.0.
    """
    if not energies.MULTI_MISMATCH:
        return 0.0

    # Get the closing pair and its adjacent bases within the multiloop.
    base_x = normalize_base(seq[base_i])
    base_y = normalize_base(seq[base_k])
    left_base = normalize_base(seq[base_i + 1]) if (base_i + 1) < base_k else "E"
    right_base = normalize_base(seq[base_k - 1]) if (base_k - 1) > base_i else "E"

    # --- Calculate Energy for Each Possible End Configuration ---
    # 1. Two-sided multiloop mismatch energy.
    mismatch_key = f"{left_base}{base_x}/{base_y}{right_base}"
    delta_g_mismatch = calculate_delta_g(energies.MULTI_MISMATCH.get(mismatch_key), temp_k)

    # 2. Single-sided 5' and 3' dangle energies.
    dangle_5_key = f"{left_base}./{base_x}{base_y}"
    dangle_3_key = f"{base_x}{base_y}/.{right_base}"
    delta_g_dangle5 = calculate_delta_g(energies.DANGLES.get(dangle_5_key), temp_k)
    delta_g_dangle3 = calculate_delta_g(energies.DANGLES.get(dangle_3_key), temp_k)

    # 3. Combined 5' and 3' dangle energy.
    both_dangles = (delta_g_dangle5 + delta_g_dangle3) if (math.isfinite(delta_g_dangle5) and math.isfinite(delta_g_dangle3)) else float("inf")

    # --- Determine the Best Option ---
    # Find the minimum among all options, including doing nothing (0.0).
    best_bonus = min(
        0.0,
        delta_g_mismatch if math.isfinite(delta_g_mismatch) else float("inf"),
        delta_g_dangle5 if math.isfinite(delta_g_dangle5) else float("inf"),
        delta_g_dangle3 if math.isfinite(delta_g_dangle3) else float("inf"),
        both_dangles
    )

    return 0.0 if best_bonus == float("inf") else best_bonus


def _terminal_au_penalty(base_x: str, base_y: str) -> float:
    """
    Returns a small destabilizing penalty for terminal AU and GU pairs.

    This penalty accounts for the reduced stability of helix ends closed by
    weaker pairs (AU, GU) compared to stronger GC pairs.

    Parameters
    ----------
    base_x : str
        The normalized 5' base of the pair.
    base_y : str
        The normalized 3' base of the pair.

    Returns
    -------
    float
        The penalty in kcal/mol (0.45 for AU/GU pairs, 0.0 otherwise).
    """
    # Combine the bases to form a pair string (e.g., "AU", "GC").
    pair = base_x + base_y

    # Return the penalty if the pair is one of the weaker types.
    if pair in ("AU", "UA", "GU", "UG"):
        return 0.45

    return 0.0
