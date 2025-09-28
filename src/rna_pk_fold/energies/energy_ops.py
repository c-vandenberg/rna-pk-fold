from __future__ import annotations

from rna_pk_fold.energies.types import SecondaryStructureEnergies
from rna_pk_fold.utils import calculate_delta_g, lookup_loop_anchor, normalize_base, dimer_key
from rna_pk_fold.rules.constraints import MIN_HAIRPIN_UNPAIRED

DEFAULT_T_K = 310.15  # 37 °C in Kelvin


def hairpin_energy(
    base_i: int,
    base_j: int,
    seq: str,
    energies: SecondaryStructureEnergies,
    temp_k: float = DEFAULT_T_K
) -> float:
    """
    Hairpin loop ΔG for a putative closing pair (i, j).

    Computes:
    - loop length L = j - i - 1
    - baseline hairpin contribution from SecondaryStructureEnergies.HAIRPIN
      using a size anchor (clamped)
    - Returns +∞ if L < MIN_HAIRPIN_UNPAIRED (disallowed) or indices invalid.

    Parameters
    ----------
    base_i, base_j : int
        0-based indices with i < j.
    seq : str
        RNA sequence (case-insensitive; T is treated as U).
    energies : SecondaryStructureEnergies
        Parameter bundle.
    temp_k : float, optional
        Temperature in Kelvin (default 310.15 ≈ 37 °C).

    Returns
    -------
    float
        Free energy ΔG (kcal/mol) of the hairpin loop baseline
        (+∞ if invalid).
    """
    if base_i < 0 or base_j >= len(seq) or base_i >= base_j:
        return float("inf")

    hairpin_len = base_j - base_i - 1
    if hairpin_len < MIN_HAIRPIN_UNPAIRED:
        return float("inf")

    base = lookup_loop_anchor(energies.HAIRPIN, hairpin_len)

    return calculate_delta_g(base, temp_k)


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
    Stacking ΔG for adjacent base pairs (i, j) and (k, l)
    where typically k = i+1, l = j-1.

    Uses nearest-neighbor table E.NN with key "XY/ZW" where:
    - X=seq[i], Y=seq[i+1] (left 5'->3')
    - Z=seq[j], W=seq[j-1] (right 3'->5')

    Notes
    -----
    This function returns only the stack term. Terminal mismatches
    and dangles are handled elsewhere (or in a more complete variant).

    Parameters
    ----------
    base_i, base_j, base_k, base_l : int
        Indices with i < k <= l < j in a stacked geometry
        (commonly k=i+1, l=j-1).
    seq : str
        RNA sequence.
    energies : SecondaryStructureEnergies
        Parameter bundle.
    temp_k : float, optional
        Temperature in Kelvin (default 310.15).

    Returns
    -------
    float
        Free energy ΔG (kcal/mol) for the stack
        (+∞ if unavailable/invalid).
    """
    # Basic sanity for a stack geometry
    if not (0 <= base_i < base_k <= base_l < base_j < len(seq)):
        return float("inf")

    key = dimer_key(seq, base_i, base_j)
    if key is None:
        return float("inf")

    dh_ds = energies.NN.get(key)

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
    Internal or bulge loop ΔG for closing pair (i, j) with inner pair (k, l).

    Let:
        a = k - i - 1  # unpaired on the left strand
        b = j - l - 1  # unpaired on the right strand

    Cases
    -----
    - Bulge: one of a or b is 0
        size = a + b
        baseline from SecondaryStructureEnergies.BULGE[size] (clamped to last anchor).
        (Typical refinements like adding the adjacent stack when size==1
        can be added later.)

    - Internal loop: a > 0 and b > 0
        size = a + b
        baseline from SecondaryStructureEnergies.INTERNAL[size] (clamped).
        Special 1×1 mismatch: if a==b==1, try E.INTERNAL_MM key using the
        2-nt motifs adjacent to the closing pair (if present).

    Parameters
    ----------
    base_i, base_j, base_k, base_l : int
        Indices with i < k <= l < j; (i, j) closes the loop and (k, l) is
        the inner pair.
    seq : str
        RNA sequence (T treated as U).
    energies : SecondaryStructureEnergies
        Parameter bundle.
    temp_k : float, optional
        Temperature in Kelvin.

    Returns
    -------
    float
        Free energy ΔG (kcal/mol) for the bulge or internal loop
        (+∞ on invalid geometry or missing data).
    """
    n = len(seq)
    if not (0 <=  base_i < base_k <= base_l < base_j < n):
        return float("inf")

    a = base_k - base_i - 1
    b = base_j - base_l - 1

    # Bulge
    if (a == 0) ^ (b == 0):
        size = a + b
        base = lookup_loop_anchor(energies.BULGE, size)
        return calculate_delta_g(base, temp_k)

    # Internal loop (including 1x1)
    if a > 0 and b > 0:
        size = a + b

        # Special 1×1 internal mismatch if we have the motif:
        if a == 1 and b == 1:
            # Build a key like "XY/ZW" using the two unpaired nucleotides
            # adjacent to the inner pair. A simple representative:
            # left-unpaired next to i -> seq[i+1], right-unpaired next to j -> seq[j-1]
            # combined with inner-pair flank nucleotides: seq[k-1], seq[l+1] if valid.
            try:
                left = normalize_base(seq[base_i + 1]) + normalize_base(seq[base_k - 1])
                right = normalize_base(seq[base_j - 1]) + normalize_base(seq[base_l + 1])
                key = f"{left}/{right}"
                if key in energies.INTERNAL_MM:
                    return calculate_delta_g(energies.INTERNAL_MM[key], temp_k)
            except IndexError:
                # fall back to baseline if neighbors not available
                pass

        base = lookup_loop_anchor(energies.INTERNAL, size)

        return calculate_delta_g(base, temp_k)

    # Not an internal/bulge geometry.
    return float("inf")


def multiloop_linear_energy(
    branches: int,
    unpaired: int,
    energies: SecondaryStructureEnergies
) -> float:
    """
    Linear multiloop model ΔG using SecondaryStructureEnergies.MULTILOOP
    coefficients (a, b, c, d).

    Model (seqfold-style)
    ---------------------
    ΔG = a + b * branches + c * unpaired + (d if unpaired == 0 else 0)

    Notes
    -----
    - SecondaryStructureEnergies.MULTILOOP coefficients are treated as
      free-energy terms (ΔG) rather than (ΔH, ΔS) pairs. Temperature is ignored here.
    - This matches the common "linear multiloop" surrogate used in practice.

    Parameters
    ----------
    branches : int
        Number of entering helices in the multiloop.
    unpaired : int
        Number of unpaired nucleotides inside the multiloop.
    energies : SecondaryStructureEnergies
        Parameter bundle.

    Returns
    -------
    float
        Free energy ΔG (kcal/mol) for the multiloop.
    """
    coeff_a, coeff_b, coeff_c, coeff_d = energies.MULTILOOP
    bonus = coeff_d if unpaired == 0 else 0.0

    return coeff_a + coeff_b * branches + coeff_c * unpaired + bonus
