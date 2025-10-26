import math
from typing import Tuple, Any, Optional, Iterable

from rna_pk_fold.energies.energy_pk_ops import dangle_outer_left, dangle_outer_right
from rna_pk_fold.utils.dynamic_programming.matrix_utils import get_gap_energy_for_named_matrix


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _get_first_attr(obj: Any, candidate_names: Iterable[str]) -> Optional[Any]:
    """
    Return the first present attribute from candidate_names on obj, else None.
    """
    for name in candidate_names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None

# ---------------------------------------------------------------------
# IS2 Bridge Energy Models
# ---------------------------------------------------------------------
def _outer_trim_sizes(i_index: int, j_index: int, r_index: int, s_index: int) -> Tuple[int, int]:
    """
    Number of trimmed/unpaired bases on the outer span when placing a bridge (r, s).

    Returns
    -------
    (n_left, n_right) with:
      n_left  = r - i       (bases removed on 5' side)
      n_right = j - s       (bases removed on 3' side)
    """
    return max(0, r_index - i_index), max(0, j_index - s_index)


def compute_is2_outer_bridge_energy(
    seq: str,
    pk_energies: Any,
    i_index: int,
    j_index: int,
    r_index: int,
    s_index: int,
) -> float:
    """
    Energy for an IS2 "outer bridge" when the OUTER pair (i,j) is already closed.

    Context
    -------
    Used from VHX/ZHX IS2 cases where the inner subproblem is ZHX or VHX and
    the outer pair (i, j) is paired by construction. Because (i,j) is already
    paired, we DO NOT add another ~P_out here (that would double-count).
    We charge for unpaired outer trim, plus a small multiloop-like overhead.

    Model
    -----
      E = (n_left + n_right) * q_tilde_out
          + (m_tilde_vhx + m_tilde_whx)
          + optional dangles at i/j if trimming occurred

    Where:
      n_left  = r - i
      n_right = j - s

    Parameters
    ----------
    seq : str
    pk_energies : PseudoknotEnergies
    i_index, j_index : int
        Outer pair indices (i < j), paired in this context.
    r_index, s_index : int
        Bridge pair indices (i <= r <= k) and (l <= s <= j)

    Returns
    -------
    float : kcal/mol
    """
    n_left, n_right = _outer_trim_sizes(i_index, j_index, r_index, s_index)

    # Per-base outer single-strand penalties
    ss_cost = (n_left + n_right) * float(pk_energies.q_tilde_out)

    # Multiloop-like overhead to connect the outer and inner helices
    ml_cost = float(pk_energies.m_tilde_vhx) + float(pk_energies.m_tilde_whx)

    # If we trimmed at either side, allow a single sequence-aware outer dangle on that side.
    dangle_cost = 0.0
    if n_left > 0:
        dangle_cost += float(dangle_outer_left(seq, i_index, pk_energies))
    if n_right > 0:
        dangle_cost += float(dangle_outer_right(seq, j_index, pk_energies))

    return ss_cost + ml_cost + dangle_cost


def compute_is2_outer_bridge_energy_yhx(
    pk_energies: Any,
    seq: str,
    i_index: int,
    j_index: int,
    r_index: int,
    s_index: int,
) -> float:
    """
    Energy for an IS2 "outer bridge" in the YHX context.

    Context
    -------
    Used when the INNER subproblem is WHX and the OUTER span is handled by YHX
    (i.e., (k,l) is paired, but (i,j) is not fixed at the time of recursion).
    Here we include a ~P_out cost because forming the bridge implies closing (i,j)
    in this pathway, plus outer unpaired penalties and a small multiloop overhead.

    Model
    -----
      E = p_tilde_out
          + (n_left + n_right) * q_tilde_out
          + (m_tilde_yhx + m_tilde_whx)
          + optional dangles at i/j if trimming occurred

    Returns
    -------
    float : kcal/mol
    """
    n_left, n_right = _outer_trim_sizes(i_index, j_index, r_index, s_index)

    # Forming the outer pair in this YHX path (count once)
    pair_cost = float(pk_energies.p_tilde_out)

    # Per-base outer single-strand penalties
    ss_cost = (n_left + n_right) * float(pk_energies.q_tilde_out)

    # Multiloop-like overhead for YHX↔WHX coupling
    ml_cost = float(pk_energies.m_tilde_yhx) + float(pk_energies.m_tilde_whx)

    # Sequence-aware dangles if we trimmed
    dangle_cost = 0.0
    if n_left > 0:
        dangle_cost += float(dangle_outer_left(seq, i_index, pk_energies))
    if n_right > 0:
        dangle_cost += float(dangle_outer_right(seq, j_index, pk_energies))

    return pair_cost + ss_cost + ml_cost + dangle_cost


# ---------------------------------------------------------------------
# Dispatcher + Full Scan
# ---------------------------------------------------------------------
def compute_is2_bridge_energy(
    config: Any,
    seq: str,
    bridge_kind_name: str,
    i_index: int,
    j_index: int,
    r_index: int,
    s_index: int,
) -> float:
    """
    Dispatch to the appropriate IS2 bridge calculator for the requested kind.
    """
    if bridge_kind_name == "yhx":
        return compute_is2_outer_bridge_energy_yhx(
            config, seq, i_index, j_index, r_index, s_index
        )
    raise ValueError(f"Unknown bridge_kind: {bridge_kind_name}")


def scan_is2_outer_bridge_candidates(
    fold_state: Any,
    config: Any,
    seq: str,
    i_index: int,
    j_index: int,
    k_index: int,
    l_index: int,
    inner_matrix_name: str,
    bridge_kind_name: str,
    backtrack_op: Any
) -> Tuple[float, Optional[Tuple[int, int]], Any]:
    """
    Scan r in [i..k], s in [l..j] for for the best IS2 outer-bridge placement.

    Returns:
        (best_energy, (r_best, s_best) or None, backtrack_op)
    """
    best_energy = math.inf
    best_coords: Optional[Tuple[int, int]] = None

    for r_index in range(i_index, k_index + 1):
        for s_index in range(l_index, j_index + 1):
            if r_index > s_index:
                continue

            inner_energy = get_gap_energy_for_named_matrix(
                fold_state, inner_matrix_name, r_index, s_index, k_index, l_index
            )
            if not math.isfinite(inner_energy):
                continue

            bridge = compute_is2_bridge_energy(
                config, seq, bridge_kind_name, i_index, j_index, r_index, s_index
            )
            candidate = bridge + inner_energy
            if candidate < best_energy:
                best_energy = candidate
                best_coords = (r_index, s_index)

    return best_energy, best_coords, backtrack_op
