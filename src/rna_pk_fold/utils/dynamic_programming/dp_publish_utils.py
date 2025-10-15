import math

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer


def use_nested_energy_if_composed_infinite(
    enable_overlap_fallback: bool,
    charged_matrix,
    nested_energy: float,
    i_idx: int,
    j_idx: int,
) -> float:
    """
    If overlap fallback is enabled and the composed (pseudoknotted) energy for (i,j)
    is infinite, write the nested energy into the charged matrix and return it.

    Parameters
    ----------
    enable_overlap_fallback : bool
        Whether the overlap fallback behavior is enabled.
    charged_matrix :
        Matrix storing composed (pseudoknotted) energies.
    nested_energy : float
        The best nested (uncharged) energy for (i,j).
    i_idx, j_idx : int
        Span indices.

    Returns
    -------
    float
        The composed energy for (i,j) after fallback is applied (if needed).
    """
    composed_energy = charged_matrix.get(i_idx, j_idx)
    if enable_overlap_fallback and not math.isfinite(composed_energy):
        charged_matrix.set(i_idx, j_idx, nested_energy)
        return nested_energy

    return composed_energy


def publish_min_energy_with_default_backpointer(
    i_idx: int,
    j_idx: int,
    nested_energy: float,
    non_nested_energy: float,
    final_matrix,
    backpointer_store,
    uncharged_op,
) -> None:
    """
    Write min(nested_energy, composed_energy) into out_matrix(i,j).

    If the nested path wins, set the defined 'uncharged' backpointer.
    If the non-nested path wins, keep the backpointer set during composition.

    Parameters
    ----------
    i_idx, j_idx : int
        Span indices.
    nested_energy : float
        Energy of the best nested (uncharged) structure.
    non_nested_energy : float
        Energy of the best composed (pseudoknotted) structure.
    final_matrix :
        Destination matrix for the final selected energy.
    backpointer_store :
        Store for backpointers associated with out_matrix.
    uncharged_op :
        Backtrack op used when the nested (uncharged) path is selected.
    """
    if nested_energy <= non_nested_energy:
        final_matrix.set(i_idx, j_idx, nested_energy)
        backpointer_store.set(i_idx, j_idx, EddyRivasBackPointer(op=uncharged_op))
    else:
        final_matrix.set(i_idx, j_idx, non_nested_energy)
