import math

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer


def fallback_composed_if_inf(
    enable_fallback: bool,
    charged_matrix,
    uncharged_energy: float,
    i: int,
    j: int
) -> float:
    """
    This is a fallback mechanism. If the overlap feature is enabled but no finite-energy
    pseudoknot was found (wxc is infinity), we consider the uncharged (nested) energy as
    the best possible 'composed' energy.
    """
    charged_energy = charged_matrix.get(i, j)
    if enable_fallback and not math.isfinite(charged_energy):
        charged_matrix.set(i, j, uncharged_energy)
        return uncharged_energy
    return charged_energy


def select_and_publish_min(
    i: int,
    j: int,
    uncharged_energy: float,
    charged_energy: float,
    final_matrix,
    backptr_store,
    uncharged_op
) -> None:
    """
    Write min(uncomposed_val, composed_val) into final_matrix(i,j).
    If the winner is 'uncomposed', set the default 'uncharged' backpointer.
    If the winner is 'composed', keep the backpointer already set during composition.
    """
    # If the nested structure is more stable (or equally stable), select it and set a
    # backpointer indicating that the uncharged (nested) path was chosen.
    if uncharged_energy <= charged_energy:
        final_matrix.set(i, j, uncharged_energy)
        backptr_store.set(i, j, EddyRivasBackPointer(op=uncharged_op))
    else:
        # If the pseudoknotted structure is more stable, select it. The backpointer for this
        # case was already set in _compose_wx (i.e. do nothing)
        final_matrix.set(i, j, charged_energy)
