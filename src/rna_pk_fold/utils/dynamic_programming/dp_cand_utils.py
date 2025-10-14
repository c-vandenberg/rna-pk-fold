import math
from typing import Tuple, Optional, Callable

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer


def take_best(
    current_best_energy: float,
    current_back_pointer: Optional[EddyRivasBackPointer],
    candidate_energy: float,
    back_pointer_factory: Callable[[], EddyRivasBackPointer],
) -> Tuple[float, Optional[EddyRivasBackPointer]]:
    """
    Update the minimum energy and its corresponding backpointer.

    This is a helper function for the dynamic programming loops. It compares a
    candidate energy with the current best energy. If the candidate is better
    (lower), it updates the best energy and creates a new backpointer using
    the provided factory function.

    Parameters
    ----------
    current_best_energy : float
        The best energy found so far for a given DP state.
    current_back_pointer : Optional[EddyRivasBackPointer]
        The backpointer corresponding to the current best energy.
    candidate_energy : float
        The new energy to compare against the current best.
    back_pointer_factory : Callable[[], EddyRivasBackPointer]
        A zero-argument function that returns a new `EddyRivasBackPointer`
        instance for the candidate case. This lazy creation avoids
        unnecessary object instantiation.

    Returns
    -------
    Tuple[float, Optional[EddyRivasBackPointer]]
        A tuple containing the updated best energy and corresponding backpointer.
    """
    if candidate_energy < current_best_energy:
        return candidate_energy, back_pointer_factory()

    return current_best_energy, current_back_pointer


def try_case(best, best_bp, cand_fn, bp_fn):
    cand = cand_fn()
    if math.isfinite(cand):
        return take_best(best, best_bp, cand, bp_fn)
    return best, best_bp


def publish_best(uncomposed, composed, final, back_ptr_store, select_uncharged_op, *,
                 overlap_enabled=False, overlap_fallback=False, i:int=None, j:int=None):
    wxu = uncomposed.get(i, j)
    wxc = composed.get(i, j)
    if overlap_enabled and overlap_fallback and not math.isfinite(wxc):
        composed.set(i, j, wxu); wxc = wxu
    if wxu <= wxc:
        final.set(i, j, wxu)
        back_ptr_store.set(i, j, EddyRivasBackPointer(op=select_uncharged_op))
    else:
        final.set(i, j, wxc)


def best_cand_check(best: float,
                    best_bp: Optional[EddyRivasBackPointer],
                    cand: float,
                    bp: EddyRivasBackPointer
                   ) -> Tuple[float, Optional[EddyRivasBackPointer]]:
    if cand < best:
        return cand, bp

    return best, best_bp