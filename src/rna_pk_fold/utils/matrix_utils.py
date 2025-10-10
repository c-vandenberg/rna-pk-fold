import math
from typing import Dict, Tuple

from rna_pk_fold.structures.gap_matrix import SparseGapMatrix
from rna_pk_fold.structures.tri_matrix import EddyRivasTriMatrix
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState

# --- Module-level Caches ---
# These global dictionaries are used for memoization to speed up expensive lookups
# during the DP composition phase. They are cleared at the start of each new fold.
_whx_lookup_cache: Dict[Tuple[int, int, int, int, bool], float] = {}
_zhx_lookup_cache: Dict[Tuple[int, int, int, int, bool], float] = {}


def clear_matrix_caches():
    """
    Resets all module-level caches for matrix lookups.

    This function must be called at the beginning of each folding prediction
    to ensure that results from previous runs do not interfere with the current
    calculation.
    """
    global _whx_lookup_cache, _zhx_lookup_cache
    _whx_lookup_cache = {}
    _zhx_lookup_cache = {}


def get_whx_with_collapse(whx_matrix: SparseGapMatrix, wx_matrix: EddyRivasTriMatrix,
                          i: int, j: int, k: int, l: int) -> float:
    """
    Retrieves a value from the WHX matrix, applying the "collapse identity".

    The collapse identity states that `WHX(i, j, k, k+1)` is equivalent to
    `WX(i, j)`. This occurs when the "hole" `(k, l)` has zero width, effectively
    reducing the 4D gapped problem to a 2D nested problem.

    Parameters
    ----------
    whx_matrix : SparseGapMatrix
        The sparse WHX energy matrix.
    wx_matrix : EddyRivasTriMatrix
        The triangular WX energy matrix to use for the collapse case.
    i, j : int
        The indices of the outer span.
    k, l : int
        The indices of the inner hole.

    Returns
    -------
    float
        The appropriate energy value, either from `WHX` or `WX`.
    """
    # If the indices do not form a valid gapped geometry, the energy is infinite.
    if not (i <= k < l <= j):
        return math.inf

    # Check for the collapse condition: a zero-width hole.
    if k + 1 == l:
        # If the hole collapses, return the value from the corresponding 2D WX matrix.
        return wx_matrix.get(i, j)

    # Otherwise, perform a standard lookup in the 4D WHX matrix.
    return whx_matrix.get(i, j, k, l)


def get_zhx_with_collapse(
    zhx_matrix: SparseGapMatrix, vx_matrix: EddyRivasTriMatrix,
    i: int, j: int, k: int, l: int
) -> float:
    """
    Retrieves a value from the ZHX matrix, applying the "collapse identity".

    The collapse identity for ZHX states that `ZHX(i, j, k, k+1)` is equivalent
    to `VX(i, j)`. This is because ZHX requires the outer span `(i, j)` to be
    paired, and when the hole collapses, it becomes a simple pair-enclosed `VX` problem.

    Parameters
    ----------
    zhx_matrix : SparseGapMatrix
        The sparse ZHX energy matrix.
    vx_matrix : EddyRivasTriMatrix
        The triangular VX energy matrix to use for the collapse case.
    i, j : int
        The indices of the outer span.
    k, l : int
        The indices of the inner hole.

    Returns
    -------
    float
        The appropriate energy value, either from `ZHX` or `VX`.
    """
    # If the indices do not form a valid gapped geometry, the energy is infinite.
    if not (i <= k < l <= j):
        return math.inf

    # Check for the collapse condition: a zero-width hole.
    if k + 1 == l:
        # If the hole collapses, return the value from the corresponding 2D VX matrix.
        return vx_matrix.get(i, j)

    # Otherwise, perform a standard lookup in the 4D ZHX matrix.
    return zhx_matrix.get(i, j, k, l)


def get_yhx_with_collapse(
    yhx_matrix: SparseGapMatrix,
    i: int, j: int, k: int, l: int,
    *, invalid_value: float = math.inf
) -> float:
    """
    Retrieves a value from the YHX matrix, handling the collapse case.

    The YHX matrix requires the inner hole `(k, l)` to be paired. When the hole
    collapses (`l == k + 1`), this condition cannot be met. Therefore, the
    collapse case for YHX is always invalid and returns an infinite energy.

    Parameters
    ----------
    yhx_matrix : SparseGapMatrix
        The sparse YHX energy matrix.
    i, j, k, l : int
        The matrix coordinates.
    invalid_value : float, optional
        The value to return for the invalid collapse case, by default `math.inf`.

    Returns
    -------
    float
        The energy from `YHX[i, j, k, l]`, or `invalid_value` for the collapse case.
    """
    if not (i <= j) or not (i <= k < l <= j):
        return math.inf

    # If the hole collapses, the state is invalid by definition.
    if k + 1 == l:
        return invalid_value

    # Otherwise, perform a standard lookup.
    return yhx_matrix.get(i, j, k, l)


def get_vhx_with_collapse(
    vhx_matrix: SparseGapMatrix,
    i: int, j: int, k: int, l: int,
    *, invalid_value: float = math.inf
) -> float:
    """
    Retrieves a value from the VHX matrix, handling the collapse case.

    The VHX matrix requires both the outer span `(i, j)` and the inner hole
    `(k, l)` to be paired. When the hole collapses (`l == k + 1`), the inner
    pair cannot exist. Therefore, the collapse case for VHX is always invalid.

    Parameters
    ----------
    vhx_matrix : SparseGapMatrix
        The sparse VHX energy matrix.
    i, j, k, l : int
        The matrix coordinates.
    invalid_value : float, optional
        The value to return for the invalid collapse case, by default `math.inf`.

    Returns
    -------
    float
        The energy from `VHX[i, j, k, l]`, or `invalid_value` for the collapse case.
    """
    if not (i <= j) or not (i <= k < l <= j):
        return math.inf

    if k + 1 == l:
        return invalid_value
    return vhx_matrix.get(i, j, k, l)


def get_wxi_or_wx(eddy_rivas_state: EddyRivasFoldState, i: int, j: int) -> float:
    """
    Retrieves a value from the multiloop-specific `wxi_matrix` if it exists,
    otherwise falls back to the standard `wx_matrix`.

    Parameters
    ----------
    eddy_rivas_state : EddyRivasFoldState
        The state object containing all DP matrices.
    i, j : int
        The indices of the span.

    Returns
    -------
    float
        The energy value from the appropriate W matrix.
    """
    # Safely access the wxi_matrix attribute.
    wxi_mat  = getattr(eddy_rivas_state, "wxi_matrix", None)

    # If it exists, get the value from it; otherwise, get from the standard wx_matrix.
    return wxi_mat .get(i, j) if wxi_mat is not None else eddy_rivas_state.wx_matrix.get(i, j)


def whx_collapse_with(eddy_rivas_state: EddyRivasFoldState, i, j, k, l, charged: bool,
                      can_pair_mask=None) -> float:
    """
    A cached lookup for WHX values that handles collapse conditions.

    This function provides a memoized way to get the energy for a WHX subproblem.
    It correctly falls back to the nested `WXU` energy if the hole is zero-width
    or if the hole endpoints `(k, l)` cannot form a base pair.

    Parameters
    ----------
    eddy_rivas_state : EddyRivasFoldState
        The state object containing all DP matrices.
    i, j, k, l : int
        The coordinates of the WHX subproblem.
    charged : bool
        Indicates if this is for a "charged" (pseudoknotted) path. This is part
        of the cache key to distinguish contexts, though not used in the logic here.
    can_pair_mask : array-like, optional
        A boolean matrix to check if `(k, l)` can pair, by default None.

    Returns
    -------
    float
        The optimal energy for the subproblem, retrieved from the cache or
        calculated.
    """
    # Create a unique key for the current request.
    cache_key = (i, j, k, l, charged)
    # Return the cached result immediately if it exists.
    if cache_key in _whx_lookup_cache:
        return _whx_lookup_cache[cache_key]

    # basic geometry guard
    if not (i <= j) or not (i <= k < l <= j):
        return math.inf

    collapse = (k + 1 == l)
    if (can_pair_mask is not None) and not collapse:
        # treat unpairable endpoints like a collapse to outer WX
        if not can_pair_mask[k][l]:
            collapse = True

    # --- Collapse Conditions ---
    # Case 1: The hole has zero width.
    if collapse:
        result = eddy_rivas_state.wxc_matrix.get(i, j) if charged else eddy_rivas_state.wxu_matrix.get(i, j)
        if math.isfinite(result):
            _whx_lookup_cache[cache_key] = result
            return result

        result = eddy_rivas_state.whx_matrix.get(i, j, k, l)
        _whx_lookup_cache[cache_key] = result
        return result

    # --- Standard Lookup ---
    # If not a collapse case, get the value from the sparse WHX matrix.
    value = eddy_rivas_state.whx_matrix.get(i, j, k, l)
    # Cache the result before returning.
    _whx_lookup_cache[cache_key] = value
    return value


def zhx_collapse_with(eddy_rivas_state: EddyRivasFoldState, i, j, k, l, charged: bool,
                      can_pair_mask=None) -> float:
    """
    A cached lookup for ZHX values that handles collapse conditions.

    This function provides a memoized way to get the energy for a ZHX subproblem.
    It correctly falls back to the nested `VXU` energy if the hole is zero-width
    or if the hole endpoints `(k, l)` cannot form a base pair.

    Parameters
    ----------
    eddy_rivas_state : EddyRivasFoldState
        The state object containing all DP matrices.
    i, j, k, l : int
        The coordinates of the ZHX subproblem.
    charged : bool
        Indicates if this is for a "charged" path (used in cache key).
    can_pair_mask : array-like, optional
        A boolean matrix to check if `(k, l)` can pair, by default None.

    Returns
    -------
    float
        The optimal energy for the subproblem, retrieved from the cache or
        calculated.
    """
    # Create a unique key for the current request.
    cache_key = (i, j, k, l, charged)

    # Return the cached result immediately if it exists.
    if cache_key in _zhx_lookup_cache:
        return _zhx_lookup_cache[cache_key]

    if not (i <= j) or not (i <= k < l <= j):
        return math.inf

    collapse = (k + 1 == l)
    if (can_pair_mask is not None) and not collapse:
        if not can_pair_mask[k][l]:
            collapse = True

    # --- Collapse Conditions ---
    # Case 1: The hole has zero width. Fall back to the nested, uncharged VX value.
    if collapse:
        result = eddy_rivas_state.vxc_matrix.get(i, j) if charged else eddy_rivas_state.vxu_matrix.get(i, j)
        if math.isfinite(result):
            _zhx_lookup_cache[cache_key] = result
            return result

        result = eddy_rivas_state.zhx_matrix.get(i, j, k, l)
        _zhx_lookup_cache[cache_key] = result
        return result

    # --- Standard Lookup ---
    # If not a collapse case, get the value from the sparse ZHX matrix.
    value = eddy_rivas_state.zhx_matrix.get(i, j, k, l)
    # Cache the result before returning.
    _zhx_lookup_cache[cache_key] = value
    return value