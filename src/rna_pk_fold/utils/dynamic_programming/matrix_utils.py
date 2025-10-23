import math
from typing import Dict, Tuple, Optional, TypeAlias

from rna_pk_fold.structures.gap_matrix import SparseGapEnergyMatrix
from rna_pk_fold.structures.tri_matrix import EddyRivasTriangularEnergyMatrix
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState

# --- Type Aliases for Cache Keys ---
# An alias for the memory address of a state object, used to keep caches 
# separate between different folding runs.
StateId: TypeAlias = int

# The full tuple used as a key for memoization caches.
# Includes state ID, 4D coordinates, and a flag for the 'charged' (pseudoknotted) context.
# Full memoization key: (state id, i, j, k, l, charged-flag)
CoordKey: TypeAlias = Tuple[StateId, int, int, int, int, bool]

# --- Module-level Caches ---
# These global dictionaries are used for memoization to speed up expensive lookups
# during the DP composition phase. They are cleared at the start of each new fold.
_whx_lookup_cache: Dict[CoordKey, float] = {}
_zhx_lookup_cache: Dict[CoordKey, float] = {}


def clear_matrix_lookup_caches():
    """
    Clear all module-level caches for matrix lookups.

    Call at the start of each folding prediction to avoid cross-run contamination.
    """
    global _whx_lookup_cache, _zhx_lookup_cache
    _whx_lookup_cache = {}
    _zhx_lookup_cache = {}


def get_whx_energy_with_collapse(
    whx_matrix: SparseGapEnergyMatrix,
    wx_matrix: EddyRivasTriangularEnergyMatrix,
    i: int, j: int, k: int, l: int,
) -> float:
    """
    Retrieves a value from the WHX matrix, applying the "collapse identity".

    The collapse identity states that `WHX(i, j, k, k+1)` is equivalent to
    `WX(i, j)`. This occurs when the "hole" `(k, l)` has zero width, effectively
    reducing the 4D gapped problem to a 2D nested problem.

    Parameters
    ----------
    whx_matrix : SparseGapEnergyMatrix
        The sparse WHX energy matrix.
    wx_matrix : EddyRivasTriangularEnergyMatrix
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
        return wx_matrix.get_energy(i, j)

    # Otherwise, perform a standard lookup in the 4D WHX matrix.
    return whx_matrix.get_energy(i, j, k, l)


def get_zhx_energy_with_collapse(
    zhx_matrix: SparseGapEnergyMatrix,
    vx_matrix: EddyRivasTriangularEnergyMatrix,
    i: int, j: int, k: int, l: int
) -> float:
    """
    Retrieves a value from the ZHX matrix, applying the "collapse identity".

    The collapse identity for ZHX states that `ZHX(i, j, k, k+1)` is equivalent
    to `VX(i, j)`. This is because ZHX requires the outer span `(i, j)` to be
    paired, and when the hole collapses, it becomes a simple pair-enclosed `VX` problem.

    Parameters
    ----------
    zhx_matrix : SparseGapEnergyMatrix
        The sparse ZHX energy matrix.
    vx_matrix : EddyRivasTriangularEnergyMatrix
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
        return vx_matrix.get_energy(i, j)

    # Otherwise, perform a standard lookup in the 4D ZHX matrix.
    return zhx_matrix.get_energy(i, j, k, l)


def get_yhx_energy_with_collapse(
    yhx_matrix: SparseGapEnergyMatrix,
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
    yhx_matrix : SparseGapEnergyMatrix
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
    # If the indices do not form a valid gapped geometry, return infinite energy.
    if not (i <= j) or not (i <= k < l <= j):
        return math.inf

    # If the hole collapses, the state is invalid by definition.
    if k + 1 == l:
        return invalid_value

    # Otherwise, perform a standard lookup.
    return yhx_matrix.get_energy(i, j, k, l)


def get_vhx_energy_with_collapse(
    vhx_matrix: SparseGapEnergyMatrix,
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
    vhx_matrix : SparseGapEnergyMatrix
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
    # If the indices do not form a valid gapped geometry, return infinite energy.
    if not (i <= j) or not (i <= k < l <= j):
        return math.inf

    # If the hole collapses, the state is invalid by definition.
    if k + 1 == l:
        return invalid_value

    return vhx_matrix.get_energy(i, j, k, l)


def get_wxi_or_wx(fold_state: EddyRivasFoldState, i: int, j: int) -> float:
    """
    Retrieves a value from the multiloop-specific `wxi_matrix` if it exists,
    otherwise falls back to the standard `wx_matrix`.

    Parameters
    ----------
    fold_state : EddyRivasFoldState
        The state object containing all DP matrices.
    i, j : int
        The indices of the span.

    Returns
    -------
    float
        The energy value from the appropriate W matrix.
    """
    # Safely access the wxi_matrix attribute.
    wxi_matrix = getattr(fold_state, "wxi_matrix", None)

    # If it exists, get the value from it; otherwise, get from the standard wx_matrix.
    return wxi_matrix.get_energy(i, j) if wxi_matrix is not None else fold_state.wx_matrix.get_energy(i, j)


def whx_collapse_with(
    fold_state: EddyRivasFoldState,
    i: int, j: int, k: int, l: int,
    charged: bool,
    can_pair_mask: Optional[list[list[bool]]] = None,
) -> float:
    """
    A cached lookup for WHX values that handles collapse conditions.

    This function provides a memoized way to get the energy for a WHX subproblem.
    It correctly falls back to the nested `WXU` energy if the hole is zero-width
    or if the hole endpoints `(k, l)` cannot form a base pair.

    Parameters
    ----------
    fold_state : EddyRivasFoldState
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
    # Create a unique key for the current request for caching.
    cache_key: CoordKey = (id(fold_state), i, j, k, l, charged)

    # Return the cached result immediately if it exists.
    if cache_key in _whx_lookup_cache:
        return _whx_lookup_cache[cache_key]

    # Check for invalid geometry.
    if not (i <= j) or not (i <= k < l <= j):
        return math.inf

    # Determine if a collapse condition is met (i.e. if hole is zero-width).
    collapse = (k + 1 == l)

    # If a collapse condition is met, get the energy from the appropriate 2D WX matrix.
    if collapse:
        result = fold_state.wxu_matrix.get_energy(i, j)
        if math.isfinite(result):
            _whx_lookup_cache[cache_key] = result
            return result

    # If not a collapse, perform a standard lookup in the sparse 4D WHX matrix and cache the result.
    result = fold_state.whx_matrix.get_energy(i, j, k, l)
    _whx_lookup_cache[cache_key] = result
    return result


def zhx_collapse_with(
    fold_state: EddyRivasFoldState,
    i: int, j: int, k: int, l: int,
    charged: bool,
    can_pair_mask: Optional[list[list[bool]]] = None,
) -> float:
    """
    A cached lookup for ZHX values that handles collapse conditions.

    This function provides a memoized way to get the energy for a ZHX subproblem.
    It correctly falls back to the nested `VXU` energy if the hole is zero-width
    or if the hole endpoints `(k, l)` cannot form a base pair.

    Parameters
    ----------
    fold_state : EddyRivasFoldState
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
    # Create a unique key for the current request for caching.
    cache_key: CoordKey = (id(fold_state), i, j, k, l, charged)

    # Return the cached result immediately if it exists.
    if cache_key in _zhx_lookup_cache:
        return _zhx_lookup_cache[cache_key]

    # Check for invalid geometry.
    if not (i <= j) or not (i <= k < l <= j):
        return math.inf

    # Determine if a collapse condition is met (i.e. if hole is zero-width).
    collapse = (k + 1 == l)

    # If a collapse condition is met, get the energy from the appropriate 2D VX matrix.
    if collapse:
        result = fold_state.vxu_matrix.get_energy(i, j)
        if math.isfinite(result):
            _zhx_lookup_cache[cache_key] = result
            return result

    # If not a collapse, perform a standard lookup in the sparse 4D ZHX matrix and cache the result.
    result = fold_state.zhx_matrix.get_energy(i, j, k, l)
    _zhx_lookup_cache[cache_key] = result

    return result


def get_inner_matrix_energy(state, inner_matrix: str, r: int, s2: int, k: int, l: int) -> float:
    """
    Generic getter for inner-gap matrices by name.
    """
    if inner_matrix == "yhx":
        return state.yhx_matrix.get_energy(r, s2, k, l)
    if inner_matrix == "zhx":
        return state.zhx_matrix.get_energy(r, s2, k, l)
    if inner_matrix == "vhx":
        return state.vhx_matrix.get_energy(r, s2, k, l)
    if inner_matrix == "whx":
        return state.whx_matrix.get_energy(r, s2, k, l)

    raise ValueError(f"Unknown inner_matrix: {inner_matrix}")

