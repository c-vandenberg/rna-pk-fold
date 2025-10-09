import math
from typing import Dict, Tuple

from rna_pk_fold.structures.gap_matrix import SparseGapMatrix
from rna_pk_fold.structures.tri_matrix import EddyRivasTriMatrix
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState

# Caches for expensive lookups
_whx_cache: Dict[Tuple[int, int, int, int, bool], float] = {}
_zhx_cache: Dict[Tuple[int, int, int, int, bool], float] = {}


def clear_matrix_caches():
    """Clear all matrix lookup caches. Call at the start of each fold."""
    global _whx_cache, _zhx_cache
    _whx_cache = {}
    _zhx_cache = {}


def get_whx_with_collapse(whx: SparseGapMatrix, wx: EddyRivasTriMatrix,
                          i: int, j: int, k: int, l: int) -> float:
    """
    Collapse identity: whx(i,j : k,k+1) == wx(i,j)
    Otherwise return whx(i,j:k,l) (default +inf if unset/invalid).
    """
    if not (i <= k < l <= j):
        return math.inf

    if k + 1 == l:
        return wx.get(i, j)

    return whx.get(i, j, k, l)


def get_zhx_with_collapse(
    zhx: SparseGapMatrix, vx: EddyRivasTriMatrix,
    i: int, j: int, k: int, l: int
) -> float:
    """
    Collapse identity: zhx(i,j : k,k+1) == vx(i,j)
    Otherwise return zhx(i,j:k,l).
    """
    if not (i <= k < l <= j):
        return math.inf

    if k + 1 == l:
        return vx.get(i, j)

    return zhx.get(i, j, k, l)


def get_yhx_with_collapse(
    yhx: SparseGapMatrix,
    i: int, j: int, k: int, l: int,
    *, invalid_value: float = math.inf
) -> float:
    """
    No valid collapse: yhx requires the inner (k,l) to be paired; with l==k+1
    the hole has zero width, so the inner "paired" object doesn't exist.
    Return +inf for the collapse case; otherwise return yhx(i,j:k,l).
    """
    if k + 1 == l:
        return invalid_value
    return yhx.get(i, j, k, l)


def get_vhx_with_collapse(
    vhx: SparseGapMatrix,
    i: int, j: int, k: int, l: int,
    *, invalid_value: float = math.inf
) -> float:
    """
    No valid collapse: vhx expects both outer and inner paired; the inner
    disappears when l==k+1. Return +inf in the collapse case; otherwise get().
    """
    if k + 1 == l:
        return invalid_value
    return vhx.get(i, j, k, l)


def wxI(re: EddyRivasFoldState, i: int, j: int) -> float:
    mat = getattr(re, "wxi_matrix", None)
    return mat.get(i, j) if mat is not None else re.wx_matrix.get(i, j)


def whx_collapse_first(re: EddyRivasFoldState, i: int, j: int, k: int, l: int) -> float:
    v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k, l)
    return v if math.isfinite(v) else re.whx_matrix.get(i, j, k, l)


def zhx_collapse_first(re: EddyRivasFoldState, i: int, j: int, k: int, l: int) -> float:
    v = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, k, l)
    return v if math.isfinite(v) else re.zhx_matrix.get(i, j, k, l)


def whx_collapse_with(re: EddyRivasFoldState, i, j, k, l, charged: bool, can_pair_mask=None) -> float:
    """
    Get WHX value with collapse optimization and caching.
    Falls back to WXU for zero-width holes OR non-pairable holes.
    """
    # Check cache first
    cache_key = (i, j, k, l, charged)
    if cache_key in _whx_cache:
        return _whx_cache[cache_key]

    # Collapse case 1: zero-width hole
    if k + 1 == l:
        result = re.wxu_matrix.get(i, j)
        _whx_cache[cache_key] = result
        return result

    # Collapse case 2: non-pairable hole (if mask provided)
    if can_pair_mask is not None and not can_pair_mask[k][l]:
        result = re.wxu_matrix.get(i, j)
        _whx_cache[cache_key] = result
        return result

    # Check sparse matrix (was computed)
    v = re.whx_matrix.get(i, j, k, l)
    if math.isfinite(v):
        _whx_cache[cache_key] = v
        return v

    # No valid value - return inf
    _whx_cache[cache_key] = math.inf
    return math.inf


def zhx_collapse_with(re: EddyRivasFoldState, i, j, k, l, charged: bool, can_pair_mask=None) -> float:
    """
    Get ZHX value with collapse optimization and caching.
    Falls back to VXU for zero-width holes OR non-pairable holes.
    """
    # Check cache first
    cache_key = (i, j, k, l, charged)
    if cache_key in _zhx_cache:
        return _zhx_cache[cache_key]

    # Collapse case 1: zero-width hole
    if k + 1 == l:
        result = re.vxu_matrix.get(i, j)  # Always use uncharged for collapse
        _zhx_cache[cache_key] = result
        return result

    # Collapse case 2: non-pairable hole (if mask provided)
    if can_pair_mask is not None and not can_pair_mask[k][l]:
        result = re.vxu_matrix.get(i, j)
        _zhx_cache[cache_key] = result
        return result

    # Check sparse matrix (was computed)
    v = re.zhx_matrix.get(i, j, k, l)
    if math.isfinite(v):
        _zhx_cache[cache_key] = v
        return v

    # No valid value - return inf
    _zhx_cache[cache_key] = math.inf
    return math.inf