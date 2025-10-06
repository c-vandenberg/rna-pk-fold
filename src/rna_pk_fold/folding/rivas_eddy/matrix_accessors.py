import math

from rna_pk_fold.structures.gap_matrix import SparseGapMatrix
from rna_pk_fold.structures.tri_matrix import RivasEddyTriMatrix
from rna_pk_fold.folding.rivas_eddy.re_fold_state import RivasEddyFoldState


def get_whx_with_collapse(whx: SparseGapMatrix, wx: RivasEddyTriMatrix,
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
    zhx: SparseGapMatrix, vx: RivasEddyTriMatrix,
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
    the hole has zero width, so the inner “paired” object doesn’t exist.
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


def wxI(re: RivasEddyFoldState, i: int, j: int) -> float:
    mat = getattr(re, "wxi_matrix", None)
    return mat.get(i, j) if mat is not None else re.wx_matrix.get(i, j)

def whx_collapse_first(re: RivasEddyFoldState, i: int, j: int, k: int, l: int) -> float:
    v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k, l)
    return v if math.isfinite(v) else re.whx_matrix.get(i, j, k, l)

def zhx_collapse_first(re: RivasEddyFoldState, i: int, j: int, k: int, l: int) -> float:
    v = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, k, l)
    return v if math.isfinite(v) else re.zhx_matrix.get(i, j, k, l)

def whx_collapse_with(re: RivasEddyFoldState, i, j, k, l, charged: bool) -> float:
    wx = re.wxc_matrix if charged else re.wxu_matrix
    v = get_whx_with_collapse(re.whx_matrix, wx, i, j, k, l)
    return v if math.isfinite(v) else re.whx_matrix.get(i, j, k, l)

def zhx_collapse_with(re: RivasEddyFoldState, i, j, k, l, charged: bool) -> float:
    vx = re.vxc_matrix if charged else re.vxu_matrix
    v = get_zhx_with_collapse(re.zhx_matrix, vx, i, j, k, l)
    return v if math.isfinite(v) else re.zhx_matrix.get(i, j, k, l)