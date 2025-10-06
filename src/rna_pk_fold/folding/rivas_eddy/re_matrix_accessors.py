import math
from rna_pk_fold.folding.fold_state import RivasEddyState
from rna_pk_fold.folding.rivas_eddy.rivas_eddy_matrices import (
    get_whx_with_collapse,
    get_zhx_with_collapse,
)

def wxI(re: RivasEddyState, i: int, j: int) -> float:
    mat = getattr(re, "wxi_matrix", None)
    return mat.get(i, j) if mat is not None else re.wx_matrix.get(i, j)

def _whx_collapse_first(re: RivasEddyState, i: int, j: int, k: int, l: int) -> float:
    v = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, i, j, k, l)
    return v if math.isfinite(v) else re.whx_matrix.get(i, j, k, l)

def _zhx_collapse_first(re: RivasEddyState, i: int, j: int, k: int, l: int) -> float:
    v = get_zhx_with_collapse(re.zhx_matrix, re.vxu_matrix, i, j, k, l)
    return v if math.isfinite(v) else re.zhx_matrix.get(i, j, k, l)

def _whx_collapse_with(re: RivasEddyState, i, j, k, l, charged: bool) -> float:
    wx = re.wxc_matrix if charged else re.wxu_matrix
    v = get_whx_with_collapse(re.whx_matrix, wx, i, j, k, l)
    return v if math.isfinite(v) else re.whx_matrix.get(i, j, k, l)

def _zhx_collapse_with(re: RivasEddyState, i, j, k, l, charged: bool) -> float:
    vx = re.vxc_matrix if charged else re.vxu_matrix
    v = get_zhx_with_collapse(re.zhx_matrix, vx, i, j, k, l)
    return v if math.isfinite(v) else re.zhx_matrix.get(i, j, k, l)