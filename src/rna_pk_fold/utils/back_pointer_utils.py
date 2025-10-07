from __future__ import annotations
from typing import Optional
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences import EddyRivasBackPointer

def wx_bp(state: EddyRivasFoldState, i: int, j: int) -> Optional[EddyRivasBackPointer]:
    return state.wx_back_ptr.get(i, j)

def whx_bp(state: EddyRivasFoldState, i: int, j: int, k: int, l: int) -> Optional[EddyRivasBackPointer]:
    return state.whx_back_ptr.get(i, j, k, l)

def yhx_bp(state: EddyRivasFoldState, i: int, j: int, k: int, l: int) -> Optional[EddyRivasBackPointer]:
    return state.yhx_back_ptr.get(i, j, k, l)

def zhx_bp(state: EddyRivasFoldState, i: int, j: int, k: int, l: int) -> Optional[EddyRivasBackPointer]:
    return state.zhx_back_ptr.get(i, j, k, l)

def vhx_bp(state: EddyRivasFoldState, i: int, j: int, k: int, l: int) -> Optional[EddyRivasBackPointer]:
    return state.vhx_back_ptr.get(i, j, k, l)
