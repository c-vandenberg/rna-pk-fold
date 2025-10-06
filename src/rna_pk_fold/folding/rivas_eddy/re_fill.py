# rna_pk_fold/folding/rivas_eddy_fill.py  (optional scaffolding module)

from __future__ import annotations
import math
from rna_pk_fold.folding.rivas_eddy.rivas_eddy_matrices import (
    get_whx_with_collapse, get_zhx_with_collapse
)
from rna_pk_fold.folding.fold_state import RivasEddyState

def fill_re_scaffold(seq: str, re: RivasEddyState) -> None:
    """
    Step 11 scaffolding: iterate in a safe topological order without
    implementing the R&E recurrences yet. Leaves +inf for unknowns.
    """
    n = re.n

    # Non-gap base cases (already set in make_re_fold_state, shown here for clarity)
    for i in range(n):
        re.wx_matrix.set(i, i, 0.0)
        re.vx_matrix.set(i, i, math.inf)

    # Outer span
    for s in range(0, n):
        for i in range(0, n - s):
            j = i + s

            # (Optional) make explicit that we haven't filled >0-length segments yet.
            if s > 0:
                # keep +inf placeholders for Step 11
                pass

            # Hole enumeration (sparse! don't prefill; just show order)
            # interior length h = l-k-1; collapse when h == 0 (l == k+1)
            for k in range(i, j):           # k in [i, j-1]
                for l in range(k + 1, j + 1):  # l in [k+1, j]
                    h = (l - k - 1)
                    if h == 0:
                        # collapse case resolved via accessors; don't store
                        _ = get_whx_with_collapse(re.whx_matrix, re.wx_matrix, i, j, k, l)
                        _ = get_zhx_with_collapse(re.zhx_matrix, re.vx_matrix, i, j, k, l)
                    else:
                        # Step 11: leave +inf (no recurrence yet)
                        # (If you want, you can set explicit +inf rows to make debugging easier)
                        # re.whx_matrix.set(i, j, k, l, math.inf)
                        # re.zhx_matrix.set(i, j, k, l, math.inf)
                        # re.yhx_matrix.set(i, j, k, l, math.inf)
                        # re.vhx_matrix.set(i, j, k, l, math.inf)
                        pass
