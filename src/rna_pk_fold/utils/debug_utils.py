import math


def debug_enabled(cfg) -> bool:
    return bool(getattr(cfg, "verbose", False))


def debug_print(cfg, *a, **kw):
    if debug_enabled(cfg):
        print(*a, **kw)


def debug_cell(cfg, cell, target) -> bool:
    return debug_enabled(cfg) and cell == target


def count_finite_cells(gap_matrix) -> int:
    total = 0
    for holes in gap_matrix.data.values():
        for v in holes.values():
            if math.isfinite(v): total += 1
    return total
