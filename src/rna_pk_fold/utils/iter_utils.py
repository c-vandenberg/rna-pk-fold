from typing import Iterator, Tuple


def iter_spans(n: int) -> Iterator[Tuple[int, int]]:
    for s in range(n):
        for i in range(0, n - s):
            yield i, i + s


def iter_holes(i: int, j: int) -> Iterator[Tuple[int, int]]:
    max_h = max(0, j - i - 1)
    for h in range(1, max_h + 1):
        for k in range(i, j - h):
            yield k, k + h + 1


def iter_complementary_tuples(i: int, j: int) -> Iterator[Tuple[int, int, int]]:
    # i < k ≤ r < l ≤ j
    for r in range(i + 1, j):
        for k in range(i + 1, r + 1):
            for l in range(r + 1, j + 1):
                yield r, k, l


def iter_inner_holes(i: int, j: int, min_hole: int = 0):
    if j - i <= 1:
        return
    for k in range(i, j):
        for l in range(k + 1 + min_hole, j + 1):
            yield k, l


def iter_holes_pairable(i: int, j: int, can_pair_mask) -> Iterator[Tuple[int, int]]:
    """Yield (k,l) with i <= k < l <= j and can_pair_mask[k][l] is True."""
    max_h = max(0, j - i - 1)
    for h in range(1, max_h + 1):
        for k in range(i, j - h):
            l = k + h + 1
            if can_pair_mask[k][l]:
                yield k, l


def iter_complementary_tuples_pairable(i: int, j: int, can_pair_mask) -> Iterator[Tuple[int, int, int]]:
    """Yield (r,k,l) with i < k <= r < l <= j and can_pair_mask[k][l] is True."""
    for r in range(i + 1, j):
        for k in range(i + 1, r + 1):
            for l in range(r + 1, j + 1):
                if can_pair_mask[k][l]:
                    yield r, k, l