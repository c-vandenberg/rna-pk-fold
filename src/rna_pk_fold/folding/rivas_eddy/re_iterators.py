from typing import Iterator, Tuple

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