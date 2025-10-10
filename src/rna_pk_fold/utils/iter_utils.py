from typing import Iterator, Tuple


def iter_spans(n: int) -> Iterator[Tuple[int, int]]:
    """
    Iterates through all possible contiguous spans `(i, j)` for a sequence of length `n`.

    A span is defined by its start and end indices `(i, j)` where `0 <= i <= j < n`.
    The iteration proceeds by increasing span length, from 0 to `n-1`.

    Parameters
    ----------
    n : int
        The length of the sequence.

    Yields
    ------
    Iterator[Tuple[int, int]]
        An iterator that yields `(i, j)` tuples representing all possible spans.
    """
    for span_length  in range(n):
        for i in range(0, n - span_length ):
            yield i, i + span_length


def iter_holes(outer_i: int, outer_j: int) -> Iterator[Tuple[int, int]]:
    """
    Iterates through all possible inner holes `(k, l)` within an outer span `(i, j)`.

    A "hole" is a sub-interval `[k, l]` that is fully contained within the
    outer interval `[i, j]`, satisfying `i <= k < l <= j`. This is a fundamental
    iteration pattern for defining gapped structures in the Eddy-Rivas algorithm.

    Parameters
    ----------
    outer_i : int
        The 5' start index of the outer span.
    outer_j : int
        The 3' end index of the outer span.

    Yields
    ------
    Iterator[Tuple[int, int]]
        An iterator that yields `(k, l)` tuples representing all possible holes.
    """
    max_h = max(0, outer_j - outer_i - 1)
    for h in range(1, max_h + 1):
        for k in range(outer_i, outer_j - h):
            yield k, k + h + 1


def iter_complementary_tuples(outer_i: int, outer_j: int) -> Iterator[Tuple[int, int, int]]:
    """
    Iterates through all `(r, k, l)` tuples satisfying the strict pseudoknot ordering.

    This function generates the core index combinations required for the O(N^6)
    composition step of the Eddy-Rivas algorithm. It finds every valid split
    point `r` and its associated inner hole `(k, l)` that adheres to the
    geometric constraint `i < k <= r < l <= j`.

    Parameters
    ----------
    outer_i : int
        The 5' start index of the outer span.
    outer_j : int
        The 3' end index of the outer span.

    Yields
    ------
    Iterator[Tuple[int, int, int]]
        An iterator that yields `(r, k, l)` tuples, where `r` is the split
        point and `(k, l)` are the hole endpoints.
    """
    # i < k ≤ r < l ≤ j
    for r in range(outer_i + 1, outer_j):
        for k in range(outer_i + 1, r + 1):
            for l in range(r + 1, outer_j + 1):
                yield r, k, l


def iter_inner_holes(outer_i: int, outer_j: int, min_hole_width: int = 0):
    """
    Iterates through all `(k, l)` holes within `[i, j]` with an optional minimum width.

    This is a simpler hole iterator than `iter_holes`, useful for cases where
    the iteration order is less constrained. It directly iterates `k` from `i`
    to `j` and `l` from `k` to `j`.

    Parameters
    ----------
    outer_i : int
        The 5' start index of the outer span.
    outer_j : int
        The 3' end index of the outer span.
    min_hole_width : int, optional
        The minimum number of unpaired bases required inside the hole, by default 0.

    Yields
    ------
    Iterator[Tuple[int, int]]
        An iterator that yields `(k, l)` tuples representing all valid holes.
    """
    if outer_j - outer_i <= 1:
        return
    for k in range(outer_i, outer_j):
        for l in range(k + 1 + min_hole_width, outer_j + 1):
            yield k, l


def iter_holes_pairable(outer_i: int, outer_j: int, can_pair_mask) -> Iterator[Tuple[int, int]]:
    """
    Iterates through all holes `(k, l)` where bases `k` and `l` can form a pair.

    This function is an efficient way to generate only the subset of holes that
    can potentially form the inner helix of a pseudoknot, by pre-filtering
    with a `can_pair_mask`.

    Parameters
    ----------
    outer_i : int
        The 5' start index of the outer span.
    outer_j : int
        The 3' end index of the outer span.
    can_pair_mask : array-like
        A 2D boolean array where `can_pair_mask[k][l]` is True if the bases
        at indices `k` and `l` can form a pair.

    Yields
    ------
    Iterator[Tuple[int, int]]
        An iterator that yields `(k, l)` tuples for all valid, pairable holes.
    """
    max_h = max(0, outer_j - outer_i - 1)
    for h in range(1, max_h + 1):
        for k in range(outer_i, outer_j - h):
            l = k + h + 1
            if can_pair_mask[k][l]:
                yield k, l


def iter_complementary_tuples_pairable_kl(outer_i: int, outer_j: int,
                                          can_pair_mask) -> Iterator[Tuple[int, int, int]]:
    """
    Iterates through `(r, k, l)` tuples where `(k, l)` is a pairable hole.

    This function combines the logic of `iter_complementary_tuples` with a
    pairing filter. It is the primary iterator for the WX/VX composition steps,
    as it generates only the geometrically valid and biochemically plausible
    configurations for forming a pseudoknot.

    Parameters
    ----------
    outer_i : int
        The 5' start index of the outer span.
    outer_j : int
        The 3' end index of the outer span.
    can_pair_mask : array-like
        A 2D boolean array where `can_pair_mask[k][l]` is True if bases `k`
        and `l` can form a pair.

    Yields
    ------
    Iterator[Tuple[int, int, int]]
        An iterator that yields `(r, k, l)` tuples that are both geometrically
        valid and have a pairable `(k, l)` hole.
    """
    for (r, k, l) in iter_complementary_tuples(outer_i, outer_j):
        if can_pair_mask[k][l]:
            yield r, k, l
