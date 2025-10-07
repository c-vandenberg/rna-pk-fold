from rna_pk_fold.utils.iter_utils import (
    iter_spans,
    iter_holes,
    iter_complementary_tuples,
    iter_inner_holes,
)


# ------------------------------
# iter_spans
# ------------------------------
def test_iter_spans_n0_is_empty():
    assert list(iter_spans(0)) == []


def test_iter_spans_n1_singleton():
    assert list(iter_spans(1)) == [(0, 0)]


def test_iter_spans_n3_exact_order_and_content():
    # For n=3 the spans by increasing s should be:
    # s=0: (0,0),(1,1),(2,2)
    # s=1: (0,1),(1,2)
    # s=2: (0,2)
    expected = [(0,0),(1,1),(2,2),(0,1),(1,2),(0,2)]
    got = list(iter_spans(3))
    assert got == expected


def test_iter_spans_count_matches_formula():
    for n in range(1, 8):
        got = list(iter_spans(n))
        # number of (i,j) with 0 <= i <= j < n is n*(n+1)/2
        expected_count = n * (n + 1) // 2
        assert len(got) == expected_count
        # all (i,j) are within bounds and ordered
        assert all(0 <= i <= j < n for (i, j) in got)


# ------------------------------
# iter_holes
# ------------------------------
def test_iter_holes_empty_when_interval_too_small():
    # j - i - 1 <= 0 → no holes
    assert list(iter_holes(0, 0)) == []
    assert list(iter_holes(2, 3)) == []  # only one base inside → none


def test_iter_holes_example_i0_j4_exact():
    # For i=0, j=4: max_h = 3; expect:
    # h=1: (0,2),(1,3),(2,4)
    # h=2: (0,3),(1,4)
    # h=3: (0,4)
    expected = [(0,2),(1,3),(2,4),
                (0,3),(1,4),
                (0,4)]
    got = list(iter_holes(0, 4))
    assert got == expected


def test_iter_holes_bounds_and_monotonicity():
    i, j = 3, 9
    for k, l in iter_holes(i, j):
        assert i <= k < l <= j
        # hole width h = l - k - 1 at least 1
        assert (l - k - 1) >= 1


# ------------------------------
# iter_complementary_tuples
# ------------------------------
def test_iter_complementary_tuples_small_example():
    # i=0, j=3
    # r in 1..2
    # r=1: k in 1..1, l in 2..3 → (1,1,2),(1,1,3)
    # r=2: k in 1..2, l in 3..3 → (2,1,3),(2,2,3)
    expected = [(1,1,2),(1,1,3),(2,1,3),(2,2,3)]
    got = list(iter_complementary_tuples(0, 3))
    assert got == expected


def test_iter_complementary_tuples_empty_when_no_room():
    assert list(iter_complementary_tuples(0, 1)) == []  # j=i+1 → empty


def test_iter_complementary_tuples_constraints_hold():
    i, j = 2, 7
    for r, k, l in iter_complementary_tuples(i, j):
        assert (i < k <= r < l <= j)


# ------------------------------
# iter_inner_holes
# ------------------------------
def test_iter_inner_holes_empty_for_too_small_interval():
    assert list(iter_inner_holes(0, 0)) == []
    assert list(iter_inner_holes(5, 6)) == []


def test_iter_inner_holes_no_min_returns_all_pairs_inclusive():
    # i..j inclusive has m = j-i+1 positions
    # here i=0, j=4 → positions 5 → C(5,2)=10 pairs
    expected = [(0,1),(0,2),(0,3),(0,4),
                (1,2),(1,3),(1,4),
                (2,3),(2,4),
                (3,4)]
    got = list(iter_inner_holes(0, 4, min_hole=0))
    assert got == expected


def test_iter_inner_holes_min_hole_1_excludes_adjacents():
    # For i=0, j=4 with min_hole=1, exclude adjacent pairs:
    expected = [(0,2),(0,3),(0,4),
                (1,3),(1,4),
                (2,4)]
    got = list(iter_inner_holes(0, 4, min_hole=1))
    assert got == expected


def test_iter_inner_holes_min_hole_too_large_yields_empty():
    # If min_hole >= (j - i), no (k,l) satisfy l >= k+1+min_hole
    assert list(iter_inner_holes(0, 4, min_hole=4)) == []


def test_iter_inner_holes_bounds_hold():
    i, j = 3, 9
    for k, l in iter_inner_holes(i, j, min_hole=2):
        assert i <= k < l <= j
        assert (l - k - 1) >= 2
