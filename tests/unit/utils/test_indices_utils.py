import pytest

from rna_pk_fold.utils.indices_utils import (
    safe_base,
    canonical_pair,
    is_interval_valid,
    get_default_split_point,
)


# ------------------------------
# safe_base
# ------------------------------
def test_safe_base_in_range_and_boundaries():
    seq = "AUGC"  # 0..3
    assert safe_base(seq, 0) == "A"
    assert safe_base(seq, 3) == "C"


def test_safe_base_out_of_range_and_empty_seq():
    seq = "AUGC"
    assert safe_base(seq, -1) is None
    assert safe_base(seq, 4) is None
    assert safe_base("", 0) is None


# ------------------------------
# canonical_pair
# ------------------------------
@pytest.mark.parametrize(
    "i,j,expected",
    [
        (0, 0, (0, 0)),  # equal
        (1, 5, (1, 5)),  # already ordered
        (7, 2, (2, 7)),  # swapped
        (-3, 4, (-3, 4)),# negative allowed, keep order if <=
        (4, -3, (-3, 4)),# negative + swap
    ],
)
def test_canonical_pair_orders_non_decreasing(i, j, expected):
    assert canonical_pair(i, j) == expected


# ------------------------------
# interval_ok
# ------------------------------
@pytest.mark.parametrize(
    "i,j,n,ok",
    [
        (0, 0, 1, True),      # smallest valid
        (0, 3, 4, True),      # full range
        (2, 2, 5, True),      # single-element interval
        (-1, 2, 5, False),    # i < 0
        (1, -1, 5, False),    # j < 0
        (3, 2, 5, False),     # i > j
        (0, 5, 5, False),     # j == n
        (0, 6, 6, False),     # j > n-1
        (0, 0, 0, False),     # empty domain
    ],
)
def test_interval_ok_various(i, j, n, ok):
    assert is_interval_valid(i, j, n) is ok


# ------------------------------
# split_default
# ------------------------------
def test_split_default_midpoint_no_fallback():
    # Even span
    assert get_default_split_point(0, 4) == 2
    # Odd span floors toward lower
    assert get_default_split_point(1, 4) == 2
    # In general, midpoint must be within [i, j]
    i, j = 5, 11
    m = get_default_split_point(i, j)
    assert i <= m <= j


def test_split_default_respects_fallback():
    # Should ignore midpoint if fallback is provided
    assert get_default_split_point(0, 100, fallback=7) == 7
    # Fallback can be outside [i, j]; function does not clamp
    assert get_default_split_point(10, 20, fallback=-3) == -3
