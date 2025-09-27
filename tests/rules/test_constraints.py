import pytest

from rna_pk_fold.rules.constraints import (
    can_pair,
    hairpin_size,
    is_min_hairpin_size,
    MIN_HAIRPIN_UNPAIRED,
)


def test_can_pair_allows_watson_crick_and_wobble():
    """
    Test whether canonical and wobble pairs are allowed.

    Notes
    -----
    Allowed (RNA): AU, UA, GC, CG, GU, UG.
    """
    allowed = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")]
    for i, j in allowed:
        assert can_pair(i, j) is True


def test_can_pair_is_case_insensitive_and_handles_T_as_U():
    """
    Verify bases are normalized to uppercase and T is treated as U.
    """
    # Case-insensitive
    assert can_pair("a", "u")
    assert can_pair("G", "c")
    # T behaves like U
    assert can_pair("A", "T")  # A–U via T->U mapping
    assert can_pair("t", "A")  # U–A via T->U mapping


def test_can_pair_rejects_invalid_inputs_and_ambiguity():
    """
    Ensure invalid or ambiguous inputs fail fast.
    """
    # Non-strings
    assert can_pair(None, "A") is False
    assert can_pair("A", 3) is False
    # Wrong length
    assert can_pair("AU", "A") is False
    assert can_pair("A", "UA") is False
    # Ambiguity code
    assert can_pair("N", "A") is False
    assert can_pair("A", "N") is False
    # Disallowed pairs
    assert can_pair("A", "G") is False
    assert can_pair("C", "U") is False


@pytest.mark.parametrize(
    "base_i,base_j,expected",
    [
        (0, 1, 0),
        (0, 2, 1),
        (2, 6, 3),
        (3, 7, 3),
    ],
)
def test_hairpin_size_formula(base_i, base_j, expected):
    """
    Check the hairpin size formula ``j - i - 1``.
    """
    assert hairpin_size(base_i, base_j) == expected


def test_is_min_hairpin_size_uses_default_threshold():
    """
    Validate default minimum hairpin size threshold.
    """
    i, j = 0, 4  # loop length = 3
    assert MIN_HAIRPIN_UNPAIRED == 3
    assert is_min_hairpin_size(i, j) is True
    assert is_min_hairpin_size(0, 3) is False  # loop length = 2


def test_is_min_hairpin_size_with_custom_threshold():
    """
    Validate custom minimum threshold argument.
    """
    assert is_min_hairpin_size(0, 4, min_unpaired=4) is False
    assert is_min_hairpin_size(0, 5, min_unpaired=4) is True
