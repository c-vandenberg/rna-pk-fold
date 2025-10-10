"""
Unit tests for fundamental RNA secondary structure rules and constraints.

This module validates the helper functions that enforce the basic biophysical
rules of RNA folding, such as which nucleotides can form pairs and the minimum
allowable size for a hairpin loop.
"""
import pytest

from rna_pk_fold.rules.constraints import (
    can_pair,
    hairpin_size,
    is_min_hairpin_size,
    MIN_HAIRPIN_UNPAIRED,
)


def test_can_pair_allows_watson_crick_and_wobble():
    """
    Tests that `can_pair` correctly identifies all canonical and wobble base pairs.
    """
    # Define the list of standard allowed RNA base pairs.
    # This includes Watson-Crick pairs (A-U, G-C) and the G-U wobble pair.
    allowed = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")]
    for i, j in allowed:
        assert can_pair(i, j) is True


def test_can_pair_is_case_insensitive_and_handles_T_as_U():
    """
    Verifies that the `can_pair` function normalizes inputs for robustness.
    It should handle lowercase letters and treat Thymine (T) as Uracil (U),
    which is useful for handling DNA sequences or mixed-format inputs.
    """
    # Test case-insensitivity.
    assert can_pair("a", "u")
    assert can_pair("G", "c")
    # Test that 'T' is treated as 'U' for pairing purposes.
    assert can_pair("A", "T")  # Should be treated as A-U.
    assert can_pair("t", "A")  # Should be treated as U-A.


def test_can_pair_rejects_invalid_inputs_and_ambiguity():
    """
    Ensures that `can_pair` returns False for disallowed pairs and invalid inputs.
    This demonstrates the function's strictness and safety against improper use.
    """
    # Test with non-string inputs.
    assert can_pair(None, "A") is False
    assert can_pair("A", 3) is False
    # Test with inputs that are not single characters.
    assert can_pair("AU", "A") is False
    assert can_pair("A", "UA") is False
    # Test with the ambiguity code 'N'.
    assert can_pair("N", "A") is False
    assert can_pair("A", "N") is False
    # Test with disallowed canonical base combinations.
    assert can_pair("A", "G") is False
    assert can_pair("C", "U") is False


@pytest.mark.parametrize(
    "base_i,base_j,expected",
    [
        (0, 1, 0), # A hairpin closed by (0,1) has 0 unpaired bases inside.
        (0, 2, 1), # A hairpin closed by (0,2) has 1 unpaired base inside.
        (2, 6, 3), # A hairpin closed by (2,6) has 3 unpaired bases inside.
        (3, 7, 3), # A hairpin closed by (3,7) also has 3 unpaired bases.
    ],
)
def test_hairpin_size_formula(base_i, base_j, expected):
    """
    Tests the `hairpin_size` calculation using several examples.
    The size of a hairpin loop is defined as the number of unpaired bases it
    contains, which is calculated by the formula `j - i - 1`.
    """
    assert hairpin_size(base_i, base_j) == expected


def test_is_min_hairpin_size_uses_default_threshold():
    """
    Validates the check for the minimum hairpin size using the default threshold.
    Biophysically, hairpin loops must contain at least 3 unpaired bases to be stable.
    This test confirms the function enforces this default rule.
    """
    # The default minimum number of unpaired bases is 3.
    assert MIN_HAIRPIN_UNPAIRED == 3

    # A pair at (0,4) creates a loop of size 4-0-1 = 3, which is allowed.
    assert is_min_hairpin_size(0, 4) is True
    # A pair at (0,3) creates a loop of size 3-0-1 = 2, which is too small.
    assert is_min_hairpin_size(0, 3) is False


def test_is_min_hairpin_size_with_custom_threshold():
    """
    Validates the minimum hairpin size check when a custom threshold is provided.
    This demonstrates the function's flexibility for use in algorithms that may
    need to explore non-canonical structures.
    """
    # A loop of size 3 (from pair 0,4) is NOT large enough for a min_unpaired of 4.
    assert is_min_hairpin_size(0, 4, min_unpaired=4) is False
    # A loop of size 4 (from pair 0,5) IS large enough for a min_unpaired of 4.
    assert is_min_hairpin_size(0, 5, min_unpaired=4) is True
