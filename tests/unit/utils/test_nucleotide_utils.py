"""
Unit tests for the `normalize_base` utility function.

This module validates the function responsible for standardizing nucleotide
characters by ensuring they are in uppercase and that the DNA base 'T'
(Thymine) is correctly mapped to the RNA base 'U' (Uracil).
"""
from rna_pk_fold.utils.nucleotide_utils import normalize_base


def test_normalize_base_uppercases_and_maps_t_to_u():
    """
    Verifies the two core transformations of the function: uppercasing and T-to-U mapping.
    """
    # Test standard uppercasing (A and C).
    assert normalize_base("a") == "A"
    assert normalize_base("c") == "C"
    # Test T-to-U mapping for lowercase 't'.
    assert normalize_base("t") == "U"
    # Test T-to-U mapping for uppercase 'T'.
    assert normalize_base("T") == "U"


def test_normalize_base_rejects_non_single_char_inputs_by_returning_original():
    """
    Tests the function's safety behavior: inputs that are not single-character
    strings are considered outside the scope of base normalization and are
    returned as-is (passthrough).
    """
    # Test passthrough for non-string, numeric input.
    assert normalize_base(5) == 5
    # Test passthrough for None.
    assert normalize_base(None) is None
    # Test passthrough for multi-character strings.
    assert normalize_base("AU") == "AU"
    # Test passthrough for empty strings.
    assert normalize_base("") == ""
