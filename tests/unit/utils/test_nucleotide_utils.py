from rna_pk_fold.utils.nucleotide_utils import normalize_base


def test_normalize_base_uppercases_and_maps_t_to_u():
    """
    Ensure bases are uppercased and T is mapped to U.

    Expected
    --------
    - Lowercase letters are uppercased; 'T' or 't' becomes 'U'.
    """
    assert normalize_base("a") == "A"
    assert normalize_base("c") == "C"
    assert normalize_base("t") == "U"
    assert normalize_base("T") == "U"


def test_normalize_base_rejects_non_single_char_inputs_by_returning_original():
    """
    Non-string or multi-character inputs are returned unchanged
    (the caller can decide how to handle them).

    Expected
    --------
    - Non-strings and strings with length != 1 are returned as-is (passthrough).
    """
    assert normalize_base(5) == 5            # Non-string passthrough
    assert normalize_base(None) is None      # Non-string passthrough
    assert normalize_base("AU") == "AU"      # Multi-char passthrough
    assert normalize_base("") == ""          # Empty string passthrough
