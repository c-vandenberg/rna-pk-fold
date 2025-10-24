"""
Unit tests for lookup and thermodynamic utility functions.

This module validates helper functions from `table_lookup_utils` that are
essential for RNA energy calculations, covering energy table lookups with various
fallback mechanisms and a core thermodynamic safety check (`clamp_non_favorable`).
"""
import math

from rna_pk_fold.utils.data.table_lookup_utils import table_lookup


# ----------------------- Table Lookup -----------------------
def test_table_lookup_returns_value_on_hit():
    """
    Tests the primary success case: when the composite key is found in the table,
    its value is returned, ignoring the fallback default.
    """
    tbl = {("A", "U"): -0.3, ("G", "C"): -0.4}
    # Successful lookup for ("A", "U").
    assert table_lookup(tbl, "A", "U", default_value=1.23) == -0.3
    # Successful lookup for ("G", "C").
    assert table_lookup(tbl, "G", "C", default_value=9.99) == -0.4


def test_table_lookup_returns_default_on_miss():
    """
    Tests the core fallback mechanism: if the key is not in the table,
    the specified `default_value` must be returned.
    """
    tbl = {("A", "U"): -0.3}
    # The key ("U", "A") is missing; should return 0.5.
    assert math.isclose(table_lookup(tbl, "U", "A", default_value=0.5), 0.5)
    # The key ("G", "C") is missing; should return 2.0.
    assert math.isclose(table_lookup(tbl, "G", "C", default_value=2.0), 2.0)


def test_table_lookup_none_inputs_use_none_value_over_default():
    """
    Tests the special guard condition for `None` key components.
    If any key component is `None`, the function should return the dedicated
    `none_value` (defaulting to 0.0), regardless of the main `default_value`.
    """
    tbl = {("A", "U"): -0.3}

    # Test default `none_value` (0.0) when one component is None.
    assert math.isclose(table_lookup(tbl, None, "U", default_value=123.0), 0.0)
    assert math.isclose(table_lookup(tbl, "A", None, default_value=123.0), 0.0)

    # Test custom `none_value` is used.
    assert math.isclose(table_lookup(tbl, None, "U", default_value=123.0, none_value=0.2), 0.2)
    assert math.isclose(table_lookup(tbl, "A", None, default_value=123.0, none_value=-1.1), -1.1)


def test_table_lookup_tuple_order_is_significant():
    """
    Verifies that the lookup is order-sensitive, as is standard for tuple keys.
    The reversed key must not be automatically checked.
    """
    tbl = {("A", "U"): -0.3}  # Only ("A", "U") is present.
    # The reversed key ("U", "A") is not a hit; should fall back to default.
    assert math.isclose(table_lookup(tbl, "U", "A", default_value=0.7), 0.7)
    # The correct key ("A", "U") is a hit.
    assert math.isclose(table_lookup(tbl, "A", "U", default_value=0.7), -0.3)
