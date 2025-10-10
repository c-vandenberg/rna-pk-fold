"""
Unit tests for the IS2 (Symmetric Internal Loop) energy utility functions.

This module validates helper functions like `IS2_outer` and `IS2_outer_yhx`,
which are responsible for retrieving specialized energy terms. The tests focus
on ensuring these functions can correctly interact with a `tables` object that
may provide the energy value either as a static attribute or via a callable method.
"""
import pytest

from rna_pk_fold.utils.is2_utils import IS2_outer, IS2_outer_yhx


# ----------------------- IS2_outer -----------------------

def test_is2_outer_uses_callable_and_returns_value():
    """
    Tests that `IS2_outer` correctly calls the `IS2_outer` method on the
    `tables` object when it is a callable.
    """
    class Tables:
        """A mock tables object with a callable IS2_outer method."""
        def IS2_outer(self, seq, i, j, r, s):
            # This method simulates a dynamic energy calculation.
            return -1.25

    # Instantiate the mock object.
    t = Tables()
    # Call the helper function, which should invoke the method on the tables object.
    got = IS2_outer("ACGU", t, 0, 5, 1, 4)

    # The result should be the value returned by the method.
    assert got == -1.25


def test_is2_outer_uses_numeric_attribute_when_non_callable():
    """
    Tests that `IS2_outer` correctly returns the `IS2_outer` attribute from
    the `tables` object when it is a simple numeric value (non-callable).
    """
    class Tables:
        """A mock tables object with a static, numeric IS2_outer attribute."""
        IS2_outer = 2.5  # This simulates a constant energy bonus/penalty.

    # Instantiate the mock object.
    t = Tables()
    # Call the helper function, which should access the attribute directly.
    got = IS2_outer("ACGU", t, 0, 5, 1, 4)

    # The result should be the value of the attribute.
    assert got == 2.5


def test_is2_outer_defaults_to_zero_when_missing_or_none():
    class TablesWithoutAttr:
        pass

    # tables is None
    assert IS2_outer("ACGU", None, 0, 5, 1, 4) == 0.0
    # attribute missing
    assert IS2_outer("ACGU", TablesWithoutAttr(), 0, 5, 1, 4) == 0.0


# -------------------- IS2_outer_yhx ----------------------

def test_is2_outer_yhx_uses_callable_and_casts_to_float():
    class Tables:
        def IS2_outer_yhx(self, seq, i, j, r, s):
            return "3.75"  # ensure cast to float happens

    class Cfg:
        tables = Tables()

    got = IS2_outer_yhx(Cfg(), "ACGU", 0, 5, 1, 4)
    assert isinstance(got, float)
    assert got == 3.75


def test_is2_outer_yhx_defaults_to_zero_when_no_tables_or_missing_attr():
    class CfgNoTables:
        tables = None

    class TablesWithoutAttr:
        pass

    class CfgWithTablesNoAttr:
        tables = TablesWithoutAttr()

    # tables is None
    assert IS2_outer_yhx(CfgNoTables(), "ACGU", 0, 5, 1, 4) == 0.0
    # IS2_outer_yhx attribute missing
    assert IS2_outer_yhx(CfgWithTablesNoAttr(), "ACGU", 0, 5, 1, 4) == 0.0


def test_is2_outer_yhx_raises_if_attribute_is_non_callable():
    # Unlike IS2_outer, IS2_outer_yhx *calls* the attribute unconditionally,
    # so a non-callable should raise a TypeError.
    class Tables:
        IS2_outer_yhx = 1.23  # non-callable

    class Cfg:
        tables = Tables()

    with pytest.raises(TypeError):
        IS2_outer_yhx(Cfg(), "ACGU", 0, 5, 1, 4)
