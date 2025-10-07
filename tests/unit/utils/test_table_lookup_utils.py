import math

from rna_pk_fold.utils.table_lookup_utils import table_lookup, clamp_non_favorable


# ----------------------- table_lookup -----------------------
def test_table_lookup_returns_value_on_hit():
    tbl = {("A", "U"): -0.3, ("G", "C"): -0.4}
    assert table_lookup(tbl, "A", "U", default=1.23) == -0.3
    assert table_lookup(tbl, "G", "C", default=9.99) == -0.4


def test_table_lookup_returns_default_on_miss():
    tbl = {("A", "U"): -0.3}
    assert math.isclose(table_lookup(tbl, "U", "A", default=0.5), 0.5)
    assert math.isclose(table_lookup(tbl, "G", "C", default=2.0), 2.0)


def test_table_lookup_none_inputs_use_none_value_over_default():
    tbl = {("A", "U"): -0.3}

    # default none_value is 0.0
    assert math.isclose(table_lookup(tbl, None, "U", default=123.0), 0.0)
    assert math.isclose(table_lookup(tbl, "A", None, default=123.0), 0.0)

    # custom none_value
    assert math.isclose(table_lookup(tbl, None, "U", default=123.0, none_value=0.2), 0.2)
    assert math.isclose(table_lookup(tbl, "A", None, default=123.0, none_value=-1.1), -1.1)


def test_table_lookup_tuple_order_is_significant():
    tbl = {("A", "U"): -0.3}  # no ("U","A") entry
    assert math.isclose(table_lookup(tbl, "U", "A", default=0.7), 0.7)
    assert math.isclose(table_lookup(tbl, "A", "U", default=0.7), -0.3)


# -------------------- clamp_non_favorable -------------------
def test_clamp_non_favorable_negative_zero_positive():
    assert math.isclose(clamp_non_favorable(-1.2), -1.2)
    assert math.isclose(clamp_non_favorable(0.0), 0.0)
    assert math.isclose(clamp_non_favorable(1e-9), 0.0)
    assert math.isclose(clamp_non_favorable(3.14), 0.0)


def test_clamp_non_favorable_infinities_and_nan():
    # +inf → clamp to 0; -inf passes through
    assert math.isclose(clamp_non_favorable(math.inf), 0.0)
    assert math.isinf(clamp_non_favorable(-math.inf)) and clamp_non_favorable(-math.inf) < 0

    # NaN compares false to any inequality; function should clamp to 0.0
    nan_val = float("nan")
    out = clamp_non_favorable(nan_val)
    assert math.isclose(out, 0.0)
