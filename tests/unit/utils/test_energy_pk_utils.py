import math
import pytest

from rna_pk_fold.utils.energy_pk_utils import (
    coax_pair_key,
    coax_energy_for_join,
)


# ------------------------------
# coax_pair_key
# ------------------------------
def test_coax_pair_key_in_range_and_boundaries():
    # 0..3 valid on a 4-mer; returns 2-char pair from indices (a,b)
    seq = "AUGC"  # indices: 0:A, 1:U, 2:G, 3:C
    assert coax_pair_key(seq, 0, 1) == "AU"
    assert coax_pair_key(seq, 2, 3) == "GC"
    # boundaries inclusive
    assert coax_pair_key(seq, 0, 3) == "AC"
    assert coax_pair_key(seq, 3, 3) == "CC"  # same index allowed, pair_str simply concatenates bases


def test_coax_pair_key_out_of_range_returns_none():
    seq = "AUGC"  # len = 4
    # negative index
    assert coax_pair_key(seq, -1, 1) is None
    # index == len(seq) (out of range)
    assert coax_pair_key(seq, 0, 4) is None
    # both bad
    assert coax_pair_key(seq, -2, 7) is None


# ------------------------------
# coax_energy_for_join
# ------------------------------
def test_coax_energy_for_join_direct_key_hit():
    # left pair (0,1) -> "AU", right pair (2,3) -> "GC"
    seq = "AUGC"
    pairs_tbl = {("AU", "GC"): -0.6}  # stabilizing
    e = coax_energy_for_join(seq, (0, 1), (2, 3), pairs_tbl)
    assert math.isclose(e, -0.6, rel_tol=1e-12)


def test_coax_energy_for_join_reversed_key_hit():
    # Only reversed ordering present in the table
    seq = "AUGC"
    pairs_tbl = {("GC", "AU"): -0.8}
    e = coax_energy_for_join(seq, (0, 1), (2, 3), pairs_tbl)
    assert math.isclose(e, -0.8, rel_tol=1e-12)


def test_coax_energy_for_join_missing_key_defaults_zero():
    # Key not present in either order
    seq = "AUGC"
    pairs_tbl = {("AU", "UA"): -0.3}
    e = coax_energy_for_join(seq, (0, 1), (2, 3), pairs_tbl)  # looking for ("AU","GC") or ("GC","AU")
    assert math.isclose(e, 0.0, rel_tol=1e-12)


def test_coax_energy_for_join_invalid_indices_return_zero():
    seq = "AUGC"
    pairs_tbl = {("AU", "GC"): -1.0}
    # Left out-of-range
    e1 = coax_energy_for_join(seq, (-1, 1), (2, 3), pairs_tbl)
    # Right out-of-range
    e2 = coax_energy_for_join(seq, (0, 1), (2, 4), pairs_tbl)
    # Both out-of-range
    e3 = coax_energy_for_join(seq, (-1, 5), (9, 9), pairs_tbl)
    assert e1 == e2 == e3 == 0.0
