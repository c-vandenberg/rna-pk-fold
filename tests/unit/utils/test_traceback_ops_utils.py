from rna_pk_fold.structures import Pair
from rna_pk_fold.utils.indices_utils import canonical_pair

from rna_pk_fold.utils.traceback_ops_utils import (
    add_pair_once,
    merge_nested_interval,
)


def test_add_pair_once_adds_new_and_records_layer():
    pairs = set()
    pair_layer = {}

    # Intentionally give reversed order to exercise canonicalization
    add_pair_once(pairs, pair_layer, i=5, j=3, layer=2)

    i2, j2 = canonical_pair(5, 3)
    assert Pair(i2, j2) in pairs
    assert pair_layer[(i2, j2)] == 2
    assert len(pairs) == 1
    assert len(pair_layer) == 1


def test_add_pair_once_does_not_overwrite_existing_layer_on_duplicate():
    pairs = set()
    pair_layer = {}

    # First insertion
    add_pair_once(pairs, pair_layer, i=1, j=4, layer=1)

    # Duplicate insertion with different layer should be ignored
    add_pair_once(pairs, pair_layer, i=4, j=1, layer=9)

    assert Pair(1, 4) in pairs
    assert pair_layer[(1, 4)] == 1        # original layer preserved
    assert len(pairs) == 1                 # still a single entry


# ---------------- merge_nested_interval ----------------
class DummyTraceResult:
    def __init__(self, pairs):
        self.pairs = pairs  # list[Pair]


def test_merge_nested_interval_collects_unique_pairs_and_applies_layer(monkeypatch):
    # Collect fn returns duplicates to ensure add_pair_once de-duplicates
    def collect_fn(seq, nested_state, i, j):
        return DummyTraceResult([Pair(1, 4), Pair(2, 3), Pair(1, 4)])

    pairs = set()
    pair_layer = {}

    merge_nested_interval(
        seq="AUGCGA",
        nested_state=None,
        i=0,
        j=5,
        layer=3,
        collect_fn=collect_fn,
        pairs=pairs,
        pair_layer=pair_layer,
    )

    # Unique pairs captured
    assert Pair(1, 4) in pairs
    assert Pair(2, 3) in pairs
    assert len(pairs) == 2

    # Layer applied to all newly added pairs
    assert pair_layer[(1, 4)] == 3
    assert pair_layer[(2, 3)] == 3


def test_merge_nested_interval_does_not_change_existing_layer_for_prepopulated_pairs():
    # Pre-populate with a pair on a specific layer
    pairs = {Pair(5, 6)}
    pair_layer = {(5, 6): 1}

    def collect_fn(seq, nested_state, i, j):
        # The nested result tries to re-add (5,6)
        return DummyTraceResult([Pair(5, 6)])

    merge_nested_interval(
        seq="CCCCCCCC",
        nested_state=None,
        i=0,
        j=7,
        layer=9,  # different layer requested
        collect_fn=collect_fn,
        pairs=pairs,
        pair_layer=pair_layer,
    )

    # Pair still present, and its original layer preserved
    assert Pair(5, 6) in pairs
    assert pair_layer[(5, 6)] == 1
    assert len(pairs) == 1
