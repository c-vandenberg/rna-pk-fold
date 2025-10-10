"""
Unit tests for traceback utility functions.

This module validates the core logic used during structure reconstruction,
particularly for pseudoknots. The tested functions ensure that base pairs are
collected uniquely (`add_pair_once`) and correctly tagged with their
pseudoknot layer, and that results from secondary structure tracebacks are
integrated without corrupting existing layer information (`merge_nested_interval`).
"""
from rna_pk_fold.structures import Pair
from rna_pk_fold.utils.indices_utils import canonical_pair

from rna_pk_fold.utils.traceback_ops_utils import (
    add_pair_once,
    merge_nested_interval,
)


def test_add_pair_once_adds_new_and_records_layer():
    """
    Tests the core functionality: adding a unique pair and recording its layer.

    It verifies that the pair is canonicalized (i < j) before being stored
    in both the pair set and the layer map.
    """
    pairs = set()
    pair_layer = {}

    # Intentionally give reversed order to exercise canonicalization: (5, 3) -> (3, 5).
    add_pair_once(pairs, pair_layer, i=5, j=3, layer=2)

    i2, j2 = canonical_pair(5, 3)
    # Assert the canonicalized pair is in the set.
    assert Pair(i2, j2) in pairs
    # Assert the layer map records the layer for the canonical key.
    assert pair_layer[(i2, j2)] == 2
    assert len(pairs) == 1
    assert len(pair_layer) == 1


def test_add_pair_once_does_not_overwrite_existing_layer_on_duplicate():
    """
    Tests the idempotency and layer preservation logic.
    When a duplicate pair is inserted, the original layer recorded during the
    first insertion must be preserved.
    """
    pairs = set()
    pair_layer = {}

    # 1. First insertion establishes the pair and its layer (1).
    add_pair_once(pairs, pair_layer, i=1, j=4, layer=1)

    # 2. Duplicate insertion with different coordinates (reversed) and a new layer (9).
    add_pair_once(pairs, pair_layer, i=4, j=1, layer=9)

    assert Pair(1, 4) in pairs
    assert pair_layer[(1, 4)] == 1      # Original layer (1) preserved.
    assert len(pairs) == 1              # No duplicate pair added to the set.
    assert len(pair_layer) == 1


# ---------------- merge_nested_interval ----------------
class DummyTraceResult:
    """
    Mock object simulating the result returned by a nested (secondary structure)
    traceback function.
    """
    def __init__(self, pairs):
        self.pairs = pairs  # list[Pair]


def test_merge_nested_interval_collects_unique_pairs_and_applies_layer(monkeypatch):
    """
    Tests that `merge_nested_interval` correctly integrates results from a
    secondary structure traceback. It must de-duplicate pairs and apply the
    current pseudoknot layer to all newly added pairs.
    """
    # Mock collection function that returns a list containing duplicates.
    def collect_fn(seq, nested_state, i, j):
        # Result contains two unique pairs, one of which is duplicated.
        return DummyTraceResult([Pair(1, 4), Pair(2, 3), Pair(1, 4)])

    pairs = set()
    pair_layer = {}

    # Merge the results, applying layer 3.
    merge_nested_interval(
        seq="AUGCGA",
        nested_state=None,
        i=0,
        j=5,
        layer=3,
        collect_pairs_fn=collect_fn,
        pairs=pairs,
        pair_layer_map=pair_layer,
    )

    # Verify that only the unique pairs were captured.
    assert Pair(1, 4) in pairs
    assert Pair(2, 3) in pairs
    assert len(pairs) == 2

    # Verify that all captured pairs were assigned the new layer (3).
    assert pair_layer[(1, 4)] == 3
    assert pair_layer[(2, 3)] == 3


def test_merge_nested_interval_does_not_change_existing_layer_for_prepopulated_pairs():
    """
    Tests the layer protection mechanism. If a pair already exists in the global
    map, the merge operation must not overwrite its original layer, even if the
    nested result tries to re-add it with a different layer.
    """
    # Pre-populate the structure with a pair and its original layer (1).
    pairs = {Pair(5, 6)}
    pair_layer = {(5, 6): 1}

    # The nested result function attempts to re-add the existing pair.
    def collect_fn(seq, nested_state, i, j):
        return DummyTraceResult([Pair(5, 6)])

    # Merge with a new layer (9) which should be ignored for the existing pair.
    merge_nested_interval(
        seq="CCCCCCCC",
        nested_state=None,
        i=0,
        j=7,
        layer=9,
        collect_pairs_fn=collect_fn,
        pairs=pairs,
        pair_layer_map=pair_layer,
    )

    # Assert that the pair is still present, and the original layer (1) is preserved.
    assert Pair(5, 6) in pairs
    assert pair_layer[(5, 6)] == 1
    assert len(pairs) == 1
