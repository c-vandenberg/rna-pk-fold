from rna_pk_fold.folding.common_traceback import pairs_to_multilayer_dotbracket
from rna_pk_fold.structures import Pair


def test_non_crossing_upper_layer_renders_parentheses():
    # Outer pair (0,5) on layer 0, inner pair (1,4) assigned to layer 1 but doesn't cross
    seq_len = 6
    pairs = [Pair(0, 5), Pair(1, 4)]
    pair_layer = {(0, 5): 0, (1, 4): 1}

    db = pairs_to_multilayer_dotbracket(seq_len, pairs, pair_layer)
    # Both should be rendered as parentheses because layer-1 pair does not cross lower layers
    assert db == '((..))'


def test_crossing_upper_layer_renders_nonparenthesis():
    # Two pairs that cross: (0,3) on layer 0 and (1,4) assigned to layer 1 -> should render [] for second
    seq_len = 5
    pairs = [Pair(0, 3), Pair(1, 4)]
    pair_layer = {(0, 3): 0, (1, 4): 1}

    db = pairs_to_multilayer_dotbracket(seq_len, pairs, pair_layer)
    # First pair on layer 0 uses (), second should use [] because it crosses the lower layer pair
    assert db == '([)].'

