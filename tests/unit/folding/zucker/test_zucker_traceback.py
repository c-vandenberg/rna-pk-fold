"""
Unit tests for the Zucker secondary structure traceback algorithm.

This module validates the `traceback_nested` function. The testing strategy
involves two main steps for each case:
1.  **Folding**: The `ZuckerFoldingEngine` is run with a controlled, `FakeEnergyModel`.
    This model is configured with simple energy functions (lambdas) to force the
    dynamic programming algorithm to populate the backpointer matrices in a
    specific, predictable way.
2.  **Traceback**: The `traceback_nested` function is then called on the resulting
    fold state. The tests assert that the traceback correctly follows the
    pre-determined backpointers to reconstruct the intended structure (list of
    base pairs and dot-bracket string).
"""
import pytest

from rna_pk_fold.folding.zucker.zucker_fold_state import make_fold_state
from rna_pk_fold.folding.zucker.zucker_recurrences import ZuckerFoldingEngine, ZuckerFoldingConfig
from rna_pk_fold.folding.zucker.zucker_traceback import traceback_nested
from rna_pk_fold.structures import Pair
from rna_pk_fold.energies import SecondaryStructureEnergies


# ---------------------- Fixtures ----------------------

@pytest.fixture
def minimal_energies():
    """
    Provides a minimal `SecondaryStructureEnergies` object with zeroed-out values.

    This fixture creates a clean slate for tests, ensuring that only the energies
    explicitly defined in the `FakeEnergyModel` influence the folding outcome. The
    multiloop parameters are set to large positive values to prevent the "close
    multiloop" recurrence from being unintentionally optimal in tests focused on
    other rules like hairpins or stacks.
    """
    return SecondaryStructureEnergies(
        BULGE={},
        COMPLEMENT_BASES={},
        DANGLES={},
        HAIRPIN={},
        MULTILOOP=(50.0, 10.0, 10.0, 0.0), # High penalties to avoid this path.
        INTERNAL={},
        INTERNAL_MISMATCH={},
        TERMINAL_MISMATCH={},
        NN_STACK={},
        SPECIAL_HAIRPINS=None,
    )


@pytest.fixture
def fake_energy_model_factory(minimal_energies):
    """
    Provides a factory for creating a mock, configurable energy model.

    This factory is key to the testing strategy. It allows each test to inject
    simple lambda functions for energy calculations, precisely controlling the
    energy landscape to guide the folding engine down a single, desired logical
    path. This isolates the test to a specific recurrence and its corresponding
    traceback step.
    """
    class FakeEnergyModel:
        """A mock class that conforms to the energy model protocol."""
        def __init__(self, params, hairpin_fn, stack_fn, internal_fn, multiloop_fn, temp_k=310.15):
            self.params = params
            self.temp_k = temp_k
            self._hairpin = hairpin_fn
            self._stack = stack_fn
            self._internal = internal_fn
            self._multiloop = multiloop_fn

        def hairpin(self, base_i, base_j, seq, *, temp_k=None):
            return self._hairpin(base_i, base_j, seq)

        def stack(self, base_i, base_j, base_k, base_l, seq, *, temp_k=None):
            return self._stack(base_i, base_j, base_k, base_l, seq)

        def internal(self, base_i, base_j, base_k, base_l, seq, *, temp_k=None):
            return self._internal(base_i, base_j, base_k, base_l, seq)

        def multiloop(self, branches, unpaired_bases):
            return self._multiloop(branches, unpaired_bases)

    def make(*, hairpin, stack, internal, multiloop, temp_k=310.15):
        return FakeEnergyModel(
            params=minimal_energies,
            hairpin_fn=hairpin,
            stack_fn=stack,
            internal_fn=internal,
            multiloop_fn=multiloop,
            temp_k=temp_k,
        )

    return make


# ----------------------------- Tests ---------------------------------

def test_traceback_empty_sequence(fake_energy_model_factory):
    """
    Tests the base case of tracing an empty sequence.
    The result should be an empty structure (no pairs, empty dot-bracket string).
    """
    seq = ""
    fold_state = make_fold_state(0)

    # The energy model is not called for n=0, but the engine requires it.
    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    # The folding engine will do nothing for an empty sequence.
    folding_eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    folding_eng.fill_all_matrices(seq, fold_state)

    # The traceback should correctly handle the n=0 case.
    result = traceback_nested(seq, fold_state)
    assert result.pairs == []
    assert result.dot_bracket == ""


def test_traceback_simple_hairpin(fake_energy_model_factory):
    """
    Tests the traceback of a single, simple hairpin loop.
    The energy model is configured to make a hairpin at (0,4) the only
    favorable structure.
    """
    seq = "AUAAU"  # The ends (0, 4) can pair.
    fold_state = make_fold_state(len(seq))

    # Configure the energy model to make only the hairpin (0,4) favorable (energy 0.0).
    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: 0.0 if (i, j) == (0, 4) else float("inf"),
        stack=lambda *args: float("inf"),
        internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    # Run the folding engine to populate the backpointer matrices.
    folding_eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    folding_eng.fill_all_matrices(seq, fold_state)

    # Run the traceback.
    result = traceback_nested(seq, fold_state)

    # Verify that the correct pair and dot-bracket string are produced.
    assert result.pairs == [Pair(0, 4)]
    assert result.dot_bracket == "(...)"


def test_traceback_stacked_helix_len2(fake_energy_model_factory):
    """
    Tests the traceback of a helix composed of two stacked pairs.
    The test guides the DP algorithm to first form an inner hairpin (1,2)
    and then stack the outer pair (0,3) on top, creating a ((...)) structure.
    """
    seq = "GCGC"
    fold_state = make_fold_state(len(seq))

    # Define a hairpin function that seeds the inner pair (1,2) with energy 0.0.
    def hairpin(i, j, s):
        return 0.0 if (i, j) == (1, 2) else float("inf")

    # Define a stack function that makes stacking (0,3) on (1,2) favorable.
    energy_model = fake_energy_model_factory(
        hairpin=hairpin,
        stack=lambda i, j, k, l, s: 0.0 if (i, j, k, l) == (0, 3, 1, 2) else float("inf"),
        internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    folding_eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    folding_eng.fill_all_matrices(seq, fold_state)

    result = traceback_nested(seq, fold_state)

    # The result should contain both pairs, sorted by the 5' index.
    assert result.pairs == [Pair(0, 3), Pair(1, 2)]
    assert result.dot_bracket == "(())"


def test_traceback_no_pairs_all_unpaired(fake_energy_model_factory):
    """
    Tests the traceback for a sequence with no favorable pairs.
    The energy model returns +infinity for all pairing options, so the optimal
    structure should be the completely unfolded state.
    """
    seq = "AAAA"
    fold_state = make_fold_state(len(seq))

    # Configure the energy model to make all pairing structures infinitely costly.
    energy_model = fake_energy_model_factory(
        hairpin=lambda *args: float("inf"),
        stack=lambda *args: float("inf"),
        internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    folding_eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    folding_eng.fill_all_matrices(seq, fold_state)

    # The traceback should follow the unpaired/bifurcation paths.
    result = traceback_nested(seq, fold_state)

    # The result should be no pairs and a dot-bracket of all unpaired bases.
    assert result.pairs == []
    assert result.dot_bracket == "...."
