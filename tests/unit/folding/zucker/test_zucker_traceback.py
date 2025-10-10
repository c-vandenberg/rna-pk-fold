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
    Provide a minimal SecondaryStructureEnergies object.

    Notes
    -----
    We give MULTILOOP coefficients large-ish values so the multiloop-closing
    option (a + WM[i+1][j-1]) does not accidentally beat the constructed
    hairpin/internal scenarios in these tests.
    """
    return SecondaryStructureEnergies(
        BULGE={},
        COMPLEMENT_BASES={},
        DANGLES={},
        HAIRPIN={},
        MULTILOOP=(50.0, 10.0, 10.0, 0.0),
        INTERNAL={},
        INTERNAL_MISMATCH={},
        TERMINAL_MISMATCH={},
        NN_STACK={},
        SPECIAL_HAIRPINS=None,
    )


@pytest.fixture
def fake_energy_model_factory(minimal_energies):
    """
    Factory for a configurable fake energy model that matches the protocol surface
    used by ZuckerFoldingEngine.
    """
    class FakeEnergyModel:
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
    """Empty sequence → empty result."""
    seq = ""
    fold_state = make_fold_state(0)

    # Energy model won't be called, but we must provide it
    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    folding_eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    folding_eng.fill_all_matrices(seq, fold_state)

    result = traceback_nested(seq, fold_state)
    assert result.pairs == []
    assert result.dot_bracket == ""


def test_traceback_simple_hairpin(fake_energy_model_factory):
    """
    Force a single hairpin at (0,4). Expect one pair and '(...)' dot-bracket.
    """
    seq = "AUAAU"  # i=0, j=4 can pair
    fold_state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: 0.0 if (i, j) == (0, 4) else float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    folding_eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    folding_eng.fill_all_matrices(seq, fold_state)

    result = traceback_nested(seq, fold_state)
    assert result.pairs == [Pair(0, 4)]
    assert result.dot_bracket == "(...)"


def test_traceback_stacked_helix_len2(fake_energy_model_factory):
    """
    Make a 2-stack helix on length-4 sequence: pairs (0,3) and (1,2).

    Construction
    ------------
    - Sequence GCGC ⇒ (0,3) = G–C, (1,2) = C–G are pairable.
    - Hairpin is ∞ everywhere except inner (1,2) where it’s 0 to seed V[1,2].
    - Stack energy for (0,3) on (1,2) is 0, so V[0,3] = 0 + V[1,2] = 0.
    - W then chooses V[0,3].
    """
    seq = "GCGC"
    fold_state = make_fold_state(len(seq))

    def hairpin(i, j, s):
        return 0.0 if (i, j) == (1, 2) else float("inf")

    energy_model = fake_energy_model_factory(
        hairpin=hairpin,
        stack=lambda i, j, k, l, s: 0.0 if (i, j, k, l) == (0, 3, 1, 2) else float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    folding_eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    folding_eng.fill_all_matrices(seq, fold_state)

    result = traceback_nested(seq, fold_state)
    # Order is sorted by (i,j) in the implementation
    assert result.pairs == [Pair(0, 3), Pair(1, 2)]
    assert result.dot_bracket == "(())"


def test_traceback_no_pairs_all_unpaired(fake_energy_model_factory):
    """
    No pairs ever form (all adapters return +∞ for pair-based paths).
    Expect all dots and zero pairs.
    """
    seq = "AAAA"
    fold_state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    folding_eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    folding_eng.fill_all_matrices(seq, fold_state)

    result = traceback_nested(seq, fold_state)
    assert result.pairs == []
    assert result.dot_bracket == "...."
