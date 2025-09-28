import math
import pytest

from rna_pk_fold.folding.recurrences import (
    SecondaryStructureFoldingEngine,
    RecurrenceConfig,
)
from rna_pk_fold.folding import make_fold_state, BacktrackOp
from rna_pk_fold.energies import SecondaryStructureEnergies


# ---------------------- Fixtures ----------------------

@pytest.fixture
def minimal_energies():
    """
    Provide a minimal SecondaryStructureEnergies object.

    Notes
    -----
    The folding engine forwards this object to adapter callables;
    the tests supply stub functions that don't actually use the
    energy tables, so empty dicts are fine.
    """
    return SecondaryStructureEnergies(
        BULGE={},
        COMPLEMENT_BASES={},
        DANGLES={},
        HAIRPIN={},
        MULTILOOP=(0.0, 0.0, 0.0, 0.0),
        INTERNAL={},
        INTERNAL_MM={},
        NN={},
        TERMINAL_MM={},
        SPECIAL_HAIRPINS=None,
    )


@pytest.fixture
def fake_energy_model_factory(minimal_energies):
    """
    Factory for a configurable fake energy model that conforms to
    SecondaryStructureEnergyModelProtocol. Each test can supply simple
    callables to define hairpin/stack/internal/multiloop behavior.
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

def test_fill_matrix_v_sets_inf_when_cannot_pair(fake_energy_model_factory):
    """
    V[i,j] should remain +∞ and back-pointer NONE when endpoints cannot pair.

    Expected
    --------
    - V[0,3] == +∞
    - back-pointer at (0,3) has operation NONE
    """
    # A and A cannot pair → expect +∞
    seq = "AAAA"
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: 1.0,
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    # Stubs
    hairpin = lambda i, j, s, energies, temp: 1.0
    stack = lambda i, j, k, l, s, energies, temp: float("inf")
    internal = lambda i, j, k, l, s, energies, temp: float("inf")
    multi = lambda branches, unpaired, energies: float("inf")

    folding_eng = SecondaryStructureFoldingEngine(
        energy_model=energy_model,
        config=RecurrenceConfig(enable_multiloop_placeholder=False),
    )

    folding_eng.fill_matrix_v(seq, state)

    assert math.isinf(state.v_matrix.get(0, 3))
    assert state.v_back_ptr.get(0, 3).operation is BacktrackOp.NONE


def test_fill_matrix_v_picks_hairpin_when_finite(fake_energy_model_factory):
    """
    When hairpin energy is finite and other cases are +∞,
    V[i,j] should equal the hairpin value and record HAIRPIN.

    Expected
    --------
    - V[0,4] == 1.23
    - back-pointer at (0,4) has operation HAIRPIN
    """
    # Ends A..U → can pair; pick a length where hairpin_fn is allowed
    seq = "AUAAU"  # i=0=A, j=4=U
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: 1.23,
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    folding_eng = SecondaryStructureFoldingEngine(
        energy_model=energy_model,
        config=RecurrenceConfig(enable_multiloop_placeholder=False),
    )

    folding_eng.fill_matrix_v(seq, state)

    cell_value = state.v_matrix.get(0, 4)
    back_ptr  = state.v_back_ptr.get(0, 4)

    assert math.isclose(cell_value, 1.23, rel_tol=1e-12)
    assert back_ptr.operation is BacktrackOp.HAIRPIN


def test_fill_matrix_v_prefers_internal_over_hairpin_when_better(fake_energy_model_factory):
    """
    If an internal loop path (E_internal + V[k,l]) is lower than the hairpin,
    it should be chosen and recorded as INTERNAL with the inner (k,l).

    Construction
    ------------
    - Use a sequence that allows (i,j) and an inner pair (k,l).
    - Make hairpin_fn expensive for the outer span but cheap for the inner (k,l),
      so V[k,l] is small.
    - Make internal_fn moderate so `internal + V[k,l]` beats the outer hairpin.

    Expected
    --------
    - V[0,5] == 2.0 + V[2,3]  (i.e., 2.5)
    - Back-pointer at (0,5) is INTERNAL with inner=(2,3)
    """
    # Seq positions: 0 1 2 3 4 5
    #                A U G C U U
    # (0,5): A–U pairs; inner (2,3): G–C pairs
    seq = "AUGCUU"
    seq_len = len(seq)
    state = make_fold_state(seq_len)

    # Hairpin: cheap for the inner (2,3) so V[2,3] becomes small; expensive elsewhere
    def hairpin(i, j, s):
        return 0.5 if (i, j) == (2, 3) else 10.0

    energy_model = fake_energy_model_factory(
        hairpin=hairpin,
        stack=lambda i, j, k, l, s: float("inf"), # Stack disabled for clarity
        internal=lambda i, j, k, l, s: 2.0, # Internal gives 2.0 for any valid geometry
        multiloop=lambda branches, unpaired: float("inf"),
    )

    folding_eng = SecondaryStructureFoldingEngine(
        energy_model=energy_model,
        config=RecurrenceConfig(enable_multiloop_placeholder=False),
    )

    folding_eng.fill_matrix_v(seq, state)

    outer_pair = state.v_matrix.get(0, 5)
    inner_pair = state.v_matrix.get(2, 3)  # This is filled earlier due to bottom-up spans
    back_ptr = state.v_back_ptr.get(0, 5)

    assert math.isclose(inner_pair, 0.5, rel_tol=1e-12)
    assert math.isclose(outer_pair, 2.0 + inner_pair, rel_tol=1e-12)
    assert back_ptr.operation is BacktrackOp.INTERNAL
    assert back_ptr.inner == (2, 3)
