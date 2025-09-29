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
        INTERNAL_MM={},
        NN={},
        TERMINAL_MM={},
        SPECIAL_HAIRPINS=None,
    )


@pytest.fixture
def fake_energy_model_factory(minimal_energies):
    """
    Factory for a configurable fake energy model that conforms to the
    `SecondaryStructureEnergyModelProtocol` surface used by the engine.
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

    folding_eng = SecondaryStructureFoldingEngine(
        energy_model=energy_model,
        config=RecurrenceConfig(),
    )

    folding_eng.fill_all_matrices(seq, state)

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
        config=RecurrenceConfig(),
    )

    folding_eng.fill_all_matrices(seq, state)

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

    # Hairpin: Cheap for the inner (2,3) so V[2,3] becomes small. Outer hairpin is expensive
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
        config=RecurrenceConfig(),
    )

    folding_eng.fill_all_matrices(seq, state)

    outer_pair = state.v_matrix.get(0, 5)
    inner_pair = state.v_matrix.get(2, 3)
    back_ptr = state.v_back_ptr.get(0, 5)

    assert math.isclose(inner_pair, 0.5, rel_tol=1e-12)
    assert math.isclose(outer_pair, 2.0 + inner_pair, rel_tol=1e-12)
    assert back_ptr.operation is BacktrackOp.INTERNAL
    assert back_ptr.inner == (2, 3)


def test_wm_unpaired_accumulates_c(fake_energy_model_factory):
    """
    WM should accumulate the unpaired cost `c` when no helix attaches.

    Construction
    ------------
    - Use a sequence with no valid pairs so `attach-helix` is impossible.
    - MULTILOOP c = 10 (from minimal_energies).
    - For span (0,3), WM[0,3] should be 3 * c = 30.

    Expected
    --------
    - WM[0,3] == 30.0
    - Back-pointer at WM[0,3] is UNPAIRED_LEFT (tie-breaker preference from implementation order).
    """
    # No valid pairs anywhere (A-A doesn't pair)
    seq = "AAAA"
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = SecondaryStructureFoldingEngine(energy_model=energy_model, config=RecurrenceConfig())
    eng.fill_all_matrices(seq, state)

    wm_val = state.wm_matrix.get(0, 3)  # 3 unpaired bases inside span (0..3)
    wm_bp  = state.wm_back_ptr.get(0, 3)

    assert math.isclose(wm_val, 30.0, rel_tol=1e-12)  # 3 * c = 3 * 10
    assert wm_bp.operation is BacktrackOp.UNPAIRED_LEFT


def test_wm_attach_helix_uses_branch_cost_and_v(fake_energy_model_factory):
    """
    WM should use the 'attach helix' path when it beats unpaired extensions.

    Construction
    ------------
    - MULTILOOP b = 10, c = 10.
    - Make V[0,1] finite (2.0) so WM[0,3] can attach a helix at k=1.
    - WM tail is WM[2,3] which (with no pairs) costs c = 10.
    - Competing unpaired-only path would cost 3*c = 30.
    - Attach path: b + V[0,1] + WM[2,3] = 10 + 2 + 10 = 22.

    Expected
    --------
    - WM[0,3] == 22.0
    - wm_back_ptr at (0,3) is MULTI_ATTACH with inner=(0,1) and split_k=1.
    """
    # Choose 0..1 to be pairable (G–C). Make everything else non-pairing for simplicity.
    seq = "GCUU"   # (0,1) can pair; others won't matter for V except tail WM[2,3]
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        # Force V[0,1] to be finite (2.0). Others will be inf because stack/internal stubs are inf.
        hairpin=lambda i, j, s: 2.0 if (i, j) == (0, 1) else float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = SecondaryStructureFoldingEngine(energy_model=energy_model, config=RecurrenceConfig())
    eng.fill_all_matrices(seq, state)

    wm_val = state.wm_matrix.get(0, 3)
    wm_bp  = state.wm_back_ptr.get(0, 3)

    assert math.isclose(wm_val, 22.0, rel_tol=1e-12)   # 10 (b) + 2 (V[0,1]) + 10 (WM[2,3])
    assert wm_bp.operation is BacktrackOp.MULTI_ATTACH
    assert wm_bp.inner == (0, 1)
    assert wm_bp.split_k == 1


def test_v_closing_multiloop_uses_wm_inside(fake_energy_model_factory):
    """
    V should be able to choose the 'close multiloop' case: a + WM[i+1][j-1].

    Construction
    ------------
    - Make hairpin, stack, internal return +∞ to force the multiloop-closing path.
    - MULTILOOP a = 50, c = 10.
    - For i=0, j=4, WM[1,3] will be 2 * c = 20 (unpaired-only inside).
    - So V[0,4] should be 50 + 20 = 70.

    Expected
    --------
    - V[0,4] == 70.0
    - v_back_ptr at (0,4) is MULTI_ATTACH (close-ml).
    """
    # Make ends pairable (A–U), interior chosen so nothing else can form viable V
    seq = "AUAAU"  # 0:A pairs with 4:U
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = SecondaryStructureFoldingEngine(energy_model=energy_model, config=RecurrenceConfig())
    eng.fill_all_matrices(seq, state)

    v_val = state.v_matrix.get(0, 4)
    v_bp  = state.v_back_ptr.get(0, 4)

    assert math.isclose(v_val, 70.0, rel_tol=1e-12)  # a + WM[1,3] = 50 + (2*c) = 50 + 20
    assert v_bp.operation is BacktrackOp.MULTI_ATTACH


def test_w_base_case_and_unpaired_propagation(fake_energy_model_factory):
    """
    W should propagate the best score by leaving ends unpaired when no pairs exist.

    Construction
    ------------
    - No pair is possible anywhere (all adapters return +∞ for V paths).

    Expected
    --------
    - W[0,3] == 0.0  (for "AAAA")
    - w_back_ptr at (0,3) prefers BIFURCATION given the tie-break rankings.
    """
    seq = "AAAA"  # A–A cannot pair under our can_pair rules
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = SecondaryStructureFoldingEngine(energy_model=energy_model, config=RecurrenceConfig())
    eng.fill_all_matrices(seq, state)

    w_val = state.w_matrix.get(0, 3)
    w_bp  = state.w_back_ptr.get(0, 3)

    assert math.isclose(w_val, 0.0, rel_tol=1e-12)
    assert w_bp.operation is BacktrackOp.BIFURCATION


def test_w_uses_v_when_pair_energy_is_better(fake_energy_model_factory):
    """
    W should choose V[i,j] when it's strictly better than leaving ends unpaired.

    Construction
    ------------
    - Sequence length 2 with a pairable end: "AU".
    - Make V[0,1] (via hairpin) negative, e.g., -1.5.
    - Unpaired options lead to W[1,1] or W[0,0], both 0, so V is strictly better.

    Expected
    --------
    - W[0,1] == -1.5
    - w_back_ptr at (0,1) is PAIR (i.e., came from V).
    """
    seq = "AU"
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: -1.5 if (i, j) == (0, 1) else float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = SecondaryStructureFoldingEngine(energy_model=energy_model, config=RecurrenceConfig())
    eng.fill_all_matrices(seq, state)

    w_val = state.w_matrix.get(0, 1)
    w_bp  = state.w_back_ptr.get(0, 1)

    assert math.isclose(w_val, -1.5, rel_tol=1e-12)
    assert w_bp.operation is BacktrackOp.PAIR


def test_w_bifurcation_beats_unpaired_and_v(fake_energy_model_factory):
    """
    W should choose the bifurcation split when it yields the lowest sum.

    Construction
    ------------
    - Sequence "AUGC":
        * make only (0,1) and (2,3) have finite V via hairpin = -2.0 each.
        * no V[0,3] (outer) and no stacks/internal otherwise.
    - Then:
        W[0,1] = -2.0, W[2,3] = -2.0
        Bifurcation at k=1 → W[0,1] + W[2,3] = -4.0
        Unpaired paths lead to >= 0, so bifurcation wins.

    Expected
    --------
    - W[0,3] == -4.0
    - w_back_ptr at (0,3) is BIFURCATION with split_k == 1.
    """
    seq = "AUGC"   # 0:A–1:U pairable, 2:G–3:C pairable; no outer 0–3 pair in our test
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: -2.0 if (i, j) in {(0, 1), (2, 3)} else float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = SecondaryStructureFoldingEngine(energy_model=energy_model, config=RecurrenceConfig())
    eng.fill_all_matrices(seq, state)

    w_val = state.w_matrix.get(0, 3)
    w_bp  = state.w_back_ptr.get(0, 3)

    assert math.isclose(w_val, -4.0, rel_tol=1e-12)
    assert w_bp.operation is BacktrackOp.BIFURCATION
    assert w_bp.split_k == 1

