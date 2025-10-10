"""
Unit tests for the Zucker secondary structure folding recurrences.

This module contains a suite of tests for the `ZuckerFoldingEngine`, focusing on
the core dynamic programming (DP) logic for the V, WM, and W matrices.

To isolate the recurrence logic from the complexities of a real energy model,
these tests utilize a `FakeEnergyModel`. This mock model is injected with simple,
deterministic energy functions (lambdas) that force the DP algorithm to take
specific paths. This allows each test to verify a single recurrence rule,
choice, or tie-break in isolation.
"""
import math
import pytest
from dataclasses import replace

from rna_pk_fold.folding.zucker.zucker_recurrences import (
    ZuckerFoldingEngine,
    ZuckerFoldingConfig,
)
from rna_pk_fold.folding.zucker import make_fold_state, ZuckerBacktrackOp
from rna_pk_fold.energies.energy_types import SecondaryStructureEnergies


# ---------------------- Fixtures ----------------------
@pytest.fixture
def minimal_energies():
    """
    Provides a minimal `SecondaryStructureEnergies` object with zeroed-out values.

    This fixture is crucial for creating a clean slate for each test, ensuring
    that only the explicitly defined energies in the `FakeEnergyModel` affect
    the outcome. The multiloop parameters (`a`, `b`, `c`) are set to large
    positive values to prevent the "close multiloop" case in the V matrix
    from being unintentionally optimal in tests focused on other rules.
    """
    return SecondaryStructureEnergies(
        BULGE={},
        COMPLEMENT_BASES={},
        DANGLES={},
        HAIRPIN={},
        MULTILOOP=(50.0, 10.0, 10.0, 0.0),  # a=50, b=10, c=10, d=0
        INTERNAL={},
        NN_STACK={},
        INTERNAL_MISMATCH={},
        TERMINAL_MISMATCH={},
        HAIRPIN_MISMATCH=None,
        MULTI_MISMATCH=None,             # Kept as None to disable end-bonuses by default.
        SPECIAL_HAIRPINS=None,
        PSEUDOKNOT=None,
    )


@pytest.fixture
def fake_energy_model_factory(minimal_energies):
    """
    Provides a factory for creating a mock, configurable energy model.

    This factory allows tests to inject simple lambda functions for hairpin,
    stack, internal, and multiloop energy calculations. This is the primary
    mechanism for controlling the behavior of the `ZuckerFoldingEngine` and
    testing its recurrence logic in isolation.
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
def test_fill_matrix_v_sets_inf_when_cannot_pair(fake_energy_model_factory):
    """
    Tests that V[i,j] remains infinite if bases i and j cannot form a pair.
    This is a fundamental constraint of the V matrix, which only stores energies
    for subsequences enclosed by a base pair.
    """
    seq = "AAAA"  # A-A cannot pair.
    state = make_fold_state(len(seq))

    # The energy model doesn't matter here as the `can_pair` check should fail first.
    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: 1.0,
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # V[0,3] should remain at its default value of +infinity.
    assert math.isinf(state.v_matrix.get(0, 3))
    # The backpointer should remain in its default NONE state.
    assert state.v_back_ptr.get(0, 3).operation is ZuckerBacktrackOp.NONE


def test_fill_matrix_v_picks_hairpin_when_finite(fake_energy_model_factory):
    """
    Tests that V[i,j] correctly calculates hairpin energy.
    When forming a hairpin is the only finite-energy option, V[i,j] should
    equal the hairpin energy, and the backpointer should be set to HAIRPIN.
    """
    seq = "AUAAU"  # i=0 (A) and j=4 (U) can pair.
    state = make_fold_state(len(seq))

    # Make only the hairpin calculation return a finite energy.
    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: 1.23,
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    assert math.isclose(state.v_matrix.get(0, 4), 1.23, rel_tol=1e-12)
    assert state.v_back_ptr.get(0, 4).operation is ZuckerBacktrackOp.HAIRPIN


def test_fill_matrix_v_prefers_internal_over_hairpin_when_better(fake_energy_model_factory):
    """
    Tests that V[i,j] chooses an internal loop over a hairpin if it's energetically better.
    """
    seq = "AUGCUU"  # Outer pair (0,5), possible inner pair (2,3).
    state = make_fold_state(len(seq))

    # V[2,3] will first be filled with its hairpin energy (0.5).
    def hairpin(i, j, s):
        return 0.5 if (i, j) == (2, 3) else 10.0

    # The recurrence for V[0,5] will be: internal_energy(2.0) + V[2,3](0.5) = 2.5.
    # This is better than the hairpin energy for V[0,5] (10.0).
    energy_model = fake_energy_model_factory(
        hairpin=hairpin,
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: 2.0,  # Constant internal loop cost.
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # Check that V[2,3] was filled correctly.
    assert math.isclose(state.v_matrix.get(2, 3), 0.5, rel_tol=1e-12)
    # Check that V[0,5] chose the better internal loop path.
    assert math.isclose(state.v_matrix.get(0, 5), 2.0 + 0.5, rel_tol=1e-12)
    # Verify the backpointer for V[0,5].
    bp = state.v_back_ptr.get(0, 5)
    assert bp.operation is ZuckerBacktrackOp.INTERNAL
    assert bp.inner == (2, 3)


def test_fill_matrix_v_prefers_stack_over_internal_and_hairpin_when_best(fake_energy_model_factory):
    """
    Tests that V[i,j] chooses stacking over other options when it is most favorable.
    """
    seq = "GCGC"  # Outer pair (0,3), inner stacked pair (1,2).
    state = make_fold_state(len(seq))

    # V[1,2] will be filled with its hairpin energy (-0.2).
    def hairpin(i, j, s):
        return -0.2 if (i, j) == (1, 2) else 10.0

    # The recurrence for V[0,3] via stacking is: stack_energy(-0.5) + V[1,2](-0.2) = -0.7.
    # This is the best option.
    energy_model = fake_energy_model_factory(
        hairpin=hairpin,
        stack=lambda i, j, k, l, s: -0.5 if (i, j, k, l) == (0, 3, 1, 2) else float("inf"),
        internal=lambda i, j, k, l, s: 10.0,
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # Verify energies and backpointer.
    v_inner = state.v_matrix.get(1, 2)
    v_outer = state.v_matrix.get(0, 3)
    assert math.isclose(v_inner, -0.2, rel_tol=1e-12)
    assert math.isclose(v_outer, -0.7, rel_tol=1e-12)
    bp = state.v_back_ptr.get(0, 3)
    assert bp.operation is ZuckerBacktrackOp.STACK
    assert bp.inner == (1, 2)


def test_v_tiebreak_prefers_stack_over_internal_on_tie(fake_energy_model_factory):
    """
    Tests the tie-breaking rule in V matrix calculations.
    When stacking and forming an internal loop have the exact same energy, the
    algorithm should prefer stacking to ensure deterministic results.
    """
    seq = "GCGC"
    state = make_fold_state(len(seq))

    # V[1,2] will have energy 0.0.
    def hairpin(i, j, s):
        return 0.0 if (i, j) == (1, 2) else 10.0

    # Both stack and internal loop options for V[0,3] will result in an energy of 2.0.
    energy_model = fake_energy_model_factory(
        hairpin=hairpin,
        stack=lambda i, j, k, l, s: 2.0 if (i, j, k, l) == (0, 3, 1, 2) else float("inf"),
        internal=lambda i, j, k, l, s: 2.0 if (i, j, k, l) == (0, 3, 1, 2) else float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # The final energy is 2.0, and the backpointer must be STACK due to the tie-break rule.
    assert math.isclose(state.v_matrix.get(0, 3), 2.0, rel_tol=1e-12)
    assert state.v_back_ptr.get(0, 3).operation is ZuckerBacktrackOp.STACK


def test_wm_unpaired_accumulates_c(fake_energy_model_factory):
    """
    Tests the WM (multiloop) recurrence for unpaired bases.
    For a subsequence where no pairs can form, the WM energy should be the
    cumulative cost `c` for each unpaired nucleotide.
    """
    seq = "AAAA"  # No pairs can form.
    state = make_fold_state(len(seq))

    # Disable all pairing options.
    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # For WM[0,3] (length 4), there are 3 unpaired extensions from WM[0,0].
    # The multiloop unpaired penalty `c` is 10.0 from the fixture.
    # Total energy = 3 * c = 30.0.
    wm_val = state.wm_matrix.get(0, 3)
    wm_bp  = state.wm_back_ptr.get(0, 3)
    assert math.isclose(wm_val, 30.0, rel_tol=1e-12)
    assert wm_bp.operation is ZuckerBacktrackOp.UNPAIRED_LEFT # Tie-break preference.


def test_wm_attach_helix_uses_branch_cost_and_v(fake_energy_model_factory):
    """
    Tests the WM recurrence for attaching a new helix (branch) in a multiloop.
    """
    seq = "GCUU"  # A pair can form at (0,1).
    state = make_fold_state(len(seq))

    # Only V[0,1] has a finite energy (2.0).
    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: 2.0 if (i, j) == (0, 1) else float("inf"),
        stack=lambda *args: float("inf"),
        internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    wm_val = state.wm_matrix.get(0, 3)
    wm_bp  = state.wm_back_ptr.get(0, 3)

    # Compare two paths for WM[0,3]:
    # 1. All unpaired: 3 * c = 30.0
    # 2. Attach helix (0,1): b + V[0,1] + WM[2,3] = 10.0 + 2.0 + (2*c=20.0) -> This is wrong.
    #    Correctly: b + V[0,1] + WM[2,3] = 10 + 2 + WM[2,2]+c = 10 + 2 + 0 + 10 = 22.0
    # Path 2 is better.
    assert math.isclose(wm_val, 22.0, rel_tol=1e-12)
    assert wm_bp.operation is ZuckerBacktrackOp.MULTI_ATTACH
    assert wm_bp.inner == (0, 1)
    assert wm_bp.split_k == 1


def test_wm_attach_helix_adds_multiloop_end_bonus_when_available(fake_energy_model_factory, minimal_energies):
    """
    Tests that the WM 'attach helix' recurrence correctly includes a terminal
    mismatch bonus for the closing pair of the multiloop if available.
    """
    seq = "CAGC"
    state = make_fold_state(len(seq))

    # Add a stabilizing multiloop mismatch energy to the parameters.
    energies = replace(
        minimal_energies,
        MULTI_MISMATCH={"AC/GA": (-0.5, 0.0)}, # ΔG = -0.5
        DANGLES={"A./CG": (-0.1, 0.0), "CG/.A": (-0.1, 0.0)}
    )

    # A more complex fake model to handle keyword arguments correctly.
    class FakeModel:
        def __init__(self, params, hairpin_fn, stack_fn, internal_fn, multiloop_fn):
            self.params = params; self._h = hairpin_fn; self._s = stack_fn; self._i = internal_fn; self._m = multiloop_fn
        def hairpin(self, *, base_i, base_j, seq, temp_k=None):   return self._h(base_i, base_j, seq)
        def stack(self, *, base_i, base_j, base_k, base_l, seq, temp_k=None): return self._s(base_i, base_j, base_k, base_l, seq)
        def internal(self, *, base_i, base_j, base_k, base_l, seq, temp_k=None): return self._i(base_i, base_j, base_k, base_l, seq)
        def multiloop(self, branches, unpaired):                   return self._m(branches, unpaired)

    # Make only V[0,2] finite.
    energy_model = FakeModel(
        energies,
        hairpin_fn=lambda i, j, s: 2.0 if (i, j) == (0, 2) else float("inf"),
        stack_fn=lambda *args: float("inf"), internal_fn=lambda *args: float("inf"), multiloop_fn=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    wm_val = state.wm_matrix.get(0, 3)
    wm_bp  = state.wm_back_ptr.get(0, 3)
    # Calculation for WM[0,3] attaching helix (0,2):
    # b + V[0,2] + WM[3,3] + end_bonus = 10.0 + 2.0 + 0.0 + (-0.5) = 11.5
    assert math.isclose(wm_val, 11.5, rel_tol=1e-12)
    assert wm_bp.operation is ZuckerBacktrackOp.MULTI_ATTACH
    assert wm_bp.inner == (0, 2)
    assert wm_bp.split_k == 2


def test_v_closing_multiloop_uses_wm_inside(fake_energy_model_factory):
    """
    Tests the V matrix recurrence for closing a multiloop.
    The energy should be calculated as `a + WM[i+1, j-1]`, where `a` is the
    multiloop initiation penalty.
    """
    seq = "AUAAU"
    state = make_fold_state(len(seq))

    # Disable all V options except closing a multiloop.
    energy_model = fake_energy_model_factory(
        hairpin=lambda *args: float("inf"), stack=lambda *args: float("inf"),
        internal=lambda *args: float("inf"), multiloop=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # For V[0,4], the inner segment is [1,3].
    # WM[1,3] will be filled based on 2 unpaired extensions: 2 * c = 2 * 10.0 = 20.0.
    # V[0,4] = a + WM[1,3] = 50.0 + 20.0 = 70.0.
    assert math.isclose(state.v_matrix.get(0, 4), 70.0, rel_tol=1e-12)
    assert state.v_back_ptr.get(0, 4).operation is ZuckerBacktrackOp.MULTI_ATTACH


def test_w_base_case_and_unpaired_propagation(fake_energy_model_factory):
    """
    Tests the W matrix for a sequence with no possible pairs.
    The optimal structure should be the unfolded state, with an energy of 0.0.
    """
    seq = "AAAA"
    state = make_fold_state(len(seq))

    # Disable all pairing.
    energy_model = fake_energy_model_factory(
        hairpin=lambda *args: float("inf"), stack=lambda *args: float("inf"),
        internal=lambda *args: float("inf"), multiloop=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # The best energy for any structure on "AAAA" is 0.0 (no structure).
    assert math.isclose(state.w_matrix.get(0, 3), 0.0, rel_tol=1e-12)
    assert state.w_back_ptr.get(0, 3).operation is ZuckerBacktrackOp.BIFURCATION # Tie-break


def test_w_uses_v_when_pair_energy_is_better(fake_energy_model_factory):
    """
    Tests that the W matrix chooses to form a pair (using V) when it is favorable.
    """
    seq = "AU"
    state = make_fold_state(len(seq))

    # Make the hairpin for (0,1) very stable (-1.5).
    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: -1.5 if (i, j) == (0, 1) else float("inf"),
        stack=lambda *args: float("inf"), internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # W[0,1] should take the value from V[0,1], as -1.5 is better than 0.0 (unpaired).
    assert math.isclose(state.w_matrix.get(0, 1), -1.5, rel_tol=1e-12)
    assert state.w_back_ptr.get(0, 1).operation is ZuckerBacktrackOp.PAIR


def test_w_tiebreak_prefers_pair_over_unpaired_when_equal(fake_energy_model_factory):
    """
    Tests the tie-breaking rule in the W matrix.
    If forming a pair (V[i,j]) has the same energy as not pairing (e.g., 0.0),
    the algorithm should prefer to form the pair.
    """
    seq = "AU"
    state = make_fold_state(len(seq))

    # The hairpin energy is 0.0, tying with the unpaired/bifurcation options.
    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: 0.0 if (i, j) == (0, 1) else float("inf"),
        stack=lambda *args: float("inf"), internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # The energy is 0.0, but the backpointer must be PAIR due to the tie-break rule.
    assert math.isclose(state.w_matrix.get(0, 1), 0.0, rel_tol=1e-12)
    assert state.w_back_ptr.get(0, 1).operation is ZuckerBacktrackOp.PAIR


def test_w_bifurcation_beats_unpaired_and_v(fake_energy_model_factory):
    """
    Tests that the W matrix correctly chooses the bifurcation option when it is best.
    """
    seq = "AUGC"
    state = make_fold_state(len(seq))

    # The sequence can be split into two stable hairpins: AU (-2.0) and GC (-2.0).
    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: -2.0 if (i, j) in {(0, 1), (2, 3)} else float("inf"),
        stack=lambda *args: float("inf"), internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # The best score for W[0,3] is the sum of W[0,1] and W[2,3], which is -2.0 + -2.0 = -4.0.
    assert math.isclose(state.w_matrix.get(0, 3), -4.0, rel_tol=1e-12)
    # The backpointer should record the bifurcation and the split point k.
    bp = state.w_back_ptr.get(0, 3)
    assert bp.operation is ZuckerBacktrackOp.BIFURCATION
    assert bp.split_k == 1


