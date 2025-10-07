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
    Provide a minimal SecondaryStructureEnergies object.

    Notes
    -----
    We give MULTILOOP coefficients large-ish values so the multiloop-closing
    option (a + WM[i+1][j-1]) does not accidentally beat the constructed
    hairpin/internal/stack scenarios in these tests.
    """
    return SecondaryStructureEnergies(
        BULGE={},
        COMPLEMENT_BASES={},
        DANGLES={},
        HAIRPIN={},
        MULTILOOP=(50.0, 10.0, 10.0, 0.0),   # a=50, b=10, c=10, d=0
        INTERNAL={},
        NN_STACK={},
        INTERNAL_MISMATCH={},
        TERMINAL_MISMATCH={},
        HAIRPIN_MISMATCH=None,
        MULTI_MISMATCH=None,                # <- keep None so WM end-bonus is off unless a test provides it
        SPECIAL_HAIRPINS=None,
        PSEUDOKNOT=None,
    )


@pytest.fixture
def fake_energy_model_factory(minimal_energies):
    """
    Factory for a configurable fake energy model that conforms to the
    SecondaryStructureEnergyModelProtocol surface used by the engine.
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
    """
    seq = "AAAA"  # A-A can't pair
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: 1.0,
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    assert math.isinf(state.v_matrix.get(0, 3))
    assert state.v_back_ptr.get(0, 3).operation is ZuckerBacktrackOp.NONE


def test_fill_matrix_v_picks_hairpin_when_finite(fake_energy_model_factory):
    """
    When hairpin energy is finite and other cases are +∞,
    V[i,j] should equal the hairpin value and record HAIRPIN.
    """
    seq = "AUAAU"  # i=0=A, j=4=U can pair
    state = make_fold_state(len(seq))

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
    If internal+inner is lower than hairpin, choose INTERNAL with correct (k,l).
    """
    seq = "AUGCUU"  # (0,5)=A-U; inner (2,3)=G-C
    state = make_fold_state(len(seq))

    def hairpin(i, j, s):
        return 0.5 if (i, j) == (2, 3) else 10.0

    energy_model = fake_energy_model_factory(
        hairpin=hairpin,
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: 2.0,  # constant internal cost
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    assert math.isclose(state.v_matrix.get(2, 3), 0.5, rel_tol=1e-12)
    assert math.isclose(state.v_matrix.get(0, 5), 2.0 + 0.5, rel_tol=1e-12)
    bp = state.v_back_ptr.get(0, 5)
    assert bp.operation is ZuckerBacktrackOp.INTERNAL
    assert bp.inner == (2, 3)


def test_fill_matrix_v_prefers_stack_over_internal_and_hairpin_when_best(fake_energy_model_factory):
    """
    If stacking (i,j) on (i+1,j-1) is most favorable, choose STACK.
    """
    seq = "GCGC"   # (0,3)=G-C; inner (1,2)=C-G
    state = make_fold_state(len(seq))

    def hairpin(i, j, s):
        # Make inner (1,2) finite so the stack path has something to add
        return -0.2 if (i, j) == (1, 2) else 10.0

    energy_model = fake_energy_model_factory(
        hairpin=hairpin,
        stack=lambda i, j, k, l, s: -0.5 if (i, j, k, l) == (0, 3, 1, 2) else float("inf"),
        internal=lambda i, j, k, l, s: 10.0,
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    v_inner = state.v_matrix.get(1, 2)          # -0.2
    v_outer = state.v_matrix.get(0, 3)          # -0.5 + (-0.2) = -0.7
    assert math.isclose(v_inner, -0.2, rel_tol=1e-12)
    assert math.isclose(v_outer, -0.7, rel_tol=1e-12)

    bp = state.v_back_ptr.get(0, 3)
    assert bp.operation is ZuckerBacktrackOp.STACK
    assert bp.inner == (1, 2)


def test_v_tiebreak_prefers_stack_over_internal_on_tie(fake_energy_model_factory):
    """
    Tie-break ranking in V: STACK (rank 0) should beat INTERNAL (rank 1) on equal energy.
    """
    seq = "GCGC"
    state = make_fold_state(len(seq))

    def hairpin(i, j, s):
        # Inner (1,2) contributes 0 so both paths tie on the additive part
        return 0.0 if (i, j) == (1, 2) else 10.0

    energy_model = fake_energy_model_factory(
        hairpin=hairpin,
        stack=lambda i, j, k, l, s: 2.0 if (i, j, k, l) == (0, 3, 1, 2) else float("inf"),
        internal=lambda i, j, k, l, s: 2.0 if (i, j, k, l) == (0, 3, 1, 2) else float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    assert math.isclose(state.v_matrix.get(0, 3), 2.0, rel_tol=1e-12)
    assert state.v_back_ptr.get(0, 3).operation is ZuckerBacktrackOp.STACK


def test_wm_unpaired_accumulates_c(fake_energy_model_factory):
    """
    WM should accumulate the unpaired cost `c` when no helix attaches.
    """
    seq = "AAAA"  # no pairs possible
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    wm_val = state.wm_matrix.get(0, 3)  # 3 unpaired inside (0..3) → 3*c = 30
    wm_bp  = state.wm_back_ptr.get(0, 3)
    assert math.isclose(wm_val, 30.0, rel_tol=1e-12)
    assert wm_bp.operation is ZuckerBacktrackOp.UNPAIRED_LEFT  # tie-break among equal paths


def test_wm_attach_helix_uses_branch_cost_and_v(fake_energy_model_factory):
    """
    WM should use the 'attach helix' path when it beats unpaired-only extensions.
    """
    seq = "GCUU"   # (0,1) pairable (G-C)
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: 2.0 if (i, j) == (0, 1) else float("inf"),
        stack=lambda i, j, k, l, s: float("inf"),
        internal=lambda i, j, k, l, s: float("inf"),
        multiloop=lambda branches, unpaired: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    wm_val = state.wm_matrix.get(0, 3)
    wm_bp  = state.wm_back_ptr.get(0, 3)
    # b + V[0,1] + WM[2,3] = 10 + 2 + 10 = 22 (beats 3*c = 30)
    assert math.isclose(wm_val, 22.0, rel_tol=1e-12)
    assert wm_bp.operation is ZuckerBacktrackOp.MULTI_ATTACH
    assert wm_bp.inner == (0, 1)
    assert wm_bp.split_k == 1


def test_wm_attach_helix_adds_multiloop_end_bonus_when_available(fake_energy_model_factory, minimal_energies):
    """
    When MULTI_MISMATCH is provided, WM 'attach helix' should add an end-bonus.
    We set up a case where the 2-sided multiloop mismatch key exists and is
    more stabilizing than single dangles.
    """
    # Span (0,3) with attach helix at (i,k)=(0,2): C–G ; loop-adjacent nts are both 'A'
    # best_multiloop_end_bonus builds key "AC/GA" for (i,k)=(0,2) on seq below.
    seq = "CAGC"   # i=0:'C', k=2:'G', neighbors: seq[1]='A' and seq[1]='A'
    state = make_fold_state(len(seq))

    # Clone minimal energies, add multi-mismatch + dangles
    energies = replace(
        minimal_energies,
        MULTI_MISMATCH={"AC/GA": (-0.5, 0.0)},                  # 2-sided mismatch → ΔG = -0.5
        DANGLES={"A./CG": (-0.1, 0.0), "CG/.A": (-0.1, 0.0)}     # each single dangle = -0.1
    )

    class FakeModel:
        def __init__(self, params, hairpin_fn, stack_fn, internal_fn, multiloop_fn):
            self.params = params
            self._h = hairpin_fn; self._s = stack_fn; self._i = internal_fn; self._m = multiloop_fn
        # NOTE: match the engine’s keyword names:
        def hairpin(self, *, base_i, base_j, seq, temp_k=None):   return self._h(base_i, base_j, seq)
        def stack(self, *, base_i, base_j, base_k, base_l, seq, temp_k=None): return self._s(base_i, base_j, base_k, base_l, seq)
        def internal(self, *, base_i, base_j, base_k, base_l, seq, temp_k=None): return self._i(base_i, base_j, base_k, base_l, seq)
        def multiloop(self, branches, unpaired):                  return self._m(branches, unpaired)

    # Make only V[0,2] finite (2.0) so WM[0,3] can attach there. Others stay +inf.
    energy_model = FakeModel(
        energies,
        hairpin_fn=lambda i, j, s: 2.0 if (i, j) == (0, 2) else float("inf"),
        stack_fn=lambda *args: float("inf"),
        internal_fn=lambda *args: float("inf"),
        multiloop_fn=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # WM = 11.5
    wm_val = state.wm_matrix.get(0, 3)
    wm_bp  = state.wm_back_ptr.get(0, 3)
    assert math.isclose(wm_val, 11.5, rel_tol=1e-12)
    assert wm_bp.operation is ZuckerBacktrackOp.MULTI_ATTACH
    assert wm_bp.inner == (0, 2)
    assert wm_bp.split_k == 2


def test_v_closing_multiloop_uses_wm_inside(fake_energy_model_factory):
    """
    V can choose the 'close multiloop' case: a + WM[i+1][j-1].
    """
    seq = "AUAAU"  # ends pair
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda *args: float("inf"),
        stack=lambda *args: float("inf"),
        internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # WM[1,3] = 2 * c = 20  → V[0,4] = a + 20 = 70
    assert math.isclose(state.v_matrix.get(0, 4), 70.0, rel_tol=1e-12)
    assert state.v_back_ptr.get(0, 4).operation is ZuckerBacktrackOp.MULTI_ATTACH


def test_w_base_case_and_unpaired_propagation(fake_energy_model_factory):
    """
    W should propagate best by leaving ends unpaired when no pairs exist.
    """
    seq = "AAAA"
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda *args: float("inf"),
        stack=lambda *args: float("inf"),
        internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    assert math.isclose(state.w_matrix.get(0, 3), 0.0, rel_tol=1e-12)
    assert state.w_back_ptr.get(0, 3).operation is ZuckerBacktrackOp.BIFURCATION


def test_w_uses_v_when_pair_energy_is_better(fake_energy_model_factory):
    """
    W should choose V[i,j] when it's strictly better than leaving ends unpaired.
    """
    seq = "AU"
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: -1.5 if (i, j) == (0, 1) else float("inf"),
        stack=lambda *args: float("inf"),
        internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    assert math.isclose(state.w_matrix.get(0, 1), -1.5, rel_tol=1e-12)
    assert state.w_back_ptr.get(0, 1).operation is ZuckerBacktrackOp.PAIR


def test_w_tiebreak_prefers_pair_over_unpaired_when_equal(fake_energy_model_factory):
    """
    If V[i,j] ties with leaving ends unpaired (both 0), W should prefer PAIR (rank 0).
    """
    seq = "AU"  # pairable
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: 0.0 if (i, j) == (0, 1) else float("inf"),
        stack=lambda *args: float("inf"),
        internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    # Unpaired option also yields 0 (W[1,1] or W[0,0]), but PAIR wins by rank
    assert math.isclose(state.w_matrix.get(0, 1), 0.0, rel_tol=1e-12)
    assert state.w_back_ptr.get(0, 1).operation is ZuckerBacktrackOp.PAIR


def test_w_bifurcation_beats_unpaired_and_v(fake_energy_model_factory):
    """
    W should choose the bifurcation split when it yields the lowest sum.
    """
    seq = "AUGC"   # (0,1) and (2,3) finite; no outer 0..3 pair
    state = make_fold_state(len(seq))

    energy_model = fake_energy_model_factory(
        hairpin=lambda i, j, s: -2.0 if (i, j) in {(0, 1), (2, 3)} else float("inf"),
        stack=lambda *args: float("inf"),
        internal=lambda *args: float("inf"),
        multiloop=lambda *args: float("inf"),
    )

    eng = ZuckerFoldingEngine(energy_model=energy_model, config=ZuckerFoldingConfig())
    eng.fill_all_matrices(seq, state)

    assert math.isclose(state.w_matrix.get(0, 3), -4.0, rel_tol=1e-12)
    bp = state.w_back_ptr.get(0, 3)
    assert bp.operation is ZuckerBacktrackOp.BIFURCATION
    assert bp.split_k == 1


