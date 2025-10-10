"""
Unit tests for the `EddyRivasFoldingEngine`, the core of the pseudoknot folding algorithm.

This module tests several key components of the folding engine:
1.  The `take_best` helper function for choosing optimal solutions.
2.  The `_seed_from_nested` method for initializing DP matrices from a
    secondary structure-only fold.
3.  The `_publish_wx` and `_publish_vx` methods that combine charged and
    uncharged substates.
4.  A smoke test to ensure the main `fill_with_costs` method orchestrates
    its subroutines in the correct sequence.
"""
import math

from rna_pk_fold.folding.zucker import make_fold_state
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import init_eddy_rivas_fold_state
from rna_pk_fold.energies.energy_types import PseudoknotEnergies

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences import (EddyRivasFoldingEngine, EddyRivasFoldingConfig,
                                                                  take_best)
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import (
    EddyRivasBackPointer,
    EddyRivasBacktrackOp,
)


# -------------------- take_best / make_bp --------------------

def test_take_best_replaces_when_better_and_calls_factory_once():
    """
    Tests that `take_best` updates the energy and backpointer if a better score is found.

    It also verifies a key optimization: the backpointer factory function (`mk`)
    should only be called if the new score is actually better, avoiding
    unnecessary object creation.
    """
    # Use a dictionary to track the number of calls to the factory.
    factory_calls = {"n": 0}
    def mk():
        factory_calls["n"] += 1
        return EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE)

    # The new energy (5.0) is better than the old one (10.0).
    best, bp = take_best(10.0, None, 5.0, mk)

    # Assert that the state was updated and the factory was called exactly once.
    assert best == 5.0
    assert isinstance(bp, EddyRivasBackPointer)
    assert factory_calls["n"] == 1


def test_take_best_keeps_old_on_tie_or_worse():
    """
    Tests that `take_best` does not update if the new score is worse or the same.
    """
    old_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_SS_LEFT)

    # Case 1: The new score (4.0) is worse than the old one (3.0).
    best, bp = take_best(3.0, old_bp, 4.0, lambda: None)
    # The original best score and backpointer object should be kept.
    assert best == 3.0 and bp is old_bp

    # Case 2: The new score is a tie.
    best2, bp2 = take_best(3.0, old_bp, 3.0, lambda: None)
    # The original state should also be kept in case of a tie.
    assert best2 == 3.0 and bp2 is old_bp


# -------------------- _seed_from_nested (static) --------------------

def test_seed_from_nested_copies_nested_into_uncharged_and_wx_vx():
    """
    Verifies the seeding process from a secondary-structure-only fold.

    The `_seed_from_nested` method initializes the Rivas-Eddy (RE) DP matrices
    by copying the results from a pre-computed Zucker-style (nested) fold.
    This test ensures the values are copied to the correct destination matrices.
    """
    n = 3
    # 1. Create and populate a mock "nested" fold state.
    nested = make_fold_state(n)
    nested.w_matrix.set(0, 2, 7.0)
    nested.v_matrix.set(0, 2, 9.5)

    # 2. Create a fresh Rivas-Eddy (RE) state.
    re_state = init_eddy_rivas_fold_state(n)
    # Sanity check: ensure initial values are as expected before seeding.
    assert math.isinf(re_state.wxc_matrix.get(0, 2))
    assert re_state.wxc_matrix.get(1, 1) == 0.0

    # 3. Run the seeding process.
    EddyRivasFoldingEngine._seed_from_nested(nested, re_state)

    # 4. Verify the results.
    # The "uncharged" matrices should mirror the nested fold.
    assert re_state.wxu_matrix.get(0, 2) == 7.0
    assert re_state.vxu_matrix.get(0, 2) == 9.5
    # The "charged" matrices should remain at their default (+inf for off-diagonal).
    assert math.isinf(re_state.wxc_matrix.get(0, 2))
    assert math.isinf(re_state.vxc_matrix.get(0, 2))
    # The main WX/VX matrices should be populated with the best score (from uncharged).
    assert re_state.wx_matrix.get(0, 2) == 7.0
    assert re_state.vx_matrix.get(0, 2) == 9.5
    # The WXI matrix should also mirror the initial W matrix.
    assert re_state.wxi_matrix.get(0, 2) == 7.0


# -------------------- publish WX/VX selection --------------------

def test_publish_wx_prefers_unscaled_uncharged_and_sets_backpointer():
    """
    Tests that `_publish_wx` selects the best score between charged/uncharged states.
    """
    n = 2
    re_state = init_eddy_rivas_fold_state(n)
    # Use zero-cost energies to isolate the logic being tested.
    cfg = EddyRivasFoldingConfig(
        costs=PseudoknotEnergies(
            q_ss=0.0, p_tilde_out=0.0, p_tilde_hole=0.0, q_tilde_out=0.0, q_tilde_hole=0.0,
            l_tilde=0.0, r_tilde=0.0, m_tilde_yhx=0.0, m_tilde_vhx=0.0, m_tilde_whx=0.0
        )
    )
    eng = EddyRivasFoldingEngine(cfg)

    # Set up the test case: make the "uncharged" score better than "charged".
    re_state.wxu_matrix.set(0, 1, 3.0)
    re_state.wxc_matrix.set(0, 1, 5.0)

    # Run the publish step.
    eng._publish_wx(re_state)

    # The final WX score should be the better one (from uncharged).
    assert re_state.wx_matrix.get(0, 1) == 3.0
    # A backpointer should be set indicating this choice.
    bp = re_state.wx_back_ptr.get(0, 1)
    assert bp is not None and bp.op is EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED


def test_publish_vx_prefers_unscaled_uncharged_and_sets_backpointer():
    """
    Tests that `_publish_vx` selects the best score between charged/uncharged states.
    """
    n = 2
    re_state = init_eddy_rivas_fold_state(n)
    cfg = EddyRivasFoldingConfig(
        costs=PseudoknotEnergies(
            q_ss=0.0, p_tilde_out=0.0, p_tilde_hole=0.0, q_tilde_out=0.0, q_tilde_hole=0.0,
            l_tilde=0.0, r_tilde=0.0, m_tilde_yhx=0.0, m_tilde_vhx=0.0, m_tilde_whx=0.0
        )
    )
    eng = EddyRivasFoldingEngine(cfg)

    # Set up the test case: make the "uncharged" score better.
    re_state.vxu_matrix.set(0, 1, 1.25)
    re_state.vxc_matrix.set(0, 1, 7.0)

    # Run the publish step.
    eng._publish_vx(re_state)

    # The final VX score should be the better one.
    assert re_state.vx_matrix.get(0, 1) == 1.25
    # A backpointer should be set indicating the choice.
    bp = re_state.vx_back_ptr.get(0, 1)
    assert bp is not None and bp.op is EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED


# -------------------- fill_with_costs: call chain smoke --------------------

def test_fill_with_costs_calls_internal_steps_in_expected_order(monkeypatch):
    """
    Smoke test to verify the calling order of subroutines in `fill_with_costs`.

    This test doesn't check for correct energy values. Instead, it uses `monkeypatch`
    to replace the internal DP methods with stubs. It then asserts that the main
    `fill_with_costs` method calls these stubs in the correct, prescribed sequence,
    confirming the overall algorithmic flow.
    """
    # Setup with minimal (zero) costs, as values don't matter for this test.
    costs = PseudoknotEnergies(
        q_ss=0.0, p_tilde_out=1.0, p_tilde_hole=1.0, q_tilde_out=0.0, q_tilde_hole=0.0,
        l_tilde=0.0, r_tilde=0.0, m_tilde_yhx=0.0, m_tilde_vhx=0.0, m_tilde_whx=0.0,
    )
    cfg = EddyRivasFoldingConfig(costs=costs)
    eng = EddyRivasFoldingEngine(cfg)
    nested = make_fold_state(3)
    re_state = init_eddy_rivas_fold_state(3)

    # This list will record the order in which the stubbed methods are called.
    calls = []

    # Patch the static `_seed_from_nested` method.
    orig_seed = EddyRivasFoldingEngine._seed_from_nested
    def seed_wrapper(nested_arg, re_arg):
        calls.append("_seed_from_nested")
        return orig_seed(nested_arg, re_arg)
    monkeypatch.setattr(EddyRivasFoldingEngine, "_seed_from_nested", staticmethod(seed_wrapper))

    # Helper function to create a stub method that just records its name.
    def make_stub(label):
        def stub(self, *args, **kwargs):
            calls.append(label)
        return stub

    # Patch all internal DP and composition methods.
    monkeypatch.setattr(EddyRivasFoldingEngine, "_dp_whx", make_stub("_dp_whx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_dp_vhx", make_stub("_dp_vhx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_dp_zhx", make_stub("_dp_zhx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_dp_yhx", make_stub("_dp_yhx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_compose_wx", make_stub("_compose_wx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_publish_wx", make_stub("_publish_wx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_compose_vx", make_stub("_compose_vx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_publish_vx", make_stub("_publish_vx"))

    # Execute the main folding method.
    eng.fill_with_costs("ACG", nested, re_state)

    # Assert that the recorded call order matches the expected algorithm flow.
    assert calls == [
        "_seed_from_nested",
        "_dp_whx",
        "_dp_vhx",
        "_dp_zhx",
        "_dp_yhx",
        "_compose_wx",
        "_publish_wx",
        "_compose_vx",
        "_publish_vx",
    ]
