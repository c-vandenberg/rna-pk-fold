"""
Unit tests for the `EddyRivasFoldingEngine`, the core of the pseudoknot folding algorithm.
"""
import math

from rna_pk_fold.folding.zucker import make_fold_state
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import init_eddy_rivas_fold_state
from rna_pk_fold.energies.energy_types import PseudoknotEnergies

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_dynamic_programming import EddyRivasFoldingEngine, EddyRivasFoldingConfig
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBacktrackOp

# -------------------- Seed From Nested (static) --------------------

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
    nested.w_matrix.set_energy(0, 2, 7.0)
    nested.v_matrix.set_energy(0, 2, 9.5)

    # 2. Create a fresh Rivas-Eddy (RE) state.
    re_state = init_eddy_rivas_fold_state(n)
    # Sanity check: ensure initial values are as expected before seeding.
    assert math.isinf(re_state.wxc_matrix.get_energy(0, 2))
    assert re_state.wxc_matrix.get_energy(1, 1) == 0.0

    # 3. Run the seeding process.
    EddyRivasFoldingEngine._seed_from_nested(nested, re_state)

    # 4. Verify the results.
    # The "uncharged" matrices should mirror the nested fold.
    assert re_state.wxu_matrix.get_energy(0, 2) == 7.0
    assert re_state.vxu_matrix.get_energy(0, 2) == 9.5
    # The "charged" matrices should remain at their default (+inf for off-diagonal).
    assert math.isinf(re_state.wxc_matrix.get_energy(0, 2))
    assert math.isinf(re_state.vxc_matrix.get_energy(0, 2))
    # The main WX/VX matrices should be populated with the best score (from uncharged).
    assert re_state.wx_matrix.get_energy(0, 2) == 7.0
    assert re_state.vx_matrix.get_energy(0, 2) == 9.5
    # The WXI matrix should also mirror the initial W matrix.
    assert re_state.wxi_matrix.get_energy(0, 2) == 7.0


# -------------------- publish WX/VX selection --------------------

def test_publish_wx_prefers_unscaled_uncharged_and_sets_backpointer():
    """
    Tests that `_publish_wx` selects the best score between charged/uncharged states.
    """
    n = 2
    re_state = init_eddy_rivas_fold_state(n)
    # Use zero-cost energies to isolate the logic being tested.
    cfg = EddyRivasFoldingConfig(
        pk_energies=PseudoknotEnergies(
            q_ss=0.0, p_tilde_out=0.0, p_tilde_hole=0.0, q_tilde_out=0.0, q_tilde_hole=0.0,
            l_tilde=0.0, r_tilde=0.0, m_tilde_yhx=0.0, m_tilde_vhx=0.0, m_tilde_whx=0.0
        )
    )
    eng = EddyRivasFoldingEngine(cfg)

    # Set up the test case: make the "uncharged" score better than "charged".
    re_state.wxu_matrix.set_energy(0, 1, 3.0)
    re_state.wxc_matrix.set_energy(0, 1, 5.0)

    # Run the publish step.
    eng._publish_wx_min_energy(re_state)

    # The final WX score should be the better one (from uncharged).
    assert re_state.wx_matrix.get_energy(0, 1) == 3.0
    # A backpointer should be set indicating this choice.
    bp = re_state.wx_back_ptr.get_backpointer(0, 1)
    assert bp is not None and bp.op is EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED


def test_publish_vx_prefers_unscaled_uncharged_and_sets_backpointer():
    """
    Tests that `_publish_vx` selects the best score between charged/uncharged states.
    """
    n = 2
    re_state = init_eddy_rivas_fold_state(n)
    cfg = EddyRivasFoldingConfig(
        pk_energies=PseudoknotEnergies(
            q_ss=0.0, p_tilde_out=0.0, p_tilde_hole=0.0, q_tilde_out=0.0, q_tilde_hole=0.0,
            l_tilde=0.0, r_tilde=0.0, m_tilde_yhx=0.0, m_tilde_vhx=0.0, m_tilde_whx=0.0
        )
    )
    eng = EddyRivasFoldingEngine(cfg)

    # Set up the test case: make the "uncharged" score better.
    re_state.vxu_matrix.set_energy(0, 1, 1.25)
    re_state.vxc_matrix.set_energy(0, 1, 7.0)

    # Run the publish step.
    eng._publish_vx_min_energy(re_state)

    # The final VX score should be the better one.
    assert re_state.vx_matrix.get_energy(0, 1) == 1.25
    # A backpointer should be set indicating the choice.
    bp = re_state.vx_back_ptr.get_backpointer(0, 1)
    assert bp is not None and bp.op is EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED


# -------------------- fill_with_costs: call chain smoke --------------------
def test_fill_with_costs_calls_internal_steps_in_expected_order(monkeypatch):
    """
    Smoke test to verify the calling order of subroutines in `fill_with_costs`.

    This test stubs the refactored method names used by `EddyRivasFoldingEngine`:
      - _fill_whx_gap_matrix
      - _fill_vhx_gap_matrix
      - _fill_zhx_gap_matrix
      - _fill_yhx_gap_matrix
      - _compose_wx_from_gapped_fragments
      - _publish_wx_min_energy
      - _compose_vx_from_zhx_fragments
      - _publish_vx_min_energy
    It asserts that `run_eddy_rivas_dp_with_costs` calls them in the expected order.
    """
    # Setup with minimal (zero) costs, as values don't matter for this test.
    costs = PseudoknotEnergies(
        q_ss=0.0, p_tilde_out=1.0, p_tilde_hole=1.0, q_tilde_out=0.0, q_tilde_hole=0.0,
        l_tilde=0.0, r_tilde=0.0, m_tilde_yhx=0.0, m_tilde_vhx=0.0, m_tilde_whx=0.0,
    )
    cfg = EddyRivasFoldingConfig(pk_energies=costs)
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

    # Helper to create a stub that records its label.
    def make_stub(label):
        def stub(self, *args, **kwargs):
            calls.append(label)
        return stub

    # Patch all internal DP and composition methods (refactored names).
    monkeypatch.setattr(EddyRivasFoldingEngine, "_fill_whx_gap_matrix",
                        make_stub("_fill_whx_gap_matrix"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_fill_vhx_gap_matrix",
                        make_stub("_fill_vhx_gap_matrix"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_fill_zhx_gap_matrix",
                        make_stub("_fill_zhx_gap_matrix"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_fill_yhx_gap_matrix",
                        make_stub("_fill_yhx_gap_matrix"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_compose_wx_from_gapped_fragments",
                        make_stub("_compose_wx_from_gapped_fragments"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_publish_wx_min_energy",
                        make_stub("_publish_wx_min_energy"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_compose_vx_from_zhx_fragments",
                        make_stub("_compose_vx_from_zhx_fragments"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_publish_vx_min_energy",
                        make_stub("_publish_vx_min_energy"))

    # Execute the main folding method.
    eng.run_eddy_rivas_dp_with_costs("ACG", nested, re_state)

    # Assert the expected call order for the refactored method names.
    assert calls == [
        "_seed_from_nested",
        "_fill_whx_gap_matrix",
        "_fill_vhx_gap_matrix",
        "_fill_zhx_gap_matrix",
        "_fill_yhx_gap_matrix",
        "_compose_wx_from_gapped_fragments",
        "_publish_wx_min_energy",
        "_compose_vx_from_zhx_fragments",
        "_publish_vx_min_energy",
    ]

