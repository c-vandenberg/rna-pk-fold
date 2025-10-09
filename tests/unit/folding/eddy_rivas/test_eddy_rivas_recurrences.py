import json
import math
import builtins
import pytest

from rna_pk_fold.folding.zucker import make_fold_state
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import init_eddy_rivas_fold_state
from rna_pk_fold.energies.energy_types import PseudoknotEnergies

from rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences import (
    EddyRivasFoldingEngine,
    EddyRivasFoldingConfig,
    take_best,
    make_bp,
    costs_from_dict,
    costs_to_dict,
    costs_from_vienna_like,
    save_costs_json,
    load_costs_json,
    quick_energy_harness,
)
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import (
    EddyRivasBackPointer,
    EddyRivasBacktrackOp,
)


# -------------------- take_best / make_bp --------------------

def test_take_best_replaces_when_better_and_calls_factory_once():
    factory_calls = {"n": 0}
    def mk():
        factory_calls["n"] += 1
        return EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE)

    best, bp = take_best(10.0, None, 5.0, mk)
    assert best == 5.0
    assert isinstance(bp, EddyRivasBackPointer)
    assert factory_calls["n"] == 1

def test_take_best_keeps_old_on_tie_or_worse():
    old_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_SS_LEFT)
    # worse
    best, bp = take_best(3.0, old_bp, 4.0, lambda: None)
    assert best == 3.0 and bp is old_bp
    # tie
    best2, bp2 = take_best(3.0, old_bp, 3.0, lambda: None)
    assert best2 == 3.0 and bp2 is old_bp

def test_make_bp_closure_sets_outer_and_hole():
    BP = make_bp(1, 7, 2, 6)
    bp = BP(EddyRivasBacktrackOp.RE_WHX_TRIM_LEFT, note="x")
    assert bp.outer == (1, 7)
    assert bp.hole == (2, 6)
    assert bp.op is EddyRivasBacktrackOp.RE_WHX_TRIM_LEFT
    assert bp.note == "x"


# -------------------- _seed_from_nested (static) --------------------

def test_seed_from_nested_copies_nested_into_uncharged_and_wx_vx():
    n = 3
    nested = make_fold_state(n)
    # Give deterministic values
    nested.w_matrix.set(0, 2, 7.0)
    nested.v_matrix.set(0, 2, 9.5)

    re_state = init_eddy_rivas_fold_state(n)
    # sanity: initial charged cells for i!=j are +inf, diagonals 0
    assert math.isinf(re_state.wxc_matrix.get(0, 2))
    assert re_state.wxc_matrix.get(1, 1) == 0.0

    # run seed
    EddyRivasFoldingEngine._seed_from_nested(nested, re_state)

    # wxu/vxu mirrors nested
    assert re_state.wxu_matrix.get(0, 2) == 7.0
    assert re_state.vxu_matrix.get(0, 2) == 9.5
    # charged (off-diagonal) set to +inf
    assert math.isinf(re_state.wxc_matrix.get(0, 2))
    assert math.isinf(re_state.vxc_matrix.get(0, 2))
    # wx/vx propagated
    assert re_state.wx_matrix.get(0, 2) == 7.0
    assert re_state.vx_matrix.get(0, 2) == 9.5
    # wxi (if present) mirrors w
    assert re_state.wxi_matrix.get(0, 2) == 7.0


# -------------------- publish WX/VX selection --------------------

def test_publish_wx_prefers_unscaled_uncharged_and_sets_backpointer():
    n = 2
    re_state = init_eddy_rivas_fold_state(n)
    cfg = EddyRivasFoldingConfig(
        costs=PseudoknotEnergies(
            q_ss=0.0, P_tilde_out=0.0, P_tilde_hole=0.0, Q_tilde_out=0.0, Q_tilde_hole=0.0,
            L_tilde=0.0, R_tilde=0.0, M_tilde_yhx=0.0, M_tilde_vhx=0.0, M_tilde_whx=0.0
        )
    )
    eng = EddyRivasFoldingEngine(cfg)

    # Make uncharged better than charged
    re_state.wxu_matrix.set(0, 1, 3.0)
    re_state.wxc_matrix.set(0, 1, 5.0)

    eng._publish_wx(re_state)
    assert re_state.wx_matrix.get(0, 1) == 3.0
    bp = re_state.wx_back_ptr.get(0, 1)
    assert bp is not None and bp.op is EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED

def test_publish_vx_prefers_unscaled_uncharged_and_sets_backpointer():
    n = 2
    re_state = init_eddy_rivas_fold_state(n)
    cfg = EddyRivasFoldingConfig(
        costs=PseudoknotEnergies(
            q_ss=0.0, P_tilde_out=0.0, P_tilde_hole=0.0, Q_tilde_out=0.0, Q_tilde_hole=0.0,
            L_tilde=0.0, R_tilde=0.0, M_tilde_yhx=0.0, M_tilde_vhx=0.0, M_tilde_whx=0.0
        )
    )
    eng = EddyRivasFoldingEngine(cfg)

    re_state.vxu_matrix.set(0, 1, 1.25)
    re_state.vxc_matrix.set(0, 1, 7.0)

    eng._publish_vx(re_state)
    assert re_state.vx_matrix.get(0, 1) == 1.25
    bp = re_state.vx_back_ptr.get(0, 1)
    assert bp is not None and bp.op is EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED


# -------------------- costs: dict/json helpers --------------------

def test_costs_from_dict_uses_defaults_for_missing_fields():
    # Provide all required tilde scalars + a couple of extras; unspecified fields
    # should fall back to the dataclass defaults (e.g., mismatch_coax_scale=0.5).
    minimal = {
        "q_ss": 0.5,
        "P_tilde_out": 1.0,
        "P_tilde_hole": 1.0,
        "Q_tilde_out": 0.0,
        "Q_tilde_hole": 0.0,
        "L_tilde": 0.0,
        "R_tilde": 0.0,
        "M_tilde_yhx": 0.0,
        "M_tilde_vhx": 0.0,
        "M_tilde_whx": 0.0,
        # plus one optional override so we can verify it sticks
        "coax_scale": 2.0,
    }
    c = costs_from_dict(minimal)

    # Required fields set as provided
    assert c.q_ss == 0.5
    assert c.P_tilde_out == 1.0 and c.P_tilde_hole == 1.0
    assert c.Q_tilde_out == 0.0 and c.Q_tilde_hole == 0.0
    assert c.L_tilde == 0.0 and c.R_tilde == 0.0
    assert c.M_tilde_yhx == 0.0 and c.M_tilde_vhx == 0.0 and c.M_tilde_whx == 0.0

    # Optional field override sticks
    assert c.coax_scale == 2.0

    # Unspecified optional fields use dataclass defaults
    assert c.coax_bonus == 0.0
    assert c.mismatch_coax_scale == 0.5
    assert c.coax_min_helix_len == 1
    assert c.short_hole_caps is None
    assert c.coax_pairs is None

def test_costs_to_dict_roundtrip_scalars_only(tmp_path):
    costs = PseudoknotEnergies(
        q_ss=0.1,
        P_tilde_out=1.0, P_tilde_hole=1.0,
        Q_tilde_out=0.0, Q_tilde_hole=0.0,
        L_tilde=0.0, R_tilde=0.0,
        M_tilde_yhx=0.0, M_tilde_vhx=0.0, M_tilde_whx=0.0,
        # leave dict fields None to keep JSON serializable keys
    )
    p = tmp_path / "costs.json"
    save_costs_json(str(p), costs)
    reloaded = load_costs_json(str(p))

    # Compare dict views (JSON-safe subset)
    d1 = costs_to_dict(costs)
    d2 = costs_to_dict(reloaded)
    # All scalar fields must survive identically (dict-valued fields are None)
    assert d1 == d2

def test_costs_from_vienna_like_maps_keys_and_types():
    # Include the required tilde scalars so costs_from_dict() can construct the dataclass.
    tbl = {
        "q_ss": 0.2,
        "P_tilde_out": 1.0, "P_tilde_hole": 1.0,
        "Q_tilde_out": 0.0, "Q_tilde_hole": 0.0,
        "L_tilde": 0.0, "R_tilde": 0.0,
        "M_tilde_yhx": 0.0, "M_tilde_vhx": 0.0, "M_tilde_whx": 0.0,

        "coax_pairs": {"GC|CG": -0.5, "AU|UA": -0.3},
        "dangle_outer_left": {"GA": -0.1},
        "dangle_outer_right": {"UC": -0.2},
        "dangle_hole_left": {"AG": -0.15},
        "dangle_hole_right": {"CU": -0.25},
        "short_hole_caps": {"1": 2.0, "2": 1.5},
    }
    c = costs_from_vienna_like(tbl)

    # scalars copied
    assert c.q_ss == 0.2
    assert c.P_tilde_out == 1.0 and c.P_tilde_hole == 1.0
    assert c.Q_tilde_out == 0.0 and c.Q_tilde_hole == 0.0
    assert c.L_tilde == 0.0 and c.R_tilde == 0.0
    assert c.M_tilde_yhx == 0.0 and c.M_tilde_vhx == 0.0 and c.M_tilde_whx == 0.0

    # coax tuple-keys generated
    assert c.coax_pairs == {("GC", "CG"): -0.5, ("AU", "UA"): -0.3}

    # dangles mapped to tuple keys
    assert c.dangle_outer_left == {("G", "A"): -0.1}
    assert c.dangle_outer_right == {("U", "C"): -0.2}
    assert c.dangle_hole_left == {("A", "G"): -0.15}
    assert c.dangle_hole_right == {("C", "U"): -0.25}

    # short-hole caps int keys
    assert c.short_hole_caps == {1: 2.0, 2: 1.5}

# -------------------- fill_with_costs: call chain smoke --------------------

def test_fill_with_costs_calls_internal_steps_in_expected_order(monkeypatch):
    # minimal costs (all zeros fine for this smoke test)
    costs = PseudoknotEnergies(
        q_ss=0.0,
        P_tilde_out=1.0, P_tilde_hole=1.0,
        Q_tilde_out=0.0, Q_tilde_hole=0.0,
        L_tilde=0.0, R_tilde=0.0,
        M_tilde_yhx=0.0, M_tilde_vhx=0.0, M_tilde_whx=0.0,
    )
    cfg = EddyRivasFoldingConfig(costs=costs)
    eng = EddyRivasFoldingEngine(cfg)

    nested = make_fold_state(3)
    re_state = init_eddy_rivas_fold_state(3)

    calls = []

    # Wrap seed to keep functionality and record order
    orig_seed = EddyRivasFoldingEngine._seed_from_nested
    def seed_wrapper(nested_arg, re_arg):
        calls.append("_seed_from_nested")
        return orig_seed(nested_arg, re_arg)
    monkeypatch.setattr(EddyRivasFoldingEngine, "_seed_from_nested", staticmethod(seed_wrapper))

    def make_stub(label):
        def stub(self, *args, **kwargs):
            calls.append(label)
        return stub

    monkeypatch.setattr(EddyRivasFoldingEngine, "_dp_whx", make_stub("_dp_whx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_dp_vhx", make_stub("_dp_vhx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_dp_zhx", make_stub("_dp_zhx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_dp_yhx", make_stub("_dp_yhx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_compose_wx", make_stub("_compose_wx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_publish_wx", make_stub("_publish_wx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_compose_vx", make_stub("_compose_vx"))
    monkeypatch.setattr(EddyRivasFoldingEngine, "_publish_vx", make_stub("_publish_vx"))

    eng.fill_with_costs("ACG", nested, re_state)

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


# -------------------- quick_energy_harness --------------------

def test_quick_energy_harness_reports_end_cells(monkeypatch):
    # dummy costs
    costs = PseudoknotEnergies(
        q_ss=0.0,
        P_tilde_out=1.0, P_tilde_hole=1.0,
        Q_tilde_out=0.0, Q_tilde_hole=0.0,
        L_tilde=0.0, R_tilde=0.0,
        M_tilde_yhx=0.0, M_tilde_vhx=0.0, M_tilde_whx=0.0,
    )
    cfg = EddyRivasFoldingConfig(costs=costs)

    # Build states
    n = 4
    nested = make_fold_state(n)
    re_state = init_eddy_rivas_fold_state(n)

    # Make fill_with_costs place known values in (0, n-1)
    def fake_fill(self, seq, nested_arg, re_arg):
        re_arg.wx_matrix.set(0, n - 1, -1.0)
        re_arg.vx_matrix.set(0, n - 1, -2.5)

    monkeypatch.setattr(EddyRivasFoldingEngine, "fill_with_costs", fake_fill)

    out = quick_energy_harness("ACGU", cfg, nested, re_state)
    assert out == {"W(0,n-1)": -1.0, "V(0,n-1)": -2.5}
