#!/usr/bin/env python3
"""
Predict RNA secondary structure (nested or pseudoknotted) from the command line.

Examples:
  python predict_rna.py "GGGAAACCCAAAGGGUUUCCC"
  python predict_rna.py --engine er --json "UUCUUUUUUAGUGGCAGUAAGCCUGGGAAUGGGGGCGACCCAGGCGUAUGAACAUAGUGUAACGCUCCCC"
  python predict_rna.py --tempC 25 --yaml /path/to/turner2004_eddyrivas1999_min.yaml "ACGU..."

Notes:
- Default engine is "auto": tries Eddy–Rivas first (full PK), falls back to nested Zuker if needed.
- Eddy–Rivas PK energies are loaded from YAML file but flags can be tweaked for exploration.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from typing import Tuple, Optional

# --- Package imports (must be installed / on PYTHONPATH) ---
from importlib.resources import files as ir_files

from rna_pk_fold.energies import SecondaryStructureEnergyLoader
from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModel

# Nested (Zuker)
from rna_pk_fold.folding.zucker import make_fold_state as make_z_state
from rna_pk_fold.folding.zucker.zucker_recurrences import ZuckerFoldingConfig, ZuckerFoldingEngine
from rna_pk_fold.folding.zucker.zucker_traceback import traceback_nested as zucker_traceback

# Interval tracer for ER full traceback
from rna_pk_fold.folding.zucker.zucker_traceback import traceback_nested_interval

# Eddy–Rivas (pseudoknots)
from rna_pk_fold.folding.eddy_rivas import eddy_rivas_recurrences
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import make_re_fold_state
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_traceback import traceback_with_pk as er_traceback_with_pk


# --------------------------
# Helpers
# --------------------------
def validate_and_normalize_seq(raw: str) -> str:
    s = raw.strip().upper().replace("T", "U")
    if not s:
        raise ValueError("Sequence is empty.")
    allowed = set("ACGU")
    bad = [i for i, ch in enumerate(s) if ch not in allowed]
    if bad:
        pos = bad[0]
        raise ValueError(
            f"Invalid character at position {pos} ('{s[pos]}'). Only A,C,G,U (or T) are allowed."
        )
    return s


def load_energy_model(yaml_path: Optional[str], temp_c: float) -> SecondaryStructureEnergyModel:
    if yaml_path is None:
        yaml_path = str(ir_files("rna_pk_fold") / "data" / "turner2004_eddyrivas1999_min.yaml")
    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=yaml_path)
    return SecondaryStructureEnergyModel(params=params, temp_k=273.15 + temp_c)


def build_er_costs(energy_model: SecondaryStructureEnergyModel,
                   q_ss: float,
                   gw_override: Optional[float]) -> eddy_rivas_recurrences.PseudoknotEnergies:
    """Minimal PK cost set; keep simple unless you want to tune."""
    # Try to take Gw from model if present, else use override, else default 7.0
    try:
        model_gw = getattr(getattr(energy_model.params, "PSEUDOKNOT", None), "pk_penalty_gw", None)
    except Exception:
        model_gw = None
    Gw = gw_override if gw_override is not None else (model_gw if model_gw is not None else 7.0)

    return eddy_rivas_recurrences.costs_from_dict({
        # Backbone / tilde terms kept neutral by default
        "q_ss": q_ss,
        "P_tilde_out": 0.0, "P_tilde_hole": 0.0,
        "Q_tilde_out": 0.0, "Q_tilde_hole": 0.0,
        "L_tilde": 0.0, "R_tilde": 0.0,
        "M_tilde_yhx": 0.0, "M_tilde_vhx": 0.0, "M_tilde_whx": 0.0,
        # Interface/charging terms often tuned; start neutral
        "Gwh": 0.0, "Gwi": 0.0,
        # Optional extras (keep 0.0 unless you’ve tuned)
        "coax_scale": 0.0, "coax_bonus": 0.0,
        # Pseudoknot penalty used in engine cfg, but keep here for completeness
        "pk_penalty_gw": Gw,
    })


def predict_nested(seq: str, energy_model: SecondaryStructureEnergyModel) -> Tuple[str, float]:
    z_cfg = ZuckerFoldingConfig()
    z_engine = ZuckerFoldingEngine(energy_model=energy_model, config=z_cfg)
    z_state = make_z_state(len(seq))
    z_engine.fill_all_matrices(seq, z_state)

    tr = zucker_traceback(seq, z_state)
    e = z_state.w_matrix.get(0, len(seq) - 1)
    return tr.dot_bracket, float(e)


def predict_er(
    seq: str,
    energy_model: SecondaryStructureEnergyModel,
    pk_penalty_gw: Optional[float],
    enable_coax: bool,
    enable_overlap: bool,
    min_hole_width: int,
    max_hole_width: int,
    q_ss: float
) -> Tuple[str, float]:
    """
    Full Eddy–Rivas prediction. Falls back to nested traceback if ER traceback
    machinery is not available.
    """
    # 1) Nested setup (Eddy Rivas builds on Zuker’s WX/V)
    z_cfg = ZuckerFoldingConfig()
    z_engine = ZuckerFoldingEngine(energy_model=energy_model, config=z_cfg)
    z_state = make_z_state(len(seq))
    z_engine.fill_all_matrices(seq, z_state)

    # 2) Eddy Rivas engine & state
    costs = build_er_costs(energy_model, q_ss=q_ss, gw_override=pk_penalty_gw)
    er_cfg = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=enable_coax,
        enable_coax_variants=enable_coax,
        enable_coax_mismatch=enable_coax,
        enable_wx_overlap=enable_overlap,
        enable_is2=False,
        enable_join_drift=False,
        min_hole_width=min_hole_width,
        max_hole_width=max_hole_width,
        pk_penalty_gw=costs.pk_penalty_gw,
        costs=costs,
    )
    er_engine = eddy_rivas_recurrences.EddyRivasFoldingEngine(er_cfg)
    re_state = make_re_fold_state(len(seq))
    er_engine.fill_with_costs(seq, z_state, re_state)

    # Energy is WX(0,n-1) from ER publish step
    e = re_state.wx_matrix.get(0, len(seq) - 1)
    if not math.isfinite(e):
        # If something went wrong, return nested as a safe fallback
        tr = zucker_traceback(seq, z_state)
        e_nested = z_state.w_matrix.get(0, len(seq) - 1)
        return tr.dot_bracket, float(e_nested)

    # Prefer full multilayer traceback if available; else nested-only shape
    if er_traceback_with_pk is not None and traceback_nested_interval is not None:
        tr_full = er_traceback_with_pk(
            seq,
            nested_state=z_state,
            re_state=re_state,
            trace_nested_interval=traceback_nested_interval,
        )
        return tr_full.dot_bracket, float(e)
    else:
        tr = zucker_traceback(seq, z_state)
        return tr.dot_bracket, float(e)


# --------------------------
# CLI
# --------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Predict RNA structure (dot-bracket) and ΔG.")
    p.add_argument("sequence", help="RNA sequence (A,C,G,U; T will be converted to U)")
    p.add_argument("--engine", choices=["auto", "zucker", "eddy_rivas"], default="auto",
                   help="Which predictor to use (default: auto).")
    p.add_argument("--yaml", default=None,
                   help="Path to parameter YAML (defaults to package data).")
    p.add_argument("--tempC", type=float, default=37.0,
                   help="Temperature in °C (default: 37.0).")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON instead of human-readable text.")

    # ER tuning (simple subset)
    p.add_argument("--pk-gw", type=float, default=None,
                   help="Override pseudoknot penalty Gw (kcal/mol).")
    p.add_argument("--coax", action="store_true",
                   help="Enable coaxial stacking terms in ER (default: off).")
    p.add_argument("--overlap", action="store_true",
                   help="Enable WX overlap path in ER (default: off).")
    p.add_argument("--min-hole-width", type=int, default=0,
                   help="Minimum hole width (k,l) seam interior (default: 0).")
    p.add_argument("--max-hole-width", type=int, default=0,
                   help="Maximum hole width (0 means no cap).")
    p.add_argument("--q-ss", type=float, default=0.0,
                   help="Backbone per-SS penalty used by ER recurrences (default: 0.0).")

    args = p.parse_args(argv)

    # Validate & normalize sequence
    try:
        seq = validate_and_normalize_seq(args.sequence)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Load energy model
    try:
        energy_model = load_energy_model(args.yaml, args.tempC)
    except Exception as e:
        print(f"Failed to load energy model YAML: {e}", file=sys.stderr)
        return 2

    # Choose engine
    engine_used = args.engine
    try:
        if args.engine == "zucker":
            db, dg = predict_nested(seq, energy_model)
        elif args.engine == "eddy_rivas":
            db, dg = predict_er(
                seq,
                energy_model,
                pk_penalty_gw=args.pk_gw,
                enable_coax=args.coax,
                enable_overlap=args.overlap,
                min_hole_width=args.min_hole_width,
                max_hole_width=args.max_hole_width,
                q_ss=args.q_ss,
            )
        else:
            # auto: try ER, fallback to nested if ER traceback is missing or energy NaN/inf
            try:
                db, dg = predict_er(
                    seq,
                    energy_model,
                    pk_penalty_gw=args.pk_gw,
                    enable_coax=args.coax,
                    enable_overlap=args.overlap,
                    min_hole_width=args.min_hole_width,
                    max_hole_width=args.max_hole_width,
                    q_ss=args.q_ss,
                )
                engine_used = "eddy_rivas"
            except Exception:
                db, dg = predict_nested(seq, energy_model)
                engine_used = "zucker"
    except Exception as e:
        print(f"Prediction failed: {e}", file=sys.stderr)
        return 1

    # Output
    if args.json:
        print(json.dumps({
            "engine": engine_used,
            "sequence": seq,
            "dot_bracket": db,
            "delta_G_kcal_per_mol": dg,
            "length": len(seq),
        }, indent=2))
    else:
        print(f"Engine : {engine_used}")
        print(f"Sequence Length : {len(seq)}")
        print(f"sequence : {seq}")
        print(f"Dot-Bracket Notation: {db}")
        print(f"ΔG (kcal/mol): {dg:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
