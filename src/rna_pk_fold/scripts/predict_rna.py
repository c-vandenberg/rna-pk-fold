#!/usr/bin/env python3
"""
Predict RNA secondary structure (nested or pseudoknotted) from the command line.

Examples:
  python predict_rna.py "GGGAAACCCAAAGGGUUUCCC"
  python predict_rna.py --engine er --json "UUCUUUUUUAGUGGCAGUAAGCCUGGGAAUGGGGGCGACCCAGGCGUAUGAACAUAGUGUAACGCUCCCC"
  python predict_rna.py -vv --tempC 25 --yaml /path/to/turner2004_eddyrivas1999_min.yaml "ACGU..."

Notes:
- Default engine is "auto": tries Eddy–Rivas first (full PK), falls back to nested Zuker if needed.
- Eddy–Rivas PK energies are loaded from YAML file but flags can be tweaked for exploration.
- Use -v for INFO logs, -vv for DEBUG logs. Logs saved to var/log/ by default.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import logging
import time
from typing import Tuple, Optional

# --- Package imports (must be installed / on PYTHONPATH) ---
from importlib.resources import files as ir_files

# Add logging setup
from rna_pk_fold.utils.logging_config import setup_logger, DEFAULT_LOG_DIR

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

# Set up module logger
logger = logging.getLogger(__name__)


# --------------------------
# Logging Configuration
# --------------------------
def setup_cli_logging(verbose: int, log_file: Optional[str] = None) -> None:
    """
    Configure logging based on CLI arguments.

    Parameters
    ----------
    verbose : int
        Verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
    log_file : str, optional
        Path to log file. If None and verbose > 0, creates default log file.
    """
    # Map verbosity to log level
    level_map = {
        0: logging.WARNING,  # Quiet mode - only warnings/errors
        1: logging.INFO,  # Normal - show progress
        2: logging.DEBUG,  # Verbose - show everything
    }
    level = level_map.get(verbose, logging.INFO)

    # Determine if we should log to file
    enable_file = (verbose > 0) or (log_file is not None)

    # Set up main script logger
    setup_logger(
        __name__,
        level=level,
        log_file=log_file,
        enable_file_logging=enable_file,
        enable_tqdm=True
    )

    # Set up module loggers
    setup_logger(
        "rna_pk_fold.folding.zucker.zucker_recurrences",
        level=level,
        log_file=log_file,
        enable_file_logging=enable_file,
        enable_tqdm=True
    )

    setup_logger(
        "rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences",
        level=level,
        log_file=log_file,
        enable_file_logging=enable_file,
        enable_tqdm=True
    )

    setup_logger(
        "rna_pk_fold.folding.eddy_rivas.eddy_rivas_traceback",
        level=level,
        log_file=log_file,
        enable_file_logging=enable_file,
        enable_tqdm=True
    )

    if enable_file and log_file is None:
        logger.info(f"Logs will be saved to: {DEFAULT_LOG_DIR.resolve()}")


# --------------------------
# Helpers
# --------------------------
def validate_and_normalize_seq(raw: str) -> str:
    logger.debug(f"Validating sequence: {raw[:50]}{'...' if len(raw) > 50 else ''}")
    s = raw.strip().upper().replace("T", "U")
    if not s:
        logger.error("Sequence is empty")
        raise ValueError("Sequence is empty.")
    allowed = set("ACGU")
    bad = [i for i, ch in enumerate(s) if ch not in allowed]
    if bad:
        pos = bad[0]
        logger.error(f"Invalid character at position {pos}: '{s[pos]}'")
        raise ValueError(
            f"Invalid character at position {pos} ('{s[pos]}'). Only A,C,G,U (or T) are allowed."
        )
    logger.info(f"Sequence validated: length={len(s)}")
    return s


def load_energy_model(yaml_path: Optional[str], temp_c: float) -> SecondaryStructureEnergyModel:
    if yaml_path is None:
        yaml_path = str(ir_files("rna_pk_fold") / "data" / "turner2004_eddyrivas1999_min.yaml")

    logger.info(f"Loading energy model from: {yaml_path}")
    logger.info(f"Temperature: {temp_c}°C ({273.15 + temp_c:.2f}K)")

    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=yaml_path)
    model = SecondaryStructureEnergyModel(params=params, temp_k=273.15 + temp_c)

    logger.debug("Energy model loaded successfully")
    return model


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
        "q_ss": q_ss,
        "P_tilde_out": 0.0, "P_tilde_hole": 0.0,
        "Q_tilde_out": 0.0, "Q_tilde_hole": 0.0,
        "L_tilde": 0.0, "R_tilde": 0.0,
        "M_tilde_yhx": 0.0, "M_tilde_vhx": 0.0, "M_tilde_whx": 0.0,
        "Gwh": 0.0, "Gwi": 0.0,
        "coax_scale": 0.0, "coax_bonus": 0.0,
        "pk_penalty_gw": Gw,
    })


def predict_nested(seq: str, energy_model: SecondaryStructureEnergyModel) -> Tuple[str, float]:
    logger.info("=" * 60)
    logger.info("Using Zucker (nested-only) algorithm")
    logger.info("=" * 60)

    start_time = time.perf_counter()

    z_cfg = ZuckerFoldingConfig(verbose=logger.isEnabledFor(logging.INFO))
    z_engine = ZuckerFoldingEngine(energy_model=energy_model, config=z_cfg)
    z_state = make_z_state(len(seq))
    z_engine.fill_all_matrices(seq, z_state)

    tr = zucker_traceback(seq, z_state)
    e = z_state.w_matrix.get(0, len(seq) - 1)

    elapsed = time.perf_counter() - start_time
    logger.info(f"Prediction completed in {elapsed:.2f}s")
    logger.info(f"Energy: {e:.3f} kcal/mol")

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
    logger.info("=" * 60)
    logger.info("Using Eddy-Rivas (pseudoknot) algorithm")
    logger.info("=" * 60)
    logger.info(f"PK penalty Gw: {pk_penalty_gw}")
    logger.info(f"Coaxial stacking: {enable_coax}")
    logger.info(f"WX overlap: {enable_overlap}")
    logger.info(f"Hole width: [{min_hole_width}, {max_hole_width if max_hole_width > 0 else '∞'}]")

    start_time = time.perf_counter()

    # 1) Nested setup (Eddy Rivas builds on Zuker's WX/V)
    logger.info("Running nested (Zuker) phase...")
    z_cfg = ZuckerFoldingConfig(verbose=logger.isEnabledFor(logging.INFO))
    z_engine = ZuckerFoldingEngine(energy_model=energy_model, config=z_cfg)
    z_state = make_z_state(len(seq))
    z_engine.fill_all_matrices(seq, z_state)

    # 2) Eddy Rivas engine & state
    logger.info("Running pseudoknot (Eddy-Rivas) phase...")
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
        verbose=logger.isEnabledFor(logging.INFO),
    )
    er_engine = eddy_rivas_recurrences.EddyRivasFoldingEngine(er_cfg)
    re_state = make_re_fold_state(len(seq))
    er_engine.fill_with_costs(seq, z_state, re_state)

    e = re_state.wx_matrix.get(0, len(seq) - 1)
    if not math.isfinite(e):
        logger.warning("ER returned infinite energy, falling back to nested")
        tr = zucker_traceback(seq, z_state)
        e_nested = z_state.w_matrix.get(0, len(seq) - 1)
        elapsed = time.perf_counter() - start_time
        logger.info(f"Prediction completed in {elapsed:.2f}s (fallback)")
        return tr.dot_bracket, float(e_nested)

    logger.info("Running traceback...")
    if er_traceback_with_pk is not None and traceback_nested_interval is not None:
        tr_full = er_traceback_with_pk(
            seq,
            nested_state=z_state,
            re_state=re_state,
            trace_nested_interval=traceback_nested_interval,
        )
        elapsed = time.perf_counter() - start_time
        logger.info(f"Prediction completed in {elapsed:.2f}s")
        logger.info(f"Energy: {e:.3f} kcal/mol")
        return tr_full.dot_bracket, float(e)
    else:
        logger.warning("Full ER traceback not available, using nested traceback")
        tr = zucker_traceback(seq, z_state)
        elapsed = time.perf_counter() - start_time
        logger.info(f"Prediction completed in {elapsed:.2f}s")
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

    # Logging arguments
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Increase verbosity (-v=INFO, -vv=DEBUG)")
    p.add_argument("--log-file", default=None,
                   help="Path to log file (default: var/log/predict_rna_TIMESTAMP.log if verbose)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress all output except final result")

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

    # Configure logging
    verbose_level = 0 if args.quiet else args.verbose
    setup_cli_logging(verbose_level, args.log_file)

    logger.info("=" * 60)
    logger.info("RNA Structure Prediction CLI")
    logger.info("=" * 60)

    # Validate & normalize sequence
    try:
        seq = validate_and_normalize_seq(args.sequence)
    except ValueError as e:
        logger.error(f"Sequence validation failed: {e}")
        if not args.json:
            print(f"Error: {e}", file=sys.stderr)
        return 2

    # Load energy model
    try:
        energy_model = load_energy_model(args.yaml, args.tempC)
    except Exception as e:
        logger.error(f"Failed to load energy model: {e}", exc_info=True)
        if not args.json:
            print(f"Failed to load energy model YAML: {e}", file=sys.stderr)
        return 2

    # Choose engine
    engine_used = args.engine
    logger.info(f"Requested engine: {args.engine}")

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
        else:  # auto
            try:
                logger.info("Attempting Eddy-Rivas (auto mode)...")
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
            except Exception as e:
                logger.warning(f"Eddy-Rivas failed, falling back to Zuker: {e}")
                db, dg = predict_nested(seq, energy_model)
                engine_used = "zucker"
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        if not args.json:
            print(f"Prediction failed: {e}", file=sys.stderr)
        return 1

    logger.info("=" * 60)
    logger.info("Prediction successful")
    logger.info("=" * 60)

    # Output (always print to stdout, regardless of logging)
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
