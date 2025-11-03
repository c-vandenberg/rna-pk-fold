#!/usr/bin/env python3
"""
Predict RNA secondary structure (nested or pseudoknotted) from the command line.

This script provides a command-line interface to predict the minimum free energy
secondary structure of an RNA sequence using either the nested-only Zuker
algorithm or the pseudoknot-aware Eddy-Rivas algorithm.

Examples:
  - python predict_rna.py "GGGAAACCCAAAGGGUUUCCC"
  - python predict_rna.py --engine eddy_rivas --json "UUCUUUUUUAGUGGCAGUAAGCCUGGGAAUGGGGGCGACCCAGGCGUAUGAACAUAGUGUAACGCUCCCC"
  - python predict_rna.py -vv --tempC 25 --yaml /path/to/turner2004_eddyrivas1999_min.yaml "ACGU..."

"""

# --- Standard Library Imports ---
from __future__ import annotations
import argparse
import json
import math
import sys
import logging
import time
from dataclasses import replace
from typing import Tuple, Optional

# --- Local Application Imports ---
# Energy model loading and definition
from rna_pk_fold.utils.logging.logging_utils import setup_logger, DEFAULT_LOG_DIR
from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModel

# Nested (Zuker) folding components
from rna_pk_fold.folding.zucker import make_fold_state as make_zucker_state
from rna_pk_fold.folding.zucker.zucker_dynamic_programming import ZuckerFoldingConfig, ZuckerFoldingEngine
from rna_pk_fold.folding.zucker.zucker_traceback import traceback_nested as zucker_traceback
from rna_pk_fold.folding.zucker.zucker_traceback import traceback_nested_interval

# Eddy-Rivas (pseudoknot) folding components
from rna_pk_fold.folding.eddy_rivas import eddy_rivas_dynamic_programming
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import init_eddy_rivas_fold_state
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_traceback import traceback_with_pseudoknots as eddy_rivas_traceback

# Utility functions
from rna_pk_fold.utils.sequences.nucleotide_utils import validate_and_normalize_seq
from rna_pk_fold.utils.energy.energy_model_utils import load_energy_model
from rna_pk_fold.utils.dynamic_programming.dp_composition_utils import set_wx_debug_outer

# Set up module logger
logger = logging.getLogger(__name__)


# --------------------------
# Logging Configuration
# --------------------------
def setup_cli_logging(verbose_level: int, log_file: Optional[str] = None) -> None:
    """
    Configures logging for the application based on command-line arguments.

    This function sets the logging level for the main script and its key modules,
    enabling console and/or file logging based on the verbosity specified by
    the user.

    Parameters
    ----------
    verbose_level : int
        The verbosity level: 0 for WARNING, 1 for INFO, 2 for DEBUG.
    log_file : Optional[str]
        The path to a specific log file. If not provided, a default timestamped
        log file is created in the `var/log/` directory when verbosity is > 0.
    """
    # Map the integer verbosity level to the corresponding logging level constant.
    level_map = {
        0: logging.WARNING,  # Quiet mode: only show warnings and errors.
        1: logging.INFO,  # Normal mode: show progress and key steps.
        2: logging.DEBUG,  # Verbose mode: show detailed internal states.
    }
    log_level = level_map.get(verbose_level, logging.INFO)

    # Determine if file logging should be enabled.
    should_log_to_file = (verbose_level > 0) or (log_file is not None)

    # Define the modules whose loggers need to be configured.
    loggers_to_configure = [
        __name__,
        "rna_pk_fold.folding.zucker.zucker_recurrences",
        "rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences",
        "rna_pk_fold.folding.eddy_rivas.eddy_rivas_traceback",
    ]

    # Configure each logger with the determined levels and file path.
    for logger_name in loggers_to_configure:
        setup_logger(
            logger_name,
            level=log_level,
            log_file=log_file,
            enable_file_logging=should_log_to_file,
            enable_tqdm=True
        )

    # Inform the user where the logs are being saved if a default file was created.
    if should_log_to_file and log_file is None:
        logger.info(f"Logs will be saved to: {DEFAULT_LOG_DIR.resolve()}")


def predict_zucker_nested(seq: str, energy_model: SecondaryStructureEnergyModel) -> Tuple[str, float]:
    """
    Runs the Zuker (nested-only) folding algorithm on a sequence.

    Parameters
    ----------
    seq : str
        The RNA sequence to fold.
    energy_model : SecondaryStructureEnergyModel
        The initialized energy model.

    Returns
    -------
    Tuple[str, float]
        A tuple containing the predicted dot-bracket structure and its
        minimum free energy in kcal/mol.
    """
    logger.info("=" * 60)
    logger.info("Using Zucker (nested-only) algorithm")
    logger.info("=" * 60)
    start_time = time.perf_counter()

    # 1. Configure and initialize the Zuker folding engine and state.
    zucker_config = ZuckerFoldingConfig(verbose=logger.isEnabledFor(logging.INFO))
    zucker_engine = ZuckerFoldingEngine(energy_model=energy_model, config=zucker_config)
    zucker_state = make_zucker_state(len(seq))

    # 2. Run the dynamic programming algorithm to fill the matrices.
    zucker_engine.fill_all_matrices(seq, zucker_state)

    # 3. Trace back through the matrices to reconstruct the optimal structure.
    trace_result = zucker_traceback(seq, zucker_state)
    # 4. Get the final minimum free energy for the entire sequence.
    energy = zucker_state.w_matrix.get_energy(0, len(seq) - 1)

    elapsed = time.perf_counter() - start_time
    logger.info(f"Prediction completed in {elapsed:.2f}s")
    logger.info(f"Energy: {energy:.3f} kcal/mol")

    return trace_result.dot_bracket, float(energy)


def predict_eddy_rivas_non_nested(
    seq: str,
    energy_model: SecondaryStructureEnergyModel,
    enable_coax: bool,
    enable_overlap: bool,
    enable_is2: bool,
    enable_join_drift: bool,
    enable_strict_complement_order: bool
) -> Tuple[str, float]:
    """
    Predict an RNA secondary structure with pseudoknots using Eddy–Rivas DP.

    The procedure runs in two stages:
     1. A full nested (Zuker) fold is computed to seed baseline W/V matrices.
     2. The Eddy–Rivas dynamic program refines this baseline to allow pseudoknotted
     topologies.

    If the Eddy–Rivas stage yields an infinite energy, the function falls back to
    the Zuker result.

    Parameters
    ----------
    seq : str
        RNA sequence to fold (characters like A, C, G, U).
    energy_model : SecondaryStructureEnergyModel
        Initialized thermodynamic model used for both the Zuker baseline and to
        derive Eddy–Rivas pseudoknot costs.
    enable_coax : bool
        Enable coaxial stacking terms during the Eddy–Rivas stage. When `True`,
        coax variants and mismatch coax are enabled with the same flag.
    enable_overlap : bool
        Enable the WX-overlap compositions used by Eddy–Rivas (may improve
        structures involving overlapping helices).
    enable_is2 : bool
        Enable energy calculations for Irreducible Surfaces of Order 2.
    enable_join_drift : bool
        Enable slight hole shifting at a join point.
    enable_strict_complement_order : bool
        Enable strict ordering i < k <= r < l <= j for pseudoknots.

    Returns
    -------
    Tuple[str, float]
        `(dot_bracket, energy)` where `dot_bracket` is a multilayer
        dot–bracket string that may include additional bracket pairs for
        pseudoknots, and `energy` is the minimum free energy (kcal/mol)
        reported by the Eddy–Rivas WX matrix (or Zuker fallback).

    Notes
    -----
    - The Zuker result is used purely as a seed/baseline; the final structure
      and energy are taken from the Eddy–Rivas WX matrix when finite.
    - On failure of the pseudoknot stage (infinite energy), the function
      returns the Zuker traceback and its energy.
    """
    logger.info("=" * 60)
    logger.info("Using Eddy-Rivas (pseudoknot) algorithm")
    logger.info("=" * 60)
    start_time = time.perf_counter()

    # --- Phase 1: Run the nested (Zuker) algorithm to provide a baseline. ---
    logger.info("Running nested (Zuker) phase...")
    zucker_config = ZuckerFoldingConfig(verbose=logger.isEnabledFor(logging.INFO))
    zucker_engine = ZuckerFoldingEngine(energy_model=energy_model, config=zucker_config)
    zucker_state = make_zucker_state(len(seq))
    zucker_engine.fill_all_matrices(seq, zucker_state)

    # --- Phase 2: Run the Eddy-Rivas algorithm. ---
    logger.info("Running pseudoknot (Eddy-Rivas) phase...")

    # Log the CLI parameters being used for the Eddy-Rivas run.
    logger.info(f"Coaxial stacking: {enable_coax}")
    logger.info(f"WX overlap: {enable_overlap}")
    logger.info(f"Join Drift: {enable_join_drift}")

    pk_energies = energy_model.params.PSEUDOKNOT

    # Configure the Eddy-Rivas engine.
    er_config = eddy_rivas_dynamic_programming.EddyRivasFoldingConfig(
        pk_energies=pk_energies,
        enable_coax=enable_coax,
        enable_wx_overlap=enable_overlap,
        enable_coax_variants=enable_coax,
        enable_coax_mismatch=enable_coax,
        enable_join_drift=enable_join_drift,
        enable_is2=enable_is2,
        enable_strict_complement_order=enable_strict_complement_order,
        verbose=logger.isEnabledFor(logging.INFO),
    )

    er_engine = eddy_rivas_dynamic_programming.EddyRivasFoldingEngine(er_config)

    # Initialize the state object for the Eddy-Rivas matrices.
    eddy_rivas_state = init_eddy_rivas_fold_state(len(seq))

    # Run the main DP algorithm, seeding it with the results from the Zuker phase.
    er_engine.run_eddy_rivas_dp_with_costs(seq, zucker_state, eddy_rivas_state)

    # Get the final energy for the entire sequence.
    energy = eddy_rivas_state.wx_matrix.get_energy(0, len(seq) - 1)

    # If the final energy is infinite, the algorithm failed; fall back to the nested result.
    if not math.isfinite(energy):
        logger.warning("Eddy-Rivas returned infinite energy, falling back to nested result.")
        trace_result = zucker_traceback(seq, zucker_state)
        nested_energy = zucker_state.w_matrix.get_energy(0, len(seq) - 1)
        elapsed = time.perf_counter() - start_time
        logger.info(f"Prediction completed in {elapsed:.2f}s (fallback)")
        return trace_result.dot_bracket, float(nested_energy)

    # --- Phase 3: Traceback ---
    logger.info("Running traceback...")
    # Use the full pseudoknot-aware traceback function.
    trace_result = eddy_rivas_traceback(
        seq,
        nested_state=zucker_state,
        eddy_rivas_fold_state=eddy_rivas_state,
        trace_nested_interval=traceback_nested_interval, # Provide the nested tracer for subproblems.
    )
    elapsed = time.perf_counter() - start_time
    logger.info(f"Prediction completed in {elapsed:.2f}s")
    logger.info(f"Energy: {energy:.3f} kcal/mol")
    return trace_result.dot_bracket, float(energy)


# --------------------------
# Command-Line Interface
# --------------------------
def main(argv=None) -> int:
    """
    Parses command-line arguments and orchestrates the RNA folding prediction.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Predict RNA structure (dot-bracket) and ΔG.")
    parser.add_argument("sequence", help="RNA sequence (A,C,G,U; T will be converted to U)")
    parser.add_argument("--engine", choices=["auto", "zucker", "eddy_rivas"], default="auto",
                        help="Which predictor to use (default: auto).")
    parser.add_argument("--yaml", default=None,
                        help="Path to parameter YAML (defaults to package data).")
    parser.add_argument("--tempC", type=float, default=37.0,
                        help="Temperature in °C (default: 37.0).")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of human-readable text.")

    # Logging arguments
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (-v=INFO, -vv=DEBUG)")
    parser.add_argument("--log-file", default=None,
                        help="Path to log file (default: var/log/predict_rna_TIMESTAMP.log if verbose)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress all output except final result")

    # Eddy-Rivas tuning arguments
    parser.add_argument("--coax", action="store_true", default=True,
                        help="Enable coaxial stacking terms in Eddy Rivas (default: on).")
    parser.add_argument("--overlap", action="store_true", default=True,
                        help="Enable WX overlap path in Eddy Rivas (default: on).")
    parser.add_argument("--is2", action="store_true", default=True,
                        help="Enable energy calculations for Irreducible Surfaces of Order 2 (default: on).")
    parser.add_argument("--join_drift", action="store_true", default=True,
                        help="Enable slight hole shifting at a join point (default: on).")
    parser.add_argument("--strict_compliment_order", action="store_true", default=True,
                        help="Enable strict ordering i < k <= r < l <= j for pseudoknots (default: on).")

    parser.add_argument("--dbg-outer", default=None,
                        help="(optional) Debug filter for WX candidates in the form i,j (e.g. 0,42).")

    cli_args = parser.parse_args(argv)

    # --- Setup ---
    # Configure logging based on --verbose, --quiet, and --log-file flags.
    verbose_level = 0 if cli_args.quiet else cli_args.verbose
    setup_cli_logging(verbose_level, cli_args.log_file)

    logger.info("=" * 60)
    logger.info("RNA Structure Prediction CLI")
    logger.info("=" * 60)

    # Validate and normalize the input sequence.
    try:
        normalized_sequence = validate_and_normalize_seq(cli_args.sequence)
    except ValueError as e:
        logger.error(f"Sequence validation failed: {e}")
        if not cli_args.json:
            print(f"Error: {e}", file=sys.stderr)
        return 2

    # Load the thermodynamic energy model from a YAML file.
    try:
        energy_model = load_energy_model(cli_args.tempC, cli_args.yaml)
    except Exception as e:
        logger.error(f"Failed to load energy model: {e}", exc_info=True)
        if not cli_args.json:
            print(f"Failed to load energy model YAML: {e}", file=sys.stderr)
        return 2

    # --- Engine Selection and Execution ---
    engine_used = cli_args.engine
    logger.info(f"Requested engine: {cli_args.engine}")

    try:
        if cli_args.engine == "zucker":
            dot_bracket, delta_g = predict_zucker_nested(normalized_sequence, energy_model)
        elif cli_args.engine == "eddy_rivas":
            dot_bracket, delta_g = predict_eddy_rivas_non_nested(
                normalized_sequence,
                energy_model,
                enable_coax=cli_args.coax,
                enable_overlap=cli_args.overlap,
                enable_is2=cli_args.is2,
                enable_join_drift=cli_args.join_drift,
                enable_strict_complement_order=cli_args.strict_compliment_order
            )
        else:  # 'auto' mode
            try:
                # First, attempt the full pseudoknot prediction.
                logger.info("Attempting Eddy-Rivas (auto mode)...")
                dot_bracket, delta_g = predict_eddy_rivas_non_nested(
                    normalized_sequence,
                    energy_model,
                    enable_coax=cli_args.coax,
                    enable_overlap=cli_args.overlap,
                    enable_is2=cli_args.is2,
                    enable_join_drift=cli_args.join_drift,
                    enable_strict_complement_order=cli_args.strict_compliment_order
                )
                engine_used = "eddy_rivas"
            except Exception as e:
                # If the pseudoknot engine fails for any reason, fall back to the nested-only engine.
                logger.warning(f"Eddy-Rivas failed, falling back to Zuker: {e}")
                dot_bracket, delta_g = predict_zucker_nested(normalized_sequence, energy_model)
                engine_used = "zucker"
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        if not cli_args.json:
            print(f"Prediction failed: {e}", file=sys.stderr)
        return 1

    logger.info("=" * 60)
    logger.info("Prediction successful")
    logger.info("=" * 60)

    # --- Output ---
    # Print the final result to standard output in the requested format.
    if cli_args.json:
        print(json.dumps({
            "engine": engine_used,
            "sequence": normalized_sequence,
            "dot_bracket": dot_bracket,
            "delta_G_kcal_per_mol": delta_g,
            "length": len(normalized_sequence),
        }, indent=2))
    else:
        print(f"Engine : {engine_used}")
        print(f"Sequence Length : {len(normalized_sequence)}")
        print(f"Sequence : {normalized_sequence}")
        print(f"Dot-Bracket Notation: {dot_bracket}")
        print(f"ΔG (kcal/mol): {delta_g:.2f}")

    return 0


if __name__ == "__main__":
    # Run the main function and exit with its return code.
    raise SystemExit(main())
