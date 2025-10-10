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
from typing import Tuple, Optional

# --- Third-Party Imports ---
from importlib.resources import files as importlib_files

# --- Local Application Imports ---
# Energy model loading and definition
from rna_pk_fold.utils.logging_utils import setup_logger, DEFAULT_LOG_DIR
from rna_pk_fold.energies import SecondaryStructureEnergyLoader
from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModel

# Nested (Zuker) folding components
from rna_pk_fold.folding.zucker import make_fold_state as make_zucker_state
from rna_pk_fold.folding.zucker.zucker_recurrences import ZuckerFoldingConfig, ZuckerFoldingEngine
from rna_pk_fold.folding.zucker.zucker_traceback import traceback_nested as zucker_traceback
from rna_pk_fold.folding.zucker.zucker_traceback import traceback_nested_interval

# Eddy-Rivas (pseudoknot) folding components
from rna_pk_fold.folding.eddy_rivas import eddy_rivas_recurrences
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import init_eddy_rivas_fold_state
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_traceback import traceback_with_pk as eddy_rivas_traceback

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


# --------------------------
# Helpers
# --------------------------
def validate_and_normalize_seq(raw_sequence: str) -> str:
    """
    Validates and normalizes an RNA sequence.

    This function strips whitespace, converts the sequence to uppercase,
    replaces 'T' with 'U', and checks for any invalid characters.

    Parameters
    ----------
    raw_sequence : str
        The input RNA sequence string.

    Returns
    -------
    str
        The validated and normalized RNA sequence.

    Raises
    ------
    ValueError
        If the sequence is empty or contains characters other than A, C, G, U, T.
    """
    logger.debug(f"Validating sequence: {raw_sequence[:50]}{'...' if len(raw_sequence) > 50 else ''}")
    # Normalize the sequence: strip whitespace, convert to uppercase, and replace T with U.
    normalized_sequence = raw_sequence.strip().upper().replace("T", "U")

    # Check if the sequence is empty after normalization.
    if not normalized_sequence:
        logger.error("Sequence is empty")
        raise ValueError("Sequence is empty.")

    # Check for any characters that are not in the allowed set (A, C, G, U).
    allowed_chars = set("ACGU")
    invalid_char_indices = [i for i, char in enumerate(normalized_sequence) if char not in allowed_chars]
    if invalid_char_indices:
        pos = invalid_char_indices[0]
        invalid_char = normalized_sequence[pos]
        error_message = f"Invalid character at position {pos} ('{invalid_char}'). Only A,C,G,U (or T) are allowed."
        logger.error(f"Invalid character at position {pos}: '{invalid_char}'")
        raise ValueError(error_message)

    logger.info(f"Sequence validated: length={len(normalized_sequence)}")
    return normalized_sequence


def load_energy_model(yaml_path: Optional[str], temp_c: float) -> SecondaryStructureEnergyModel:
    """
    Loads the RNA thermodynamic parameters and creates an energy model.

    If no YAML file path is provided, it loads the default parameters bundled
    with the package.

    Parameters
    ----------
    yaml_path : Optional[str]
        The file path to the energy parameter YAML file.
    temp_c : float
        The temperature in Celsius for the energy calculations.

    Returns
    -------
    SecondaryStructureEnergyModel
        An initialized energy model object ready for use by the folding engines.
    """
    # Use the default bundled parameter file if no path is provided.
    if yaml_path is None:
        yaml_path = str(importlib_files("rna_pk_fold") / "data" / "turner2004_eddyrivas1999_min.yaml")

    logger.info(f"Loading energy model from: {yaml_path}")
    temp_k = 273.15 + temp_c
    logger.info(f"Temperature: {temp_c}°C ({temp_k:.2f}K)")

    # Load the raw parameters from the YAML file.
    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=yaml_path)

    # Create the energy model instance with the loaded parameters and specified temperature.
    model = SecondaryStructureEnergyModel(params=params, temp_k=temp_k)

    logger.debug("Energy model loaded successfully")

    return model


def build_eddy_rivas_costs(energy_model: SecondaryStructureEnergyModel,
                           q_ss_override: Optional[float],
                           gw_override: Optional[float]) -> eddy_rivas_recurrences.PseudoknotEnergies:
    """
    Constructs the pseudoknot energy parameter object, applying CLI overrides.

    This function takes the base pseudoknot parameters from the loaded energy
    model and replaces specific values with any overrides provided via
    command-line arguments.

    Parameters
    ----------
    energy_model : SecondaryStructureEnergyModel
        The fully loaded energy model.
    q_ss_override : Optional[float]
        An optional override for the single-stranded base penalty (`q_ss`).
    gw_override : Optional[float]
        An optional override for the pseudoknot initiation penalty (`Gw`).

    Returns
    -------
    eddy_rivas_recurrences.PseudoknotEnergies
        A data object containing the final set of pseudoknot energy parameters.

    Raises
    ------
    ValueError
        If the loaded energy model does not contain pseudoknot parameters.
    """
    # Extract the base pseudoknot parameters from the energy model.
    base_pk_params = energy_model.params.PSEUDOKNOT
    if base_pk_params is None:
        raise ValueError("No pseudoknot parameters found in the provided YAML file!")

    # Collect any command-line overrides into a dictionary.
    cli_overrides = {}
    if gw_override is not None:
        cli_overrides["pk_penalty_gw"] = gw_override
    if q_ss_override is not None:
        cli_overrides["q_ss"] = q_ss_override

    # If there are no overrides, return the original parameters.
    if not cli_overrides:
        return base_pk_params

    # Otherwise, create a new dataclass instance with the overrides applied.
    from dataclasses import replace
    return replace(base_pk_params, **cli_overrides)


def predict_nested(seq: str, energy_model: SecondaryStructureEnergyModel) -> Tuple[str, float]:
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
    energy = zucker_state.w_matrix.get(0, len(seq) - 1)

    elapsed = time.perf_counter() - start_time
    logger.info(f"Prediction completed in {elapsed:.2f}s")
    logger.info(f"Energy: {energy:.3f} kcal/mol")

    return trace_result.dot_bracket, float(energy)


def predict_eddy_rivas(
    seq: str,
    energy_model: SecondaryStructureEnergyModel,
    pk_penalty_gw: Optional[float],
    enable_coax: bool,
    enable_overlap: bool,
    min_hole_width: int,
    max_hole_width: int,
    q_ss: Optional[float]
) -> Tuple[str, float]:
    """
    Runs the Eddy-Rivas (pseudoknot-aware) folding algorithm.

    This involves a two-phase process: first, a complete nested fold is
    performed using the Zuker algorithm, and then the Eddy-Rivas algorithm
    builds upon those results to find the optimal structure including pseudoknots.

    Parameters
    ----------
    seq : str
        The RNA sequence to fold.
    energy_model : SecondaryStructureEnergyModel
        The initialized energy model.
    All other parameters are tuning options for the Eddy-Rivas algorithm.

    Returns
    -------
    Tuple[str, float]
        A tuple containing the predicted multilayer dot-bracket structure and
        its minimum free energy in kcal/mol.
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
    # Build the specific cost model for pseudoknots, applying any CLI overrides.
    er_costs = build_eddy_rivas_costs(energy_model, q_ss_override=q_ss, gw_override=pk_penalty_gw)

    # Log the final parameters being used for the Eddy-Rivas run.
    logger.info(f"PK penalty Gw: {er_costs.pk_penalty_gw}")
    logger.info(f"Coaxial stacking: {enable_coax}")
    logger.info(f"WX overlap: {enable_overlap}")
    logger.info(f"Hole width: [{min_hole_width}, {max_hole_width if max_hole_width > 0 else '∞'}]")

    # Configure the Eddy-Rivas engine.
    er_config = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=enable_coax,
        enable_coax_variants=enable_coax,
        enable_coax_mismatch=enable_coax,
        enable_wx_overlap=enable_overlap,
        enable_is2=False, # Experimental feature, disabled by default.
        enable_join_drift=False, # Experimental feature, disabled by default.
        min_hole_width=min_hole_width,
        max_hole_width=max_hole_width,
        pk_penalty_gw=er_costs.pk_penalty_gw,
        costs=er_costs,
        verbose=logger.isEnabledFor(logging.INFO),
    )
    er_engine = eddy_rivas_recurrences.EddyRivasFoldingEngine(er_config)

    # Initialize the state object for the Eddy-Rivas matrices.
    eddy_rivas_state = init_eddy_rivas_fold_state(len(seq))

    # Run the main DP algorithm, seeding it with the results from the Zuker phase.
    er_engine.fill_with_costs(seq, zucker_state, eddy_rivas_state)

    # Get the final energy for the entire sequence.
    energy = eddy_rivas_state.wx_matrix.get(0, len(seq) - 1)

    # If the final energy is infinite, the algorithm failed; fall back to the nested result.
    if not math.isfinite(energy):
        logger.warning("Eddy-Rivas returned infinite energy, falling back to nested result.")
        trace_result = zucker_traceback(seq, zucker_state)
        nested_energy = zucker_state.w_matrix.get(0, len(seq) - 1)
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
    parser.add_argument("--pk-gw", type=float, default=None,
                      help="Override pseudoknot penalty Gw (kcal/mol).")
    parser.add_argument("--coax", action="store_true",
                      help="Enable coaxial stacking terms in ER (default: off).")
    parser.add_argument("--overlap", action="store_true",
                      help="Enable WX overlap path in ER (default: off).")
    parser.add_argument("--min-hole-width", type=int, default=0,
                      help="Minimum hole width (k,l) seam interior (default: 0).")
    parser.add_argument("--max-hole-width", type=int, default=0,
                      help="Maximum hole width (0 means no cap).")
    parser.add_argument("--q-ss", type=float, default=None,
                      help="Backbone per-SS penalty used by ER recurrences (default: from YAML).")

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
        energy_model = load_energy_model(cli_args.yaml, cli_args.tempC)
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
            dot_bracket, delta_g = predict_nested(normalized_sequence, energy_model)
        elif cli_args.engine == "eddy_rivas":
            dot_bracket, delta_g = predict_eddy_rivas(
                normalized_sequence,
                energy_model,
                pk_penalty_gw=cli_args.pk_gw,
                enable_coax=cli_args.coax,
                enable_overlap=cli_args.overlap,
                min_hole_width=cli_args.min_hole_width,
                max_hole_width=cli_args.max_hole_width,
                q_ss=cli_args.q_ss,
            )
        else:  # 'auto' mode
            try:
                # First, attempt the full pseudoknot prediction.
                logger.info("Attempting Eddy-Rivas (auto mode)...")
                dot_bracket, delta_g = predict_eddy_rivas(
                    normalized_sequence,
                    energy_model,
                    pk_penalty_gw=cli_args.pk_gw,
                    enable_coax=cli_args.coax,
                    enable_overlap=cli_args.overlap,
                    min_hole_width=cli_args.min_hole_width,
                    max_hole_width=cli_args.max_hole_width,
                    q_ss=cli_args.q_ss,
                )
                engine_used = "eddy_rivas"
            except Exception as e:
                # If the pseudoknot engine fails for any reason, fall back to the nested-only engine.
                logger.warning(f"Eddy-Rivas failed, falling back to Zuker: {e}")
                dot_bracket, delta_g = predict_nested(normalized_sequence, energy_model)
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
