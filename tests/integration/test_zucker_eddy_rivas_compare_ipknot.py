"""
Integration tests for comparing the Eddy-Rivas (pseudoknot-aware) folding
engine against IPknot, a tool for predicting RNA secondary structures including
pseudoknots.

Attributes
----------
pytestmark : list
    A list of pytest markers. It marks all tests in this module as 'integration'
    and skips them if the 'ipknot' command-line tool is not found in the system's PATH.
"""
from __future__ import annotations

# --- Standard Library Imports ---
import math
import os
import re
import shutil
import subprocess
import tempfile
from importlib.resources import files as importlib_files
from typing import Dict, List, Optional, Set, Tuple

# --- Third-Party Imports ---
import pytest

# --- Local Application Imports ---
import rna_pk_fold

# Energy model components
from rna_pk_fold.energies import SecondaryStructureEnergyLoader
from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModel

# Zuker (nested) folding components
from rna_pk_fold.folding.zucker import make_fold_state
from rna_pk_fold.folding.zucker.zucker_recurrences import ZuckerFoldingConfig, ZuckerFoldingEngine

# Eddy-Rivas (pseudoknot) folding components
from rna_pk_fold.folding.eddy_rivas import eddy_rivas_recurrences as eddy_rivas_engine
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import init_eddy_rivas_fold_state
from rna_pk_fold.folding.zucker.zucker_traceback import traceback_nested_interval as zucker_traceback_interval
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_traceback import traceback_with_pk as eddy_rivas_traceback

# --- Constants for Multilayer Bracket Parsing ---
# A dictionary mapping opening brackets to their corresponding closing brackets.
BRACKET_PAIRS: Dict[str, str] = {'(': ')', '[': ']', '{': '}', '<': '>'}
# A reverse mapping from closing brackets back to their opening counterparts.
REVERSE_BRACKETS: Dict[str, str] = {v: k for k, v in BRACKET_PAIRS.items()}

# Mark all tests in this file as integration tests and skip if ipknot is not installed.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("ipknot") is None, reason="ipknot not found on PATH"),
]


def run_ipknot(seq: str) -> Tuple[str, float]:
    """
    Calls the IPknot command-line tool to predict a pseudoknotted structure.

    This function writes the input sequence to a temporary FASTA file, runs
    the `ipknot` executable as an external process, and parses its standard
    output to extract the predicted multilayer dot-bracket structure and the
    associated score (treated as an energy).

    Parameters
    ----------
    seq : str
        The RNA sequence to fold.

    Returns
    -------
    Tuple[str, float]
        A tuple containing the predicted dot-bracket string and the score.
    """
    # Find the ipknot executable, allowing for an environment variable override.
    ipknot_executable = os.environ.get("IPKNOT_BIN") or shutil.which("ipknot")
    assert ipknot_executable, "ipknot binary not found (set IPKNOT_BIN or add to PATH)."

    # Create a temporary file to pass the sequence to ipknot in FASTA format.
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as temp_fasta:
        temp_fasta.write(">query_sequence\n")
        temp_fasta.write(seq + "\n")
        fasta_path = temp_fasta.name

    try:
        # Run the ipknot command with the '-E' flag to output the score.
        process = subprocess.run(
            [ipknot_executable, "-E", fasta_path],
            text=True,
            capture_output=True,
            check=True,
        )
    finally:
        # Ensure the temporary file is deleted after the process runs.
        try:
            os.unlink(fasta_path)
        except OSError:
            pass

    # --- Parse IPknot Output ---
    output = process.stdout.strip()
    assert output, f"IPknot produced no output.\nSTDERR:\n{process.stderr}"
    lines = [line.strip() for line in output.splitlines() if line.strip()]

    # Regular expressions to find the dot-bracket string and the energy score.
    dot_bracket_regex = re.compile(r"^[.\(\)\[\]\{\}\<\>]+$")
    number_regex = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")

    dot_bracket: Optional[str] = None
    energy: Optional[float] = None

    # Search for the structure and energy from the last lines of the output.
    for line in reversed(lines):
        if energy is None and number_regex.fullmatch(line):
            energy = float(line)
            continue
        if dot_bracket is None and dot_bracket_regex.fullmatch(line):
            dot_bracket = line
            continue
        if dot_bracket is not None and energy is not None:
            break

    # Assert that a valid dot-bracket string was found.
    assert dot_bracket is not None, (
        "Could not find a multilayer dot-bracket string in IPknot output.\n"
        f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
    )

    # As a fallback, search for any floating-point number if a standalone one wasn't found.
    if energy is None:
        for line in reversed(lines):
            match = re.search(r"([-+]?[\d.]+(?:[eE][-+]?\d+)?)", line)
            if match:
                try:
                    energy = float(match.group(1))
                    break
                except ValueError:
                    pass

    # Assert that an energy/score was found.
    assert energy is not None, (
        "Could not find an energy score in IPknot output (expected with -E flag).\n"
        f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
    )
    return dot_bracket, energy


# ------------------------
# Pytest Fixtures
# ------------------------
@pytest.fixture(scope="module")
def energy_model() -> SecondaryStructureEnergyModel:
    """A pytest fixture that loads the Turner 2004 energy parameters."""
    # Locate the default parameter file within the installed package.
    yaml_path = importlib_files(rna_pk_fold) / "data" / "turner2004_eddyrivas1999_min.yaml"
    assert yaml_path.exists(), "No parameter YAML found."
    # Load the parameters and construct the energy model at 37 °C.
    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=str(yaml_path))
    return SecondaryStructureEnergyModel(params=params, temp_k=310.15)


@pytest.fixture(scope="module")
def engines_and_costs(energy_model: SecondaryStructureEnergyModel) -> Tuple[
    ZuckerFoldingEngine, eddy_rivas_engine.EddyRivasFoldingEngine, eddy_rivas_engine.PseudoknotEnergies]:
    """
    A pytest fixture that provides initialized Zuker and Eddy-Rivas folding engines.
    """
    # 1. Initialize the Zuker (nested-only) engine.
    zucker_engine = ZuckerFoldingEngine(
        energy_model=energy_model,
        config=ZuckerFoldingConfig(),
    )

    # 2. Extract pseudoknot parameters from the loaded energy model.
    pk_params = energy_model.params.PSEUDOKNOT
    if pk_params is None:
        raise ValueError("No pseudoknot parameters found in the YAML file!")

    # 3. Configure and initialize the Eddy-Rivas (pseudoknot) engine.
    # These settings are chosen to be broadly permissive for testing.
    er_config = eddy_rivas_engine.EddyRivasFoldingConfig(
        enable_coax=True,
        enable_coax_variants=True,
        enable_coax_mismatch=True,
        enable_wx_overlap=True,
        strict_complement_order=False,
        enable_join_drift=False,  # Experimental feature
        enable_is2=False,  # Experimental feature
        pk_penalty_gw=pk_params.pk_penalty_gw,
        costs=pk_params,
    )
    er_engine = eddy_rivas_engine.EddyRivasFoldingEngine(er_config)

    return zucker_engine, er_engine, pk_params


# ------------------------
# Test Helper Functions (Duplicated from other test file for standalone execution)
# ------------------------
def dotbracket_to_pairs_multilayer(dot_bracket: str) -> Set[Tuple[int, int]]:
    """Converts a layered dot-bracket string into a set of base pair tuples."""
    bracket_stacks: Dict[str, List[int]] = {opener: [] for opener in BRACKET_PAIRS}
    pairs: Set[Tuple[int, int]] = set()
    for index, char in enumerate(dot_bracket):
        if char in BRACKET_PAIRS:
            bracket_stacks[char].append(index)
        elif char in REVERSE_BRACKETS:
            opener = REVERSE_BRACKETS[char]
            if bracket_stacks[opener]:
                i = bracket_stacks[opener].pop()
                pairs.add((i, index))
    return pairs


def bp_distance_multilayer(dot_bracket_1: str, dot_bracket_2: str) -> int:
    """Calculates the base-pair distance between two multilayer dot-bracket strings."""
    pairs_1 = dotbracket_to_pairs_multilayer(dot_bracket_1)
    pairs_2 = dotbracket_to_pairs_multilayer(dot_bracket_2)
    return len(pairs_1 ^ pairs_2)


def project_parentheses(dot_bracket: str) -> str:
    """Converts a multilayer dot-bracket to a single-layer (parentheses-only) string."""
    return ''.join(char if char in '().' else '.' for char in dot_bracket)


# ------------------------
# Test Sequences
# ------------------------
# A known pseudoknotted sequence to test against.
TEST_SEQUENCES = [
    "UUCUUUUUUAGUGGCAGUAAGCCUGGGAAUGGGGGCGACCCAGGCGUAUGAACAUAGUGUAACGCUCCCC"
]


# ------------------------
# The Main Test Function
# ------------------------
@pytest.mark.parametrize("seq", TEST_SEQUENCES)
def test_full_vs_ipknot_shape_and_energy(seq: str, engines_and_costs: tuple):
    """
    Compares the Eddy-Rivas engine's output against IPknot.

    This test runs our full pseudoknot prediction pipeline and compares the
    resulting structure against the prediction from IPknot. It checks that the
    multilayer base-pair distance is within an acceptable tolerance.

    Parameters
    ----------
    seq : str
        The RNA sequence to test.
    engines_and_costs : tuple
        The pytest fixture providing the initialized folding engines.
    """
    zucker_engine, er_engine, _ = engines_and_costs

    # --- 1. Run Zuker (nested) fold to get the baseline matrices ---
    zucker_state = make_fold_state(len(seq))
    zucker_engine.fill_all_matrices(seq, zucker_state)

    # --- 2. Run Eddy-Rivas (pseudoknot) fold ---
    eddy_rivas_state = init_eddy_rivas_fold_state(len(seq))
    er_engine.fill_with_costs(seq, zucker_state, eddy_rivas_state)
    our_full_energy = eddy_rivas_state.wx_matrix.get(0, len(seq) - 1)
    assert math.isfinite(our_full_energy), "Eddy-Rivas prediction resulted in a non-finite energy."

    # --- 3. Run IPknot to get the reference structure ---
    ipknot_dot_bracket, ipknot_energy = run_ipknot(seq)

    # --- 4. Perform the full multilayer traceback for our engine ---
    # Skip this test if the necessary traceback functions could not be imported.
    if eddy_rivas_traceback is None or zucker_traceback_interval is None:
        pytest.skip("Full PK traceback is unavailable (required imports failed).")

    full_trace_result = eddy_rivas_traceback(
        seq,
        nested_state=zucker_state,
        eddy_rivas_fold_state=eddy_rivas_state,
        trace_nested_interval=zucker_traceback_interval,
    )
    our_full_dot_bracket = full_trace_result.dot_bracket

    # --- 5. Compare the structures ---
    # A tolerance is used to allow for minor differences between the energy models.
    tolerance_bp_multi = 4
    distance_multi = bp_distance_multilayer(our_full_dot_bracket, ipknot_dot_bracket)
    assert distance_multi <= tolerance_bp_multi, (
        f"Pseudoknot multilayer shape mismatch for sequence: {seq}\n"
        f"Our prediction: {our_full_dot_bracket}\n"
        f"IPknot        : {ipknot_dot_bracket}\n"
        f"Multilayer base-pair distance ({distance_multi}) exceeds tolerance ({tolerance_bp_multi}).\n"
        f"Our MFE: {our_full_energy:.2f} kcal/mol, IPknot score: {ipknot_energy:.2f}"
    )
