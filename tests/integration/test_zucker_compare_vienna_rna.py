"""
Integration tests for comparing the nested RNA folding engine against ViennaRNA's RNAfold.

This test suite validates the correctness of the Zuker (nested-only) folding
algorithm by comparing its predictions for structure and minimum free energy (MFE)
against the results from RNAfold, a widely-used benchmark tool.

Attributes
----------
pytestmark : list
    A list of pytest markers. It marks all tests in this module as 'integration'
    and skips them if the 'RNAfold' command-line tool is not found in the system's PATH.
"""
from __future__ import annotations

# --- Standard Library Imports ---
import math
import re
import shutil
import subprocess
from importlib.resources import files as importlib_files
from typing import Dict, List, Set, Tuple

# --- Third-Party Imports ---
import pytest

# --- Local Application Imports ---
import rna_pk_fold
from rna_pk_fold.energies import SecondaryStructureEnergyLoader
from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModel
from rna_pk_fold.folding.common_traceback import dotbracket_to_pairs
from rna_pk_fold.folding.zucker import make_fold_state
from rna_pk_fold.folding.zucker.zucker_recurrences import (ZuckerFoldingConfig,
                                                           ZuckerFoldingEngine)
from rna_pk_fold.folding.zucker.zucker_traceback import traceback_nested

# Mark all tests in this file as integration tests.
# Skip these tests if the 'RNAfold' executable is not found in the system's PATH.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("RNAfold") is None, reason="RNAfold (ViennaRNA) not found on PATH"),
]

# Define constants for multilayer bracket parsing.
BRACKET_PAIRS: Dict[str, str] = {'(': ')', '[': ']', '{': '}', '<': '>'}
REVERSE_BRACKETS: Dict[str, str] = {v: k for k, v in BRACKET_PAIRS.items()}


# --------------------------
# Test Helper Functions
# --------------------------

def dotbracket_to_pairs_multilayer(dot_bracket: str) -> Set[Tuple[int, int]]:
    """
    Converts a layered dot-bracket string into a set of base pair tuples.

    This function parses a dot-bracket string that may contain multiple types
    of brackets (e.g., `()`, `[]`, `{}`) and converts them into a canonical
    set of `(i, j)` tuples where `i < j`. It handles each bracket type
    independently, making it suitable for representing pseudoknots.

    Parameters
    ----------
    dot_bracket : str
        The dot-bracket string to parse.

    Returns
    -------
    Set[Tuple[int, int]]
        A set of tuples, where each tuple represents a base pair.
    """
    # Create a dictionary of stacks, one for each type of opening bracket.
    bracket_stacks: Dict[str, List[int]] = {opener: [] for opener in BRACKET_PAIRS}
    # Use a set to store the output pairs, automatically handling duplicates.
    pairs: Set[Tuple[int, int]] = set()

    # Iterate through each character and its index in the dot-bracket string.
    for index, char in enumerate(dot_bracket):
        # If the character is an opening bracket, push its index onto the corresponding stack.
        if char in BRACKET_PAIRS:
            bracket_stacks[char].append(index)
        # If the character is a closing bracket...
        elif char in REVERSE_BRACKETS:
            # ...find its corresponding opening bracket type.
            opener = REVERSE_BRACKETS[char]
            # If there is a matching opener on the stack, pop its index and form a pair.
            if bracket_stacks[opener]:
                i = bracket_stacks[opener].pop()
                pairs.add((i, index))

    # Any leftover, unmatched opening brackets are ignored by design.
    return pairs


def bp_distance(dot_bracket_1: str, dot_bracket_2: str) -> int:
    """
    Calculates the base-pair distance between two single-layer dot-bracket strings.

    The distance is defined as the size of the symmetric difference between the
    sets of base pairs represented by the two structures. It counts the number
    of pairs that are in one structure but not the other.

    Parameters
    ----------
    dot_bracket_1 : str
        The first dot-bracket string (using parentheses only).
    dot_bracket_2 : str
        The second dot-bracket string (using parentheses only).

    Returns
    -------
    int
        The number of base pairs that differ between the two structures.
    """
    # Convert each dot-bracket string to a set of pairs.
    pairs_1 = dotbracket_to_pairs(dot_bracket_1)
    pairs_2 = dotbracket_to_pairs(dot_bracket_2)
    # The distance is the number of elements in the symmetric difference of the two sets.
    return len(pairs_1 ^ pairs_2)


def bp_distance_multilayer(dot_bracket_1: str, dot_bracket_2: str) -> int:
    """
    Calculates the base-pair distance between two multilayer dot-bracket strings.

    This function is similar to `bp_distance` but correctly parses all supported
    bracket types (`()`, `[]`, `{}`, `<>`).

    Parameters
    ----------
    dot_bracket_1 : str
        The first multilayer dot-bracket string.
    dot_bracket_2 : str
        The second multilayer dot-bracket string.

    Returns
    -------
    int
        The number of base pairs that differ between the two structures.
    """
    # Convert each multilayer dot-bracket string to a set of pairs.
    pairs_1 = dotbracket_to_pairs_multilayer(dot_bracket_1)
    pairs_2 = dotbracket_to_pairs_multilayer(dot_bracket_2)
    # Calculate the size of the symmetric difference.
    return len(pairs_1 ^ pairs_2)


def project_parentheses(dot_bracket: str) -> str:
    """
    Converts a multilayer dot-bracket string to a single-layer (parentheses-only) string.

    This is used for comparing a potentially multilayer structure from our engine
    with the strictly nested output of ViennaRNA's RNAfold. It replaces all
    non-parenthesis brackets with dots.

    Parameters
    ----------
    dot_bracket : str
        The multilayer dot-bracket string.

    Returns
    -------
    str
        A new string where only `(`, `)`, and `.` characters are preserved.
    """
    return ''.join(char if char in '().' else '.' for char in dot_bracket)


def run_rnafold(seq: str) -> Tuple[str, float]:
    """
    Calls the ViennaRNA RNAfold command-line tool to predict a structure.

    This function executes `RNAfold` as an external process, captures its
    standard output, and parses it to extract the predicted dot-bracket
    structure and the minimum free energy (MFE).

    Parameters
    ----------
    seq : str
        The RNA sequence to fold.

    Returns
    -------
    Tuple[str, float]
        A tuple containing the predicted dot-bracket string and the MFE in kcal/mol.
    """
    # Run the RNAfold command with the '--noPS' flag to suppress PostScript file generation.
    process = subprocess.run(
        ["RNAfold", "--noPS"],
        input=seq + "\n",  # RNAfold expects the sequence followed by a newline.
        text=True,
        capture_output=True,
        check=True,
    )
    # The output format is two lines: the sequence, then the structure with MFE.
    lines = process.stdout.strip().splitlines()
    assert len(lines) >= 2, f"Unexpected RNAfold output:\n{process.stdout}"
    result_line = lines[1]

    # Use a regular expression to parse the dot-bracket string and the MFE from the output line.
    match = re.match(r"\s*([().]+)\s+\(\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*\)\s*$", result_line)
    assert match, f"Could not parse RNAfold line: {result_line!r}"

    # Extract the matched groups.
    dot_bracket = match.group(1)
    mfe = float(match.group(2))
    return dot_bracket, mfe


# --------------------------
# Pytest Fixtures
# --------------------------
@pytest.fixture(scope="module")
def energy_model() -> SecondaryStructureEnergyModel:
    """
    A pytest fixture that loads the Turner 2004 energy parameters.

    This fixture is scoped to the module, so the energy model is loaded only
    once per test file run, improving performance.

    Returns
    -------
    SecondaryStructureEnergyModel
        An initialized energy model object at 310.15 K (37 °C).
    """
    # Locate the default parameter file within the installed package.
    yaml_path = importlib_files(rna_pk_fold) / "data" / "turner2004_eddyrivas1999_min.yaml"
    assert yaml_path.exists(), "No parameter YAML found in rna_pk_fold/data."
    # Load the parameters and construct the energy model.
    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=str(yaml_path))
    return SecondaryStructureEnergyModel(params=params, temp_k=310.15)


@pytest.fixture(scope="module")
def nested_engine(energy_model: SecondaryStructureEnergyModel) -> ZuckerFoldingEngine:
    """
    A pytest fixture that provides an initialized Zuker (nested-only) folding engine.

    Parameters
    ----------
    energy_model : SecondaryStructureEnergyModel
        The energy model fixture.

    Returns
    -------
    ZuckerFoldingEngine
        An instance of the nested folding engine.
    """
    return ZuckerFoldingEngine(
        energy_model=energy_model,
        config=ZuckerFoldingConfig()
    )


# A comprehensive list of pseudoknot-free test sequences with varied properties.
TEST_SEQUENCES = [
    "GCGC",
    "GCAUCUAUGC",
    "GGGAAAUCCC",
    "AUGCUAGCUAUGC",
    "AUAUAUAUAU",
    "GCAAAGC", "GCAAAAGC", "GCAAAAAGC",
    "AUGGGAU", "AUGGGGAU", "GUAAAAGU", "UGAAAUG",
    "GCGCAAGC", "GCUUCGGC", "GCGGAGGC",
    "GGCGAACGCC", "GGCGAAUGCC", "GGCAAUUGCC", "GGCACAUUGCC", "GGCAAAUUGCC",
    "GGGAAACCCAAAGGGUUUCCC", "GCGAAUCCGAUUGGCUAAGCG",
    "GGAUCCGAAGGCUCGAUCC", "GGGAAAUCCAUUGGAUCCCUCC", "GCCGAUACGUAUCGGCGAU",
    "GCGCGCGCGCAUUGCGCGCGCGC", "GGGGCCCCGGGGCCCC",
    "GUGUGUGUACACACAC", "UGUGUGAAACACACA", "GUGUAAUUGUGU",
    "AUAUAUAUAU", "AAUAAAUAAAUAA", "AUAUAAUAUAUAUAU",
    "GCGCGCAGCGCGC", "GGCGCCGCGGCC",
    "GCAUCUAUGC", "AUGCUAGCUAUGC", "GGGAAAUCCC", "GCGC",
    "GGAUACGUACCU", "CGAUGCAGCUAG",
    "AAAAUAAAAUAAAAUAAAA", "UUUUUAAAUUUUUAAAUUUU",
    "AUCCCUA", "GUCCUGU",
]


@pytest.mark.parametrize("seq", TEST_SEQUENCES)
def test_nested_vs_vienna_shape_and_energy(seq: str, nested_engine: ZuckerFoldingEngine):
    """
    Compares the nested folding engine's output against ViennaRNA's RNAfold.

    This test runs both our implementation and RNAfold on a given sequence and
    asserts that:
    1. The predicted secondary structures are similar (base-pair distance is within a tolerance).
    2. The calculated minimum free energies are close (absolute difference is within a tolerance).

    Parameters
    ----------
    seq : str
        The RNA sequence to test, provided by `pytest.mark.parametrize`.
    nested_engine : ZuckerFoldingEngine
        The initialized nested folding engine, provided by the pytest fixture.
    """
    # --- 1. Run our nested prediction engine ---
    state = make_fold_state(len(seq))
    nested_engine.fill_all_matrices(seq, state)
    our_dot_bracket = traceback_nested(seq, state).dot_bracket
    our_mfe = state.w_matrix.get(0, len(seq) - 1)
    assert math.isfinite(our_mfe), "Predicted MFE should be a finite number."
    assert len(our_dot_bracket) == len(seq), "Dot-bracket length must match sequence length."

    # --- 2. Run ViennaRNA's RNAfold as the ground truth ---
    vienna_dot_bracket, vienna_mfe = run_rnafold(seq)
    assert len(vienna_dot_bracket) == len(seq), "ViennaRNA dot-bracket length must match sequence length."

    # --- 3. Compare the results ---
    # Compare the structures using base-pair distance. A small tolerance allows for minor discrepancies.
    tolerance_bp = 1
    distance = bp_distance(our_dot_bracket, vienna_dot_bracket)
    assert distance <= tolerance_bp, (
        f"Structure mismatch for sequence: {seq}\n"
        f"Our prediction  : {our_dot_bracket}\n"
        f"ViennaRNA       : {vienna_dot_bracket}\n"
        f"Base-pair distance ({distance}) exceeds tolerance ({tolerance_bp}).\n"
        f"Our MFE: {our_mfe:.2f} kcal/mol, Vienna MFE: {vienna_mfe:.2f} kcal/mol"
    )

    # Compare the minimum free energies. A small tolerance accounts for minor differences in parameter sets.
    tolerance_energy = 2.0
    energy_difference = abs(our_mfe - vienna_mfe)
    assert energy_difference <= tolerance_energy, (
        f"Energy mismatch for sequence: {seq}\n"
        f"Our MFE       : {our_mfe:.2f} kcal/mol\n"
        f"ViennaRNA MFE : {vienna_mfe:.2f} kcal/mol\n"
        f"Difference ({energy_difference:.2f}) exceeds tolerance ({tolerance_energy:.2f} kcal/mol).\n"
        f"Our structure: {our_dot_bracket}, Vienna structure: {vienna_dot_bracket}"
    )
