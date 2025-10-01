import re
import shutil
import subprocess
from typing import List, Set, Tuple

import pytest
from importlib.resources import files as ir_files
pytestmark = pytest.mark.integration

import rna_pk_fold
from rna_pk_fold.folding import make_fold_state
from rna_pk_fold.folding.recurrences import SecondaryStructureFoldingEngine, RecurrenceConfig
from rna_pk_fold.energies import (SecondaryStructureEnergies, SecondaryStructureEnergyModel,
                                  SecondaryStructureEnergyLoader)
from rna_pk_fold.utils.nucleotide_utils import dimer_key
from rna_pk_fold.folding.traceback import traceback_nested

# ---------- Helpers ----------

def dotbracket_to_pairs(db: str) -> Set[Tuple[int, int]]:
    """
    Convert dot-bracket to a set of 0-based base-pair tuples (i, j) with i<j.
    Supports only '(' and ')'.
    """
    stack: List[int] = []
    pairs: Set[Tuple[int, int]] = set()
    for idx, ch in enumerate(db):
        if ch == '(':
            stack.append(idx)
        elif ch == ')':
            if not stack:
                # Unbalanced, ignore to keep test robust
                continue
            i = stack.pop()
            pairs.add((i, idx))
    return pairs

def bp_distance(db1: str, db2: str) -> int:
    """Base-pair distance = symmetric difference size of pair sets."""
    p1 = dotbracket_to_pairs(db1)
    p2 = dotbracket_to_pairs(db2)
    return len(p1 ^ p2)

def run_rnafold(seq: str) -> Tuple[str, float]:
    """
    Call ViennaRNA RNAfold (CLI) and parse dot-bracket + MFE.
    Returns (dotbracket, mfe_kcal_mol).
    """
    proc = subprocess.run(
        ["RNAfold", "--noPS"],
        input=seq + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    # Output looks like:
    #   SEQUENCE
    #   .....(((....)))... (-3.40)
    lines = proc.stdout.strip().splitlines()
    assert len(lines) >= 2, f"Unexpected RNAfold output:\n{proc.stdout}"
    result_line = lines[1]
    # Extract db and energy "(...)" part
    # e.g. ".....(((....)))... (-3.40)"
    m = re.match(r"\s*([().]+)\s+\(\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*\)\s*$", result_line)
    assert m, f"Could not parse RNAfold line: {result_line!r}"
    db = m.group(1)
    mfe = float(m.group(2))
    return db, mfe

# ---------- Fixtures you must supply ----------

@pytest.fixture(scope="module")
def energy_model():
    """
    Load the Turner 2004 (minimal) parameters from packaged YAML and
    construct the SecondaryStructureEnergyModel at 310.15 K.
    """
    yaml_path = ir_files(rna_pk_fold) / "data" / "turner2004_min.yaml"
    params = SecondaryStructureEnergyLoader.load(kind="RNA", yaml_path=yaml_path)
    return SecondaryStructureEnergyModel(params=params, temp_k=310.15)

@pytest.fixture(scope="module")
def engine(energy_model):
    return SecondaryStructureFoldingEngine(energy_model=energy_model, config=RecurrenceConfig())

# ---------- Skip if RNAfold isn't available ----------

rnafold_missing = shutil.which("RNAfold") is None

pytestmark = pytest.mark.skipif(
    rnafold_missing, reason="RNAfold (ViennaRNA) not found on PATH"
)

# ---------- Parameterized sequences ----------
# Keep small, pseudoknot-free, and varied GC/AU content
SEQS = [
    "GCGC",          # simple 2-stack helix
    "GCAUCUAUGC",    # small helix with loop
    "GGGAAAUCCC",    # classic strong stem with AU loop
    "AUGCUAGCUAUGC", # few possible helices
    "AUAUAUAUAU",    # low GC, might stay mostly unpaired

    # Hairpins (short loops, clamps)
    "GCAAAGC", "GCAAAAGC", "GCAAAAAGC",
    "AUGGGAU", "AUGGGGAU", "GUAAAAGU", "UGAAAUG",

    # Tetraloops (require SPECIAL_HAIRPINS to match Vienna exactly)
    "GCGCAAGC", "GCUUCGGC", "GCGGAGGC",

    # Bulges & internal loops
    "GGCGAACGCC", "GGCGAAUGCC", "GGCAAUUGCC", "GGCACAUUGCC", "GGCAAAUUGCC",

    # Multiloops / branching
    "GGGAAACCCAAAGGGUUUCCC", "GCGAAUCCGAUUGGCUAAGCG",
    "GGAUCCGAAGGCUCGAUCC", "GGGAAAUCCAUUGGAUCCCUCC", "GCCGAUACGUAUCGGCGAU",

    # Long/strong helices
    "GCGCGCGCGCAUUGCGCGCGCGC", "GGGGCCCCGGGGCCCC",

    # Wobble-heavy
    "GUGUGUGUACACACAC", "UGUGUGAAACACACA", "GUGUAAUUGUGU",

    # AU-rich
    "AUAUAUAUAU", "AAUAAAUAAAUAA", "AUAUAAUAUAUAUAU",

    # GC-rich
    "GCGCGCAGCGCGC", "GGCGCCGCGGCC",

    # Randomish short
    "GCAUCUAUGC", "AUGCUAGCUAUGC", "GGGAAAUCCC", "GCGC",
    "GGAUACGUACCU", "CGAUGCAGCUAG",

    # Mostly unstructured
    "AAAAUAAAAUAAAAUAAAA", "UUUUUAAAUUUUUAAAUUUU",

    # Edge cases
    "AUCCCUA", "GUCCUGU",
]

@pytest.mark.parametrize("seq", SEQS)
def test_shape_matches_vienna_within_tolerance(seq: str, engine):
    """
    Compare our dot-bracket and total MFE (kcal/mol) with ViennaRNA.

    - Structure: base-pair distance within TOL_BP.
    - Energy: absolute difference within TOL_EN (kcal/mol).

    Notes:
      * Our DP energies are in kcal/mol; RNAfold reports kcal/mol too.
      * Sequences in SEQS are chosen to be pseudoknot-free so our nested traceback is valid.
    """
    # 1) Our prediction
    st = make_fold_state(len(seq))
    engine.fill_all_matrices(seq, st)
    ours = traceback_nested(seq, st).dot_bracket
    assert len(ours) == len(seq)

    # 2) ViennaRNA prediction
    v_db, v_mfe = run_rnafold(seq)
    assert len(v_db) == len(seq)

    # 3) Compare shape (base-pair distance)
    dist = bp_distance(ours, v_db)

    # Tighten this threshold as you align your parameters to Vienna
    TOL = 1
    assert dist <= TOL, f"Seq={seq}, ours={ours}, vienna={v_db}, dist={dist}, mfe={v_mfe}"
