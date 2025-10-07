import re
import shutil
import subprocess
import math
from typing import List, Set, Tuple, Dict

import pytest
from importlib.resources import files as ir_files

import rna_pk_fold
from rna_pk_fold.folding.zucker import make_fold_state
from rna_pk_fold.folding.zucker.zucker_recurrences import ZuckerFoldingConfig, ZuckerFoldingEngine
from rna_pk_fold.energies import SecondaryStructureEnergyLoader
from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModel
from rna_pk_fold.folding.zucker.zucker_traceback import traceback_nested
from rna_pk_fold.folding.common_traceback import dotbracket_to_pairs

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("RNAfold") is None, reason="RNAfold (ViennaRNA) not found on PATH"),
]

BRACKET_PAIRS: Dict[str, str] = {'(': ')', '[': ']', '{': '}', '<': '>'}
REV_BRACKET: Dict[str, str] = {v: k for k, v in BRACKET_PAIRS.items()}


# ---------- Helpers ----------
BRACKET_PAIRS: Dict[str, str] = {'(': ')', '[': ']', '{': '}', '<': '>'}
REV_BRACKET: Dict[str, str] = {v: k for k, v in BRACKET_PAIRS.items()}

def dotbracket_to_pairs_multilayer(db: str) -> Set[Tuple[int, int]]:
    """
    Convert layered dot-bracket ((),[],{},<>) into pair tuples (i,j) with i<j.
    Each bracket type is matched independently (supports crossings).
    """
    stacks: Dict[str, List[int]] = {op: [] for op in BRACKET_PAIRS}
    pairs: Set[Tuple[int, int]] = set()
    for idx, ch in enumerate(db):
        if ch in BRACKET_PAIRS:               # opener
            stacks[ch].append(idx)
        elif ch in REV_BRACKET:               # closer
            op = REV_BRACKET[ch]
            if stacks[op]:
                i = stacks[op].pop()
                pairs.add((i, idx))
    # (Any leftover unmatched openers are ignored by design.)
    return pairs

def bp_distance(db1: str, db2: str) -> int:
    """Base-pair distance (parentheses only) = symmetric difference size."""
    p1 = dotbracket_to_pairs(db1)
    p2 = dotbracket_to_pairs(db2)
    return len(p1 ^ p2)

def bp_distance_multilayer(db1: str, db2: str) -> int:
    """Base-pair distance for layered brackets (counts all bracket types)."""
    p1 = dotbracket_to_pairs_multilayer(db1)
    p2 = dotbracket_to_pairs_multilayer(db2)
    return len(p1 ^ p2)

def project_parentheses(db: str) -> str:
    """Replace any non-parenthesis bracket with '.' to compare shapes to Vienna."""
    return ''.join(ch if ch in '().' else '.' for ch in db)

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
    yaml_path = ir_files(rna_pk_fold) / "data" / "turner2004_eddyrivas1999_min.yaml"
    assert yaml_path is not None, "No parameter YAML found in rna_pk_fold/data."
    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=yaml_path)
    return SecondaryStructureEnergyModel(params=params, temp_k=310.15)

@pytest.fixture(scope="module")
def engine_nested(energy_model):
    # Pseudoknots disabled for Vienna comparison
    return ZuckerFoldingEngine(
        energy_model=energy_model,
        config=ZuckerFoldingConfig(enable_pk_h=False)
    )

@pytest.fixture(scope="module")
def engine_pk(energy_model):
    # Pseudoknots enabled (minimal H-type term)
    return ZuckerFoldingEngine(
        energy_model=energy_model,
        config=ZuckerFoldingConfig(enable_pk_h=True, pk_h_penalty=1.0)
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
def test_nested_vs_vienna_shape_and_energy(seq: str, engine_nested):
    """
    Compare our nested-only dot-bracket and total MFE (kcal/mol) with ViennaRNA.

    - Structure: base-pair distance within TOL_BP.
    - Energy: absolute difference within TOL_EN (kcal/mol).
    """
    # Our nested prediction
    st = make_fold_state(len(seq))
    engine_nested.fill_all_matrices(seq, st)
    ours_db = traceback_nested(seq, st).dot_bracket
    ours_mfe = st.w_matrix.get(0, len(seq) - 1)
    assert math.isfinite(ours_mfe)
    assert len(ours_db) == len(seq)

    # ViennaRNA prediction
    v_db, v_mfe = run_rnafold(seq)
    assert len(v_db) == len(seq)

    # Structure comparison (parentheses only)
    TOL_BP = 1
    dist = bp_distance(ours_db, v_db)
    assert dist <= TOL_BP, (
        f"Seq= {seq}\nours= {ours_db}\nvienna= {v_db}\n"
        f"bp_distance= {dist} > {TOL_BP}\nours_mfe= {ours_mfe:.2f}, vienna_mfe= {v_mfe:.2f}"
    )

    # Energy comparison (kcal/mol)
    TOL_EN = 2.0
    delta_g = abs(ours_mfe - v_mfe)
    assert delta_g <= TOL_EN, (
        f"MFE differs by {delta_g:.2f} kcal/mol > {TOL_EN:.2f}\n"
        f"Seq={seq}\nours_db={ours_db}\nvienna_db={v_db}\n"
        f"ours_mfe={ours_mfe:.2f}, vienna_mfe={v_mfe:.2f}"
    )
