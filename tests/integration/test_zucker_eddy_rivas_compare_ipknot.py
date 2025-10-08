import os
import re
import shutil
import subprocess
import math
import tempfile
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
    pytest.mark.skipif(shutil.which("ipknot") is None, reason="ipknot not found on PATH"),
]

BRACKET_PAIRS: Dict[str, str] = {'(': ')', '[': ']', '{': '}', '<': '>'}
REV_BRACKET: Dict[str, str] = {v: k for k, v in BRACKET_PAIRS.items()}

def dotbracket_to_pairs_multilayer(db: str) -> Set[Tuple[int, int]]:
    stacks: Dict[str, List[int]] = {op: [] for op in BRACKET_PAIRS}
    pairs: Set[Tuple[int, int]] = set()
    for idx, ch in enumerate(db):
        if ch in BRACKET_PAIRS:
            stacks[ch].append(idx)
        elif ch in REV_BRACKET:
            op = REV_BRACKET[ch]
            if stacks[op]:
                i = stacks[op].pop()
                pairs.add((i, idx))
    return pairs

def bp_distance(db1: str, db2: str) -> int:
    p1 = dotbracket_to_pairs(db1)   # parentheses only
    p2 = dotbracket_to_pairs(db2)
    return len(p1 ^ p2)

def bp_distance_multilayer(db1: str, db2: str) -> int:
    p1 = dotbracket_to_pairs_multilayer(db1)
    p2 = dotbracket_to_pairs_multilayer(db2)
    return len(p1 ^ p2)

def project_parentheses(db: str) -> str:
    return ''.join(ch if ch in '().' else '.' for ch in db)

def run_ipknot(seq: str) -> Tuple[str, float]:
    """
    Call IPknot with -E to print free energy and parse multilayer dot-bracket + energy.
    Returns (dotbracket, energy_kcal_mol-ish).
    """
    ipknot_bin = os.environ.get("IPKNOT_BIN") or shutil.which("ipknot")
    assert ipknot_bin, "ipknot binary not found (set IPKNOT_BIN or add to PATH)."

    # Write a FASTA file (IPknot expects a file).
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fh:
        fh.write(">q\n")
        fh.write(seq + "\n")
        fasta_path = fh.name

    try:
        proc = subprocess.run(
            [ipknot_bin, "-E", fasta_path],
            text=True,
            capture_output=True,
            check=True,
        )
    finally:
        try:
            os.unlink(fasta_path)
        except OSError:
            pass

    out = proc.stdout.strip()
    assert out, f"IPknot produced no output.\nSTDERR:\n{proc.stderr}"
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]

    # Patterns
    db_re = re.compile(r"^[.\(\)\[\]\{\}\<\>]+$")  # multilayer dot-bracket only
    num_re = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")  # real number
    e_header_re = re.compile(r"\(e\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)")

    db = None
    energy: float | None = None

    # 1) Prefer energy in header like: ">q (e=0)" or similar.
    for ln in lines:
        m = e_header_re.search(ln)
        if m:
            try:
                energy = float(m.group(1))
                break
            except ValueError:
                pass

    # 2) Find the last multilayer dot-bracket line.
    for ln in reversed(lines):
        if db_re.fullmatch(ln):
            db = ln
            break

    assert db is not None, (
        "Could not find multilayer dot-bracket in IPknot output.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert len(db) == len(seq), (
        f"Dot-bracket length mismatch from IPknot: len(db)={len(db)} vs len(seq)={len(seq)}\n"
        f"db={db}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    # 3) If header energy missing, try a strict standalone number somewhere (fallback).
    if energy is None:
        for ln in reversed(lines):
            if num_re.fullmatch(ln):
                try:
                    energy = float(ln)
                    break
                except ValueError:
                    pass
        # 4) Last-resort: search any number inside a line (still strict float shape).
        if energy is None:
            for ln in reversed(lines):
                m = num_re.search(ln)
                if m:
                    try:
                        energy = float(m.group(0))
                        break
                    except ValueError:
                        pass

    assert energy is not None, (
        "Could not find an energy in IPknot output (with -E). "
        "Tried header '(e=...)', then standalone/embedded numbers.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    return db, energy

# ---------- Fixtures ----------
@pytest.fixture(scope="module")
def energy_model():
    yaml_path = ir_files(rna_pk_fold) / "data" / "turner2004_eddyrivas1999_min.yaml"
    assert yaml_path is not None, "No parameter YAML found in rna_pk_fold/data."
    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=yaml_path)
    return SecondaryStructureEnergyModel(params=params, temp_k=310.15)

@pytest.fixture(scope="module")
def engine_nested(energy_model):
    # Compare nested-only (parentheses) vs IPknot parentheses-projection.
    return ZuckerFoldingEngine(
        energy_model=energy_model,
        config=ZuckerFoldingConfig(enable_pk_h=False)
    )

# ---------- Sequences ----------
SEQS = [
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

@pytest.mark.parametrize("seq", SEQS)
def test_nested_vs_ipknot_shape_and_energy(seq: str, engine_nested):
    # Our nested prediction
    st = make_fold_state(len(seq))
    engine_nested.fill_all_matrices(seq, st)
    ours_db = traceback_nested(seq, st).dot_bracket
    ours_mfe = st.w_matrix.get(0, len(seq) - 1)
    assert math.isfinite(ours_mfe)
    assert len(ours_db) == len(seq)

    # IPknot prediction
    ip_db, ip_e = run_ipknot(seq)
    assert len(ip_db) == len(seq)

    # Structure: compare parentheses onl
    TOL_BP = 2  # slightly looser than Vienna, since IPknot ≠ Turner MFE
    dist = bp_distance(ours_db, ip_db)  # ignores [],{},<> automatically
    assert dist <= TOL_BP, (
        f"Seq= {seq}\nours= {ours_db}\nipknot= {ip_db}\n"
        f"bp_distance= {dist} > {TOL_BP}\nours_mfe= {ours_mfe:.2f}, ipknot_energy= {ip_e:.2f}"
    )

    # Energy: different models—use a looser tolerance
    TOL_EN = 3.0
    delta_g = abs(ours_mfe - ip_e)
    assert delta_g <= TOL_EN, (
        f"MFE differs by {delta_g:.2f} kcal/mol > {TOL_EN:.2f}\n"
        f"Seq={seq}\nours_db={ours_db}\nipknot_db={ip_db}\n"
        f"ours_mfe={ours_mfe:.2f}, ipknot_energy={ip_e:.2f}"
    )
