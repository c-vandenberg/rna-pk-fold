# tests/integration/test_full_vs_ipknot.py
import os
import re
import shutil
import subprocess
import math
import tempfile
from importlib.resources import files as ir_files
from typing import List, Set, Tuple, Dict, Optional

import pytest

import rna_pk_fold
# --- Zucker (nested) ---
from rna_pk_fold.folding.zucker import make_fold_state
from rna_pk_fold.folding.zucker.zucker_recurrences import ZuckerFoldingConfig, ZuckerFoldingEngine
from rna_pk_fold.folding.zucker.zucker_traceback import traceback_nested as zucker_traceback

# --- Eddy–Rivas (pseudoknots) ---
from rna_pk_fold.folding.eddy_rivas import eddy_rivas_recurrences as ER
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import make_re_fold_state

# --- Energy model ---
from rna_pk_fold.energies import SecondaryStructureEnergyLoader
from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModel

# --- Common helpers you provided ---
from rna_pk_fold.folding.common_traceback import (
    pairs_to_multilayer_dotbracket,
    dotbracket_to_pairs,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("ipknot") is None, reason="ipknot not found on PATH"),
]

# Try to import an optional full PK traceback from your ER module.
# If you don’t have it yet, we’ll fall back to parentheses-only checks.
try:
    # expected signature: traceback_full(seq: str, z_state, re_state) -> TraceResult
    from rna_pk_fold.folding.eddy_rivas.eddy_rivas_traceback import traceback_full as er_traceback_full  # type: ignore
except Exception:
    er_traceback_full = None  # type: ignore


# ------------------------
# IPknot runner
# ------------------------
def run_ipknot(seq: str) -> Tuple[str, float]:
    """
    Call IPknot with -E to print free energy and parse multilayer dot-bracket + energy.
    Returns (dotbracket, energy_kcal_mol).
    """
    ipknot_bin = os.environ.get("IPKNOT_BIN") or shutil.which("ipknot")
    assert ipknot_bin, "ipknot binary not found (set IPKNOT_BIN or add to PATH)."

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

    db_re = re.compile(r"^[.\(\)\[\]\{\}\<\>]+$")  # multilayer db
    num_re = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")

    db: Optional[str] = None
    energy: Optional[float] = None

    for ln in reversed(lines):
        if energy is None and num_re.fullmatch(ln):
            energy = float(ln)
            continue
        if db is None and db_re.fullmatch(ln):
            db = ln
            continue
        if db is not None and energy is not None:
            break

    assert db is not None, (
        "Could not find multilayer dot-bracket in IPknot output.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    if energy is None:
        # fall back: scan for a numeric token anywhere
        for ln in reversed(lines):
            m = re.search(r"([-+]?[\d.]+(?:[eE][-+]?\d+)?)", ln)
            if m:
                try:
                    energy = float(m.group(1))
                    break
                except ValueError:
                    pass

    assert energy is not None, (
        "Could not find an energy in IPknot output (with -E).\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return db, energy


# ------------------------
# Fixtures
# ------------------------
@pytest.fixture(scope="module")
def energy_model():
    # Use your full YAML (with ER params) if available:
    yaml_path = ir_files(rna_pk_fold) / "data" / "turner2004_eddyrivas1999_min.yaml"
    assert yaml_path is not None, "No parameter YAML found."
    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=yaml_path)
    return SecondaryStructureEnergyModel(params=params, temp_k=310.15)


@pytest.fixture(scope="module")
def engines_and_costs(energy_model):
    """
    Return (zucker_engine, er_engine, er_costs_template).
    We keep ER costs minimal-but-complete unless you wire a YAML->PseudoknotEnergies bridge.
    """
    z_engine = ZuckerFoldingEngine(
        energy_model=energy_model,
        config=ZuckerFoldingConfig(enable_pk_h=True, pk_h_penalty=1.0),
    )

    # Required scalars for PseudoknotEnergies — zeros are fine for structural tests
    # (if you have a real loader, replace this with costs_from_vienna_like(...) + costs_from_dict(...)).
    costs = ER.costs_from_dict({
        "q_ss": 0.0,
        "P_tilde_out": 0.0, "P_tilde_hole": 0.0,
        "Q_tilde_out": 0.0, "Q_tilde_hole": 0.0,
        "L_tilde": 0.0, "R_tilde": 0.0,
        "M_tilde_yhx": 0.0, "M_tilde_vhx": 0.0, "M_tilde_whx": 0.0,
    })

    # Pull a pk penalty if present in your YAML; fallback otherwise.
    try:
        pk_gw = getattr(getattr(energy_model.params, "PSEUDOKNOT", None), "pk_penalty_gw", 1.0)
    except Exception:
        pk_gw = 1.0

    er_cfg = ER.EddyRivasFoldingConfig(
        enable_coax=True,
        enable_coax_variants=True,
        enable_coax_mismatch=True,
        enable_wx_overlap=True,
        enable_join_drift=False,
        pk_penalty_gw=pk_gw,
        costs=costs,
    )
    er_engine = ER.EddyRivasFoldingEngine(er_cfg)
    return z_engine, er_engine, costs


# ------------------------
# Helpers for distances
# ------------------------
def dotbracket_to_pairs_multilayer(db: str) -> Set[Tuple[int, int]]:
    """Parse layered dot-bracket ((),[],{},<>) by treating each layer independently."""
    stacks: Dict[str, List[int]] = {'(': [], '[': [], '{': [], '<': []}
    close_to_open = {')': '(', ']': '[', '}': '{', '>': '<'}
    out: Set[Tuple[int, int]] = set()
    for i, ch in enumerate(db):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in close_to_open:
            op = close_to_open[ch]
            if stacks[op]:
                j = stacks[op].pop()
                out.add((j, i))
    return out


def bp_distance(db1: str, db2: str) -> int:
    p1 = dotbracket_to_pairs(db1)
    p2 = dotbracket_to_pairs(db2)
    return len(p1 ^ p2)


def bp_distance_multilayer(db1: str, db2: str) -> int:
    p1 = dotbracket_to_pairs_multilayer(db1)
    p2 = dotbracket_to_pairs_multilayer(db2)
    return len(p1 ^ p2)


def project_parentheses(db: str) -> str:
    """Keep only () for shape comparison to nested structures."""
    return ''.join(ch if ch in '().' else '.' for ch in db)


# ------------------------
# Sequences
# ------------------------
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


# ------------------------
# The test
# ------------------------
@pytest.mark.parametrize("seq", SEQS)
def test_full_vs_ipknot_shape_and_energy(seq: str, energy_model, engines_and_costs):
    z_engine, er_engine, _ = engines_and_costs

    # 1) Run your nested engine and traceback (always available)
    z_state = make_fold_state(len(seq))
    z_engine.fill_all_matrices(seq, z_state)
    nested_tr = zucker_traceback(seq, z_state)
    ours_nested_db = nested_tr.dot_bracket
    ours_nested_e = z_state.w_matrix.get(0, len(seq) - 1)
    assert math.isfinite(ours_nested_e)

    # 2) Run your Eddy–Rivas engine (fills its own state)
    re_state = make_re_fold_state(len(seq))
    er_engine.fill_with_costs(seq, z_state, re_state)  # use the combined fill your ER engine expects
    # WX total is the published nested+PK energy in ER path:
    ours_full_e = re_state.wx_matrix.get(0, len(seq) - 1)
    assert math.isfinite(ours_full_e)

    # 3) IPknot external reference
    ip_db, ip_e = run_ipknot(seq)

    # 4) Structure comparison
    if callable(er_traceback_full):
        # If you add a full traceback later, we'll use it to get multilayer db:
        full_tr = er_traceback_full(seq, z_state, re_state)  # expected to return TraceResult
        ours_full_db = full_tr.dot_bracket
        assert len(ours_full_db) == len(seq) == len(ip_db)
        TOL_BP_MULTI = 4  # looser than nested-only
        dist = bp_distance_multilayer(ours_full_db, ip_db)
        assert dist <= TOL_BP_MULTI, (
            f"Seq= {seq}\nours(full)= {ours_full_db}\nipknot= {ip_db}\n"
            f"bp_distance_multilayer= {dist} > {TOL_BP_MULTI}\n"
            f"ours_full_e= {ours_full_e:.2f}, ipknot_e= {ip_e:.2f}"
        )
    else:
        # No full traceback yet: compare parentheses projection against our nested-only result.
        # This still exercises the ER engine for energy but uses nested structure for shape.
        ip_paren = project_parentheses(ip_db)
        assert len(ours_nested_db) == len(ip_paren) == len(seq)
        TOL_BP = 2
        dist = bp_distance(ours_nested_db, ip_paren)
        assert dist <= TOL_BP, (
            f"(Fallback: no full ER traceback in your code yet)\n"
            f"Seq= {seq}\nours(nested)= {ours_nested_db}\nipknot(paren)= {ip_paren}\n"
            f"bp_distance= {dist} > {TOL_BP}\n"
            f"ours_full_e= {ours_full_e:.2f}, ipknot_e= {ip_e:.2f}"
        )

    # 5) Energy comparison
    # Different objective (IPknot ≠ Turner MFE), so keep this loose.
    TOL_EN = 5.0
    delta = abs(ours_full_e - ip_e)
    assert delta <= TOL_EN, (
        f"Energy gap {delta:.2f} > {TOL_EN:.2f}\n"
        f"Seq= {seq}\nours_full_e= {ours_full_e:.2f}, ipknot_e= {ip_e:.2f}"
    )

