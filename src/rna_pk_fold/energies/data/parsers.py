from __future__ import annotations
from typing import Any, Iterable, Mapping, Tuple, Callable, Optional

from rna_pk_fold.energies.types import (
    BasePairMap,
    MultiLoopCoeffs,
    PairEnergies,
    LoopEnergies,
)

from rna_pk_fold.energies.data.thermo_math import resolve_dh_ds


# ---------- Top-level config helpers ----------

def get_temperature_kelvin(data: Mapping[str, Any]) -> float:
    """
    Prefer metadata.temperature_kelvin, else top-level temperature_kelvin,
    else default 310.15 K.
    """
    metadata = data.get("metadata") or {}
    temp_k = metadata.get("temperature_kelvin") or data.get("temperature_kelvin") or 310.15

    return float(temp_k)


def parse_complements(data: Mapping[str, Any]) -> BasePairMap:
    """
    Parse and normalize the complements map (uppercase).
    """
    complements_data = data.get("complements")
    if not isinstance(complements_data, dict) or not complements_data:
        raise ValueError("YAML must contain a non-empty 'complements' mapping.")

    return {str(k).upper(): str(v).upper() for k, v in complements_data.items()}


def validate_rna_complements(complements: BasePairMap) -> None:
    """
    Must include U; must not include T (keys or values).
    """
    if "U" not in complements.keys() and "U" not in complements.values():
        raise ValueError("RNA complements must include uracil ('U').")
    if "T" in complements.keys() or "T" in complements.values():
        raise ValueError("RNA complements must not contain thymine ('T') (DNA-specific).")


# ---------- Generic Cell Helpers ----------

def _cell(matrix: Any, i: int, j: int) -> float | None:
    if matrix is None:
        return None
    try:
        cell_value = matrix[i][j]
    except Exception:
        return None

    return None if cell_value is None else float(cell_value)

def _resolve_cell_thermo(
    dh_matrix: Any, ds_matrix: Any, dg_matrix: Any, idx_i: int, idx_j: int, temp_k: float
) -> Optional[Tuple[float, float]]:
    """
    Pull (dh, ds, dg) from [i,j] across three matrices, skip if all None,
    otherwise return a normalized (ΔH, ΔS) tuple. None → skip.
    """
    dh = _cell(dh_matrix, idx_i, idx_j)
    ds = _cell(ds_matrix, idx_i, idx_j)
    dg = _cell(dg_matrix, idx_i, idx_j)
    if dh is None and ds is None and dg is None:
        return None

    delta_h, delta_s = resolve_dh_ds(dh=dh, ds=ds, dg=dg, temp_k=temp_k)

    return delta_h, delta_s


# ---------- Multiloop ----------

def parse_multiloop(data: Mapping[str, Any]) -> MultiLoopCoeffs:
    multiloop_data = data.get("multiloop")
    if not isinstance(multiloop_data, dict):
        raise ValueError("Missing 'multiloop' section.")

    coeff_a = float(multiloop_data.get("a", 0.0))
    coeff_b = float(multiloop_data.get("b", 0.0))
    coeff_c = float(multiloop_data.get("c", 0.0))
    coeff_d = float(multiloop_data.get("d", 0.0))

    return coeff_a, coeff_b, coeff_c, coeff_d


# ---------- Loop length tables ----------

def parse_loop_table(
    data: Mapping[str, Any],
    keys: Iterable[str],
    temp_k: float,
) -> LoopEnergies:
    """
    Parse hairpin/bulge/internal loop baselines.
    Supports either '..._loops' or singular alias in `keys`.
    """
    loop = None
    for k in keys:
        if k in data:
            loop = data[k]
            break

    if not isinstance(loop, dict):
        return {}

    loop_energies: LoopEnergies = {}
    for length_str, entry in loop.items():
        loop_len = int(length_str)
        dh = entry.get("dh")
        ds = entry.get("ds")
        dg = entry.get("dg") if "dg" in entry else entry.get("dg_37")
        if dh is None and ds is None and dg is None:
            continue

        delta_h, delta_s = resolve_dh_ds(dh=dh,ds=ds,dg=dg,temp_k=temp_k)
        loop_energies[loop_len] = (delta_h, delta_s)

    return loop_energies


# ---------- Stacks (nearest-neighbor) ----------

def parse_stacks_matrix(data: Mapping[str, Any], temp_k: float) -> PairEnergies:
    """
    Matrix with 'rows', 'cols' and any two of dh/ds/dg (or dg_37).
    Produces flat keys "XY/ZW".
    """
    stacks_matrix = data.get("stacks_matrix")
    if not isinstance(stacks_matrix, dict):
        return {}

    rows = list(map(str, stacks_matrix.get("rows", [])))
    cols = list(map(str, stacks_matrix.get("cols", [])))
    dh_rows = stacks_matrix.get("dh")
    ds_rows = stacks_matrix.get("ds")
    dg_rows = stacks_matrix.get("dg") if "dg" in stacks_matrix else stacks_matrix.get("dg_37")

    stack_pair_energies: PairEnergies = {}
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            delta_h_delta_s = _resolve_cell_thermo(dh_rows, ds_rows, dg_rows, i, j, temp_k)
            stack_pair_energies[f"{r}/{c}"] = delta_h_delta_s

    return stack_pair_energies


# ---------- Dangles ----------

def _parse_dangle_matrix(
    dangle_matrix: Mapping[str, Any] | None,
    temp_k: float,
    key_fmt: Callable[[str, str], str],
) -> PairEnergies:
    """Parse a single dangle matrix (5' or 3') into flat PairEnergies."""
    if not isinstance(dangle_matrix, dict):
        return {}

    pairs = [str(x) for x in dangle_matrix.get("rows", [])]
    nucs  = [str(x) for x in dangle_matrix.get("cols", [])]
    dh_rows = dangle_matrix.get("dh")
    ds_rows = dangle_matrix.get("ds")
    dg_rows = dangle_matrix.get("dg") if "dg" in dangle_matrix else dangle_matrix.get("dg_37")

    dangle_energies: PairEnergies = {}
    for i, pair in enumerate(pairs):
        for j, nuc in enumerate(nucs):
            delta_h_delta_s = _resolve_cell_thermo(dh_rows, ds_rows, dg_rows, i, j, temp_k)
            dangle_energies[key_fmt(pair, nuc)] = delta_h_delta_s

    return dangle_energies


def parse_dangles(data: Mapping[str, Any], temp_k: float) -> PairEnergies:
    """
    dangle5_matrix: rows=pairs, cols=nucs → key "N./PAIR"
    dangle3_matrix: rows=pairs, cols=nucs → key "PAIR/.N"
    """
    dangles_energies: PairEnergies = {}
    dangles_energies.update(_parse_dangle_matrix(
        data.get("dangle5_matrix"), temp_k, lambda pair, nuc: f"{nuc}./{pair}"
    ))
    dangles_energies.update(_parse_dangle_matrix(
        data.get("dangle3_matrix"), temp_k, lambda pair, nuc: f"{pair}/.{nuc}"
    ))

    return dangles_energies

# ---------- Mismatches (two schemas → flat "XY/ZW") ----------

def parse_mismatch(data: Mapping[str, Any], section: str, temp_k: float) -> PairEnergies:
    """
    1) Sparse XY/ZW matrix:
       { rows: [...], cols: [...], dh/ds/dg grids }
    2) Closing-pair × nucleotide grid:
       { pairs: [...], nucs: [...], dh/ds/dg per pair as 2D lists }
       Flatten: closing pair X–Y, left nuc L, right nuc R → key = f"{L}{X}/{Y}{R}".
       Cells with placeholder nucs like 'E' are skipped.
    """
    mm_data = data.get(section)
    if not isinstance(mm_data, dict):
        return {}

    # Case 1: "XY/ZW" Sparse Matrix
    if "rows" in mm_data and "cols" in mm_data:
        rows = list(map(str, mm_data.get("rows", [])))
        cols = list(map(str, mm_data.get("cols", [])))
        dh_rows = mm_data.get("dh")
        ds_rows = mm_data.get("ds")
        dg_rows = mm_data.get("dg") if "dg" in mm_data else mm_data.get("dg_37")
        mm_energies: PairEnergies = {}
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                dh = _cell(dh_rows, i, j)
                ds = _cell(ds_rows, i, j)
                dg = _cell(dg_rows, i, j)
                if dh is None and ds is None and dg is None:
                    continue

                delta_h, delta_s = resolve_dh_ds(dh=dh, ds=ds, dg=dg, temp_k=temp_k)
                mm_energies[f"{r}/{c}"] = (delta_h, delta_s)

        return mm_energies

    # Case 2: closing-pair × nucleotide grid
    if "pairs" in mm_data and "nucs" in mm_data:
        pairs = list(map(str, mm_data["pairs"]))
        nucs = list(map(str, mm_data["nucs"]))
        dh_all = mm_data.get("dh") or {}
        ds_all = mm_data.get("ds") or {}
        dg_all = mm_data.get("dg") or mm_data.get("dg_37") or {}
        mm_energies: PairEnergies = {}
        for pair in pairs:
            if len(pair) != 2:
                continue
            base_x, base_y = pair[0], pair[1]
            dh_matrix = dh_all.get(pair)
            ds_matrix = ds_all.get(pair)
            dg_matrix = dg_all.get(pair)
            for i, base_left in enumerate(nucs):
                for j, base_right in enumerate(nucs):
                    if base_left in ("E",) or base_right in ("E",):  # Skip placeholders XY/ZW"
                        continue

                    delta_h_delta_s = _resolve_cell_thermo(dh_matrix, ds_matrix, dg_matrix, i, j, temp_k)
                    left_right_dimer = f"{base_left}{base_x}/{base_y}{base_right}"
                    mm_energies[left_right_dimer] = delta_h_delta_s

        return mm_energies
    return {}


# ---------- Special hairpins (optional) ----------

def parse_special_hairpins(data: Mapping[str, Any], temp_k: float) -> PairEnergies:
    special_hairpins_data = data.get("special_hairpins")
    if not isinstance(special_hairpins_data, dict):
        return {}
    special_hairpins_energies: PairEnergies = {}
    for k, entry in special_hairpins_data.items():
        if entry is None:
            continue
        dh = entry.get("dh")
        ds = entry.get("ds")
        dg = entry.get("dg") if "dg" in entry else entry.get("dg_37")
        if dh is None and ds is None and dg is None:
            continue

        delta_h, delta_s = resolve_dh_ds(dh=dh, ds=ds, dg=dg, temp_k=temp_k)
        special_hairpins_energies[str(k)] = (delta_h, delta_s)

    return special_hairpins_energies
