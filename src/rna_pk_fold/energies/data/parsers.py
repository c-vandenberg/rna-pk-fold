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
    Return the thermodynamic temperature (Kelvin) to use for conversions.

    Parameters
    ----------
    data : Mapping[str, Any]
        Parsed YAML tree (top-level dict).

    Returns
    -------
    float
        Temperature in Kelvin.

    Notes
    -----
    Prefer metadata.temperature_kelvin, else top-level temperature_kelvin,
    else default 310.15 K.
    """
    metadata = data.get("metadata") or {}
    temp_k = metadata.get("temperature_kelvin") or data.get("temperature_kelvin") or 310.15

    return float(temp_k)


def parse_complements(data: Mapping[str, Any]) -> BasePairMap:
    """
    Parse and normalize the base complement map.

    All keys and values are uppercased. The map must be present and non-empty.

    Parameters
    ----------
    data : Mapping[str, Any]
        Parsed YAML tree containing a ``"complements"`` mapping.

    Returns
    -------
    BasePairMap
        Mapping from nucleotide to its canonical complement.

    Raises
    ------
    ValueError
        If the complements mapping is missing or empty.
    """
    complements_data = data.get("complements")
    if not isinstance(complements_data, dict) or not complements_data:
        raise ValueError("YAML must contain a non-empty 'complements' mapping.")

    return {str(k).upper(): str(v).upper() for k, v in complements_data.items()}


def validate_rna_complements(complements: BasePairMap) -> None:
    """
    Validate that an RNA complement map contains U and does not contain T.

    Parameters
    ----------
    complements : BasePairMap
        Complement mapping as returned by :func: `parse_complements()`.

    Raises
    ------
    ValueError
        If 'U' is not present in keys or values, or if 'T' is present
        in keys or values.
    """
    if "U" not in complements.keys() and "U" not in complements.values():
        raise ValueError("RNA complements must include uracil ('U').")
    if "T" in complements.keys() or "T" in complements.values():
        raise ValueError("RNA complements must not contain thymine ('T') (DNA-specific).")


# ---------- Generic Cell Helpers ----------

def _cell(matrix: Any, idx_i: int, idx_j: int) -> float | None:
    """
    Safely fetch a numeric cell from a 2D matrix-like object.

    Parameters
    ----------
    matrix : Any
        2D indexable object (e.g., list of lists) or `None`.
    idx_i : int
        Row index.
    idx_j : int
        Column index.

    Returns
    -------
    float | None
        Cell value coerced to `float`, or `None` if out of bounds,
        matrix is `None`, or the cell itself is `None`.
    """
    if matrix is None:
        return None
    try:
        cell_value = matrix[idx_i][idx_j]
    except Exception:
        return None

    return None if cell_value is None else float(cell_value)

def _resolve_cell_thermo(
    dh_matrix: Any, ds_matrix: Any, dg_matrix: Any, idx_i: int, idx_j: int, temp_k: float
) -> Optional[Tuple[float, float]]:
    """
    Resolve a single cell's thermodynamic tuple (ΔH, ΔS) from up to three grids.

    The three inputs are parallel 2D matrices for ΔH, ΔS, and ΔG(T). Any two
    of the three quantities suffice; the missing one is derived using the
    temperature via:
        ΔG = ΔH − T * (ΔS / 1000)

    Parameters
    ----------
    dh_matrix : Any
        2D matrix for ΔH [kcal/mol], or `None`.
    ds_matrix : Any
        2D matrix for ΔS [cal/(K·mol)], or `None`.
    dg_matrix : Any
        2D matrix for ΔG(T) [kcal/mol] (may be named `dg` or `dg_37` upstream), or `None`.
    idx_i : int
        Row index.
    idx_j : int
        Column index.
    temp_k : float
        Temperature in Kelvin used for conversions.

    Returns
    -------
    tuple of float, float or None
        `(ΔH, ΔS)` if at least two values were present for the cell;
        `None` if the cell is entirely empty (all three are missing).

    Raises
    ------
    ValueError
        If one or zero of ΔH/ΔS/ΔG is provided (insufficient to resolve).
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
    """
    Parse multiloop coefficients `(a, b, c, d)` from the YAML tree.

    Parameters
    ----------
    data : Mapping[str, Any]
        Parsed YAML tree containing a `multiloop` mapping with numeric fields.

    Returns
    -------
    MultiLoopCoeffs
        Tuple `(a, b, c, d)` used in the affine multibranch model.

    Raises
    ------
    ValueError
        If the `multiloop` section is missing or not a mapping.

    Notes
    -----
    Typical usage in scoring is ``a + b * branches + c * unpaired``,
    with ``d`` optionally applied when zero unpaired nucleotides are enclosed.
    """
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
    Parse baseline loop energies indexed by loop length (nt).

    The function searches the first present key among `keys` (e.g.,
    `("hairpin_loops", "hairpin_loop")`) and expects a mapping of
    loop length → {dh|ds|dg}. Any two of ``dh``, ``ds``, ``dg`` suffice.

    Parameters
    ----------
    data : Mapping[str, Any]
        Parsed YAML tree.
    keys : Iterable[str]
        Candidate keys (loop types) for the loop table (checked in order).
    temp_k : float
        Temperature in Kelvin for ΔG/ΔH/ΔS conversions.

    Returns
    -------
    LoopEnergies
        Mapping `length:int → (ΔH, ΔS)`. Empty if no table is present.

    Raises
    ------
    ValueError
        If a loop entry provides only one of ΔH/ΔS/ΔG (insufficient to resolve).

    Notes
    -----
    Entries with all three missing values are skipped.
    """
    loop = None
    for loop_type in keys:
        if loop_type in data:
            loop = data[loop_type]
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
    Parse the nearest-neighbor stacking matrix into flat "XY/ZW" keys.

    The YAML section must have:
      - `rows` : list of left dimers (5'→3')
      - `cols` : list of right dimers (3'→5')
      - any two of `dh`, `ds`, `dg` (or `dg_37`) as 2D lists.

    Parameters
    ----------
    data : Mapping[str, Any]
        Parsed YAML tree containing a `stacks_matrix` section.
    temp_k : float
        Temperature in Kelvin for conversions.

    Returns
    -------
    PairEnergies
        Mapping "XY/ZW" → (ΔH, ΔS)" for all non-empty cells.
        Cells where all of ΔH/ΔS/ΔG are missing are skipped (not included).

    Notes
    -----
    This function uses :func:`_resolve_cell_thermo()` per cell to normalize
    to the common `(ΔH, ΔS)` representation.
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
            if delta_h_delta_s is None:
                continue

            stack_pair_energies[f"{r}/{c}"] = delta_h_delta_s

    return stack_pair_energies


# ---------- Dangles ----------

def _parse_dangle_matrix(
    dangle_matrix: Mapping[str, Any] | None,
    temp_k: float,
    key_fmtr: Callable[[str, str], str],
) -> PairEnergies:
    """
    Parse a single dangle matrix (5' or 3') into flat `PairEnergies`.

    The matrix is defined with:
      - `rows` : list of closing pairs (e.g., `"AU"``)
      - `cols` : list of flanking nucleotides (e.g., `"A","C","G","U"`)
      - any two of `dh`, `ds`, `dg` (or `dg_37`) as 2D lists.

    The final key is produced by `key_fmtr(pair, nuc)`, enabling
    orientations like `"N./PAIR"` (5' dangle) or `"PAIR/.N"` (3' dangle).

    Parameters
    ----------
    dangle_matrix : Mapping[str, Any] | None
        Matrix mapping from pairs × nucleotides to thermodynamic values.
    temp_k : float
        Temperature in Kelvin for conversions.
    key_fmtr : Callable[[str, str], str]
        Formatter that builds the final flat key from `(pair, nuc)`.

    Returns
    -------
    PairEnergies
        Mapping of formatted key → `(ΔH, ΔS)` for all non-empty cells.

    Notes
    -----
    Cells with all three values missing are omitted.
    """
    if not isinstance(dangle_matrix, dict):
        return {}

    pairs = [str(x) for x in dangle_matrix.get("rows", [])]
    nucs = [str(x) for x in dangle_matrix.get("cols", [])]
    dh_rows = dangle_matrix.get("dh")
    ds_rows = dangle_matrix.get("ds")
    dg_rows = dangle_matrix.get("dg") if "dg" in dangle_matrix else dangle_matrix.get("dg_37")

    dangle_energies: PairEnergies = {}
    for i, pair in enumerate(pairs):
        for j, nuc in enumerate(nucs):
            delta_h_delta_s = _resolve_cell_thermo(dh_rows, ds_rows, dg_rows, i, j, temp_k)
            if delta_h_delta_s is None:
                continue

            dangle_energies[key_fmtr(pair, nuc)] = delta_h_delta_s

    return dangle_energies


def parse_dangles(data: Mapping[str, Any], temp_k: float) -> PairEnergies:
    """
    Parse both 5' and 3' dangle matrices into a unified flat map.

    Expected YAML sections:
      - `dangle5_matrix` : → keys of the form `"N./PAIR"`
      - `dangle3_matrix` : → keys of the form `"PAIR/.N"`

    Parameters
    ----------
    data : Mapping[str, Any]
        Parsed YAML tree containing optional dangle sections.
    temp_k : float
        Temperature in Kelvin for conversions.

    Returns
    -------
    PairEnergies
        Combined mapping of dangle contributions for all present cells.
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
    Parse internal/terminal mismatch tables into flat `"XY/ZW"` keys.

    Supports two schemas:

    1) Sparse dimer/dimer matrix
       A dict with `rows`, `cols` and any two of `dh`, `ds`, `dg` (or `dg_37`)
       represented as 2D lists. Keys are directly ``"row/col"``.

    2) Closing-pair × nucleotide grid (Turner-2004 style)
       A dict with:
       - `pairs` : list of closing pairs `["CG","GC","GU","UG","AU","UA", ...]`
       - `nucs`  : list of nucleotides (may include placeholder like `"E"`)
       - per-pair 2D lists for any two of `dh`, `ds`, `dg` (or `dg_37`)

       Each cell is flattened to the legacy dimer/dimer key by:
       `key = f"{L}{X}/{Y}{R}"` where the closing pair is `X–Y`,
       left nucleotide is `L` and right nucleotide is `R`.

    Parameters
    ----------
    data : Mapping[str, Any]
        Parsed YAML tree containing the section.
    section : str
        Section name (e.g., `"internal_mismatches"` or `"terminal_mismatches"`).
    temp_k : float
        Temperature in Kelvin for conversions.

    Returns
    -------
    PairEnergies
        Mapping `"XY/ZW" → (ΔH, ΔS)"` for all non-empty cells.

    Raises
    ------
    ValueError
        If an addressable cell provides only one of ΔH/ΔS/ΔG (insufficient to resolve).

    Notes
    -----
    Placeholder nucleotides like `"E"` in the pair×nuc schema are skipped.
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
                    if delta_h_delta_s is None:
                        continue

                    left_right_dimer = f"{base_left}{base_x}/{base_y}{base_right}"
                    mm_energies[left_right_dimer] = delta_h_delta_s

        return mm_energies
    return {}


# ---------- Special hairpins (optional) ----------

def parse_special_hairpins(data: Mapping[str, Any], temp_k: float) -> PairEnergies:
    """
    Parse sequence-specific hairpin overrides (optional).

    Parameters
    ----------
    data : Mapping[str, Any]
        Parsed YAML tree containing an optional `special_hairpins` mapping
        of sequence → {dh|ds|dg}.
    temp_k : float
        Temperature in Kelvin for conversions.

    Returns
    -------
    PairEnergies
        Mapping `sequence → (ΔH, ΔS)` for entries with sufficient data.
        Entries with all three missing are skipped.

    Raises
    ------
    ValueError
        If an entry provides exactly one of ΔH/ΔS/ΔG (insufficient to resolve).
    """
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
