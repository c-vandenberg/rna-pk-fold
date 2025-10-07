from __future__ import annotations
import yaml
from pathlib import Path

from typing import Literal, Dict, Tuple, Any, Optional

from .data.yaml_io import read_yaml
from .data.parsers import (
    get_temperature_kelvin, parse_complements, validate_rna_complements,
    parse_multiloop, parse_loop_table, parse_stacks_matrix,
    parse_dangles, parse_mismatch, parse_special_hairpins,
)

from rna_pk_fold.energies.types import (
    SecondaryStructureEnergies,
    PseudoknotEnergies,
    BasePairMap,
    MultiLoopCoeffs,
    PairEnergies,
    LoopEnergies,
)

Kind = Literal["RNA"]


class SecondaryStructureEnergyLoader:
    def load(self, kind: Kind = "RNA", yaml_path: str | Path | None = None) -> SecondaryStructureEnergies:
        """
        Load the secondary structure thermodynamic energy bundle for a given
        nucleic acid class.

        Current implementation only supports RNA secondary structure energies.
        However, the design is open to extension for DNA.

        Parameters
        ----------
        kind : {"RNA"}, optional
            Parameter set to load. The current implementation only supports
            RNA secondary structure energies. The design is open to
            extension for DNA.

        Returns
        -------
        SecondaryStructureEnergies
            Immutable container with nearest-neighbor, loop, dangling,
            and multiloop coefficients used by the DP layer.

        Notes
        -----
        - Energies are stored as `(ΔH [kcal/mol], ΔS [cal/(K·mol)])`.
        - Conversion to free energy at temperature `T` (Kelvin) is:
          `ΔG = ΔH - T * (ΔS / 1000)`.
        """
        if kind.upper() != "RNA":
            raise ValueError("Only 'RNA' is supported for now.")
        return self._build_rna(yaml_path)

    def _build_rna(self, yaml_path: str | Path | None = None) -> SecondaryStructureEnergies:
        """
        Construct RNA thermodynamic tables (starter subset).

        Returns
        -------
        SecondaryStructureEnergies
            Immutable collection of RNA parameter tables suitable for driving
            the base Zuker-style DP (and extensions).

        References
        ----------
        1. Xia, T. et al. (1998). Thermodynamic parameters for an expanded nearest
          neighbor model for formation of RNA duplexes with Watson–Crick base pairs.
          Biochemistry, 37(42), 14719–14735.
        2. Mathews, D. H., Sabina, J., Zuker, M., & Turner, D. H. (1999). Expanded
          sequence dependence of thermodynamic parameters provides robust prediction
          of RNA secondary structure. J. Mol. Biol., 288(5), 911–940.
        """
        if yaml_path is None:
            raise ValueError("yaml_path is required.")

        data = read_yaml(yaml_path)
        temp_k = get_temperature_kelvin(data)

        complements = parse_complements(data)
        validate_rna_complements(complements)

        multiloop = parse_multiloop(data)
        nn_stack = parse_stacks_matrix(data, temp_k)
        dangles = parse_dangles(data, temp_k)
        hairpin = parse_loop_table(data, ("hairpin_loops", "hairpin_loop"), temp_k)
        bulge = parse_loop_table(data, ("bulge_loops", "bulge_loop"), temp_k)
        internal = parse_loop_table(data, ("internal_loops", "internal_loop"), temp_k)
        internal_mm = parse_mismatch(data, "internal_mismatches", temp_k)
        terminal_mm = parse_mismatch(data, "terminal_mismatches", temp_k)
        hairpin_mm = parse_mismatch(data, "hairpin_mismatches", temp_k)
        multi_mm = parse_mismatch(data, "multi_mismatch", temp_k)
        special_hairpin = parse_special_hairpins(data, temp_k)
        pseudoknots = self._parse_pseudoknot_block(data)

        return SecondaryStructureEnergies(
            BULGE=bulge,
            COMPLEMENT_BASES=complements,
            DANGLES=dangles,
            HAIRPIN=hairpin,
            MULTILOOP=multiloop,
            INTERNAL=internal,
            NN_STACK=nn_stack,
            INTERNAL_MISMATCH=internal_mm,
            TERMINAL_MISMATCH=terminal_mm,
            HAIRPIN_MISMATCH=hairpin_mm,
            SPECIAL_HAIRPINS=special_hairpin,
            PSEUDOKNOT=pseudoknots
        )

    @staticmethod
    def _parse_pseudoknot_block(data: Dict[str, Any]) -> Optional[PseudoknotEnergies]:
        """
        Parse optional 'pseudoknot' YAML section and return a PseudoknotEnergies
        instance (or None if absent). Missing keys fall back to defaults.
        """
        node = data.get("pseudoknot")
        if not node:
            return None

        def _f(key: str, default: float) -> float:
            return float(node.get(key, default))

        def _i(key: str, default: int) -> int:
            return int(node.get(key, default))

        def _bigram_map(key: str) -> Dict[Tuple[str, str], float]:
            out: Dict[Tuple[str, str], float] = {}
            raw = node.get(key, {}) or {}
            for bigram, val in raw.items():
                if isinstance(bigram, str) and len(bigram) == 2:
                    out[(bigram[0], bigram[1])] = float(val)
            return out

        def _coax_map(key: str) -> Dict[Tuple[str, str], float]:
            out: Dict[Tuple[str, str], float] = {}
            raw = node.get(key, {}) or {}
            for k, val in raw.items():
                if isinstance(k, str) and "|" in k:
                    left, right = k.split("|", 1)
                    out[(left.strip(), right.strip())] = float(val)
            return out

        def _caps(key: str) -> Dict[int, float]:
            out: Dict[int, float] = {}
            raw = node.get(key, {}) or {}
            for h, val in raw.items():
                try:
                    out[int(h)] = float(val)
                except Exception:
                    continue
            return out

        return PseudoknotEnergies(
            q_ss=_f("q_ss", 0.2),
            P_tilde_out=_f("P_tilde_out", 1.0),
            P_tilde_hole=_f("P_tilde_hole", 1.0),
            Q_tilde_out=_f("Q_tilde_out", 0.2),
            Q_tilde_hole=_f("Q_tilde_hole", 0.2),
            L_tilde=_f("L_tilde", 0.0),
            R_tilde=_f("R_tilde", 0.0),
            M_tilde_yhx=_f("M_tilde_yhx", 0.0),
            M_tilde_vhx=_f("M_tilde_vhx", 0.0),
            M_tilde_whx=_f("M_tilde_whx", 0.0),

            dangle_hole_L=_bigram_map("dangle_hole_L") or None,
            dangle_hole_R=_bigram_map("dangle_hole_R") or None,
            dangle_outer_L=_bigram_map("dangle_outer_L") or None,
            dangle_outer_R=_bigram_map("dangle_outer_R") or None,
            coax_pairs=_coax_map("coax_pairs") or None,

            coax_bonus=_f("coax_bonus", 0.0),
            coax_scale_oo=_f("coax_scale_oo", 1.0),
            coax_scale_oi=_f("coax_scale_oi", 1.0),
            coax_scale_io=_f("coax_scale_io", 1.0),
            coax_min_helix_len=_i("coax_min_helix_len", 1),
            coax_scale=_f("coax_scale", 1.0),

            mismatch_coax_scale=_f("mismatch_coax_scale", 0.5),
            mismatch_coax_bonus=_f("mismatch_coax_bonus", 0.0),

            join_drift_penalty=_f("join_drift_penalty", 0.0),

            short_hole_caps=_caps("short_hole_caps") or None,

            Gwh=_f("Gwh", 0.0),
            Gwi=_f("Gwi", 0.0),
            Gwh_wx=_f("Gwh_wx", 0.0),
            Gwh_whx=_f("Gwh_whx", 0.0),

            pk_penalty_gw=_f("pk_penalty_gw", 1.0),
        )


def pk_costs_dict(src: SecondaryStructureEnergies | PseudoknotEnergies) -> Dict[str, Any]:
    """
    Convert a PseudoknotEnergies instance (directly) or a SecondaryStructureEnergies
    (by reading .PSEUDOKNOT) into the flat dict that RivasEddyCosts expects.
    """
    if isinstance(src, SecondaryStructureEnergies):
        if src.PSEUDOKNOT is None:
            raise ValueError("No pseudoknot block found in SecondaryStructureEnergies.")
        pk = src.PSEUDOKNOT
    elif isinstance(src, PseudoknotEnergies):
        pk = src
    else:
        raise TypeError("pk_costs_dict expects SecondaryStructureEnergies or PseudoknotEnergies.")

    return {
        "q_ss": pk.q_ss,
        "P_tilde_out": pk.P_tilde_out,
        "P_tilde_hole": pk.P_tilde_hole,
        "Q_tilde_out": pk.Q_tilde_out,
        "Q_tilde_hole": pk.Q_tilde_hole,
        "L_tilde": pk.L_tilde,
        "R_tilde": pk.R_tilde,
        "M_tilde_yhx": pk.M_tilde_yhx,
        "M_tilde_vhx": pk.M_tilde_vhx,
        "M_tilde_whx": pk.M_tilde_whx,

        "dangle_hole_L": pk.dangle_hole_L or {},
        "dangle_hole_R": pk.dangle_hole_R or {},
        "dangle_outer_L": pk.dangle_outer_L or {},
        "dangle_outer_R": pk.dangle_outer_R or {},
        "coax_pairs": pk.coax_pairs or {},

        "coax_bonus": pk.coax_bonus,
        "coax_scale_oo": pk.coax_scale_oo,
        "coax_scale_oi": pk.coax_scale_oi,
        "coax_scale_io": pk.coax_scale_io,
        "coax_min_helix_len": pk.coax_min_helix_len,
        "coax_scale": pk.coax_scale,

        "mismatch_coax_scale": pk.mismatch_coax_scale,
        "mismatch_coax_bonus": pk.mismatch_coax_bonus,

        "join_drift_penalty": pk.join_drift_penalty,

        "short_hole_caps": pk.short_hole_caps or {},

        "Gwh": pk.Gwh,
        "Gwi": pk.Gwi,
        "Gwh_wx": pk.Gwh_wx,
        "Gwh_whx": pk.Gwh_whx,
    }
