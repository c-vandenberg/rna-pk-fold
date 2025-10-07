from __future__ import annotations
from pathlib import Path
from typing import Literal, Dict, Any, Optional

from .data.yaml_io import read_yaml
from .data.parsers import (
    get_temperature_kelvin, parse_complements, validate_rna_complements,
    parse_multiloop, parse_loop_table, parse_stacks_matrix,
    parse_dangles, parse_mismatch, parse_special_hairpins,
)

from rna_pk_fold.energies.energy_types import (
    SecondaryStructureEnergies,
    PseudoknotEnergies
)
from rna_pk_fold.energies.data.parsers import (get_float, get_int, parse_bigram_float_map, parse_coax_pairs_map,
                                               parse_int_float_map)

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
            MULTI_MISMATCH=multi_mm,
            SPECIAL_HAIRPINS=special_hairpin,
            PSEUDOKNOT=pseudoknots
        )

    @staticmethod
    def _parse_pseudoknot_block(data: Dict[str, Any]) -> Optional[PseudoknotEnergies]:
        node = data.get("pseudoknot")
        if not node:
            return None

        return PseudoknotEnergies(
            # scalars
            q_ss=get_float(node, "q_ss", 0.2),
            P_tilde_out=get_float(node, "P_tilde_out", 1.0),
            P_tilde_hole=get_float(node, "P_tilde_hole", 1.0),
            Q_tilde_out=get_float(node, "Q_tilde_out", 0.2),
            Q_tilde_hole=get_float(node, "Q_tilde_hole", 0.2),
            L_tilde=get_float(node, "L_tilde", 0.0),
            R_tilde=get_float(node, "R_tilde", 0.0),
            M_tilde_yhx=get_float(node, "M_tilde_yhx", 0.0),
            M_tilde_vhx=get_float(node, "M_tilde_vhx", 0.0),
            M_tilde_whx=get_float(node, "M_tilde_whx", 0.0),

            # maps
            dangle_hole_left=(parse_bigram_float_map(node, "dangle_hole_left") or None),
            dangle_hole_right=(parse_bigram_float_map(node, "dangle_hole_right") or None),
            dangle_outer_left=(parse_bigram_float_map(node, "dangle_outer_left") or None),
            dangle_outer_right=(parse_bigram_float_map(node, "dangle_outer_right") or None),
            coax_pairs=(parse_coax_pairs_map(node, "coax_pairs") or None),

            # coax controls
            coax_bonus=get_float(node, "coax_bonus", 0.0),
            coax_scale_oo=get_float(node, "coax_scale_oo", 1.0),
            coax_scale_oi=get_float(node, "coax_scale_oi", 1.0),
            coax_scale_io=get_float(node, "coax_scale_io", 1.0),
            coax_min_helix_len=get_int(node, "coax_min_helix_len", 1),
            coax_scale=get_float(node, "coax_scale", 1.0),

            mismatch_coax_scale=get_float(node, "mismatch_coax_scale", 0.5),
            mismatch_coax_bonus=get_float(node, "mismatch_coax_bonus", 0.0),

            # composition variants / penalties
            join_drift_penalty=get_float(node, "join_drift_penalty", 0.0),
            short_hole_caps=(parse_int_float_map(node, "short_hole_caps") or None),

            # composition offsets
            Gwh=get_float(node, "Gwh", 0.0),
            Gwi=get_float(node, "Gwi", 0.0),
            Gwh_wx=get_float(node, "Gwh_wx", 0.0),
            Gwh_whx=get_float(node, "Gwh_whx", 0.0),

            # optional global PK penalty scaling (if you included this in the dataclass)
            pk_penalty_gw=get_float(node, "pk_penalty_gw", 1.0),
        )
