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
    """
    Loads and parses thermodynamic energy parameters from a YAML file.

    This class is responsible for reading a specially formatted YAML file
    containing thermodynamic parameters (ΔH and ΔS) for various RNA
    secondary structure motifs and parsing them into a structured,
    immutable `SecondaryStructureEnergies` object.
    """
    def load(self, kind: Kind = "RNA", yaml_path: str | Path | None = None) -> SecondaryStructureEnergies:
        """
        Loads the thermodynamic energy parameter bundle for a given nucleic acid.

        This is the main entry point for the loader. It validates the requested
        nucleic acid type and dispatches to the appropriate builder method.

        Parameters
        ----------
        kind : {"RNA"}, optional
            The type of parameter set to load. Currently, only "RNA" is
            supported, by default "RNA".
        yaml_path : str | Path | None
            The file path to the YAML file containing the energy parameters.

        Returns
        -------
        SecondaryStructureEnergies
            An immutable data object containing all the parsed thermodynamic
            parameter tables required by the folding engines.

        Raises
        ------
        ValueError
            If a `kind` other than "RNA" is specified or if `yaml_path` is not provided.

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
        Constructs the RNA thermodynamic parameter tables from a YAML file.

        This method reads the specified YAML file and uses a series of specialized
        parsers to extract and structure the data for each type of thermodynamic
        parameter (e.g., stacking, loops, dangles).

        Parameters
        ----------
        yaml_path : str | Path | None
            The file path to the energy parameter YAML file.

        Returns
        -------
        SecondaryStructureEnergies
            An immutable collection of RNA parameter tables.

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

        # Read the raw dictionary data from the YAML file.
        data = read_yaml(yaml_path)
        temp_k = get_temperature_kelvin(data)

        # --- Parse Nested Structure Parameters ---
        # Each function call here is responsible for a specific section of the YAML file.
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

        # --- Parse Pseudoknot-Specific Parameters ---
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
        """
        Parses the 'pseudoknot' section of the YAML data.

        This static method extracts all parameters specific to the Eddy-Rivas
        pseudoknot folding algorithm from the raw data dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            The raw dictionary data loaded from the YAML file.

        Returns
        -------
        Optional[PseudoknotEnergies]
            An immutable data object with the pseudoknot parameters, or None if
            the 'pseudoknot' section is not present in the YAML file.
        """
        node = data.get("pseudoknot")
        if not node:
            return None

        # Use helper functions to parse each specific parameter from the node,
        # providing a default value if the key is missing.
        return PseudoknotEnergies(
            # Scalar energy penalties and bonuses.
            q_ss=get_float(node, "q_ss", 0.2),
            p_tilde_out=get_float(node, "P_tilde_out", 1.0),
            p_tilde_hole=get_float(node, "P_tilde_hole", 1.0),
            q_tilde_out=get_float(node, "Q_tilde_out", 0.2),
            q_tilde_hole=get_float(node, "Q_tilde_hole", 0.2),
            l_tilde=get_float(node, "L_tilde", 0.0),
            r_tilde=get_float(node, "R_tilde", 0.0),
            m_tilde_yhx=get_float(node, "M_tilde_yhx", 0.0),
            m_tilde_vhx=get_float(node, "M_tilde_vhx", 0.0),
            m_tilde_whx=get_float(node, "M_tilde_whx", 0.0),

            # Sequence-dependent energy maps.
            dangle_hole_left=(parse_bigram_float_map(node, "dangle_hole_left") or None),
            dangle_hole_right=(parse_bigram_float_map(node, "dangle_hole_right") or None),
            dangle_outer_left=(parse_bigram_float_map(node, "dangle_outer_left") or None),
            dangle_outer_right=(parse_bigram_float_map(node, "dangle_outer_right") or None),
            coax_pairs=(parse_coax_pairs_map(node, "coax_pairs") or None),

            # Coaxial stacking control parameters.
            coax_bonus=get_float(node, "coax_bonus", 0.0),
            coax_scale_oo=get_float(node, "coax_scale_oo", 1.0),
            coax_scale_oi=get_float(node, "coax_scale_oi", 1.0),
            coax_scale_io=get_float(node, "coax_scale_io", 1.0),
            coax_min_helix_len=get_int(node, "coax_min_helix_len", 1),
            coax_scale=get_float(node, "coax_scale", 1.0),
            mismatch_coax_scale=get_float(node, "mismatch_coax_scale", 0.5),
            mismatch_coax_bonus=get_float(node, "mismatch_coax_bonus", 0.0),

            # Penalties for specific pseudoknot compositions.
            join_drift_penalty=get_float(node, "join_drift_penalty", 0.0),
            short_hole_caps=(parse_int_float_map(node, "short_hole_caps") or None),

            # Global penalties for initiating pseudoknots or overlaps.
            g_wh=get_float(node, "Gwh", 0.0),
            g_wi=get_float(node, "Gwi", 0.0),
            g_wh_wx=get_float(node, "Gwh_wx", 0.0),
            g_wh_whx=get_float(node, "Gwh_whx", 0.0),
            pk_penalty_gw=get_float(node, "pk_penalty_gw", 1.0),
        )
