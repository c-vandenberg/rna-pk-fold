from __future__ import annotations
import yaml
from pathlib import Path

from typing import Literal

from .data.yaml_io import read_yaml
from .data.parsers import (
    get_temperature_kelvin, parse_complements, validate_rna_complements,
    parse_multiloop, parse_loop_table, parse_stacks_matrix,
    parse_dangles, parse_mismatch, parse_special_hairpins,
)

from rna_pk_fold.energies.types import (
    SecondaryStructureEnergies,
    BasePairMap,
    MultiLoopCoeffs,
    PairEnergies,
    LoopEnergies,
)

Kind = Literal["RNA"]


class SecondaryStructureEnergyLoader:
    @classmethod
    def load(cls, kind: Kind = "RNA", yaml_path: str | Path | None = None) -> SecondaryStructureEnergies:
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
        return cls._build_rna(yaml_path)

    @staticmethod
    def _build_rna(yaml_path: str | Path | None = None) -> SecondaryStructureEnergies:
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
        special_hairpin = parse_special_hairpins(data, temp_k)

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
            SPECIAL_HAIRPINS=special_hairpin,
        )
