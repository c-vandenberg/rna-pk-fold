from __future__ import annotations
import yaml
from pathlib import Path

from typing import Literal

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
    def load(cls, kind: Kind = "RNA") -> SecondaryStructureEnergies:
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
        return cls._build_rna()

    @staticmethod
    def _load_thermo_from_yaml(path: str | Path):
        thermo_data = yaml.safe_load(Path(path).read_text())

        return thermo_data

    @staticmethod
    def _build_rna() -> SecondaryStructureEnergies:
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
        # Map of canonical complements for RNA: `{"A":"U","U":"A","G":"C","C":"G","N":"N"}`.
        # `N` represents an IUPAC ambiguity (unknown) nucleotide; its complement
        #  is kept as `N`
        complement: BasePairMap = {"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"}

        # Minimal nearest-neighbor stack table using `"ij/kl"` keys
        nn_stack: PairEnergies = {
            "AU/UA": (-7.7, -20.6),
            "UA/AU": (-7.7, -20.6),
            "GC/CG": (-14.9, -37.1),
            "CG/GC": (-10.6, -26.4),
            "GU/CA": (-11.4, -29.7),
            "UG/AC": (-10.4, -26.8),
            "AG/UU": (-3.21, -8.6),
            "AU/UG": (-8.81, -24.0),
            "CG/GU": (-5.61, -13.5),
            "CU/GG": (-12.11, -32.2),
            "GG/CU": (-8.33, -21.9),
            "GU/CG": (-12.59, -32.5),
            "GA/UU": (-12.83, -37.3),
            "GG/UU": (-13.47, -44.9),
            "GU/UG": (-14.59, -51.2),
            "UG/AU": (-6.99, -19.3),
            "UG/GU": (-9.26, -30.8),
        }


        # Dangling-end contributions where one side has a single unpaired
        # nucleotide adjacent to a closing pair. Keys can include `"."`
        # to mark the absent partner (e.g. `"A./UA"` means a 5' dangle on
        # the left).
        dangles: PairEnergies = {
            "A./UA": (-0.5, -0.6),  # 5' dangle left
            "UA/A.": (-0.5, -0.6),  # 3' dangle right
        }

        # Hairpin loop baseline as a function of total loop length (nt)
        # For lengths outside the table, typical practice is to
        # use Jacobson–Stockmayer extrapolation (to be added).
        hairpin: LoopEnergies = {
            3: (1.3, -13.2),
            4: (4.8, -2.6),
            6: (-2.9, -26.8),
            10: (5.0, -4.8),
            30: (5.0, -8.7),
        }

        # Bulge loop baseline as a function of total loop length (nt).
        bulge: LoopEnergies = {
            1: (10.6, 21.9),
            2: (7.1, 13.9),
            4: (7.1, 11.3),
            10: (7.1, 7.1),
            30: (7.1, 3.2),
        }

        # Internal loop baseline by total loop length (nt) (excluding 1×1).
        internal: LoopEnergies = {
            4: (-7.2, -26.8),
            6: (-1.3, -10.6),
            10: (-1.3, -12.3),
            30: (-1.3, -16.1),
        }

        # Linear multibranch model coefficients for multiloops
        multiloop: MultiLoopCoeffs = (2.5, 0.1, 0.4, 2.0)  # (a, b, c, d)

        return SecondaryStructureEnergies(
            BULGE=bulge,
            COMPLEMENT_BASES=complement,
            DANGLES=dangles,
            HAIRPIN=hairpin,
            MULTILOOP=multiloop,
            INTERNAL=internal,
            NN_STACK=nn_stack,
            INTERNAL_MISMATCH=internal_mm,
            TERMINAL_MISMATCH=terminal_mm,
            SPECIAL_HAIRPINS=None,
        )
