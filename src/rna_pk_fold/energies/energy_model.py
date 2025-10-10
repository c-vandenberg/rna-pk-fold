from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Tuple

from rna_pk_fold.energies.energy_types import SecondaryStructureEnergies, PseudoknotEnergies
from rna_pk_fold.energies.energy_ops import (
    hairpin_energy, stack_energy, internal_loop_energy, multiloop_linear_energy,
)
from rna_pk_fold.energies.energy_pk_ops import (
    dangle_hole_left as pk_dangle_hole_left_fn,
    dangle_hole_right as pk_dangle_hole_right_fn,
    dangle_outer_left as pk_dangle_outer_left_fn,
    dangle_outer_right as pk_dangle_outer_right_fn,
    coax_pack as pk_coax_pack_fn,
    short_hole_penalty as pk_short_hole_penalty_fn,
)
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_recurrences import EddyRivasFoldingConfig


class SecondaryStructureEnergyModelProtocol(Protocol):
    """
    Defines the interface for a secondary structure thermodynamic energy model.

    This protocol specifies the set of methods that any energy model must
    implement to be compatible with the folding engines (Zuker and Eddy-Rivas).
    It ensures that the engines can request energy calculations for various
    structural motifs in a consistent way.
    """
    params: SecondaryStructureEnergies
    temp_k: float

    # --- Core (Zucker) Secondary Structure Operations ---
    def hairpin(self, base_i: int, base_j: int, seq: str, *, temp_k: Optional[float] = None) -> float: ...

    def stack(self, base_i: int, base_j: int, base_k: int, base_l: int, seq: str, *,
              temp_k: Optional[float] = None) -> float: ...

    def internal(self, base_i: int, base_j: int, base_k: int, base_l: int, seq:str, *,
                 temp_k: Optional[float] = None) -> float: ...

    def multiloop(self, branches:int, unpaired_bases: int) -> float: ...

    # --- Pseudoknot (Eddy & Rivas) Secondary Structure Operations ---
    def pk_dangle_hole_left(self, k: int, seq: str, costs: Optional[PseudoknotEnergies] = None) -> float: ...

    def pk_dangle_hole_right(self, l: int, seq: str, costs: Optional[PseudoknotEnergies] = None) -> float: ...

    def pk_dangle_outer_left(self, i: int, seq: str, costs: Optional[PseudoknotEnergies] = None) -> float: ...

    def pk_dangle_outer_right(self, j: int, seq: str, costs: Optional[PseudoknotEnergies] = None) -> float: ...

    def pk_coax_pack(self, i: int, j: int, r: int, k: int, l: int, seq: str,
                     cfg: EddyRivasFoldingConfig,
                     costs: Optional[PseudoknotEnergies] = None,
                     adjacent: bool = False) -> Tuple[float, float]: ...

    # Short-hole penalty (once per seam)
    def pk_short_hole_penalty(self, k: int, l: int, costs: Optional[PseudoknotEnergies] = None) -> float: ...


@dataclass(frozen=True, slots=True)
class SecondaryStructureEnergyModel:
    """
    A concrete implementation of the energy model protocol.

    This class acts as a high-level interface to the underlying energy
    calculation functions. It holds the loaded thermodynamic parameters and the
    reference temperature, and it dispatches calls to the appropriate low-level
    functions that compute the free energy for specific structural motifs.

    Attributes
    ----------
    params : SecondaryStructureEnergies
        An immutable data object containing all the parsed thermodynamic
        parameter tables required by the folding engines.
    temp_k : float
        The reference temperature in Kelvin for free energy calculations.
        Defaults to 310.15 K (37 °C).
    """
    params: SecondaryStructureEnergies
    temp_k: float = 310.15  # 37 °C

    def _temp(self, temp_k: Optional[float]) -> float:
        """
        Returns the override temperature if provided, otherwise the default.

        Parameters
        ----------
        temp_k : int, optional
            Temperature in Kelvin.

        Returns
        ----------
        float
            The temperature value.
        """
        return self.temp_k if temp_k is None else temp_k

    def hairpin(self, base_i: int, base_j: int, seq: str, *, temp_k: Optional[float] = None) -> float:
        """
       Calculates the free energy of a hairpin loop closed by the pair (i, j).

       Parameters
       ----------
       base_i : int
           The 5' index of the closing base pair.
       base_j : int
           The 3' index of the closing base pair.
       seq : str
           The RNA sequence.
       temp_k : Optional[float], optional
           An optional override for the temperature in Kelvin. If None, the
           model's default temperature is used.

       Returns
       -------
       float
           The calculated free energy (ΔG) in kcal/mol.
       """
        return hairpin_energy(base_i, base_j, seq, self.params, self._temp(temp_k))

    def stack(self, base_i: int, base_j: int, base_k: int, base_l: int, seq: str, *,
              temp_k: Optional[float] = None) -> float:
        """
        Calculates the free energy of a base pair (i, j) stacking on an adjacent pair (k, l).

        Parameters
        ----------
        base_i : int
            The 5' index of the outer base pair.
        base_j : int
            The 3' index of the outer base pair.
        base_k : int
            The 5' index of the inner base pair.
        base_l : int
            The 3' index of the inner base pair.
        seq : str
            The RNA sequence.
        temp_k : Optional[float], optional
            An optional override for the temperature in Kelvin.

        Returns
        -------
        float
            The calculated stacking energy (ΔG) in kcal/mol.
        """
        return stack_energy(base_i, base_j, base_k, base_l, seq, self.params, self._temp(temp_k))

    def internal(self, base_i: int, base_j: int, base_k: int, base_l: int, seq: str, *,
                 temp_k: Optional[float] = None) -> float:
        """
        Calculates the free energy of an internal loop.

        This function handles symmetric and asymmetric internal loops as well as bulges,
        enclosed by an outer pair (i, j) and an inner pair (k, l).

        Parameters
        ----------
        base_i : int
            The 5' index of the outer base pair.
        base_j : int
            The 3' index of the outer base pair.
        base_k : int
            The 5' index of the inner base pair.
        base_l : int
            The 3' index of the inner base pair.
        seq : str
            The RNA sequence.
        temp_k : Optional[float], optional
            An optional override for the temperature in Kelvin.

        Returns
        -------
        float
            The calculated loop energy (ΔG) in kcal/mol.
        """
        return internal_loop_energy(base_i, base_j, base_k, base_l, seq, self.params, self._temp(temp_k))

    def multiloop(self, branches: int, unpaired_bases: int) -> float:
        """
        Calculates the free energy of a multiloop based on a linear model.

        The energy is a function of a fixed penalty, the number of branching
        helices, and the number of unpaired nucleotides within the loop.

        Parameters
        ----------
        branches : int
            The number of helices branching from the multiloop.
        unpaired_bases : int
            The number of unpaired nucleotides in the multiloop.

        Returns
        -------
        float
            The calculated multiloop energy (ΔG) in kcal/mol.
        """
        return multiloop_linear_energy(branches, unpaired_bases, self.params)

    def pk_dangle_hole_left(self, k: int, seq: str, costs: Optional[PseudoknotEnergies] = None) -> float:
        """
        Calculates the energy of a 5' dangle inside a pseudoknot hole.

        Parameters
        ----------
        k : int
            The index of the 5' base of the inner helix adjacent to the dangle.
        seq : str
            The RNA sequence.
        costs : Optional[PseudoknotEnergies], optional
            An optional override for the pseudoknot energy parameters.

        Returns
        -------
        float
            The dangling end energy contribution in kcal/mol.
        """
        return pk_dangle_hole_left_fn(seq, k, self._get_pk_params(costs))

    def pk_dangle_hole_right(self, l: int, seq: str, costs: Optional[PseudoknotEnergies] = None) -> float:
        """
        Calculates the energy of a 3' dangle inside a pseudoknot hole.

        Parameters
        ----------
        l : int
            The index of the 3' base of the inner helix adjacent to the dangle.
        seq : str
            The RNA sequence.
        costs : Optional[PseudoknotEnergies], optional
            An optional override for the pseudoknot energy parameters.

        Returns
        -------
        float
            The dangling end energy contribution in kcal/mol.
        """
        return pk_dangle_hole_right_fn(seq, l, self._get_pk_params(costs))

    def pk_dangle_outer_left(self, i: int, seq: str, costs: Optional[PseudoknotEnergies] = None) -> float:
        """
        Calculates the energy of a 5' dangle on the outer span of a pseudoknot.

        Parameters
        ----------
        i : int
            The index of the 5' base of the outer helix adjacent to the dangle.
        seq : str
            The RNA sequence.
        costs : Optional[PseudoknotEnergies], optional
            An optional override for the pseudoknot energy parameters.

        Returns
        -------
        float
            The dangling end energy contribution in kcal/mol.
        """
        return pk_dangle_outer_left_fn(seq, i, self._get_pk_params(costs))

    def pk_dangle_outer_right(self, j: int, seq: str, costs: Optional[PseudoknotEnergies] = None) -> float:
        """
        Calculates the energy of a 3' dangle on the outer span of a pseudoknot.

        Parameters
        ----------
        j : int
            The index of the 3' base of the outer helix adjacent to the dangle.
        seq : str
            The RNA sequence.
        costs : Optional[PseudoknotEnergies], optional
            An optional override for the pseudoknot energy parameters.

        Returns
        -------
        float
            The dangling end energy contribution in kcal/mol.
        """
        return pk_dangle_outer_right_fn(seq, j, self._get_pk_params(costs))

    def pk_coax_pack(self, i: int, j: int, r: int, k: int, l: int, seq: str,
                     cfg: EddyRivasFoldingConfig,
                     costs: Optional[PseudoknotEnergies] = None,
                     adjacent: bool = False) -> Tuple[float, float]:
        """
        Calculates the coaxial stacking energy for two helices in a pseudoknot context.

        Parameters
        ----------
        i, j : int
            Indices of the outer closing pair of the multiloop context.
        r : int
            The split point between the two stacking helices.
        k, l : int
            Indices of the inner hole.
        seq : str
            The RNA sequence.
        cfg : EddyRivasFoldingConfig
            The folding configuration object.
        costs : Optional[PseudoknotEnergies], optional
            An optional override for the pseudoknot energy parameters.
        adjacent : bool, optional
            True if the helices are flush against each other, by default False.

        Returns
        -------
        Tuple[float, float]
            A tuple containing `(coax_total, coax_bonus)`, representing the total
            coaxial stacking energy and any additional bonus term.
        """
        return pk_coax_pack_fn(seq, i, j, r, k, l, cfg, self._get_pk_params(costs), adjacent)

    def pk_short_hole_penalty(self, k: int, l: int, costs: Optional[PseudoknotEnergies] = None) -> float:
        """
        Calculates the penalty for a short loop (hole) between pseudoknot helices.

        Parameters
        ----------
        k : int
            The 5' index of the inner hole.
        l : int
            The 3' index of the inner hole.
        costs : Optional[PseudoknotEnergies], optional
            An optional override for the pseudoknot energy parameters.

        Returns
        -------
        float
            The penalty energy in kcal/mol.
        """
        return pk_short_hole_penalty_fn(self._get_pk_params(costs), k, l)

    def _get_pk_params(self, costs: Optional[PseudoknotEnergies]) -> PseudoknotEnergies:
        """
        A helper to retrieve the correct set of pseudoknot energy parameters.

        It returns the `costs_override` if provided, otherwise it falls back to the
        pseudoknot parameters loaded with the main energy model.
        """
        if costs is not None:
            return costs
        pk = self.params.PSEUDOKNOT
        if pk is None:
            raise ValueError("Pseudoknot parameters are not loaded in self.params.PSEUDOKNOT.")

        return pk
