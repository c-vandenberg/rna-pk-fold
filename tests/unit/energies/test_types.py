"""
Unit tests for the energy parameter dataclasses.

This module verifies the structural properties and helper methods of the
`SecondaryStructureEnergies` and `PseudoknotEnergies` dataclasses. The tests
ensure that these classes are immutable and memory-efficient, and that their
helper methods, like the Gibbs free energy calculation, are correct.
"""

import math
import pytest
from dataclasses import FrozenInstanceError

from rna_pk_fold.energies.energy_types import (
    SecondaryStructureEnergies,
    PseudoknotEnergies,
)


def test_secondary_structure_energies_is_frozen_and_slotted():
    """
    Validates the `SecondaryStructureEnergies` dataclass is immutable and uses slots.

    This test confirms two important behaviors:
    1.  `frozen=True`: The energy parameters, once loaded, cannot be accidentally
        changed at runtime. This prevents hard-to-debug errors.
    2.  `slots=True`: The class uses `__slots__` instead of `__dict__`, which
        reduces its memory footprint and can speed up attribute access.
    """
    energies = SecondaryStructureEnergies(
        BULGE={1: (10.6, 21.9)},
        COMPLEMENT_BASES={"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"},
        DANGLES={"A./UA": (-0.5, -0.6)},
        HAIRPIN={3: (1.3, -13.2)},
        MULTILOOP=(2.5, 0.1, 0.4, 2.0),
        INTERNAL={4: (-7.2, -26.8)},
        NN_STACK={"AU/UA": (-7.7, -20.6)},
        INTERNAL_MISMATCH={"UU/AG": (-12.8, -37.1)},
        TERMINAL_MISMATCH={"UA/UA": (0.0, 0.0)},
        # Optional fields default to None, so they are omitted here.
    )

    # Test for immutability: attempting to change a field should raise an error.
    with pytest.raises(FrozenInstanceError):
        energies.BULGE = {}

    # Test for memory optimization (`slots=True`).
    # A slotted class should have a `__slots__` attribute but not a `__dict__`.
    assert hasattr(energies, "__slots__")
    assert not hasattr(energies, "__dict__")
    # Attempting to add a new, undeclared attribute should fail.
    with pytest.raises((AttributeError, TypeError)):
        setattr(energies, "new_field", 123)


def test_delta_g_helper_matches_formula():
    """
    Ensures the `delta_g` helper correctly calculates Gibbs free energy.

    The static method must match the standard thermodynamic formula:
    ΔG = ΔH - T * ΔS. The test uses typical values for enthalpy (ΔH) and
    entropy (ΔS) and a standard biological temperature.
    """
    # Sample thermodynamic parameters: ΔH in kcal/mol, ΔS in cal/(K·mol).
    dh, ds = (-7.7, -20.6)
    # Standard biological temperature of 37 °C, converted to Kelvin.
    T = 310.15

    # The formula requires consistent units, so ΔS is converted from cal to kcal.
    expected = dh - T * (ds / 1000.0)
    got = SecondaryStructureEnergies.delta_g(dh, ds, T)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_delta_g_with_none_returns_inf():
    """
    Tests the guard behavior of `delta_g` for missing input.

    If either enthalpy (ΔH) or entropy (ΔS) data is missing (i.e., `None`),
    the free energy is undefined. Returning positive infinity is a standard way
    to represent an energetically impossible state, effectively penalizing
    any structure with incomplete data in a folding algorithm.
    """
    T = 310.15
    # If ΔH is None, the result should be infinity.
    assert math.isinf(SecondaryStructureEnergies.delta_g(None, -1.0, T))
    # If ΔS is None, the result should also be infinity.
    assert math.isinf(SecondaryStructureEnergies.delta_g(-1.0, None, T))


def test_pseudoknot_energies_is_frozen_slotted_and_defaults():
    """
    Validates the `PseudoknotEnergies` dataclass for immutability, slots, and defaults.

    Similar to the `SecondaryStructureEnergies` test, this confirms that the
    pseudoknot energy dataclass is also immutable and memory-efficient. It also
    spot-checks that optional parameters correctly fall back to their default
    values when not provided during instantiation.
    """
    pk = PseudoknotEnergies(
        q_ss=0.2,
        p_tilde_out=1.0,
        p_tilde_hole=1.0,
        q_tilde_out=0.2,
        q_tilde_hole=0.2,
        l_tilde=0.0,
        r_tilde=0.0,
        m_tilde_yhx=0.0,
        m_tilde_vhx=0.0,
        m_tilde_whx=0.0,
        # Optional maps and other parameters are omitted to test their defaults.
    )

    # Test for immutability (`frozen=True`).
    with pytest.raises(FrozenInstanceError):
        pk.q_ss = 0.0

    # Test for memory optimization (`slots=True`).
    assert hasattr(pk, "__slots__")
    assert not hasattr(pk, "__dict__")

    # Spot-check that several optional fields have their expected default values.
    assert pk.coax_pairs is None
    assert pk.coax_min_helix_len == 1
    assert pk.pk_penalty_gw == 1.0
