import math
import pytest
from dataclasses import FrozenInstanceError

from rna_pk_fold.energies.energy_types import (
    SecondaryStructureEnergies,
    PseudoknotEnergies,
)


def test_secondary_structure_energies_is_frozen_and_slotted():
    """
    Validate dataclass immutability and slotted behavior for the refactored fields.
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
        # Optional fields default to None: HAIRPIN_MISMATCH, MULTI_MISMATCH, SPECIAL_HAIRPINS, PSEUDOKNOT
    )

    # Frozen: changing a field should fail
    with pytest.raises(FrozenInstanceError):
        energies.BULGE = {}

    # Slotted: has __slots__, no __dict__, adding new attribute should fail
    assert hasattr(energies, "__slots__")
    assert not hasattr(energies, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        setattr(energies, "new_field", 123)


def test_delta_g_helper_matches_formula():
    """
    ΔG helper must match the analytical formula: ΔG = ΔH - T * (ΔS / 1000).
    """
    dh, ds = (-7.7, -20.6)      # kcal/mol, cal/(K·mol)
    T = 310.15                  # 37 °C in Kelvin
    expected = dh - T * (ds / 1000.0)
    got = SecondaryStructureEnergies.delta_g(dh, ds, T)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_delta_g_with_none_returns_inf():
    """
    Guard behavior: if either ΔH or ΔS is None, delta_g returns +∞.
    """
    T = 310.15
    assert math.isinf(SecondaryStructureEnergies.delta_g(None, -1.0, T))
    assert math.isinf(SecondaryStructureEnergies.delta_g(-1.0, None, T))


def test_pseudoknot_energies_is_frozen_slotted_and_defaults():
    """
    Validate immutability/slots and a couple of defaults on PseudoknotEnergies.
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
        # optional maps omitted → default None
        # coax_* controls & others use their defaults
    )

    # Frozen & slotted
    with pytest.raises(FrozenInstanceError):
        pk.q_ss = 0.0
    assert hasattr(pk, "__slots__")
    assert not hasattr(pk, "__dict__")

    # Defaults: optional maps None; some scalar defaults
    assert pk.coax_pairs is None
    assert pk.coax_min_helix_len == 1
    assert pk.pk_penalty_gw == 1.0
