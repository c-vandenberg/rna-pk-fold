import math
import pytest
from dataclasses import FrozenInstanceError

from rna_pk_fold.thermo.types import SecondaryStructureEnergies


def test_secondary_structure_energies_is_frozen_and_slotted():
    energies = SecondaryStructureEnergies(
        BULGE={1: (10.6, 21.9)},
        COMPLEMENT_BASES={"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"},
        DANGLES={"A./UA": (-0.5, -0.6)},
        HAIRPIN={3: (1.3, -13.2)},
        MULTILOOP=(2.5, 0.1, 0.4, 2.0),
        INTERNAL={4: (-7.2, -26.8)},
        INTERNAL_MM={"UU/AG": (-12.8, -37.1)},
        NN={"AU/UA": (-7.7, -20.6)},
        TERMINAL_MM={"UA/UA": (0.0, 0.0)},
        SPECIAL_HAIRPINS=None,
    )

    with pytest.raises(FrozenInstanceError, AttributeError) as e:
        energies.BULGE = {}

    assert "read-only" in str(e.value).lower()

    # `slots=True`. Therefore, __slots__ should exist and __dict__ should not
    assert hasattr(energies, "__slots__")
    assert not hasattr(energies, "__dict__")


def test_delta_g_helper_matches_formula():
    # ΔG = ΔH - T * (ΔS/1000)
    delta_h, delta_s = (-7.7, -20.6)         # kcal/mol, cal/(K·mol)
    temp = 310.15                            # 37°C in Kelvin
    expected = delta_h - temp * (delta_s / 1000.0)

    got = SecondaryStructureEnergies.delta_g(delta_h, delta_s, temp)
    assert math.isclose(got, expected, rel_tol=1e-12)
