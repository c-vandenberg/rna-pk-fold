import math
import pytest

from rna_pk_fold.energies import SecondaryStructureEnergyLoader
from rna_pk_fold.energies import SecondaryStructureEnergies


def test_load_returns_rna_bundle():
    """
    Verify that `load("RNA")` returns a populated RNA energy bundle.

    Returns
    -------
    None

    Notes
    -----
    Confirms presence and basic types of core tables used by the DP layer.
    """
    bundle = SecondaryStructureEnergyLoader.load("RNA")
    assert isinstance(bundle, SecondaryStructureEnergies)

    # Core tables present and non-empty where expected
    assert bundle.NN and isinstance(bundle.NN, dict)
    assert bundle.HAIRPIN and isinstance(bundle.HAIRPIN, dict)
    assert bundle.BULGE and isinstance(bundle.BULGE, dict)
    assert bundle.INTERNAL and isinstance(bundle.INTERNAL, dict)
    assert isinstance(bundle.COMPLEMENT_BASES, dict)
    assert isinstance(bundle.MULTILOOP, tuple) and len(bundle.MULTILOOP) == 4


def test_only_rna_supported():
    """
    Ensure attempting to load DNA currently raises a ValueError.
    """
    with pytest.raises(ValueError):
        SecondaryStructureEnergyLoader.load("DNA")


def test_complement_map_has_expected_pairs_and_N():
    """
    Validate canonical RNA complements and the 'N' ambiguity mapping.

    Notes
    -----
    'N' should map to 'N' to preserve ambiguity through complement lookups.
    """
    rna_energies = SecondaryStructureEnergyLoader.load()
    complement_bases = rna_energies.COMPLEMENT_BASES
    for k, v in [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("N", "N")]:
        assert complement_bases[k] == v


def test_nn_contains_expected_entries_and_types():
    """
    Check representative nearest-neighbor entries and value types.

    Notes
    -----
    Each NN entry must exist and provide numeric (ΔH, ΔS) pairs.
    """
    rna_energies = SecondaryStructureEnergyLoader.load()
    nn = rna_energies.NN

    expected_pairs = [
        "AU/UA", "UA/AU", "GC/CG", "CG/GC",
        "GU/CA", "UG/AC",
        "AG/UU", "AU/UG", "CG/GU", "CU/GG",
        "GG/CU", "GU/CG", "GA/UU", "GG/UU",
        "GU/UG", "UG/AU", "UG/GU",
    ]
    for key in expected_pairs:
        assert key in nn, f"Missing NN key: {key}"
        dh, ds = nn[key]
        assert isinstance(dh, (float, int))
        assert isinstance(ds, (float, int))


def test_terminal_and_dangles_present():
    """
    Confirm presence of terminal mismatch and dangling-end parameters.

    Notes
    -----
    Dangles use '.' in keys to denote a single unpaired nucleotide.
    """
    rna_energies = SecondaryStructureEnergyLoader.load()

    # Terminal mismatches
    assert "UA/UA" in rna_energies.TERMINAL_MM

    # Dangles keys include dots as per convention
    assert any("." in base_key for base_key in rna_energies.DANGLES.keys())


def test_delta_g_at_37C_matches_known_stack_within_tolerance():
    """
    Compute ΔG at 37 °C for a known stack and compare to an expected value.

    Notes
    -----
    Uses the relation ΔG = ΔH - T * (ΔS / 1000) with T = 310.15 K.
    """
    rna_energies = SecondaryStructureEnergyLoader.load()
    temp = 310.15  # 37°C in K
    delta_h, delta_s = rna_energies.NN["AU/UA"]
    delta_g = rna_energies.delta_g(delta_h, delta_s, temp)

    assert math.isclose(delta_g, -1.30, abs_tol=0.05)


def test_multiloop_coefficients_shape_and_values():
    """
    Validate multiloop coefficient tuple shape and numeric types.
    """
    rna_energies = SecondaryStructureEnergyLoader.load()
    a, c1, c2, d = rna_energies.MULTILOOP
    for x in (a, c1, c2, d):
        assert isinstance(x, (float, int))


def test_special_hairpins_default_none():
    """
    SPECIAL_HAIRPINS should default to None in the base parameter set.
    """
    rna_energies = SecondaryStructureEnergyLoader.load()
    assert rna_energies.SPECIAL_HAIRPINS is None


def test_class_method_allows_subclass_override_of_build():
    """
    Demonstrate subclass customization via `_build_rna()` override.

    Notes
    -----
    The class method `load` should dispatch to the subclass's static
    `_build_rna()` implementation, allowing project-specific tweaks.
    """
    class MyLoader(SecondaryStructureEnergyLoader):
        @staticmethod
        def _build_rna():
            base_energy_loader = super(MyLoader, MyLoader)._build_rna()

            # Return a modified copy with a distinct value to prove override is used
            return SecondaryStructureEnergies(
                BULGE=base_energy_loader.BULGE,
                COMPLEMENT_BASES=base_energy_loader.COMPLEMENT_BASES,
                DANGLES=base_energy_loader.DANGLES,
                HAIRPIN=base_energy_loader.HAIRPIN,
                MULTILOOP=(9.9, 9.9, 9.9, 9.9),  # changed
                INTERNAL=base_energy_loader.INTERNAL,
                INTERNAL_MM=base_energy_loader.INTERNAL_MM,
                NN=base_energy_loader.NN,
                TERMINAL_MM=base_energy_loader.TERMINAL_MM,
                SPECIAL_HAIRPINS=base_energy_loader.SPECIAL_HAIRPINS,
            )

    b = MyLoader.load("RNA")
    assert b.MULTILOOP == (9.9, 9.9, 9.9, 9.9)
