import math
import pytest
from importlib.resources import files as ir_files

import rna_pk_fold
from rna_pk_fold.energies.energy_loader import SecondaryStructureEnergyLoader
from rna_pk_fold.energies.energy_types import SecondaryStructureEnergies


@pytest.fixture(scope="module")
def yaml_path() -> str:
    return ir_files(rna_pk_fold) / "data" / "xia1998_zucker_turner1999_min.yaml"


def test_load_returns_rna_bundle(yaml_path):
    """
    Verify that `load("RNA")` returns a populated RNA energy bundle.

    Expected
    --------
    - Return value is a `SecondaryStructureEnergies` instance.
    - NN_STACK, HAIRPIN, BULGE, INTERNAL are present and non-empty dicts.
    - COMPLEMENT_BASES is a dict.
    - MULTILOOP is a 4-tuple.
    """
    bundle = SecondaryStructureEnergyLoader().load("RNA", yaml_path=yaml_path)
    assert isinstance(bundle, SecondaryStructureEnergies)

    # Core tables present and non-empty where expected
    assert bundle.NN_STACK and isinstance(bundle.NN_STACK, dict)
    assert bundle.HAIRPIN and isinstance(bundle.HAIRPIN, dict)
    assert bundle.BULGE and isinstance(bundle.BULGE, dict)
    assert bundle.INTERNAL and isinstance(bundle.INTERNAL, dict)
    assert isinstance(bundle.COMPLEMENT_BASES, dict)
    assert isinstance(bundle.MULTILOOP, tuple) and len(bundle.MULTILOOP) == 4


def test_only_rna_supported(yaml_path):
    """
    Ensure attempting to load DNA currently raises a ValueError.
    """
    with pytest.raises(ValueError):
        SecondaryStructureEnergyLoader().load("DNA", yaml_path=yaml_path)


def test_complement_map_has_expected_pairs_and_N(yaml_path):
    """
    Validate canonical RNA complements and the 'N' ambiguity mapping.
    """
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)
    complement_bases = bundle.COMPLEMENT_BASES
    for k, v in [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("N", "N")]:
        assert complement_bases[k] == v


def test_nn_contains_expected_entries_and_types(yaml_path):
    """
    Check representative nearest-neighbor entries and value types.

    Notes
    -----
    Keep the set conservative to avoid YAML sparsity issues.
    """
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)
    nn = bundle.NN_STACK

    expected_pairs = [
        "AU/UA", "UA/AU", "GC/CG", "CG/GC",  # canonical
        "GU/UG", "UG/GU",                    # wobble
    ]
    for key in expected_pairs:
        assert key in nn, f"Missing NN key: {key}"
        dh, ds = nn[key]
        assert isinstance(dh, (float, int))
        assert isinstance(ds, (float, int))


def test_terminal_and_dangles_present(yaml_path):
    """
    Confirm presence of terminal mismatch and dangling-end parameters.
    """
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)

    # Terminal mismatches
    assert "UA/UA" in bundle.TERMINAL_MISMATCH

    # Dangles keys include dots as per convention
    assert any("." in base_key for base_key in bundle.DANGLES.keys())


def test_delta_g_at_37C_matches_known_stack_within_tolerance(yaml_path):
    """
    Compute ΔG at 37 °C for a known stack and compare to an expected value.

    Uses the relation ΔG = ΔH - T * (ΔS / 1000) with T = 310.15 K.
    """
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)
    temp = 310.15  # 37°C in K
    delta_h, delta_s = bundle.NN_STACK["AU/UA"]
    delta_g = bundle.delta_g(delta_h, delta_s, temp)

    # Expected Turner/Xia value at 37°C
    assert math.isclose(delta_g, -1.30, abs_tol=0.05)


def test_multiloop_coefficients_shape_and_values(yaml_path):
    """
    Validate multiloop coefficient tuple shape and numeric types.
    """
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)
    a, c1, c2, d = bundle.MULTILOOP
    for x in (a, c1, c2, d):
        assert isinstance(x, (float, int))


def test_special_hairpins_default_none(yaml_path):
    """
    SPECIAL_HAIRPINS should default to None in the base parameter set.
    """
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)
    assert not bundle.SPECIAL_HAIRPINS


def test_subclass_override_of_build(yaml_path):
    """
    Demonstrate subclass customization via `_build_rna()` override.

    Expected
    --------
    - `MyLoader().load("RNA", yaml_path)` returns a bundle whose `MULTILOOP`
      is `(9.9, 9.9, 9.9, 9.9)`.
    """
    class MyLoader(SecondaryStructureEnergyLoader):
        def _build_rna(self, yaml_path):
            base = super()._build_rna(yaml_path)

            # Return a modified copy with a distinct value to prove override is used
            return SecondaryStructureEnergies(
                BULGE=base.BULGE,
                COMPLEMENT_BASES=base.COMPLEMENT_BASES,
                DANGLES=base.DANGLES,
                HAIRPIN=base.HAIRPIN,
                MULTILOOP=(9.9, 9.9, 9.9, 9.9),  # changed
                INTERNAL=base.INTERNAL,
                NN_STACK=base.NN_STACK,
                INTERNAL_MISMATCH=base.INTERNAL_MISMATCH,
                TERMINAL_MISMATCH=base.TERMINAL_MISMATCH,
                HAIRPIN_MISMATCH=base.HAIRPIN_MISMATCH,
                MULTI_MISMATCH=base.MULTI_MISMATCH,
                SPECIAL_HAIRPINS=base.SPECIAL_HAIRPINS,
                PSEUDOKNOT=base.PSEUDOKNOT,
            )

    b = MyLoader().load("RNA", yaml_path=yaml_path)
    assert b.MULTILOOP == (9.9, 9.9, 9.9, 9.9)

