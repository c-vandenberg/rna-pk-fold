"""
Tests for the SecondaryStructureEnergyLoader to ensure correct parsing and
structuring of thermodynamic parameters from YAML files.
"""
from __future__ import annotations

# --- Standard Library Imports ---
import math
from importlib.resources import files as ir_files

# --- Third-Party Imports ---
import pytest

# --- Local Application Imports ---
import rna_pk_fold
from rna_pk_fold.energies.energy_loader import SecondaryStructureEnergyLoader
from rna_pk_fold.energies.energy_types import SecondaryStructureEnergies


@pytest.fixture(scope="module")
def yaml_path() -> str:
    """
    A pytest fixture that provides the file path to the minimal test energy parameter YAML file.

    This fixture is scoped to the module level, so the path is determined only
    once per test file run.

    Returns
    -------
    str
        The absolute file path to the test YAML data file.
    """
    # Locate the test parameter file within the installed package's data directory.
    return str(ir_files(rna_pk_fold) / "data" / "xia1998_zucker_turner1999_min.yaml")


def test_load_returns_rna_bundle(yaml_path):
    """
    Tests that loading "RNA" parameters returns a correctly structured object.

    This test verifies that the `SecondaryStructureEnergyLoader` successfully
    instantiates a `SecondaryStructureEnergies` object and that its core
    attributes are populated with the expected data types (e.g., dicts, tuples).

    Parameters
    ----------
    yaml_path : str
        The path to the energy parameter file, provided by the pytest fixture.
    """
    # Load the energy parameters from the specified YAML file.
    bundle = SecondaryStructureEnergyLoader().load("RNA", yaml_path=yaml_path)
    # Assert that the loaded object is of the correct dataclass type.
    assert isinstance(bundle, SecondaryStructureEnergies)

    # Assert that the core thermodynamic tables are present and have the correct type.
    assert bundle.NN_STACK and isinstance(bundle.NN_STACK, dict)
    assert bundle.HAIRPIN and isinstance(bundle.HAIRPIN, dict)
    assert bundle.BULGE and isinstance(bundle.BULGE, dict)
    assert bundle.INTERNAL and isinstance(bundle.INTERNAL, dict)
    assert isinstance(bundle.COMPLEMENT_BASES, dict)
    # Assert that the multiloop parameters are a tuple of four numeric coefficients.
    assert isinstance(bundle.MULTILOOP, tuple) and len(bundle.MULTILOOP) == 4


def test_only_rna_supported(yaml_path):
    """
    Tests that the loader raises a ValueError for unsupported nucleic acid types.

    Currently, the implementation only supports "RNA". This test ensures that
    attempting to load parameters for "DNA" or any other type correctly fails.

    Parameters
    ----------
    yaml_path : str
        The path to the energy parameter file, provided by the pytest fixture.
    """
    # Use pytest.raises to assert that a ValueError is thrown when loading "DNA".
    with pytest.raises(ValueError):
        SecondaryStructureEnergyLoader().load("DNA", yaml_path=yaml_path)


def test_complement_map_has_expected_pairs_and_N(yaml_path):
    """
    Tests that the loaded complement map contains the correct canonical RNA pairs.

    This test validates that the `COMPLEMENT_BASES` mapping correctly defines
    the standard Watson-Crick pairs (A-U, G-C) and the self-complementary
    ambiguity code 'N'.

    Parameters
    ----------
    yaml_path : str
        The path to the energy parameter file, provided by the pytest fixture.
    """
    # Load the energy parameters.
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)
    complement_bases = bundle.COMPLEMENT_BASES
    # Check for the presence and correctness of all expected key-value pairs.
    for k, v in [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("N", "N")]:
        assert complement_bases[k] == v


def test_nn_contains_expected_entries_and_types(yaml_path):
    """
    Tests that the nearest-neighbor stacking table contains representative entries.

    This test verifies that the `NN_STACK` table is populated and checks for a
    few key stacking interactions (canonical and wobble pairs). It also ensures
    that the energy values are stored as tuples of numbers (ΔH, ΔS).

    Parameters
    ----------
    yaml_path : str
        The path to the energy parameter file, provided by the pytest fixture.
    """
    # Load the energy parameters.
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)
    nn = bundle.NN_STACK

    # Define a list of essential stacking keys that must be present.
    expected_pairs = [
        "AU/UA", "UA/AU", "GC/CG", "CG/GC",  # Canonical Watson-Crick stacks
        "GU/UG", "UG/GU",                    # Wobble pair stacks
    ]
    # Check that each expected key exists in the table.
    for key in expected_pairs:
        assert key in nn, f"Missing NN key: {key}"
        # Verify that the value is a tuple of two numbers (float or int).
        dh, ds = nn[key]
        assert isinstance(dh, (float, int))
        assert isinstance(ds, (float, int))


def test_terminal_and_dangles_present(yaml_path):
    """
    Tests for the presence of terminal mismatch and dangling end parameter tables.

    Parameters
    ----------
    yaml_path : str
        The path to the energy parameter file, provided by the pytest fixture.
    """
    # Load the energy parameters.
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)

    # Check for a representative key in the terminal mismatch table.
    assert "UA/UA" in bundle.TERMINAL_MISMATCH

    # Check that the dangling ends table uses the conventional key format (containing a dot).
    assert any("." in base_key for base_key in bundle.DANGLES.keys())


def test_delta_g_at_37C_matches_known_stack_within_tolerance(yaml_path):
    """
    Tests the ΔG calculation for a known stacking interaction at 37 °C.

    This test extracts the ΔH and ΔS for a specific stack (`AU/UA`), calculates
    the free energy (ΔG) at 310.15 K, and asserts that the result is close to
    the published Turner/Xia value.

    Parameters
    ----------
    yaml_path : str
        The path to the energy parameter file, provided by the pytest fixture.
    """
    # Load the energy parameters.
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)
    # Define the standard temperature for biological systems.
    temp = 310.15  # 37°C in Kelvin
    # Get the enthalpy and entropy for a known stacking interaction.
    delta_h, delta_s = bundle.NN_STACK["AU/UA"]
    # Calculate the free energy using the model's static method.
    delta_g = bundle.delta_g(delta_h, delta_s, temp)

    # Assert that the calculated ΔG is within a small tolerance of the expected value.
    assert math.isclose(delta_g, -1.30, abs_tol=0.05)


def test_multiloop_coefficients_shape_and_values(yaml_path):
    """
    Tests that the multiloop parameters are a correctly formatted 4-tuple of numbers.

    Parameters
    ----------
    yaml_path : str
        The path to the energy parameter file, provided by the pytest fixture.
    """
    # Load the energy parameters.
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)
    # Unpack the four coefficients of the linear multiloop model.
    a, c1, c2, d = bundle.MULTILOOP
    # Verify that each coefficient is a numeric type.
    for x in (a, c1, c2, d):
        assert isinstance(x, (float, int))


def test_special_hairpins_default_none(yaml_path):
    """
    Tests that the `SPECIAL_HAIRPINS` table is None in the minimal parameter set.

    This confirms that optional tables are correctly parsed as None when they are
    not present in the YAML file.

    Parameters
    ----------
    yaml_path : str
        The path to the energy parameter file, provided by the pytest fixture.
    """
    # Load the energy parameters from the minimal YAML file.
    bundle = SecondaryStructureEnergyLoader().load(yaml_path=yaml_path)
    # The minimal file does not contain special hairpins, so the attribute should be None.
    assert not bundle.SPECIAL_HAIRPINS


def test_subclass_override_of_build(yaml_path):
    """
    Tests that subclassing `SecondaryStructureEnergyLoader` to override `_build_rna` works as expected.

    This test demonstrates a customization pattern where a user can create a
    subclass to modify the loading process, for example, to inject custom or
    altered parameters.

    Parameters
    ----------
    yaml_path : str
        The path to the energy parameter file, provided by the pytest fixture.
    """
    # Define a custom loader that inherits from the base loader.
    class MyLoader(SecondaryStructureEnergyLoader):
        # Override the method responsible for building the RNA energy bundle.
        def _build_rna(self, yaml_path):
            # First, call the parent method to get the base-loaded parameters.
            base = super()._build_rna(yaml_path)

            # Create a new, modified copy of the bundle, replacing the MULTILOOP parameters.
            return SecondaryStructureEnergies(
                BULGE=base.BULGE,
                COMPLEMENT_BASES=base.COMPLEMENT_BASES,
                DANGLES=base.DANGLES,
                HAIRPIN=base.HAIRPIN,
                MULTILOOP=(9.9, 9.9, 9.9, 9.9),  # The overridden value.
                INTERNAL=base.INTERNAL,
                NN_STACK=base.NN_STACK,
                INTERNAL_MISMATCH=base.INTERNAL_MISMATCH,
                TERMINAL_MISMATCH=base.TERMINAL_MISMATCH,
                HAIRPIN_MISMATCH=base.HAIRPIN_MISMATCH,
                MULTI_MISMATCH=base.MULTI_MISMATCH,
                SPECIAL_HAIRPINS=base.SPECIAL_HAIRPINS,
                PSEUDOKNOT=base.PSEUDOKNOT,
            )

    # Instantiate the custom loader and load the parameters.
    b = MyLoader().load("RNA", yaml_path=yaml_path)
    # Assert that the MULTILOOP attribute has the new, overridden value.
    assert b.MULTILOOP == (9.9, 9.9, 9.9, 9.9)

