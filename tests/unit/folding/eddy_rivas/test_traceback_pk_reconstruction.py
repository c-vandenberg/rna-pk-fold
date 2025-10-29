import pytest
from rna_pk_fold.scripts.predict_rna import predict_eddy_rivas_non_nested
from rna_pk_fold.utils.energy.energy_model_utils import load_energy_model

SEQUENCE = "AGCUUUGAAAGCUUUCGAGUCUGUUUCGAAAUCACAAGGACCU"


def test_pk_reconstruction_shows_nonzero_layer():
    # Load default energy model used by CLI (temperature 37C, default YAML)
    energy_model = load_energy_model(37.0, None)

    dot_bracket, energy = predict_eddy_rivas_non_nested(
        SEQUENCE,
        energy_model,
        enable_coax=True,
        enable_overlap=True,
        enable_is2=True,
        enable_join_drift=True,
        enable_strict_complement_order=True,
    )

    # The reconstructed structure should include at least one non-parenthesis
    # bracket type, indicating a pseudoknot layer (e.g., '[', '{', '<').
    assert any(ch in dot_bracket for ch in "[]{}<>"), (
        "Expected at least one pseudoknot layer bracket in the multilayer dot-bracket,"
        f" got: {dot_bracket}"
    )

    # Also sanity-check that energy is finite and negative (typical for folded RNA)
    assert energy is not None and float(energy) < 0.0

