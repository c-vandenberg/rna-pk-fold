import logging
from typing import Optional

from importlib.resources import files as importlib_files

from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModel
from rna_pk_fold.energies.energy_loader import SecondaryStructureEnergyLoader

logger = logging.getLogger(__name__)


def load_energy_model(yaml_path: Optional[str], temp_c: float) -> SecondaryStructureEnergyModel:
    """
    Loads the RNA thermodynamic parameters and creates an energy model.

    If no YAML file path is provided, it loads the default parameters bundled
    with the package.

    Parameters
    ----------
    yaml_path : Optional[str]
        The file path to the energy parameter YAML file.
    temp_c : float
        The temperature in Celsius for the energy calculations.

    Returns
    -------
    SecondaryStructureEnergyModel
        An initialized energy model object ready for use by the folding engines.
    """
    # Use the default bundled parameter file if no path is provided.
    if yaml_path is None:
        yaml_path = str(importlib_files("rna_pk_fold") / "data" / "turner2004_eddyrivas1999_min.yaml")

    logger.info(f"Loading energy model from: {yaml_path}")
    temp_k = 273.15 + temp_c
    logger.info(f"Temperature: {temp_c}°C ({temp_k:.2f}K)")

    # Load the raw parameters from the YAML file.
    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=yaml_path)

    # Create the energy model instance with the loaded parameters and specified temperature.
    model = SecondaryStructureEnergyModel(params=params, temp_k=temp_k)

    logger.debug("Energy model loaded successfully")

    return model