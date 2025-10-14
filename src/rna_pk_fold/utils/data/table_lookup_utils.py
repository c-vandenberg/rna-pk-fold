from __future__ import annotations
from typing import Dict, Tuple, Optional


def table_lookup(
    table: Dict[Tuple[str, str], float],
    key_x: Optional[str],
    key_y: Optional[str],
    default_value: float,
    none_value: float = 0.0
) -> float:
    """
    Performs a safe lookup in a dictionary keyed by a tuple of strings.

    This function is a robust wrapper for dictionary lookups where the keys
    might be `None` (e.g., due to an out-of-bounds sequence access). It provides
    a specific return value for `None` inputs and a different default for
    keys that are not found in the dictionary.

    Parameters
    ----------
    table : Dict[Tuple[str, str], float]
        The dictionary to perform the lookup in.
    key_x : Optional[str]
        The first part of the tuple key.
    key_y : Optional[str]
        The second part of the tuple key.
    default_value : float
        The value to return if the `(key_x, key_y)` tuple is not found in the table.
    none_value : float, optional
        The value to return if either `key_x` or `key_y` is `None`, by default 0.0.

    Returns
    -------
    float
        The looked-up value, the default value, or the none value.
    """
    if key_x is None or key_y is None:
        return none_value

    return table.get((key_x, key_y), default_value)
