def normalize_base(base_raw: str) -> str:
    """
    Upper-case a nucleotide base and map T->U so RNA logic can be applied uniformly.

    Parameters
    ----------
    base_raw : str
        Raw single-character nucleotide base.

    Returns
    -------
    str
        Normalized base in {A, U, G, C, N}.
    """
    if not isinstance(base_raw, str):
        return base_raw

    if len(base_raw) != 1:
        return base_raw

    base_norm = base_raw.upper()

    return "U" if base_norm == "T" else base_norm
