from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

from rna_pk_fold.structures import Pair  # single canonical Pair

# Layers → bracket glyphs
BRACKETS: List[Tuple[str, str]] = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]


@dataclass(frozen=True, slots=True)
class TraceResult:
    """
    A standard container for the results of an RNA folding traceback.

    This simple, immutable data structure bundles the two primary representations
    of a secondary structure: a list of base pairs and the corresponding
    dot-bracket string. It is used as a consistent return type by both the
    nested and pseudoknot-aware traceback engines.

    Attributes
    ----------
    pairs : List[Pair]
        A list of `Pair` objects, where each `Pair` represents a canonical
        base pair (i, j) with i < j. The list is typically sorted by the
        5' index `i`.
    dot_bracket : str
        The dot-bracket string representation of the secondary structure.
    """
    pairs: List[Pair]
    dot_bracket: str


def pairs_to_dotbracket(seq_len: int, pairs: List[Pair]) -> str:
    """
    Converts a list of base pairs into a standard (single-layer) dot-bracket string.

    This function generates a simple dot-bracket string where all base pairs are
    represented by parentheses `()` and unpaired bases are represented by dots `.`.
    It does not support pseudoknots or multilayer notation.

    Parameters
    ----------
    seq_len : int
        The total length of the RNA sequence.
    pairs : List[Pair]
        A list of `Pair` objects representing the nested secondary structure.

    Returns
    -------
    str
        The single-layer dot-bracket string representation of the structure.
    """
    chars = ['.'] * seq_len
    for pr in pairs:
        i, j = pr.base_i, pr.base_j
        if 0 <= i < j < seq_len:
            chars[i] = '('
            chars[j] = ')'
    return ''.join(chars)


def pairs_to_multilayer_dotbracket(
    seq_len: int,
    pairs: List[Pair],
    pair_layer: Dict[Tuple[int, int], int],
) -> str:
    """
    Converts a list of pairs and layer assignments into a multilayer dot-bracket string.

    This function renders a dot-bracket string that can represent pseudoknots
    by using different types of brackets for base pairs on different "layers".
    The layer for each pair determines which bracket style from the `BRACKETS`
    list is used.

    Parameters
    ----------
    seq_len : int
        The total length of the RNA sequence.
    pairs : List[Pair]
        A list of `Pair` objects representing the full secondary structure.
    pair_layer : Dict[Tuple[int, int], int]
        A dictionary mapping each pair `(i, j)` to an integer layer index.
        If a pair is not in the dictionary, it defaults to layer 0.

    Returns
    -------
    str
        The multilayer dot-bracket string representation of the structure.
    """
    chars = ['.'] * seq_len
    for pr in pairs:
        i, j = pr.base_i, pr.base_j
        layer = pair_layer.get((i, j), 0)
        br_open, br_close = BRACKETS[layer % len(BRACKETS)]
        if 0 <= i < j < seq_len:
            chars[i] = br_open
            chars[j] = br_close

    return ''.join(chars)


def dotbracket_to_pairs(db: str) -> Set[Tuple[int, int]]:
    """
    Parses a simple, single-layer dot-bracket string into a set of base pairs.

    This function reads a dot-bracket string containing only `(`, `)`, and `.`
    characters and reconstructs the set of base pairs it represents. It does not
    support multilayer/pseudoknotted notation.

    Parameters
    ----------
    db : str
        The single-layer dot-bracket string to parse.

    Returns
    -------
    Set[Tuple[int, int]]
        A set of tuples, where each tuple `(i, j)` represents a base pair.
    """
    stack: List[int] = []
    out: Set[Tuple[int, int]] = set()
    for idx, ch in enumerate(db):
        if ch == '(':
            stack.append(idx)
        elif ch == ')':
            if stack:
                i = stack.pop()
                out.add((i, idx))
    return out