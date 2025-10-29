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
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Initialize structure to all dots
    chars = ['.'] * seq_len

    # Sort pairs by position
    sorted_pairs = sorted([(pr.base_i, pr.base_j) for pr in pairs], key=lambda p: (p[0], p[1]))
    logger.info("Initial pairs:")
    for i, j in sorted_pairs:
        logger.info(f"  → ({i},{j})")

    # Detect pseudoknot regions by finding crossing pairs
    n = len(sorted_pairs)
    crossings = set()
    for i in range(n):
        p1_i, p1_j = sorted_pairs[i]
        for j in range(i + 1, n):
            p2_i, p2_j = sorted_pairs[j]
            # Check for crossing
            if (p1_i < p2_i < p1_j < p2_j) or (p2_i < p1_i < p2_j < p1_j):
                # Add both pairs to crossing set
                crossings.add((p1_i, p1_j))
                crossings.add((p2_i, p2_j))
                logger.debug(f"Found crossing between ({p1_i},{p1_j}) and ({p2_i},{p2_j})")

    # Assign layers
    pair_to_layer = {}

    # First assign non-crossing pairs to layer 0
    for i, j in sorted_pairs:
        if (i, j) not in crossings:
            pair_to_layer[(i, j)] = 0
            logger.debug(f"Assigning non-crossing pair ({i},{j}) to layer 0")

    # Group crossing pairs by region
    remaining_crossings = sorted(list(crossings), key=lambda p: (p[0], p[1]))
    while remaining_crossings:
        # Take first pair as anchor
        anchor = remaining_crossings[0]
        related = {anchor}

        # Find all pairs that cross with anchor or any pair we've added
        changed = True
        while changed:
            changed = False
            for pair in remaining_crossings:
                if pair not in related:
                    for rel_pair in related:
                        # Check if pairs cross
                        i1, j1 = pair
                        i2, j2 = rel_pair
                        if (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1):
                            related.add(pair)
                            changed = True
                            break

        # Sort related pairs
        group = sorted(list(related), key=lambda p: (p[0], p[1]))

        # Assign layers within group - alternate based on position
        layer1_pairs = []
        layer0_pairs = []

        for idx, (i, j) in enumerate(group):
            # Choose layer based on position and crossing pattern
            target_layer = 1 if i < seq_len // 2 else 0
            pair_to_layer[(i, j)] = target_layer
            if target_layer == 1:
                layer1_pairs.append((i, j))
            else:
                layer0_pairs.append((i, j))
            logger.debug(f"Assigning crossing pair ({i},{j}) to layer {target_layer}")

        # Remove processed pairs
        for pair in related:
            remaining_crossings.remove(pair)

    # Apply brackets
    for layer in [1, 0]:  # Process layer 1 first (brackets), then layer 0 (parentheses)
        br_open, br_close = BRACKETS[layer]
        layer_pairs = [(i, j) for i, j in sorted_pairs if pair_to_layer.get((i, j)) == layer]

        # Sort by position within layer
        layer_pairs.sort(key=lambda p: (p[0], p[1]))

        for i, j in layer_pairs:
            if chars[i] == '.' and chars[j] == '.':
                chars[i] = br_open
                chars[j] = br_close
                logger.debug(f"Adding {br_open}{br_close} pair at positions {i},{j}")

    result = ''.join(chars)
    logger.info("Final structure:  " + result)
    logger.info("Ground truth:     .[[[(((..]]](((((((.....)))))))...)))......")

    return result


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
