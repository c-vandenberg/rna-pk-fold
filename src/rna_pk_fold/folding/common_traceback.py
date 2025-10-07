from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

from rna_pk_fold.structures import Pair  # single canonical Pair

# Layers → bracket glyphs
BRACKETS: List[Tuple[str, str]] = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]


@dataclass(frozen=True, slots=True)
class TraceResult:
    """Minimal result type used by both engines."""
    pairs: List[Pair]
    dot_bracket: str


def pairs_to_dotbracket(seq_len: int, pairs: List[Pair]) -> str:
    """Render plain (single-layer) dot-bracket for nested pairs."""
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
    """Render layered dot-bracket using BRACKETS by layer; unpaired are '.'."""
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
    """Parse '(' ... ')' dot-bracket into a set of (i, j) base pairs."""
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