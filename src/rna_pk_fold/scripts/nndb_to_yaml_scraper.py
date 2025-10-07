#!/usr/bin/env python3
import re
import sys
import itertools as it
from typing import Dict, Tuple, List, Iterable
import requests
from bs4 import BeautifulSoup as BS
import yaml

NNDB_BASE = "https://rna.urmc.rochester.edu/NNDB/"

# Canonical pairs (for a uniform coax table when pair-specific values are absent)
CANONICAL = ["AU","UA","CG","GC","GU","UG"]

# ---------- HTTP helpers ----------

def fetch(url: str) -> BS:
    # Handle Unicode minus and timeouts nicely; raise on HTTP errors.
    r = requests.get(url, timeout=20, headers={"User-Agent": "rna-pk-fold/1.0 (+github)"})
    r.raise_for_status()
    return BS(r.text, "html.parser")

def find_link_by_text(soup: BS, pattern: str) -> str:
    """
    Find first anchor whose text matches `pattern` (case-insensitive).
    Returns absolute URL (NNDB_BASE + href if relative).
    """
    a = soup.find("a", string=re.compile(pattern, re.I))
    if not a or not a.get("href"):
        raise RuntimeError(f"Could not find link matching /{pattern}/i")
    href = a["href"]
    if href.startswith("http"):
        return href
    return NNDB_BASE.rstrip("/") + "/" + href.lstrip("/")

# ---------- Parsing helpers ----------

def _text(soup: BS) -> str:
    # Normalize whitespace; convert Unicode minus to ASCII hyphen for floats.
    txt = soup.get_text("\n", strip=True)
    return txt.replace("−", "-")  # U+2212 → '-'

_float = r"[-+]?\d+(?:\.\d+)?"
FOUR_FLOATS = re.compile(rf"\s*({_float})\s+({_float})\s+({_float})\s+({_float})")

def _extract_4cols(lines: Iterable[str]) -> List[List[float]]:
    """
    From a section containing repeated
      A C G U
      v1 v2 v3 v4
    blocks, collect all 4-tuples (one per block).
    """
    blocks: List[List[float]] = []
    it_lines = iter(lines)
    for line in it_lines:
        if line.strip().upper() == "A C G U":
            # Next non-empty line should be the numbers
            for nums_line in it_lines:
                nums_line = nums_line.strip()
                if not nums_line:
                    continue
                m = FOUR_FLOATS.match(nums_line)
                if m:
                    blocks.append([float(m.group(i)) for i in range(1, 5)])
                break
    return blocks

def _mean_columns(blocks: List[List[float]]) -> List[float]:
    """Average each column across all blocks."""
    if not blocks:
        return [0.0, 0.0, 0.0, 0.0]
    cols = list(zip(*blocks))  # 4 tuples
    return [sum(col)/len(col) for col in cols]

def _expand_to_bigram_map(vec_by_base: List[float], *, side: str) -> Dict[Tuple[str,str], float]:
    """
    Expand a 4-vector (energies for A,C,G,U) into a 4×4 bigram map:
    - side='left'  → use the FIRST base of (X,Y) (the dangling base)
    - side='right' → use the SECOND base of (X,Y) (the dangling base)
    """
    idx = {"A":0, "C":1, "G":2, "U":3}
    bases = ["A","C","G","U"]
    out: Dict[Tuple[str,str], float] = {}
    for x in bases:
        for y in bases:
            key_base = x if side == "left" else y
            out[(x,y)] = vec_by_base[idx[key_base]]
    return out

# ---------- Scrapers ----------

def scrape_dangles() -> Dict[str, Dict[Tuple[str,str], float]]:
    """
    Scrape ΔG°37 dangling end parameters from NNDB:
      NNDB → "Dangling Ends" → "html" parameters page.

    Returns a dict with four bigram maps:
      - dangle_hole_left,  dangle_outer_left  (from 5' tables)
      - dangle_hole_right, dangle_outer_right (from 3' tables)
    """
    # Navigate to the parameter tables
    idx = fetch(NNDB_BASE)
    dang_page = fetch(find_link_by_text(idx, r"Dangling Ends"))
    dang_html = fetch(find_link_by_text(dang_page, r"\bhtml\b"))

    txt = _text(dang_html)
    lines = txt.splitlines()

    # Find ΔG° section bounds
    try:
        g_start = next(i for i,l in enumerate(lines) if l.strip().startswith("# ΔG"))
    except StopIteration:
        raise RuntimeError("Could not find ΔG° section on dangles page.")

    # Locate “## 3' Dangling Ends” and “## 5' Dangling Ends”
    try:
        three_i = next(i for i in range(g_start, len(lines)) if lines[i].strip().startswith("## 3'"))
        five_i  = next(i for i in range(three_i+1, len(lines)) if lines[i].strip().startswith("## 5'"))
    except StopIteration:
        raise RuntimeError("Could not find 3' / 5' subsections on dangles page.")

    three_block = lines[three_i: five_i]
    # End ΔG° at the next major header (“# ΔH°”) or EOF
    try:
        end_i = next(i for i in range(five_i+1, len(lines)) if lines[i].strip().startswith("# ΔH"))
    except StopIteration:
        end_i = len(lines)
    five_block = lines[five_i: end_i]

    # Extract all A C G U → four-float rows and average columns
    three_blocks = _extract_4cols(three_block)
    five_blocks  = _extract_4cols(five_block)

    three_mean = _mean_columns(three_blocks)  # per dangling base
    five_mean  = _mean_columns(five_blocks)

    # Expand to 4×4 maps as explained above
    left_map  = _expand_to_bigram_map(five_mean, side="left")   # 5' dangles
    right_map = _expand_to_bigram_map(three_mean, side="right") # 3' dangles

    return {
        "dangle_hole_left":  left_map,   # (k-1, k)
        "dangle_outer_left": left_map,   # (i, i+1)
        "dangle_hole_right": right_map,  # (l, l+1)
        "dangle_outer_right": right_map, # (j-1, j)
    }

def scrape_coax_defaults() -> Dict[str, float | Dict[str, float]]:
    """
    Scrape the Coaxial Stacking 'html' page and pull ΔG°37 constants.
    These are global constants on the page (not pair-specific).
    """
    idx = fetch(NNDB_BASE)
    coax_page = fetch(find_link_by_text(idx, r"Coaxial Stacking"))
    coax_html = fetch(find_link_by_text(coax_page, r"\bhtml\b"))

    txt = _text(coax_html)

    def grab(pattern: str, default: float) -> float:
        m = re.search(pattern, txt, re.I | re.S)
        return float(m.group(1)) if m else default

    # Heuristics based on NNDB wording; defaults are sensible if not found
    g_flush = grab(r"flush coaxial stacking.*?ΔG°?\s*37\s*=\s*([\-+]?\d+(?:\.\d+)?)", -2.1)
    g_mm1   = grab(r"mismatch.*?ΔG°?\s*37\s*=\s*([\-+]?\d+(?:\.\d+)?)", -0.4)
    g_mm2   = grab(r"\(([\-+]?\d+(?:\.\d+)?)\s*kcal/mol\)\s*\)", -0.2)

    pairs = [f"{x}|{y}" for x,y in it.product(CANONICAL, CANONICAL)]
    coax_pairs_uniform = {p: g_flush for p in pairs}

    return {
        "coax_pairs_uniform": coax_pairs_uniform,
        "flush_dG37": g_flush,
        "mismatch_dG37_major": g_mm1,
        "mismatch_dG37_minor": g_mm2,
    }

# ---------- YAML builder ----------

def build_yaml() -> dict:
    dangles = scrape_dangles()
    coax    = scrape_coax_defaults()

    pseudoknot = {
        # Scalars / tilde terms
        "q_ss": 0.2,
        "P_tilde_out": 1.0,
        "P_tilde_hole": 1.0,
        "Q_tilde_out": 0.2,
        "Q_tilde_hole": 0.2,
        "L_tilde": 0.0,
        "R_tilde": 0.0,
        "M_tilde_yhx": 0.0,
        "M_tilde_vhx": 0.0,
        "M_tilde_whx": 0.0,

        # Dangle tables (ΔG°37)
        "dangle_hole_L": dangles["dangle_hole_left"],
        "dangle_hole_R": dangles["dangle_hole_right"],
        "dangle_outer_L": dangles["dangle_outer_left"],
        "dangle_outer_R": dangles["dangle_outer_right"],

        # Coax seam energies (uniform table; your code can scale by variants)
        "coax_pairs": coax["coax_pairs_uniform"],
        "coax_bonus": 0.0,
        "coax_scale_oo": 1.0,
        "coax_scale_oi": 1.0,
        "coax_scale_io": 1.0,
        "coax_min_helix_len": 1,
        "coax_scale": 1.0,

        "mismatch_coax_scale": 0.5,
        "mismatch_coax_bonus": 0.0,

        # Drift & short-hole caps
        "join_drift_penalty": 0.0,
        "short_hole_caps": {0: 0.0, 1: 2.0, 2: 1.0},

        # Overlap penalties (left 0 by default)
        "Gwh": 0.0,
        "Gwi": 0.0,
        "Gwh_wx": 0.0,
        "Gwh_whx": 0.0,

        "pk_penalty_gw": 1.0,
    }
    return {"pseudoknot": pseudoknot}

# ---------- CLI ----------

def main(out_path: str = "pseudoknot_from_nndb.yaml"):
    y = build_yaml()
    with open(out_path, "w") as f:
        yaml.safe_dump(y, f, sort_keys=False)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "pseudoknot_from_nndb.yaml"
    main(out)

