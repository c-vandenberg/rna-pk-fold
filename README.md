# rna-pk-fold

# RNA-PK-FOLD: Optimal RNA Folding with Pseudoknots

This project implements the dynamic programming algorithm for RNA secondary structure prediction including **pseudoknots**, as described by Rivas & Eddy (1999). This method extends the conventional Zuker approach to handle non-nested base pairings using multi-dimensional gap matrices, aiming to find the globally optimal minimum free energy structure.

## 1. Installation

### 1.1. Prerequisites
* Python (3.8+)
* NumPy
* Numba
* PyYAML

### 1.2. Installation (From Source)
1. **Clone the repository:**
```
git clone https://github.com/c-vandenberg/rna-pk-fold.git
cd rna-pk-fold
```
2. **Install the package (editable mode recommended):**
```
pip install -e .
```

### 1.3. Usage
The folding routine can be run using the primary prediction script within the project root:
```
rna-pk-fold GCGCGCGCGCAUUGCGCGCGCGC
```

The following arguments can be passed to the script:
```
usage: rna-pk-fold [-h] [--engine {auto,zucker,eddy_rivas}] [--yaml YAML] [--tempC TEMPC] [--json] [-v] [--log-file LOG_FILE] [--quiet]
                   [--pk-gw PK_GW] [--coax] [--overlap] [--min-hole-width MIN_HOLE_WIDTH] [--max-hole-width MAX_HOLE_WIDTH] [--q-ss Q_SS]
                   sequence

```

**Example Output:**
```
Engine : eddy_rivas
Sequence Length : 23
Sequence : GCGCGCGCGCAUUGCGCGCGCGC
Dot-Bracket Notation: ((((((((((...))))))))))
ΔG (kcal/mol): -23.00
```

## 2. Discussion
### 2.1 Core Dynamic Programming Approach

The implementation strictly follows the recursive relations laid out in the Rivas & Eddy (1999) paper (Equations 8, 9, 11, 12, 13, and 15). The algorithm utilizes five main dynamic programming matrices to track optimal folding energies:

* **2D Matrices:**
  * $W(i, j)$: Minimum free energy (MFE) for the subsequence $i..j$ (unconstrained).
  * $V(i, j)$: MFE for $i..j$, given that $i$ and $j$ are paired.
* **4D Gap Matrices (for Pseudoknots):**
  * $WHX(i, j: k, l)$: General optimal fold over segments $[i, k]$ and $[l, j]$.
  * $ZHX(i, j: k, l)$: $i$ and $j$ are paired; $k$ and $l$ are paired (stem-stem interaction).
  * $YHX(i, j: k, l)$: $i$ and $j$ are paired; $k$ and $l$ are unpaired/unconstrained.

The energy model combines standard nearest-neighbor rules (e.g., Turner 2004 parameters) for nested structures with explicit parameters for coaxial stacking and pseudoknot initiation/extension (following Rivas & Eddy Table 3 heuristics).

### 2.2. Optimization Techniques

1. **High-Performance Kernels (Numba):** The calculation of the recurrence relations is the critical performance bottleneck ($O(N^6)$ time complexity). Python functions for calculating the core DP loops (`eddy_rivas_recurrences.py`, specifically iterators over matrix dimensions) are compiled using **Numba's Just-In-Time (JIT) compilation** (`@numba.njit`). This achieves native C/Fortran performance, drastically reducing the effective runtime compared to standard Python loops.

2. **Specialized Data Structures:** Custom data structures are used for the memory-intensive matrices. The **Gap Matrices ($WHX, ZHX, YHX$)** are implemented using specialized data types and indexing schemes optimized for sparse storage where possible, although the underlying complexity remains $O(N^4)$.

### 2.3 Algorithm Performance Evaluation

The Rivas & Eddy algorithm is known for its computational intensity.

| Metric | Theoretical Complexity | Practical Bottleneck | 
 | ----- | ----- | ----- | 
| **Time (Speed)** | $O(N^{6})$ | Despite Numba optimization, runtime still scales steeply. Max sequence length is severely constrained (typically $N < 150$). | 
| **Space (Memory)** | $O(N^{4})$ | Primarily dictated by the three 4D gap matrices. This quickly exhausts standard memory resources for $N \gtrsim 100$ nucleotides. | 

The primary limiting factor for scalability is the $O(N^4)$ memory requirement. For practical use on larger RNA molecules, constraints (like restricting loop sizes or maximum pseudoknot spans) are required, or the use of high-memory computing clusters. The Numba optimizations successfully mitigate the **time** complexity, allowing folding of sequences in the 70–100 nt range within reasonable timeframes, but the **memory** cost remains the hard limit.

### 2.3. Test RNA Sequences Predictions
All test RNA predictions were carried out using the `turner2004_eddyrivas1999_min.yaml` configuration file. This configuration combined energy data and parameters from both ViennaRNA**<sup>1</sup>** for the core Zucker (nested) algorithm, and the paper by Eddy & Rivas**<sup>2</sup>** for the pseudoknot algorithm. The output was then compared to the predicted structure for [IPknot](https://ws.sato-lab.org/rtips/ipknot/) and ViennaRNA.

### Non-Psuedoknot RNA Sequences
| Sequence | Predicted Dot-Bracket Notation | Predicted $\Delta G$ (kcal/mol) | IPknot Prediction Match |
| :--- | :--- | :--- | :--- |
| GCGC | .... | 0.00 | ✅ |
| GCAUCUAUGC | (((....))) | -1.80 | ✅ |
| GGGAAAUCCC | (((....))) | -2.90 | ✅ |
| AUGCUAGCUAUGC | ......((...)) | -0.10 | ✅ |
| AUAUAUAUAU | .......... | 0.00 | ✅ |
| GCAAAGC | ....... | 0.00 | ✅ |
| GCAAAAGC | ........ | 0.00 | ✅ |
| GCAAAAAGC | ......... | 0.00 | ✅ |
| AUGGGAU | ....... | 0.00 | ✅ |
| AUGGGGAU | ........ | 0.00 | ✅ |
| GUAAAAGU | ........ | 0.00 | ✅ |
| UGAAAUG | ....... | 0.00 | ✅ |
| GCGCAAGC | ((....)) | -0.10 | ✅ |
| GCUUCGGC | ........ | 0.00 | ✅ |
| GCGGAGGC | ((....)) | -0.20 | ✅ |
| GGCGAACGCC | (((....))) | -3.00 | ✅ |
| GGCGAAUGCC | (((....))) | -3.00 | ✅ |
| GGCAAUUGCC | (((....))) | -3.00 | ✅ |
| GGCACAUUGCC | ((((...)))) | -5.20 | ✅ |
| GGCAAAUUGCC | ((((...)))). | -5.20 | ✅ |
| GGGAAACCCAAAGGGUUUCCC | (((((((((...))))))))) | -16.01 | ✅ |
| GCGAAUCCGAUUGGCUAAGCG | ((....((....))....)). | -0.70 | ✅ |
| GGAUCCGAAGGCUCGAUCC | ....((...))........ | -11.30 | ❌ |
| GGGAAAUCCAUUGGAUCCCUCC | (((...)))...(((....))) | -14.81 | ❌ |
| GCCGAUACGUAUCGGCGAU | ((((((....))))))... | -18.90 | ✅ |
| GCGCGCGCGCAUUGCGCGCGCGC | ((((((((((...)))))))))) | -23.00 | ✅ |
| GGGGCCCCGGGGCCCC | ((((((....)))))) | -12.91 | ✅ |
| GUGUGUGUACACACAC | ((((((....)))))). | -7.10 | ✅ |
| UGUGUGAAACACACA | ((((((...)))))) | -7.10 | ✅ |
| GUGUAAUUGUGU | ............ | 0.00 | ✅ |
| AUAUAUAUAU | .......... | 0.00 | ✅ |
| AAUAAAUAAAUAA | .............. | 0.00 | ✅ |
| AUAUAAUAUAUAUAU | (((((...))))).. | -1.20 | ❌ |
| GCGCGCAGCGCGC | (((((...))))) | -8.00 | ✅ |
| GGCGCCGCGGCC | (((......))) | -3.70 | ✅ |
| GCAUCUAUGC | (((....))) | -1.80 | ✅ |
| AUGCUAGCUAUGC | ......((...)) | -0.10 | ✅ |
| GGGAAAUCCC | (((....))) | -2.90 | ✅ |
| GGAUACGUACCU | ............ | 0.00 | ✅ |
| CGAUGCAGCUAG | ............ | 0.00 | ✅ |
| AAAAUAAAAUAAAAUAAAA | ................... | 0.00 | ✅ |
| UUUUUAAAUUUUUAAAUUUU | ..(((((....))))).... | -0.30 | ❌ |
| AUCCCUA | ....... | 0.00 | ✅ |
| GUCCUGU | ....... | 0.00 | ✅ |

## Pseudoknot RNA Sequences
| Sequence | Predicted Dot-Bracket Notation | Predicted $\Delta G$ (kcal/mol) | IPknot Prediction Match |
| :--- | :--- | :--- | :--- |
| UUCUUUUUUAGUGGCAGUAAGCCUGGGAAUGGGGGCGACCCAGGCGUAUGAACAUAGUGUAACGCUCCCC | (((((.......(((.....))).)))))((((.....))))(((((....(((...))).))))).... | -33.06 | ❌ |
| AGCUUUGAAAGCUUUCGAGUCUGUUUCGAAAUCACAAGGACCU | (((((...)))))((((((.....))))))............. | -9.81 | ❌ |

### 2.4. Pseudoknot Prediction: Known Issues and Debugging Analysis
### Problem Statement
The Eddy-Rivas pseudoknot prediction algorithm currently fails to predict pseudoknot structures for sequences where other tools (e.g., IPknot) successfully identify H-type pseudoknots. For example, for the test sequence:
```
UUCUUUUUAGUGGCAGUAAGCCUGGGAAUGGGGGCGACCCAGGCGUAUGAACAUAGUGUAACGCUCCCC
```
**Expected Structure (IPknot Prediction)**
```
................(((.(((((((....[[[[[[.))))))).))).............]]]]]]..
```
This contains a pseudoknot with crossing stems at positions 31-36 paired with 63-68.

**Actual Output (rna-pk-knot Prediction)**
```
(((((.......(((.....))).)))))((((.....))))(((((....(((...))).)))))....
```
This is predicted structure is purely nested (no crossings). Similar issues have been observed with other pseudoknot test sequences (e.g. `AGCUUUGAAAGCUUUCGAGUCUGUUUCGAAAUCACAAGGACCU`). Further investigation showed that the algorithm was indeed predicted charged (i.e. crossed/pseudoknot) composition paths with a lower free energy, however these were being filtered out.

### Debugging

All debugging was carried out via the use of the PyCharm debugger tool, logging, and print statemnts.

### Phase 1: Composition Layer Investigation
**Finding 1:** The WX composition correctly identifies pseudoknot holes with favorable energies:
* Hole (30,67): ΔG = -49.91 kcal/mol
* Hole (31,68): ΔG = -48.36 kcal/mol

These energies are significantly better than nested alternatives, and the kernel correctly selects them as winners during composition.
Issue: Despite printing "✓ NEW WINNER!", these holes were not being committed to the final structure.

**Root Cause:** The update logic (`best_c = cand`) was accidentally placed inside a debug conditional block, causing it to only update for specifically monitored holes. This was fixed by moving the update logic outside the debug block.

### Phase 2: Outer Interval Mismatch
**Finding:** The composition was querying incorrect outer intervals in gap matrices:
```
# WRONG (original):
Ru[t] = whx_collapse_with(re, k + 1, j, r + 1, l, ...)  # Outer: (k+1, j)

# CORRECT (fixed):
Ru[t] = whx_collapse_with(re, r + 1, j, r + 1, l, ...)  # Outer: (r+1, j)
```

**Impact:** This mismatch caused the composition to query different cells than traceback expected, leading to inconsistent energy calculations.
**Fix:** Corrected outer intervals to match traceback expectations.

### Phase 3: Filter Logic Analysis
Multiple filters were investigated that could block pseudoknot formation:
1. **Sparse matrix filter:** Checked if both WHX/YHX sides had finite values in raw sparse matrices. This was too strict because it rejected compositions using valid WXU baseline collapse.
2. **Zero-width hole filter:** Rejected holes where either `hole_left` or `hole_right` had width ≤ 1. Initially placed after `best_c` update, causing orphan energies (updated energy but no backpointer).

**Fixes Applied:**
* Removed sparse matrix filter entirely
* Moved zero-width filter before best_c update to ensure atomic updates

### Phase 4: Gap Matrix Coverage Analysis
**Critical Finding:** Even after all composition fixes, pseudoknot holes still fail to form true crossings. Traceback shows:
```
Winner: hole=(11,68), split=65
WHX_L=inf, WHX_R=inf  ← Both sides collapsed to baseline!
YHX_L=inf, YHX_R=inf
[MERGE] Found 20 nested pairs
crossings_within_layer=0  ← No actual pseudoknot formed!
```

**Root Cause:** The gap filling phase only computes holes where endpoints can form Watson-Crick pairs:
```
# In _dp_whx(), _dp_yhx(), _dp_vhx(), _dp_zhx():
for k, l in iter_holes_pairable(i, j, can_pair_mask):  # ← Only pairable holes
```

**Why This Breaks Pseudoknots:**
For the IPknot pseudoknot at hole (31,68) with split r=63, composition needs:
* Left gap: `WHX[0, 63, 31, 63]` or `YHX[0, 63, 31, 63]`
* Right gap: `WHX[64, 69, 64, 68]` or `YHX[64, 69, 64, 68]`

However:
* Bases 31 & 63: **G-A** (not pairable)
* Bases 64 & 68: **C-C** (not pairable)

Since these hole boundaries don't form Watson-Crick pairs, gap filling skips them entirely, leaving the sparse matrices at infinity. Composition then has no choice but to fall back to the WXU baseline (nested structure).

### Suspected Root Cause
The pairability constraint on hole boundaries is fundamentally incompatible with pseudoknot prediction. In the Eddy-Rivas model:
1. **Hole endpoints (k,l) are structural markers**, not actual base pairs
2. **Pseudoknot crossings arise from gap structure**, not from (k,l) pairing
3. **The algorithm should compute gaps for ALL holes**, regardless of endpoint pairability

The current implementation incorrectly assumes that if (k,l) can't pair, no meaningful gap structure exists between them. This is false for pseudoknots, where the crossing stems may involve completely different base pairs inside the hole boundaries.

### Attempted Solution
The fix should be to remove pairability constraints from gap filling:
```
# Change in _dp_whx(), _dp_yhx():
for k, l in iter_holes(i, j):  # ← All holes, not just pairable
    # Existing logic remains unchanged
```

**Expected Impact:**
* Gap matrices would contain entries for non-pairable holes
* Composition could find genuine gap structure instead of always collapsing to baseline
* Pseudoknot holes would form true crossings in traceback

It should be noted that this will affect:
1. **Performance:** Removes O(N²) filtering, significantly increasing computation
2. **Memory:** Sparse matrices would need many more entries

The above has been implemented, but the issue has persisted. further testing is required to ensure that logic elsewhere is not filtering non-pariable holes.

### Current Status (10/10/2025)
✅ Performs O(N⁶) composition with proper energy calculations<br>
✅ Identifies pseudoknot holes with favorable free energies<br>
✅ Handles backpointer creation and traceback mechanics<br>

But fails to:
❌ Compute gap matrices for holes with non-pairable endpoints<br>
❌ Form true crossing structures (always collapses to nested baseline)<br>
❌ Match predictions from specialized pseudoknot tools like IPknot<br>

### Recommendations for Future Work
1. Validate that pairability constraint is not present elsewhere, and if it is, remove it and re-test.
2. Profile performance impact of computing all holes vs. pairable-only
3. Add sparse matrix optimization to handle increased entry count
4. Validate against [CRW database](https://crw2-comparative-rna-web.org/16s-rrnas/) of known pseudoknots
5. Consider selective gap filling: compute non-pairable holes only when composition requests them (lazy evaluation)
6. Refactoring of larger, monolithic modules (in particular the DP modules `eddy_rivas_recurrences.py` and `zucker_recurrences.py`) once algorithm is successfully handling pseudoknots.

### 2.4. Computational Generation of Optimized RNA Sequences
### Approach: Simulated Annealing with Structure-Based Fitness
**Method:**
* Start with random sequence of desired length
* Use **simulated annealing** to explore sequence space via single-nucleotide mutations
* Evaluate fitness using existing Zucker DP implementation (returns number of base pairs and ΔG)

**Objective Functions:**
* **More pairings:** Minimize ΔG (more stable structure = more pairs)
* **Fewer pairings:** Maximize ΔG or minimize pair_count directly

**Implementation (Python Pseudocode):**
```
for temperature in annealing_schedule:
    mutated_seq = random_mutation(current_seq)
    new_energy = zucker_fold(mutated_seq).energy
    if accept_move(new_energy, old_energy, temperature):
        current_seq = mutated_seq
```

**Tools:** Python, NumPy, existing ZuckerFoldingEngine from this project
**Complexity (Estimated):** O(iterations × N³) where N³ is folding cost per evaluation

