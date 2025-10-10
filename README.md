# rna-pk-fold

# RNA-PK-FOLD: Optimal RNA Folding with Pseudoknots

This project implements the dynamic programming algorithm for RNA secondary structure prediction including **pseudoknots**, as described by Rivas & Eddy (1999). This method extends the conventional Zuker approach to handle non-nested base pairings using multi-dimensional gap matrices, aiming to find the globally optimal minimum free energy structure.

## 1. Approach and Optimization Techniques

### Core Dynamic Programming Approach

The implementation strictly follows the recursive relations laid out in the Rivas & Eddy (1999) paper (Equations 8, 9, 11, 12, 13, and 15). The algorithm utilizes five main dynamic programming matrices to track optimal folding energies:

* **2D Matrices:**

  * $W(i, j)$: Minimum free energy (MFE) for the subsequence $i..j$ (unconstrained).

  * $V(i, j)$: MFE for $i..j$, given that $i$ and $j$ are paired.

* **4D Gap Matrices (for Pseudoknots):**

  * $WHX(i, j: k, l)$: General optimal fold over segments $[i, k]$ and $[l, j]$.

  * $ZHX(i, j: k, l)$: $i$ and $j$ are paired; $k$ and $l$ are paired (stem-stem interaction).

  * $YHX(i, j: k, l)$: $i$ and $j$ are paired; $k$ and $l$ are unpaired/unconstrained.

The energy model combines standard nearest-neighbor rules (e.g., Turner 2004 parameters) for nested structures with explicit parameters for coaxial stacking and pseudoknot initiation/extension (following Rivas & Eddy Table 3 heuristics).

### Optimization Techniques

1. **High-Performance Kernels (Numba):** The calculation of the recurrence relations is the critical performance bottleneck ($O(N^6)$ time complexity). Python functions for calculating the core DP loops (`eddy_rivas_recurrences.py`, specifically iterators over matrix dimensions) are compiled using **Numba's Just-In-Time (JIT) compilation** (`@numba.njit`). This achieves native C/Fortran performance, drastically reducing the effective runtime compared to standard Python loops.

2. **Specialized Data Structures:** Custom data structures are used for the memory-intensive matrices. The **Gap Matrices ($WHX, ZHX, YHX$)** are implemented using specialized data types and indexing schemes optimized for sparse storage where possible, although the underlying complexity remains $O(N^4)$.

## 2. Algorithm Performance Evaluation

The Rivas & Eddy algorithm is known for its computational intensity.

| Metric | Theoretical Complexity | Practical Bottleneck | 
 | ----- | ----- | ----- | 
| **Time (Speed)** | $O(N^{6})$ | Despite Numba optimization, runtime still scales steeply. Max sequence length is severely constrained (typically $N < 150$). | 
| **Space (Memory)** | $O(N^{4})$ | Primarily dictated by the three 4D gap matrices. This quickly exhausts standard memory resources for $N \gtrsim 100$ nucleotides. | 

The primary limiting factor for scalability is the $O(N^4)$ memory requirement. For practical use on larger RNA molecules, constraints (like restricting loop sizes or maximum pseudoknot spans) are required, or the use of high-memory computing clusters. The Numba optimizations successfully mitigate the **time** complexity, allowing folding of sequences in the 70–100 nt range within reasonable timeframes, but the **memory** cost remains the hard limit.

## 3. Installation and Usage

### Prerequisites

* Python (3.8+)

* NumPy

* Numba

* PyYAML

### Installation (From Source)

1. **Clone the repository:**
```
git clone https://github.com/c-vandenberg/rna-pk-fold.git
cd rna-pk-fold
```
2. **Install the package (editable mode recommended):**
```
pip install -e .
```

### Usage

The folding routine can be run using the primary prediction script within the project root:
```
rna-pk-fold "GCCCGGGGC"
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
Sequence Length : 9
Sequence : GCCCGGGGC
Dot-Bracket Notation: (((...)))
ΔG (kcal/mol): -3.10
```

## 4. Discussion
### 4.1. Pseudoknot Prediction: Known Issues and Debugging Analysis
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
This is predicted structure is purely nested (no crossings). Similar issues have been observed with other pseudoknot test sequences