from __future__ import annotations
from dataclasses import dataclass
import math
import time
import logging

from tqdm import tqdm

from rna_pk_fold.folding.zucker import ZuckerFoldState, ZuckerBacktrackOp, ZuckerBackPointer
from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModelProtocol
from rna_pk_fold.rules import can_pair, MIN_HAIRPIN_UNPAIRED
from rna_pk_fold.energies.energy_ops import best_multiloop_end_bonus

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ZuckerFoldingConfig:
    """
    Configuration settings for the Zuker folding algorithm.

    Attributes
    ----------
    temp_k : float
        The temperature in Kelvin used for energy calculations. Defaults to 310.15 K (37 °C).
    verbose : bool
        If True, enables verbose output, including a progress bar.
    """
    temp_k: float = 310.15
    verbose: bool = False


@dataclass(slots=True)
class ZuckerFoldingEngine:
    """
    Implements the Zuker dynamic programming algorithm for nested RNA folding.

    This class orchestrates the bottom-up filling of the DP matrices (W, V, WM)
    to find the minimum free energy secondary structure of an RNA sequence,
    considering only nested (non-pseudoknotted) structures.

    Attributes
    ----------
    energy_model : SecondaryStructureEnergyModelProtocol
        An object that provides methods to calculate the free energy contributions
        of various structural motifs (hairpins, stacks, etc.).
    config : ZuckerFoldingConfig
        A configuration object containing settings for the folding process.
    """
    energy_model: SecondaryStructureEnergyModelProtocol
    config: ZuckerFoldingConfig

    def fill_all_matrices(self, seq: str, state: ZuckerFoldState) -> None:
        """
        Executes the main Zuker dynamic programming algorithm.

        This method drives the entire folding process by iterating through all
        possible subsequence lengths (spans `d`) and filling the `WM`, `V`, and `W`
        matrices in a bottom-up fashion.

        Parameters
        ----------
        seq : str
            The RNA sequence to fold.
        state : ZuckerFoldState
            The state object containing the DP matrices to be filled.
        """
        start_time = time.perf_counter()
        n = len(seq)
        a, b, c, d_bonus = self.energy_model.params.MULTILOOP

        if n == 0:
            logger.info("Zucker DP: empty sequence; nothing to fill.")
            return

        # Log algorithm start
        logger.info("=" * 60)
        logger.info(f"Zucker (Nested) DP for sequence length N={n}")
        logger.info(f"Expected complexity: O(N³) ≈ {n ** 3:,} operations")
        logger.info("=" * 60)

        show_progress = self.config.verbose or logger.isEnabledFor(logging.INFO)
        span_iter = tqdm(range(0, n), desc="Zucker DP", leave=True, disable=not show_progress)

        # The main DP loop: Iterates by subsequence length 'd' (from 0 to N-1).
        for d in span_iter:
            # Inner loop: iterates by the start index 'i' of the subsequence.
            for i in range(0, n - d):
                # Calculate the end index 'j'.
                j = i + d

                # ---------- 1. Step 1: Fill WM matrix cell ----------
                # WM accumulates energies for structures within a multiloop.
                self._fill_wm_cell(seq, i, j, state, b, c)

                # ---------- 2. Fill V matrix cell ----------
                # V calculates energies for subsequences closed by a pair (i,j).
                # A hairpin requires a minimum length to form.
                if i < j:
                    self._fill_v_cell(seq, i, j, state, a)

                # ---------- 3. Fill W matrix cell ----------
                # W calculates the overall optimal energy for the subsequence [i,j].
                self._fill_w_cell(i, j, state, seq)

        elapsed = time.perf_counter() - start_time
        final_energy = state.w_matrix.get(0, n - 1)

        logger.info(f"Zucker DP completed in {elapsed:.2f}s ({elapsed * 1000:.0f}ms)")
        logger.info(f"Final W[0,{n - 1}] = {final_energy:.3f} kcal/mol")
        logger.info(f"Average time per cell: {elapsed * 1000 * 1000 / (n * (n + 1) / 2):.2f} μs")

    def _fill_wm_cell(
        self,
        seq: str,
        i: int,
        j: int,
        state: ZuckerFoldState,
        branch_cost_b: float,
        unpaired_cost_c: float
    ) -> None:
        """
        Fills a single cell WM[i, j] for the multiloop interior matrix.

        The WM matrix stores the minimum free energy for a subsequence `[i, j]`
        that is located *inside* a multiloop. The structure within `[i, j]`
        can consist of one or more helices branching off, interspersed with
        unpaired bases. This function calculates `WM[i, j]` by taking the
        minimum over all possible ways to form such a structure.

        Parameters
        ----------
        seq : str
            The RNA sequence.
        i : int
            The 5' start index of the subsequence.
        j : int
            The 3' end index of the subsequence.
        state : ZuckerFoldState
            The state object containing the DP matrices.
        branch_cost_b : float
            The thermodynamic penalty for adding a new branching helix to a multiloop.
        unpaired_cost_c : float
            The thermodynamic penalty for each unpaired nucleotide within a multiloop.

        Notes
        -----
        The value of `WM[i, j]` is the minimum of three cases:
        1.  `WM[i+1, j] + c`: Base `i` is unpaired.
        2.  `WM[i, j-1] + c`: Base `j` is unpaired.
        3.  `min_{i<k<=j} (V[i, k] + b + WM[k+1, j])`: A new helix `(i, k)`
            branches off, incurring a penalty `b`. The remainder `[k+1, j]`
            continues the multiloop interior.
        """
        # Get references to the required matrices from the state object.
        wm_matrix = state.wm_matrix
        wm_back_ptr = state.wm_back_ptr
        v_matrix = state.v_matrix

        # The base case for a single nucleotide (i == j) is already initialized to 0.
        if i == j:
            wm_back_ptr.set(i, j, ZuckerBackPointer(operation=ZuckerBacktrackOp.NONE))
            return

        # Initialize the best energy and backpointer for this cell.
        best_energy = math.inf
        best_rank = math.inf
        best_back_ptr = ZuckerBackPointer()

        # --- Recurrence Cases for WM ---
        # Case 1: Add an unpaired base at the 5' end.
        cand_energy = wm_matrix.get(i + 1, j) + unpaired_cost_c
        cand_rank = 1
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.UNPAIRED_LEFT)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Case 2: Add an unpaired base at the 3' end.
        cand_energy = wm_matrix.get(i, j - 1) + unpaired_cost_c
        cand_rank = 1
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.UNPAIRED_RIGHT)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Case 3: Attach a new helix starting at 'i' and closing at 'k'.
        for k in range(i + 1, j + 1):
            # The helix must be formed by a valid base pair.
            if not can_pair(seq[i], seq[k]):
                continue

            # Get the energy of the closing helix V(i,k).
            v_ik = v_matrix.get(i, k)
            if math.isinf(v_ik):
                continue

            # Calculate any bonus for terminal mismatches adjacent to the multiloop.
            end_bonus = 0.0
            if self.energy_model.params.MULTI_MISMATCH is not None:
                end_bonus = best_multiloop_end_bonus(i, k, seq, self.energy_model.params, self.config.temp_k)

            # Get the energy of the remaining segment of the multiloop.
            tail = 0.0 if k + 1 > j else wm_matrix.get(k + 1, j)

            # Total energy is the sum of the branch penalty, helix energy, bonuses, and tail energy.
            cand_energy = branch_cost_b + v_ik + end_bonus + tail
            cand_rank = 0
            cand_back_ptr = ZuckerBackPointer(
                operation=ZuckerBacktrackOp.MULTI_ATTACH, inner=(i, k), split_k=k, note="attach-helix"
            )
            best_energy, best_rank, best_back_ptr = self._compare_candidates(
                cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
            )

        # Store the optimal energy and backpointer for this cell.
        wm_matrix.set(i, j, best_energy)
        wm_back_ptr.set(i, j, best_back_ptr)

    def _fill_v_cell(self, seq: str, i: int, j: int, state: ZuckerFoldState,
                     multi_close_a: float) -> None:
        """
        Fills a single cell V[i, j] for the pair-closed matrix.

        The V matrix stores the minimum free energy for a subsequence `[i, j]`
        given the constraint that bases `i` and `j` form a pair. This function
        calculates `V[i, j]` by considering all possible nested structures that
        can be enclosed by this pair.

        Parameters
        ----------
        seq : str
            The RNA sequence.
        i : int
            The 5' start index of the subsequence (forms a pair with j).
        j : int
            The 3' end index of the subsequence (forms a pair with i).
        state : ZuckerFoldState
            The state object containing the DP matrices.
        multi_close_a : float
            The thermodynamic penalty for closing a multiloop.

        Notes
        -----
        The value of `V[i, j]` is the minimum of four cases:
        1.  **Hairpin**: The pair `(i, j)` closes a hairpin loop. The energy is
            calculated by the energy model.
        2.  **Stack**: The pair `(i, j)` stacks on an adjacent inner pair
            `(i+1, j-1)`. The total energy is `stacking_energy + V[i+1, j-1]`.
        3.  **Internal Loop**: The pair `(i, j)` encloses an inner pair `(k, l)`,
            forming an internal loop or bulge. The total energy is
            `internal_loop_energy + V[k, l]`.
        4.  **Multiloop**: The pair `(i, j)` closes a multiloop. The total
            energy is `multiloop_penalty + WM[i+1, j-1]`.
        """
        # Get references to the required matrices.
        v_matrix = state.v_matrix
        v_back_ptr = state.v_back_ptr

        # V(i,j) is only defined if 'i' and 'j' can form a base pair.
        if not can_pair(seq[i], seq[j]):
            v_matrix.set(i, j, math.inf)
            v_back_ptr.set(i, j, ZuckerBackPointer())
            return

        # Initialize the best energy and backpointer for this cell.
        best_energy = math.inf
        best_rank = math.inf
        best_back_ptr = ZuckerBackPointer()

        # --- Recurrence Cases for V ---
        # Case 1: (i,j) closes a hairpin loop.
        delta_g_hp = self.energy_model.hairpin(base_i=i, base_j=j, seq=seq, temp_k=self.config.temp_k)
        cand_energy = delta_g_hp
        cand_rank = 3
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.HAIRPIN)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Case 2: (i,j) stacks on an inner pair (i+1, j-1).
        if i + 1 <= j - 1 and can_pair(seq[i + 1], seq[j - 1]):
            delta_g_stk = self.energy_model.stack(
                base_i=i, base_j=j, base_k=i + 1, base_l=j - 1, seq=seq, temp_k=self.config.temp_k
            )
            if math.isfinite(delta_g_stk):
                inner = v_matrix.get(i + 1, j - 1)
                cand_energy = delta_g_stk + inner
                cand_rank = 0
                cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.STACK, inner=(i + 1, j - 1))
                best_energy, best_rank, best_back_ptr = self._compare_candidates(
                    cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
                )

        # Case 3: (i,j) closes an internal loop or bulge loop.
        for k in range(i + 1, j):
            for l in range(k + 1, j):
                if not can_pair(seq[k], seq[l]):
                    continue

                # Calculate the energy of the internal loop itself.
                delta_g_intl = self.energy_model.internal(
                    base_i=i, base_j=j, base_k=k, base_l=l, seq=seq, temp_k=self.config.temp_k
                )
                if not math.isfinite(delta_g_intl):
                    continue

                # Total energy is the loop energy plus the energy of the enclosed helix V(k,l).
                cand_energy = delta_g_intl + v_matrix.get(k, l)
                cand_rank = 1
                cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.INTERNAL, inner=(k, l))
                best_energy, best_rank, best_back_ptr = self._compare_candidates(
                    cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
                )

        # Case 3: (i,j) closes a multiloop.
        if j - i - 1 >= MIN_HAIRPIN_UNPAIRED:
            wm_inside = state.wm_matrix.get(i + 1, j - 1)

            # Total energy is the multiloop closing penalty plus the energy of the interior.
            cand_energy = multi_close_a + wm_inside
            cand_rank = 2
            cand_back_ptr = ZuckerBackPointer(
                operation=ZuckerBacktrackOp.MULTI_ATTACH, inner=(i + 1, j - 1), note="close-multiloop"
            )
            best_energy, best_rank, best_back_ptr = self._compare_candidates(
                cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
            )

        # Store the optimal energy and backpointer for this cell.
        v_matrix.set(i, j, best_energy)
        v_back_ptr.set(i, j, best_back_ptr)

    def _fill_w_cell(self, i: int, j: int, state: ZuckerFoldState, seq) -> None:
        """
        Fills a single cell W[i, j] for the main energy matrix.

        The W matrix stores the overall minimum free energy for the subsequence
        `[i, j]`, considering all possible valid nested secondary structures.
        This function calculates `W[i, j]` by taking the minimum over all
        possible decompositions of the subsequence.

        Parameters
        ----------
        i : int
            The 5' start index of the subsequence.
        j : int
            The 3' end index of the subsequence.
        state : ZuckerFoldState
            The state object containing the DP matrices.
        seq : str
            The RNA sequence (used for debugging output).

        Notes
        -----
        The value of `W[i, j]` is the minimum of four cases:
        1.  `W[i+1, j]`: Base `i` is left unpaired.
        2.  `W[i, j-1]`: Base `j` is left unpaired.
        3.  `V[i, j]`: Bases `i` and `j` form a pair, enclosing an optimal
            substructure.
        4.  `min_{i<=k<j} (W[i, k] + W[k+1, j])`: The structure is a
            bifurcation, composed of two independent adjacent substructures.
        """
        # Get references to the required matrices.
        w_matrix = state.w_matrix
        v_matrix = state.v_matrix
        w_back_ptr = state.w_back_ptr

        # Base case: a single nucleotide has 0 energy and no structure.
        if i == j:
            w_matrix.set(i, j, 0.0)
            w_back_ptr.set(i, j, ZuckerBackPointer(operation=ZuckerBacktrackOp.NONE))
            return

        # Initialize the best energy and backpointer for this cell.
        best_energy = math.inf
        best_rank = math.inf
        best_back_ptr = ZuckerBackPointer()

        # --- Recurrence Cases for W ---
        # Case 1: Leave base 'i' unpaired.
        cand_energy = w_matrix.get(i + 1, j)
        cand_rank = 2
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.UNPAIRED_LEFT)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Case 2: Leave base 'j' unpaired.
        cand_energy = w_matrix.get(i, j - 1)
        cand_rank = 2
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.UNPAIRED_RIGHT)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Case 3: The pair (i,j) is formed.
        cand_energy = v_matrix.get(i, j)
        cand_rank = 0
        cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.PAIR)
        best_energy, best_rank, best_back_ptr = self._compare_candidates(
            cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
        )

        # Case 4: Bifurcation. Split the interval [i,j] into two independent subproblems.
        for k in range(i, j):
            cand_energy = w_matrix.get(i, k) + w_matrix.get(k + 1, j)
            cand_rank = 1
            cand_back_ptr = ZuckerBackPointer(operation=ZuckerBacktrackOp.BIFURCATION, split_k=k)
            best_energy, best_rank, best_back_ptr = self._compare_candidates(
                cand_energy, cand_rank, cand_back_ptr, best_energy, best_rank, best_back_ptr
            )

        # This debugging block logs details for the final cell of the matrix.
        if i == 0 and j == len(seq) - 1:
            logger.debug(f"\n=== W[0,{j}] Final ===")
            logger.debug(f"Best energy: {best_energy:.2f}")
            logger.debug(f"Best operation: {best_back_ptr.operation}")
            logger.debug(f"V[0,{j}]: {v_matrix.get(i, j):.2f}")

        # Store the optimal energy and backpointer for this cell.
        w_matrix.set(i, j, best_energy)
        w_back_ptr.set(i, j, best_back_ptr)

    @staticmethod
    def _compare_candidates(
        cand_energy: float,
        cand_rank: float,
        cand_back_ptr: ZuckerBackPointer,
        best_energy: float,
        best_rank: float,
        best_back_ptr: ZuckerBackPointer,
    )-> tuple[float, float, ZuckerBackPointer]:
        """
        Selects the best of two candidates based on energy and rank.

        This helper function implements the tie-breaking rule: if two recursion
        cases produce the same energy, the one with the lower rank is preferred.
        This helps produce more canonical structures (e.g., smaller loops).

        Returns
        -------
        tuple[float, float, ZuckerBackPointer]
            The winning candidate's (energy, rank, backpointer) tuple.
        """
        if (cand_energy < best_energy) or (cand_energy == best_energy and cand_rank < best_rank):
            return cand_energy, cand_rank, cand_back_ptr

        return best_energy, best_rank, best_back_ptr
