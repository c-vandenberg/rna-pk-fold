# -----------------------
# Standard Library Imports
# -----------------------
import math
import json
import time
import logging
from dataclasses import dataclass, fields
from typing import Tuple, Dict, Optional, Any, Callable

# -----------------------
# Third-party Imports
# -----------------------
import numpy as np
from tqdm import tqdm

# -----------------------
# Local Application Imports
# -----------------------
from rna_pk_fold.energies.energy_types import PseudoknotEnergies
from rna_pk_fold.folding.zucker.zucker_fold_state import ZuckerFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer, EddyRivasBacktrackOp
from rna_pk_fold.utils.is2_utils import IS2_outer, IS2_outer_yhx
from rna_pk_fold.utils.iter_utils import (iter_spans, iter_holes, iter_complementary_tuples, iter_inner_holes,
                                          iter_holes_pairable, iter_complementary_tuples_pairable_kl)
from rna_pk_fold.utils.matrix_utils import (clear_matrix_caches, get_whx_with_collapse, get_zhx_with_collapse, wxI,
                                                             whx_collapse_with, zhx_collapse_with)
from rna_pk_fold.energies.energy_pk_ops import (dangle_hole_left, dangle_hole_right, dangle_outer_left,
                                                dangle_outer_right, coax_pack, short_hole_penalty)
from rna_pk_fold.folding.eddy_rivas.numba_kernels import (
    compose_wx_best_over_r_arrays,
    compose_vx_best_over_r,
    best_sum,
    best_sum_with_penalty,
)
from rna_pk_fold.utils.logging_utils import setup_logger

# -----------------------
# Logging Setup
# -----------------------
logger = setup_logger(
    name=__name__,
    level=logging.DEBUG,
    console_level=logging.INFO,
    file_level=logging.DEBUG,
)

# -----------------------
# Helper Functions
# -----------------------
def take_best(
    current_best_energy: float,
    current_back_pointer: Optional[EddyRivasBackPointer],
    candidate_energy: float,
    back_pointer_factory: Callable[[], EddyRivasBackPointer],
) -> Tuple[float, Optional[EddyRivasBackPointer]]:
    """
    Update the minimum energy and its corresponding backpointer.

    This is a helper function for the dynamic programming loops. It compares a
    candidate energy with the current best energy. If the candidate is better
    (lower), it updates the best energy and creates a new backpointer using
    the provided factory function.

    Parameters
    ----------
    current_best_energy : float
        The best energy found so far for a given DP state.
    current_back_pointer : Optional[EddyRivasBackPointer]
        The backpointer corresponding to the current best energy.
    candidate_energy : float
        The new energy to compare against the current best.
    back_pointer_factory : Callable[[], EddyRivasBackPointer]
        A zero-argument function that returns a new `EddyRivasBackPointer`
        instance for the candidate case. This lazy creation avoids
        unnecessary object instantiation.

    Returns
    -------
    Tuple[float, Optional[EddyRivasBackPointer]]
        A tuple containing the updated best energy and corresponding backpointer.
    """
    if candidate_energy < current_best_energy:
        return candidate_energy, back_pointer_factory()
    return current_best_energy, current_back_pointer


def make_back_pointer_factory(
    i: int, j: int, k: int, l: int
) -> Callable[..., EddyRivasBackPointer]:
    """
    Create a factory for generating `EddyRivasBackPointer` objects.

    This function captures the primary indices (i, j, k, l) and returns a
    closure that can be called later to create a backpointer. This simplifies
    the main DP loops by pre-filling the coordinate information.

    Parameters
    ----------
    i : int
        The 5' index of the outer span.
    j : int
        The 3' index of the outer span.
    k : int
        The 5' index of the inner hole.
    l : int
        The 3' index of the inner hole.

    Returns
    -------
    Callable[..., EddyRivasBackPointer]
        A function that takes a backtrack operation (`op`) and keyword
        arguments to produce a fully-formed `EddyRivasBackPointer`.
    """
    def create_back_pointer(
        op: EddyRivasBacktrackOp, **kwargs: Any
    ) -> EddyRivasBackPointer:
        """Instantiate a backpointer with pre-filled coordinates."""
        return EddyRivasBackPointer(
            op=op,
            outer=(i, j),
            hole=(k, l),
            **kwargs
        )
    return create_back_pointer


@dataclass(slots=True)
class EddyRivasFoldingConfig:
    """
    Configuration settings for the Eddy-Rivas folding algorithm.

    This dataclass holds all parameters that control the behavior of the
    dynamic programming algorithm, including penalties, feature toggles,
    and beam search parameters.

    Attributes
    ----------
    enable_coax : bool
        If True, enables coaxial stacking energy bonuses.
    enable_wx_overlap : bool
        If True, enables WX same-hole overlap terms.
    enable_coax_variants : bool
        If True, adds extra coaxial stacking topologies in VX composition.
    enable_coax_mismatch : bool
        If True, allows coaxial stacking at seams with a one-nucleotide gap.
    enable_join_drift : bool
        If True, allows the hole to shift slightly at a join point.
    drift_radius : int
        The maximum distance the hole can shift if `enable_join_drift` is True.
    enable_is2 : bool
        If True, includes energy calculations for Irreducible Surfaces of Order 2.
    pk_penalty_gw : float
        The free energy penalty (in kcal/mol) for initiating a pseudoknot (Gw).
    max_hole_width : int
        The maximum allowed width of a pseudoknot hole (l - k).
    min_hole_width : int
        The minimum allowed width of a pseudoknot hole.
    min_outer_left : int
        The minimum length of the 5' outer segment [i..r].
    min_outer_right : int
        The minimum length of the 3' outer segment [r+1..j].
    beam_k : int
        If > 0, enables beam search, keeping at most K holes (k, l) per outer
        span (i, j).
    beam_v_threshold : float
        Threshold for beam search; keeps holes (k, l) only if the nested
        energy V[k][l] is below this value.
    strict_complement_order : bool
        If True, enforces the strict ordering i < k <= r < l <= j for pseudoknots.
    costs : Optional[PseudoknotEnergies]
        A data object containing all thermodynamic energy parameters.
    tables : Optional[Any]
        An object containing pre-computed energy tables (e.g., for dangle ends).
    verbose : bool
        If True, enables verbose logging.
    """
    enable_coax: bool = False # keep off initially
    enable_wx_overlap: bool = False # turn on WX same-hole overlap terms
    enable_coax_variants: bool = False  # NEW: add extra coax topologies in VX composition
    enable_coax_mismatch: bool = False  # allow |k-r|==1 seam as "mismatch coax"
    enable_join_drift: bool = False  # enable slight hole drift at join
    drift_radius: int = 0  # how far to drift (0 = off)
    enable_is2: bool = True
    pk_penalty_gw: float = 1.0 # Gw: pseudoknot introduction penalty (kcal/mol)
    max_hole_width: int = 0
    min_hole_width: int = 0  # 0 = identical behavior; 1+ prunes zero/narrow holes
    min_outer_left: int = 0  # minimal length of [i..r]
    min_outer_right: int = 0  # minimal length of [r+1..j]
    beam_k: int = 0                 # 0 = disabled, else keep at most K (k,l) per (i,j)
    beam_v_threshold: float = 0.0  # keep (k,l) only if nested V[k][l] <= this (e.g. -0.1)
    strict_complement_order: bool = True  # enforce i<k<=r<l<=j
    costs: Optional[PseudoknotEnergies] = None
    tables: object = None
    verbose: bool = False

# -----------------------
# Engine
# -----------------------
class EddyRivasFoldingEngine:
    """
    Implements the Rivas and Eddy dynamic programming algorithm for RNA folding.

    This class orchestrates the filling of the DP matrices (`wx`, `vx`, and the
    four "gap" matrices `whx`, `vhx`, `zhx`, `yhx`) to find the minimum free
    energy secondary structure of an RNA sequence, including pseudoknots.

    The algorithm proceeds in three main phases:
    1. Seeding: Initialize `wx` and `vx` from a pre-computed nested fold.
    2. Gap Matrix Filling: Populate the four O(N⁴) gap matrices.
    3. Composition: Use the gap matrices to update `wx` and `vx` with
       pseudoknotted structures in O(N⁶) time.
    """
    def __init__(self, config: EddyRivasFoldingConfig):
        self.cfg = config
        self.timings = {}

    @staticmethod
    def _build_can_pair_mask(seq: str) -> list[list[bool]]:
        """
        Creates a boolean mask indicating which nucleotides can form pairs.

        Parameters
        ----------
        seq : str
            The RNA sequence.

        Returns
        -------
        np.ndarray
            A 2D numpy array of booleans where `mask[i, j]` is True if the
            bases at `sequence[i]` and `sequence[j]` can form a Watson-Crick
            or wobble pair.
        """
        from rna_pk_fold.rules.constraints import can_pair
        seq_len = len(seq)
        mask = [[False] * seq_len for _ in range(seq_len)]
        for k in range(seq_len):
            base_k = seq[k]
            for l in range(k + 1, seq_len):
                mask[k][l] = can_pair(base_k, seq[l])
        return mask

    def fill_with_costs(self, seq: str, nested: ZuckerFoldState, eddy_rivas_fold_state: EddyRivasFoldState) -> None:
        """
        Executes the main Eddy-Rivas dynamic programming algorithm.

        This method drives the entire folding process. It initializes the DP
        matrices, fills the gap matrices, and then composes the final matrices
        to find the optimal folding energy including pseudoknots.

        Parameters
        ----------
        seq : str
            The RNA sequence to fold.
        nested : ZuckerFoldState
            A pre-filled state object containing the results of a nested-only
            (e.g., Zuker) folding algorithm.
        eddy_rivas_fold_state : EddyRivasFoldState
            The state object that will be populated with the results of this
            algorithm. It contains all the DP matrices.

        Notes
        -----
        The algorithm follows a precise sequence of steps as outlined in the
        Rivas and Eddy paper:

        1.  **Seeding**: The process begins by populating the primary DP
            matrices, `wx` (best energy for subsequence `i` to `j`) and `vx`
            (best energy given `i` and `j` are paired), with the results from
            the `nested_fold_state`. This establishes a baseline of optimal
            non-pseudoknotted structures.

        2.  **Gap Matrix Filling (O(N⁴) Complexity)**: This is the core of the
            pseudoknot detection. The algorithm fills four "gap matrices" that
            store energies for structures spanning two disconnected segments,
            `[i..k]` and `[l..j]`, leaving a "hole" `[k+1..l-1]`.
            - `vhx(i,j:k,l)`: Energy where `(i,j)` and `(k,l)` are both paired.
            - `zhx(i,j:k,l)`: Energy where `(i,j)` is paired, `(k,l)` is not.
            - `yhx(i,j:k,l)`: Energy where `(k,l)` is paired, `(i,j)` is not.
            - `whx(i,j:k,l)`: Energy where pairing of `(i,j)` and `(k,l)` is undetermined.
            These are filled iteratively, building larger gapped structures
            from smaller nested and gapped ones.

        3.  **Composition (O(N⁶) Complexity)**: After the gap matrices are
            complete, this phase updates the `wx` and `vx` matrices by
            considering all possible ways to form a pseudoknot. For each span
            `(i, j)`, the algorithm iterates through all split points `r` to
            combine two complementary gapped fragments, one spanning `[i..r]`
            and the other `[r+1..j]`. This step finds the minimum energy by
            either maintaining the existing nested structure or introducing a
            more stable pseudoknotted one.

        4.  **Final Energy**: The optimal free energy for the entire sequence
            is the value stored in `wx[0, n-1]`. The structure itself can be
            reconstructed via a traceback procedure using the backpointers
            stored during the DP fill.
        """
        total_start = time.perf_counter()

        seq_len = eddy_rivas_fold_state.seq_len
        # Log algorithm start
        logger.info("=" * 60)
        logger.info(f"Eddy-Rivas DP for sequence length N={seq_len}")
        logger.info(f"Expected complexity:")
        logger.info(f"  Gap matrices: O(N⁴) ≈ {seq_len ** 4:,} operations")
        logger.info(f"  Compositions: O(N⁶) ≈ {seq_len ** 6:,} operations")
        logger.info("=" * 60)

        clear_matrix_caches()

        q_ss = self.cfg.costs.q_ss
        g_w = self.cfg.pk_penalty_gw
        g_wh = getattr(self.cfg.costs, "Gwh", 0.0)
        g_wi = self.cfg.costs.Gwi
        g_wh_wx = getattr(self.cfg.costs, "Gwh_wx", 0.0)
        g_wh_whx = getattr(self.cfg.costs, "Gwh_whx", 0.0)
        tables = getattr(self.cfg, "tables", None)
        g = self.cfg.costs.coax_scale

        # tilde scalars (names preserved)
        p_out = getattr(tables, "P_tilde_out", getattr(self.cfg.costs, "P_tilde_out", 1.0))
        p_hole = getattr(tables, "P_tilde_hole", getattr(self.cfg.costs, "P_tilde_hole", 1.0))
        l_tilde = getattr(tables, "L_tilde", 0.0)
        r_tilde = getattr(tables, "R_tilde", 0.0)
        q_tilde_out = getattr(tables, "Q_tilde_out", getattr(self.cfg.costs, "Q_tilde_out", 0.0))
        q_tilde_hole = getattr(tables, "Q_tilde_hole", getattr(self.cfg.costs, "Q_tilde_hole", 0.0))
        m_tilde_yhx = getattr(tables, "M_tilde_yhx", getattr(self.cfg.costs, "M_tilde_yhx", 0.0))
        m_tilde_vhx = getattr(tables, "M_tilde_vhx", getattr(self.cfg.costs, "M_tilde_vhx", 0.0))
        m_tilde_whx = getattr(tables, "M_tilde_whx", getattr(self.cfg.costs, "M_tilde_whx", 0.0))

        # --- Phase 1: Seeding ---
        seed_start = time.perf_counter()
        self._seed_from_nested(nested, eddy_rivas_fold_state)
        eddy_rivas_fold_state.wxu_matrix.enable_dense()
        eddy_rivas_fold_state.wxc_matrix.enable_dense()
        eddy_rivas_fold_state.vxu_matrix.enable_dense()
        eddy_rivas_fold_state.vxc_matrix.enable_dense()
        can_pair_mask = self._build_can_pair_mask(seq)
        self.timings['seed'] = time.perf_counter() - seed_start
        logger.info(f"Seeding completed in {self.timings['seed']:.2f}s")

        # --- Phase 2: Gap Matrix Filling ---
        # WHX
        logger.info("Filling WHX matrix...")
        whx_start = time.perf_counter()
        self._dp_whx(seq, eddy_rivas_fold_state, q_ss, g_wh_whx, can_pair_mask)
        self.timings['whx'] = time.perf_counter() - whx_start
        logger.info(f"WHX filled in {self.timings['whx']:.2f}s")

        # VHX
        logger.info("Filling VHX matrix...")
        vhx_start = time.perf_counter()
        self._dp_vhx(seq, eddy_rivas_fold_state, g_wi, p_hole, l_tilde, r_tilde,
                     q_tilde_hole, m_tilde_vhx, m_tilde_whx, can_pair_mask)
        self.timings['vhx'] = time.perf_counter() - vhx_start
        logger.info(f"VHX filled in {self.timings['vhx']:.2f}s")

        # ZHX
        logger.info("Filling ZHX matrix...")
        zhx_start = time.perf_counter()
        self._dp_zhx(seq, eddy_rivas_fold_state, g_wi, p_hole, q_tilde_hole, can_pair_mask)
        self.timings['zhx'] = time.perf_counter() - zhx_start
        logger.info(f"ZHX filled in {self.timings['zhx']:.2f}s")

        # YHX
        logger.info("Filling YHX matrix...")
        yhx_start = time.perf_counter()
        self._dp_yhx(seq, eddy_rivas_fold_state, q_ss, g_wi, p_out, q_tilde_out,
                     m_tilde_yhx, m_tilde_whx, can_pair_mask)
        self.timings['yhx'] = time.perf_counter() - yhx_start
        logger.info(f"YHX filled in {self.timings['yhx']:.2f}s")

        # --- Phase 3: Composition ---
        # WX Composition
        logger.info("Composing WX matrix...")
        wx_start = time.perf_counter()
        self._compose_wx(seq, eddy_rivas_fold_state, g_w, g_wh_wx, can_pair_mask)
        self._publish_wx(eddy_rivas_fold_state)
        self.timings['wx_compose'] = time.perf_counter() - wx_start
        logger.info(f"WX composed in {self.timings['wx_compose']:.2f}s")

        # VX Composition
        logger.info("Composing VX matrix...")
        vx_start = time.perf_counter()
        self._compose_vx(seq, eddy_rivas_fold_state, g_w, g, can_pair_mask)
        self._publish_vx(eddy_rivas_fold_state)
        self.timings['vx_compose'] = time.perf_counter() - vx_start
        logger.info(f"VX composed in {self.timings['vx_compose']:.2f}s")

        # --- Final Logging ---
        self.timings['total'] = time.perf_counter() - total_start
        final_energy = eddy_rivas_fold_state.wx_matrix.get(0, seq_len - 1)
        logger.info("=" * 60)
        logger.info(f"Eddy-Rivas DP completed in {self.timings['total']:.2f}s")
        logger.info(f"Final WX[0,{seq_len - 1}] = {final_energy:.3f} kcal/mol")
        logger.info("")
        logger.info("Timing breakdown:")
        logger.info(
            f"  Seeding:        {self.timings['seed']:7.2f}s ({self.timings['seed'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  WHX fill:       {self.timings['whx']:7.2f}s ({self.timings['whx'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  VHX fill:       {self.timings['vhx']:7.2f}s ({self.timings['vhx'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  ZHX fill:       {self.timings['zhx']:7.2f}s ({self.timings['zhx'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  YHX fill:       {self.timings['yhx']:7.2f}s ({self.timings['yhx'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  WX composition: {self.timings['wx_compose']:7.2f}s ({self.timings['wx_compose'] / self.timings['total'] * 100:5.1f}%)")
        logger.info(
            f"  VX composition: {self.timings['vx_compose']:7.2f}s ({self.timings['vx_compose'] / self.timings['total'] * 100:5.1f}%)")
        gap_total = self.timings['whx'] + self.timings['vhx'] + self.timings['zhx'] + self.timings['yhx']
        comp_total = self.timings['wx_compose'] + self.timings['vx_compose']
        logger.info(f"  Gap matrices:   {gap_total:7.2f}s ({gap_total / self.timings['total'] * 100:5.1f}%)")
        logger.info(f"  Compositions:   {comp_total:7.2f}s ({comp_total / self.timings['total'] * 100:5.1f}%)")
        logger.info("=" * 60)

    # --------- Seeding ---------
    @staticmethod
    def _seed_from_nested(nested_fold_state: ZuckerFoldState, eddy_rivas_fold_state: EddyRivasFoldState) -> None:
        """
        Initializes the Eddy-Rivas DP matrices from a pre-computed nested fold.

        This method populates the initial state of the pseudoknot-aware DP
        matrices (`wx`, `vx`, etc.) with the optimal energies found by a
        nested-only algorithm (e.g., Zuker). This provides an energy baseline
        for every possible subsequence, which the subsequent composition steps
        will attempt to improve upon by introducing pseudoknots.

        Parameters
        ----------
        nested_fold_state : ZuckerFoldState
            The state object containing the results of a completed nested-only
            folding calculation. This is the source of the initial energies.
        eddy_rivas_fold_state : EddyRivasFoldState
            The state object for the pseudoknot algorithm, which will be
            initialized by this method.

        Notes
        -----
        - The `wxu` and `vxu` matrices ("uncomposed") store these initial
          nested energies.
        - The `wxc` and `vxc` matrices ("composed") are initialized to infinity,
          as they will later store the optimal energies derived from combining
          gapped fragments to form pseudoknots.
        - The final `wx` and `vx` matrices are also set to the nested values,
          acting as the starting point for the DP updates.
        """
        n = eddy_rivas_fold_state.seq_len
        for i, j in iter_spans(n):
            base_w = nested_fold_state.w_matrix.get(i, j)
            base_v = nested_fold_state.v_matrix.get(i, j)

            eddy_rivas_fold_state.wxu_matrix.set(i, j, base_w)
            eddy_rivas_fold_state.vxu_matrix.set(i, j, base_v)

            if i != j:
                eddy_rivas_fold_state.wxc_matrix.set(i, j, math.inf)
                eddy_rivas_fold_state.vxc_matrix.set(i, j, math.inf)

            eddy_rivas_fold_state.wx_matrix.set(i, j, base_w)
            eddy_rivas_fold_state.vx_matrix.set(i, j, base_v)

            if hasattr(eddy_rivas_fold_state, "wxi_matrix") and eddy_rivas_fold_state.wxi_matrix is not None:
                eddy_rivas_fold_state.wxi_matrix.set(i, j, base_w)

    # --------- WHX ---------
    def _dp_whx(self, seq: str, eddy_rivas_fold_state: EddyRivasFoldState,
                unpaired_base_penalty: float, overlap_penalty: float,
                can_pair_mask: list[list[bool]]) -> None:
        """
        Fills the WHX gap matrix using dynamic programming.

        WHX(i, j: k, l) stores the minimum free energy for a structure spanning
        the disconnected segments [i..k] and [l..j]. This is the most general
        of the four gap matrices, as the pairing status of the external bases
        (i, j) and the internal hole bases (k, l) is undetermined.

        Parameters
        ----------
        seq : str
            The RNA sequence.
        eddy_rivas_fold_state : EddyRivasFoldState
            The state object containing all DP matrices.
        unpaired_base_penalty : float
            The energy cost (q) for a single unpaired nucleotide.
        overlap_penalty : float
            The energy cost (Gwh) for forming an overlapping pseudoknot.
        can_pair_mask : list[list[bool]]
            A boolean matrix where `mask[i, j]` is True if bases at i and j
            can form a pair.

        Notes
        -----
        The method calculates the optimal energy for each state `WHX(i, j: k, l)`
        by taking the minimum over several recursive cases, which correspond to
        different ways of forming the structure:
        - **Add Unpaired Base**: Add an unpaired nucleotide to one of the four
          endpoints (i, j, k, or l).
        - **Collapse**: The hole [k+1..l-1] collapses, resulting in a nested
          structure from i to j, represented by `WX(i, j)`.
        - **Bifurcation**: The structure is split into a gapped part and a
          nested part (e.g., `WHX(i, r: k, l) + WX(r+1, j)`).
        - **Overlap**: Two gapped structures with the same hole are joined,
          incurring a penalty. This models overlapping pseudoknots.
        - **IS2 Motif**: A specific structure (Irreducible Surface of order 2)
          is formed by combining a `YHX` subproblem with a bridge energy.
        """
        spans = list(iter_spans(eddy_rivas_fold_state.seq_len))
        for i, j in tqdm(spans, desc="WHX", leave=False):
            for k, l in iter_holes_pairable(i, j, can_pair_mask):
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                # Pruning for beam search
                if self.cfg.beam_v_threshold != 0.0:
                    if eddy_rivas_fold_state.vxu_matrix.get(k, l) > self.cfg.beam_v_threshold:
                        continue

                best = math.inf
                best_bp: Optional[EddyRivasBackPointer] = None

                # Case 1: Add an unpaired base at the 5' end of the hole.
                v = get_whx_with_collapse(eddy_rivas_fold_state.whx_matrix,
                                          eddy_rivas_fold_state.wxu_matrix, i, j, k + 1, l)
                cand = v + unpaired_base_penalty
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SHRINK_LEFT,
                                                   outer=(i, j), hole=(k, l))

                # Case 2: Add an unpaired base at the 3' end of the hole.
                v = get_whx_with_collapse(eddy_rivas_fold_state.whx_matrix,
                                          eddy_rivas_fold_state.wxu_matrix, i, j, k, l - 1)
                cand = v + unpaired_base_penalty
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SHRINK_RIGHT,
                                                   outer=(i, j), hole=(k, l))

                # Case 3: Add an unpaired base at the 5' end of the outer span.
                v = eddy_rivas_fold_state.whx_matrix.get(i + 1, j, k, l)
                cand = v + unpaired_base_penalty
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_TRIM_LEFT,
                                                   outer=(i, j), hole=(k, l))

                # Case 4: Add an unpaired base at the 3' end of the outer span.
                v = eddy_rivas_fold_state.whx_matrix.get(i, j - 1, k, l)
                cand = v + unpaired_base_penalty
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_TRIM_RIGHT,
                                                   outer=(i, j), hole=(k, l))

                # Case 5: Collapse the hole to form a nested structure WX(i, j).
                v = get_whx_with_collapse(eddy_rivas_fold_state.whx_matrix,
                                          eddy_rivas_fold_state.wxu_matrix, i, j, k, l)
                cand = v
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_COLLAPSE,
                                                   outer=(i, j), hole=(k, l))

                # Case 6: Add unpaired bases at both outer ends.
                v = eddy_rivas_fold_state.whx_matrix.get(i + 1, j - 1, k, l)
                if math.isfinite(v):
                    cand = v + 2.0 * unpaired_base_penalty
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_SS_BOTH,
                                                       outer=(i, j), hole=(k, l))

                # Case 7: Bifurcation into WHX + WX (left-gapped).
                lr = max(0, j - i)
                if lr > 0:
                    left_vec = np.full(lr, np.inf, dtype=np.float64)
                    right_vec = np.full(lr, np.inf, dtype=np.float64)
                    for t in range(lr):
                        r = i + t
                        lv = eddy_rivas_fold_state.whx_matrix.get(i, r, k, l)
                        rv = wxI(eddy_rivas_fold_state, r + 1, j)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv

                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        r_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_WHX_SPLIT_LEFT_WHX_WX,
                            outer=(i, j), hole=(k, l), split=r_star
                        )

                # Case 8: Bifurcation into WX + WHX (right-gapped).
                ls = max(0, j - i)
                if ls > 0:
                    left_vec = np.full(ls, np.inf, dtype=np.float64)
                    right_vec = np.full(ls, np.inf, dtype=np.float64)
                    for t in range(ls):
                        s2 = i + t
                        lv = wxI(eddy_rivas_fold_state, i, s2)
                        rv = eddy_rivas_fold_state.whx_matrix.get(s2 + 1, j, k, l)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv

                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        s2_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_WHX_SPLIT_RIGHT_WX_WHX,
                            outer=(i, j), hole=(k, l), split=s2_star
                        )

                # Case 9: Overlapping pseudoknot (WHX + WHX).
                if overlap_penalty != 0.0:
                    lr = max(0, j - i)
                    if lr > 0:
                        left_vec = np.full(lr, np.inf, dtype=np.float64)
                        right_vec = np.full(lr, np.inf, dtype=np.float64)
                        for t in range(lr):
                            r = i + t
                            lv = eddy_rivas_fold_state.whx_matrix.get(i, r, k, l)
                            rv = eddy_rivas_fold_state.whx_matrix.get(r + 1, j, k, l)
                            if math.isfinite(lv): left_vec[t] = lv
                            if math.isfinite(rv): right_vec[t] = rv

                        cand_overlap, t_star = best_sum_with_penalty(left_vec, right_vec, float(overlap_penalty))
                        if t_star >= 0 and cand_overlap < best:
                            r_star = i + t_star
                            best = cand_overlap
                            best_bp = EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_WHX_OVERLAP_SPLIT,
                                outer=(i, j), hole=(k, l), split=r_star
                            )

                # Case 10: IS2 motif (Irreducible Surface of order 2).
                if self.cfg.enable_is2:
                    for r2 in range(i, k + 1):
                        for s2 in range(l, j + 1):
                            if r2 <= k and l <= s2 and r2 <= s2:
                                inner_y = eddy_rivas_fold_state.yhx_matrix.get(r2, s2, k, l)
                                if math.isfinite(inner_y):
                                    bridge = IS2_outer_yhx(self.cfg, seq, i, j, r2, s2)
                                    cand = bridge + inner_y
                                    if cand < best:
                                        best = cand
                                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WHX_IS2_INNER_YHX,
                                                                       outer=(i, j), hole=(k, l), bridge=(r2, s2))

                eddy_rivas_fold_state.whx_matrix.set(i, j, k, l, best)
                eddy_rivas_fold_state.whx_back_ptr.set(i, j, k, l, best_bp)

    # --------- VHX ---------
    def _dp_vhx(
        self,
        seq: str,
        eddy_rivas_fold_state: EddyRivasFoldState,
        internal_pk_penalty: float,
        tilde_p_hole: float,
        tilde_l_hole: float,
        tilde_r_hole: float,
        tilde_q_hole: float,
        tilde_m_vhx: float,
        tilde_m_whx: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        Fills the VHX gap matrix using dynamic programming.

        VHX(i, j: k, l) stores the minimum free energy for a structure where
        the outer span (i, j) AND the inner hole span (k, l) are both closed
        by base pairs. This represents a core pseudoknot motif of two helices.

        Parameters
        ----------
        seq : str
            The RNA sequence.
        eddy_rivas_fold_state : EddyRivasFoldState
            The state object containing all DP matrices.
        internal_pk_penalty : float
            Penalty for forming an internal pseudoknot (Gwi).
        tilde_p_hole : float
            Penalty for a base pair adjacent to the hole (~P).
        tilde_l_hole : float
            Energy contribution of a 5' dangle in the hole (~L).
        tilde_r_hole : float
            Energy contribution of a 3' dangle in the hole (~R).
        tilde_q_hole : float
            Penalty for an unpaired base in the hole (~Q).
        tilde_m_vhx : float
            Penalty for a multiloop originating from a VHX state (~M).
        tilde_m_whx : float
            Penalty for a multiloop originating from a WHX state (~M).
        can_pair_mask : np.ndarray
            A boolean matrix indicating allowed base pairs.

        Notes
        -----
        The recursion for VHX involves several cases:
        - **Dangles**: Adding a dangling base next to the (k,l) pair inside
          the hole.
        - **Unpaired Base**: Adding a single-stranded base adjacent to the hole,
          transitioning from a ZHX state.
        - **Bifurcation**: Splitting the region between the outer and inner
          helices into a nested part (WX) and a gapped part (ZHX).
        - **IS2 Motif**: Forming an Irreducible Surface of order 2 by bridging
          the (i, j) pair with an inner ZHX structure.
        - **Multiloop**: Closing a multiloop around a WHX subproblem.
        """
        spans = list(iter_spans(eddy_rivas_fold_state.seq_len))
        for i, j in tqdm(spans, desc="VHX", leave=False):
            for k, l in iter_holes_pairable(i, j, can_pair_mask):
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                if self.cfg.beam_v_threshold != 0.0:
                    if eddy_rivas_fold_state.vxu_matrix.get(k, l) > self.cfg.beam_v_threshold:
                        continue

                best = eddy_rivas_fold_state.vhx_matrix.get(i, j, k, l)
                best_bp: Optional[EddyRivasBackPointer] = None

                # Case 1: Dangle on the 5' side of the inner pair (k,l).
                v = eddy_rivas_fold_state.vhx_matrix.get(i, j, k + 1, l)
                cand = tilde_p_hole + tilde_l_hole + v
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_DANGLE_L,
                        outer=(i, j), hole=(k, l)
                    )

                # Case 2: Dangle on the 3' side of the inner pair (k,l).
                v = eddy_rivas_fold_state.vhx_matrix.get(i, j, k, l - 1)
                cand = tilde_p_hole + tilde_r_hole + v
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_DANGLE_R,
                        outer=(i, j), hole=(k, l)
                    )

                # Case 3: Dangles on both sides of the inner pair (k,l).
                v = eddy_rivas_fold_state.vhx_matrix.get(i, j, k + 1, l - 1)
                cand = tilde_p_hole + tilde_l_hole + tilde_r_hole + v
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_DANGLE_LR,
                        outer=(i, j), hole=(k, l)
                    )

                # Case 4: Add an unpaired base in the hole (from a ZHX state).
                # This logic block has a special tie-breaking rule.
                v_zhx = get_zhx_with_collapse(eddy_rivas_fold_state.zhx_matrix,
                                              eddy_rivas_fold_state.vxu_matrix, i, j, k, l)
                cand = tilde_q_hole + v_zhx
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_SS_LEFT,
                        outer=(i, j), hole=(k, l)
                    )
                elif (cand == best and isinstance(best_bp, EddyRivasBackPointer) and
                      best_bp.op in (EddyRivasBacktrackOp.RE_VHX_SS_LEFT,
                                     EddyRivasBacktrackOp.RE_VHX_SS_RIGHT)):
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_SS_RIGHT,
                        outer=(i, j), hole=(k, l)
                    )

                # Case 5: Bifurcation on the 5' side (ZHX + WX).
                lr = max(0, k - i)
                if lr > 0:
                    left_vec = np.full(lr, np.inf, dtype=np.float64)
                    right_vec = np.full(lr, np.inf, dtype=np.float64)
                    for t in range(lr):
                        r = i + t
                        lv = get_zhx_with_collapse(eddy_rivas_fold_state.zhx_matrix,
                                                   eddy_rivas_fold_state.vxu_matrix, i, j, r, l)
                        rv = wxI(eddy_rivas_fold_state, r + 1, k)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv

                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        r_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_SPLIT_LEFT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=r_star
                        )

                # Case 6: Bifurcation on the 3' side (ZHX + WX).
                ls = max(0, j - (l + 1) + 1)  # = j - l
                if ls > 0:
                    left_vec = np.full(ls, np.inf, dtype=np.float64)
                    right_vec = np.full(ls, np.inf, dtype=np.float64)
                    for t in range(ls):
                        s2 = (l + 1) + t
                        lv = get_zhx_with_collapse(eddy_rivas_fold_state.zhx_matrix,
                                                   eddy_rivas_fold_state.vxu_matrix, i, j, k, s2)
                        rv = wxI(eddy_rivas_fold_state, l, s2 - 1)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv

                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        s2_star = (l + 1) + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_SPLIT_RIGHT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=s2_star
                        )

                # Case 7: IS2 motif (Irreducible Surface of order 2).
                if self.cfg.enable_is2:
                    for r in range(i, k + 1):
                        for s2 in range(l, j + 1):
                            if r <= k and l <= s2 and r <= s2:
                                inner = get_zhx_with_collapse(eddy_rivas_fold_state.zhx_matrix,
                                                              eddy_rivas_fold_state.vxu_matrix, r, s2, k, l)
                                cand = IS2_outer(seq, self.cfg.tables, i, j, r, s2) + inner
                                if cand < best:
                                    best = cand
                                    best_bp = EddyRivasBackPointer(
                                        op=EddyRivasBacktrackOp.RE_VHX_IS2_INNER_ZHX,
                                        outer=(i, j), hole=(k, l), bridge=(r, s2)
                                    )

                # Case 8: Form a multiloop enclosing a smaller gapped structure.
                close = get_whx_with_collapse(eddy_rivas_fold_state.whx_matrix,
                                              eddy_rivas_fold_state.wxu_matrix, i + 1, j - 1, k - 1, l + 1)
                if math.isfinite(close):
                    cand = 2.0 * tilde_p_hole + tilde_m_vhx + close + internal_pk_penalty + tilde_m_whx
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_CLOSE_BOTH,
                            outer=(i, j), hole=(k, l)
                        )

                # Case 9: Wrap a smaller gapped structure in a multiloop.
                wrap = get_whx_with_collapse(eddy_rivas_fold_state.whx_matrix,
                                             eddy_rivas_fold_state.wxu_matrix, i + 1, j - 1, k, l)
                cand = tilde_p_hole + tilde_m_vhx + wrap + internal_pk_penalty + tilde_m_whx
                if cand < best:
                    best = cand
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_VHX_WRAP_WHX,
                        outer=(i, j), hole=(k, l)
                    )

                eddy_rivas_fold_state.vhx_matrix.set(i, j, k, l, best)
                eddy_rivas_fold_state.vhx_back_ptr.set(i, j, k, l, best_bp)

    # --------- ZHX ---------
    def _dp_zhx(
        self,
        seq: str,
        eddy_rivas_fold_state: EddyRivasFoldState,
        internal_pk_penalty: float,
        tilde_p_hole: float,
        tilde_q_hole: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        Fills the ZHX gap matrix using dynamic programming.

        ZHX(i, j: k, l) stores the minimum free energy for a structure where
        the outer span (i, j) is **closed by a base pair**, but the pairing
        status of the inner hole endpoints (k, l) is undetermined.

        Parameters
        ----------
        seq : str
            The RNA sequence.
        eddy_rivas_fold_state : EddyRivasFoldState
            The state object containing all DP matrices.
        internal_pk_penalty : float
            Penalty for forming an internal pseudoknot (Gwi).
        tilde_p_hole : float
            Penalty for a base pair adjacent to the hole (~P).
        tilde_q_hole : float
            Penalty for an unpaired base in the hole (~Q).
        can_pair_mask : np.ndarray
            A boolean matrix indicating allowed base pairs.

        Notes
        -----
        The recursion for ZHX involves several cases:
        - **From VHX**: The primary case where the undetermined hole (k,l) in
          ZHX becomes a defined pair, transitioning from a VHX subproblem.
        - **Dangles**: Adding dangling bases next to the (k,l) pair, which
          also derives from a VHX subproblem.
        - **Add Unpaired Base**: Adding a single-stranded base to the 5' or 3'
          side of the hole, recursing on a smaller ZHX state.
        - **Bifurcation**: Splitting the region between i and k (or l and j)
          into a nested part (WX) and another gapped part (ZHX).
        - **IS2 Motif**: Forming an Irreducible Surface of order 2 with an
          inner VHX structure.
        """
        spans = list(iter_spans(eddy_rivas_fold_state.seq_len))
        for i, j in tqdm(spans, desc="ZHX", leave=False):
            for k, l in iter_holes_pairable(i, j, can_pair_mask):
                # Apply hole width and beam search filters
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                if self.cfg.beam_v_threshold != 0.0:
                    if eddy_rivas_fold_state.vxu_matrix.get(k, l) > self.cfg.beam_v_threshold:
                        continue

                best = math.inf
                best_bp: Optional[EddyRivasBackPointer] = None

                # Case 1: Form a pair at (k,l), transitioning from VHX.
                v = eddy_rivas_fold_state.vhx_matrix.get(i, j, k, l)
                if math.isfinite(v):
                    cand = tilde_p_hole + v + internal_pk_penalty
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_FROM_VHX, outer=(i, j),
                                                       hole=(k, l))

                # Case 2: Dangles around the newly formed (k,l) pair.
                # Dangles on both sides of the inner pair (k-1, l+1).
                v = eddy_rivas_fold_state.vhx_matrix.get(i, j, k - 1, l + 1)
                if math.isfinite(v):
                    Lh = dangle_hole_left(seq, k, self.cfg.costs)
                    Rh = dangle_hole_right(seq, l, self.cfg.costs)
                    cand = Lh + Rh + tilde_p_hole + v + internal_pk_penalty
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_DANGLE_LR, outer=(i, j),
                                                       hole=(k, l))

                # DANGLE_R from VHX
                v = eddy_rivas_fold_state.vhx_matrix.get(i, j, k - 1, l)
                if math.isfinite(v):
                    Rh = dangle_hole_right(seq, l - 1, self.cfg.costs)
                    cand = Rh + tilde_p_hole + v + internal_pk_penalty
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_DANGLE_R, outer=(i, j),
                                                       hole=(k, l))

                # DANGLE_L from VHX
                v = eddy_rivas_fold_state.vhx_matrix.get(i, j, k, l + 1)
                if math.isfinite(v):
                    Lh = dangle_hole_left(seq, k + 1, self.cfg.costs)
                    cand = Lh + tilde_p_hole + v + internal_pk_penalty
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_DANGLE_L, outer=(i, j),
                                                       hole=(k, l))

                # Case 3: Add an unpaired base to the 5' or 3' side of the hole.
                # This section includes special tie-breaking logic.
                # SS_LEFT: Add base on the 5' side.
                v = eddy_rivas_fold_state.zhx_matrix.get(i, j, k - 1, l)
                if math.isfinite(v):
                    cand = tilde_q_hole + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_SS_LEFT, outer=(i, j),
                                                       hole=(k, l))

                # SS_RIGHT: Add base on the 3' side (with tie-breaking).
                v = eddy_rivas_fold_state.zhx_matrix.get(i, j, k, l + 1)
                if math.isfinite(v):
                    cand = tilde_q_hole + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT,
                            outer=(i, j), hole=(k, l)
                        )
                    elif (cand == best and isinstance(best_bp, EddyRivasBackPointer) and
                          best_bp.op in (EddyRivasBacktrackOp.RE_ZHX_SS_LEFT,
                                         EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT)):
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT,
                            outer=(i, j), hole=(k, l)
                        )

                # Case 4: Bifurcation into ZHX + WX.
                # Split on the 5' side.
                lr = max(0, k - i)
                if lr > 0:
                    left_vec = np.full(lr, np.inf, dtype=np.float64)
                    right_vec = np.full(lr, np.inf, dtype=np.float64)
                    for t in range(lr):
                        r = i + t
                        lv = eddy_rivas_fold_state.zhx_matrix.get(i, j, r, l)
                        rv = wxI(eddy_rivas_fold_state, r + 1, k)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv
                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        r_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SPLIT_LEFT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=r_star
                        )

                # Split on the 3' side.
                ls = max(0, j - l)
                if ls > 0:
                    left_vec = np.full(ls, np.inf, dtype=np.float64)
                    right_vec = np.full(ls, np.inf, dtype=np.float64)
                    for t in range(ls):
                        s2 = (l + 1) + t
                        lv = eddy_rivas_fold_state.zhx_matrix.get(i, j, k, s2)
                        rv = wxI(eddy_rivas_fold_state, l, s2 - 1)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv
                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        s2_star = (l + 1) + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SPLIT_RIGHT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=s2_star
                        )

                # Case 5: IS2 motif (Irreducible Surface of order 2).
                if self.cfg.enable_is2:
                    for r in range(i, k + 1):
                        for s2 in range(l, j + 1):
                            if r <= s2:
                                inner = eddy_rivas_fold_state.vhx_matrix.get(r, s2, k, l)
                                if math.isfinite(inner):
                                    bridge = IS2_outer(seq, self.cfg.tables, i, j, r, s2)
                                    cand = bridge + inner
                                    if cand < best:
                                        best = cand
                                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_IS2_INNER_VHX,
                                                                       outer=(i, j), hole=(k, l), bridge=(r, s2))

                eddy_rivas_fold_state.zhx_matrix.set(i, j, k, l, best)
                eddy_rivas_fold_state.zhx_back_ptr.set(i, j, k, l, best_bp)

    # --------- YHX ---------
    def _dp_yhx(
        self,
        seq: str,
        re: EddyRivasFoldState,
        q: float,
        Gwi: float,
        P_out: float,
        Q_out: float,
        M_yhx: float,
        M_whx: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        YHX: outer helix context around an inner (k,l) helix.
        Now iterates only pairable (k,l) and respects min/max hole width.
        """
        for i, j in iter_spans(re.seq_len):
            for k, l in iter_holes_pairable(i, j, can_pair_mask):
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                if self.cfg.beam_v_threshold != 0.0:
                    if re.vxu_matrix.get(k, l) > self.cfg.beam_v_threshold:
                        continue

                best = math.inf
                best_bp: Optional[EddyRivasBackPointer] = None

                # Outer dangle L
                v = re.vhx_matrix.get(i + 1, j, k, l)
                if math.isfinite(v):
                    Lo = dangle_outer_left(seq, i, self.cfg.costs)
                    cand = Lo + P_out + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_DANGLE_L, outer=(i, j), hole=(k, l))

                # Outer dangle R
                v = re.vhx_matrix.get(i, j - 1, k, l)
                if math.isfinite(v):
                    Ro = dangle_outer_right(seq, j, self.cfg.costs)
                    cand = Ro + P_out + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_DANGLE_R,
                                                       outer=(i, j), hole=(k, l))

                # Outer dangle LR
                v = re.vhx_matrix.get(i + 1, j - 1, k, l)
                if math.isfinite(v):
                    Lo = dangle_outer_left(seq, i, self.cfg.costs)
                    Ro = dangle_outer_right(seq, j, self.cfg.costs)
                    cand = Lo + Ro + P_out + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_DANGLE_LR, outer=(i, j),
                                                       hole=(k, l))

                # SS trims: Left
                v = re.yhx_matrix.get(i + 1, j, k, l)
                if math.isfinite(v):
                    cand = Q_out + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_SS_LEFT, outer=(i, j),
                                                       hole=(k, l))

                # SS trims: Right (flip on tie)
                v = re.yhx_matrix.get(i, j - 1, k, l)
                if math.isfinite(v):
                    cand = Q_out + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_YHX_SS_RIGHT,
                            outer=(i, j), hole=(k, l)
                        )
                    elif (cand == best and isinstance(best_bp, EddyRivasBackPointer) and
                          best_bp.op in (EddyRivasBacktrackOp.RE_YHX_SS_LEFT,
                                         EddyRivasBacktrackOp.RE_YHX_SS_RIGHT)):
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_YHX_SS_RIGHT,
                            outer=(i, j), hole=(k, l)
                        )

                # SS both sides
                v = re.yhx_matrix.get(i + 1, j - 1, k, l)
                if math.isfinite(v):
                    cand = 2.0 * Q_out + v
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_SS_BOTH, outer=(i, j),
                                                       hole=(k, l))

                # Wrap via WHX(i,j:k-1,l+1)
                v = re.whx_matrix.get(i, j, k - 1, l + 1)
                if math.isfinite(v):
                    cand = P_out + M_yhx + M_whx + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX, outer=(i, j), hole=(k, l))

                # Wrap + outer dangles
                v = re.whx_matrix.get(i + 1, j, k - 1, l + 1)
                if math.isfinite(v):
                    Lo = dangle_outer_left(seq, i, self.cfg.costs)
                    cand = Lo + P_out + M_yhx + M_whx + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_L, outer=(i, j), hole=(k, l))

                v = re.whx_matrix.get(i, j - 1, k - 1, l + 1)
                if math.isfinite(v):
                    Ro = dangle_outer_right(seq, j, self.cfg.costs)
                    cand = Ro + P_out + M_yhx + M_whx + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_R, outer=(i, j), hole=(k, l))

                v = re.whx_matrix.get(i + 1, j - 1, k - 1, l + 1)
                if math.isfinite(v):
                    Lo = dangle_outer_left(seq, i, self.cfg.costs)
                    Ro = dangle_outer_right(seq, j, self.cfg.costs)
                    cand = Lo + Ro + P_out + M_yhx + M_whx + v + Gwi
                    if cand < best:
                        best = cand
                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR, outer=(i, j), hole=(k, l))

                # Outer splits with WX
                Lr = max(0, j - i)
                if Lr > 0:
                    left_vec = np.full(Lr, np.inf, dtype=np.float64)
                    right_vec = np.full(Lr, np.inf, dtype=np.float64)
                    for t in range(Lr):
                        r = i + t
                        lv = re.yhx_matrix.get(i, r, k, l)
                        rv = wxI(re, r + 1, j)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv
                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        r_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_YHX_SPLIT_LEFT_YHX_WX,
                            outer=(i, j), hole=(k, l), split=r_star
                        )

                Ls = max(0, j - i)
                if Ls > 0:
                    left_vec = np.full(Ls, np.inf, dtype=np.float64)
                    right_vec = np.full(Ls, np.inf, dtype=np.float64)
                    for t in range(Ls):
                        s2 = i + t
                        lv = wxI(re, i, s2)
                        rv = re.yhx_matrix.get(s2 + 1, j, k, l)
                        if math.isfinite(lv): left_vec[t] = lv
                        if math.isfinite(rv): right_vec[t] = rv
                    cand_split, t_star = best_sum(left_vec, right_vec)
                    if t_star >= 0 and cand_split < best:
                        s2_star = i + t_star
                        best = cand_split
                        best_bp = EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_YHX_SPLIT_RIGHT_WX_YHX,
                            outer=(i, j), hole=(k, l), split=s2_star
                        )

                # IS2 (optional, expensive)
                if self.cfg.enable_is2:
                    for r2 in range(i, k + 1):
                        for s2 in range(l, j + 1):
                            if r2 <= s2:
                                inner_w = get_whx_with_collapse(re.whx_matrix, re.wxu_matrix, r2, s2, k, l)
                                if math.isfinite(inner_w):
                                    bridge = IS2_outer_yhx(self.cfg, seq, i, j, r2, s2)
                                    cand = bridge + inner_w
                                    if cand < best:
                                        best = cand
                                        best_bp = EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_YHX_IS2_INNER_WHX,
                                                                       outer=(i, j), hole=(k, l), bridge=(r2, s2))


                re.yhx_matrix.set(i, j, k, l, best)
                re.yhx_back_ptr.set(i, j, k, l, best_bp)

    # --------- WX Composition & Publish ---------
    def _compose_wx(
            self,
            seq: str,
            re: EddyRivasFoldState,
            Gw: float,
            Gwh_wx: float,
            can_pair_mask: list[list[bool]],
    ) -> None:
        """
        Compose WX from uncharged (WXU) + charged candidates built around pairable (k,l).
        Per-(k,l) we precompute vector components over r and let the kernel minimize across r.
        """
        spans = list(iter_spans(re.seq_len))
        for i, j in tqdm(spans, desc="WX Compose", leave=False):
            best_c = re.wxc_matrix.get(i, j)
            best_bp: Optional[EddyRivasBackPointer] = None

            # iterate pairable holes only
            for (k, l) in iter_holes_pairable(i, j, can_pair_mask):
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                # Beam pruning on inner helix
                if self.cfg.beam_v_threshold != 0.0:
                    v_inner = re.vxu_matrix.get(k, l)
                    if v_inner > self.cfg.beam_v_threshold:
                        continue

                # Precompute r-range vectors: r ∈ [k, l-1]
                L = l - k
                Lu = np.full(L, np.inf, dtype=np.float64)
                Ru = np.full(L, np.inf, dtype=np.float64)
                Lc = np.full(L, np.inf, dtype=np.float64)
                Rc = np.full(L, np.inf, dtype=np.float64)
                left_y = np.full(L, np.inf, dtype=np.float64)
                right_y = np.full(L, np.inf, dtype=np.float64)

                for t in range(L):
                    r = k + t
                    if self.cfg.strict_complement_order:
                        if not (i < k <= r < l <= j):
                            continue
                    # min outer lengths mask
                    if (r - i) < self.cfg.min_outer_left or (j - (r + 1)) < self.cfg.min_outer_right:
                        continue

                    # collapse terms (uncharged/charged)
                    Lu[t] = whx_collapse_with(re, i, r, k, r, charged=False, can_pair_mask=can_pair_mask)
                    Ru[t] = whx_collapse_with(re, k + 1, j, r + 1, l, charged=False, can_pair_mask=can_pair_mask)
                    Lc[t] = whx_collapse_with(re, i, r, k, r, charged=True, can_pair_mask=can_pair_mask)
                    Rc[t] = whx_collapse_with(re, k + 1, j, r + 1, l, charged=True, can_pair_mask=can_pair_mask)

                    if can_pair_mask[k][r]:
                        ly = re.yhx_matrix.get(i, r, k, r)
                        if math.isfinite(ly): left_y[t] = ly

                    if can_pair_mask[r + 1][l]:
                        ry = re.yhx_matrix.get(k + 1, j, r + 1, l)
                        if math.isfinite(ry): right_y[t] = ry

                    # YHX terms (left and right). IMPORTANT: right uses l (not l-1)
                    ly = re.yhx_matrix.get(i, r, k, r)
                    ry = re.yhx_matrix.get(k + 1, j, r + 1, l)
                    if math.isfinite(ly): left_y[t] = ly
                    if math.isfinite(ry): right_y[t] = ry

                cap_pen = short_hole_penalty(self.cfg.costs, k, l)

                # Kernel: best over r for this (k,l)
                cand, t_star, case_id = compose_wx_best_over_r_arrays(
                    Lu, Ru, Lc, Rc, left_y, right_y, float(Gw), float(cap_pen)
                )

                # Right after the pairability checks and kernel call:
                if i == 0 and j == re.seq_len - 1 and (k, l) == (28, 59):
                    print(f"\n[HOLE (28,59) EVAL]", flush=True)
                    print(f"  Built vectors for L={L}", flush=True)
                    finite_Lu = np.sum(np.isfinite(Lu))
                    finite_Ru = np.sum(np.isfinite(Ru))
                    finite_ly = np.sum(np.isfinite(left_y))
                    finite_ry = np.sum(np.isfinite(right_y))
                    print(f"  Finite: Lu={finite_Lu}/{L}, Ru={finite_Ru}/{L}, ly={finite_ly}/{L}, ry={finite_ry}/{L}",
                          flush=True)

                    # Check specific positions
                    for test_r in [28, 35, 43, 51, 54, 58]:
                        t = test_r - k
                        if 0 <= t < L:
                            print(
                                f"  r={test_r}: Lu={Lu[t]:.2f}, Ru={Ru[t]:.2f}, ly={left_y[t]:.2f}, ry={right_y[t]:.2f}",
                                flush=True)

                if i == 0 and j == re.seq_len - 1 and (k, l) == (28, 59):
                    r_star = k + t_star if t_star >= 0 else -1
                    print(f"  Kernel: cand={cand:.2f}, r={r_star}, case={case_id}", flush=True)
                    if cand < best_c:
                        print(f"  ✓ NEW WINNER! (beats current best {best_c:.2f})", flush=True)
                    else:
                        print(f"  ✗ LOSES to current best {best_c:.2f}", flush=True)

                    if r_star >= 0:
                        whx_l = re.whx_matrix.get(i, r_star, k, r_star)
                        whx_r = re.whx_matrix.get(r_star + 1, j, r_star + 1, l)
                        yhx_l = re.yhx_matrix.get(i, r_star, k, r_star)
                        yhx_r = re.yhx_matrix.get(r_star + 1, j, r_star + 1, l)
                        print(
                            f"  At r={r_star}: WHX_L={whx_l:.2f}, WHX_R={whx_r:.2f}, YHX_L={yhx_l:.2f}, YHX_R={yhx_r:.2f}",
                            flush=True)

                        left_ok = math.isfinite(whx_l) or math.isfinite(yhx_l)
                        right_ok = math.isfinite(whx_r) or math.isfinite(yhx_r)
                        if not (left_ok and right_ok):
                            print(f"  ✗ WILL BE FILTERED! (left_ok={left_ok}, right_ok={right_ok})", flush=True)


                if cand < best_c:
                    r_star = k + t_star if t_star >= 0 else -1

                    if r_star >= 0:
                        whx_l_sparse = re.whx_matrix.get(i, r_star, k, r_star)
                        whx_r_sparse = re.whx_matrix.get(r_star + 1, j, r_star + 1, l)
                        yhx_l_sparse = re.yhx_matrix.get(i, r_star, k, r_star)
                        yhx_r_sparse = re.yhx_matrix.get(r_star + 1, j, r_star + 1, l)

                        # Need at least one finite SPARSE value on each side
                        left_has_structure = math.isfinite(whx_l_sparse) or math.isfinite(yhx_l_sparse)
                        right_has_structure = math.isfinite(whx_r_sparse) or math.isfinite(yhx_r_sparse)

                        if not (left_has_structure and right_has_structure):
                            continue  # Not a true PK - at least one side is collapsed baseline

                    # OK, valid PK candidate with both sides having structure
                    best_c = cand
                    r_star = k + t_star

                    # Map kernel case -> backpointer op + holes
                    if case_id in (0, 1, 2, 3):
                        # WHX + WHX
                        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX
                        hole_left = (k, r_star)
                        hole_right = (r_star + 1, l)  # WHX uses l
                    elif case_id == 4:
                        # YHX + YHX
                        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX
                        hole_left = (k, r_star)
                        hole_right = (r_star + 1, l)  # FIXED: YHX uses l
                    elif case_id in (5, 6):
                        # YHX + WHX
                        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_WHX
                        hole_left = (k, r_star)
                        hole_right = (r_star + 1, l)  # WHX uses l
                    else:  # case_id in (7, 8): WHX + YHX
                        op = EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_WHX_YHX
                        hole_left = (k, r_star)
                        hole_right = (r_star + 1, l)  # FIXED: YHX uses l

                    k_l, l_l = hole_left
                    k_r, l_r = hole_right
                    if (l_l - k_l <= 1) or (l_r - k_r <= 1):
                        continue

                    best_bp = EddyRivasBackPointer(
                        op=op,
                        outer=(i, j),
                        hole=(k, l),
                        hole_left=hole_left,
                        hole_right=hole_right,
                        split=r_star,
                        charged=True,
                    )

            # Optional overlap path
            if self.cfg.enable_wx_overlap and Gwh_wx != 0.0:
                for (k2, l2) in iter_inner_holes(i, j, min_hole=self.cfg.min_hole_width):
                    for r2 in range(i, j):
                        left_yv = re.yhx_matrix.get(i, r2, k2, l2)
                        right_yv = re.yhx_matrix.get(r2 + 1, j, k2, l2)
                        if math.isfinite(left_yv) and math.isfinite(right_yv):
                            cand_overlap = (
                                    Gwh_wx + left_yv + right_yv + short_hole_penalty(self.cfg.costs, k2, l2)
                            )
                            if cand_overlap < best_c:
                                best_c = cand_overlap
                                best_bp = EddyRivasBackPointer(
                                    op=EddyRivasBacktrackOp.RE_PK_COMPOSE_WX_YHX_OVERLAP,
                                    outer=(i, j),
                                    hole=(k2, l2),
                                    split=r2,
                                    charged=True,
                                )

            # Commit value + backpointer
            re.wxc_matrix.set(i, j, best_c)
            if best_bp is not None:
                re.wx_back_ptr.set(i, j, best_bp)

            if i == 0 and j == re.seq_len - 1:
                print(f"\n[WX FINAL] best_c={best_c:.2f}, best_bp={best_bp}", flush=True)
                if best_bp and hasattr(best_bp, 'hole'):
                    k_win, l_win = best_bp.hole
                    r_win = best_bp.split
                    print(f"  Winner: hole=({k_win},{l_win}), split={r_win}", flush=True)

                    # Check what values contributed
                    whx_l = re.whx_matrix.get(i, r_win, k_win, r_win)
                    whx_r = re.whx_matrix.get(r_win + 1, j, r_win + 1, l_win)
                    yhx_l = re.yhx_matrix.get(i, r_win, k_win, r_win)
                    yhx_r = re.yhx_matrix.get(r_win + 1, j, r_win + 1, l_win)

                    print(f"  WHX_L={whx_l:.2f}, WHX_R={whx_r:.2f}", flush=True)
                    print(f"  YHX_L={yhx_l:.2f}, YHX_R={yhx_r:.2f}", flush=True)

                    # Check backpointers
                    whx_bp_l = re.whx_back_ptr.get(i, r_win, k_win, r_win)
                    whx_bp_r = re.whx_back_ptr.get(r_win + 1, j, r_win + 1, l_win)
                    print(f"  WHX_L backptr: {whx_bp_l}", flush=True)
                    print(f"  WHX_R backptr: {whx_bp_r}", flush=True)

    def _publish_wx(self, re: EddyRivasFoldState) -> None:
        for i, j in iter_spans(re.seq_len):
            wxu = re.wxu_matrix.get(i, j)
            wxc = re.wxc_matrix.get(i, j)
            if self.cfg.enable_wx_overlap and not math.isfinite(wxc):
                re.wxc_matrix.set(i, j, wxu)
                wxc = wxu
            if wxu <= wxc:
                re.wx_matrix.set(i, j, wxu)
                re.wx_back_ptr.set(i, j, EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED))
            else:
                re.wx_matrix.set(i, j, wxc)

    # --------- VX Composition & Publish ---------
    def _compose_vx(
        self,
        seq: str,
        re: EddyRivasFoldState,
        Gw: float,
        g: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        Compose VX from uncharged (VXU) + charged candidates around pairable (k,l).
        Numba-accelerated: precompute vector components over r and minimize in kernel.
        """
        spans = list(iter_spans(re.seq_len))
        for i, j in tqdm(spans, desc="VX Compose", leave=False):
            best_c = re.vxc_matrix.get(i, j)
            best_bp: Optional[EddyRivasBackPointer] = None

            for (k, l) in iter_holes_pairable(i, j, can_pair_mask):
                hole_w = (l - k - 1)
                if self.cfg.min_hole_width and hole_w < self.cfg.min_hole_width:
                    continue
                if self.cfg.max_hole_width and hole_w > self.cfg.max_hole_width:
                    continue

                L = l - k
                Lu = np.full(L, np.inf, dtype=np.float64)
                Ru = np.full(L, np.inf, dtype=np.float64)
                Lc = np.full(L, np.inf, dtype=np.float64)
                Rc = np.full(L, np.inf, dtype=np.float64)
                coax_total = np.zeros(L, dtype=np.float64)
                coax_bonus = np.zeros(L, dtype=np.float64)

                for t in range(L):
                    r = k + t
                    if self.cfg.strict_complement_order:
                        if not (i < k <= r < l <= j):
                            continue
                    if (r - i) < self.cfg.min_outer_left or (j - (r + 1)) < self.cfg.min_outer_right:
                        continue

                    # collapse terms for VX composition
                    Lu[t] = zhx_collapse_with(re, i, r, k, r, charged=False, can_pair_mask=can_pair_mask)
                    Ru[t] = zhx_collapse_with(re, k + 1, j, r + 1, l, charged=False, can_pair_mask=can_pair_mask)
                    Lc[t] = zhx_collapse_with(re, i, r, k, r, charged=True, can_pair_mask=can_pair_mask)
                    Rc[t] = zhx_collapse_with(re, k + 1, j, r + 1, l, charged=True, can_pair_mask=can_pair_mask)

                    # coax terms for this r
                    adjacent = (r == k)
                    cx_total, cx_bonus = coax_pack(
                        seq, i, j, r, k, l, self.cfg, self.cfg.costs, adjacent
                    )
                    coax_total[t] = cx_total
                    coax_bonus[t] = cx_bonus

                cap_pen = short_hole_penalty(self.cfg.costs, k, l)

                cand, t_star, base_case = compose_vx_best_over_r(
                    Lu, Ru, Lc, Rc, coax_total, coax_bonus,
                    float(Gw), float(cap_pen), float(g)
                )

                if cand < best_c:
                    best_c = cand
                    r_star = k + t_star
                    best_bp = EddyRivasBackPointer(
                        op=EddyRivasBacktrackOp.RE_PK_COMPOSE_VX,
                        outer=(i, j), hole=(k, l), split=r_star, charged=True
                    )

            re.vxc_matrix.set(i, j, best_c)
            if best_bp is not None:
                re.vx_back_ptr.set(i, j, best_bp)

    @staticmethod
    def _publish_vx(re: EddyRivasFoldState) -> None:
        for i, j in iter_spans(re.seq_len):
            vxu = re.vxu_matrix.get(i, j)
            vxc = re.vxc_matrix.get(i, j)
            if vxu <= vxc:
                re.vx_matrix.set(i, j, vxu)
                re.vx_back_ptr.set(i, j, EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED))
            else:
                re.vx_matrix.set(i, j, vxc)


def load_costs_json(path: str) -> PseudoknotEnergies:
    with open(path, "r") as fh:
        d = json.load(fh)
    return costs_from_dict(d)

def save_costs_json(path: str, costs: PseudoknotEnergies) -> None:
    with open(path, "w") as fh:
        json.dump(costs_to_dict(costs), fh, indent=2, sort_keys=True)

def costs_from_dict(d: Dict) -> PseudoknotEnergies:
    """Create RERECosts from a flat dict; keys not present use dataclass defaults."""
    field_names = {f.name for f in fields(PseudoknotEnergies)}
    kwargs = {k: v for k, v in d.items() if k in field_names}
    return PseudoknotEnergies(**kwargs)

def costs_to_dict(costs: PseudoknotEnergies) -> Dict:
    """Round-trip exporter useful for saving tuned params."""
    return {f.name: getattr(costs, f.name) for f in fields(PseudoknotEnergies)}

def costs_from_vienna_like(tbl: Dict[str, Any]) -> PseudoknotEnergies:
    """
    Map a Vienna-like dict to RERECosts.
    Expected keys (suggested, adapt to your source):
      - 'q_ss', 'Gw', 'Gwi', 'Gwh', 'coax_scale', 'coax_bonus',
      - 'coax_pairs': { "GC|CG": -0.5, "AU|UA": -0.3, ... },
      - 'dangle_outer_left/R', 'dangle_hole_left/R': { "GA": -0.1, ... },
      - 'short_hole_caps': { "1": 2.0, "2": 1.0 },
      - optional: 'mismatch_coax_scale', 'mismatch_coax_bonus',
                  'coax_min_helix_len', 'coax_scale_oo/oi/io',
                  'P_tilde_out', 'P_tilde_hole', 'Q_tilde_out', 'Q_tilde_hole',
                  'M_tilde_yhx', 'M_tilde_vhx', 'M_tilde_whx'.
    """
    d = {}

    # simple scalars
    for k in [
        "q_ss","Gwh","Gwi","coax_scale","coax_bonus",
        "mismatch_coax_scale","mismatch_coax_bonus",
        "coax_min_helix_len","coax_scale_oo","coax_scale_oi","coax_scale_io",
        "P_tilde_out","P_tilde_hole","Q_tilde_out","Q_tilde_hole",
        "L_tilde", "R_tilde", "M_tilde_yhx","M_tilde_vhx","M_tilde_whx"
    ]:
        if k in tbl: d[k] = tbl[k]

    # coax pairs: accept "XY|UV" keys
    coax_pairs = {}
    for key, val in tbl.get("coax_pairs", {}).items():
        left,right = key.split("|")
        coax_pairs[(left, right)] = float(val)
    d["coax_pairs"] = coax_pairs

    # dangles: accept bigrams like "GA": value
    for name in ["dangle_outer_left","dangle_outer_right","dangle_hole_left","dangle_hole_right"]:
        m = {}
        for bigram, val in tbl.get(name, {}).items():
            if len(bigram) == 2:
                m[(bigram[0], bigram[1])] = float(val)
        d[name] = m

    # short-hole caps
    caps = {}
    for h, val in tbl.get("short_hole_caps", {}).items():
        caps[int(h)] = float(val)
    d["short_hole_caps"] = caps

    return costs_from_dict(d)

def quick_energy_harness(seq: str, cfg: EddyRivasFoldingConfig, nested: ZuckerFoldState, re: EddyRivasFoldState) -> Dict[str, float]:
    """
    Run fill_with_costs and report a few sentinel energies for regression:
    """
    eng = EddyRivasFoldingEngine(cfg)
    eng.fill_with_costs(seq, nested, re)
    out = {
        "W(0,n-1)": re.wx_matrix.get(0, re.seq_len - 1),
        "V(0,n-1)": re.vx_matrix.get(0, re.seq_len - 1),
    }
    # Add any other coordinates you want to track here.
    return out

