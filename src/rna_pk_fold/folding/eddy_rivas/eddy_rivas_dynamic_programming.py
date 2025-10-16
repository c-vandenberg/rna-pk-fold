import math
import time
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from rna_pk_fold.energies.energy_types import PseudoknotEnergies
from rna_pk_fold.folding.zucker.zucker_fold_state import ZuckerFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import EddyRivasFoldState
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_back_pointer import EddyRivasBackPointer, EddyRivasBacktrackOp
from rna_pk_fold.utils.sequences.iter_utils import iter_spans, iter_holes_pairable
from rna_pk_fold.utils.dynamic_programming.matrix_utils import (clear_matrix_lookup_caches, get_whx_energy_with_collapse,
                                                                get_zhx_energy_with_collapse)
from rna_pk_fold.rules.constraints import build_can_pair_mask
from rna_pk_fold.utils.dynamic_programming.dp_gap_matrix_utils import (
    should_skip_dp_cell, BestCandidateTracker, update_tracker_for_vhx_inner_dangles, scan_is2_outer_min_bridge,
    update_tracker_for_vhx_multiloop_close_and_wrap, update_tracker_for_hole_dangles_from_vhx, update_tracker_for_hole_ss_right_tiebreak,
    update_tracker_for_outer_dangles_from_vhx, update_tracker_for_outer_ss_right_tiebreak, update_tracker_for_yhx_wrap_whx,
    update_tracker_for_outer_ss_both, update_tracker_for_whx_hole_shrinks, update_tracker_for_whx_outer_trims, update_tracker_for_whx_collapse,
    update_tracker_for_whx_single_strand_both, update_tracker_for_whx_splits, update_tracker_for_whx_overlap_split, update_tracker_for_whx_is2
)
from rna_pk_fold.utils.dynamic_programming.dp_composition_utils import (evaluate_wx_composition_for_hole,
                                                                        evaluate_wx_yhx_overlap_for_span,
                                                                        set_span_cell_with_backpointer,
                                                                        evaluate_vx_composition_for_hole)
from rna_pk_fold.utils.dynamic_programming.dp_split_utils import (zhx_wx_split_min_vhx, compute_zhx_split_min_over_zhx_wx,
                                                                  yhx_wx_split_min, VhxSplitMode, ZhxSplitMode,
                                                                  YhxSplitMode)
from rna_pk_fold.utils.dynamic_programming.dp_publish_utils import (use_nested_energy_if_composed_infinite,
                                                                    publish_min_energy_with_default_backpointer)
from rna_pk_fold.utils.logging.debug_utils import debug_print, count_finite_cells
from rna_pk_fold.utils.logging.logging_utils import setup_logger

logger = setup_logger(
    name=__name__,
    level=logging.DEBUG,
    console_level=logging.INFO,
    file_level=logging.DEBUG,
)


# -----------------------
# Configuration
# -----------------------
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
    enable_coax: bool = True
    enable_wx_overlap: bool = False
    enable_coax_variants: bool = False
    enable_coax_mismatch: bool = False
    enable_join_drift: bool = False
    drift_radius: int = 0
    enable_is2: bool = False
    pk_penalty_gw: float = 1.0
    max_hole_width: int = 0
    min_hole_width: int = 0
    min_outer_left: int = 0
    min_outer_right: int = 0
    beam_k: int = 0
    beam_v_threshold: float = 0.0
    strict_complement_order: bool = True
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
        self.config = config
        self.timings = {}

    def run_eddy_rivas_dp_with_costs(
        self,
        seq: str,
        nested_state: ZuckerFoldState,
        eddy_rivas_fold_state: EddyRivasFoldState
    ) -> None:
        """
        Executes the main Eddy-Rivas dynamic programming algorithm.

        This method drives the entire folding process. It initializes the DP
        matrices, fills the gap matrices, and then composes the final matrices
        to find the optimal folding energy including pseudoknots.

        Parameters
        ----------
        seq : str
            The RNA sequence to fold.
        nested_state : ZuckerFoldState
            A pre-filled state object containing the results of a nested-only
            (e.g., Zuker) folding algorithm.
        eddy_rivas_fold_state : EddyRivasFoldState
            The state object that will be populated with the results of this
            algorithm. It contains all the DP matrices.

        Notes
        -----
        The algorithm follows a precise sequence of steps as outlined in the
        Rivas and Eddy paper:

        1.  Seeding: The process begins by populating the primary DP
            matrices, `wx` (best energy for subsequence `i` to `j`) and `vx`
            (best energy given `i` and `j` are paired), with the results from
            the `nested_fold_state`. This establishes a baseline of optimal
            non-pseudoknotted structures.

        2.  Gap Matrix Filling (O(N⁴) Complexity): This is the core of the
            pseudoknot detection. The algorithm fills four "gap matrices" that
            store energies for structures spanning two disconnected segments,
            `[i..k]` and `[l..j]`, leaving a "hole" `[k+1..l-1]`.
            - `vhx(i,j:k,l)`: Energy where `(i,j)` and `(k,l)` are both paired.
            - `zhx(i,j:k,l)`: Energy where `(i,j)` is paired, `(k,l)` is not.
            - `yhx(i,j:k,l)`: Energy where `(k,l)` is paired, `(i,j)` is not.
            - `whx(i,j:k,l)`: Energy where pairing of `(i,j)` and `(k,l)` is undetermined.
            These are filled iteratively, building larger gapped structures
            from smaller nested and gapped ones.

        3.  Composition (O(N⁶) Complexity): After the gap matrices are
            complete, this phase updates the `wx` and `vx` matrices by
            considering all possible ways to form a pseudoknot. For each span
            `(i, j)`, the algorithm iterates through all split points `r` to
            combine two complementary gapped fragments, one spanning `[i..r]`
            and the other `[r+1..j]`. This step finds the minimum energy by
            either maintaining the existing nested structure or introducing a
            more stable pseudoknotted one.

        4.  Final Energy: The optimal free energy for the entire sequence
            is the value stored in `wx[0, n-1]`. The structure itself can be
            reconstructed via a traceback procedure using the backpointers
            stored during the DP fill.
        """
        total_start = time.perf_counter()

        seq_len = eddy_rivas_fold_state.seq_len

        # --- Log header ---
        logger.info("=" * 60)
        logger.info(f"Eddy-Rivas DP for sequence length N={seq_len}")
        logger.info(f"Expected complexity:")
        logger.info(f"  Gap matrices: O(N⁴) ≈ {seq_len ** 4:,} operations")
        logger.info(f"  Compositions: O(N⁶) ≈ {seq_len ** 6:,} operations")
        logger.info("=" * 60)

        # Reset any memoized matrix lookups (module-level caches).
        clear_matrix_lookup_caches()

        # --- Load model configuration
        config = self._load_config()
        q_ss = config["q_ss"]
        g_w = config["g_w"]
        g_wh = config["g_wh"]  # kept for completeness
        g_wi = config["g_wi"]
        g_wh_wx = config["g_wh_wx"]
        g_wh_whx = config["g_wh_whx"]
        g_coax_scale = config["g"]
        p_out = config["p_out"]
        p_hole = config["p_hole"]
        l_tilde = config["l_tilde"]
        r_tilde = config["r_tilde"]
        q_tilde_out = config["q_tilde_out"]
        q_tilde_hole = config["q_tilde_hole"]
        m_tilde_yhx = config["m_tilde_yhx"]
        m_tilde_vhx = config["m_tilde_vhx"]
        m_tilde_whx = config["m_tilde_whx"]

        # --- Phase 1: Seeding ---
        seed_start = time.perf_counter()
        self._seed_from_nested(nested_state, eddy_rivas_fold_state)

        # Densify working matrices that will be heavily written.
        eddy_rivas_fold_state.wxu_matrix.enable_dense()
        eddy_rivas_fold_state.wxc_matrix.enable_dense()
        eddy_rivas_fold_state.vxu_matrix.enable_dense()
        eddy_rivas_fold_state.vxc_matrix.enable_dense()

        can_pair_mask = build_can_pair_mask(seq)

        self.timings['seed'] = time.perf_counter() - seed_start
        logger.info(f"Seeding completed in {self.timings['seed']:.2f}s")

        # --- Phase 2: Gap Matrix Filling ---
        # WHX
        logger.info("Filling WHX matrix...")
        whx_start = time.perf_counter()
        self._fill_whx_gap_matrix(seq, eddy_rivas_fold_state, q_ss, g_wh_whx, can_pair_mask)
        self.timings['whx'] = time.perf_counter() - whx_start
        logger.info(f"WHX filled in {self.timings['whx']:.2f}s")

        # Targeted debug checks (guarded)
        debug_print(
            self.config,
            f"[REF CHECK] WHX[0,33:23,33] = {eddy_rivas_fold_state.whx_matrix.get(0, 33, 23, 33):.2f}"
        )
        debug_print(
            self.config,
            f"[REF CHECK] WHX[34,69:63,68] = {eddy_rivas_fold_state.whx_matrix.get(34, 69, 63, 68):.2f}"
        )

        # VHX
        logger.info("Filling VHX matrix...")
        vhx_start = time.perf_counter()
        self._fill_vhx_gap_matrix(
            seq, eddy_rivas_fold_state,
            g_wi, p_hole, l_tilde, r_tilde,
            q_tilde_hole, m_tilde_vhx, m_tilde_whx,
            can_pair_mask
        )
        self.timings['vhx'] = time.perf_counter() - vhx_start
        logger.info(f"VHX filled in {self.timings['vhx']:.2f}s")

        # ZHX
        logger.info("Filling ZHX matrix...")
        zhx_start = time.perf_counter()
        self._fill_zhx_gap_matrix(seq, eddy_rivas_fold_state, g_wi, p_hole, q_tilde_hole, can_pair_mask)
        self.timings['zhx'] = time.perf_counter() - zhx_start
        logger.info(f"ZHX filled in {self.timings['zhx']:.2f}s")

        # YHX
        logger.info("Filling YHX matrix...")
        yhx_start = time.perf_counter()
        self._fill_yhx_gap_matrix(
            seq, eddy_rivas_fold_state,
            g_wi, p_out, q_tilde_out,
            m_tilde_yhx, m_tilde_whx,
            can_pair_mask
        )
        self.timings['yhx'] = time.perf_counter() - yhx_start
        logger.info(f"YHX filled in {self.timings['yhx']:.2f}s")

        # Optional inspection (guarded)
        debug_print(
            self.config,
            f"YHX[37,42:37,40] = {eddy_rivas_fold_state.yhx_matrix.get(37, 42, 37, 40):.2f}"
        )
        debug_print(self.config, f"YHX BP: {eddy_rivas_fold_state.yhx_back_ptr.get(37, 42, 37, 40)}")

        # Gap stats (using helper)
        whx_count = count_finite_cells(eddy_rivas_fold_state.whx_matrix)
        yhx_count = count_finite_cells(eddy_rivas_fold_state.yhx_matrix)
        zhx_count = count_finite_cells(eddy_rivas_fold_state.zhx_matrix)
        vhx_count = count_finite_cells(eddy_rivas_fold_state.vhx_matrix)
        debug_print(self.config, "\n[GAP STATS]")
        debug_print(self.config, f"  WHX: {whx_count} finite cells")
        debug_print(self.config, f"  YHX: {yhx_count} finite cells")
        debug_print(self.config, f"  ZHX: {zhx_count} finite cells")
        debug_print(self.config, f"  VHX: {vhx_count} finite cells")

        # --- Phase 3: Composition & Publish ---
        # WX Composition & Publish
        logger.info("Composing WX matrix...")
        wx_start = time.perf_counter()
        self._compose_wx_from_gapped_fragments(seq, eddy_rivas_fold_state, g_w, g_wh_wx, can_pair_mask)
        self._publish_wx_min_energy(eddy_rivas_fold_state)
        self.timings['wx_compose'] = time.perf_counter() - wx_start
        logger.info(f"WX composed in {self.timings['wx_compose']:.2f}s")

        # VX Composition & Publish
        logger.info("Composing VX matrix...")
        vx_start = time.perf_counter()
        self._compose_vx_from_zhx_fragments(seq, eddy_rivas_fold_state, g_w, g_coax_scale, can_pair_mask)
        self._publish_vx_min_energy(eddy_rivas_fold_state)
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
            f"  Seeding:        {self.timings['seed']:7.2f}s ({self.timings['seed'] / self.timings['total'] * 100:5.1f}%)"
        )
        logger.info(
            f"  WHX fill:       {self.timings['whx']:7.2f}s ({self.timings['whx'] / self.timings['total'] * 100:5.1f}%)"
        )
        logger.info(
            f"  VHX fill:       {self.timings['vhx']:7.2f}s ({self.timings['vhx'] / self.timings['total'] * 100:5.1f}%)"
        )
        logger.info(
            f"  ZHX fill:       {self.timings['zhx']:7.2f}s ({self.timings['zhx'] / self.timings['total'] * 100:5.1f}%)"
        )
        logger.info(
            f"  YHX fill:       {self.timings['yhx']:7.2f}s ({self.timings['yhx'] / self.timings['total'] * 100:5.1f}%)"
        )
        logger.info(
            f"  WX composition: {self.timings['wx_compose']:7.2f}s ({self.timings['wx_compose'] / self.timings['total'] * 100:5.1f}%)"
        )
        logger.info(
            f"  VX composition: {self.timings['vx_compose']:7.2f}s ({self.timings['vx_compose'] / self.timings['total'] * 100:5.1f}%)"
        )

        gap_total = self.timings['whx'] + self.timings['vhx'] + self.timings['zhx'] + self.timings['yhx']
        comp_total = self.timings['wx_compose'] + self.timings['vx_compose']

        logger.info(f"  Gap matrices:   {gap_total:7.2f}s ({gap_total / self.timings['total'] * 100:5.1f}%)")
        logger.info(f"  Compositions:   {comp_total:7.2f}s ({comp_total / self.timings['total'] * 100:5.1f}%)")
        logger.info("=" * 60)

    def _load_config(self):
        costs_config = self.config.costs
        config_tables = getattr(self.config, "tables", None)
        return dict(
            q_ss=costs_config.q_ss,
            g_w=self.config.pk_penalty_gw,
            g_wh=getattr(costs_config, "Gwh", 0.0),
            g_wi=costs_config.g_wi,
            g_wh_wx=getattr(costs_config, "Gwh_wx", 0.0),
            g_wh_whx=getattr(costs_config, "Gwh_whx", 0.0),
            g=costs_config.coax_scale,
            p_out=getattr(config_tables, "P_tilde_out", getattr(costs_config, "P_tilde_out", 1.0)),
            p_hole=getattr(config_tables, "P_tilde_hole", getattr(costs_config, "P_tilde_hole", 1.0)),
            l_tilde=getattr(config_tables, "L_tilde", 0.0),
            r_tilde=getattr(costs_config, "R_tilde", 0.0),
            q_tilde_out=getattr(config_tables, "Q_tilde_out", getattr(costs_config, "Q_tilde_out", 0.0)),
            q_tilde_hole=getattr(config_tables, "Q_tilde_hole", getattr(costs_config, "Q_tilde_hole", 0.0)),
            m_tilde_yhx=getattr(config_tables, "M_tilde_yhx", getattr(costs_config, "M_tilde_yhx", 0.0)),
            m_tilde_vhx=getattr(config_tables, "M_tilde_vhx", getattr(costs_config, "M_tilde_vhx", 0.0)),
            m_tilde_whx=getattr(config_tables, "M_tilde_whx", getattr(costs_config, "M_tilde_whx", 0.0)),
        )

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
        seq_len = eddy_rivas_fold_state.seq_len
        for i, j in iter_spans(seq_len):
            # Nested (Zucker) Energies
            base_wx_energy = nested_fold_state.w_matrix.get(i, j)
            base_vx_energy = nested_fold_state.v_matrix.get(i, j)

            # Uncomposed (Baseline) Energies
            eddy_rivas_fold_state.wxu_matrix.set(i, j, base_wx_energy)
            eddy_rivas_fold_state.vxu_matrix.set(i, j, base_vx_energy)

            # Composed energies start as +inf for non-trivial spans
            if i != j:
                eddy_rivas_fold_state.wxc_matrix.set(i, j, math.inf)
                eddy_rivas_fold_state.vxc_matrix.set(i, j, math.inf)

            # Final matrices start with the nested baseline
            eddy_rivas_fold_state.wx_matrix.set(i, j, base_wx_energy)
            eddy_rivas_fold_state.vx_matrix.set(i, j, base_vx_energy)

            if getattr(eddy_rivas_fold_state, "wxi_matrix", None) is not None:
                eddy_rivas_fold_state.wxi_matrix.set(i, j, base_wx_energy)

    # --------- WHX ---------
    def _fill_whx_gap_matrix(
        self,
        seq: str,
        eddy_rivas_fold_state: EddyRivasFoldState,
        unpaired_base_penalty: float,
        overlap_penalty: float,
        can_pair_mask: list[list[bool]]
    ) -> None:
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
                # ---------- Guards/Filters (Hole Width, Beam Threshold, Watson-Crick Base Pairing) ----------
                if should_skip_dp_cell(i, j, k, l, self.config, eddy_rivas_fold_state.vxu_matrix.get,
                                       can_pair_mask=can_pair_mask, require_kl_pairable=True):
                    continue

                # ---------- Targeted Debug Probes ----------
                debug_cell = (i, j, k, l) == (0, 33, 23, 33)
                debug_print(debug_cell, "\n[WHX DEBUG] Filling (0,33:23,33)")

                # ---------- Initialize Best Candidate Tracker ----------
                tracker = BestCandidateTracker()

                # ---------- Cases 1 & 2: Add An Unpaired (Single Stranded) Base Either Ends of the Hole. ----------
                update_tracker_for_whx_hole_shrinks(tracker, eddy_rivas_fold_state, i, j, k, l, unpaired_base_penalty)

                # ---------- Cases 3 & 4: Add an Unpaired Base at Either Ends of the Outer Span. ----------
                update_tracker_for_whx_outer_trims(tracker, eddy_rivas_fold_state, i, j, k, l, unpaired_base_penalty)

                # ---------- Case 5: Collapse The Hole to Form a Nested Structure WX(i, j). ----------
                update_tracker_for_whx_collapse(tracker, eddy_rivas_fold_state, i, j, k, l)

                # ---------- Case 6: Add Unpaired Bases at Both Outer Ends. ----------
                update_tracker_for_whx_single_strand_both(tracker, eddy_rivas_fold_state, i, j, k, l, unpaired_base_penalty)

                # ---------- Cases 7 & 8: Left & Right Splits into (WHX + WX) & (WX+WHX). ----------
                update_tracker_for_whx_splits(tracker, eddy_rivas_fold_state, i, j, k, l)

                # ---------- Case 9: Overlapping Pseudoknot Splut into (WHX + WHX) with Penalty. ----------
                update_tracker_for_whx_overlap_split(tracker, eddy_rivas_fold_state, i, j, k, l, overlap_penalty)

                # ---------- Case 10: IS2 motif (Outer Bridge + Inner YHX). ----------
                if self.config.enable_is2:
                    update_tracker_for_whx_is2(tracker, eddy_rivas_fold_state, self.config, seq, i, j, k, l)

                # -------- Publish Cell --------
                debug_print(debug_cell, f"  FINAL: best={tracker.best_energy:.2f} bp={tracker.backpointer}")
                eddy_rivas_fold_state.whx_matrix.set(i, j, k, l, tracker.best_energy)
                eddy_rivas_fold_state.whx_back_ptr.set(i, j, k, l, tracker.backpointer)

                if i == 0 and j >= 33 and 20 <= k <= 30 <= l <= 35:
                    status = "SUCCESS" if math.isfinite(tracker.best_energy) else "FAIL"
                    print(f"[WHX {status}] ({i},{j}:{k},{l}) = {tracker.best_energy:.2f}", flush=True)

    # --------- VHX ---------
    def _fill_vhx_gap_matrix(
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
        - Dangles: Adding a dangling base next to the (k,l) pair inside
          the hole.
        - Unpaired Base: Adding a single-stranded base adjacent to the hole,
          transitioning from a ZHX state.
        - Bifurcation: Splitting the region between the outer and inner
          helices into a nested part (WX) and a gapped part (ZHX).
        - IS2 Motif: Forming an Irreducible Surface of order 2 by bridging
          the (i, j) pair with an inner ZHX structure.
        - Multiloop: Closing a multiloop around a WHX subproblem.
        """
        spans = list(iter_spans(eddy_rivas_fold_state.seq_len))
        for i, j in tqdm(spans, desc="VHX", leave=False):
            for k, l in iter_holes_pairable(i, j, can_pair_mask):
                # ---------- Guards/Filters (Hole Width, Beam Threshold) ----------
                if should_skip_dp_cell(i, j, k, l, self.config, eddy_rivas_fold_state.vxu_matrix.get):
                    continue

                # ---------- Initialize Best Candidate Tracker ----------
                tracker = BestCandidateTracker(best_energy=eddy_rivas_fold_state.vhx_matrix.get(i, j, k, l))

                # -------- Cases 1, 2 & 3: Inner Pair (k,l) Dangles. --------
                update_tracker_for_vhx_inner_dangles(
                    tracker, eddy_rivas_fold_state.vhx_matrix.get,
                    i, j, k, l,
                    tilde_p_hole, tilde_l_hole, tilde_r_hole,
                    EddyRivasBacktrackOp.RE_VHX_DANGLE_L,
                    EddyRivasBacktrackOp.RE_VHX_DANGLE_R,
                    EddyRivasBacktrackOp.RE_VHX_DANGLE_LR,
                )

                # -------- Case 4: Add an Unpaired Base in the Hole (From ZHX) With Tie-Break to Right. --------
                zhx_energy = get_zhx_energy_with_collapse(
                    eddy_rivas_fold_state.zhx_matrix,
                    eddy_rivas_fold_state.vxu_matrix,
                    i, j, k, l
                )
                if math.isfinite(zhx_energy):
                    tracker.update_pair_with_right_tiebreak(
                        tilde_q_hole + zhx_energy,  # Left view
                        tilde_q_hole + zhx_energy,  # Right view (same energy)
                        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_VHX_SS_LEFT, outer=(i, j), hole=(k, l)),
                        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_VHX_SS_RIGHT, outer=(i, j), hole=(k, l)),
                    )

                # -------- Cases 5: Split on the 5' (Left Side) - r in [i..k-1]  →  ZHX(i,j:r,l) + WX(r+1,k) --------
                cand_left, t_left = zhx_wx_split_min_vhx(
                    VhxSplitMode.LEFT_ZHX_WX, eddy_rivas_fold_state, i, j, k, l
                )
                if t_left >= 0:
                    r_star = i + t_left
                    tracker.update_if_better(
                        cand_left,
                        EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_SPLIT_LEFT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=r_star
                        ),
                    )

                # -------- Case 6: Split on the 3' (Right) Side - s2 in [l+1..j]  →  ZHX(i,j:k,s2) + WX(l, s2-1) --------
                cand_right, t_right = zhx_wx_split_min_vhx(
                    VhxSplitMode.RIGHT_ZHX_WX, eddy_rivas_fold_state, i, j, k, l
                )
                if t_right >= 0:
                    s2_star = (l + 1) + t_right
                    tracker.update_if_better(
                        cand_right,
                        EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_VHX_SPLIT_RIGHT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=s2_star
                        ),
                    )

                # -------- Case 7: IS2 (Outer Bridge + Inner ZHX) --------
                if self.config.enable_is2:
                    is2_best, is2_bp, _ = scan_is2_outer_min_bridge(
                        eddy_rivas_fold_state, self.config, seq, i, j, k, l,
                        inner_matrix="zhx", # VHX uses ZHX as the inner problem here
                        bridge_kind="default",
                        op=EddyRivasBacktrackOp.RE_VHX_IS2_INNER_ZHX
                    )
                    if is2_bp is not None:
                        r2, s2 = is2_bp
                        tracker.update_if_better(
                            is2_best, EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_VHX_IS2_INNER_ZHX,
                                outer=(i, j), hole=(k, l), bridge=(r2, s2)
                            )
                        )

                # -------- Case 8: Multiloop - Close Around/Wrap on WHX Sub-problem --------
                update_tracker_for_vhx_multiloop_close_and_wrap(
                    tracker,
                    lambda idx_i, idx_j, idx_k, idx_l: get_whx_energy_with_collapse(
                        eddy_rivas_fold_state.whx_matrix,
                        eddy_rivas_fold_state.wxu_matrix, idx_i, idx_j, idx_k, idx_l
                    ),
                    i, j, k, l,
                    tilde_p_hole, tilde_m_vhx, tilde_m_whx, internal_pk_penalty,
                    EddyRivasBacktrackOp.RE_VHX_CLOSE_BOTH,
                    EddyRivasBacktrackOp.RE_VHX_WRAP_WHX,
                )

                # -------- Publish Cell --------
                eddy_rivas_fold_state.vhx_matrix.set(i, j, k, l, tracker.best_energy)
                eddy_rivas_fold_state.vhx_back_ptr.set(i, j, k, l, tracker.backpointer)

    # --------- ZHX ---------
    def _fill_zhx_gap_matrix(
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
                # ---------- Guards/Filters (Hole Width, Beam Threshold) ----------
                if should_skip_dp_cell(i, j, k, l, self.config, eddy_rivas_fold_state.vxu_matrix.get):
                    continue

                # ---------- Initialize Best Candidate Tracker ----------
                tracker = BestCandidateTracker()

                # ---------- Case 1: From VHX, Form a pair at (k,l) ----------
                vhx_energy = eddy_rivas_fold_state.vhx_matrix.get(i, j, k, l)
                if math.isfinite(vhx_energy):
                    tracker.update_if_better(
                        tilde_p_hole + vhx_energy + internal_pk_penalty,
                        EddyRivasBackPointer(op=EddyRivasBacktrackOp.RE_ZHX_FROM_VHX, outer=(i, j), hole=(k, l))
                    )

                # ---------- Case 2: Dangles around the newly formed (k,l) pair from VHX. ----------
                update_tracker_for_hole_dangles_from_vhx(
                    tracker, eddy_rivas_fold_state.vhx_matrix.get,
                    seq, self.config.costs,
                    i, j, k, l,
                    tilde_p_hole, internal_pk_penalty,
                    EddyRivasBacktrackOp.RE_ZHX_DANGLE_L,
                    EddyRivasBacktrackOp.RE_ZHX_DANGLE_R,
                    EddyRivasBacktrackOp.RE_ZHX_DANGLE_LR,
                )

                # ---------- Case 3: Add an Unpaired (Single Stranded) Base to the 5' or 3' side of the hole. Tie-Break to Right ----------
                update_tracker_for_hole_ss_right_tiebreak(
                    tracker, eddy_rivas_fold_state.zhx_matrix.get,
                    i, j, k, l, tilde_q_hole,
                    EddyRivasBacktrackOp.RE_ZHX_SS_LEFT,
                    EddyRivasBacktrackOp.RE_ZHX_SS_RIGHT,
                )

                # ---------- Case 4: Split into ZHX + WX. ----------
                # 4.1. Split on the 5' (Left) Side: ZHX(i,j:r,l) + WX(r+1,k)
                cand_left, t_left = compute_zhx_split_min_over_zhx_wx(
                    ZhxSplitMode.LEFT_ZHX_WX, eddy_rivas_fold_state, i, j, k, l
                )
                if t_left >= 0:
                    r_star = i + t_left
                    tracker.update_if_better(
                        cand_left,
                        EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SPLIT_LEFT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=r_star
                        ),
                    )

                # 4.2. Split on the 3' (Right) Side: ZHX(i,j:k,s2) + WX(l, s2-1)
                cand_right, t_right = compute_zhx_split_min_over_zhx_wx(
                    ZhxSplitMode.RIGHT_ZHX_WX, eddy_rivas_fold_state, i, j, k, l
                )
                if t_right >= 0:
                    s2_star = (l + 1) + t_right
                    tracker.update_if_better(
                        cand_right,
                        EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_ZHX_SPLIT_RIGHT_ZHX_WX,
                            outer=(i, j), hole=(k, l), split=s2_star
                        ),
                    )

                # ---------- Case 5: IS2 Motif (Outer Bridge + Inner VHX)). ----------
                if self.config.enable_is2:
                    is2_best, is2_bp, _ = scan_is2_outer_min_bridge(
                        eddy_rivas_fold_state, self.config, seq, i, j, k, l,
                        inner_matrix="vhx", bridge_kind="default",
                        op=EddyRivasBacktrackOp.RE_ZHX_IS2_INNER_VHX
                    )
                    if is2_bp is not None:
                        r2, s2 = is2_bp
                        tracker.update_if_better(
                            is2_best, EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_ZHX_IS2_INNER_VHX,
                                outer=(i, j), hole=(k, l), bridge=(r2, s2)
                            )
                        )

                # -------- Publish Cells --------
                eddy_rivas_fold_state.zhx_matrix.set(i, j, k, l, tracker.best_energy)
                eddy_rivas_fold_state.zhx_back_ptr.set(i, j, k, l, tracker.backpointer)

    # --------- YHX ---------
    def _fill_yhx_gap_matrix(
        self,
        seq: str,
        eddy_rivas_fold_state: EddyRivasFoldState,
        internal_pk_penalty: float,
        tilde_p_out: float,
        tilde_q_out: float,
        tilde_m_yhx: float,
        tilde_m_whx: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        Fills the YHX gap matrix using dynamic programming.

        YHX(i, j: k, l) stores the minimum free energy for a structure where
        the inner hole span (k, l) is **closed by a base pair**, but the pairing
        status of the outer endpoints (i, j) is undetermined. This is the
        symmetric counterpart to the ZHX matrix.

        Parameters
        ----------
        seq : str
            The RNA sequence.
        eddy_rivas_fold_state : EddyRivasFoldState
            The state object containing all DP matrices.
        internal_pk_penalty : float
            Penalty for forming an internal pseudoknot (Gwi).
        tilde_p_out : float
            Penalty for a base pair in the outer span (~P).
        tilde_q_out : float
            Penalty for an unpaired base in the outer span (~Q).
        tilde_m_yhx : float
            Penalty for a multiloop originating from a YHX state (~M).
        tilde_m_whx : float
            Penalty for a multiloop originating from a WHX state (~M).
        can_pair_mask : np.ndarray
            A boolean matrix indicating allowed base pairs.

        Notes
        -----
        The recursion for YHX mirrors that of ZHX but acts on the outer span:
        - **Dangles**: Adding dangling bases next to the (i, j) pair, deriving
          from a VHX subproblem.
        - **Add Unpaired Base**: Adding a single-stranded base to the 5' or 3'
          end of the outer span (i.e., trimming), recursing on a smaller YHX.
        - **Bifurcation**: Splitting the outer span into a nested part (WX) and
          another gapped part (YHX).
        - **Multiloop Wrap**: Forming a multiloop by closing the (i,j) pair
          around a WHX subproblem.
        - **IS2 Motif**: Forming an Irreducible Surface of order 2 with an
          inner WHX structure.
        """
        for i, j in iter_spans(eddy_rivas_fold_state.seq_len):
            for k, l in iter_holes_pairable(i, j, can_pair_mask):
                # ---------- Guards/Filters (Hole Width, Beam Threshold) ----------
                if should_skip_dp_cell(i, j, k, l, self.config, eddy_rivas_fold_state.vxu_matrix.get):
                    continue

                # ---------- Initialize Best Candidate Tracker ----------
                tracker = BestCandidateTracker()

                # ---------- Case 1: Dangles on the Outer Pair (i,j) From VHX. ----------
                update_tracker_for_outer_dangles_from_vhx(
                    tracker, eddy_rivas_fold_state.vhx_matrix.get,
                    seq, self.config.costs,
                    i, j, k, l,
                    tilde_p_out, internal_pk_penalty,
                    EddyRivasBacktrackOp.RE_YHX_DANGLE_L,
                    EddyRivasBacktrackOp.RE_YHX_DANGLE_R,
                    EddyRivasBacktrackOp.RE_YHX_DANGLE_LR,
                )

                # ---------- Case 2: Add an Unpaired Base to the Outer Span (Trimming). Tie-Break to Right ----------
                update_tracker_for_outer_ss_right_tiebreak(
                    tracker, eddy_rivas_fold_state.yhx_matrix.get,
                    i, j, k, l, tilde_q_out,
                    EddyRivasBacktrackOp.RE_YHX_SS_LEFT,
                    EddyRivasBacktrackOp.RE_YHX_SS_RIGHT,
                )
                both_trim_val = eddy_rivas_fold_state.yhx_matrix.get(i + 1, j - 1, k, l)
                update_tracker_for_outer_ss_both(
                    tracker, both_trim_val, tilde_q_out, i, j, k, l,
                    EddyRivasBacktrackOp.RE_YHX_SS_BOTH
                )

                # ---------- Case 3: Multiloop Wrap of WHX. ----------
                update_tracker_for_yhx_wrap_whx(
                    tracker, eddy_rivas_fold_state.whx_matrix.get,
                    seq, self.config.costs,
                    i, j, k, l,
                    tilde_p_out, tilde_m_yhx, tilde_m_whx, internal_pk_penalty,
                    EddyRivasBacktrackOp.RE_YHX_WRAP_WHX,
                    EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_L,
                    EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_R,
                    EddyRivasBacktrackOp.RE_YHX_WRAP_WHX_LR,
                )

                # ---------- Case 4: Split of the Outer Span Into YHX + WX. ----------
                # 4.1. Left Split: YHX(i, r) + WX(r+1, j)
                cand_left, t_left = yhx_wx_split_min(
                    YhxSplitMode.LEFT_YHX_WX, eddy_rivas_fold_state, i, j, k, l
                )
                if t_left >= 0:
                    r_star = i + t_left
                    tracker.update_if_better(
                        cand_left,
                        EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_YHX_SPLIT_LEFT_YHX_WX,
                            outer=(i, j), hole=(k, l), split=r_star
                        ),
                    )

                # 4.2. Right Split: WX(i, s) + YHX(s+1, j)
                cand_right, t_right = yhx_wx_split_min(
                    YhxSplitMode.RIGHT_WX_YHX, eddy_rivas_fold_state, i, j, k, l
                )
                if t_right >= 0:
                    s_star = i + t_right
                    tracker.update_if_better(
                        cand_right,
                        EddyRivasBackPointer(
                            op=EddyRivasBacktrackOp.RE_YHX_SPLIT_RIGHT_WX_YHX,
                            outer=(i, j), hole=(k, l), split=s_star
                        ),
                    )

                # ---------- Case 5: IS2 motif (Outer Bridge + Inner WHX. ----------
                if self.config.enable_is2:
                    is2_best, is2_bp, _ = scan_is2_outer_min_bridge(
                        eddy_rivas_fold_state, self.config, seq, i, j, k, l,
                        inner_matrix="whx", bridge_kind="yhx",
                        op=EddyRivasBacktrackOp.RE_YHX_IS2_INNER_WHX
                    )
                    if is2_bp is not None:
                        r2, s2 = is2_bp
                        tracker.update_if_better(
                            is2_best, EddyRivasBackPointer(
                                op=EddyRivasBacktrackOp.RE_YHX_IS2_INNER_WHX,
                                outer=(i, j), hole=(k, l), bridge=(r2, s2)
                            )
                        )

                # ---------- Publish Cells ----------
                eddy_rivas_fold_state.yhx_matrix.set(i, j, k, l, tracker.best_energy)
                eddy_rivas_fold_state.yhx_back_ptr.set(i, j, k, l, tracker.backpointer)

    # --------- WX Composition & Publish ---------
    def _compose_wx_from_gapped_fragments(
        self,
        seq: str,
        eddy_rivas_fold_state: EddyRivasFoldState,
        pseudoknot_penalty: float,
        yhx_overlap_penalty: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        Updates the WX matrix by composing gapped fragments into pseudoknots.

        This is the O(N⁶) core of the algorithm where the optimal non-nested
        structures are identified. It iterates through all spans (i, j), all
        potential holes (k, l) within that span, and all split points (r)
        that can partition the structure into two complementary gapped fragments.
        The minimum energy found across all possible pseudoknotted configurations
        updates the initial nested energy for WX(i, j).

        Parameters
        ----------
        seq : str
            The RNA sequence.
        eddy_rivas_fold_state : EddyRivasFoldState
            The state object containing all DP matrices.
        pseudoknot_penalty : float
            The energy penalty for initiating a pseudoknot (Gw).
        can_pair_mask : np.ndarray
            A boolean matrix indicating allowed base pairs.

        Notes
        -----
        The composition logic is as follows:
        For each span `(i, j)` and hole `(k, l)`:
        1.  A split point `r` is chosen, `k <= r < l`.
        2.  This `r` defines two complementary gapped subproblems:
            - A "left" fragment covering `[i..r]` with a hole `[k..r]`.
            - A "right" fragment covering `[r+1..j]` with a hole `[r+1..l]`.
        3.  The energies for these fragments (from `whx` and `yhx` matrices)
            are combined with the `pseudoknot_penalty`.
        4.  An efficient kernel (`compose_wx_best_over_r_arrays`) finds the
            optimal split point `r` and the best combination of `whx`/`yhx`
            matrices for the given `(i, j, k, l)`.
        5.  The resulting minimum energy is compared to the current best
            composed energy for `WX(i, j)`, and updated if it's better.
        """
        # Iterate over all possible outer spans (i, j) of the structure.
        spans = list(iter_spans(eddy_rivas_fold_state.seq_len))
        for i, j in tqdm(spans, desc="WX Compose", leave=False):

            # Initialize with the best composed energy found so far for this span.
            best_composed_energy = eddy_rivas_fold_state.wxc_matrix.get(i, j)
            best_backpointer: Optional[EddyRivasBackPointer] = None

            # Iterate over all possible inner holes (k, l) that could form a pseudoknot.
            for (k, l) in iter_holes_pairable(i, j, can_pair_mask):
                # ---------- Guards/Filters (Hole Width, Beam Threshold) ----------
                if should_skip_dp_cell(i, j, k, l, self.config, eddy_rivas_fold_state.vxu_matrix.get):
                    continue

                cand_energy, cand_bp = evaluate_wx_composition_for_hole(
                    eddy_rivas_fold_state, self.config, seq, i, j, k, l, pseudoknot_penalty, can_pair_mask
                )
                if cand_energy < best_composed_energy and cand_bp is not None:
                    best_composed_energy, best_backpointer = cand_energy, cand_bp


            if i == 0 and j == eddy_rivas_fold_state.seq_len - 1:
                wxu_val = eddy_rivas_fold_state.wxu_matrix.get(i, j)
                print(
                    f"[COMPOSE END] WXU={wxu_val:.2f}, best_c={best_composed_energy:.2f}, best_bp={best_backpointer}", flush=True
                )

            # Optional YHX-overlap path
            cand_overlap, bp_overlap = evaluate_wx_yhx_overlap_for_span(
                eddy_rivas_fold_state, self.config, i, j, yhx_overlap_penalty
            )
            if cand_overlap < best_composed_energy and bp_overlap is not None:
                best_composed_energy, best_backpointer = cand_overlap, bp_overlap

            # After checking all possible holes (k, l) for the current span (i, j),
            # commit the best result to the composed matrix and its backpointer store.
            set_span_cell_with_backpointer(
                eddy_rivas_fold_state.wxc_matrix,
                eddy_rivas_fold_state.wx_back_ptr,
                i, j,
                best_composed_energy,
                best_backpointer
            )

            # --- Final Debugging Block ---
            # This block prints the final winning configuration for the entire sequence.
            if i == 0 and j == eddy_rivas_fold_state.seq_len - 1:
                print(f"\n[WX FINAL] best_c={best_composed_energy:.2f}, best_bp={best_backpointer}", flush=True)
                if best_backpointer and hasattr(best_backpointer, 'hole'):
                    k_win, l_win = best_backpointer.hole
                    r_win = best_backpointer.split
                    print(f"  Winner: hole=({k_win},{l_win}), split={r_win}", flush=True)

                    # Check what energy values contributed to the winning structure.
                    whx_l = eddy_rivas_fold_state.whx_matrix.get(i, r_win, k_win, r_win)
                    whx_r = eddy_rivas_fold_state.whx_matrix.get(r_win + 1, j, r_win + 1, l_win)
                    yhx_l = eddy_rivas_fold_state.yhx_matrix.get(i, r_win, k_win, r_win)
                    yhx_r = eddy_rivas_fold_state.yhx_matrix.get(r_win + 1, j, r_win + 1, l_win)

                    print(f"  WHX_L={whx_l:.2f}, WHX_R={whx_r:.2f}", flush=True)
                    print(f"  YHX_L={yhx_l:.2f}, YHX_R={yhx_r:.2f}", flush=True)

                    # Check the backpointers of the contributing subproblems.
                    whx_bp_l = eddy_rivas_fold_state.whx_back_ptr.get(i, r_win, k_win, r_win)
                    whx_bp_r = eddy_rivas_fold_state.whx_back_ptr.get(r_win + 1, j, r_win + 1, l_win)
                    print(f"  WHX_L backptr: {whx_bp_l}", flush=True)
                    print(f"  WHX_R backptr: {whx_bp_r}", flush=True)

    # --------- VX Composition & Publish ---------
    def _compose_vx_from_zhx_fragments(
        self,
        seq: str,
        eddy_rivas_fold_state: EddyRivasFoldState,
        pseudoknot_penalty: float,
        coaxial_scale: float,
        can_pair_mask: list[list[bool]],
    ) -> None:
        """
        Updates the VX matrix by composing gapped fragments into pseudoknots.

        This method calculates the energy of the optimal pseudoknotted structure
        that can be formed *within* a closing base pair (i, j). It is analogous
        to the multiloop calculation in a standard nested folding algorithm. This
        is an O(N^6) operation that is accelerated by vectorizing the innermost
        loop and executing it with a Numba-optimized kernel.

        Parameters
        ----------
        seq : str
            The RNA sequence.
        eddy_rivas_fold_state : EddyRivasFoldState
            The state object containing all DP matrices for the Eddy-Rivas fold.
        pseudoknot_penalty : float
            The energy penalty for initiating a pseudoknot.
        coaxial_scale : float
            A scaling factor for coaxial stacking energies in pseudoknots.
        can_pair_mask : list[list[bool]]
            A boolean matrix indicating which nucleotides can form pairs.

        Notes
        -----
        For each closing pair `(i, j)`, the algorithm finds the best pseudoknot
        by iterating through all possible inner holes `(k, l)` and split points
        `r`. The structure is formed by combining two `ZHX` subproblems, as `ZHX`
        requires its outer span to be paired, fitting the context of being
        enclosed by the `(i, j)` pair. This composition also includes energy
        bonuses for coaxial stacking between the inner helices.
        """
        # Iterate over all possible outer spans (i, j) that could form a closing pair.
        spans = list(iter_spans(eddy_rivas_fold_state.seq_len))
        for i, j in tqdm(spans, desc="VX Compose", leave=False):

            # Initialize with the best composed energy found so far for this span.
            best_composed_energy = eddy_rivas_fold_state.vxc_matrix.get(i, j)
            best_backpointer: Optional[EddyRivasBackPointer] = None

            # Iterate over all possible inner holes (k, l) that could form a pseudoknot.
            for (k, l) in iter_holes_pairable(i, j, can_pair_mask):
                # ---------- Guards/Filters (Hole Width, Beam Threshold) ----------
                if should_skip_dp_cell(i, j, k, l, self.config, eddy_rivas_fold_state.vxc_matrix.get):
                    continue

                candidate_energy, candidate_backpointer = evaluate_vx_composition_for_hole(
                    eddy_rivas_fold_state, self.config, seq, i, j, k, l, pseudoknot_penalty, coaxial_scale, can_pair_mask
                )
                if candidate_energy < best_composed_energy and candidate_backpointer is not None:
                    best_composed_energy, best_backpointer = candidate_energy, candidate_backpointer

            # After checking all possible holes (k, l) for the current span (i, j),
            # commit the best result to the composed matrix and its backpointer store.
            set_span_cell_with_backpointer(
                eddy_rivas_fold_state.vxc_matrix,
                eddy_rivas_fold_state.vx_back_ptr,
                i, j,
                best_composed_energy,
                best_backpointer
            )

    def _publish_wx_min_energy(self, eddy_rivas_fold_state: EddyRivasFoldState) -> None:
        """
        Finalizes the WX matrix by selecting the optimal energy for each span.

        This method compares the energy of the best nested-only structure
        (the "uncomposed" energy from `wxu_matrix`) with the energy of the
        best pseudoknotted structure (the "composed" energy from `wxc_matrix`).
        It selects the minimum of the two and populates the final `wx_matrix`.

        Parameters
        ----------
        eddy_rivas_fold_state : EddyRivasFoldState
            The state object containing all DP matrices for the Eddy-Rivas fold.

        Notes
        -----
        For each span (i, j), this function performs the final selection:
        - If `uncomposed_energy <= composed_energy`, the optimal structure is
          nested. The final `wx_matrix` is set to the uncomposed energy, and
          a backpointer is created to indicate this choice.
        - If `composed_energy < uncomposed_energy`, the optimal structure
          contains a pseudoknot. The final `wx_matrix` is set to the composed
          energy. The backpointer for this case was already set during the
          `_compose_wx` step and is implicitly retained.
        """
        # Iterate over all possible spans (i, j) in the sequence.
        for i, j in iter_spans(eddy_rivas_fold_state.seq_len):
            # Retrieve the optimal energy for the nested-only structure for this span.
            # This 'uncharged' energy comes from the initial Zuker-style fold.
            wxu_nested_energy = eddy_rivas_fold_state.wxu_matrix.get(i, j)

            # Retrieve the optimal energy for any pseudoknotted structure for this span.
            # This 'charged' energy was calculated in the _compose_wx step. If `enable_fallback`
            # is `True` and `charged` energy is `+inf`, we use the `uncharged` (Zucker) energy
            # for `wxc`.
            wxc_charged_energy = use_nested_energy_if_composed_infinite(
                enable_overlap_fallback=self.config.enable_wx_overlap,
                charged_matrix=eddy_rivas_fold_state.wxc_matrix,
                nested_energy=wxu_nested_energy,
                i_idx=i,
                j_idx=j
            )

            # --- Final Selection ---
            # Compare the energy of the best nested structure with the best pseudoknotted one.
            publish_min_energy_with_default_backpointer(
                i_idx=i,
                j_idx=j,
                nested_energy=wxu_nested_energy,
                non_nested_energy=wxc_charged_energy,
                final_matrix=eddy_rivas_fold_state.wx_matrix,
                backpointer_store=eddy_rivas_fold_state.wx_back_ptr,
                uncharged_op=EddyRivasBacktrackOp.RE_WX_SELECT_UNCHARGED,
            )

            # Optional debug
            if i == 0 and j == 27:
                existing_bp = eddy_rivas_fold_state.wx_back_ptr.get(i, j)
                print(f"[PUBLISH ELSE] Keeping WXC, existing BP: {existing_bp}", flush=True)

    @staticmethod
    def _publish_vx_min_energy(re: EddyRivasFoldState) -> None:
        """
        Finalizes the VX matrix by selecting the optimal energy for each pair.

        This method compares the energy of the best nested structure enclosed by
        the pair (i, j) (the "uncomposed" energy from `vxu_matrix`) with the
        energy of the best pseudoknotted structure enclosed by the same pair
        (the "composed" energy from `vxc_matrix`). It selects the minimum of
        the two and populates the final `vx_matrix`.

        Parameters
        ----------
        re : EddyRivasFoldState
            The state object containing all DP matrices for the Eddy-Rivas fold.

        Notes
        -----
        For each span (i, j), this function performs the final selection:
        - If `uncomposed_energy <= composed_energy`, the optimal structure
          enclosed by the pair (i, j) is nested.
        - If `composed_energy < uncomposed_energy`, the optimal enclosed
          structure contains a pseudoknot. The backpointer for this case was
          already set during the `_compose_vx` step and is implicitly retained.
        """
        # Iterate over all possible spans (i, j) in the sequence.
        for i, j in iter_spans(re.seq_len):
            # Retrieve the optimal energy for a nested structure enclosed by the pair (i, j).
            # This 'uncomposed' energy is the baseline from the initial Zuker-style fold.
            vxu_nested_energy = re.vxu_matrix.get(i, j)

            # Retrieve the optimal energy for a pseudoknotted structure enclosed by the pair (i, j).
            # This 'composed' energy was calculated in the _compose_vx step.
            vxc_charged_energy = re.vxc_matrix.get(i, j)

            # --- Final Selection ---
            publish_min_energy_with_default_backpointer(
                i_idx=i,
                j_idx=j,
                nested_energy=vxu_nested_energy,
                non_nested_energy=vxc_charged_energy,
                final_matrix=re.vx_matrix,
                backpointer_store=re.vx_back_ptr,
                uncharged_op=EddyRivasBacktrackOp.RE_VX_SELECT_UNCHARGED,
            )
