"""
A diagnostic test module for the Zucker folding algorithm.

This module contains a specific, detailed debugging test designed to provide a
human-readable trace of the dynamic programming calculations. Its purpose is not
for automated validation in a CI/CD pipeline, but rather to serve as a tool for
developers to inspect the step-by-step energy evaluation for a specific cell in
the DP matrix, aiding in debugging the recurrence relations or the energy model.
"""

def test_wm_36_debug():
    """
    Provides a detailed breakdown of the energy calculation for the WM[3,6] cell.

    This function acts as a "probe" into the Zucker folding algorithm. It performs
    the following steps:
    1. Loads the standard Turner 2004 energy parameters.
    2. Initializes and runs the Zucker folding engine on the sequence "GCAUCUAUGC".
    3. After the DP matrices are filled, it focuses on the cell WM[3,6].
    4. It prints the final, optimal energy stored in WM[3,6].
    5. It then re-calculates and prints the energy contributions from each possible
       subproblem (e.g., adding an unpaired base, attaching a new helix) that
       could have led to the final value.
    6. Finally, it prints the backpointer that was stored for WM[3,6], revealing
       which of the possible subproblems was chosen as the optimal path.

    This provides a clear, step-by-step trace that is invaluable for verifying
    the correctness of the recurrence implementation.
    """
    # --- Setup: Load energy model and initialize the folding engine ---
    from rna_pk_fold.folding.zucker import make_fold_state
    from rna_pk_fold.folding.zucker.zucker_recurrences import ZuckerFoldingEngine, ZuckerFoldingConfig
    from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModel
    from rna_pk_fold.energies import SecondaryStructureEnergyLoader
    from importlib.resources import files as ir_files
    import rna_pk_fold

    # Load the Turner 2004 energy parameters from the package data.
    yaml_path = ir_files(rna_pk_fold) / "data" / "turner2004_eddyrivas1999_min.yaml"
    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=yaml_path)
    energy_model = SecondaryStructureEnergyModel(params=params, temp_k=310.15)
    config = ZuckerFoldingConfig(temp_k=310.15, verbose=False)

    # --- Execution: Run the folding algorithm ---
    seq = "GCAUCUAUGC"
    engine = ZuckerFoldingEngine(energy_model=energy_model, config=config)
    state = make_fold_state(len(seq))
    engine.fill_all_matrices(seq, state) # This populates all DP matrices.

    # Unpack multiloop energy parameters for use in print statements.
    a, b, c, d = params.MULTILOOP

    # --- Debug Output: Print a detailed trace for the target cell WM[3,6] ---
    print(f"\n=== WM[3,6] Debug ===")
    print(f"Sequence [3:7]: {seq[3:7]}")  # The subsequence is UCUA
    print(f"WM[3,6] = {state.wm_matrix.get(3, 6):.2f}")

    # Display the energy contributions from the two "unpaired" recurrence options.
    print(f"\nOptions for WM[3,6]:")
    # Option 1: Add an unpaired base on the left (5' side).
    print(f"  Unpaired left: WM[4,6] + c = {state.wm_matrix.get(4, 6):.2f} + {c} = {state.wm_matrix.get(4, 6) + c:.2f}")
    # Option 2: Add an unpaired base on the right (3' side).
    print(
        f"  Unpaired right: WM[3,5] + c = {state.wm_matrix.get(3, 5):.2f} + {c} = {state.wm_matrix.get(3, 5) + c:.2f}")

    # Display contributions from the "attach helix" / bifurcation options.
    print(f"\n  Attach helix options:")
    for k in range(4, 7):
        from rna_pk_fold.rules import can_pair
        # Check if the ends of the potential new helix can form a base pair.
        if can_pair(seq[3], seq[k]):
            # This appears to be the bifurcation WM(i,j) -> V(i,k) + WM(k+1,j).
            # Note: The print statement uses 'V' for clarity, but the value is
            # fetched from the WM matrix in this specific implementation detail.
            v_3k = state.wm_matrix.get(3, k)
            # Energy of the remaining subsequence on the 3' side.
            tail = 0.0 if k + 1 > 6 else state.wm_matrix.get(k + 1, 6)
            print(f"    k={k} ({seq[3]}-{seq[k]}): V[3,{k}]={v_3k:.2f}, tail={tail:.2f}, total={b + v_3k + tail:.2f}")

    # Retrieve and print the backpointer to show which option was chosen.
    wm_bp = state.wm_back_ptr.get(3, 6)
    print(f"\nWM[3,6] chose: {wm_bp.operation}")
    if wm_bp.inner:
        print(f"  inner: {wm_bp.inner}")