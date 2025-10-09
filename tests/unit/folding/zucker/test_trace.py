def test_wm_36_debug():
    """Debug WM[3,6] energy"""
    from rna_pk_fold.folding.zucker import make_fold_state
    from rna_pk_fold.folding.zucker.zucker_recurrences import ZuckerFoldingEngine, ZuckerFoldingConfig
    from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModel
    from rna_pk_fold.energies import SecondaryStructureEnergyLoader
    from importlib.resources import files as ir_files
    import rna_pk_fold

    yaml_path = ir_files(rna_pk_fold) / "data" / "turner2004_eddyrivas1999_min.yaml"
    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=yaml_path)
    energy_model = SecondaryStructureEnergyModel(params=params, temp_k=310.15)
    config = ZuckerFoldingConfig(temp_k=310.15, verbose=False)

    seq = "GCAUCUAUGC"
    engine = ZuckerFoldingEngine(energy_model=energy_model, config=config)
    state = make_fold_state(len(seq))
    engine.fill_all_matrices(seq, state)

    a, b, c, d = params.MULTILOOP

    print(f"\n=== WM[3,6] Debug ===")
    print(f"Sequence [3:7]: {seq[3:7]}")  # UCUA
    print(f"WM[3,6] = {state.wm_matrix.get(3, 6):.2f}")

    # Check options
    print(f"\nOptions for WM[3,6]:")
    print(f"  Unpaired left: WM[4,6] + c = {state.wm_matrix.get(4, 6):.2f} + {c} = {state.wm_matrix.get(4, 6) + c:.2f}")
    print(
        f"  Unpaired right: WM[3,5] + c = {state.wm_matrix.get(3, 5):.2f} + {c} = {state.wm_matrix.get(3, 5) + c:.2f}")

    print(f"\n  Attach helix options:")
    for k in range(4, 7):
        from rna_pk_fold.rules import can_pair
        if can_pair(seq[3], seq[k]):
            v_3k = state.wm_matrix.get(3, k)
            tail = 0.0 if k + 1 > 6 else state.wm_matrix.get(k + 1, 6)
            print(f"    k={k} ({seq[3]}-{seq[k]}): V[3,{k}]={v_3k:.2f}, tail={tail:.2f}, total={b + v_3k + tail:.2f}")

    wm_bp = state.wm_back_ptr.get(3, 6)
    print(f"\nWM[3,6] chose: {wm_bp.operation}")
    if wm_bp.inner:
        print(f"  inner: {wm_bp.inner}")