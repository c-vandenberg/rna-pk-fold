#!/usr/bin/env python3
"""
Performance evaluation script for the Eddy-Rivas pseudoknot folding algorithm.

This script benchmarks the runtime and memory usage of the $O(N^{6})$
Eddy-Rivas dynamic programming algorithm across a range of sequence lengths.
It analyzes the empirical time complexity and generates plots for visualization.
"""

import time
import tracemalloc
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib.resources import files as importlib_files

# Import folding components (matching predict_rna.py structure)
from rna_pk_fold.energies import SecondaryStructureEnergyLoader
from rna_pk_fold.energies.energy_model import SecondaryStructureEnergyModel
from rna_pk_fold.folding.zucker import make_fold_state as make_zucker_state
from rna_pk_fold.folding.zucker.zucker_recurrences import ZuckerFoldingConfig, ZuckerFoldingEngine
from rna_pk_fold.folding.eddy_rivas import eddy_rivas_recurrences
from rna_pk_fold.folding.eddy_rivas.eddy_rivas_fold_state import init_eddy_rivas_fold_state


def generate_random_sequence(length: int, seed: int = None) -> str:
    """
    Generate a random RNA sequence of a given length.

    Parameters
    ----------
    length : int
        The desired length of the RNA sequence ($N$).
    seed : int, optional
        Seed for the random number generator for reproducibility.
        The default is None.

    Returns
    -------
    str
        A random RNA sequence composed of 'A', 'C', 'G', 'U' bases.
    """
    if seed is not None:
        random.seed(seed)
    return ''.join(random.choices(['A', 'C', 'G', 'U'], k=length))


def load_energy_model(temp_c: float = 37.0) -> SecondaryStructureEnergyModel:
    """
    Load the RNA thermodynamic energy model.

    This function loads Turner 2004 parameters augmented with Rivas & Eddy
    (1999) pseudoknot heuristic parameters.

    Parameters
    ----------
    temp_c : float, optional
        Temperature in Celsius for $\Delta G$ calculations. The default is $37.0$.

    Returns
    -------
    SecondaryStructureEnergyModel
        An initialized energy model object.
    """
    # Use default bundled parameter file
    yaml_path = str(importlib_files("rna_pk_fold") / "data" / "turner2004_eddyrivas1999_min.yaml")

    temp_k = 273.15 + temp_c

    # Load raw parameters from YAML
    params = SecondaryStructureEnergyLoader().load(kind="RNA", yaml_path=yaml_path)

    # Create energy model
    model = SecondaryStructureEnergyModel(params=params, temp_k=temp_k)

    return model


def build_eddy_rivas_costs(energy_model: SecondaryStructureEnergyModel) -> eddy_rivas_recurrences.PseudoknotEnergies:
    """
    Extract and build pseudoknot energy parameters from the loaded energy model.

    Parameters
    ----------
    energy_model : SecondaryStructureEnergyModel
        The loaded energy model object.

    Returns
    -------
    eddy_rivas_recurrences.PseudoknotEnergies
        A structured object containing pseudoknot-specific $\Delta G$ parameters.

    Raises
    ------
    ValueError
        If no pseudoknot parameters are defined in the energy model.
    """
    base_pk_params = energy_model.params.PSEUDOKNOT
    if base_pk_params is None:
        raise ValueError("No pseudoknot parameters found in YAML!")

    return base_pk_params


def eddy_rivas_fold(sequence: str, energy_model: SecondaryStructureEnergyModel) -> dict:
    """
    Fold an RNA sequence using the full Eddy-Rivas algorithm.

    The folding proceeds in two phases: first the nested Zuker component,
    followed by the pseudoknot component.

    Parameters
    ----------
    sequence : str
        The RNA sequence to fold.
    energy_model : SecondaryStructureEnergyModel
        The energy model containing $\Delta G$ parameters.

    Returns
    -------
    dict
        A dictionary containing:
        'energy' : float
            The final minimum free energy of the structure (kcal/mol).
        'length' : int
            The length of the sequence ($N$).
        'state' : object
            The final Eddy-Rivas DP state object.
        'nested_state' : object
            The intermediate Zuker DP state object.
    """
    # Phase 1: Run Zucker (nested) algorithm
    zucker_config = ZuckerFoldingConfig(verbose=False)
    zucker_engine = ZuckerFoldingEngine(energy_model=energy_model, config=zucker_config)
    zucker_state = make_zucker_state(len(sequence))
    zucker_engine.fill_all_matrices(sequence, zucker_state)

    # Phase 2: Run Eddy-Rivas (pseudoknot) algorithm
    er_costs = build_eddy_rivas_costs(energy_model)

    # Configuration matches hardcoded defaults used in the actual prediction script
    er_config = eddy_rivas_recurrences.EddyRivasFoldingConfig(
        enable_coax=False,
        enable_coax_variants=False,
        enable_coax_mismatch=False,
        enable_wx_overlap=False,
        enable_is2=False,
        enable_join_drift=False,
        min_hole_width=0,
        max_hole_width=0,
        pk_penalty_gw=-10.0, # Use favorable PK penalty for testing
        costs=er_costs,
        verbose=False,
    )

    er_engine = eddy_rivas_recurrences.EddyRivasFoldingEngine(er_config)
    eddy_rivas_state = init_eddy_rivas_fold_state(len(sequence))

    # Run DP algorithm
    er_engine.fill_with_costs(sequence, zucker_state, eddy_rivas_state)

    # Get final energy
    final_energy = eddy_rivas_state.wx_matrix.get(0, len(sequence) - 1)

    return {
        'energy': final_energy,
        'length': len(sequence),
        'state': eddy_rivas_state,
        'nested_state': zucker_state
    }


def benchmark_runtime(sequence_lengths: list[int], num_trials: int = 3) -> dict:
    """
    Benchmark the mean runtime across different sequence lengths ($N$).

    Parameters
    ----------
    sequence_lengths : list of int
        List of sequence lengths ($N$) to test.
    num_trials : int, optional
        Number of folding runs per length for averaging. The default is 3.

    Returns
    -------
    dict
        A dictionary containing:
        'lengths' : list of int
            The sequence lengths tested.
        'mean_times' : list of float
            The mean runtime (in seconds) for each length.
        'std_times' : list of float
            The standard deviation of runtime for each length.
        'energies' : list of float
            The final MFE for each tested length.
    """
    # Load energy model once (reused for all sequences)
    print("Loading energy model...")
    energy_model = load_energy_model()

    results = {
        'lengths': sequence_lengths,
        'mean_times': [],
        'std_times': [],
        'energies': []
    }

    for n in sequence_lengths:
        print(f"\nBenchmarking N={n}...")
        trial_times = []

        for trial in range(num_trials):
            # Sequence changes per trial to avoid caching effects
            seq = generate_random_sequence(n, seed=42 + trial)

            start = time.perf_counter()
            result = eddy_rivas_fold(seq, energy_model)
            elapsed = time.perf_counter() - start

            trial_times.append(elapsed)
            print(f"  Trial {trial + 1}/{num_trials}: {elapsed:.2f}s")

        results['mean_times'].append(np.mean(trial_times))
        results['std_times'].append(np.std(trial_times))
        results['energies'].append(result['energy'])

        print(f"  Mean: {results['mean_times'][-1]:.2f}s $\pm$ {results['std_times'][-1]:.2f}s")
        print(f"  Energy: {results['energies'][-1]:.2f} kcal/mol")

    return results


def benchmark_memory(sequence_lengths: list[int]) -> dict:
    """
    Benchmark peak memory usage across different sequence lengths ($N$).

    Uses Python's `tracemalloc` to measure the peak memory allocated
    during the DP calculation.

    Parameters
    ----------
    sequence_lengths : list of int
        List of sequence lengths ($N$) to test.

    Returns
    -------
    dict
        A dictionary containing:
        'lengths' : list of int
            The sequence lengths tested.
        'peak_memory_mb' : list of float
            The peak memory usage (in megabytes) for each length.
    """
    # Load energy model once
    print("\nLoading energy model for memory tests...")
    energy_model = load_energy_model()

    results = {
        'lengths': sequence_lengths,
        'peak_memory_mb': []
    }

    for n in sequence_lengths:
        print(f"\nMeasuring memory for N={n}...")
        seq = generate_random_sequence(n, seed=42)

        tracemalloc.start()
        # tracemalloc tracks peak memory until tracemalloc.stop()
        result = eddy_rivas_fold(seq, energy_model)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 ** 2
        results['peak_memory_mb'].append(peak_mb)

        print(f"  Peak memory: {peak_mb:.2f} MB")
        print(f"  Energy: {result['energy']:.2f} kcal/mol")

    return results


def analyze_complexity(lengths: list[int], times: list[float]) -> tuple[float, np.ndarray]:
    """
    Fit empirical runtime data to the relationship $T \propto N^{k}$ and
    estimate the time complexity exponent $k$.

    This is done by performing a linear regression on the log-log transformed data:
    $\log(T) = k \cdot \log(N) + c$

    Parameters
    ----------
    lengths : list of int
        Sequence lengths ($N$).
    times : list of float
        Mean runtimes ($T$) corresponding to each length.

    Returns
    -------
    tuple of (float, numpy.ndarray)
        The estimated exponent $k$ and an array of fitted times.
    """
    log_n = np.log(lengths)
    log_time = np.log(times)

    # Linear fit in log-log space: log(T) = k*log(N) + c
    coeffs = np.polyfit(log_n, log_time, 1)
    k = coeffs[0]  # Exponent
    c = coeffs[1]  # Constant

    # Generate fitted curve: T = e^c * N^k
    fitted_times = np.exp(c) * np.array(lengths) ** k

    print(f"\n{'=' * 60}")
    print(f"COMPLEXITY ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Empirical complexity: $O(N^{{{k:.2f}}})$")
    print(f"Theoretical (paper):  $O(N^{6})$ (composition) + $O(N^{4})$ (gaps)")

    if k < 6.0:
        print(f"Speedup factor: {6.0 / k:.2f}x faster than theoretical $O(N^{6})$")
    else:
        print(f"Note: Empirical complexity matches or exceeds theoretical $O(N^{6})$")

    print(f"{'=' * 60}\n")

    return k, fitted_times


def plot_results(runtime_results: dict, memory_results: dict, fitted_times: np.ndarray, complexity_k: float):
    """
    Create and save performance visualization plots for runtime and memory.

    Parameters
    ----------
    runtime_results : dict
        Results from `benchmark_runtime`.
    memory_results : dict
        Results from `benchmark_memory`.
    fitted_times : numpy.ndarray
        The array of fitted times from `analyze_complexity`.
    complexity_k : float
        The estimated runtime complexity exponent $k$.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Runtime plot
    ax1 = axes[0]
    lengths = runtime_results['lengths']
    means = runtime_results['mean_times']
    stds = runtime_results['std_times']

    ax1.errorbar(lengths, means, yerr=stds, fmt='o-', capsize=5,
                 label='Measured', linewidth=2, markersize=8)
    ax1.plot(lengths, fitted_times, '--',
             label=f'Fitted $O(N^{{{complexity_k:.2f}}})$', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Sequence Length ($N$)', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Runtime Performance (Log-Log)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    # Memory plot
    ax2 = axes[1]
    memory_mb = memory_results['peak_memory_mb']

    ax2.plot(lengths, memory_mb, 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Sequence Length ($N$)', fontsize=12)
    ax2.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax2.set_title('Memory Usage (Log-Log)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    plt.tight_layout()

    # Save figure
    output_dir = Path('performance_results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_dir / 'performance_analysis.png'}")

    plt.show()


def generate_markdown_table(runtime_results: dict, memory_results: dict):
    """
    Generate and print the performance results in a raw Markdown table format.

    Parameters
    ----------
    runtime_results : dict
        Results from `benchmark_runtime`.
    memory_results : dict
        Results from `benchmark_memory`.
    """
    print("\n" + "=" * 60)
    print("MARKDOWN TABLE FOR README")
    print("=" * 60 + "\n")

    print("| Sequence Length ($N$) | Runtime (s) | Peak Memory (MB) | Energy ($\Delta G$, kcal/mol) |")
    print("|-----------------------|-------------|------------------|-------------------------------|")

    for i, n in enumerate(runtime_results['lengths']):
        time_mean = runtime_results['mean_times'][i]
        time_std = runtime_results['std_times'][i]
        memory = memory_results['peak_memory_mb'][i]
        energy = runtime_results['energies'][i]

        print(
            f"| {n:20d}| {time_mean:6.2f} $\pm$ {time_std:.2f} | {memory:16.2f} | {energy:27.2f}|")

    print("\n")


def main():
    """
    Main performance evaluation workflow.

    Executes runtime and memory benchmarks, analyzes the empirical complexity,
    generates plots, and prints the results table.
    """
    print("=" * 60)
    print("RNA PSEUDOKNOT FOLDING - PERFORMANCE EVALUATION")
    print("=" * 60)

    # Configuration
    sequence_lengths = [20, 30, 40, 50, 60, 70] # Adjust based on time constraints
    num_trials = 3  # Number of runs per length for averaging

    print(f"\nSequence lengths to test: {sequence_lengths}")
    print(f"Trials per length: {num_trials}")

    # Benchmark runtime
    print("\n" + "=" * 60)
    print("PHASE 1: RUNTIME BENCHMARKING")
    print("=" * 60)
    runtime_results = benchmark_runtime(sequence_lengths, num_trials)

    # Benchmark memory
    print("\n" + "=" * 60)
    print("PHASE 2: MEMORY BENCHMARKING")
    print("=" * 60)
    memory_results = benchmark_memory(sequence_lengths)

    # Analyze complexity
    complexity_k, fitted_times = analyze_complexity(
        runtime_results['lengths'],
        runtime_results['mean_times']
    )

    # Generate visualizations
    plot_results(runtime_results, memory_results, fitted_times, complexity_k)

    # Generate markdown table
    generate_markdown_table(runtime_results, memory_results)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
