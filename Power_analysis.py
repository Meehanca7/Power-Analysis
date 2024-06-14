import numpy as np
from scipy.stats import norm
import warnings
import psutil
import os
import shutil
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

# Filter out convergence warnings from statsmodels.stats.power
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels.stats.power")

def perform_power_analysis(params):
    skew_scale, n_structures, structure_id, alpha_level, power_level, prob_structure, total_reads, effect_size = params

    # Calculate the expected number of observations for the structure
    expected_observations = int(round(total_reads * prob_structure))

    # Calculate the minimum required sample size
    z_alpha = norm.ppf(1 - alpha_level / 2)
    z_beta = norm.ppf(power_level)

    if expected_observations > 0:
        min_required_sample_size = int(np.ceil((((z_alpha + z_beta) ** 2) * 2) / (effect_size ** 2 * expected_observations)))
    else:
        min_required_sample_size = float('inf')  # Set to infinity if expected observations are zero or very small

    # Calculate the z-score using the effect size and expected observations
    z_score = effect_size * np.sqrt(expected_observations)

    return {
        'Skew Scale': skew_scale,
        'Structure ID': structure_id,
        'Unique Structures': n_structures,
        'Alpha': alpha_level,
        'Power': power_level,
        'Probability of Structure': prob_structure,
        'Expected Observations': expected_observations,
        'Minimum Required Sample Size': min_required_sample_size,
        'Effect Size': effect_size,
        'Z-Score': z_score
    }

def param_combinations(skew_scales, n_structures, alpha_levels, power_levels, total_reads, effect_sizes):
    combinations = []
    # Generate skew values for each structure
    skew_values = np.random.exponential(scale=1, size=n_structures)
    skew_values /= skew_values.sum()  # Normalize skew values to sum up to 1
    sorted_skew_values = np.sort(skew_values)[::-1]  # Sort skew values in descending order

    for structure_id in range(n_structures):
        prob_structure = sorted_skew_values[structure_id]
        for skew_scale in skew_scales:
            for alpha_level in alpha_levels:
                for power_level in power_levels:
                    for effect_size in effect_sizes:
                        combinations.append((skew_scale, n_structures, structure_id, alpha_level, power_level, prob_structure, total_reads, effect_size))
    return combinations

def aggregate_results(results):
    aggregate_results = []
    sorted_results = sorted(results, key=lambda x: x['Structure ID'])

    # Calculate percentiles for minimum required sample size, expected observations, and z-score
    percentiles = [5, 25, 50, 75, 95, 100]
    min_required_sample_size_percentiles = np.percentile([r['Minimum Required Sample Size'] for r in sorted_results], percentiles)
    expected_observations_percentiles = np.percentile([r['Expected Observations'] for r in sorted_results], percentiles)
    z_score_percentiles = np.percentile([r['Z-Score'] for r in sorted_results], percentiles)

    aggregate_result = {
        'Skew Scale': sorted_results[0]['Skew Scale'],
        'Unique Structures': sorted_results[0]['Unique Structures'],
        'Alpha': sorted_results[0]['Alpha'],
        'Power': sorted_results[0]['Power'],
        'Effect Size': sorted_results[0]['Effect Size'],
        'Min Required Sample Size 5th Percentile': min_required_sample_size_percentiles[0],
        'Min Required Sample Size 25th Percentile': min_required_sample_size_percentiles[1],
        'Min Required Sample Size 50th Percentile': min_required_sample_size_percentiles[2],
        'Min Required Sample Size 75th Percentile': min_required_sample_size_percentiles[3],
        'Min Required Sample Size 95th Percentile': min_required_sample_size_percentiles[4],
        'Min Required Sample Size 100th Percentile': min_required_sample_size_percentiles[5],
        'Expected Observations 5th Percentile': expected_observations_percentiles[0],
        'Expected Observations 25th Percentile': expected_observations_percentiles[1],
        'Expected Observations 50th Percentile': expected_observations_percentiles[2],
        'Expected Observations 75th Percentile': expected_observations_percentiles[3],
        'Expected Observations 95th Percentile': expected_observations_percentiles[4],
        'Z-Score 5th Percentile': z_score_percentiles[0],
        'Z-Score 25th Percentile': z_score_percentiles[1],
        'Z-Score 50th Percentile': z_score_percentiles[2],
        'Z-Score 75th Percentile': z_score_percentiles[3],
        'Z-Score 95th Percentile': z_score_percentiles[4]
    }

    aggregate_results.append(aggregate_result)

    return aggregate_results

def worker(n_structures, skew_scales, alpha_levels, power_levels, total_reads, effect_sizes):
    # Generate parameter combinations for the current number of structures
    param_combos = param_combinations(skew_scales, n_structures, alpha_levels, power_levels, total_reads, effect_sizes)

    # Process parameter combinations and aggregate results
    results = []
    for params in tqdm(param_combos, desc=f"Processing {n_structures} structures"):
        result = perform_power_analysis(params)
        results.append(result)

    # Aggregate results for the current number of structures
    aggregated_results = aggregate_results(results)

    return aggregated_results

if __name__ == '__main__':
    total_reads = 1e7  # 10 million observations
    total_possible_structures = 16.8e6  # 16.8 million possible structures

    skew_scales = [1, 2, 5, 10, 50, 100, 500, 1000]
    unique_structures = [1000, 5000, 10000, 25000, 50000, 100000, 125000, 150000, 200000, 250000, 300000]
    alpha_levels = [0.01, 0.05]
    power_levels = [0.8, 0.9]
    effect_sizes = [0.5, 1, 2, 5, 10, 100, 1000]  # Specify the desired effect size

    print("Starting power analysis...")

    # Create a pool of processes
    num_processes = min(mp.cpu_count(), len(unique_structures))
    pool = mp.Pool(processes=num_processes)

    # Process unique structures in parallel
    results = pool.starmap(worker, [(n_structures, skew_scales, alpha_levels, power_levels, total_reads, effect_sizes) for n_structures in unique_structures])

    # Flatten the results
    aggregated_results = [result for sub_results in results for result in sub_results]

    print("Power analysis completed.")

    print("Writing aggregated results to CSV...")
    # Write aggregated results to CSV
    import csv
    output_filename = "aggregated_results.csv"
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['Skew Scale', 'Unique Structures', 'Alpha', 'Power', 'Effect Size',
                      'Min Required Sample Size 5th Percentile', 'Min Required Sample Size 25th Percentile',
                      'Min Required Sample Size 50th Percentile', 'Min Required Sample Size 75th Percentile',
                      'Min Required Sample Size 95th Percentile', 'Min Required Sample Size 100th Percentile',
                      'Expected Observations 5th Percentile', 'Expected Observations 25th Percentile',
                      'Expected Observations 50th Percentile', 'Expected Observations 75th Percentile',
                      'Expected Observations 95th Percentile',
                      'Z-Score 5th Percentile', 'Z-Score 25th Percentile',
                      'Z-Score 50th Percentile', 'Z-Score 75th Percentile', 'Z-Score 95th Percentile']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in aggregated_results:
            writer.writerow(result)

    print("Aggregated results written to", output_filename)

    pool.close()
    pool.join()

    # Print aggregated results
    print("Aggregated results:")
    for result in aggregated_results:
        print(result)