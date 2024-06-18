import numpy as np
from scipy.stats import norm
import warnings
import psutil
import os
import shutil
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels.stats.power")

def perform_power_analysis(params):
    skew_scale, n_structures, structure_id, alpha_level, power_level, prob_structure, total_reads, cohens_d = params

    # Calculate the expected number of observations for the structure
    expected_observations = int(round(total_reads * prob_structure))

    # Calculate the minimum required sample size
    z_alpha = norm.ppf(1 - alpha_level / 2)
    z_beta = norm.ppf(power_level)

    if expected_observations > 0:
        min_required_sample_size = int(
            np.ceil((((z_alpha + z_beta) ** 2) * 2) / (cohens_d ** 2)))
    else:
        min_required_sample_size = float('inf')  # Set to infinity if expected observations are zero or very small

    # Calculate the z-score using Cohen's d and expected observations
    z_score = cohens_d * np.sqrt(expected_observations / 2)

    return {
        'Skew Scale': skew_scale,
        'Unique Structures': n_structures,
        'Alpha': alpha_level,
        'Power': power_level,
        'Cohen\'s d': cohens_d,
        'Minimum Required Sample Size': min_required_sample_size,
        'Expected Observations': expected_observations,
        'Z-Score': z_score
    }


def param_combinations(skew_scales, n_structures, alpha_levels, power_levels, total_reads, cohens_ds):
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
                    for cohens_d in cohens_ds:
                        combinations.append((skew_scale, n_structures, structure_id, alpha_level, power_level,
                                             prob_structure, total_reads, cohens_d))
    return combinations


def aggregate_results(results):
    aggregated_results = []

    # Group results by unique parameter combinations
    grouped_results = {}
    for result in results:
        key = (
        result['Skew Scale'], result['Unique Structures'], result['Alpha'], result['Power'], result['Cohen\'s d'])
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(
            (result['Minimum Required Sample Size'], result['Expected Observations'], result['Z-Score']))

    # Calculate average and percentiles for each unique parameter combination
    for key, values in grouped_results.items():
        skew_scale, n_structures, alpha_level, power_level, cohens_d = key
        min_required_sample_sizes, expected_observations, z_scores = zip(*values)

        avg_min_required_sample_size = np.mean([size for size in min_required_sample_sizes if size != float('inf')])
        avg_expected_observations = np.mean(expected_observations)
        avg_z_score = np.mean(z_scores)

        percentiles = [5, 25, 50, 75, 95, 100]
        min_required_sample_size_percentiles = np.percentile(
            [size for size in min_required_sample_sizes if size != float('inf')], percentiles)
        expected_observations_percentiles = np.percentile(expected_observations, percentiles)
        z_score_percentiles = np.percentile(z_scores, percentiles)

        aggregated_result = {
            'Skew Scale': skew_scale,
            'Unique Structures': n_structures,
            'Alpha': alpha_level,
            'Power': power_level,
            'Cohen\'s d': cohens_d,
            'Average Minimum Required Sample Size': avg_min_required_sample_size,
            'Min Required Sample Size 5th Percentile': min_required_sample_size_percentiles[0],
            'Min Required Sample Size 25th Percentile': min_required_sample_size_percentiles[1],
            'Min Required Sample Size 50th Percentile': min_required_sample_size_percentiles[2],
            'Min Required Sample Size 75th Percentile': min_required_sample_size_percentiles[3],
            'Min Required Sample Size 95th Percentile': min_required_sample_size_percentiles[4],
            'Min Required Sample Size 100th Percentile': min_required_sample_size_percentiles[5],
            'Average Expected Observations': avg_expected_observations,
            'Expected Observations 5th Percentile': expected_observations_percentiles[0],
            'Expected Observations 25th Percentile': expected_observations_percentiles[1],
            'Expected Observations 50th Percentile': expected_observations_percentiles[2],
            'Expected Observations 75th Percentile': expected_observations_percentiles[3],
            'Expected Observations 95th Percentile': expected_observations_percentiles[4],
            'Average Z-Score': avg_z_score,
            'Z-Score 5th Percentile': z_score_percentiles[0],
            'Z-Score 25th Percentile': z_score_percentiles[1],
            'Z-Score 50th Percentile': z_score_percentiles[2],
            'Z-Score 75th Percentile': z_score_percentiles[3],
            'Z-Score 95th Percentile': z_score_percentiles[4]
        }
        aggregated_results.append(aggregated_result)

    return aggregated_results


def worker(n_structures, skew_scales, alpha_levels, power_levels, total_reads, cohens_ds):
    # Generate parameter combinations for the current number of structures
    param_combos = param_combinations(skew_scales, n_structures, alpha_levels, power_levels, total_reads, cohens_ds)

    # Process parameter combinations and aggregate results
    results = []
    for params in tqdm(param_combos, desc=f"Processing {n_structures} structures"):
        result = perform_power_analysis(params)
        results.append(result)

    # Aggregate results for the current number of structures
    aggregated_results = aggregate_results(results)

    return aggregated_results


if __name__ == '__main__':
    total_reads_list = [1e6, 5e6, 1e7, 5e7, 1e8]  # Multiple read sizes
    total_possible_structures = 16.8e6  # 16.8 million possible structures

    skew_scales = [1, 2, 5, 10, 50, 100, 500, 1000]
    unique_structures = [1000, 5000, 10000, 25000, 50000, 100000, 125000, 150000, 200000, 300000]
    alpha_levels = [0.01, 0.05]
    power_levels = [0.8, 0.9]
    cohens_ds = [0.25, 0.5, 1, 2, 3, 5, 10]  # Small, medium, and large effect sizes using Cohen's d

    print("Starting power analysis...")

    for total_reads in total_reads_list:
        print(f"Processing {total_reads} total reads...")

        # Create a pool of processes
        num_processes = min(mp.cpu_count(), len(unique_structures))
        pool = mp.Pool(processes=num_processes)

        # Process unique structures in parallel
        results = pool.starmap(worker,
                               [(n_structures, skew_scales, alpha_levels, power_levels, total_reads, cohens_ds) for
                                n_structures in unique_structures])

        # Flatten the results
        aggregated_results = [result for sub_results in results for result in sub_results]

        print(f"Power analysis completed for {total_reads} total reads.")

        print("Writing aggregated results to CSV...")
        # Write aggregated results to CSV
        import csv

        output_filename = f"aggregated_results_{int(total_reads)}.csv"
        with open(output_filename, 'w', newline='') as csvfile:
            fieldnames = ['Skew Scale', 'Unique Structures', 'Alpha', 'Power', 'Cohen\'s d',
                          'Average Minimum Required Sample Size',
                          'Min Required Sample Size 5th Percentile', 'Min Required Sample Size 25th Percentile',
                          'Min Required Sample Size 50th Percentile', 'Min Required Sample Size 75th Percentile',
                          'Min Required Sample Size 95th Percentile', 'Min Required Sample Size 100th Percentile',
                          'Average Expected Observations',
                          'Expected Observations 5th Percentile', 'Expected Observations 25th Percentile',
                          'Expected Observations 50th Percentile', 'Expected Observations 75th Percentile',
                          'Expected Observations 95th Percentile',
                          'Average Z-Score',
                          'Z-Score 5th Percentile', 'Z-Score 25th Percentile',
                          'Z-Score 50th Percentile', 'Z-Score 75th Percentile', 'Z-Score 95th Percentile']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in aggregated_results:
                writer.writerow(result)

        print("Aggregated results written to", output_filename)

        pool.close()
        pool.join()

    print("All power analyses completed.")