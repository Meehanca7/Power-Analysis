import csv
import multiprocessing as mp
from functools import partial
import numpy as np
from scipy.stats import norm
import warnings
from statsmodels.stats.power import TTestIndPower
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

def param_combinations(skew_scales, unique_structures, alpha_levels, power_levels, total_reads, effect_sizes):
    for skew_scale in skew_scales:
        for n_structures in unique_structures:
            # Generate skew values for each structure
            skew_values = np.random.exponential(scale=skew_scale, size=n_structures)
            skew_values /= skew_values.sum()  # Normalize skew values to sum up to 1
            sorted_skew_values = np.sort(skew_values)[::-1]  # Sort skew values in descending order

            for structure_id in range(n_structures):
                prob_structure = sorted_skew_values[structure_id]
                for alpha_level in alpha_levels:
                    for power_level in power_levels:
                        for effect_size in effect_sizes:
                            yield (skew_scale, n_structures, structure_id, alpha_level, power_level, prob_structure, total_reads, effect_size)

def aggregate_results(results, skew_scales, unique_structures, alpha_levels, power_levels, effect_sizes):
    aggregate_results = []
    for skew_scale in skew_scales:
        for n_structures in unique_structures:
            for alpha_level in alpha_levels:
                for power_level in power_levels:
                    for effect_size in effect_sizes:
                        # Filter the results for the current parameters
                        subset_results = [r for r in results if r['Skew Scale'] == skew_scale and r['Unique Structures'] == n_structures and r['Alpha'] == alpha_level and r['Power'] == power_level and r['Effect Size'] == effect_size]

                        # Sort the subset results by structure ID
                        sorted_subset_results = sorted(subset_results, key=lambda x: x['Structure ID'])

                        # Calculate percentiles for minimum required sample size, expected observations, and z-score
                        percentiles = [5, 25, 50, 75, 95]
                        min_required_sample_size_percentiles = np.percentile([r['Minimum Required Sample Size'] for r in sorted_subset_results], percentiles)
                        expected_observations_percentiles = np.percentile([r['Expected Observations'] for r in sorted_subset_results], percentiles)
                        z_score_percentiles = np.percentile([r['Z-Score'] for r in sorted_subset_results], percentiles)

                        aggregate_results.append({
                            'Skew Scale': skew_scale,
                            'Unique Structures': n_structures,
                            'Alpha': alpha_level,
                            'Power': power_level,
                            'Effect Size': effect_size,
                            'Min Required Sample Size 5th Percentile': min_required_sample_size_percentiles[0],
                            'Min Required Sample Size 25th Percentile': min_required_sample_size_percentiles[1],
                            'Min Required Sample Size 50th Percentile': min_required_sample_size_percentiles[2],
                            'Min Required Sample Size 75th Percentile': min_required_sample_size_percentiles[3],
                            'Min Required Sample Size 95th Percentile': min_required_sample_size_percentiles[4],
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
                        })

    return aggregate_results

if __name__ == '__main__':
    total_reads = 1e7  # 10 million observations
    total_possible_structures = 16.8e6  # 16.8 million possible structures

    skew_scales = [1,2,5,10,25,50]
    unique_structures = [1000, 5000, 10000, 25000, 50000, 100000, 125000, 150000, 200000]
    alpha_levels = [0.01, 0.05, 0.1]
    power_levels = [0.8, 0.85, 0.9, 0.95]
    effect_sizes = [0.5, 1, 2, 5, 10]  # Specify the desired effect size

    # Generate parameter combinations
    param_combos = list(param_combinations(skew_scales, unique_structures, alpha_levels, power_levels, total_reads, effect_sizes))

    # Create a pool of 32 processes
    pool = mp.Pool(processes=32)

    # Perform power analysis for each parameter combination using multiprocessing
    perform_power_analysis_partial = partial(perform_power_analysis)
    results = list(tqdm(pool.imap(perform_power_analysis_partial, param_combos), total=len(param_combos), desc='Performing power analysis', disable=False))
    pool.close()
    pool.join()

    # Write individual results to CSV
    with open('individual_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Skew Scale', 'Structure ID', 'Unique Structures', 'Alpha', 'Power', 'Probability of Structure',
                      'Expected Observations', 'Minimum Required Sample Size', 'Effect Size', 'Z-Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # Write aggregated results to CSV
    aggregated_results = aggregate_results(results, skew_scales, unique_structures, alpha_levels, power_levels, effect_sizes)
    with open('aggregated_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Skew Scale', 'Unique Structures', 'Alpha', 'Power', 'Effect Size',
                      'Min Required Sample Size 5th Percentile', 'Min Required Sample Size 25th Percentile',
                      'Min Required Sample Size 50th Percentile', 'Min Required Sample Size 75th Percentile',
                      'Min Required Sample Size 95th Percentile',
                      'Expected Observations 5th Percentile', 'Expected Observations 25th Percentile',
                      'Expected Observations 50th Percentile', 'Expected Observations 75th Percentile',
                      'Expected Observations 95th Percentile',
                      'Z-Score 5th Percentile', 'Z-Score 25th Percentile',
                      'Z-Score 50th Percentile', 'Z-Score 75th Percentile', 'Z-Score 95th Percentile']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in aggregated_results:
            writer.writerow(result)

    # Print aggregated results
    for result in aggregated_results:
        print(result)