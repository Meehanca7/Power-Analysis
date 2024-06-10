import csv
import multiprocessing as mp
from functools import partial
import numpy as np
from scipy.stats import poisson
from statsmodels.stats.power import TTestIndPower
from tqdm import tqdm  # For progress bar

def perform_power_analysis(params):
    skew_scale, n_structures, structure_id, alpha_level, power_level, prob_structure, total_reads = params

    # Calculate the expected number of observations for the structure
    expected_observations = total_reads * prob_structure

    # Perform power analysis
    analysis = TTestIndPower()
    effect_sizes = np.arange(0.1, 1.1, 0.1)  # Adjust the range and step size as needed
    required_sample_sizes = []

    for effect_size in effect_sizes:
        sample_size = analysis.solve_power(effect_size=effect_size, nobs1=None, alpha=alpha_level, power=power_level, ratio=1)
        required_sample_sizes.append(sample_size)

    # Find the minimum required sample size
    min_required_sample_size = int(np.ceil(min(required_sample_sizes)))

    return {
        'Skew Scale': skew_scale,
        'Structure ID': structure_id,
        'Unique Structures': n_structures,
        'Alpha': alpha_level,
        'Power': power_level,
        'Probability of Structure': prob_structure,
        'Expected Observations': expected_observations,
        'Minimum Required Sample Size': min_required_sample_size
    }

def param_combinations(skew_scales, unique_structures, alpha_levels, power_levels, total_reads):
    for skew_scale in skew_scales:
        for n_structures in unique_structures:
            # Generate skew values for each structure
            skew_values = np.random.exponential(scale=skew_scale, size=n_structures)
            skew_values /= skew_values.sum()  # Normalize skew values to sum up to 1

            for structure_id in range(n_structures):
                prob_structure = skew_values[structure_id]
                for alpha_level in alpha_levels:
                    for power_level in power_levels:
                        yield (skew_scale, n_structures, structure_id, alpha_level, power_level, prob_structure, total_reads)

def aggregate_results(results, skew_scales, unique_structures, alpha_levels, power_levels):
    aggregate_results = []
    for skew_scale in skew_scales:
        for n_structures in unique_structures:
            for alpha_level in alpha_levels:
                for power_level in power_levels:
                    # Filter the results for the current parameters
                    subset_results = [r for r in results if r['Skew Scale'] == skew_scale and r['Unique Structures'] == n_structures and r['Alpha'] == alpha_level and r['Power'] == power_level]

                    # Calculate aggregate metrics for minimum required sample size
                    avg_required_sample_size = np.mean([r['Minimum Required Sample Size'] for r in subset_results])

                    # Calculate aggregate metrics for expected observations
                    avg_expected_observations = np.mean([r['Expected Observations'] for r in subset_results])
                    p5_expected_observations = np.percentile([r['Expected Observations'] for r in subset_results], 5)
                    p25_expected_observations = np.percentile([r['Expected Observations'] for r in subset_results], 25)
                    p50_expected_observations = np.percentile([r['Expected Observations'] for r in subset_results], 50)
                    p75_expected_observations = np.percentile([r['Expected Observations'] for r in subset_results], 75)
                    p95_expected_observations = np.percentile([r['Expected Observations'] for r in subset_results], 95)

                    aggregate_results.append({
                        'Skew Scale': skew_scale,
                        'Unique Structures': n_structures,
                        'Alpha': alpha_level,
                        'Power': power_level,
                        'Average Required Sample Size': avg_required_sample_size,
                        'Average Expected Observations': avg_expected_observations,
                        'P5 Expected Observations': p5_expected_observations,
                        'P25 Expected Observations': p25_expected_observations,
                        'P50 Expected Observations': p50_expected_observations,
                        'P75 Expected Observations': p75_expected_observations,
                        'P95 Expected Observations': p95_expected_observations
                    })

    return aggregate_results

if __name__ == '__main__':
    total_reads = 1e7  # 10 million observations
    total_possible_structures = 16.8e6  # 16.8 million possible structures

    skew_scales = [0.5, 1.0, 2.0]
    unique_structures = [1000]
    alpha_levels = [0.01, 0.05]
    power_levels = [0.8, 0.9]

    # Generate parameter combinations
    param_combos = list(param_combinations(skew_scales, unique_structures, alpha_levels, power_levels, total_reads))

    # Perform power analysis for each parameter combination using multiprocessing
    pool = mp.Pool()
    perform_power_analysis_partial = partial(perform_power_analysis)
    results = list(tqdm(pool.imap(perform_power_analysis_partial, param_combos), total=len(param_combos), desc='Performing power analysis'))
    pool.close()
    pool.join()

    # Write individual results to CSV
    with open('individual_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Skew Scale', 'Structure ID', 'Unique Structures', 'Alpha', 'Power', 'Probability of Structure',
                      'Expected Observations', 'Minimum Required Sample Size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # Write aggregated results to CSV
    aggregated_results = aggregate_results(results, skew_scales, unique_structures, alpha_levels, power_levels)
    with open('aggregated_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Skew Scale', 'Unique Structures', 'Alpha', 'Power', 'Average Required Sample Size',
                      'Average Expected Observations', 'P5 Expected Observations', 'P25 Expected Observations',
                      'P50 Expected Observations', 'P75 Expected Observations', 'P95 Expected Observations']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in aggregated_results:
            writer.writerow(result)

    # Print aggregated results
    for result in aggregated_results:
        print(result)