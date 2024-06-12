import csv
import multiprocessing as mp
from functools import partial
import numpy as np
from scipy.stats import norm
import warnings
from tqdm import tqdm
import psutil

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
    combinations = []
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
                            combinations.append((skew_scale, n_structures, structure_id, alpha_level, power_level, prob_structure, total_reads, effect_size))
    return combinations

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

def get_available_memory():
    mem = psutil.virtual_memory()
    available_memory_gb = mem.available / (1024 ** 3)
    return available_memory_gb

def adjust_batch_size(batch_size, available_memory_gb):
    memory_per_process = 0.5  # Adjust this value based on your observations
    max_processes = int(available_memory_gb // memory_per_process)
    adjusted_batch_size = max(1, min(batch_size, max_processes))
    return adjusted_batch_size

if __name__ == '__main__':
    total_reads = 1e7  # 10 million observations
    total_possible_structures = 16.8e6  # 16.8 million possible structures

    skew_scales = [1, 2, 5, 10, 25, 50, 100, 250, 1000]
    unique_structures = [1000, 5000, 10000, 25000, 50000, 100000, 125000, 150000, 200000, 250000, 300000]
    alpha_levels = [0.01, 0.05, 0.1]
    power_levels = [0.8, 0.85, 0.9, 0.95]
    effect_sizes = [0.5, 1, 2, 5, 10, 25, 50]  # Specify the desired effect size

    # Generate parameter combinations
    param_combos = param_combinations(skew_scales, unique_structures, alpha_levels, power_levels, total_reads, effect_sizes)

    # Calculate the total number of iterations
    total_iterations = sum(len(unique_structures) * len(skew_scales) * len(alpha_levels) * len(power_levels) * len(effect_sizes) for n_structures in unique_structures)

    # Create a pool of processes
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)

    # Process results in batches
    batch_size = 10000
    results = []

    available_memory_gb = get_available_memory()
    adjusted_batch_size = adjust_batch_size(batch_size, available_memory_gb)
    print(f"Adjusted batch size: {adjusted_batch_size}")

    completed_iterations = 0

    with tqdm(total=total_iterations, desc='Progress', unit='iteration') as pbar:
        for batch in pool.imap(perform_power_analysis, param_combos, chunksize=adjusted_batch_size):
            if isinstance(batch, dict):
                results.append(batch)
            else:
                results.extend(batch)

            completed_iterations += len(batch)
            pbar.update(len(batch))
            pbar.set_postfix({'Completed': f'{completed_iterations}/{total_iterations}', 'Percentage': f'{completed_iterations/total_iterations:.2%}'})

            if len(results) >= adjusted_batch_size:
                # Write individual results to CSV
                with open('individual_results.csv', 'a', newline='') as csvfile:
                    fieldnames = ['Skew Scale', 'Structure ID', 'Unique Structures', 'Alpha', 'Power', 'Probability of Structure',
                                  'Expected Observations', 'Minimum Required Sample Size', 'Effect Size', 'Z-Score']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    if csvfile.tell() == 0:  # Write header only if the file is empty
                        writer.writeheader()
                    writer.writerows(results)

                results = []  # Clear the results list for the next batch

                # Check available memory and adjust batch size if necessary
                available_memory_gb = get_available_memory()
                adjusted_batch_size = adjust_batch_size(batch_size, available_memory_gb)

    # Write any remaining results to CSV
    if results:
        with open('individual_results.csv', 'a', newline='') as csvfile:
            fieldnames = ['Skew Scale', 'Structure ID', 'Unique Structures', 'Alpha', 'Power', 'Probability of Structure',
                          'Expected Observations', 'Minimum Required Sample Size', 'Effect Size', 'Z-Score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:  # Write header only if the file is empty
                writer.writeheader()
            writer.writerows(results)

    pool.close()
    pool.join()

    # Aggregate results
    aggregated_results = aggregate_results(results, skew_scales, unique_structures, alpha_levels, power_levels, effect_sizes)

    # Write aggregated results to CSV
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