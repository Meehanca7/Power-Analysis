import numpy as np
from scipy.stats import norm
import warnings
import multiprocessing as mp
from tqdm import tqdm
import csv

warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels.stats.power")

def perform_power_analysis(params):
    skew_scale, n_structures, structure_id, alpha_level, power_level, prob_structure, total_reads, effect_size = params

    expected_observations = int(round(total_reads * prob_structure))

    z_alpha = norm.ppf(1 - alpha_level / 2)
    z_beta = norm.ppf(power_level)

    if expected_observations > 0:
        min_required_sample_size = int(
            np.ceil((((z_alpha + z_beta) ** 2) * 2) / (effect_size ** 2 * expected_observations)))
    else:
        min_required_sample_size = float('inf')

    z_score = effect_size * np.sqrt(expected_observations)

    return {
        'Total Reads': total_reads,
        'Skew Scale': skew_scale,
        'Unique Structures': n_structures,
        'Alpha': alpha_level,
        'Power': power_level,
        'Effect Size': effect_size,
        'Minimum Required Sample Size': min_required_sample_size,
        'Expected Observations': expected_observations,
        'Z-Score': z_score
    }

def param_combinations(skew_scales, n_structures, alpha_levels, power_levels, total_reads, effect_sizes):
    combinations = []
    skew_values = np.random.exponential(scale=1, size=n_structures)
    skew_values /= skew_values.sum()
    sorted_skew_values = np.sort(skew_values)[::-1]

    for structure_id in range(n_structures):
        prob_structure = sorted_skew_values[structure_id]
        for skew_scale in skew_scales:
            for alpha_level in alpha_levels:
                for power_level in power_levels:
                    for effect_size in effect_sizes:
                        for total_read in total_reads:
                            combinations.append((skew_scale, n_structures, structure_id, alpha_level, power_level,
                                                 prob_structure, total_read, effect_size))
    return combinations

def aggregate_results(results):
    aggregated_results = []

    grouped_results = {}
    for result in results:
        key = (result['Total Reads'], result['Skew Scale'], result['Unique Structures'], result['Alpha'], result['Power'], result['Effect Size'])
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append((result['Minimum Required Sample Size'], result['Expected Observations'], result['Z-Score']))

    for key, values in grouped_results.items():
        total_reads, skew_scale, n_structures, alpha_level, power_level, effect_size = key
        min_required_sample_sizes, expected_observations, z_scores = zip(*values)

        sorted_data = sorted(zip(expected_observations, min_required_sample_sizes, z_scores), reverse=True)
        sorted_expected_observations, sorted_min_required_sample_sizes, sorted_z_scores = zip(*sorted_data)

        avg_min_required_sample_size = np.mean([size for size in sorted_min_required_sample_sizes if size != float('inf')])
        avg_expected_observations = np.mean(sorted_expected_observations)
        avg_z_score = np.mean(sorted_z_scores)

        percentiles = [1, 2.5, 5, 25, 50, 75, 95, 97.5, 99, 100]
        min_required_sample_size_percentiles = np.percentile(
            [size for size in sorted_min_required_sample_sizes if size != float('inf')],
            percentiles
        )

        expected_observations_percentiles = np.percentile(sorted_expected_observations, percentiles)
        z_score_percentiles = np.percentile(sorted_z_scores, percentiles)

        aggregated_result = {
            'Total Reads': total_reads,
            'Skew Scale': skew_scale,
            'Unique Structures': n_structures,
            'Alpha': alpha_level,
            'Power': power_level,
            'Effect Size': effect_size,
            'Average Minimum Required Sample Size': avg_min_required_sample_size,
            'Min Required Sample Size 1st Percentile': min_required_sample_size_percentiles[0],
            'Min Required Sample Size 2.5th Percentile': min_required_sample_size_percentiles[1],
            'Min Required Sample Size 5th Percentile': min_required_sample_size_percentiles[2],
            'Min Required Sample Size 25th Percentile': min_required_sample_size_percentiles[3],
            'Min Required Sample Size 50th Percentile': min_required_sample_size_percentiles[4],
            'Min Required Sample Size 75th Percentile': min_required_sample_size_percentiles[5],
            'Min Required Sample Size 95th Percentile': min_required_sample_size_percentiles[6],
            'Min Required Sample Size 97.5th Percentile': min_required_sample_size_percentiles[7],
            'Min Required Sample Size 99th Percentile': min_required_sample_size_percentiles[8],
            'Min Required Sample Size 100th Percentile': min_required_sample_size_percentiles[9],
            'Average Expected Observations': avg_expected_observations,
            'Expected Observations 1st Percentile': expected_observations_percentiles[0],
            'Expected Observations 2.5th Percentile': expected_observations_percentiles[1],
            'Expected Observations 5th Percentile': expected_observations_percentiles[2],
            'Expected Observations 25th Percentile': expected_observations_percentiles[3],
            'Expected Observations 50th Percentile': expected_observations_percentiles[4],
            'Expected Observations 75th Percentile': expected_observations_percentiles[5],
            'Expected Observations 95th Percentile': expected_observations_percentiles[6],
            'Expected Observations 97.5th Percentile': expected_observations_percentiles[7],
            'Expected Observations 99th Percentile': expected_observations_percentiles[8],
            'Expected Observations 100th Percentile': expected_observations_percentiles[9],
            'Average Z-Score': avg_z_score,
            'Z-Score 1st Percentile': z_score_percentiles[0],
            'Z-Score 2.5th Percentile': z_score_percentiles[1],
            'Z-Score 5th Percentile': z_score_percentiles[2],
            'Z-Score 25th Percentile': z_score_percentiles[3],
            'Z-Score 50th Percentile': z_score_percentiles[4],
            'Z-Score 75th Percentile': z_score_percentiles[5],
            'Z-Score 95th Percentile': z_score_percentiles[6],
            'Z-Score 97.5th Percentile': z_score_percentiles[7],
            'Z-Score 99th Percentile': z_score_percentiles[8],
            'Z-Score 100th Percentile': z_score_percentiles[9]
        }
        aggregated_results.append(aggregated_result)

    return aggregated_results

def worker(n_structures, skew_scales, alpha_levels, power_levels, total_reads, effect_sizes):
    param_combos = param_combinations(skew_scales, n_structures, alpha_levels, power_levels, total_reads, effect_sizes)

    results = []
    for params in tqdm(param_combos, desc=f"Processing {n_structures} structures"):
        result = perform_power_analysis(params)
        results.append(result)

    aggregated_results = aggregate_results(results)

    return aggregated_results

if __name__ == '__main__':
    total_possible_structures = 16.8e6  # 16.8 million possible structures

    total_reads = [1e6, 5e6, 1e7, 5e7, 1e8]
    skew_scales = [1, 2, 5, 10, 15, 20]
    unique_structures = [1000,10000,25000,50000,100000,125000,150000,175000,200000,250000,300000]
    alpha_levels = [0.05]
    power_levels = [0.8]
    effect_sizes = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10]

    print("Starting power analysis...")

    num_processes = min(mp.cpu_count(), len(unique_structures))
    pool = mp.Pool(processes=num_processes)

    results = pool.starmap(worker,
                           [(n_structures, skew_scales, alpha_levels, power_levels, total_reads, effect_sizes) for
                            n_structures in unique_structures])

    aggregated_results = [result for sub_results in results for result in sub_results]

    print("Power analysis completed.")

    print("Writing aggregated results to CSV...")
    output_filename = "aggregated_results.csv"
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['Total Reads', 'Skew Scale', 'Unique Structures', 'Alpha', 'Power', 'Effect Size',
                      'Average Minimum Required Sample Size',
                      'Min Required Sample Size 1st Percentile',
                      'Min Required Sample Size 2.5th Percentile',
                      'Min Required Sample Size 5th Percentile',
                      'Min Required Sample Size 25th Percentile',
                      'Min Required Sample Size 50th Percentile',
                      'Min Required Sample Size 75th Percentile',
                      'Min Required Sample Size 95th Percentile',
                      'Min Required Sample Size 97.5th Percentile',
                      'Min Required Sample Size 99th Percentile',
                      'Min Required Sample Size 100th Percentile',
                      'Average Expected Observations',
                      'Expected Observations 1st Percentile',
                      'Expected Observations 2.5th Percentile',
                      'Expected Observations 5th Percentile',
                      'Expected Observations 25th Percentile',
                      'Expected Observations 50th Percentile',
                      'Expected Observations 75th Percentile',
                      'Expected Observations 95th Percentile',
                      'Expected Observations 97.5th Percentile',
                      'Expected Observations 99th Percentile',
                      'Expected Observations 100th Percentile',
                      'Average Z-Score',
                      'Z-Score 1st Percentile',
                      'Z-Score 2.5th Percentile',
                      'Z-Score 5th Percentile',
                      'Z-Score 25th Percentile',
                      'Z-Score 50th Percentile',
                      'Z-Score 75th Percentile',
                      'Z-Score 95th Percentile',
                      'Z-Score 97.5th Percentile',
                      'Z-Score 99th Percentile',
                      'Z-Score 100th Percentile']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in aggregated_results:
            writer.writerow(result)

    print("Aggregated results written to", output_filename)

    pool.close()
    pool.join()

    print("Aggregated results:")
    for result in aggregated_results:
        print(result)