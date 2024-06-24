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

def read_structure_frequencies(file_path):
    structure_frequencies = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            frequencies = [float(row[sample]) for sample in reader.fieldnames[1:]]
            structure_frequencies.append(np.mean(frequencies))
    return structure_frequencies

def param_combinations(skew_scales, n_structures, structure_frequencies, alpha_levels, power_levels, total_reads, effect_sizes):
    combinations = []
    for structure_id, prob_structure in enumerate(structure_frequencies):
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

        avg_min_required_sample_size = np.mean([size for size in min_required_sample_sizes if size != float('inf')])
        avg_expected_observations = np.mean(expected_observations)
        avg_z_score = np.mean(z_scores)

        percentiles = [0, 1, 2.5, 5, 25, 50, 75, 95, 97.5, 99, 100]
        min_required_sample_size_percentiles = np.percentile(
            [size for size in min_required_sample_sizes if size != float('inf')],
            percentiles
        )
        expected_observations_percentiles = np.percentile(expected_observations, percentiles)
        z_score_percentiles = np.percentile(z_scores, percentiles)

        aggregated_result = {
            'Total Reads': total_reads,
            'Skew Scale': skew_scale,
            'Unique Structures': n_structures,
            'Alpha': alpha_level,
            'Power': power_level,
            'Effect Size': effect_size,
            'Average Minimum Required Sample Size': avg_min_required_sample_size,
            **{f'Min Required Sample Size {p}th Percentile': min_required_sample_size_percentiles[i] for i, p in enumerate(percentiles)},
            'Average Expected Observations': avg_expected_observations,
            **{f'Expected Observations {p}th Percentile': expected_observations_percentiles[i] for i, p in enumerate(percentiles)},
            'Average Z-Score': avg_z_score,
            **{f'Z-Score {p}th Percentile': z_score_percentiles[i] for i, p in enumerate(percentiles)}
        }
        aggregated_results.append(aggregated_result)

    return aggregated_results

def worker(structure_file, skew_scales, alpha_levels, power_levels, total_reads, effect_sizes):
    structure_frequencies = read_structure_frequencies(structure_file)
    n_structures = len(structure_frequencies)
    param_combos = param_combinations(skew_scales, n_structures, structure_frequencies, alpha_levels, power_levels, total_reads, effect_sizes)

    results = []
    for params in tqdm(param_combos, desc=f"Processing {n_structures} structures"):
        result = perform_power_analysis(params)
        results.append(result)

    aggregated_results = aggregate_results(results)

    return aggregated_results

if __name__ == '__main__':
    structure_file = '/Users/cathalmeehan/Documents/Power_analysis/structure_frequencies.txt'  # Path to the structure frequency file

    total_reads = [1e6, 5e6, 1e7, 5e7, 1e8]
    skew_scales = [1, 2, 5, 10, 15, 20]
    alpha_levels = [0.05]
    power_levels = [0.8]
    effect_sizes = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10]

    print("Starting power analysis...")

    results = worker(structure_file, skew_scales, alpha_levels, power_levels, total_reads, effect_sizes)
    aggregated_results = results

    print("Power analysis completed.")

    print("Writing aggregated results to CSV...")
    output_filename = "aggregated_results.csv"
    fieldnames = list(aggregated_results[0].keys())
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in aggregated_results:
            writer.writerow(result)

    print("Aggregated results written to", output_filename)