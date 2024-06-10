import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestIndPower
from multiprocessing import Pool
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm

warnings.filterwarnings('ignore', category=ConvergenceWarning)


def perform_power_analysis(params):
    skew_scale, n_structures, structure_id, alpha_level, power_level, reads_per_structure_target, reads_per_structure_counter_target = params

    # Calculate the effect size
    effect_size = (reads_per_structure_target[structure_id] - reads_per_structure_counter_target[
        structure_id]) / np.sqrt(
        (reads_per_structure_target[structure_id] + reads_per_structure_counter_target[structure_id]) / 2)

    if effect_size == 0:
        required_sample_size_per_condition = 0
    else:
        # Calculate the required sample size per condition
        analysis = TTestIndPower()
        result = analysis.solve_power(effect_size=effect_size,
                                      nobs1=None,
                                      alpha=alpha_level,
                                      power=power_level,
                                      ratio=1)  # Assuming equal sample sizes
        required_sample_size_per_condition = int(result)

    return {
        'Skew Scale': skew_scale,
        'Structure ID': structure_id,
        'Unique Structures': n_structures,
        'Alpha': alpha_level,
        'Power': power_level,
        'Required Sample Size per Condition': required_sample_size_per_condition,
        'Total Required Sample Size': required_sample_size_per_condition * 2,  # For both conditions
        'Target Reads': reads_per_structure_target[structure_id],
        'Counter Target Reads': reads_per_structure_counter_target[structure_id]
    }


def param_combinations(skew_scales, unique_structures, alpha_levels, power_levels, total_reads_target,
                       total_reads_counter_target):
    for skew_scale in skew_scales:
        for n_structures in unique_structures:
            # Generate skew values for each structure
            skew_values = np.random.exponential(scale=skew_scale, size=n_structures)
            skew_values /= skew_values.sum()  # Normalize skew values to sum up to 1

            # Calculate the reads per structure based on skew for target and counter target
            reads_per_structure_target = (total_reads_target * skew_values).astype(int)
            reads_per_structure_counter_target = (total_reads_counter_target * skew_values).astype(int)

            for structure_id in range(n_structures):
                for alpha_level in alpha_levels:
                    for power_level in power_levels:
                        yield (
                        skew_scale, n_structures, structure_id, alpha_level, power_level, reads_per_structure_target,
                        reads_per_structure_counter_target)


if __name__ == '__main__':
    # Change the working directory
    os.chdir('./')

    # Define the range of parameters
    unique_structures = [1000, 5000, 10000, 25000, 50000, 100000, 125000, 150000, 200000]
    alpha_levels = [0.01, 0.05, 0.1]  # Significance levels
    power_levels = [0.75, 0.8, 0.85, 0.9, 0.95]  # Desired power levels
    skew_scales = [0.5, 1.0, 2.0, 5.0]  # Different skew scales

    # Set the total number of reads for target and counter target
    total_reads_target = 10000000
    total_reads_counter_target = 10000000

    # Create a multiprocessing pool with 32 processes
    pool = Pool(processes=32)

    # Generate parameter combinations using the generator
    param_combos = param_combinations(skew_scales, unique_structures, alpha_levels, power_levels, total_reads_target,
                                      total_reads_counter_target)

    # Process results using imap_unordered
    results = []
    with tqdm(
            total=len(skew_scales) * len(unique_structures) * len(alpha_levels) * len(power_levels) * unique_structures[
                -1]) as pbar:
        for result in pool.imap_unordered(perform_power_analysis, param_combos):
            results.append(result)
            pbar.update()

    # Close the multiprocessing pool
    pool.close()
    pool.join()

    print("Power analysis completed.")

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Aggregate the results for each library type (skew scale)
    aggregate_results = []
    for skew_scale in skew_scales:
        for n_structures in unique_structures:
            for alpha_level in alpha_levels:
                for power_level in power_levels:
                    # Filter the DataFrame for the current parameters
                    df_subset = df[(df['Skew Scale'] == skew_scale) & (df['Unique Structures'] == n_structures) & (
                                df['Alpha'] == alpha_level) & (df['Power'] == power_level)]

                    # Calculate aggregate stats for the library
                    avg_required_sample_size_per_condition = df_subset['Required Sample Size per Condition'].mean()
                    avg_total_required_sample_size = df_subset['Total Required Sample Size'].mean()
                    p5_required_sample_size_per_condition = np.percentile(
                        df_subset['Required Sample Size per Condition'], 5)
                    p10_required_sample_size_per_condition = np.percentile(
                        df_subset['Required Sample Size per Condition'], 10)
                    p15_required_sample_size_per_condition = np.percentile(
                        df_subset['Required Sample Size per Condition'], 15)
                    p20_required_sample_size_per_condition = np.percentile(
                        df_subset['Required Sample Size per Condition'], 20)
                    p25_required_sample_size_per_condition = np.percentile(
                        df_subset['Required Sample Size per Condition'], 25)
                    p75_required_sample_size_per_condition = np.percentile(
                        df_subset['Required Sample Size per Condition'], 75)
                    p80_required_sample_size_per_condition = np.percentile(
                        df_subset['Required Sample Size per Condition'], 80)
                    p85_required_sample_size_per_condition = np.percentile(
                        df_subset['Required Sample Size per Condition'], 85)
                    p90_required_sample_size_per_condition = np.percentile(
                        df_subset['Required Sample Size per Condition'], 90)
                    p95_required_sample_size_per_condition = np.percentile(
                        df_subset['Required Sample Size per Condition'], 95)

                    # Append the aggregate results to the list
                    aggregate_results.append({
                        'Skew Scale': skew_scale,
                        'Unique Structures': n_structures,
                        'Alpha': alpha_level,
                        'Power': power_level,
                        'Average Required Sample Size per Condition': avg_required_sample_size_per_condition,
                        'Average Total Required Sample Size': avg_total_required_sample_size,
                        'P5 Required Sample Size per Condition': p5_required_sample_size_per_condition,
                        'P10 Required Sample Size per Condition': p10_required_sample_size_per_condition,
                        'P15 Required Sample Size per Condition': p15_required_sample_size_per_condition,
                        'P20 Required Sample Size per Condition': p20_required_sample_size_per_condition,
                        'P25 Required Sample Size per Condition': p25_required_sample_size_per_condition,
                        'P75 Required Sample Size per Condition': p75_required_sample_size_per_condition,
                        'P80 Required Sample Size per Condition': p80_required_sample_size_per_condition,
                        'P85 Required Sample Size per Condition': p85_required_sample_size_per_condition,
                        'P90 Required Sample Size per Condition': p90_required_sample_size_per_condition,
                        'P95 Required Sample Size per Condition': p95_required_sample_size_per_condition
                    })

    print("Aggregation completed.")

    # Create a DataFrame from the aggregate results
    aggregate_df = pd.DataFrame(aggregate_results)

    # Save the aggregate results to a CSV file
    aggregate_df.to_csv('power_analysis_aggregate_results.csv', index=False)
    print("Aggregate results saved to power_analysis_aggregate_results.csv")