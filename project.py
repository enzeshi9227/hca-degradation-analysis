import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def load_data(filepath):
    """Load and extract clean normalized HCA concentration data
    for all three temperature conditions from a raw CSV file."""
    df = pd.read_csv(filepath)

    def extract_condition(df, time_col, conc_col, temp_label):
        subset = df[[time_col, conc_col]].copy()
        subset.columns = ['time_days', 'concentration']
        subset['time_days'] = pd.to_numeric(subset['time_days'], errors='coerce')
        subset['concentration'] = pd.to_numeric(subset['concentration'], errors='coerce')
        subset = subset.dropna()
        subset['replicate'] = (subset['time_days'] == 0).cumsum()
        subset['temperature'] = temp_label
        return subset

    df_50c = extract_condition(df, 'Unnamed: 23', 'Unnamed: 24', '50C')
    df_30c = extract_condition(df, 'Unnamed: 26', 'Unnamed: 27', '30C')
    df_2c  = extract_condition(df, 'Unnamed: 29', 'Unnamed: 30', '2C')

    combined = pd.concat([df_50c, df_30c, df_2c], ignore_index=True)
    return combined

def compute_stats(df):
    """Compute mean and std of concentration at each timepoint
    per temperature condition across replicates."""
    stats_df = (
        df.groupby(['temperature', 'time_days'])['concentration']
        .agg(mean='mean', std='std', count='count')
        .reset_index()
    )
    return stats_df

def compare_conditions(df, temp1, temp2, timepoint):
    """Run a t-test comparing concentration between two temperature
    conditions at a specific timepoint across replicates."""
    group1 = df[(df['temperature'] == temp1) & (df['time_days'] == timepoint)]['concentration']
    group2 = df[(df['temperature'] == temp2) & (df['time_days'] == timepoint)]['concentration']

    t_stat, p_value = stats.ttest_ind(group1, group2)

    return {
        'temp1': temp1,
        'temp2': temp2,
        'timepoint': timepoint,
        't_statistic': round(t_stat, 4),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05
    }


def plot_degradation(df, output_path='degradation_plot.png'):
    """Plot normalized HCA concentration over time for each
    temperature condition and save as a PNG file."""
    stats_df = compute_stats(df)

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {'50C': 'red', '30C': 'orange', '2C': 'blue'}

    for temp in ['50C', '30C', '2C']:
        subset = stats_df[stats_df['temperature'] == temp]
        ax.errorbar(
            subset['time_days'],
            subset['mean'],
            yerr=subset['std'],
            label=f"{temp}",
            color=colors[temp],
            marker='o',
            capsize=4,
            linewidth=2
        )

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Normalized HCA Concentration')
    ax.set_title('Hexachloroethane Degradation at Different Temperatures')
    ax.legend(title='Temperature')
    ax.set_ylim(0, 1.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def main():
    df = load_data('EST_Tae_CCR_Rawdata_Figure1.csv')
    stats_df = compute_stats(df)
    print(stats_df)
    result = compare_conditions(df, '50C', '30C', 30.0)
    print(result)
    path = plot_degradation(df)
    print(f"Plot saved to {path}")

if __name__ == "__main__":
    main()