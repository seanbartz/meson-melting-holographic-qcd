#!/usr/bin/env python3
"""
Script to compile existing axial peak data across temperatures and plot peak positions vs temperature
without rerunning the full simulations.
"""
import os, re, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def main(data_dir, lambda1, mu, mq, out_dir):
    # Pattern to match peak files for given parameters
    pattern = re.compile(
        rf"axial_peaks_data_T_(?P<T>[0-9]+\.?[0-9]*)_mu_{mu:.1f}_mq_{mq:.1f}_lambda1_{lambda1:.1f}\.csv"
    )
    files = [f for f in os.listdir(data_dir) if pattern.match(f)]
    if not files:
        print(f"No peak files found in {data_dir} for lambda1={lambda1}, mu={mu}, mq={mq}")
        return

    # Collect peak data
    all_dfs = []
    for fname in sorted(files, key=lambda x: float(pattern.match(x).group('T'))):
        temp = float(pattern.match(fname).group('T'))
        df = pd.read_csv(os.path.join(data_dir, fname))
        df['temperature'] = temp
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)

    # Generate peak numbers based on the order of the data
    combined['peak_number'] = combined.groupby('temperature').cumcount() + 1

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(
        out_dir,
        f"axial_temperature_summary_l1_{lambda1:.1f}_mu_{mu:.1f}_mq_{mq:.1f}.csv"
    )
    combined.to_csv(summary_csv, index=False)
    print(f"Combined summary saved to {summary_csv}")

    # Ensure required columns exist in the DataFrame
    required_columns = ['peak_number', 'temperature', 'peak_omega_squared_dimensionless']
    for col in required_columns:
        if col not in combined.columns:
            raise KeyError(f"Missing required column '{col}' in the combined DataFrame.")

    # Plot peak positions vs temperature
    # Improved aesthetics for the plot
    plt.figure(figsize=(12, 8))
    cmap = plt.cm.viridis
    peak_numbers = sorted(combined['peak_number'].unique())
    colors = [cmap(i / max(len(peak_numbers) - 1, 1)) for i in range(len(peak_numbers))]

    # Plot each peak number as a separate line
    for i, peak_num in enumerate(peak_numbers):
        peak_data = combined[combined['peak_number'] == peak_num]
        if len(peak_data) > 0:
            peak_data = peak_data.sort_values('temperature')
            plt.plot(peak_data['temperature'], peak_data['peak_omega_squared_dimensionless'],
                     label=f'Peak {peak_num}', color=colors[i], marker='o', markersize=5, linewidth=1.5)

    # Customize the plot
    plt.xlabel('Temperature (MeV)', fontsize=14)
    plt.ylabel(r'$(\omega/\mu_g)^2$', fontsize=14)
    plt.title(f'Axial Peak Positions vs Temperature ($\lambda_1$={lambda1:.1f}, $\mu$={mu:.1f}, $m_q$={mq:.1f})', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    ax = plt.gca()

    # Determine y-axis limits and set ticks at multiples of 4
    y_max = combined['peak_omega_squared_dimensionless'].max()
    y_ticks = np.arange(0, y_max + 4, 4)
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # Configure grid
    ax.grid(which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, alpha=0.4)

    # Include legend
    plt.legend(loc='upper right', fontsize=12, ncol=2)

    # Adjust axis limits for better visualization
    plt.ylim(bottom=0)

    # Save the plot
    plot_png = os.path.join(
        out_dir,
        f"axial_peak_positions_vs_temperature_l1_{lambda1:.1f}_mu_{mu:.1f}_mq_{mq:.1f}.png"
    )
    plt.savefig(plot_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_png}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile and plot axial peak data across temperatures')
    parser.add_argument('--data-dir', default='mu_g_440/axial_data', help='Directory containing peak CSV files')
    parser.add_argument('--lambda1', type=float, required=True, help='Lambda1 value')
    parser.add_argument('--mu', type=float, required=True, help='Chemical potential')
    parser.add_argument('--mq', type=float, required=True, help='Quark mass')
    parser.add_argument('--out-dir', default='mu_g_440/axial_plots/compile_summary', help='Output directory for summary and plot')
    args = parser.parse_args()
    main(args.data_dir, args.lambda1, args.mu, args.mq, args.out_dir)
