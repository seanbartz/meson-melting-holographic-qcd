#!/usr/bin/env python3
"""
Simple script to run axial_spectra.py for a range of temperature values.
This approach uses direct command line execution instead of importing the module.
"""

import os
import sys
import numpy as np
import subprocess
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

def run_temperature_scan(t_min, t_max, t_step, mu_value=0, mq_value=9, lambda1_value=7.438, expected_peaks_override=None,
                         wi_value=700, wf_value=2400, wcount_value=1700, wresolution_value=0.1, normalize=False):
    """
    Run axial_spectra.py for a range of temperatures by calling it directly.
    
    Parameters:
    t_min (float): Minimum temperature in MeV
    t_max (float): Maximum temperature in MeV
    t_step (float): Temperature step size
    mu_value (float): Chemical potential value to use for all runs
    mq_value (float): Quark mass value to use for all runs
    lambda1_value (float): Lambda1 parameter value to use for all runs
    wi_value (float): Initial frequency in MeV
    wf_value (float): Final frequency in MeV
    wcount_value (int): Number of frequency points
    """
    # Create output directory for summary
    summary_dir = os.path.join('mu_g_440', 'axial_temperature_scan')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Generate temperature values
    temperatures = np.arange(t_min, t_max + t_step/2, t_step)
    print(f"Running axial calculations for {len(temperatures)} temperature values")
    
    # Save scan parameters
    with open(os.path.join(summary_dir, 'scan_parameters.txt'), 'w') as f:
        f.write(f"Axial temperature scan from {t_min} to {t_max} MeV with step {t_step} MeV\n")
        f.write(f"Chemical potential: {mu_value} MeV\n")
        f.write(f"Quark mass: {mq_value}\n")
        f.write(f"Lambda1: {lambda1_value}\n")
        f.write(f"Frequency range: {wi_value} to {wf_value} MeV\n")
        f.write(f"Frequency points: {wcount_value}\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Dictionary to store the number of peaks found for each temperature
    peaks_by_temperature = {}
    sigma_by_temperature = {}

    # Run axial_spectra.py for each temperature
    for i, temp in enumerate(temperatures):
        print(f"\n[{i+1}/{len(temperatures)}] Running axial calculation for T = {temp:.1f} MeV")
        start_time = time.time()
        
        # Determine expected peak count (user override or from previous)
        if expected_peaks_override is not None:
            expected_peaks = expected_peaks_override
            print(f"Using user override expected peak count: {expected_peaks}")
        else:
            expected_peaks = None
            if i > 0:
                prev_temp = temperatures[i-1]
                if prev_temp in peaks_by_temperature:
                    expected_peaks = peaks_by_temperature[prev_temp]
                    print(f"Using expected peak count of {expected_peaks} from previous temperature T = {prev_temp:.1f} MeV")
        
        # Build command with all parameters
        cmd = [
            "python", "axial_spectra.py",
            "-T", f"{temp:.1f}",
            "-mu", f"{mu_value:.1f}",
            "-mq", f"{mq_value:.1f}",
            "-l1", f"{lambda1_value:.3f}",
            "-wi", f"{wi_value:.1f}",
            "-wf", f"{wf_value:.1f}",
            "-wc", f"{wcount_value}",
            "-wr", f"{wresolution_value}",
            "--no-plot"  # Don't display plots during batch processing
        ]
        
        # Add expected peak count if available
        if expected_peaks is not None:
            cmd.extend(["-ep", f"{expected_peaks}"])
        # Provide sigma output file for direct sigma passback
        sigma_fn = f"sigma_T{temp:.1f}_mu{mu_value:.1f}_mq{mq_value:.1f}_l1{lambda1_value:.1f}.csv"
        sigma_out = os.path.join(summary_dir, sigma_fn)
        cmd.extend(["--sigma-out", sigma_out])
        
        # Add normalize flag if requested
        if normalize:
            cmd.append('--normalize')
        # Run the command
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # Read sigma directly from CSV
            import pandas as pd
            sigma_df = pd.read_csv(sigma_out)
            sigma_val = sigma_df['sigma'].iloc[0]
            sigma_by_temperature[temp] = [sigma_val]

            # Existing stdout print for debugging
            print(result.stdout)
            if result.stderr:
                print(f"Stderr: {result.stderr}")
            
            elapsed_time = time.time() - start_time
            print(f"Completed T = {temp:.1f} MeV in {elapsed_time:.2f} seconds")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running axial_spectra.py for T = {temp:.1f} MeV: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")

    # After all calculations, create a summary from the individual peak data files
    create_summary_from_peak_files(summary_dir, temperatures, mu_value, mq_value, lambda1_value)
    
    # Save the peaks by temperature data for reference
    peaks_by_temp_df = pd.DataFrame(list(peaks_by_temperature.items()), columns=['temperature', 'peak_count'])
    peaks_by_temp_df = peaks_by_temp_df.sort_values('temperature')
    peaks_by_temp_file = os.path.join(summary_dir, 'peaks_by_temperature.csv')
    peaks_by_temp_df.to_csv(peaks_by_temp_file, index=False)
    print(f"Peak counts by temperature saved to {peaks_by_temp_file}")
    # Save sigma vs temperature
    if sigma_by_temperature:
        # Build DataFrame with one sigma per temperature
        rows = [{'temperature': T, 'sigma': sigma_by_temperature[T][0]}
                for T in sorted(sigma_by_temperature)]
        sigma_df = pd.DataFrame(rows)
        fn = f'sigma_vs_temperature_l1_{lambda1_value:.1f}_mu_{mu_value:.1f}_mq_{mq_value:.1f}.csv'
        sigma_csv = os.path.join(summary_dir, fn)
        sigma_df.to_csv(sigma_csv, index=False)
        print(f"Sigma values saved to {sigma_csv}")
        # Plot
        plt.figure(figsize=(8,6))
        plt.plot(sigma_df['temperature'], sigma_df['sigma'], 'o-')
        plt.xlabel('Temperature (MeV)')
        plt.ylabel('Sigma')
        plt.title(f'Sigma vs Temperature (l1={lambda1_value:.1f}, μ={mu_value:.1f}, mq={mq_value:.1f})')
        sigma_png = sigma_csv.replace('.csv', '.png')
        plt.savefig(sigma_png, dpi=300, bbox_inches='tight')
        print(f"Sigma plot saved to {sigma_png}")
    else:
        print("No sigma output files found; skipping sigma output.")

    # Update scan parameters with completion time
    with open(os.path.join(summary_dir, 'scan_parameters.txt'), 'a') as f:
        f.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Peak counts by temperature: {dict(sorted(peaks_by_temperature.items()))}\n")
    
    return summary_dir

def create_summary_from_peak_files(summary_dir, temperatures, mu_value, mq_value, lambda1_value):
    """
    Create a summary dataframe from individual peak files in axial_data directory.
    
    Parameters:
    summary_dir (str): Directory to save the summary data
    temperatures (list): List of temperatures that were calculated
    mu_value (float): Chemical potential value used
    mq_value (float): Quark mass value used
    lambda1_value (float): Lambda1 parameter value used
    """
    print("\nCompiling peak data from individual files...")
    # Use the axial_data directory (used by compile_peak_summary.py) for peak files
    data_dir = os.path.join("mu_g_440", "axial_data")
    # Save plots in the summary directory
    plot_dir = summary_dir
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create an empty list to hold peak data
    all_peaks_data = []
    
    # Loop through temperatures and read the corresponding peak files
    for temp in temperatures:
        # Adjust file path logic to ensure correct directory and naming conventions
        peak_file = os.path.join(data_dir, f"axial_peaks_data_T{temp:.1f}_mu{mu_value:.1f}_mq{mq_value:.1f}_lambda1{lambda1_value:.1f}.csv")

        # Debugging: Print the expected file path
        print(f"Looking for peak file: {peak_file}")

        try:
            if os.path.exists(peak_file):
                peaks_df = pd.read_csv(peak_file)
                peaks_df['temperature'] = temp
                peaks_df['chemical_potential'] = mu_value
                peaks_df['quark_mass'] = mq_value
                peaks_df['lambda1'] = lambda1_value
                all_peaks_data.append(peaks_df)
                print(f"Added {len(peaks_df)} peaks from T = {temp:.1f} MeV")
            else:
                print(f"Warning: Peak file not found for T = {temp:.1f} MeV")
        except Exception as e:
            print(f"Error processing peak file for T = {temp:.1f} MeV: {e}")
    if all_peaks_data:
        combined_df = pd.concat(all_peaks_data, ignore_index=True)
        # Generate peak numbers based on temperature groups, as in compile_peak_summary.py
        combined_df['peak_number'] = combined_df.groupby('temperature').cumcount() + 1
        # Save summary data
        summary_file = os.path.join(summary_dir, 'axial_temperature_summary.csv')
        combined_df.to_csv(summary_file, index=False)
        print(f"Summary data saved to {summary_file}")
        # Create a plot of peak positions vs temperature
        plt.figure(figsize=(12, 8))
        cmap = plt.cm.viridis
        peak_numbers = sorted(combined_df['peak_number'].unique())
        colors = [cmap(i / max(len(peak_numbers) - 1, 1)) for i in range(len(peak_numbers))]

        # Plot each peak number as a separate line
        for i, peak_num in enumerate(peak_numbers):
            peak_data = combined_df[combined_df['peak_number'] == peak_num]
            if len(peak_data) > 0:
                peak_data = peak_data.sort_values('temperature')
                plt.plot(peak_data['temperature'], peak_data['peak_omega_squared_dimensionless'],
                         label=f'Peak {peak_num}', color=colors[i], marker='o', markersize=5, linewidth=1.5)

        # Customize the plot
        plt.xlabel('Temperature (MeV)', fontsize=14)
        plt.ylabel(r'$(\omega/\mu_g)^2$', fontsize=14)
        plt.title(f'Axial Peak Positions vs Temperature ($\lambda_1$={lambda1_value:.1f}, $\mu$={mu_value:.1f}, $m_q$={mq_value:.1f})', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        ax = plt.gca()

        # Determine y-axis limits and set ticks at multiples of 4
        y_max = combined_df['peak_omega_squared_dimensionless'].max()
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
            plot_dir,
            f"axial_peak_positions_vs_temperature_l1_{lambda1_value:.1f}_mu_{mu_value:.1f}_mq_{mq_value:.1f}.png"
        )
        plt.savefig(plot_png, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_png}")
        # Also save scan_parameters.txt to data_dir
        scan_params_src = os.path.join(summary_dir, 'scan_parameters.txt')
        scan_params_dst = os.path.join(data_dir, 'scan_parameters.txt')
        if os.path.exists(scan_params_src):
            import shutil
            shutil.copyfile(scan_params_src, scan_params_dst)
            print(f"Scan parameters also saved to {scan_params_dst}")
    else:
        print("No peak data found to create summary.")

def plot_peaks_vs_temperature(df, output_file):
    """
    Create a plot of peak positions vs temperature.
    
    Args:
        df: DataFrame with the peak data
        output_file: Path to save the output plot
    """
    plt.figure(figsize=(12, 8))
    
    # Group by temperature
    temps = sorted(df['temperature'].unique())
    
    # Plot each peak as a point
    for temp in temps:
        temp_df = df[df['temperature'] == temp]
        plt.scatter([temp] * len(temp_df), temp_df['peak_omega_squared_dimensionless'], 
                   alpha=0.7, s=50)
    
    # Connect peaks that are close in frequency
    for i in range(len(temps) - 1):
        t1 = temps[i]
        t2 = temps[i + 1]
        
        df1 = df[df['temperature'] == t1]
        df2 = df[df['temperature'] == t2]
        
        for _, peak1 in df1.iterrows():
            # Find the closest peak in the next temperature
            if len(df2) > 0:
                diffs = abs(df2['peak_omega_squared_dimensionless'] - peak1['peak_omega_squared_dimensionless'])
                closest_idx = diffs.idxmin()
                closest_peak = df2.loc[closest_idx]
                
                # Only connect if the peaks are close enough (adjust threshold as needed)
                if abs(closest_peak['peak_omega_squared_dimensionless'] - peak1['peak_omega_squared_dimensionless']) < 4.0:
                    plt.plot([t1, t2], 
                            [peak1['peak_omega_squared_dimensionless'], closest_peak['peak_omega_squared_dimensionless']], 
                            'b-', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Temperature (MeV)', fontsize=14)
    plt.ylabel(r'$(ω/μ_g)^2$', fontsize=14)
    plt.title('Axial Peak Positions vs Temperature', fontsize=16)
    
    # Set up grid with custom y-ticks at multiples of 4
    plt.grid(True, linestyle='--', alpha=0.7)
    ax = plt.gca()
    
    # Fixed y-axis ticks at multiples of 4 up to 40
    y_ticks = np.arange(0, 41, 4)
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    # Configure grid
    ax.grid(which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, alpha=0.4)
    
    # Set y-axis limits
    plt.ylim(0, 40)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Also save as PDF for publication quality
    pdf_output_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_output_file, format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    # Default parameters
    t_min = 17  # MeV
    t_max = 60  # MeV
    t_step = 1   # MeV
    mu_value = 0  # MeV
    mq_value = 9  # Default quark mass
    lambda1_value = 7.438  # Default lambda1 value
    wi_value = 700  # Default initial frequency
    wf_value = 2400  # Default final frequency
    wcount_value = 1700  # Default frequency count
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run axial vector temperature scan using direct command line execution.')
    parser.add_argument('-t_min', '--min-temperature', type=float, default=t_min, help='Minimum temperature in MeV')
    parser.add_argument('-t_max', '--max-temperature', type=float, default=t_max, help='Maximum temperature in MeV')  
    parser.add_argument('-t_step', '--temperature-step', type=float, default=t_step, help='Temperature step size in MeV')
    parser.add_argument('-mu', '--chemical-potential', type=float, default=mu_value, help='Chemical potential in MeV')
    parser.add_argument('-mq', '--quark-mass', type=float, default=mq_value, help='Quark mass value')
    parser.add_argument('-l1', '--lambda1', type=float, default=lambda1_value, help='Lambda1 parameter value')
    parser.add_argument('-wi', '--omega-initial', type=float, default=wi_value, help='Initial frequency in MeV')
    parser.add_argument('-wf', '--omega-final', type=float, default=wf_value, help='Final frequency in MeV')
    parser.add_argument('-wc', '--omega-count', type=int, default=wcount_value, help='Number of frequency points')
    parser.add_argument('-wr', '--omega-resolution', type=float, default=0.1, help='Minimum frequency resolution (default: 0.1 MeV)')
    parser.add_argument('-ep', '--expected-peaks', type=int, help='Override expected peak count')
    parser.add_argument('--normalize', action='store_true', help='Normalize the output')
    
    args = parser.parse_args()
    
    # Run the temperature scan
    summary_dir = run_temperature_scan(
        args.min_temperature,
        args.max_temperature,
        args.temperature_step,
        args.chemical_potential,
        args.quark_mass,
        args.lambda1,
        args.expected_peaks,
        args.omega_initial,
        args.omega_final,
        args.omega_count,
        args.omega_resolution,
        args.normalize
    )
    
    print(f"\nTemperature scan complete. Results saved to {summary_dir}")