import numpy as np
import os
import sys
import time
import datetime
import importlib
import vector_spectra
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
import argparse
import subprocess
import pandas as pd

def clean_peaks_data(peakws, peakBAs, omega_threshold=5.0):
    """
    Clean peaks data by removing duplicate peaks within threshold MeV of each other.
    Keep only the peak with the largest spectral function value.
    
    Args:
        peakws: Array of peak omega values
        peakBAs: Array of peak spectral function values
        omega_threshold: Threshold in MeV to consider peaks as duplicates
    
    Returns:
        cleaned_peakws, cleaned_peakBAs: Arrays with duplicates removed
    """
    if len(peakws) <= 1:
        # No duplicates possible with 0 or 1 peak
        return peakws, peakBAs
    
    # Sort peaks by omega
    sort_indices = np.argsort(peakws)
    sorted_peakws = peakws[sort_indices]
    sorted_peakBAs = peakBAs[sort_indices]
    
    # Group peaks that are within omega_threshold MeV of each other
    groups = []
    current_group = [0]  # Start with the first peak
    
    for i in range(1, len(sorted_peakws)):
        # If this peak is within threshold MeV of the first peak in the current group
        if abs(sorted_peakws[i] - sorted_peakws[current_group[0]]) <= omega_threshold:
            current_group.append(i)
        else:
            # This peak is not within threshold MeV, so start a new group
            groups.append(current_group)
            current_group = [i]
    
    # Add the last group
    if current_group:
        groups.append(current_group)
    
    # For each group, keep only the peak with the highest spectral function value
    cleaned_peakws = []
    cleaned_peakBAs = []
    
    for group in groups:
        if len(group) == 1:
            # No duplicates in this group
            cleaned_peakws.append(sorted_peakws[group[0]])
            cleaned_peakBAs.append(sorted_peakBAs[group[0]])
        else:
            # Find the peak with the highest spectral function value in this group
            best_idx = group[np.argmax([sorted_peakBAs[i] for i in group])]
            cleaned_peakws.append(sorted_peakws[best_idx])
            cleaned_peakBAs.append(sorted_peakBAs[best_idx])
    
    return np.array(cleaned_peakws), np.array(cleaned_peakBAs)

def track_peaks_across_temperatures(df):
    """
    Track peaks across temperature values to maintain consistent numbering.
    Use a sophisticated approach that tracks peaks by continuity of omega squared values.
    
    Args:
        df: DataFrame with peaks data
        
    Returns:
        DataFrame with consistent peak numbers across temperatures
    """
    # First assign sequential peak numbers for each temperature
    result_df = df.copy()
    result_df['peak_number'] = -1
    
    # Process each temperature group
    for temp in result_df['temperature'].unique():
        # Get peaks for this temperature and sort by omega
        temp_peaks = result_df[result_df['temperature'] == temp].sort_values('peak_omega')
        
        # Assign sequential numbers
        peak_numbers = list(range(1, len(temp_peaks) + 1))
        result_df.loc[temp_peaks.index, 'peak_number'] = peak_numbers
    
    # Get unique temperatures sorted
    temperatures = sorted(df['temperature'].unique())
    
    if len(temperatures) <= 1:
        # No tracking needed for a single temperature
        return result_df
    
    # Create a dictionary to store the tracks of each peak
    tracks = {}
    
    # Start with the first temperature and use its numbering
    first_temp = temperatures[0]
    first_temp_peaks = result_df[result_df['temperature'] == first_temp].sort_values('peak_omega_squared_dimensionless')
    
    # Initialize tracks with first temperature's peaks
    for _, row in first_temp_peaks.iterrows():
        peak_num = int(row['peak_number'])
        tracks[peak_num] = [(first_temp, row['peak_omega'], row['peak_omega_squared_dimensionless'])]
    
    # For each subsequent temperature, match peaks to existing tracks
    # based on continuity of peak_omega_squared_dimensionless values
    for i in range(1, len(temperatures)):
        curr_temp = temperatures[i]
        curr_peaks = result_df[result_df['temperature'] == curr_temp].sort_values('peak_omega_squared_dimensionless')
        
        # Get the dimensionless omega squared values from previous temperature
        prev_temp = temperatures[i-1]
        # Extract the last point in each track that matches the previous temperature
        prev_tracks = {pnum: [(t, o, osq) for t, o, osq in track if t == prev_temp][-1] 
                      for pnum, track in tracks.items() if any(t == prev_temp for t, _, _ in track)}
        
        # Get the dimensionless omega squared values
        prev_osq_values = {pnum: osq for pnum, (_, _, osq) in prev_tracks.items()}
        
        # For each peak at current temperature, find best match based on dimensionless omega squared value
        assigned_peak_nums = set()
        assigned_curr_indices = set()
        
        # First, create all possible matches and sort by difference in dimensionless omega squared
        all_matches = []
        for curr_idx, curr_row in curr_peaks.iterrows():
            curr_osq = curr_row['peak_omega_squared_dimensionless']
            for peak_num, prev_osq in prev_osq_values.items():
                diff = abs(curr_osq - prev_osq)
                # Only consider matches if the dimensionless omega squared values are reasonably close
                if diff < 4.0:  # Threshold of 4.0 (approximately one mode spacing)
                    all_matches.append((diff, curr_idx, peak_num))
        
        # Sort matches by difference
        all_matches.sort()
        
        # Assign matches in order of increasing difference
        for diff, curr_idx, peak_num in all_matches:
            if peak_num not in assigned_peak_nums and curr_idx not in assigned_curr_indices:
                # This is a good match
                curr_row = curr_peaks.loc[curr_idx]
                tracks[peak_num].append((curr_temp, curr_row['peak_omega'], curr_row['peak_omega_squared_dimensionless']))
                assigned_peak_nums.add(peak_num)
                assigned_curr_indices.add(curr_idx)
        
        # Handle unassigned peaks as new tracks
        new_peak_num = max(tracks.keys()) + 1 if tracks else 1
        for curr_idx, curr_row in curr_peaks.iterrows():
            if curr_idx not in assigned_curr_indices:
                # This is a new peak
                tracks[new_peak_num] = [(curr_temp, curr_row['peak_omega'], curr_row['peak_omega_squared_dimensionless'])]
                new_peak_num += 1
    
    # Create a mapping from (temperature, omega) to peak number
    peak_mapping = {}
    for peak_num, track in tracks.items():
        for temp, omega, _ in track:
            peak_mapping[(temp, omega)] = peak_num
    
    # Update peak numbers in the DataFrame
    for idx, row in result_df.iterrows():
        temp = row['temperature']
        omega = row['peak_omega']
        # Find the closest match if exact match is not found
        if (temp, omega) in peak_mapping:
            result_df.loc[idx, 'peak_number'] = peak_mapping[(temp, omega)]
        else:
            # Find closest omega for this temperature
            closest_matches = [(abs(omega - om), pnum) for (t, om), pnum in peak_mapping.items() if t == temp]
            if closest_matches:
                closest_matches.sort()
                if closest_matches[0][0] < 1.0:  # Within 1 MeV
                    result_df.loc[idx, 'peak_number'] = closest_matches[0][1]
    
    return result_df

def plot_peaks_vs_temperature(summary_df, output_file):
    """
    Create an improved peak positions vs temperature plot with numbered peaks.
    
    Args:
        summary_df: DataFrame containing the peak data
        output_file: Path to save the output plot
    """
    # First track peaks across temperatures
    df = track_peaks_across_temperatures(summary_df)
    
    # Get unique peak numbers after tracking
    peak_numbers = sorted(df['peak_number'].unique())
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Use a colormap for different peak numbers
    try:
        # For newer matplotlib versions
        from matplotlib import colormaps
        cmap = colormaps['viridis']
    except (ImportError, AttributeError):
        # For older matplotlib versions
        cmap = plt.cm.viridis
    
    colors = [cmap(i / max(len(peak_numbers) - 1, 1)) for i in range(len(peak_numbers))]
    
    # Plot each peak number as a separate line
    for i, peak_num in enumerate(peak_numbers):
        peak_data = df[df['peak_number'] == peak_num]
        
        if len(peak_data) > 0:
            # Sort by temperature for proper line connection
            peak_data = peak_data.sort_values('temperature')
            
            # Only plot if we have at least one point
            if len(peak_data) >= 1:
                plt.plot(peak_data['temperature'], peak_data['peak_omega_squared_dimensionless'], 
                         'o-', label=f'n={peak_num}', color=colors[i % len(colors)], 
                         linewidth=2, markersize=6)
    
    # Customize the plot
    plt.xlabel('Temperature (MeV)', fontsize=14)
    plt.ylabel(r'$(ω/μ_g)^2$', fontsize=14)
    plt.title('Peak Positions vs Temperature with Mode Numbering', fontsize=16)
    
    # Set up grid with custom y-ticks at multiples of 4
    plt.grid(True, linestyle='--', alpha=0.7)
    ax = plt.gca()
    
    # Determine y-axis limits and set ticks at multiples of 4
    y_max = df['peak_omega_squared_dimensionless'].max()
    y_ticks = np.arange(0, y_max + 4, 4)  # Start at 0, go up to y_max in steps of 4
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    # Configure grid
    ax.grid(which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, alpha=0.4)
    
    # Only include the most important modes in the legend to avoid overcrowding
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 10:
        # Select the first few and last few modes
        first_few = 5
        last_few = 5
        selected_indices = list(range(first_few)) + list(range(len(handles) - last_few, len(handles)))
        handles = [handles[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
    
    plt.legend(handles, labels, loc='upper right', fontsize=12, ncol=2)
    
    # Adjust axis limits for better visualization
    plt.ylim(bottom=0)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Also save as PDF for publication quality
    pdf_output_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_output_file, format='pdf', bbox_inches='tight')
    
    return output_file

def run_temperature_scan(t_min, t_max, t_step, mu_value=0, wi=700, wf=1800, wcount=1100, wresolution=0.1, normalize=False):
    """
    Run vector spectra calculations for a range of temperature values.
    
    Parameters:
    t_min (float): Minimum temperature in MeV
    t_max (float): Maximum temperature in MeV
    t_step (float): Temperature step size
    mu_value (float): Chemical potential value to use for all runs
    wi, wf, wcount, wresolution: Frequency scan parameters passed to vector_spectra.py
    """
    # Remove all timestamp usage
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'mu_g_440', 'vector_data')
    plots_dir = os.path.join(script_dir, 'mu_g_440', 'vector_plots')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Format temperature and mu value for consistent file naming
    def format_param(value):
        """Format parameter with consistent decimal places for file naming"""
        if value == int(value):
            return f"{value:.1f}"
        else:
            return f"{value:.1f}"

    # Save scan parameters (no timestamp)
    with open(os.path.join(data_dir, 'scan_parameters.txt'), 'w') as f:
        f.write(f"Temperature scan from {t_min} to {t_max} MeV with step {t_step} MeV\n")
        f.write(f"Chemical potential: {mu_value} MeV\n")
        f.write(f"Frequency range: {wi} to {wf} MeV\n")
        f.write(f"Frequency resolution: {wresolution} MeV (count={wcount})\n")

    # Generate temperature values
    temperatures = np.arange(t_min, t_max + t_step/2, t_step)

    # Log the total number of temperatures
    print(f"Running temperature scan with {len(temperatures)} temperature values")
    print(f"Results will be saved in {data_dir} and plots in {plots_dir}")

    # Create summary dataframe to store peak positions vs temperature
    summary_data = []

    # Loop through temperatures
    max_peaks = None  # Track the maximum number of peaks allowed at each step
    for i, temp in enumerate(temperatures):
        start_time = time.time()
        print(f"\n[{i+1}/{len(temperatures)}] Starting calculation for T = {temp} MeV")

        try:
            # Call vector_spectra.py as subprocess with frequency scan parameters
            cmd = [sys.executable, os.path.join(script_dir, 'vector_spectra.py'),
                   '--wi', str(wi), '--wf', str(wf), '--wcount', str(wcount),
                   '--wresolution', str(wresolution), '--T', str(temp), '--mu', str(mu_value)]
            # Append normalize flag if requested
            if normalize:
                cmd.append('--normalize')
            print(f"Running vector_spectra subprocess: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Compute variables for summary
            vector_spectra.T = temp
            vector_spectra.mu = mu_value
            mug, zh, Q, lam = vector_spectra.calculate_variables()
            
            # Load generated data and peaks CSV
            data_file = os.path.join(data_dir, f"spectral_data_T{format_param(temp)}_mu{format_param(mu_value)}.csv")
            peaks_file = os.path.join(data_dir, f"peaks_data_T{format_param(temp)}_mu{format_param(mu_value)}.csv")
            df_data = pd.read_csv(data_file)
            ws = df_data['omega'].values
            BAs = df_data['spectral_function'].values
            df_peaks = pd.read_csv(peaks_file)
            peakws = df_peaks['peak_omega'].values
            peakBAs = df_peaks['peak_spectral_function'].values

            # Clean the peaks data to remove duplicates within 5 MeV of each other
            cleaned_peakws, cleaned_peakBAs = clean_peaks_data(peakws, peakBAs, omega_threshold=5.0)

            # Limit the number of peaks to the minimum found so far (not maximum)
            if max_peaks is not None and len(cleaned_peakws) > max_peaks:
                # Keep only the first max_peaks peaks (lowest omega)
                sort_indices = np.argsort(cleaned_peakws)
                cleaned_peakws = cleaned_peakws[sort_indices][:max_peaks]
                cleaned_peakBAs = cleaned_peakBAs[sort_indices][:max_peaks]
            # Update max_peaks for next temperature (minimum so far)
            max_peaks = len(cleaned_peakws) if max_peaks is None else min(max_peaks, len(cleaned_peakws))

            # Save plot and data (with consistent file naming)
            plot_file = os.path.join(plots_dir, f"vector_plot_T{format_param(temp)}_mu{format_param(mu_value)}.png")
            vector_spectra.plot_results(ws, BAs, cleaned_peakws, cleaned_peakBAs, mug, temp, mu_value, plot_file)

            data_file = os.path.join(data_dir, f"vector_data_T{format_param(temp)}_mu{format_param(mu_value)}.csv")
            vector_spectra.save_data(ws, BAs, cleaned_peakws, cleaned_peakBAs, mug, temp, mu_value, Q, zh, data_file)

            # Add peaks to summary data
            for i, peak_w in enumerate(cleaned_peakws):
                summary_data.append({
                    'temperature': temp,
                    'peak_omega': peak_w,
                    'peak_omega_squared_dimensionless': (peak_w/mug)**2,
                    'peak_spectral_function': cleaned_peakBAs[i],
                    'mu_g': mug,
                    'zh': zh,
                    'Q': Q
                })

            # Report completion time
            elapsed_time = time.time() - start_time
            print(f"Completed T = {temp} MeV in {elapsed_time:.2f} seconds")
            print(f"Found {len(cleaned_peakws)} peaks at omega values: {cleaned_peakws}")

        except Exception as e:
            print(f"Error processing temperature T = {temp} MeV: {str(e)}")
            import traceback
            traceback.print_exc()

    # Save summary data
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(data_dir, 'temperature_scan_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary data saved to {summary_file}")

        # Create improved peak positions vs temperature plot
        try:
            plot_file = os.path.join(plots_dir, f'peak_positions_vs_temperature_mu{mu_value}.png')
            plot_peaks_vs_temperature(summary_df, plot_file)
            print(f"Summary plot saved to {plot_file}")

        except Exception as e:
            print(f"Error creating summary plot: {str(e)}")
            import traceback
            traceback.print_exc()

    # Update scan parameters with completion time (no timestamp)
    with open(os.path.join(data_dir, 'scan_parameters.txt'), 'a') as f:
        f.write(f"\nScan completed. Total peaks found: {len(summary_data)}\n")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run vector temperature scan.')
    parser.add_argument('-t_min', '--min-temperature', type=float, required=True, help='Minimum temperature in MeV')
    parser.add_argument('-t_max', '--max-temperature', type=float, required=True, help='Maximum temperature in MeV')
    parser.add_argument('-t_step', '--temperature-step', type=float, required=True, help='Temperature step size in MeV')
    parser.add_argument('-mu', '--chemical-potential', type=float, required=True, help='Chemical potential in MeV')
    parser.add_argument('--normalize', action='store_true', help='Normalize spectrum by dividing by (ω/μ_g)^2')
    # Frequency scan parameters for vector_spectra.py
    parser.add_argument('-wi', type=float, default=700, help='Initial frequency in MeV')
    parser.add_argument('-wf', type=float, default=1800, help='Final frequency in MeV')
    parser.add_argument('-wcount', type=int, default=1100, help='Number of frequency points')
    parser.add_argument('-wresolution', type=float, default=0.1, help='Target frequency resolution in MeV')

    args = parser.parse_args()

    # Run the temperature scan with the provided arguments
    run_temperature_scan(
        t_min=args.min_temperature,
        t_max=args.max_temperature,
        t_step=args.temperature_step,
        mu_value=args.chemical_potential,
        wi=args.wi,
        wf=args.wf,
        wcount=args.wcount,
        wresolution=args.wresolution,
        normalize=args.normalize
    )
