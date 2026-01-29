#!/usr/bin/env python3
# run_axial_temperature_scan.py
# Run a temperature scan for axial vector meson spectral functions

import numpy as np
import os
import sys
import time
import datetime
import importlib
import axial_spectra
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
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
            spectral_values = [np.abs(np.imag(sorted_peakBAs[i])) for i in group]
            best_idx = group[np.argmax(spectral_values)]
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
        import matplotlib.colormaps as colormaps
        cmap = colormaps['viridis']
    except ImportError:
        # For older matplotlib versions
        cmap = plt.get_cmap('viridis')
    
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
    plt.title('Axial Peak Positions vs Temperature with Mode Numbering', fontsize=16)
    
    # Set up grid with custom y-ticks at multiples of 4
    plt.grid(True, linestyle='--', alpha=0.7)
    ax = plt.gca()
    
    # Fixed y-axis ticks at multiples of 4 up to 24
    y_ticks = np.arange(0, 25, 4)
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
    
    # Set fixed y-axis limits
    plt.ylim(0, 24)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Also save as PDF for publication quality
    pdf_output_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_output_file, format='pdf', bbox_inches='tight')
    
    return output_file

def save_spectrum_plot(ws, BAs, peakws, peakBAs, mug, T, mu, mq, output_dir):
    """
    Save the spectral function plot for a specific temperature.
    
    Args:
        ws: Frequency values
        BAs: Spectral function values
        peakws: Peak frequency values
        peakBAs: Peak spectral function values
        mug: mu_g parameter
        T: Temperature in MeV
        mu: Chemical potential in MeV
        mq: Quark mass parameter
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the main spectral function
    spec_vals = np.abs(np.imag(BAs))
    plt.plot((ws/mug)**2, spec_vals)
    
    # Mark the peaks
    plt.scatter((peakws/mug)**2, np.abs(np.imag(peakBAs)), color='red', s=50)
    
    # Set plot labels and title
    plt.xlabel(r"$(\omega/\mu_g)^2$", fontsize=14)
    plt.ylabel(r"|Im$(B/A)$|", fontsize=14)
    plt.title(f"Axial Spectral Function (T={T} MeV, μ={mu} MeV, m_q={mq})", fontsize=16)
    
    # Set y-axis limits to either 0-2000 or max value (whichever is smaller)
    max_val = np.max(spec_vals)
    y_limit = min(3000, max_val * 1.1)  # Add 10% padding
    plt.ylim(0, y_limit)
    
    # Add grid and improve appearance
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Ensure the plots directory exists
    plots_dir = os.path.join(output_dir, 'spectrum_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save the plot (without timestamp so it can be overwritten on re-runs)
    plot_filename = os.path.join(plots_dir, f'axial_spectrum_T{T:.1f}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    # Also save as PDF for publication quality
    pdf_filename = os.path.join(plots_dir, f'axial_spectrum_T{T:.1f}.pdf')
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    
    plt.close()  # Close the figure to free memory

def run_axial_temperature_scan(t_min, t_max, t_step, mu_value=0, mq_value=9, lambda1_value=7.438,
                              wi_value=700, wf_value=2400, wcount_value=1700, load_if_exists=False):
    """
    Run axial vector spectra calculations for a range of temperature values.
    
    Parameters:
    t_min (float): Minimum temperature in MeV
    t_max (float): Maximum temperature in MeV
    t_step (float): Temperature step size
    mu_value (float): Chemical potential value to use for all runs
    mq_value (float): Quark mass value to use for all runs
    lambda1_value (float): Lambda1 parameter value to use for all runs
    wi_value (float): Initial frequency in MeV
    wf_value (float): Final frequency in MeV
    wcount_value (float): Number of frequency points
    load_if_exists (bool): Load cached spectra if available
    """
    # Create a directory to keep a record of the scan without timestamp
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scan_dir = os.path.join(script_dir, 'mu_g_440', 'axial_temperature_scan')
    os.makedirs(scan_dir, exist_ok=True)
    
    # Format parameter with consistent decimal places for file naming
    def format_param(value):
        """Format parameter with consistent decimal places for file naming"""
        if value == int(value):
            # For integer values, use 1 decimal place: '10.0' instead of '10'
            return f"{value:.1f}"
        else:
            # For non-integer values, use 1 decimal place
            return f"{value:.1f}"
    
    # Save scan parameters
    with open(os.path.join(scan_dir, 'scan_parameters.txt'), 'w') as f:
        f.write(f"Axial temperature scan from {t_min} to {t_max} MeV with step {t_step} MeV\n")
        f.write(f"Chemical potential: {mu_value} MeV\n")
        f.write(f"Quark mass: {mq_value}\n")
        f.write(f"Lambda1: {lambda1_value}\n")
        f.write(f"Frequency range: {wi_value} to {wf_value} MeV\n")
        f.write(f"Frequency points: {wcount_value}\n")
        f.write(f"Load cached spectra if exists: {load_if_exists}\n")
        f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Generate temperature values
    temperatures = np.arange(t_min, t_max + t_step/2, t_step)
    
    # Log the total number of temperatures
    print(f"Running axial temperature scan with {len(temperatures)} temperature values")
    print(f"Results will be saved in {scan_dir}")
    
    # Create summary dataframe to store peak positions vs temperature
    summary_data = []
    
    # Loop through temperatures
    for i, temp in enumerate(temperatures):
        start_time = time.time()
        print(f"\n[{i+1}/{len(temperatures)}] Starting axial calculation for T = {temp} MeV")
        
        # Format temperature for consistent file naming
        temp_str = format_param(temp)
        mu_str = format_param(mu_value)
        mq_str = format_param(mq_value)
        lambda1_str = format_param(lambda1_value)
        
        try:
            # Run the calculation using the main function with specified parameters
            ws, BAs, peakws, peakBAs, mug, _ = axial_spectra.main(
                T_value=temp,
                mu_value=mu_value,
                mq_value=mq_value,
                lambda1_value=lambda1_value,
                wi_value=wi_value,
                wf_value=wf_value,
                wcount_value=wcount_value,
                show_plot=False,  # Don't show plots during batch processing
                load_if_exists=load_if_exists
            )
            
            # Save spectrum plot for this temperature unless cached
            if not axial_spectra.last_used_cache:
                save_spectrum_plot(ws, BAs, peakws, peakBAs, mug, temp, mu_value, mq_value, scan_dir)
            
            # Clean the peaks data to remove duplicates within 5 MeV of each other
            cleaned_peakws, cleaned_peakBAs = clean_peaks_data(peakws, peakBAs, omega_threshold=5.0)
            
            # Add peaks to summary data
            for j, peak_w in enumerate(cleaned_peakws):
                summary_data.append({
                    'temperature': temp,
                    'peak_omega': peak_w,
                    'peak_omega_squared_dimensionless': (peak_w/mug)**2,
                    'peak_spectral_function_real': np.real(cleaned_peakBAs[j]),
                    'peak_spectral_function_imag': np.imag(cleaned_peakBAs[j]),
                    'peak_spectral_function_abs': np.abs(np.imag(cleaned_peakBAs[j])),
                    'mu_g': mug,
                    'chemical_potential': mu_value,
                    'quark_mass': mq_value,
                    'lambda1': lambda1_value,
                    'width': None  # Placeholder for width, will be calculated later if needed
                })
            
            # Report completion time
            elapsed_time = time.time() - start_time
            print(f"Completed T = {temp} MeV in {elapsed_time:.2f} seconds")
            print(f"Found {len(cleaned_peakws)} peaks at omega values: {[round(w, 1) for w in cleaned_peakws]}")
            
        except Exception as e:
            print(f"Error processing temperature T = {temp} MeV: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save summary data
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(scan_dir, 'axial_temperature_scan_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary data saved to {summary_file}")
        
        # Create improved peak positions vs temperature plot
        try:
            plot_file = os.path.join(scan_dir, 'axial_peak_positions_vs_temperature.png')
            plot_peaks_vs_temperature(summary_df, plot_file)
            print(f"Summary plot saved to {plot_file}")
            
        except Exception as e:
            print(f"Error creating summary plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Update scan parameters with completion time
    with open(os.path.join(scan_dir, 'scan_parameters.txt'), 'a') as f:
        f.write(f"\nCompleted at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total peaks found: {len(summary_data)}\n")
        f.write(f"Spectrum plots saved in: {os.path.join(scan_dir, 'spectrum_plots')}\n")
    
    return scan_dir, summary_file

if __name__ == "__main__":
    # Default parameters
    t_min = 10  # MeV
    t_max = 55  # MeV
    t_step = 5   # MeV
    mu_value = 0  # MeV
    mq_value = 9  # Default quark mass
    lambda1_value = 7.438  # Default lambda1 value
    wi_value = 700  # Default initial frequency
    wf_value = 2400  # Default final frequency
    wcount_value = 1700  # Default frequency count
    
    # Parse command line arguments if provided
    import argparse
    parser = argparse.ArgumentParser(description='Run axial vector temperature scan.')
    parser.add_argument('-t_min', '--min-temperature', type=float, default=t_min, help='Minimum temperature in MeV')
    parser.add_argument('-t_max', '--max-temperature', type=float, default=t_max, help='Maximum temperature in MeV')  
    parser.add_argument('-t_step', '--temperature-step', type=float, default=t_step, help='Temperature step size in MeV')
    parser.add_argument('-mu', '--chemical-potential', type=float, default=mu_value, help='Chemical potential in MeV')
    parser.add_argument('-mq', '--quark-mass', type=float, default=mq_value, help='Quark mass value')
    parser.add_argument('-l1', '--lambda1', type=float, default=lambda1_value, help='Lambda1 parameter value')
    parser.add_argument('-wi', '--omega-initial', type=float, default=wi_value, help='Initial frequency in MeV')
    parser.add_argument('-wf', '--omega-final', type=float, default=wf_value, help='Final frequency in MeV')
    parser.add_argument('-wc', '--omega-count', type=int, default=wcount_value, help='Number of frequency points')
    parser.add_argument('--load-if-exists', action='store_true', help='Load cached spectra if available')
    
    args = parser.parse_args()
    
    # Run the temperature scan
    run_axial_temperature_scan(
        args.min_temperature, 
        args.max_temperature, 
        args.temperature_step, 
        args.chemical_potential,
        args.quark_mass,
        args.lambda1,
        args.omega_initial,
        args.omega_final,
        args.omega_count,
        args.load_if_exists
    )
