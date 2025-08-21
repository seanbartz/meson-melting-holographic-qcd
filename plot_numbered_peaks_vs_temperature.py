#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import get_cmap
from scipy.interpolate import interp1d

def find_temperature_scan_summary_files():
    """
    Find all temperature scan summary files in the workspace.
    
    Returns:
        List of paths to summary files
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for summary files in the temperature scan directories
    temp_scan_files = []
    temp_scan_dirs = glob.glob(os.path.join(script_dir, 'temperature_scan_*'))
    for scan_dir in temp_scan_dirs:
        if os.path.isdir(scan_dir):
            summary_files = glob.glob(os.path.join(scan_dir, 'temperature_scan_summary_*.csv'))
            temp_scan_files.extend(summary_files)
    
    return temp_scan_files

def assign_peak_numbers(df):
    """
    For each temperature group, assign sequential numbers to peaks ordered by omega.
    
    Args:
        df: DataFrame containing temperature and peak data
        
    Returns:
        DataFrame with an additional 'peak_number' column
    """
    # Work with a copy to avoid modifying the original
    result_df = df.copy()
    
    # Add a peak_number column
    result_df['peak_number'] = -1
    
    # Process each temperature group
    for temp in result_df['temperature'].unique():
        # Get peaks for this temperature and sort by omega
        temp_peaks = result_df[result_df['temperature'] == temp].sort_values('peak_omega')
        
        # Assign sequential numbers
        peak_numbers = list(range(1, len(temp_peaks) + 1))
        result_df.loc[temp_peaks.index, 'peak_number'] = peak_numbers
    
    return result_df

def track_peaks_across_temperatures(df):
    """
    Track peaks across temperature values to maintain consistent numbering.
    Use a more sophisticated approach that tracks peaks by continuity of omega squared values.
    
    Args:
        df: DataFrame with peaks data
        
    Returns:
        DataFrame with consistent peak numbers across temperatures
    """
    # First, assign initial peak numbers based on ordering at each temperature
    df = assign_peak_numbers(df)
    
    # Get unique temperatures sorted
    temperatures = sorted(df['temperature'].unique())
    
    if len(temperatures) <= 1:
        # No tracking needed for a single temperature
        return df
    
    # Create a dictionary to store the tracks of each peak
    tracks = {}
    
    # Start with the first temperature and use its numbering
    first_temp = temperatures[0]
    first_temp_peaks = df[df['temperature'] == first_temp].sort_values('peak_omega_squared_dimensionless')
    
    # Initialize tracks with first temperature's peaks
    for _, row in first_temp_peaks.iterrows():
        peak_num = int(row['peak_number'])
        tracks[peak_num] = [(first_temp, row['peak_omega'], row['peak_omega_squared_dimensionless'])]
    
    # For each subsequent temperature, match peaks to existing tracks
    # based on continuity of peak_omega_squared_dimensionless values
    for i in range(1, len(temperatures)):
        curr_temp = temperatures[i]
        curr_peaks = df[df['temperature'] == curr_temp].sort_values('peak_omega_squared_dimensionless')
        
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
    
    # Now update the DataFrame with consistent peak numbers
    result_df = df.copy()
    
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

def load_peak_widths(summary_df):
    """
    Load width information from peak data files and merge with the summary DataFrame.
    
    Args:
        summary_df: DataFrame with peak summary data
        
    Returns:
        DataFrame with added width information
    """
    # Create a copy to avoid modifying the original
    df = summary_df.copy()
    
    # Add width column if it doesn't exist
    if 'width' not in df.columns:
        df['width'] = np.nan
    
    # Get all unique temperatures
    temperatures = df['temperature'].unique()
    
    # Get script directory for finding data files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # For each temperature, find the corresponding peaks data file
    for temp in temperatures:
        # Find peaks data file for this temperature
        peaks_files = glob.glob(os.path.join(data_dir, f"peaks_data_T{temp:.1f}_mu0*.csv"))
        
        if not peaks_files:
            print(f"Warning: No peaks data file found for T={temp}")
            continue
            
        # Use the first matching file
        peaks_file = peaks_files[0]
        
        try:
            # Load peaks data
            peaks_df = pd.read_csv(peaks_file)
            
            # Check if width column exists
            if 'width' not in peaks_df.columns:
                print(f"Warning: No width information in {os.path.basename(peaks_file)}")
                continue
                
            # Match peaks by omega values and add width information
            for _, row in df[df['temperature'] == temp].iterrows():
                peak_omega = row['peak_omega']
                
                # Find matching peak in peaks_df (allow small numerical differences)
                matching_peaks = peaks_df[np.isclose(peaks_df['peak_omega'], peak_omega, atol=0.5)]
                
                if len(matching_peaks) > 0:
                    # Get the width value
                    width = matching_peaks.iloc[0]['width']
                    
                    # Update width in summary DataFrame
                    df.loc[(df['temperature'] == temp) & (df['peak_omega'] == peak_omega), 'width'] = width
                
        except Exception as e:
            print(f"Error loading width data from {peaks_file}: {str(e)}")
    
    return df

def omega_to_omega_squared_dimensionless(omega, mug=388.0):
    """
    Convert omega (in MeV) to dimensionless (omega/mug)^2
    
    Args:
        omega: Energy in MeV
        mug: Energy scale (default 388.0 MeV)
        
    Returns:
        Dimensionless (omega/mug)^2 value
    """
    return (omega / mug) ** 2

def width_to_dimensionless_range(omega, width, mug=388.0):
    """
    Convert width in MeV to upper and lower bounds in dimensionless (omega/mug)^2
    
    Args:
        omega: Peak position in MeV
        width: Width (FWHM) in MeV
        mug: Energy scale (default 388.0 MeV)
    
    Returns:
        Tuple of (lower, upper) bounds in dimensionless units
    """
    # Convert omega to dimensionless units
    omega_dim = omega_to_omega_squared_dimensionless(omega, mug)
    
    # Limit width to a reasonable value to avoid excessively large bands
    # Maximum width is 20% of omega value for visualization purposes
    max_width = 0.2 * omega
    width = min(width, max_width)
    
    # Calculate upper and lower bounds in MeV
    omega_lower = omega - width / 2
    omega_upper = omega + width / 2
    
    # Convert bounds to dimensionless units
    lower_dim = omega_to_omega_squared_dimensionless(max(0, omega_lower), mug)
    upper_dim = omega_to_omega_squared_dimensionless(omega_upper, mug)
    
    return (lower_dim, upper_dim)

def plot_peaks_vs_temperature(summary_file, output_dir=None):
    """
    Create an improved peak positions vs temperature plot with width bands.
    
    Args:
        summary_file: Path to temperature scan summary file
        output_dir: Directory to save the plot (if None, use the summary file's directory)
    """
    # Read the summary data
    df = pd.read_csv(summary_file)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(summary_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp from filename for output file naming
    timestamp = os.path.basename(summary_file).split('_')[-1].split('.')[0]
    
    # Load width information from peak data files
    df = load_peak_widths(df)
    
    # Track peaks across temperatures to maintain consistent numbering
    df = track_peaks_across_temperatures(df)
    
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
            color = colors[i % len(colors)]
            
            # Only plot if we have at least one point
            if len(peak_data) >= 1:
                # First plot the width bands if width data is available
                for _, row in peak_data.iterrows():
                    # Only include widths that are reasonable (positive and not excessively large)
                    if (pd.notnull(row['width']) and 
                        row['width'] > 0 and 
                        row['width'] < row['peak_omega'] and
                        row['peak_omega_squared_dimensionless'] < 24):  # Only plot peaks within our y-axis range
                        
                        temp = row['temperature']
                        omega = row['peak_omega']
                        width = row['width']
                        
                        # Calculate width bands in dimensionless units
                        lower_dim, upper_dim = width_to_dimensionless_range(omega, width)
                        
                        # Only plot if the band is visible in our y-axis range
                        if lower_dim < 24 and upper_dim > 0:
                            # Limit band to our y-axis range
                            lower_dim = max(0, lower_dim)
                            upper_dim = min(24, upper_dim)
                            
                            # Plot vertical line with width
                            plt.plot([temp, temp], [lower_dim, upper_dim], '-', color=color, alpha=0.3, linewidth=8)
                
                # Then plot the line connecting peak positions
                # Only include points that are within our y-axis range
                visible_data = peak_data[peak_data['peak_omega_squared_dimensionless'] <= 24]
                
                if len(visible_data) > 0:
                    plt.plot(visible_data['temperature'], visible_data['peak_omega_squared_dimensionless'], 
                             'o-', label=f'n={peak_num}', color=color, 
                             linewidth=2, markersize=6)
    
    # Customize the plot
    plt.xlabel('Temperature (MeV)', fontsize=14)
    plt.ylabel(r'$(ω/μ_g)^2$', fontsize=14)
    plt.title('Peak Positions vs Temperature with Mode Numbering and Width', fontsize=16)
    
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
    
    # If we have too many legend items, select a subset
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
    output_file = os.path.join(output_dir, f'peak_positions_vs_temperature_numbered_with_width_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Also save as PDF for publication quality
    pdf_output_file = os.path.join(output_dir, f'peak_positions_vs_temperature_numbered_with_width_{timestamp}.pdf')
    plt.savefig(pdf_output_file, format='pdf', bbox_inches='tight')
    
    print(f"Plot saved to {output_file}")
    print(f"PDF saved to {pdf_output_file}")
    
    return output_file

def main():
    """Main function to find and process the temperature scan summary files."""
    summary_files = find_temperature_scan_summary_files()
    
    if not summary_files:
        print("No temperature scan summary files found.")
        return
    
    print(f"Found {len(summary_files)} temperature scan summary files.")
    
    for i, summary_file in enumerate(summary_files, 1):
        print(f"\nProcessing file {i}/{len(summary_files)}: {os.path.basename(summary_file)}")
        plot_file = plot_peaks_vs_temperature(summary_file)
        print(f"Created plot: {os.path.basename(plot_file)}")

if __name__ == "__main__":
    main()