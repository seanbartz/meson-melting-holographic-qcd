#!/usr/bin/env python3
# calculate_peak_widths.py
# Script to calculate FWHM for spectral peaks and update peak data files

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def quadratic_background(x, a, b, c):
    """
    Quadratic background function: a*x^2 + b*x + c
    """
    return a * x**2 + b * x + c

def find_nearest_idx(array, value):
    """Find index of nearest value in array"""
    return np.abs(array - value).argmin()

def find_fwhm_with_background_subtraction(x, y, peak_index, max_search_range=100):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a peak with background subtraction.
    
    Args:
        x: Array of x-values (omega)
        y: Array of y-values (spectral function)
        peak_index: Index of the peak in the arrays
        max_search_range: Maximum number of points to search on each side
        
    Returns:
        fwhm: The full width at half maximum
        background_subtracted: Boolean indicating if background subtraction was used
    """
    peak_height = y[peak_index]
    peak_position = x[peak_index]
    
    # Define regions for background fitting (away from the peak)
    # We'll use points that are far enough from the peak but still within a reasonable range
    left_bound = max(0, peak_index - max_search_range)
    right_bound = min(len(y) - 1, peak_index + max_search_range)
    
    # Create mask to exclude the peak region (use ~20% of points around the peak)
    peak_width_estimate = (x[right_bound] - x[left_bound]) * 0.2
    mask = np.abs(x - peak_position) > peak_width_estimate
    
    # Restrict mask to our search range
    mask = mask & (np.arange(len(x)) >= left_bound) & (np.arange(len(x)) <= right_bound)
    
    # Check if we have enough points for fitting
    if np.sum(mask) < 10:  # Need at least 10 points for a reasonable fit
        # Fall back to original method if we don't have enough points
        return find_fwhm_original(x, y, peak_index, max_search_range), False
    
    try:
        # Fit background to quadratic function
        popt, _ = curve_fit(quadratic_background, x[mask], y[mask])
        
        # Calculate background at all points
        background = quadratic_background(x, *popt)
        
        # Subtract background from original data
        y_corrected = y - background
        
        # Ensure the corrected peak is still positive
        if y_corrected[peak_index] <= 0:
            # If background subtraction makes the peak negative, fall back to original method
            return find_fwhm_original(x, y, peak_index, max_search_range), False
            
        # Find FWHM on the background-subtracted data
        return find_fwhm_original(x, y_corrected, peak_index, max_search_range), True
        
    except (RuntimeError, ValueError) as e:
        # If curve fitting fails, fall back to original method
        print(f"  Background fitting failed: {str(e)}")
        return find_fwhm_original(x, y, peak_index, max_search_range), False

def find_fwhm_original(x, y, peak_index, max_search_range=100):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a peak.
    Using a robust approach that handles asymmetric peaks better.
    
    Args:
        x: Array of x-values (omega)
        y: Array of y-values (spectral function)
        peak_index: Index of the peak in the arrays
        max_search_range: Maximum number of points to search on each side
        
    Returns:
        fwhm: The full width at half maximum
    """
    peak_height = y[peak_index]
    half_max = peak_height / 2.0
    
    # Search left of the peak for the half-maximum point
    left_idx = peak_index
    search_limit = max(0, peak_index - max_search_range)
    
    while left_idx > search_limit:
        if y[left_idx] <= half_max:
            # Found the left crossing point
            break
        left_idx -= 1
    
    if left_idx == search_limit and y[left_idx] > half_max:
        # Couldn't find a crossing point, use the search limit
        left_idx = search_limit
    
    # Search right of the peak for the half-maximum point
    right_idx = peak_index
    search_limit = min(len(y) - 1, peak_index + max_search_range)
    
    while right_idx < search_limit:
        if y[right_idx] <= half_max:
            # Found the right crossing point
            break
        right_idx += 1
    
    if right_idx == search_limit and y[right_idx] > half_max:
        # Couldn't find a crossing point, use the search limit
        right_idx = search_limit
    
    # If we have valid crossing points, use linear interpolation for more accurate width
    if left_idx > 0 and right_idx < len(y) - 1:
        # Interpolate to find more precise crossing points
        try:
            left_x = x[left_idx-1:left_idx+1]
            left_y = y[left_idx-1:left_idx+1]
            if left_y[0] != left_y[1]:  # Avoid division by zero
                left_interp = interp1d(left_y, left_x, fill_value="extrapolate")
                left_intercept = float(left_interp([half_max])[0])
            else:
                left_intercept = x[left_idx]
                
            right_x = x[right_idx:right_idx+2]
            right_y = y[right_idx:right_idx+2]
            if right_y[0] != right_y[1]:  # Avoid division by zero
                right_interp = interp1d(right_y, right_x, fill_value="extrapolate")
                right_intercept = float(right_interp([half_max])[0])
            else:
                right_intercept = x[right_idx]
                
            fwhm = right_intercept - left_intercept
        except (ValueError, IndexError):
            # Fallback to simple calculation if interpolation fails
            fwhm = x[right_idx] - x[left_idx]
    else:
        # Simple calculation if we're at the edge of data
        fwhm = x[right_idx] - x[left_idx]
    
    # Return a minimum width of at least one data point spacing to avoid negative or zero widths
    return max(fwhm, np.mean(np.diff(x)))

def process_temperature_scan(summary_file):
    """
    Process a temperature scan summary file to calculate FWHM for each peak
    
    Args:
        summary_file: Path to temperature scan summary file
    """
    print(f"Processing summary file: {summary_file}")
    
    # Load temperature scan summary
    scan_df = pd.read_csv(summary_file)
    
    # Get unique temperatures
    temperatures = scan_df['temperature'].unique()
    
    # Process each temperature
    for temp in temperatures:
        peaks_for_temp = scan_df[scan_df['temperature'] == temp]
        
        # Find spectral data file for this temperature
        spectral_data_pattern = f"spectral_data_T{temp:.1f}_mu0*.csv"
        spectral_files = glob.glob(os.path.join(os.path.dirname(__file__), "data", spectral_data_pattern))
        
        # Find peaks data file for this temperature
        peaks_data_pattern = f"peaks_data_T{temp:.1f}_mu0*.csv"
        peaks_files = glob.glob(os.path.join(os.path.dirname(__file__), "data", peaks_data_pattern))
        
        if not spectral_files:
            print(f"Warning: No spectral data file found for T={temp:.1f}")
            continue
            
        if not peaks_files:
            print(f"Warning: No peaks data file found for T={temp:.1f}")
            continue
        
        # Use the first matching file
        spectral_file = spectral_files[0]
        peaks_file = peaks_files[0]
        
        print(f"Processing T={temp:.1f} using {os.path.basename(spectral_file)}")
        
        # Load spectral data
        spectral_df = pd.read_csv(spectral_file)
        omega = spectral_df['omega'].values
        spectral_func = spectral_df['spectral_function'].values
        
        # Load peaks data
        peaks_df = pd.read_csv(peaks_file)
        
        # Check if 'width' column already exists
        if 'width' not in peaks_df.columns:
            peaks_df['width'] = np.nan
            
        # Check if 'background_subtracted' column already exists
        if 'background_subtracted' not in peaks_df.columns:
            peaks_df['background_subtracted'] = False
        
        # Process each peak for this temperature
        updated = False
        for _, peak_row in peaks_for_temp.iterrows():
            peak_omega = peak_row['peak_omega']
            
            # Skip if peak_omega is outside the range of omega in spectral data
            if peak_omega < omega.min() or peak_omega > omega.max():
                print(f"Warning: Peak at omega={peak_omega:.4f} is outside spectral data range [{omega.min():.4f}, {omega.max():.4f}]")
                continue
            
            # Find the index of this peak in the spectral data
            peak_index = np.argmin(np.abs(omega - peak_omega))
            
            # Verify if this is truly a local maximum
            # Check +/- 5 MeV around the reported peak
            search_range = 100  # Number of points to check
            start_idx = max(0, peak_index - search_range)
            end_idx = min(len(omega), peak_index + search_range)
            
            local_max_idx = start_idx + np.argmax(spectral_func[start_idx:end_idx])
            
            # Check if the peak position needs to be corrected
            if local_max_idx != peak_index:
                print(f"  Peak at {peak_omega} MeV (T={temp}) was adjusted from {omega[peak_index]} to {omega[local_max_idx]} MeV")
                peak_index = local_max_idx
            
            # First try with background subtraction
            fwhm, bg_subtracted = find_fwhm_with_background_subtraction(
                omega, spectral_func, peak_index, max_search_range=100
            )
            
            status = "with background subtraction" if bg_subtracted else "without background subtraction"
            print(f"  Peak at omega={omega[peak_index]:.4f}: FWHM = {fwhm:.4f} MeV ({status})")
            
            # Find the matching peak in peaks_df
            matching_peak = peaks_df[np.isclose(peaks_df['peak_omega'], peak_omega, atol=0.5)]
            
            if len(matching_peak) == 0:
                print(f"Warning: No matching peak found in peaks data file for peak_omega={peak_omega:.4f}")
                continue
                
            # Update the width for the peak
            peak_idx = matching_peak.index[0]
            peaks_df.at[peak_idx, 'width'] = fwhm
            peaks_df.at[peak_idx, 'background_subtracted'] = bg_subtracted
            
            # If peak location was adjusted, update that too
            if abs(omega[peak_index] - peak_omega) > 1e-6:
                peaks_df.at[peak_idx, 'peak_omega'] = omega[peak_index]
            
            updated = True
        
        # Save updated peaks data if any changes were made
        if updated:
            # Create backup of original file
            backup_file = f"{peaks_file}.bak"
            if not os.path.exists(backup_file):
                os.rename(peaks_file, backup_file)
                print(f"Backup created: {os.path.basename(backup_file)}")
                
            # Save updated file
            peaks_df.to_csv(peaks_file, index=False)
            print(f"Updated peaks file: {os.path.basename(peaks_file)}")

def find_temperature_scan_summary_files():
    """Find all temperature scan summary files in the workspace"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_scan_files = []
    
    # Look in script directory
    temp_scan_files.extend(glob.glob(os.path.join(script_dir, 'temperature_scan_summary_*.csv')))
    
    # Look in subdirectories
    for subdir in glob.glob(os.path.join(script_dir, '*/')):
        if os.path.isdir(subdir):
            temp_scan_files.extend(glob.glob(os.path.join(subdir, 'temperature_scan_summary_*.csv')))
            
    return temp_scan_files

def main():
    # Find temperature scan summary files
    summary_files = find_temperature_scan_summary_files()
    
    if not summary_files:
        print("Error: No temperature scan summary files found.")
        return
    
    print(f"Found {len(summary_files)} temperature scan summary file(s).")
    
    # Process each summary file
    for summary_file in summary_files:
        process_temperature_scan(summary_file)
    
    print("Done.")

if __name__ == "__main__":
    main()