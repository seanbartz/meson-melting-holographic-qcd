#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import re

def extract_timestamp_and_params(filename):
    """Extract temperature, mu, and timestamp from a filename."""
    # Different regex patterns to match all file naming conventions
    patterns = [
        r'T(\d+\.\d+)_mu(\d+\.?\d*)_(\d{8}_\d{6})\.csv',  # T10.0_mu0.0_timestamp
        r'T(\d+\.\d+)_mu(\d+)_(\d{8}_\d{6})\.csv',        # T10.0_mu0_timestamp
        r'T(\d+)_mu(\d+)_(\d{8}_\d{6})\.csv'              # T15_mu0_timestamp
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return float(match.group(1)), float(match.group(2)), match.group(3)
    
    return None, None, None

def get_peaks_data_file(data_dir, temperature, summary_timestamp=None):
    """
    Find the peaks_data file for a given temperature with smart matching.
    
    Args:
        data_dir: Directory containing peaks_data files
        temperature: Temperature value to match
        summary_timestamp: Optional timestamp to match
    
    Returns:
        Path to the matching peaks_data file or None if not found
    """
    # Generate potential file pattern formats for this temperature
    temp_formats = [
        f"T{temperature:.1f}",  # T10.0
        f"T{int(temperature)}" if temperature == int(temperature) else None  # T10
    ]
    temp_formats = [t for t in temp_formats if t is not None]
    
    # Try to find matching files for any of these formats
    for temp_format in temp_formats:
        # Try with different mu formats
        for mu_format in ["mu0.0", "mu0"]:
            pattern = os.path.join(data_dir, f"peaks_data_{temp_format}_{mu_format}_*.csv")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                # If we have multiple matches and a timestamp, try to find the matching timestamp
                if summary_timestamp and len(matching_files) > 1:
                    for file_path in matching_files:
                        _, _, file_timestamp = extract_timestamp_and_params(os.path.basename(file_path))
                        if file_timestamp == summary_timestamp:
                            return file_path
                
                # Otherwise return the first match
                return matching_files[0]
    
    return None

def update_summary_files():
    """
    Update temperature scan summary files by adding spectral function values
    from the corresponding peaks_data files.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # Find all temperature scan summary files
    temp_scan_files = []
    
    # Look for summary files in the main directory
    temp_scan_files.extend(glob.glob(os.path.join(script_dir, 'temperature_scan_summary_*.csv')))
    
    # Look for summary files in temperature scan subdirectories
    temp_scan_dirs = glob.glob(os.path.join(script_dir, 'temperature_scan_*'))
    for scan_dir in temp_scan_dirs:
        if os.path.isdir(scan_dir):
            summary_files = glob.glob(os.path.join(scan_dir, 'temperature_scan_summary_*.csv'))
            temp_scan_files.extend(summary_files)
    
    if not temp_scan_files:
        print("No temperature scan summary files found.")
        return
    
    print(f"Found {len(temp_scan_files)} temperature scan summary files.")
    
    # Process each summary file
    for summary_file in temp_scan_files:
        print(f"\nProcessing {os.path.basename(summary_file)}...")
        
        try:
            # Read the summary file
            summary_df = pd.read_csv(summary_file)
            if 'peak_spectral_function' in summary_df.columns:
                print(f"  File already has spectral function values, skipping.")
                continue
            
            # Get the summary timestamp
            summary_timestamp = os.path.basename(summary_file).split('_')[-1].split('.')[0]
            
            # Create a new dataframe to store updated data
            updated_rows = []
            
            # Get unique temperatures in this summary file
            temperatures = summary_df['temperature'].unique()
            
            found_spectral_values = 0
            missing_spectral_values = 0
            
            for temp in temperatures:
                # Find matching peaks_data file using smart matching
                matching_peaks_file = get_peaks_data_file(data_dir, temp, summary_timestamp)
                
                if matching_peaks_file:
                    print(f"  Found peaks data for T={temp}: {os.path.basename(matching_peaks_file)}")
                    peaks_df = pd.read_csv(matching_peaks_file)
                    
                    # Get summary rows for this temperature
                    temp_summary = summary_df[summary_df['temperature'] == temp]
                    
                    # For each peak in the summary, find the corresponding peak in the peaks data
                    for _, row in temp_summary.iterrows():
                        peak_omega = row['peak_omega']
                        
                        # Find closest peak in peaks data (might not be exact due to rounding)
                        if len(peaks_df) > 0:
                            closest_peak = peaks_df.iloc[(peaks_df['peak_omega'] - peak_omega).abs().argsort()[0]]
                            omega_diff = abs(closest_peak['peak_omega'] - peak_omega)
                            
                            # Only match if the peaks are within 1 MeV of each other (to avoid false matches)
                            if omega_diff <= 1.0:
                                # Create updated row with spectral function value
                                updated_row = row.to_dict()
                                updated_row['peak_spectral_function'] = closest_peak['peak_spectral_function']
                                updated_rows.append(updated_row)
                                found_spectral_values += 1
                            else:
                                # No close match found, use original row
                                updated_rows.append(row.to_dict())
                                missing_spectral_values += 1
                                print(f"    Warning: No matching peak for omega={peak_omega} in peaks data (closest was {closest_peak['peak_omega']} with diff {omega_diff:.2f} MeV)")
                        else:
                            # Empty peaks file, use original row
                            updated_rows.append(row.to_dict())
                            missing_spectral_values += 1
                else:
                    print(f"  No peaks data file found for T={temp}")
                    # Add original rows without spectral function values
                    temp_rows = summary_df[summary_df['temperature'] == temp].to_dict('records')
                    updated_rows.extend(temp_rows)
                    missing_spectral_values += len(temp_rows)
            
            # Create updated dataframe and save it back to the file
            if updated_rows:
                updated_df = pd.DataFrame(updated_rows)
                updated_df.to_csv(summary_file, index=False)
                print(f"  Updated {summary_file} with spectral function values.")
                print(f"  Found spectral values for {found_spectral_values} peaks, missing for {missing_spectral_values} peaks.")
            else:
                print(f"  No updates made to {summary_file}.")
        
        except Exception as e:
            print(f"  Error processing {summary_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nUpdating complete.")
    print("Next, run clean_temperature_scan_summary.py to remove duplicate peaks based on spectral function values.")

def main():
    update_summary_files()

if __name__ == "__main__":
    main()