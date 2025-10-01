#!/usr/bin/env python3

import os
import pandas as pd
import glob
import numpy as np

def clean_temperature_scan_summary(file_path, omega_threshold=5.0):
    """
    Clean temperature scan summary by removing duplicate peaks within 5 MeV of each other
    for each temperature value. Keep only the peak with the highest spectral function value
    within each group.
    
    Args:
        file_path: Path to the temperature scan summary CSV file
        omega_threshold: Threshold in MeV to consider peaks as duplicates
    
    Returns:
        True if file was modified, False otherwise
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    if len(df) <= 1:
        # No duplicates possible with 0 or 1 row
        return False
    
    # Check if peak_spectral_function column exists
    has_spectral_values = 'peak_spectral_function' in df.columns
    
    # Group data by temperature
    temperatures = df['temperature'].unique()
    
    # Create a new dataframe to store cleaned data
    cleaned_rows = []
    modified = False
    
    # Process each temperature group separately
    for temp in temperatures:
        temp_df = df[df['temperature'] == temp].copy()
        
        # If there's only one peak for this temperature, keep it
        if len(temp_df) <= 1:
            cleaned_rows.extend(temp_df.to_dict('records'))
            continue
        
        # Sort by peak_omega
        temp_df = temp_df.sort_values('peak_omega')
        
        # Group peaks that are within threshold MeV of each other
        groups = []
        current_group = [0]  # Start with the first peak
        
        for i in range(1, len(temp_df)):
            # If this peak is within threshold MeV of the first peak in the current group
            if abs(temp_df.iloc[i]['peak_omega'] - temp_df.iloc[current_group[0]]['peak_omega']) <= omega_threshold:
                current_group.append(i)
            else:
                # This peak is not within threshold MeV, so start a new group
                groups.append(current_group)
                current_group = [i]
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        # For each group, keep only the peak with the highest spectral function value
        indices_to_keep = []
        
        for group in groups:
            if len(group) == 1:
                # No duplicates in this group
                indices_to_keep.append(group[0])
            else:
                if has_spectral_values:
                    # Use spectral function value to determine which peak to keep
                    values = [temp_df.iloc[i]['peak_spectral_function'] for i in group]
                    max_index = group[values.index(max(values))]
                    indices_to_keep.append(max_index)
                else:
                    # If spectral function values aren't available, we can't determine which peak to keep
                    # So just keep the first one in the group (or all peaks if preferred)
                    print(f"Warning: No spectral function values found for temperature {temp}.")
                    print(f"         Cannot determine which peak to keep in group with omega values:")
                    omega_values = [temp_df.iloc[i]['peak_omega'] for i in group]
                    print(f"         {omega_values}")
                    print(f"         Keeping the first peak in the group.")
                    indices_to_keep.append(group[0])
                
                # If we're removing peaks, the file is being modified
                if len(group) > 1:
                    modified = True
        
        # Add the selected rows to our cleaned data
        for idx in indices_to_keep:
            cleaned_rows.append(temp_df.iloc[idx].to_dict())
    
    # If we made modifications, write the cleaned data back to file
    if modified:
        cleaned_df = pd.DataFrame(cleaned_rows)
        # Sort by temperature and then by peak_omega
        cleaned_df = cleaned_df.sort_values(['temperature', 'peak_omega'])
        cleaned_df.to_csv(file_path, index=False)
        return True
    
    return False

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all temperature_scan_summary_*.csv files
    # Look in the workspace directory and any subdirectories
    temp_scan_files = []
    
    # First check current directory
    temp_scan_files.extend(glob.glob(os.path.join(script_dir, 'temperature_scan_summary_*.csv')))
    
    # Then check in subdirectories that might contain temperature scan results
    for subdir in glob.glob(os.path.join(script_dir, 'temperature_scan_*')):
        if os.path.isdir(subdir):
            temp_scan_files.extend(glob.glob(os.path.join(subdir, 'temperature_scan_summary_*.csv')))
    
    total_files = len(temp_scan_files)
    modified_files = 0
    
    if total_files == 0:
        print("No temperature scan summary files found.")
        return
    
    print(f"Found {total_files} temperature scan summary files to process...")
    
    # Process each file
    for i, file_path in enumerate(temp_scan_files, 1):
        print(f"Processing file {i}/{total_files}: {os.path.basename(file_path)}", end="... ")
        
        was_modified = clean_temperature_scan_summary(file_path)
        
        if was_modified:
            print("Cleaned up duplicates")
            modified_files += 1
        else:
            print("No duplicates found or removed")
    
    print(f"\nProcessing complete. Modified {modified_files} out of {total_files} files.")
    
    # Check if the data had spectral function values or not
    if total_files > 0:
        df = pd.read_csv(temp_scan_files[0])
        if 'peak_spectral_function' not in df.columns:
            print("\nNote: The summary files do not have spectral function values.")
            print("Run update_summary_with_spectral_values.py first to add these values.")
            print("This will ensure that duplicate peaks are removed based on spectral function values.")

if __name__ == "__main__":
    main()