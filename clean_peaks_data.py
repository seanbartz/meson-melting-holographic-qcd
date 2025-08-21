#!/usr/bin/env python3

import os
import pandas as pd
import glob
import numpy as np

def clean_peaks_data(file_path):
    """
    Clean peaks data by removing duplicate peaks within 5 MeV of each other.
    Keep only the peak with the largest spectral function value.
    
    Args:
        file_path: Path to the peaks_data CSV file
    
    Returns:
        True if file was modified, False otherwise
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    if len(df) <= 1:
        # No duplicates possible with 0 or 1 row
        return False
    
    # Sort by peak_omega
    df = df.sort_values('peak_omega')
    
    # Group peaks that are within 5 MeV of each other
    groups = []
    current_group = [0]  # Start with the first peak
    
    for i in range(1, len(df)):
        # If this peak is within 5 MeV of the first peak in the current group
        if abs(df.iloc[i]['peak_omega'] - df.iloc[current_group[0]]['peak_omega']) <= 5:
            current_group.append(i)
        else:
            # This peak is not within 5 MeV, so start a new group
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
            # Find the peak with the highest spectral function value in this group
            spectral_values = [df.iloc[i]['peak_spectral_function'] for i in group]
            max_index = group[spectral_values.index(max(spectral_values))]
            indices_to_keep.append(max_index)
    
    # Create a new DataFrame with only the peaks to keep
    cleaned_df = df.iloc[indices_to_keep].copy()
    
    # Check if any rows were removed
    if len(cleaned_df) < len(df):
        # Sort by peak_omega again to maintain the original order
        cleaned_df = cleaned_df.sort_values('peak_omega')
        
        # Write the cleaned data back to the file
        cleaned_df.to_csv(file_path, index=False)
        return True
    
    return False

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the data directory
    data_dir = os.path.join(script_dir, 'data')
    
    # Find all peaks_data_*.csv files
    peaks_files = glob.glob(os.path.join(data_dir, 'peaks_data_*.csv'))
    
    total_files = len(peaks_files)
    modified_files = 0
    
    print(f"Found {total_files} peaks data files to process...")
    
    # Process each file
    for i, file_path in enumerate(peaks_files, 1):
        print(f"Processing file {i}/{total_files}: {os.path.basename(file_path)}", end="... ")
        
        was_modified = clean_peaks_data(file_path)
        
        if was_modified:
            print("Cleaned up duplicates")
            modified_files += 1
        else:
            print("No duplicates found")
    
    print(f"\nProcessing complete. Modified {modified_files} out of {total_files} files.")

if __name__ == "__main__":
    main()