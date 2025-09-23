#!/usr/bin/env python3
"""
Script to find vector meson melting temperatures as a function of chemical potential.

This script runs vector_spectra.py over a range of mu values to find the melting temperature
(temperature at which all peaks disappear) for each mu value. It starts at high temperatures
where no peaks are expected and decreases temperature until the first peak appears.

The melting temperature is defined as the highest temperature where no peaks are found.
"""

import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from datetime import datetime

def run_vector_spectra(wi, wf, wcount, wresolution, T, mu, normalize=False):
    """
    Run vector_spectra.py with given parameters and return the number of peaks found.
    
    Returns:
        int: Number of peaks found, or -1 if there was an error
    """
    cmd = [
        sys.executable, 'vector_spectra.py',
        '-wi', str(wi),
        '-wf', str(wf), 
        '-wcount', str(wcount),
        '-wresolution', str(wresolution),
        '-T', str(T),
        '-mu', str(mu)
    ]
    
    if normalize:
        cmd.append('--normalize')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Parse output to count peaks
        lines = result.stdout.split('\n')
        for line in lines:
            if 'peaks found:' in line.lower():
                # Extract number after "peaks found:"
                try:
                    num_peaks = int(line.split(':')[-1].strip())
                    return num_peaks
                except ValueError:
                    continue
        
        # If we can't find the peak count in stdout, check if there were any errors
        if result.returncode != 0:
            print(f"Error running vector_spectra.py: {result.stderr}")
            return -1
        
        # Default to 0 peaks if we can't parse the output
        return 0
        
    except subprocess.TimeoutExpired:
        print(f"Timeout running vector_spectra.py at T={T}, mu={mu}")
        return -1
    except Exception as e:
        print(f"Error running vector_spectra.py: {e}")
        return -1

def find_melting_temperature(mu, start_temp, temp_increment, wi, wf, wcount, wresolution, normalize=False):
    """
    Find the melting temperature for a given mu value.
    
    Args:
        mu: Chemical potential in MeV
        start_temp: Starting temperature (should be high enough that no peaks exist)
        temp_increment: Temperature increment in MeV
        wi, wf, wcount, wresolution: Frequency scan parameters
        normalize: Whether to use normalized spectrum
    
    Returns:
        float: Melting temperature in MeV, or None if not found (including if starting temp is too low)
    """
    print(f"\nFinding melting temperature for mu = {mu} MeV...")
    print(f"Starting at T = {start_temp} MeV, decrementing by {temp_increment} MeV")
    
    current_temp = start_temp
    melting_temp = None
    first_check = True
    
    while current_temp > 0:
        print(f"Testing T = {current_temp} MeV, mu = {mu} MeV...")
        
        num_peaks = run_vector_spectra(wi, wf, wcount, wresolution, current_temp, mu, normalize)
        
        if num_peaks == -1:
            print(f"Error at T = {current_temp} MeV, skipping...")
            current_temp -= temp_increment
            first_check = False
            continue
            
        print(f"Found {num_peaks} peaks at T = {current_temp} MeV")
        
        # Check if we already have peaks at the starting temperature
        if first_check and num_peaks > 0:
            print(f"Error: Already found {num_peaks} peaks at starting temperature {start_temp} MeV")
            print(f"Starting temperature is too low for mu = {mu} MeV")
            return None
        
        first_check = False
        
        if num_peaks == 0:
            # No peaks found, this could be our melting temperature
            melting_temp = current_temp
            current_temp -= temp_increment
        else:
            # Peaks found, we've gone too low
            break
    
    if melting_temp is not None:
        print(f"Melting temperature for mu = {mu} MeV: {melting_temp} MeV")
        return melting_temp
    else:
        print(f"Reached T <= 0 without finding melting point for mu = {mu} MeV; recording melting_temperature = 0 MeV")
        return 0

def load_or_create_melting_data(filename):
    """
    Load existing melting data or create new DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['mu', 'melting_temperature']
    """
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        print(f"Loaded existing melting data from {filename}")
        return df
    else:
        df = pd.DataFrame(columns=['mu', 'melting_temperature'])
        print(f"Created new melting data file: {filename}")
        return df

def save_melting_data(df, filename):
    """
    Save melting data to CSV file, sorted by mu.
    """
    # Sort by mu
    df_sorted = df.sort_values('mu').reset_index(drop=True)
    
    # Save to CSV
    df_sorted.to_csv(filename, index=False)
    print(f"Melting data saved to {filename}")
    
    return df_sorted

def plot_melting_curve(df, normalize=False):
    """
    Plot melting temperature vs mu.
    """
    if df.empty:
        print("No data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['mu'], df['melting_temperature'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Chemical potential μ (MeV)')
    plt.ylabel('Melting temperature T (MeV)')
    plt.title('Vector Meson Melting Temperature vs Chemical Potential')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, 'mu_g_440', 'vector_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save plot
    prefix = 'normalized_' if normalize else ''
    plot_filename = f'{prefix}vector_melting_curve.png'
    plot_path = os.path.join(plot_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    print(f"Melting curve plot saved to {plot_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Find vector meson melting temperatures vs chemical potential')
    
    # Mu scan parameters
    parser.add_argument('--mu_start', type=float, required=True, help='Starting mu value in MeV')
    parser.add_argument('--mu_end', type=float, required=True, help='Final mu value in MeV')
    parser.add_argument('--mu_step', type=float, required=True, help='Mu increment in MeV')
    
    # Temperature scan parameters
    parser.add_argument('--T_start', type=float, required=True, help='Starting temperature for first mu value in MeV')
    parser.add_argument('--T_step', type=float, default=5.0, help='Temperature increment in MeV (default: 5.0)')
    
    # Frequency scan parameters
    parser.add_argument('--wi', type=float, default=700, help='Initial frequency in MeV (default: 700)')
    parser.add_argument('--wf', type=float, default=1800, help='Final frequency in MeV (default: 1800)')
    parser.add_argument('--wcount', type=int, default=1100, help='Number of frequency points (default: 1100)')
    parser.add_argument('--wresolution', type=float, default=0.1, help='Target frequency resolution (default: 0.1)')
    
    # Normalization option
    parser.add_argument('--normalize', action='store_true', help='Use normalized spectrum for peak finding')
    
    # Global override for updating existing data
    parser.add_argument('--update', choices=['yes', 'no'], default=None, 
                       help='Global override for updating existing data points: "yes" to update all, "no" to skip all, or omit to prompt for each')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.mu_start >= args.mu_end:
        print("Error: mu_start must be less than mu_end")
        sys.exit(1)
    
    if args.mu_step <= 0:
        print("Error: mu_step must be positive")
        sys.exit(1)
    
    if args.T_step <= 0:
        print("Error: T_step must be positive")
        sys.exit(1)
    
    # Create output filename
    prefix = 'normalized_' if args.normalize else ''
    melting_data_filename = f'{prefix}vector_melting_data.csv'
    
    # Load existing data
    df = load_or_create_melting_data(melting_data_filename)
    
    # Generate mu values
    mu_values = np.arange(args.mu_start, args.mu_end + args.mu_step/2, args.mu_step)
    
    print(f"Scanning mu values: {mu_values}")
    print(f"Starting temperature: {args.tstart} MeV")
    print(f"Temperature step: {args.T_step} MeV")
    print(f"Frequency range: {args.wi} - {args.wf} MeV")
    print(f"Normalized spectrum: {args.normalize}")
    
    # Keep track of melting temperatures for next starting point
    current_start_temp = args.tstart
    
    for mu in mu_values:
        print(f"\n{'='*60}")
        print(f"Processing mu = {mu} MeV")
        print(f"{'='*60}")
        
        # Check if this mu value already exists in the data
        existing_row = df[df['mu'] == mu]
        if not existing_row.empty:
            print(f"Data for mu = {mu} MeV already exists: T_melt = {existing_row['melting_temperature'].iloc[0]} MeV")
            
            # Handle update decision based on global override or user prompt
            should_update = False
            if args.update == 'yes':
                print("Global override: Updating existing value")
                should_update = True
            elif args.update == 'no':
                print("Global override: Skipping existing value")
                should_update = False
            else:
                # Ask user if they want to update
                response = input("Update this value? (y/n): ")
                should_update = response.lower() in ['y', 'yes']
            
            if not should_update:
                # Use existing melting temperature for next starting point
                current_start_temp = existing_row['melting_temperature'].iloc[0] + args.T_step
                continue
        
        # Find melting temperature
        melting_temp = find_melting_temperature(
            mu, current_start_temp, args.T_step, 
            args.wi, args.wf, args.wcount, args.wresolution, 
            args.normalize
        )
        
        # Handle None (skip) and zero (terminate scan)
        if melting_temp is None:
            # Skip this mu value
            continue
        if melting_temp == 0:
            # Record zero and stop further mu values
            print(f"Recording melting_temperature = 0 for mu = {mu} MeV and terminating scan.")
            new_row = pd.DataFrame({'mu': [mu], 'melting_temperature': [0]})
            df = pd.concat([df, new_row], ignore_index=True)
            save_melting_data(df, melting_data_filename)
            break
        
        # Normal case: melting_temp > 0
        # Update or add data
        if not existing_row.empty:
            df.loc[df['mu'] == mu, 'melting_temperature'] = melting_temp
        else:
            new_row = pd.DataFrame({'mu': [mu], 'melting_temperature': [melting_temp]})
            df = pd.concat([df, new_row], ignore_index=True)
        
        # Save and update temperature for next mu
        save_melting_data(df, melting_data_filename)
        current_start_temp = melting_temp + args.T_step
    
    # Final save and plot
    if not df.empty:
        df_sorted = save_melting_data(df, melting_data_filename)
        plot_melting_curve(df_sorted, args.normalize)
    else:
        print("No melting temperatures found.")

if __name__ == "__main__":
    main()
