#!/usr/bin/env python3
"""
Batch Phase Diagram Scanner

This script runs map_phase_diagram_improved.py for a range of gamma or lambda4 values
and creates combined plots showing how the phase diagram changes with these parameters.

Usage:
    # Scan over gamma values
    python batch_phase_diagram_scan.py --parameter gamma --values -25.0 -22.6 -20.0 -lambda1 5.0 -mq 9.0
    
    # Scan over lambda4 values  
    python batch_phase_diagram_scan.py --parameter lambda4 --values 3.0 4.2 5.5 -lambda1 5.0 -mq 9.0
    
    # Use range specification
    python batch_phase_diagram_scan.py --parameter gamma --range -25.0 -20.0 --num-values 6 -lambda1 5.0 -mq 9.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os
import sys
import subprocess
import time
from pathlib import Path

def run_phase_diagram(lambda1, ml, gamma=None, lambda4=None, mu_min=0.0, mu_max=200.0, mu_points=20, 
                      tmin=80.0, tmax=210.0, max_iterations=10):
    """
    Run map_phase_diagram_improved.py with specified parameters.
    
    Args:
        lambda1: Lambda1 parameter
        ml: Quark mass
        gamma: Gamma parameter (if None, uses default)
        lambda4: Lambda4 parameter (if None, uses default)
        mu_min: Minimum chemical potential
        mu_max: Maximum chemical potential
        mu_points: Number of mu points
        tmin: Minimum temperature
        tmax: Maximum temperature
        max_iterations: Maximum iterations
        
    Returns:
        tuple: (success, output_file, gamma_used, lambda4_used)
    """
    # Construct command
    cmd = [
        sys.executable, 'map_phase_diagram_improved.py',
        '-lambda1', str(lambda1), '-mq', str(ml),
        '-mumin', str(mu_min),
        '-mumax', str(mu_max),
        '-mupoints', str(mu_points),
        '-tmin', str(tmin),
        '-tmax', str(tmax),
        '-maxiterations', str(max_iterations),
        '--no-display'  # Don't display plots
    ]
    
    # Add gamma or lambda4 if specified
    gamma_used = -22.4  # default
    lambda4_used = 4.2  # default
    
    if gamma is not None:
        cmd.extend(['--gamma', str(gamma)])
        gamma_used = gamma
        
    if lambda4 is not None:
        cmd.extend(['--lambda4', str(lambda4)])
        lambda4_used = lambda4
    
    # Expected output file name
    output_file = f"phase_data/phase_diagram_improved_mq_{ml:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma_used:.1f}_lambda4_{lambda4_used:.1f}.csv"
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Expected output: {output_file}")
    
    try:
        # Run the command with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✓ Successfully completed for gamma={gamma_used:.1f}, lambda4={lambda4_used:.1f}")
            return True, output_file, gamma_used, lambda4_used
        else:
            print(f"✗ Error running phase diagram:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False, None, gamma_used, lambda4_used
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout (>1 hour) for gamma={gamma_used:.1f}, lambda4={lambda4_used:.1f}")
        return False, None, gamma_used, lambda4_used
    except Exception as e:
        print(f"✗ Exception: {str(e)}")
        return False, None, gamma_used, lambda4_used

def load_phase_diagram_data(csv_file):
    """
    Load phase diagram data from CSV file.
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        DataFrame or None if failed
    """
    try:
        df = pd.read_csv(csv_file)
        # Filter out failed calculations
        df_valid = df[df['order'].notna()].copy()
        return df_valid
    except Exception as e:
        print(f"Error loading {csv_file}: {str(e)}")
        return None

def create_combined_phase_diagram(data_files, parameter_values, parameter_name, lambda1, ml, 
                                 output_dir='phase_plots', show_axial=True, show_vector=True):
    """
    Create a combined phase diagram plot showing multiple parameter values.
    
    Args:
        data_files: List of CSV file paths
        parameter_values: List of parameter values corresponding to data files
        parameter_name: Name of the parameter being varied ('gamma' or 'lambda4')
        lambda1: Lambda1 value
        ml: Quark mass value
        output_dir: Directory to save plots
        show_axial: Whether to show axial melting lines
        show_vector: Whether to show vector melting line
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Color map for different parameter values
    colors = cm.viridis(np.linspace(0, 1, len(data_files)))
    
    # Load and plot each dataset
    valid_datasets = []
    for i, (csv_file, param_val, color) in enumerate(zip(data_files, parameter_values, colors)):
        if csv_file is None:
            continue
            
        df = load_phase_diagram_data(csv_file)
        if df is None or len(df) == 0:
            print(f"Skipping {parameter_name}={param_val:.1f} - no valid data")
            continue
            
        valid_datasets.append((df, param_val, color))
        
        # Separate first order and crossover points
        first_order = df[df['order'] == 1]
        crossover = df[df['order'] == 2]
        
        # Plot with parameter value in label
        param_latex = parameter_name.replace("lambda", "\\lambda_").replace("gamma", "\\gamma")
        label_base = f'${param_latex} = {param_val:.1f}$'
        
        if len(first_order) > 0:
            plt.plot(first_order['mu'], first_order['Tc'], 'o-', color=color,
                    label=f'{label_base} (1st order)', linewidth=2, markersize=6, alpha=0.8)
        
        if len(crossover) > 0:
            plt.plot(crossover['mu'], crossover['Tc'], 's--', color=color,
                    label=f'{label_base} (crossover)', linewidth=2, markersize=6, alpha=0.8)
    
    if len(valid_datasets) == 0:
        print("No valid datasets to plot!")
        return
    
    # Add reference lines if requested
    if show_vector:
        try:
            vector_data = np.loadtxt('vector_melting.csv', delimiter=',', skiprows=1)
            mu_vector = vector_data[:, 0]
            T_vector = vector_data[:, 1]
            plt.plot(mu_vector, T_vector, 'r--', linewidth=2, alpha=0.6, 
                    label='Vector Melting', zorder=0)
        except (FileNotFoundError, OSError):
            print("Vector melting data not found")
    
    if show_axial:
        # Try to load axial melting data for the first valid dataset's parameters
        first_df, first_param, _ = valid_datasets[0]
        gamma_val = first_df['gamma'].iloc[0] if parameter_name != 'gamma' else parameter_values[0]
        lambda4_val = first_df['lambda4'].iloc[0] if parameter_name != 'lambda4' else parameter_values[0]
        
        axial_filename = f'axial_data/axial_melting_data_mq_{ml:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma_val:.1f}_lambda4_{lambda4_val:.1f}.csv'
        try:
            axial_data = np.loadtxt(axial_filename, delimiter=',', skiprows=4)
            mu_axial = axial_data[:, 0]
            T_axial = axial_data[:, 1]
            plt.plot(mu_axial, T_axial, 'k-', linewidth=2, alpha=0.6,
                    label='Axial Melting', zorder=0)
        except (FileNotFoundError, OSError):
            print(f"Axial melting data not found: {axial_filename}")
    
    # Formatting
    plt.xlabel('Chemical Potential μ (MeV)', fontsize=14)
    plt.ylabel('Critical Temperature Tc (MeV)', fontsize=14)
    
    param_display = parameter_name.replace("lambda", "\\lambda_").replace("gamma", "\\gamma")
    plt.title(f'QCD Phase Diagram - ${param_display}$ Scan\n'
              f'$m_q = {ml}$ MeV, $\\lambda_1 = {lambda1:.3f}$', fontsize=16)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best', ncol=2)
    plt.tight_layout()
    
    # Save plot
    param_str = parameter_name.replace('lambda', 'lambda')  # For filename
    plot_filename = os.path.join(output_dir, 
                                f"combined_phase_diagram_{param_str}_scan_mq_{ml:.1f}_lambda1_{lambda1:.1f}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {plot_filename}")
    plt.close()  # Don't display, just save
    
    # Create a summary plot showing just the critical lines
    create_summary_plot(valid_datasets, parameter_values, parameter_name, lambda1, ml, output_dir)

def create_summary_plot(valid_datasets, parameter_values, parameter_name, lambda1, ml, output_dir):
    """
    Create a summary plot showing just the critical temperature lines.
    """
    plt.figure(figsize=(12, 8))
    
    colors = cm.plasma(np.linspace(0, 1, len(valid_datasets)))
    param_latex = parameter_name.replace("lambda", "\\lambda_").replace("gamma", "\\gamma")
    
    for (df, param_val, _), color in zip(valid_datasets, colors):
        # Sort by mu for smooth lines
        df_sorted = df.sort_values('mu')
        
        plt.plot(df_sorted['mu'], df_sorted['Tc'], 'o-', color=color,
                linewidth=3, markersize=6, alpha=0.8,
                label=f'${param_latex} = {param_val:.1f}$')
    
    plt.xlabel('Chemical Potential μ (MeV)', fontsize=14)
    plt.ylabel('Critical Temperature Tc (MeV)', fontsize=14)
    
    param_display = parameter_name.replace("lambda", "\\lambda_").replace("gamma", "\\gamma")
    plt.title(f'Critical Temperature Lines - ${param_display}$ Scan\n'
              f'$m_q = {ml}$ MeV, $\\lambda_1 = {lambda1:.3f}$', fontsize=16)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    
    param_str = parameter_name.replace('lambda', 'lambda')
    summary_filename = os.path.join(output_dir, 
                                   f"summary_critical_lines_{param_str}_scan_mq_{ml:.1f}_lambda1_{lambda1:.1f}.png")
    plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
    print(f"Summary plot saved to: {summary_filename}")
    plt.close()

def create_parameter_evolution_plot(valid_datasets, parameter_values, parameter_name, lambda1, ml, output_dir):
    """
    Create a plot showing how critical temperature at specific mu values changes with the parameter.
    """
    # Choose a few representative mu values
    mu_targets = [0, 50, 100, 150, 200]
    
    plt.figure(figsize=(12, 8))
    
    for mu_target in mu_targets:
        tc_values = []
        param_vals_with_data = []
        
        for df, param_val, _ in valid_datasets:
            # Find closest mu value to target
            if len(df) > 0:
                closest_idx = np.argmin(np.abs(df['mu'] - mu_target))
                closest_mu = df.iloc[closest_idx]['mu']
                
                # Only use if mu is within 5 MeV of target
                if abs(closest_mu - mu_target) <= 5:
                    tc_values.append(df.iloc[closest_idx]['Tc'])
                    param_vals_with_data.append(param_val)
        
        if len(tc_values) >= 2:  # Need at least 2 points for a line
            plt.plot(param_vals_with_data, tc_values, 'o-', linewidth=2, markersize=6,
                    label=f'μ = {mu_target} MeV', alpha=0.8)
    
    param_display = parameter_name.replace("lambda", "\\lambda_").replace("gamma", "\\gamma")
    plt.xlabel(f'${param_display}$', fontsize=14)
    plt.ylabel('Critical Temperature Tc (MeV)', fontsize=14)
    plt.title(f'Critical Temperature vs ${param_display}$\n'
              f'$m_q = {ml}$ MeV, $\\lambda_1 = {lambda1:.3f}$', fontsize=16)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    
    param_str = parameter_name.replace('lambda', 'lambda')
    evolution_filename = os.path.join(output_dir, 
                                     f"parameter_evolution_{param_str}_scan_mq_{ml:.1f}_lambda1_{lambda1:.1f}.png")
    plt.savefig(evolution_filename, dpi=300, bbox_inches='tight')
    print(f"Evolution plot saved to: {evolution_filename}")
    plt.close()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Batch scan phase diagrams over parameter ranges')
    
    # Required arguments
    parser.add_argument('-lambda1', type=float, required=True, help='Lambda1 parameter')
    parser.add_argument('-mq', type=float, required=True, help='Quark mass in MeV')
    
    # Parameter to scan
    parser.add_argument('--parameter', choices=['gamma', 'lambda4'], required=True,
                       help='Parameter to scan over')
    
    # Parameter values - either explicit list or range
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--values', type=float, nargs='+', 
                      help='Explicit list of parameter values')
    group.add_argument('--range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                      help='Parameter range (min max)')
    
    parser.add_argument('--num-values', type=int, default=5,
                       help='Number of values in range (default: 5)')
    
    # Phase diagram parameters
    parser.add_argument('-mumin', type=float, default=0.0, help='Min chemical potential (MeV)')
    parser.add_argument('-mumax', type=float, default=200.0, help='Max chemical potential (MeV)')
    parser.add_argument('-mupoints', type=int, default=20, help='Number of mu points')
    parser.add_argument('-tmin', type=float, default=80.0, help='Min temperature (MeV)')
    parser.add_argument('-tmax', type=float, default=210.0, help='Max temperature (MeV)')
    parser.add_argument('-maxiter', type=int, default=10, help='Max iterations')
    
    # Fixed parameter values (for the parameter not being scanned)
    parser.add_argument('-gammafixed', type=float, default=-22.4,
                       help='Fixed gamma value when scanning lambda4 (default: -22.4)')
    parser.add_argument('-lambda4fixed', type=float, default=4.2,
                       help='Fixed lambda4 value when scanning gamma (default: 4.2)')
    
    # Plot options
    parser.add_argument('--no-axial', action='store_true', help='Don\'t show axial melting lines')
    parser.add_argument('--no-vector', action='store_true', help='Don\'t show vector melting line')
    parser.add_argument('--skip-existing', action='store_true', 
                       help='Skip calculations if output file already exists')
    
    args = parser.parse_args()
    
    # Determine parameter values to scan
    if args.values is not None:
        parameter_values = args.values
    else:
        parameter_values = np.linspace(args.range[0], args.range[1], args.num_values)
    
    print("=" * 70)
    print("BATCH PHASE DIAGRAM SCANNER")
    print("=" * 70)
    print(f"Parameter to scan: {args.parameter}")
    print(f"Parameter values: {parameter_values}")
    print(f"Fixed parameters: lambda1={args.lambda1}, mq={args.mq}")
    if args.parameter == 'gamma':
        print(f"Fixed lambda4: {args.lambda4_fixed}")
    else:
        print(f"Fixed gamma: {args.gamma_fixed}")
    print("=" * 70)
    
    # Run phase diagram calculations
    data_files = []
    successful_params = []
    total_start_time = time.time()
    
    for i, param_val in enumerate(parameter_values):
        print(f"\nStep {i+1}/{len(parameter_values)}: {args.parameter} = {param_val:.3f}")
        print("-" * 50)
        
        # Set parameters
        if args.parameter == 'gamma':
            gamma_val = param_val
            lambda4_val = args.lambda4_fixed
        else:
            gamma_val = args.gamma_fixed
            lambda4_val = param_val
        
        # Check if output file already exists
        expected_file = f"phase_data/phase_diagram_improved_mq_{args.mq:.1f}_lambda1_{args.lambda1:.1f}_gamma_{gamma_val:.1f}_lambda4_{lambda4_val:.1f}.csv"
        
        if args.skip_existing and os.path.exists(expected_file):
            print(f"Output file exists, skipping: {expected_file}")
            data_files.append(expected_file)
            successful_params.append(param_val)
            continue
        
        # Run calculation
        success, output_file, gamma_used, lambda4_used = run_phase_diagram(
            lambda1=args.lambda1,
            ml=args.mq,
            gamma=gamma_val,
            lambda4=lambda4_val,
            mu_min=args.mumin,
            mu_max=args.mumax,
            mu_points=args.mupoints,
            tmin=args.tmin,
            tmax=args.tmax,
            max_iterations=args.max_iterations
        )
        
        if success and output_file and os.path.exists(output_file):
            data_files.append(output_file)
            successful_params.append(param_val)
        else:
            print(f"Failed to generate data for {args.parameter}={param_val:.3f}")
            data_files.append(None)
    
    total_time = time.time() - total_start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("CALCULATION SUMMARY:")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Successful calculations: {len(successful_params)}/{len(parameter_values)}")
    if successful_params:
        print(f"Successful {args.parameter} values: {successful_params}")
    
    # Create combined plots if we have data
    if len(successful_params) >= 2:
        print("\nCreating combined plots...")
        
        # Filter to only successful datasets
        successful_files = [f for f in data_files if f is not None]
        
        create_combined_phase_diagram(
            successful_files, successful_params, args.parameter,
            args.lambda1, args.mq, show_axial=not args.no_axial, show_vector=not args.no_vector
        )
        
        # Also create parameter evolution plot
        valid_datasets = []
        for csv_file, param_val in zip(successful_files, successful_params):
            df = load_phase_diagram_data(csv_file)
            if df is not None and len(df) > 0:
                valid_datasets.append((df, param_val, None))
        
        if len(valid_datasets) >= 2:
            create_parameter_evolution_plot(valid_datasets, successful_params, args.parameter,
                                          args.lambda1, args.mq, 'phase_plots')
        
        print("All plots created successfully!")
    else:
        print(f"\nNeed at least 2 successful calculations to create plots (got {len(successful_params)})")
    
    return successful_params, data_files

if __name__ == '__main__':
    successful_params, data_files = main()
