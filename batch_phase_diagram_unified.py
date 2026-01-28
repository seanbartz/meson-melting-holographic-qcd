#!/usr/bin/env python3
"""
Unified Batch Phase Diagram Scanner

This script runs map_phase_diagram_improved.py for ranges or fixed values of any combination
of parameters (mq, lambda1, gamma, lambda4) and creates combined plots showing how the 
phase diagram changes with these parameters.

Usage Examples:
    # Scan over gamma values, keeping other parameters fixed
    python batch_phase_diagram_unified.py -gammarange -25.0 -20.0 -gammapoints 6 -mq 9.0 -lambda1 5.0 -lambda4 4.2
    
    # Scan over multiple parameters simultaneously (Cartesian product)
    python batch_phase_diagram_unified.py -mqvalues 9.0 12.0 15.0 -lambda1range 3.0 7.0 -lambda1points 5 -gamma -22.4 -lambda4 4.2
    
    # Explicit values for any parameter
    python batch_phase_diagram_unified.py -gammavalues -25.0 -22.6 -20.0 -mq 9.0 -lambda1 5.0 -lambda4 4.2
    
    # Minimal scan (gamma and lambda4 use defaults: -22.4 and 4.2)
    python batch_phase_diagram_unified.py -mq 9.0 -lambda1 5.0
    
    # Single parameter scan with defaults
    python batch_phase_diagram_unified.py -mqvalues 9.0 12.0 15.0 -lambda1 5.0
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
import itertools
from pathlib import Path
from datetime import datetime

def parse_parameter_specification(args, param_name, default_value):
    """
    Parse parameter specification from command line arguments.
    
    Args:
        args: Parsed command line arguments
        param_name: Name of parameter (e.g., 'mq', 'lambda1', 'gamma', 'lambda4')
        default_value: Default value if parameter not specified
    
    Returns:
        list: List of parameter values to use
    """
    # Check for explicit values
    values_attr = f'{param_name}values'
    if hasattr(args, values_attr) and getattr(args, values_attr) is not None:
        return getattr(args, values_attr)
    
    # Check for range specification
    range_attr = f'{param_name}range'
    points_attr = f'{param_name}points'
    
    if hasattr(args, range_attr) and getattr(args, range_attr) is not None:
        range_vals = getattr(args, range_attr)
        num_points = getattr(args, points_attr, 5)  # default 5 points
        return np.linspace(range_vals[0], range_vals[1], num_points).tolist()
    
    # Check for single value
    if hasattr(args, param_name) and getattr(args, param_name) is not None:
        return [getattr(args, param_name)]
    
    # Use default
    return [default_value]

def run_phase_diagram(mq, lambda1, gamma, lambda4, mu_min=0.0, mu_max=200.0, mu_points=20, 
                      tmin=80.0, tmax=210.0, max_iterations=10, output_dir='phase_data'):
    """
    Run map_phase_diagram_improved.py with specified parameters.
    
    Args:
        mq: Quark mass
        lambda1: Lambda1 parameter
        gamma: Gamma parameter
        lambda4: Lambda4 parameter
        mu_min: Minimum chemical potential
        mu_max: Maximum chemical potential
        mu_points: Number of mu points
        tmin: Minimum temperature
        tmax: Maximum temperature
        max_iterations: Maximum iterations
        output_dir: Directory for output files
        
    Returns:
        tuple: (success, output_file)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct output filename with all parameters
    output_file = os.path.join(output_dir, 
        f"phase_diagram_improved_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv")
    
    # Construct command
    cmd = [
        sys.executable, 'map_phase_diagram_improved.py',
        '-mq', str(mq),
        '-lambda1', str(lambda1),
        '-gamma', str(gamma),
        '-lambda4', str(lambda4),
        '-mumin', str(mu_min),
        '-mumax', str(mu_max),
        '-mupoints', str(mu_points),
        '-tmin', str(tmin),
        '-tmax', str(tmax),
        '-maxiterations', str(max_iterations),
        '--no-display'  # Don't display plots
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Expected output: {output_file}")
    
    try:
        # Run the command with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✓ Successfully completed for mq={mq:.1f}, lambda1={lambda1:.1f}, gamma={gamma:.1f}, lambda4={lambda4:.1f}")
            return True, output_file
        else:
            print(f"✗ Error running phase diagram:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False, None
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout (>1 hour) for mq={mq:.1f}, lambda1={lambda1:.1f}, gamma={gamma:.1f}, lambda4={lambda4:.1f}")
        return False, None
    except Exception as e:
        print(f"✗ Exception: {str(e)}")
        return False, None

def log_physics_results(task_id, data_dir):
    if not os.path.exists('extract_physics_results.py'):
        print("Skipping physics summary logging: extract_physics_results.py not found")
        return False

    cmd = [
        sys.executable, 'extract_physics_results.py',
        '--task-id', task_id,
        '--data-dir', data_dir
    ]
    print(f"Logging physics summary: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Logged physics summary for {task_id}")
            return True
        print(f"⚠️  Physics summary logging failed for {task_id}")
        print(result.stdout)
        print(result.stderr)
        return False
    except Exception as e:
        print(f"⚠️  Physics summary logging exception: {e}")
        return False

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

def identify_varying_parameters(parameter_combinations):
    """
    Identify which parameters are varying across the combinations.
    
    Args:
        parameter_combinations: List of (mq, lambda1, gamma, lambda4) tuples
    
    Returns:
        dict: Parameter names mapped to their unique values
    """
    param_names = ['mq', 'lambda1', 'gamma', 'lambda4']
    varying_params = {}
    
    for i, param_name in enumerate(param_names):
        values = [combo[i] for combo in parameter_combinations]
        unique_values = sorted(set(values))
        if len(unique_values) > 1:
            varying_params[param_name] = unique_values
    
    return varying_params

def create_combined_phase_diagram(successful_runs, output_dir='phase_plots'):
    """
    Create combined phase diagram plots showing parameter variations.
    
    Args:
        successful_runs: List of (parameters, output_file) tuples
        output_dir: Directory to save plots
    """
    if not successful_runs:
        print("No successful runs to plot")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameter combinations
    parameter_combinations = [run[0] for run in successful_runs]
    output_files = [run[1] for run in successful_runs]
    
    # Identify varying parameters
    varying_params = identify_varying_parameters(parameter_combinations)
    
    if not varying_params:
        print("All parameters are fixed - creating single phase diagram")
        varying_params = {'fixed': [True]}  # Dummy for single plot
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Color map for different parameter combinations
    colors = cm.viridis(np.linspace(0, 1, len(successful_runs)))
    
    # Load and plot each dataset
    valid_datasets = []
    for i, ((mq, lambda1, gamma, lambda4), csv_file, color) in enumerate(zip(parameter_combinations, output_files, colors)):
        df = load_phase_diagram_data(csv_file)
        if df is not None and len(df) > 0:
            valid_datasets.append((mq, lambda1, gamma, lambda4, df))
            
            # Create label based on varying parameters
            label_parts = []
            if 'mq' in varying_params:
                label_parts.append(f'mq={mq:.1f}')
            if 'lambda1' in varying_params:
                label_parts.append(f'λ₁={lambda1:.1f}')
            if 'gamma' in varying_params:
                label_parts.append(f'γ={gamma:.1f}')
            if 'lambda4' in varying_params:
                label_parts.append(f'λ₄={lambda4:.1f}')
            
            if not label_parts:
                label = f'mq={mq:.1f}, λ₁={lambda1:.1f}, γ={gamma:.1f}, λ₄={lambda4:.1f}'
            else:
                label = ', '.join(label_parts)
            
            # Plot phase boundaries
            # Handle both numeric and string order values
            # Based on the CSV data: order=1 seems to be first order, order=2 seems to be crossover
            
            # First order boundary (order=1 or 'first_order')
            first_order_data = df[(df['order'] == 1) | (df['order'] == 'first_order')]
            if len(first_order_data) > 0:
                plt.scatter(first_order_data['mu'], first_order_data['Tc'], 
                           c=[color], alpha=0.7, s=30, marker='s', label=f'{label} (1st order)')
            
            # Crossover boundary (order=2 or 'crossover')
            crossover_data = df[(df['order'] == 2) | (df['order'] == 'crossover')]
            if len(crossover_data) > 0:
                plt.scatter(crossover_data['mu'], crossover_data['Tc'], 
                           c=[color], alpha=0.7, s=30, marker='o', label=f'{label} (crossover)')
            
            # Critical points (order=0 or 'critical')
            critical_data = df[(df['order'] == 0) | (df['order'] == 'critical')]
            if len(critical_data) > 0:
                plt.scatter(critical_data['mu'], critical_data['Tc'], 
                           c=[color], s=100, marker='*', label=f'{label} (critical)', 
                           edgecolors='black', linewidth=1)
    
    if not valid_datasets:
        print("No valid datasets found for plotting")
        return
    
    # Format plot
    plt.xlabel('Chemical Potential μ (MeV)', fontsize=14)
    plt.ylabel('Temperature T (MeV)', fontsize=14)
    plt.title('QCD Phase Diagram - Parameter Scan', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create filename based on varying parameters
    filename_parts = []
    for param, values in varying_params.items():
        if param != 'fixed':
            if len(values) == 2:
                filename_parts.append(f'{param}_{values[0]:.1f}to{values[-1]:.1f}')
            else:
                filename_parts.append(f'{param}_scan')
    
    if filename_parts:
        base_filename = f"phase_diagram_combined_{'_'.join(filename_parts)}"
    else:
        base_filename = f"phase_diagram_single"
    
    # Save plot
    plot_file = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Combined phase diagram saved to: {plot_file}")
    
    plt.close()

def run_batch_scan(mq_values, lambda1_values, gamma_values, lambda4_values, 
                   mu_min=0.0, mu_max=200.0, mu_points=20, 
                   tmin=80.0, tmax=210.0, max_iterations=10,
                   output_dir='phase_data', plot_dir='phase_plots'):
    """
    Run batch scan over parameter combinations.
    
    Args:
        mq_values: List of mq values
        lambda1_values: List of lambda1 values  
        gamma_values: List of gamma values
        lambda4_values: List of lambda4 values
        Other args: Parameters for map_phase_diagram_improved.py
    """
    # Create all parameter combinations (Cartesian product)
    parameter_combinations = list(itertools.product(mq_values, lambda1_values, gamma_values, lambda4_values))
    
    print("=" * 80)
    print("UNIFIED BATCH PHASE DIAGRAM SCANNER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Parameter Ranges:")
    print(f"  mq: {mq_values} ({len(mq_values)} values)")
    print(f"  lambda1: {lambda1_values} ({len(lambda1_values)} values)")
    print(f"  gamma: {gamma_values} ({len(gamma_values)} values)")
    print(f"  lambda4: {lambda4_values} ({len(lambda4_values)} values)")
    print(f"  Total combinations: {len(parameter_combinations)}")
    print(f"  Chemical potential: {mu_min} to {mu_max} MeV ({mu_points} points)")
    print(f"  Temperature search: {tmin} to {tmax} MeV")
    print("=" * 80)
    
    successful_runs = []
    failed_runs = []
    
    for i, (mq, lambda1, gamma, lambda4) in enumerate(parameter_combinations):
        print(f"\nProgress: {i+1}/{len(parameter_combinations)}")
        print(f"Processing: mq={mq:.1f}, lambda1={lambda1:.1f}, gamma={gamma:.1f}, lambda4={lambda4:.1f}")
        
        success, output_file = run_phase_diagram(
            mq, lambda1, gamma, lambda4, mu_min, mu_max, mu_points,
            tmin, tmax, max_iterations, output_dir
        )
        
        if success:
            successful_runs.append(((mq, lambda1, gamma, lambda4), output_file))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_id = (
                f"batch_{timestamp}_{i+1:03d}_"
                f"mq{mq:.1f}_lambda1{lambda1:.1f}_gamma{gamma:.1f}_lambda4{lambda4:.1f}"
            )
            log_physics_results(task_id, os.getcwd())
        else:
            failed_runs.append((mq, lambda1, gamma, lambda4))
    
    print("\n" + "=" * 80)
    print("BATCH SCAN SUMMARY")
    print("=" * 80)
    print(f"Successful runs: {len(successful_runs)}/{len(parameter_combinations)}")
    print(f"Failed runs: {len(failed_runs)}/{len(parameter_combinations)}")
    
    if failed_runs:
        print("\nFailed parameter combinations:")
        for mq, lambda1, gamma, lambda4 in failed_runs:
            print(f"  mq={mq:.1f}, lambda1={lambda1:.1f}, gamma={gamma:.1f}, lambda4={lambda4:.1f}")
    
    # Create combined plots
    if successful_runs:
        print("\nCreating combined phase diagram plots...")
        create_combined_phase_diagram(successful_runs, plot_dir)
    
    print("=" * 80)
    print(f"Batch scan completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description='Unified Batch Phase Diagram Scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Scan gamma while keeping other parameters fixed
    python batch_phase_diagram_unified.py -gammarange -25.0 -20.0 -gammapoints 6 -mq 9.0 -lambda1 5.0
    
    # Scan multiple parameters (Cartesian product)  
    python batch_phase_diagram_unified.py -mqvalues 9.0 12.0 -lambda1range 3.0 7.0 -lambda1points 5
    
    # Explicit values (gamma and lambda4 use defaults)
    python batch_phase_diagram_unified.py -gammavalues -25.0 -22.6 -20.0 -mq 9.0 -lambda1 5.0
    
    # Minimal required arguments (only mq and lambda1)
    python batch_phase_diagram_unified.py -mq 9.0 -lambda1 5.0
        """)
    
    # Parameter specifications - can use single value, explicit values, or range
    # mq parameter
    parser.add_argument('-mq', type=float, help='Single mq value')
    parser.add_argument('-mqvalues', type=float, nargs='+', help='Explicit mq values')
    parser.add_argument('-mqrange', type=float, nargs=2, metavar=('MIN', 'MAX'), help='mq range')
    parser.add_argument('-mqpoints', type=int, default=5, help='Number of mq points in range (default: 5)')
    
    # lambda1 parameter  
    parser.add_argument('-lambda1', type=float, help='Single lambda1 value')
    parser.add_argument('-lambda1values', type=float, nargs='+', help='Explicit lambda1 values')
    parser.add_argument('-lambda1range', type=float, nargs=2, metavar=('MIN', 'MAX'), help='lambda1 range')
    parser.add_argument('-lambda1points', type=int, default=5, help='Number of lambda1 points in range (default: 5)')
    
    # gamma parameter (optional, defaults to -22.4)
    parser.add_argument('-gamma', type=float, help='Single gamma value (default: -22.4)')
    parser.add_argument('-gammavalues', type=float, nargs='+', help='Explicit gamma values')
    parser.add_argument('-gammarange', type=float, nargs=2, metavar=('MIN', 'MAX'), help='gamma range')
    parser.add_argument('-gammapoints', type=int, default=5, help='Number of gamma points in range (default: 5)')
    
    # lambda4 parameter (optional, defaults to 4.2)
    parser.add_argument('-lambda4', type=float, help='Single lambda4 value (default: 4.2)')
    parser.add_argument('-lambda4values', type=float, nargs='+', help='Explicit lambda4 values')
    parser.add_argument('-lambda4range', type=float, nargs=2, metavar=('MIN', 'MAX'), help='lambda4 range')
    parser.add_argument('-lambda4points', type=int, default=5, help='Number of lambda4 points in range (default: 5)')
    
    # Physical parameters
    parser.add_argument('-mumin', type=float, default=0.0, help='Minimum chemical potential (default: 0.0)')
    parser.add_argument('-mumax', type=float, default=200.0, help='Maximum chemical potential (default: 200.0)')
    parser.add_argument('-mupoints', type=int, default=20, help='Number of mu points (default: 20)')
    parser.add_argument('-tmin', type=float, default=80.0, help='Minimum temperature (default: 80.0)')
    parser.add_argument('-tmax', type=float, default=210.0, help='Maximum temperature (default: 210.0)')
    parser.add_argument('-maxiter', type=int, default=10, help='Maximum iterations (default: 10)')
    
    # Output options (use double dash for non-numeric options)
    parser.add_argument('--output-dir', default='phase_data', help='Output directory for CSV files (default: phase_data)')
    parser.add_argument('--plot-dir', default='phase_plots', help='Output directory for plots (default: phase_plots)')
    
    args = parser.parse_args()
    
    # Parse parameter specifications with defaults
    # Required parameters (no defaults - must be specified or use defaults)
    mq_values = parse_parameter_specification(args, 'mq', 9.0)
    lambda1_values = parse_parameter_specification(args, 'lambda1', 5.0)
    # Optional parameters with standard defaults
    gamma_values = parse_parameter_specification(args, 'gamma', -22.4)
    lambda4_values = parse_parameter_specification(args, 'lambda4', 4.2)
    
    # Run the batch scan
    run_batch_scan(
        mq_values, lambda1_values, gamma_values, lambda4_values,
        args.mumin, args.mumax, args.mupoints,
        args.tmin, args.tmax, args.maxiter,
        args.output_dir, args.plot_dir
    )

if __name__ == '__main__':
    main()
