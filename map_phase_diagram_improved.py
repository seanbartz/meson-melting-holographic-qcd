#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Phase Diagram Mapper

This script calls the critical_zoom_improved function to map the phase diagram over a range 
of chemical potential (mu) values for given input lambda1 and ml (quark mass).
Uses the improved sigma calculation method and optimizes parallelization.

Saves the critical points to CSV along with the order (1 for first order, 2 for crossover).

Created on July 24, 2025
@author: GitHub Copilot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
import time
import subprocess

# Import the improved critical_zoom function
from critical_zoom_improved import critical_zoom_improved

def map_phase_diagram_improved(mu_min, mu_max, mu_points, lambda1, ml, 
                              tmin=80, tmax=210, 
                              ui=1e-4, uf=None,
                              d0_lower=0, d0_upper=10, mq_tolerance=0.1,
                              max_iterations=10, gamma=-22.4, lambda4=4.2,
                              output_file=None, plot_results=True, display_plot=True):
    """
    Map the phase diagram over a range of chemical potential values using improved method.
    
    Args:
        mu_min: Minimum chemical potential (MeV)
        mu_max: Maximum chemical potential (MeV) 
        mu_points: Number of mu points to sample
        lambda1: Lambda1 parameter for mixing between dilaton and chiral field
        ml: Light quark mass (MeV)
        tmin: Minimum temperature for search (MeV)
        tmax: Maximum temperature for search (MeV)
        ui: Lower integration bound
        uf: Upper integration bound (if None, set to 1-ui)
        d0_lower: Lower bound for d0 search
        d0_upper: Upper bound for d0 search
        mq_tolerance: Tolerance for quark mass matching
        max_iterations: Maximum number of zoom iterations
        gamma: Background metric parameter (default: -22.4)
        lambda4: Fourth-order coupling parameter (default: 4.2)
        output_file: Output CSV filename (if None, auto-generated)
        plot_results: Whether to create a phase diagram plot
        display_plot: Whether to display the plot (default: True)
        
    Returns:
        DataFrame with critical points and phase transition information
    """
    
    # Optimize number of temperature points based on available CPUs
    cpu_count = os.cpu_count()
    numtemp = cpu_count  # Use exactly the number of CPUs for optimal parallelization
    
    print(f"Detected {cpu_count} CPU cores - using {numtemp} temperature points per iteration")
    
    if uf is None:
        uf = 1 - ui
    
    # Convert gamma and lambda4 to internal v3 and v4 parameters
    # v3 = gamma / (6 * sqrt(2))
    # v4 = lambda4
    import math
    v3 = gamma / (6 * math.sqrt(2))
    v4 = lambda4
    
    print(f"Using gamma = {gamma:.3f}, lambda4 = {lambda4:.3f}")
    print(f"Converted to internal parameters: v3 = {v3:.6f}, v4 = {v4:.3f}")
    
    # Create mu array
    mu_values = np.linspace(mu_min, mu_max, mu_points)
    
    # Lists to store results
    results = []
    
    # Record total start time
    total_start_time = time.time()
    
    print(f"Mapping phase diagram for ml={ml} MeV, lambda1={lambda1}")
    print(f"Chemical potential range: {mu_min} to {mu_max} MeV ({mu_points} points)")
    print(f"Using improved sigma calculation with optimized parallelization")
    print("=" * 70)
    
    for i, mu in enumerate(mu_values):
        print(f"\nProgress: {i+1}/{mu_points} - Processing mu = {mu:.2f} MeV")
        
        # Record start time for this mu value
        start_time = time.time()
        
        # Adaptive temperature bounds: for subsequent mu values, use previous Tc as guide
        if i > 0 and len(results) > 0:
            # Get the last successful critical temperature
            last_valid_results = [r for r in results if not np.isnan(r['Tc'])]
            if last_valid_results:
                last_Tc = last_valid_results[-1]['Tc']
                last_tmin_search = last_valid_results[-1]['tmin_search']
                
                # Set new tmax to previous Tc + 2 MeV buffer, but don't exceed original tmax
                adaptive_tmax = min(last_Tc + 2.0, tmax)
                
                # Adaptive tmin: if previous Tc was close to tmin, decrease tmin for next search
                if last_Tc < last_tmin_search + 10.0:
                    adaptive_tmin = max(last_tmin_search - 10.0, 10.0)  # Don't go below 10 MeV (positive definite)
                    print(f"  Previous Tc ({last_Tc:.1f}) close to tmin - decreasing tmin by 10 MeV")
                else:
                    adaptive_tmin = min(tmin, adaptive_tmax - 5.0)  # Ensure at least 5 MeV range
                
                # Ensure adaptive_tmin is always positive and reasonable
                adaptive_tmin = max(adaptive_tmin, 10.0)  # Absolute minimum of 10 MeV
                
                # Ensure adaptive_tmin < adaptive_tmax with reasonable range
                if adaptive_tmax - adaptive_tmin < 5.0:
                    adaptive_tmin = max(adaptive_tmax - 5.0, 10.0)  # Maintain minimum range but keep positive
                
                print(f"  Adaptive temperature range: {adaptive_tmin:.1f} - {adaptive_tmax:.1f} MeV")
                print(f"  (Based on previous Tc = {last_Tc:.1f} MeV)")
            else:
                adaptive_tmin, adaptive_tmax = tmin, tmax
        else:
            adaptive_tmin, adaptive_tmax = tmin, tmax
            print(f"  Initial temperature range: {adaptive_tmin:.1f} - {adaptive_tmax:.1f} MeV")
        
        try:
            # Call improved critical_zoom function with adaptive bounds
            order, iterationNumber, sigma_list, temps_list, Tc = critical_zoom_improved(
                adaptive_tmin, adaptive_tmax, numtemp, ml, mu, lambda1, ui, uf, 
                d0_lower, d0_upper, mq_tolerance, max_iterations, v3, v4
            )
            
            # Extract maximum sigma value from all iterations by flattening the list of arrays
            if sigma_list and len(sigma_list) > 0:
                # Flatten all sigma arrays from all iterations and find the maximum
                all_sigmas = np.concatenate([np.array(sigma_iter) for sigma_iter in sigma_list if len(sigma_iter) > 0])
                max_sigma = np.max(all_sigmas) if len(all_sigmas) > 0 else np.nan
            else:
                max_sigma = np.nan
            
            # Store results with additional metadata
            result_dict = {
                'mu': mu,
                'Tc': Tc,
                'order': order,
                'max_sigma': max_sigma,
                'iterations': iterationNumber,
                'tmin_search': adaptive_tmin,
                'tmax_search': adaptive_tmax,
                'numtemp_per_iter': numtemp,
                'total_temp_points': sum(len(temps) for temps in temps_list),
                'ui': ui,
                'uf': uf,
                'd0_lower': d0_lower,
                'd0_upper': d0_upper,
                'mq_tolerance': mq_tolerance,
                'max_iterations': max_iterations,
                'ml': ml,
                'lambda1': lambda1,
                'gamma': gamma,
                'lambda4': lambda4,
                'adaptive_bounds_used': i > 0 and len([r for r in results if not np.isnan(r['Tc'])]) > 0
            }
            
            results.append(result_dict)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Print result for this mu
            order_str = "First order" if order == 1 else "Crossover/2nd order"
            print(f"  Result: Tc = {Tc:.3f} MeV, Order = {order} ({order_str})")
            print(f"  Max sigma = {max_sigma:.6f}")
            print(f"  Calculation time: {elapsed_time:.2f} seconds")
            print(f"  Iterations: {iterationNumber}")
            print(f"  Total temperature points evaluated: {result_dict['total_temp_points']}")
            
        except Exception as e:
            # Calculate elapsed time even for failed calculations
            elapsed_time = time.time() - start_time
            
            print(f"  ERROR: Failed to process mu = {mu:.2f} MeV: {str(e)}")
            print(f"  Failed calculation time: {elapsed_time:.2f} seconds")
            # Store error result
            result_dict = {
                'mu': mu,
                'Tc': np.nan,
                'order': np.nan,
                'max_sigma': np.nan,
                'iterations': np.nan,
                'tmin_search': adaptive_tmin,
                'tmax_search': adaptive_tmax,
                'numtemp_per_iter': numtemp,
                'total_temp_points': np.nan,
                'ui': ui,
                'uf': uf,
                'd0_lower': d0_lower,
                'd0_upper': d0_upper,
                'mq_tolerance': mq_tolerance,
                'max_iterations': max_iterations,
                'ml': ml,
                'lambda1': lambda1,
                'gamma': gamma,
                'lambda4': lambda4
            }
            results.append(result_dict)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate total elapsed time
    total_elapsed_time = time.time() - total_start_time
    
    # Ensure output directories exist
    data_dir = 'CP_data'
    plot_dir = 'CP_plots' 
    axial_dir = 'axial_data'
    axial_plot_dir = 'axial_plots'
    
    for directory in [data_dir, plot_dir, axial_dir, axial_plot_dir]:
        os.makedirs(directory, exist_ok=True)

    # Generate output filename if not provided
    if output_file is None:
        output_file = os.path.join(data_dir, f"phase_diagram_improved_ml_{ml:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv")

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"Total points processed: {len(df)}")
    successful = df['order'].notna().sum()
    print(f"Successful calculations: {successful}")
    if successful > 0:
        first_order = (df['order'] == 1).sum()
        crossover = (df['order'] == 2).sum()
        print(f"First order transitions: {first_order}")
        print(f"Crossover/2nd order: {crossover}")
        print(f"Critical temperature range: {df['Tc'].min():.3f} - {df['Tc'].max():.3f} MeV")
        
        # Print efficiency statistics
        total_iterations = df['iterations'].sum()
        total_temp_points = df['total_temp_points'].sum()
        avg_iterations = df['iterations'].mean()
        avg_temp_points = df['total_temp_points'].mean()
        
        print(f"Total iterations across all mu values: {total_iterations}")
        print(f"Total temperature points evaluated: {total_temp_points}")
        print(f"Average iterations per mu: {avg_iterations:.1f}")
        print(f"Average temperature points per mu: {avg_temp_points:.1f}")
        print(f"CPU cores utilized: {cpu_count}")
        
    # Print total time information
    print(f"Total calculation time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
    if successful > 0:
        print(f"Average time per successful calculation: {total_elapsed_time/successful:.2f} seconds")
    
    # Create phase diagram plot if requested
    if plot_results and successful > 0:
        create_phase_diagram_plot(df, lambda1, ml, plot_dir, gamma, lambda4, display_plot)
    
    return df

def generate_axial_melting_data(ml, lambda1, mu_min=0.0, mu_max=200.0, mu_points=21, gamma=-22.4, lambda4=4.2):
    """
    Generate axial melting data by calling axial_melting_scan.py
    
    Args:
        ml: Quark mass value
        lambda1: Lambda1 parameter value
        mu_min: Minimum chemical potential
        mu_max: Maximum chemical potential
        mu_points: Number of mu points
        gamma: Chiral parameter gamma (default: -22.4)
        lambda4: Chiral parameter lambda4 (default: 4.2)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure axial_data directory exists
    os.makedirs('axial_data', exist_ok=True)
    
    axial_filename = f'axial_data/axial_melting_data_mq_{ml:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv'
    
    print(f"Generating axial melting data for mq={ml:.1f}, lambda1={lambda1:.1f}...")
    
    # Construct command to run axial_melting_scan.py
    cmd = [
        sys.executable, 'axial_melting_scan.py',
        '--mq', str(ml),
        '--lambda1', str(lambda1),
        '--gamma', str(gamma),
        '--lambda4', str(lambda4),
        '--mu-min', str(mu_min),
        '--mu-max', str(mu_max),
        '--mu-points', str(mu_points),
        '--no-display'  # Don't display plot, just save it
    ]
    
    try:
        # Run the axial melting scan
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            print(f"Successfully generated axial melting data: {axial_filename}")
            return True
        else:
            print(f"Error generating axial melting data:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout generating axial melting data (>30 minutes)")
        return False
    except Exception as e:
        print(f"Exception while generating axial melting data: {str(e)}")
        return False

def create_phase_diagram_plot(df, lambda1, ml, plot_dir, gamma=-22.4, lambda4=4.2, display_plot=True):
    """
    Create a phase diagram plot showing the critical line.
    
    Args:
        df: DataFrame with phase diagram data
        lambda1: Lambda1 parameter value
        ml: Quark mass value
        plot_dir: Directory to save the plot
        gamma: Chiral parameter gamma (default: -22.4)
        lambda4: Chiral parameter lambda4 (default: 4.2)
        display_plot: Whether to display the plot (default: True)
    """
    # Filter out failed calculations
    df_valid = df[df['order'].notna()].copy()
    
    if len(df_valid) == 0:
        print("No valid data points for plotting")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Load and plot vector melting data if it exists
    try:
        vector_data = np.loadtxt('vector_melting.csv', delimiter=',', skiprows=1)
        mu_vector = vector_data[:, 0]
        T_vector = vector_data[:, 1]
        plt.plot(mu_vector, T_vector, color='red', linestyle='--', 
                label='Vector Melting Line', linewidth=2, alpha=0.8)
    except (FileNotFoundError, OSError):
        print("Vector melting data file not found, skipping vector melting line")
    
    # Load and plot axial melting data if it exists, or generate it
    axial_filename = f'axial_data/axial_melting_data_mq_{ml:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv'
    axial_data_loaded = False
    
    try:
        axial_data = np.loadtxt(axial_filename, delimiter=',', skiprows=4)
        mu_axial = axial_data[:, 0]
        T_axial = axial_data[:, 1]
        plt.plot(mu_axial, T_axial, color='black', 
                label='Axial Melting Line', linewidth=2, alpha=0.8)
        axial_data_loaded = True
        print(f"Loaded existing axial melting data from {axial_filename}")
    except (FileNotFoundError, OSError):
        print(f"Axial melting data file {axial_filename} not found")
        
        # Try to generate the axial melting data
        mu_range = df_valid['mu'].max() - df_valid['mu'].min()
        mu_min_axial = max(0, df_valid['mu'].min() - 0.1 * mu_range)
        mu_max_axial = df_valid['mu'].max() + 0.1 * mu_range
        mu_points_axial = max(21, len(df_valid) + 5)  # At least as many points as phase diagram
        
        if generate_axial_melting_data(ml, lambda1, mu_min_axial, mu_max_axial, mu_points_axial, gamma, lambda4):
            # Try to load the newly generated data
            try:
                axial_data = np.loadtxt(axial_filename, delimiter=',', skiprows=4)
                mu_axial = axial_data[:, 0]
                T_axial = axial_data[:, 1]
                plt.plot(mu_axial, T_axial, color='black', 
                        label='Axial Melting Line', linewidth=2, alpha=0.8)
                axial_data_loaded = True
                print(f"Successfully generated and loaded axial melting data")
            except Exception as e:
                print(f"Failed to load newly generated axial melting data: {str(e)}")
    
    if not axial_data_loaded:
        print("Proceeding without axial melting line")
    
    # Separate first order and crossover points
    first_order = df_valid[df_valid['order'] == 1]
    crossover = df_valid[df_valid['order'] == 2]
    
    # Plot critical line with improved styling
    if len(first_order) > 0:
        plt.plot(first_order['mu'], first_order['Tc'], 'ro-', 
                label='First order transition', linewidth=3, markersize=8, alpha=0.9)
    
    if len(crossover) > 0:
        plt.plot(crossover['mu'], crossover['Tc'], 'bo-', 
                label='Crossover/2nd order', linewidth=3, markersize=8, alpha=0.9)
    
    plt.xlabel('Chemical Potential μ (MeV)', fontsize=14)
    plt.ylabel('Critical Temperature Tc (MeV)', fontsize=14)
    plt.title(f'QCD Phase Diagram\n'
              f'$m_q = {ml}$ MeV, $\\lambda_1 = {lambda1:.3f}$', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Add text annotations for critical endpoint if transition order changes
    if len(first_order) > 0 and len(crossover) > 0:
        # Find potential critical endpoint (transition between orders)
        all_data = df_valid.sort_values('mu')
        order_changes = np.where(np.diff(all_data['order']) != 0)[0]
        if len(order_changes) > 0:
            cep_idx = order_changes[0]
            cep_mu = all_data.iloc[cep_idx]['mu']
            cep_tc = all_data.iloc[cep_idx]['Tc']
            plt.plot(cep_mu, cep_tc, 'ks', markersize=12, 
                    label='Critical endpoint candidate')
            plt.annotate(f'CEP?\n(μ={cep_mu:.1f}, T={cep_tc:.1f})', 
                        xy=(cep_mu, cep_tc), xytext=(15, 15),
                        textcoords='offset points', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.tight_layout()
    
    # Save plot with improved naming
    plot_filename = os.path.join(plot_dir, f"phase_diagram_improved_ml_{ml:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Phase diagram plot saved to: {plot_filename}")
    
    # Only display plot if requested
    if display_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

def compare_with_original(df_improved, ml, lambda1):
    """
    Compare results with original method if available.
    
    Args:
        df_improved: DataFrame with improved method results
        ml: Quark mass value
        lambda1: Lambda1 parameter value
    """
    try:
        # Try to load original results
        original_file = f"CP_data/phase_diagram_ml{ml:.1f}_lambda1{lambda1:.1f}.csv"
        df_original = pd.read_csv(original_file)
        
        print("\n" + "=" * 70)
        print("COMPARISON WITH ORIGINAL METHOD:")
        
        # Filter valid data
        df_orig_valid = df_original[df_original['order'].notna()]
        df_impr_valid = df_improved[df_improved['order'].notna()]
        
        print(f"Original method: {len(df_orig_valid)} successful calculations")
        print(f"Improved method: {len(df_impr_valid)} successful calculations")
        
        # Compare critical temperatures for common mu values
        if len(df_orig_valid) > 0 and len(df_impr_valid) > 0:
            # Find overlapping mu values
            mu_orig = set(df_orig_valid['mu'].round(2))
            mu_impr = set(df_impr_valid['mu'].round(2))
            mu_common = mu_orig.intersection(mu_impr)
            
            if mu_common:
                print(f"Common mu values for comparison: {len(mu_common)}")
                
                tc_diffs = []
                for mu_val in mu_common:
                    tc_orig = df_orig_valid[df_orig_valid['mu'].round(2) == mu_val]['Tc'].iloc[0]
                    tc_impr = df_impr_valid[df_impr_valid['mu'].round(2) == mu_val]['Tc'].iloc[0]
                    tc_diff = abs(tc_orig - tc_impr)
                    tc_diffs.append(tc_diff)
                
                if tc_diffs:
                    print(f"Average |ΔTc|: {np.mean(tc_diffs):.4f} MeV")
                    print(f"Maximum |ΔTc|: {np.max(tc_diffs):.4f} MeV")
                    print(f"Standard deviation: {np.std(tc_diffs):.4f} MeV")
    
    except FileNotFoundError:
        print(f"\nOriginal results file not found: {original_file}")
        print("Cannot perform comparison with original method")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Map QCD phase diagram using improved critical_zoom function')
    
    # Required arguments
    parser.add_argument('lambda1', type=float, help='Lambda1 parameter for mixing between dilaton and chiral field')
    parser.add_argument('ml', type=float, help='Light quark mass in MeV')
    
    # Chemical potential range
    parser.add_argument('--mu-min', type=float, default=0.0, help='Minimum chemical potential in MeV (default: 0.0)')
    parser.add_argument('--mu-max', type=float, default=200.0, help='Maximum chemical potential in MeV (default: 200.0)')
    parser.add_argument('--mu-points', type=int, default=20, help='Number of mu points to sample (default: 20)')
    
    # Temperature search parameters
    parser.add_argument('--tmin', type=float, default=80.0, help='Minimum temperature for search in MeV (default: 80.0)')
    parser.add_argument('--tmax', type=float, default=210.0, help='Maximum temperature for search in MeV (default: 210.0)')
    
    # Integration parameters
    parser.add_argument('--ui', type=float, default=1e-2, help='Lower integration bound (default: 1e-2)')
    parser.add_argument('--uf', default=1-1e-4, type=float, help='Upper integration bound (default: 1-1e-4)')
    
    # Search parameters
    parser.add_argument('--d0-lower', type=float, default=0.0, help='Lower bound for d0 search (default: 0.0)')
    parser.add_argument('--d0-upper', type=float, default=10.0, help='Upper bound for d0 search (default: 10.0)')
    parser.add_argument('--mq-tolerance', type=float, default=0.01, help='Tolerance for quark mass matching (default: 0.01)')
    parser.add_argument('--max-iterations', type=int, default=10, help='Maximum number of zoom iterations (default: 10)')
    
    # Model parameters
    parser.add_argument('--gamma', type=float, default=-22.4, help='Background metric parameter (default: -22.4)')
    parser.add_argument('--lambda4', type=float, default=4.2, help='Fourth-order coupling parameter (default: 4.2)')
    
    # Output options
    parser.add_argument('-o', '--output', type=str, help='Output CSV filename (if not specified, auto-generated)')
    parser.add_argument('--no-plot', action='store_true', help='Do not create phase diagram plot')
    parser.add_argument('--no-display', action='store_true', help='Do not display plot (still saves plot file)')
    parser.add_argument('--compare', action='store_true', help='Compare with original method results if available')
    
    args = parser.parse_args()
    
    # Print system information
    print(f"System Information:")
    print(f"CPU cores detected: {os.cpu_count()}")
    print(f"Using {os.cpu_count()} temperature points per iteration for optimal parallelization")
    print()
    
    # Run improved phase diagram mapping
    df = map_phase_diagram_improved(
        mu_min=args.mu_min,
        mu_max=args.mu_max, 
        mu_points=args.mu_points,
        lambda1=args.lambda1,
        ml=args.ml,
        tmin=args.tmin,
        tmax=args.tmax,
        ui=args.ui,
        uf=args.uf,
        d0_lower=args.d0_lower,
        d0_upper=args.d0_upper,
        mq_tolerance=args.mq_tolerance,
        max_iterations=args.max_iterations,
        gamma=args.gamma,
        lambda4=args.lambda4,
        output_file=args.output,
        plot_results=not args.no_plot,
        display_plot=not args.no_display
    )
    
    # Compare with original method if requested
    if args.compare:
        compare_with_original(df, args.ml, args.lambda1)
    
    return df

if __name__ == '__main__':
    df = main()
