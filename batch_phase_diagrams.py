#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Phase Diagram Runner

This script runs map_phase_diagram.py for a range of lambda1 values to systematically
explore how the mixing parameter affects the QCD phase diagram.

Created on July 15, 2025
@author: GitHub Copilot
"""

import subprocess
import numpy as np
import os
import sys
import time
import argparse
from datetime import datetime

def run_phase_diagram_batch(lambda1_values, ml, mu_min=0.0, mu_max=300.0, mu_points=30,
                           tmin=50.0, tmax=100.0, numtemp=50, 
                           minsigma=0.0, maxsigma=400.0, a0=0.0,
                           base_output_dir='batch_results', 
                           keep_individual_plots=True):
    """
    Run map_phase_diagram.py for multiple lambda1 values.
    
    Args:
        lambda1_values: Array or list of lambda1 values to explore
        ml: Light quark mass (MeV)
        mu_min: Minimum chemical potential (MeV)
        mu_max: Maximum chemical potential (MeV)
        mu_points: Number of mu points to sample
        tmin: Minimum temperature for search (MeV)
        tmax: Maximum temperature for search (MeV)
        numtemp: Number of temperature points per iteration
        minsigma: Minimum sigma value for search
        maxsigma: Maximum sigma value for search
        a0: Additional parameter
        base_output_dir: Base directory for organizing results
        keep_individual_plots: Whether to keep individual plots for each lambda1
    """
    
    # Create timestamp for this batch run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(base_output_dir, f"batch_{timestamp}_ml{ml:.1f}")
    os.makedirs(batch_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"BATCH PHASE DIAGRAM CALCULATION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  Quark mass (ml): {ml} MeV")
    print(f"  Lambda1 range: {lambda1_values[0]:.3f} to {lambda1_values[-1]:.3f} ({len(lambda1_values)} values)")
    print(f"  Chemical potential: {mu_min} to {mu_max} MeV ({mu_points} points)")
    print(f"  Temperature search: {tmin} to {tmax} MeV ({numtemp} points)")
    print(f"  Output directory: {batch_dir}")
    print("=" * 80)
    
    # Log file for batch results
    log_file = os.path.join(batch_dir, "batch_log.txt")
    
    successful_runs = []
    failed_runs = []
    
    with open(log_file, 'w') as log:
        log.write(f"Batch Phase Diagram Calculation Log\n")
        log.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Parameters: ml={ml}, mu_range=[{mu_min},{mu_max}], mu_points={mu_points}\n")
        log.write(f"Lambda1 values: {lambda1_values}\n\n")
        
        for i, lambda1 in enumerate(lambda1_values):
            print(f"\nProgress: {i+1}/{len(lambda1_values)} - Processing lambda1 = {lambda1:.6f}")
            log.write(f"Processing lambda1 = {lambda1:.6f}\n")
            
            # Create output filename for this lambda1
            output_file = os.path.join(batch_dir, f"phase_diagram_ml{ml:.1f}_lambda1{lambda1:.6f}.csv")
            
            # Construct command to run map_phase_diagram.py
            cmd = [
                sys.executable, "map_phase_diagram.py",
                str(lambda1), str(ml),
                "--mu-min", str(mu_min),
                "--mu-max", str(mu_max),
                "--mu-points", str(mu_points),
                "--tmin", str(tmin),
                "--tmax", str(tmax),
                "--numtemp", str(numtemp),
                "--minsigma", str(minsigma),
                "--maxsigma", str(maxsigma),
                "--a0", str(a0),
                "--output", output_file,
                "--no-display"
            ]
            
            # Add no-plot flag if we don't want individual plots
            if not keep_individual_plots:
                cmd.append("--no-plot")
            
            start_time = time.time()
            
            try:
                # Run the command
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
                
                elapsed_time = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"  ✓ SUCCESS - Completed in {elapsed_time:.1f} seconds")
                    log.write(f"  SUCCESS - Completed in {elapsed_time:.1f} seconds\n")
                    successful_runs.append((lambda1, output_file, elapsed_time))
                    
                    # Move plot to batch directory if it exists and we're keeping plots
                    if keep_individual_plots:
                        plot_file = f"CP_data/phase_diagram_ml{ml:.1f}_lambda1{lambda1:.6f}.png"
                        if os.path.exists(plot_file):
                            new_plot_file = os.path.join(batch_dir, f"phase_diagram_ml{ml:.1f}_lambda1{lambda1:.6f}.png")
                            os.rename(plot_file, new_plot_file)
                else:
                    print(f"  ✗ FAILED - Return code: {result.returncode}")
                    print(f"  Error: {result.stderr}")
                    log.write(f"  FAILED - Return code: {result.returncode}\n")
                    log.write(f"  Error: {result.stderr}\n")
                    failed_runs.append((lambda1, result.stderr))
                    
            except subprocess.TimeoutExpired:
                print(f"  ✗ TIMEOUT - Exceeded 1 hour limit")
                log.write(f"  TIMEOUT - Exceeded 1 hour limit\n")
                failed_runs.append((lambda1, "Timeout"))
                
            except Exception as e:
                print(f"  ✗ ERROR - {str(e)}")
                log.write(f"  ERROR - {str(e)}\n")
                failed_runs.append((lambda1, str(e)))
        
        # Write summary
        log.write(f"\n" + "=" * 50 + "\n")
        log.write(f"BATCH SUMMARY\n")
        log.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Successful runs: {len(successful_runs)}/{len(lambda1_values)}\n")
        log.write(f"Failed runs: {len(failed_runs)}\n")
        
        if successful_runs:
            total_time = sum(t[2] for t in successful_runs)
            avg_time = total_time / len(successful_runs)
            log.write(f"Average time per run: {avg_time:.1f} seconds\n")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"Successful runs: {len(successful_runs)}/{len(lambda1_values)}")
    print(f"Failed runs: {len(failed_runs)}")
    
    if successful_runs:
        total_time = sum(t[2] for t in successful_runs)
        print(f"Total computation time: {total_time/60:.1f} minutes")
        print(f"Average time per run: {total_time/len(successful_runs):.1f} seconds")
    
    if failed_runs:
        print(f"\nFailed lambda1 values:")
        for lambda1, error in failed_runs:
            print(f"  λ₁ = {lambda1:.6f}: {error[:100]}...")
    
    print(f"\nResults saved in: {batch_dir}")
    print(f"Log file: {log_file}")
    
    return successful_runs, failed_runs, batch_dir

def create_comparison_plots(batch_dir, ml):
    """
    Create comparison plots from all successful runs in the batch.
    
    Args:
        batch_dir: Directory containing batch results
        ml: Quark mass value
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Find all CSV files in batch directory
    csv_files = [f for f in os.listdir(batch_dir) if f.endswith('.csv') and f.startswith('phase_diagram')]
    
    if not csv_files:
        print("No phase diagram data files found for comparison plots")
        return
    
    plt.figure(figsize=(12, 8))
    
    lambda1_values = []
    
    for csv_file in sorted(csv_files):
        # Extract lambda1 from filename
        try:
            lambda1_str = csv_file.split('lambda1')[1].split('.csv')[0]
            lambda1 = float(lambda1_str)
            lambda1_values.append(lambda1)
            
            # Load data
            filepath = os.path.join(batch_dir, csv_file)
            df = pd.read_csv(filepath)
            
            # Filter valid data
            df_valid = df[df['order'].notna()]
            
            if len(df_valid) > 0:
                # Plot critical line
                plt.plot(df_valid['mu'], df_valid['Tc'], 'o-', 
                        label=f'λ₁ = {lambda1:.3f}', linewidth=1.5, markersize=4)
        
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    plt.xlabel('Chemical Potential μ (MeV)', fontsize=12)
    plt.ylabel('Critical Temperature Tc (MeV)', fontsize=12)
    plt.title(f'Phase Diagram Comparison\n$m_q = {ml}$ MeV, Various λ₁ Values', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save comparison plot
    comparison_plot = os.path.join(batch_dir, f"phase_diagram_comparison_ml{ml:.1f}.png")
    plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved: {comparison_plot}")
    plt.show()

def main():
    """Main function with argument parsing for batch runs"""
    parser = argparse.ArgumentParser(description='Run phase diagram mapping for multiple lambda1 values')
    
    # Required arguments
    parser.add_argument('ml', type=float, help='Light quark mass in MeV')
    
    # Lambda1 range options
    parser.add_argument('--lambda1-min', type=float, default=0.0, help='Minimum lambda1 value (default: 0.0)')
    parser.add_argument('--lambda1-max', type=float, default=10.0, help='Maximum lambda1 value (default: 10.0)')
    parser.add_argument('--lambda1-points', type=int, default=11, help='Number of lambda1 points (default: 11)')
    parser.add_argument('--lambda1-values', type=str, help='Custom lambda1 values as comma-separated list (e.g., "0,1,2,5,10")')
    
    # Chemical potential range
    parser.add_argument('--mu-min', type=float, default=0.0, help='Minimum chemical potential in MeV (default: 0.0)')
    parser.add_argument('--mu-max', type=float, default=400.0, help='Maximum chemical potential in MeV (default: 400.0)')
    parser.add_argument('--mu-points', type=int, default=40, help='Number of mu points to sample (default: 40)')
    
    # Temperature search parameters
    parser.add_argument('--tmin', type=float, default=50.0, help='Minimum temperature for search in MeV (default: 40.0)')
    parser.add_argument('--tmax', type=float, default=100.0, help='Maximum temperature for search in MeV (default: 100.0)')
    parser.add_argument('--numtemp', type=int, default=50, help='Number of temperature points per iteration (default: 50)')
    
    # Other parameters
    parser.add_argument('--minsigma', type=float, default=0.0, help='Minimum sigma value (default: 0.0)')
    parser.add_argument('--maxsigma', type=float, default=200.0, help='Maximum sigma value (default: 200.0)')
    parser.add_argument('--a0', type=float, default=0.0, help='Parameter a0 (default: 0.0)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='batch_results', help='Base output directory (default: batch_results)')
    parser.add_argument('--no-individual-plots', action='store_true', help='Do not keep individual plots for each lambda1')
    parser.add_argument('--no-comparison-plot', action='store_true', help='Do not create comparison plot')
    
    args = parser.parse_args()
    
    # Determine lambda1 values
    if args.lambda1_values:
        # Custom values provided
        lambda1_values = [float(x.strip()) for x in args.lambda1_values.split(',')]
    else:
        # Use range
        lambda1_values = np.linspace(args.lambda1_min, args.lambda1_max, args.lambda1_points)
    
    print(f"Lambda1 values to process: {lambda1_values}")
    
    # Run batch calculation
    successful_runs, failed_runs, batch_dir = run_phase_diagram_batch(
        lambda1_values=lambda1_values,
        ml=args.ml,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        mu_points=args.mu_points,
        tmin=args.tmin,
        tmax=args.tmax,
        numtemp=args.numtemp,
        minsigma=args.minsigma,
        maxsigma=args.maxsigma,
        a0=args.a0,
        base_output_dir=args.output_dir,
        keep_individual_plots=not args.no_individual_plots
    )
    
    # Create comparison plot if requested and we have successful runs
    if not args.no_comparison_plot and successful_runs:
        create_comparison_plots(batch_dir, args.ml)
    
    return successful_runs, failed_runs, batch_dir

if __name__ == '__main__':
    successful_runs, failed_runs, batch_dir = main()
