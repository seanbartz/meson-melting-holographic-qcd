#!/usr/bin/env python3
"""
Run Axial Melting Temperature Scan for Multiple Parameters

This script runs axial_melting_scan.py for a range of mq (quark mass) and lambda1 values.
It allows the user to specify the chemical potential range and starting temperature,
which will be used for all parameter combinations.

Usage:
    python run_axial_melting_scan.py --mq-values 9.0 15.0 --lambda1-values 4.5 5.0 6.0 --mu-min 0.1 --mu-max 200 --mu-points 21 --T-start 200
"""

import subprocess
import argparse
import sys
import os
import itertools
import time

def run_single_scan(mq, lambda1, mu_min, mu_max, mu_points, T_start, additional_args=None):
    """
    Run axial_melting_scan.py for a single combination of mq and lambda1.
    
    Parameters:
    -----------
    mq : float
        Quark mass value
    lambda1 : float
        Lambda1 parameter value
    mu_min : float
        Minimum chemical potential
    mu_max : float
        Maximum chemical potential
    mu_points : int
        Number of chemical potential points
    T_start : float
        Starting temperature
    additional_args : list
        Additional command line arguments
        
    Returns:
    --------
    bool : True if successful, False if failed
    """
    
    # Build the command
    cmd = [
        'python', 'axial_melting_scan.py',
        '-mq', str(mq),
        '-lambda1', str(lambda1),
        '-mumin', str(mu_min),
        '-mumax', str(mu_max),
        '-mupoints', str(mu_points),
        '-tstart', str(T_start),
        '--no-display'  # Don't display plots
    ]
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"Running: mq={mq}, λ₁={lambda1}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: mq={mq}, λ₁={lambda1}")
            print("STDOUT:")
            print(result.stdout)
            return True
        else:
            print(f"✗ FAILED: mq={mq}, λ₁={lambda1}")
            print("STDERR:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: mq={mq}, λ₁={lambda1} (exceeded 1 hour)")
        return False
    except Exception as e:
        print(f"✗ ERROR: mq={mq}, λ₁={lambda1} - {str(e)}")
        return False

def run_parameter_scan(mq_values, lambda1_values, mu_min, mu_max, mu_points, T_start, 
                      additional_args=None, continue_on_error=True):
    """
    Run axial melting scans for all combinations of mq and lambda1 values.
    
    Parameters:
    -----------
    mq_values : list
        List of quark mass values
    lambda1_values : list
        List of lambda1 parameter values
    mu_min : float
        Minimum chemical potential
    mu_max : float
        Maximum chemical potential
    mu_points : int
        Number of chemical potential points
    T_start : float
        Starting temperature
    additional_args : list
        Additional command line arguments
    continue_on_error : bool
        Whether to continue if one scan fails
        
    Returns:
    --------
    tuple : (successful_runs, failed_runs, total_runs)
    """
    
    # Generate all combinations
    parameter_combinations = list(itertools.product(mq_values, lambda1_values))
    total_runs = len(parameter_combinations)
    
    print("=" * 80)
    print("AXIAL MELTING TEMPERATURE PARAMETER SCAN")
    print("=" * 80)
    print(f"mq values: {mq_values}")
    print(f"λ₁ values: {lambda1_values}")
    print(f"μ range: {mu_min} - {mu_max} MeV ({mu_points} points)")
    print(f"Starting temperature: {T_start} MeV")
    print(f"Total parameter combinations: {total_runs}")
    print("=" * 80)
    
    successful_runs = []
    failed_runs = []
    
    start_time = time.time()
    
    for i, (mq, lambda1) in enumerate(parameter_combinations):
        print(f"\n[{i+1}/{total_runs}] Processing mq={mq}, λ₁={lambda1}")
        print("-" * 60)
        
        success = run_single_scan(mq, lambda1, mu_min, mu_max, mu_points, T_start, additional_args)
        
        if success:
            successful_runs.append((mq, lambda1))
        else:
            failed_runs.append((mq, lambda1))
            if not continue_on_error:
                print(f"\nStopping due to failure at mq={mq}, λ₁={lambda1}")
                break
        
        # Print progress
        elapsed_time = time.time() - start_time
        avg_time_per_run = elapsed_time / (i + 1)
        estimated_remaining = avg_time_per_run * (total_runs - i - 1)
        
        print(f"Progress: {i+1}/{total_runs} ({100*(i+1)/total_runs:.1f}%)")
        print(f"Elapsed time: {elapsed_time/60:.1f} min")
        print(f"Estimated remaining: {estimated_remaining/60:.1f} min")
    
    return successful_runs, failed_runs, total_runs

def print_summary(successful_runs, failed_runs, total_runs, start_time):
    """
    Print a summary of the parameter scan results.
    """
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("PARAMETER SCAN SUMMARY")
    print("=" * 80)
    print(f"Total runs: {total_runs}")
    print(f"Successful: {len(successful_runs)}")
    print(f"Failed: {len(failed_runs)}")
    print(f"Success rate: {100*len(successful_runs)/total_runs:.1f}%")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    
    if successful_runs:
        print("\nSuccessful runs:")
        for mq, lambda1 in successful_runs:
            print(f"  ✓ mq={mq}, λ₁={lambda1}")
    
    if failed_runs:
        print("\nFailed runs:")
        for mq, lambda1 in failed_runs:
            print(f"  ✗ mq={mq}, λ₁={lambda1}")
    
    print("\nGenerated files:")
    print("  - axial_melting_data_mq{mq}_lambda{lambda1}.csv (data)")
    print("  - axial_melting_curve_mq{mq}_lambda{lambda1}.png (plots)")
    
    print("=" * 80)

def parse_float_list(values):
    """
    Convert a list of strings to a list of floats.
    """
    try:
        return [float(x) for x in values]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid float value: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run axial melting scans for multiple mq and lambda1 values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan for mq=9,15 and lambda1=4.5,5.0,6.0
  python run_axial_melting_scan.py --mq-values 9.0 15.0 --lambda1-values 4.5 5.0 6.0
  
  # Custom μ range and starting temperature
  python run_axial_melting_scan.py --mq-values 9.0 --lambda1-values 6.0 7.0 --mu-min 0.1 --mu-max 500 --mu-points 50 --T-start 150
  
  # Stop on first error
  python run_axial_melting_scan.py --mq-values 9.0 15.0 --lambda1-values 4.5 5.0 --stop-on-error
        """
    )
    
    # Parameter ranges
    parser.add_argument('--mq-values', type=float, nargs='+', required=True,
                       help='Space-separated list of quark mass values (MeV)')
    parser.add_argument('--lambda1-values', type=float, nargs='+', required=True,
                       help='Space-separated list of lambda1 parameter values')
    
    # Chemical potential settings
    parser.add_argument('-mumin', type=float, default=0.1,
                       help='Minimum chemical potential (MeV) [default: 0.1]')
    parser.add_argument('-mumax', type=float, default=200.0,
                       help='Maximum chemical potential (MeV) [default: 200.0]')
    parser.add_argument('-mupoints', type=int, default=21,
                       help='Number of chemical potential points [default: 21]')
    
    # Temperature settings
    parser.add_argument('-tstart', type=float, default=200.0,
                       help='Starting temperature for scan (MeV) [default: 200.0]')
    
    # Execution options
    parser.add_argument('--stop-on-error', action='store_true',
                       help='Stop execution if any scan fails [default: continue]')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.mq_values:
        print("Error: At least one mq value must be specified")
        sys.exit(1)
    
    if not args.lambda1_values:
        print("Error: At least one lambda1 value must be specified")
        sys.exit(1)
    
    if args.mumin <= 0:
        print("Error: mu-min must be positive (avoid mu=0)")
        sys.exit(1)
    
    if args.mumax <= args.mumin:
        print("Error: mu-max must be greater than mu-min")
        sys.exit(1)
    
    if args.mupoints < 2:
        print("Error: mu-points must be at least 2")
        sys.exit(1)
    
    # Check if axial_melting_scan.py exists
    if not os.path.exists('axial_melting_scan.py'):
        print("Error: axial_melting_scan.py not found in current directory")
        sys.exit(1)
    
    # Dry run mode
    if args.dry_run:
        parameter_combinations = list(itertools.product(args.mq_values, args.lambda1_values))
        print("DRY RUN - Commands that would be executed:")
        print("=" * 60)
        for mq, lambda1 in parameter_combinations:
            cmd = [
                'python', 'axial_melting_scan.py',
                '-mq', str(mq),
                '-lambda1', str(lambda1),
                '-mumin', str(args.mumin),
                '-mumax', str(args.mumax),
                '-mupoints', str(args.mupoints),
                '-tstart', str(args.tstart),
                '--no-display'
            ]
            print(' '.join(cmd))
        print("=" * 60)
        print(f"Total commands: {len(parameter_combinations)}")
        sys.exit(0)
    
    # Run the parameter scan
    start_time = time.time()
    
    successful_runs, failed_runs, total_runs = run_parameter_scan(
        args.mq_values, args.lambda1_values,
        args.mumin, args.mumax, args.mupoints, args.tstart,
        continue_on_error=not args.stop_on_error
    )
    
    # Print summary
    print_summary(successful_runs, failed_runs, total_runs, start_time)
    
    # Exit with appropriate code
    if failed_runs and args.stop_on_error:
        sys.exit(1)
    elif failed_runs:
        print(f"\nWarning: {len(failed_runs)} scan(s) failed, but continuing as requested")
        sys.exit(0)
    else:
        print("\nAll scans completed successfully!")
        sys.exit(0)
