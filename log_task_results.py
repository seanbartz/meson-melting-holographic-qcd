#!/usr/bin/env python3
"""
Simple Task Summary Integration

A lightweight script to add task summary logging to existing calculation workflows.
Can be called at the end of batch jobs to automatically log results.

Usage examples:

# At the end of a batch job script:
python log_task_results.py batch_990_5 --mq 9.0 --lambda1 7.438

# With specific results:
python log_task_results.py my_task --mq 9.0 --lambda1 7.438 \\
    --axial-melting-T 85.3 --chiral-critical-T 102.1 --has-critical-point

# Quick log with minimal info:
python log_task_results.py quick_test --mq 15.0 --lambda1 7.0 --notes "test run"
"""

import sys
import os
import argparse
import fcntl
import time
from datetime import datetime

def simple_log_task_summary(task_id, **params):
    """
    Simple function to log task summary without heavy dependencies.
    Creates a CSV file with basic task information.
    Uses file locking to prevent race conditions in concurrent environments.
    """
    
    # Define the summary CSV file - use QUENCH project directory on Obsidian
    if os.path.exists('/net/project/QUENCH'):
        # On Obsidian cluster
        summary_dir = '/net/project/QUENCH/summary_data'
        print(f"Using Obsidian cluster directory: {summary_dir}")
    else:
        # Local development
        summary_dir = 'summary_data'
        print(f"Using local directory: {summary_dir}")
    
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(summary_dir, 'task_summary.csv')
    
    # Define CSV headers (same as in TaskSummaryLogger)
    headers = [
        'task_id', 'calculation_date', 'timestamp',
        'mq', 'lambda1', 'gamma', 'lambda4',
        'T_min', 'T_max', 'mu_min', 'mu_max', 'num_mu_values',
        'axial_melting_T_mu0', 'axial_melting_mu_mu0',
        'chiral_critical_T_mu0', 'chiral_critical_mu_mu0', 'chiral_transition_order',
        'has_critical_point', 'critical_point_T', 'critical_point_mu',
        'axial_vector_merge', 'merge_T', 'merge_mu',
        'axial_chiral_cross', 'cross_T', 'cross_mu',
        'total_calculations', 'convergence_issues', 'notes'
    ]
    
    # Prepare the data row
    row_data = {
        'task_id': task_id,
        'calculation_date': params.get('calculation_date', datetime.now().strftime("%Y-%m-%d")),
        'timestamp': datetime.now().isoformat()
    }
    
    # Add all provided parameters
    for header in headers[3:]:  # Skip metadata columns
        row_data[header] = params.get(header, '')
    
    # Write the row with file locking to prevent race conditions
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            # Check if we need to write headers (for new file or empty file)
            write_headers = not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0
            
            with open(summary_file, 'a') as f:
                # Acquire exclusive lock
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except IOError:
                    if attempt < max_retries - 1:
                        print(f"  File locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise IOError("Could not acquire file lock after maximum retries")
                
                try:
                    # Check again if we need headers after acquiring lock
                    if write_headers:
                        f.write(','.join(headers) + '\n')
                        print(f"Created new task summary file: {summary_file}")
                    
                    # Write the data row
                    row_values = [str(row_data.get(header, '')) for header in headers]
                    f.write(','.join(row_values) + '\n')
                    f.flush()  # Ensure data is written
                    
                    print(f"✓ Logged task summary: {task_id}")
                    print(f"   Summary file: {summary_file}")
                    print(f"   Attempt: {attempt + 1}/{max_retries}")
                    return True
                    
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error on attempt {attempt + 1}: {e}")
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"Error logging task summary after {max_retries} attempts: {e}")
                return False
    
    print(f"Failed to log task summary after {max_retries} attempts")
    return False

def count_csv_lines(filename):
    """Count lines in a CSV file, handling large files efficiently."""
    if not os.path.exists(filename):
        return 0
    
    try:
        with open(filename, 'r') as f:
            return sum(1 for line in f) - 1  # Subtract header
    except:
        return 0

def auto_detect_results():
    """
    Automatically detect calculation results from common file patterns.
    Returns a dictionary of detected results.
    """
    results = {}
    
    # Look for sigma calculations file
    sigma_files = [f for f in os.listdir('.') if 'sigma_calculations' in f and f.endswith('.csv')]
    if sigma_files:
        sigma_count = count_csv_lines(sigma_files[0])
        if sigma_count > 0:
            results['total_calculations'] = sigma_count
            print(f"  Detected {sigma_count} sigma calculations")
    
    # Look for melting data files
    melting_files = [f for f in os.listdir('.') if 'melting_data' in f and f.endswith('.csv')]
    if melting_files:
        print(f"  Found {len(melting_files)} melting data files")
    
    # Look for phase diagram files  
    phase_files = [f for f in os.listdir('.') if 'phase_diagram' in f and f.endswith('.csv')]
    if phase_files:
        print(f"  Found {len(phase_files)} phase diagram files")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Log task summary for physics calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'task_id',
        help='Unique identifier for this calculation task'
    )
    
    # Input parameters
    parser.add_argument('--mq', type=float, help='Quark mass parameter')
    parser.add_argument('--lambda1', type=float, help='Lambda1 parameter')
    parser.add_argument('--gamma', type=float, help='Gamma parameter')
    parser.add_argument('--lambda4', type=float, help='Lambda4 parameter')
    
    # Temperature and mu ranges
    parser.add_argument('--T-min', type=float, dest='T_min', help='Minimum temperature (MeV)')
    parser.add_argument('--T-max', type=float, dest='T_max', help='Maximum temperature (MeV)')
    parser.add_argument('--mu-min', type=float, dest='mu_min', help='Minimum chemical potential (MeV)')
    parser.add_argument('--mu-max', type=float, dest='mu_max', help='Maximum chemical potential (MeV)')
    parser.add_argument('--num-mu-values', type=int, dest='num_mu_values', help='Number of mu values')
    
    # Physics results
    parser.add_argument('--axial-melting-T', type=float, dest='axial_melting_T_mu0', 
                       help='Axial melting temperature at mu=0 (MeV)')
    parser.add_argument('--chiral-critical-T', type=float, dest='chiral_critical_T_mu0',
                       help='Chiral critical temperature at mu=0 (MeV)')
    parser.add_argument('--transition-order', type=int, dest='chiral_transition_order',
                       choices=[1, 2], help='Chiral transition order (1=first order, 2=crossover)')
    
    # Critical point
    parser.add_argument('--has-critical-point', action='store_true', dest='has_critical_point',
                       help='Critical point is present')
    parser.add_argument('--cp-T', type=float, dest='critical_point_T', 
                       help='Critical point temperature (MeV)')
    parser.add_argument('--cp-mu', type=float, dest='critical_point_mu',
                       help='Critical point chemical potential (MeV)')
    
    # Analysis results
    parser.add_argument('--axial-vector-merge', action='store_true', dest='axial_vector_merge',
                       help='Axial and vector melting temperatures merge')
    parser.add_argument('--merge-T', type=float, dest='merge_T', help='Merge temperature (MeV)')
    parser.add_argument('--merge-mu', type=float, dest='merge_mu', help='Merge chemical potential (MeV)')
    
    parser.add_argument('--axial-chiral-cross', action='store_true', dest='axial_chiral_cross',
                       help='Axial and chiral lines cross')
    parser.add_argument('--cross-T', type=float, dest='cross_T', help='Crossing temperature (MeV)')
    parser.add_argument('--cross-mu', type=float, dest='cross_mu', help='Crossing chemical potential (MeV)')
    
    # Additional info
    parser.add_argument('--total-calculations', type=int, dest='total_calculations',
                       help='Total number of calculations performed')
    parser.add_argument('--convergence-issues', type=int, dest='convergence_issues', default=0,
                       help='Number of convergence issues encountered')
    parser.add_argument('--notes', help='Additional notes about the calculation')
    
    # Options
    parser.add_argument('--auto-detect', action='store_true',
                       help='Try to automatically detect results from files in current directory')
    parser.add_argument('--date', help='Calculation date (YYYY-MM-DD, default: today)')
    
    args = parser.parse_args()
    
    # Convert args to dictionary
    params = {k: v for k, v in vars(args).items() if v is not None and k not in ['task_id', 'auto_detect']}
    
    # Auto-detect results if requested
    if args.auto_detect:
        print("Auto-detecting calculation results...")
        detected = auto_detect_results()
        
        # Only add detected values if not explicitly provided
        for key, value in detected.items():
            if key not in params:
                params[key] = value
    
    # Check for required parameters
    if not any(param in params for param in ['mq', 'lambda1']):
        print("Warning: No physics parameters provided. At minimum, specify --mq and --lambda1")
        print("Use --help to see all available options")
    
    # Log the task summary
    success = simple_log_task_summary(args.task_id, **params)
    
    if success:
        print("\\nTask summary logged successfully!")
        
        # Show what was logged
        if params:
            print("\\nLogged parameters:")
            for key, value in params.items():
                if value not in ['', None]:
                    print(f"  {key}: {value}")
    else:
        print("Failed to log task summary")
        sys.exit(1)

if __name__ == "__main__":
    main()
