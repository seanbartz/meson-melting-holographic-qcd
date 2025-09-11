#!/usr/bin/env python3
"""
Physics Results Extractor and Logger

Analyzes completed calculation results and extracts key physics quantities
for logging in the task summary CSV. This script can be run after completing
a batch of calculations to automatically extract and log the results.

Usage:
    # Extract results from current directory
    python extract_physics_results.py --task-id batch_990_5
    
    # Extract from specific directory
    python extract_physics_results.py --task-id batch_990_5 --data-dir /path/to/results
    
    # Manual entry mode (for existing analysis)
    python extract_physics_results.py --manual --task-id batch_990_5
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import glob

# Import our task logger
from task_summary_logger import TaskSummaryLogger

class PhysicsResultsExtractor:
    """
    Extracts key physics results from calculation data for task summary logging.
    """
    
    def __init__(self, data_dir='.'):
        """
        Initialize the results extractor.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing calculation results
        """
        self.data_dir = Path(data_dir)
        self.logger = TaskSummaryLogger()
    
    def find_result_files(self):
        """Find relevant result files in the data directory."""
        result_files = {
            'melting_data': list(self.data_dir.glob('*melting_data*.csv')),
            'phase_diagram': list(self.data_dir.glob('phase_diagram*.csv')),
            'critical_points': list(self.data_dir.glob('critical*.csv')),
            'temperature_scan': list(self.data_dir.glob('*temperature_scan*.csv')),
            'sigma_calculations': list(self.data_dir.glob('sigma_calculations*.csv'))
        }
        
        print("Found result files:")
        for category, files in result_files.items():
            if files:
                print(f"  {category}: {len(files)} files")
                for f in files[:3]:  # Show first 3
                    print(f"    - {f.name}")
                if len(files) > 3:
                    print(f"    ... and {len(files)-3} more")
            else:
                print(f"  {category}: none found")
        
        return result_files
    
    def extract_parameters_from_filename(self, filename):
        """
        Extract physics parameters from standardized filenames.
        
        Expected format: phase_diagram_improved_mq_X_lambda1_Y_gamma_Z_lambda4_W.csv
        """
        import re
        params = {}
        
        filename_str = str(filename)
        
        # Enhanced parameter patterns for phase diagram files
        patterns = {
            'mq': r'mq[_-]([0-9]+\.?[0-9]*)',
            'lambda1': r'lambda1[_-]([0-9]+\.?[0-9]*)',
            'gamma': r'gamma[_-]([-]?[0-9]+\.?[0-9]*)',
            'lambda4': r'lambda4[_-]([0-9]+\.?[0-9]*)',
            # Alternative patterns
            'ml': r'ml[_-]([0-9]+\.?[0-9]*)'  # Convert ml to mq
        }
        
        for param, pattern in patterns.items():
            match = re.search(pattern, filename_str)
            if match:
                value = float(match.group(1))
                if param == 'ml':
                    params['mq'] = value  # Convert ml to mq
                else:
                    params[param] = value
        
        print(f"  Extracted from filename '{filename.name}': {params}")
        return params
    
    def analyze_melting_data(self, melting_files):
        """
        Analyze melting temperature data to extract key results.
        
        Returns:
        --------
        dict: Extracted melting temperature results
        """
        results = {}
        
        if not melting_files:
            return results
        
        # Use the most recent melting data file
        melting_file = max(melting_files, key=lambda x: x.stat().st_mtime)
        print(f"Analyzing melting data: {melting_file.name}")
        
        try:
            df = pd.read_csv(melting_file)
            
            # Extract axial melting temperature at mu=0 (or closest to 0)
            if 'mu' in df.columns and 'T' in df.columns:
                # Find row with mu closest to 0
                min_mu_idx = df['mu'].abs().idxmin()
                closest_mu = df.loc[min_mu_idx, 'mu']
                
                if abs(closest_mu) < 5.0:  # Within 5 MeV of mu=0
                    results['axial_melting_T_mu0'] = df.loc[min_mu_idx, 'T']
                    results['axial_melting_mu_mu0'] = closest_mu
                    print(f"  Axial melting at μ≈0: T={results['axial_melting_T_mu0']:.1f} MeV")
            
            # Determine temperature and mu ranges
            if 'T' in df.columns:
                results['T_min'] = df['T'].min()
                results['T_max'] = df['T'].max()
            
            if 'mu' in df.columns:
                results['mu_min'] = df['mu'].min()
                results['mu_max'] = df['mu'].max()
                results['num_mu_values'] = df['mu'].nunique()
                print(f"  Temperature range: {results.get('T_min', 'unknown'):.1f} - {results.get('T_max', 'unknown'):.1f} MeV")
                print(f"  Chemical potential: {results['num_mu_values']} values from {results['mu_min']:.1f} to {results['mu_max']:.1f} MeV")
        
        except Exception as e:
            print(f"  Error analyzing melting data: {e}")
        
        return results
    
    def analyze_phase_diagram_data(self, phase_files):
        """
        Analyze phase diagram data for chiral transitions and critical points.
        
        Returns:
        --------
        dict: Extracted phase transition results  
        """
        results = {}
        
        if not phase_files:
            return results
        
        # Use the most recent phase diagram file
        phase_file = max(phase_files, key=lambda x: x.stat().st_mtime)
        print(f"Analyzing phase diagram: {phase_file.name}")
        
        try:
            df = pd.read_csv(phase_file)
            
            # Look for chiral critical temperature at mu=0
            if 'mu' in df.columns and 'T' in df.columns:
                # Find transitions at mu close to 0
                low_mu_data = df[df['mu'].abs() < 5.0]
                
                if len(low_mu_data) > 0:
                    # Look for phase transition indicators
                    if 'phase' in df.columns or 'chiral_phase' in df.columns:
                        phase_col = 'phase' if 'phase' in df.columns else 'chiral_phase'
                        
                        # Find transition temperature (where phase changes)
                        mu_zero_data = low_mu_data.iloc[0]
                        results['chiral_critical_T_mu0'] = mu_zero_data.get('T', None)
                        results['chiral_critical_mu_mu0'] = mu_zero_data.get('mu', 0.0)
                    
                    # Check for critical point indicators
                    if any(col in df.columns for col in ['critical_point', 'cp_T', 'cp_mu']):
                        cp_data = df.dropna(subset=[col for col in ['cp_T', 'cp_mu'] if col in df.columns])
                        
                        if len(cp_data) > 0:
                            results['has_critical_point'] = True
                            if 'cp_T' in df.columns:
                                results['critical_point_T'] = cp_data['cp_T'].iloc[0]
                            if 'cp_mu' in df.columns:
                                results['critical_point_mu'] = cp_data['cp_mu'].iloc[0]
                            print(f"  Critical point found: T≈{results.get('critical_point_T', 'unknown'):.1f}, μ≈{results.get('critical_point_mu', 'unknown'):.1f} MeV")
                        else:
                            results['has_critical_point'] = False
                    
                    # Estimate transition order (this would need more sophisticated analysis)
                    results['chiral_transition_order'] = 2  # Assume crossover unless proven otherwise
        
        except Exception as e:
            print(f"  Error analyzing phase diagram: {e}")
        
        return results
    
    def estimate_parameter_ranges(self, all_files, task_id=None):
        """Estimate physics parameters from filenames and available data, with task context."""
        params = {}
        
        # Try to extract from filenames
        for file_list in all_files.values():
            for file_path in file_list:
                extracted = self.extract_parameters_from_filename(file_path)
                params.update(extracted)
        
        # Use task context if available (extract from task_id)
        if task_id:
            # Check if we can get parameters from existing task summary
            try:
                df = self.logger.get_summary_dataframe()
                if not df.empty and task_id in df['task_id'].values:
                    task_row = df[df['task_id'] == task_id].iloc[0]
                    print(f"  Found existing task data for {task_id}")
                    for param in ['mq', 'lambda1', 'gamma', 'lambda4']:
                        if pd.notna(task_row[param]) and task_row[param] != '':
                            params[param] = float(task_row[param])
                            print(f"    Using {param} = {params[param]} from task summary")
            except Exception as e:
                print(f"  Could not load existing task data: {e}")
        
        # Set defaults for common parameters if not found
        if 'gamma' not in params:
            params['gamma'] = -22.4  # Standard default
            print(f"  Using default gamma = {params['gamma']}")
        
        if 'lambda4' not in params:
            params['lambda4'] = 4.2  # Standard default  
            print(f"  Using default lambda4 = {params['lambda4']}")
        
        return params
    
    def auto_extract_results(self, task_id, calculation_date=None):
        """
        Automatically extract physics results from available data files.
        
        Parameters:
        -----------
        task_id : str
            Unique identifier for this calculation task
        calculation_date : str, optional
            Date of calculation
        """
        print(f"Auto-extracting results for task: {task_id}")
        print(f"Data directory: {self.data_dir}")
        
        # Find all relevant result files
        result_files = self.find_result_files()
        
        # Extract results from each data type
        results = {}
        
        # Melting temperature analysis
        melting_results = self.analyze_melting_data(result_files['melting_data'])
        results.update(melting_results)
        
        # Phase diagram analysis
        phase_results = self.analyze_phase_diagram_data(result_files['phase_diagram'])
        results.update(phase_results)
        
        # Estimate physics parameters
        param_estimates = self.estimate_parameter_ranges(result_files, task_id)
        results.update(param_estimates)
        
        # Set defaults for missing values
        defaults = {
            'axial_vector_merge': False,
            'axial_chiral_cross': False,
            'has_critical_point': False,
            'chiral_transition_order': 2,
            'convergence_issues': 0
        }
        
        for key, default_val in defaults.items():
            if key not in results:
                results[key] = default_val
        
        # Count total calculations if sigma data available
        if result_files['sigma_calculations']:
            try:
                sigma_file = result_files['sigma_calculations'][0]
                df = pd.read_csv(sigma_file)
                results['total_calculations'] = len(df)
            except:
                pass
        
        # Log the extracted results
        if results:
            print(f"\nExtracted results:")
            for key, value in results.items():
                if value is not None:
                    print(f"  {key}: {value}")
            
            success = self.logger.log_task_summary(
                task_id=task_id,
                calculation_date=calculation_date,
                **results
            )
            
            if success:
                print(f"\n✓ Successfully logged results for task {task_id}")
                return True
            else:
                print(f"\n✗ Failed to log results for task {task_id}")
                return False
        else:
            print("No results could be extracted from available files")
            return False
    
    def manual_entry_mode(self, task_id):
        """
        Interactive mode for manually entering physics results.
        """
        print(f"Manual entry mode for task: {task_id}")
        print("Enter the physics results (press Enter to skip optional fields):")
        
        results = {}
        
        # Required parameters
        required = ['mq', 'lambda1', 'gamma', 'lambda4']
        for param in required:
            while True:
                try:
                    value = input(f"{param}: ")
                    if value.strip():
                        results[param] = float(value)
                        break
                    else:
                        print(f"Error: {param} is required")
                except ValueError:
                    print("Error: Please enter a numeric value")
        
        # Temperature and mu ranges
        range_params = [
            ('T_min', 'Minimum temperature (MeV)'),
            ('T_max', 'Maximum temperature (MeV)'),
            ('mu_min', 'Minimum chemical potential (MeV)'),
            ('mu_max', 'Maximum chemical potential (MeV)'),
            ('num_mu_values', 'Number of mu values calculated')
        ]
        
        for param, description in range_params:
            value = input(f"{description}: ")
            if value.strip():
                results[param] = float(value) if param != 'num_mu_values' else int(value)
        
        # Physics results
        physics_params = [
            ('axial_melting_T_mu0', 'Axial melting temperature at μ=0 (MeV)'),
            ('chiral_critical_T_mu0', 'Chiral critical temperature at μ=0 (MeV)'),
            ('chiral_transition_order', 'Chiral transition order (1=first order, 2=crossover)'),
        ]
        
        for param, description in physics_params:
            value = input(f"{description}: ")
            if value.strip():
                if param == 'chiral_transition_order':
                    results[param] = int(value)
                else:
                    results[param] = float(value)
        
        # Critical point
        has_cp = input("Critical point present? (y/n): ").lower().startswith('y')
        results['has_critical_point'] = has_cp
        
        if has_cp:
            cp_T = input("Critical point temperature (MeV): ")
            cp_mu = input("Critical point chemical potential (MeV): ")
            if cp_T.strip():
                results['critical_point_T'] = float(cp_T)
            if cp_mu.strip():
                results['critical_point_mu'] = float(cp_mu)
        
        # Additional analysis
        merge = input("Axial and vector melting temperatures merge? (y/n): ").lower().startswith('y')
        results['axial_vector_merge'] = merge
        
        cross = input("Axial melting and chiral transition lines cross? (y/n): ").lower().startswith('y')
        results['axial_chiral_cross'] = cross
        
        # Notes
        notes = input("Additional notes: ")
        if notes.strip():
            results['notes'] = notes
        
        # Log the results
        success = self.logger.log_task_summary(task_id=task_id, **results)
        
        if success:
            print(f"\n✓ Successfully logged manual entry for task {task_id}")
        else:
            print(f"\n✗ Failed to log manual entry for task {task_id}")
        
        return success

def main():
    parser = argparse.ArgumentParser(
        description="Extract and log physics results from calculation data"
    )
    parser.add_argument(
        '--task-id',
        required=True,
        help='Unique identifier for this calculation task'
    )
    parser.add_argument(
        '--data-dir',
        default='.',
        help='Directory containing calculation results (default: current directory)'
    )
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Use interactive manual entry mode'
    )
    parser.add_argument(
        '--date',
        help='Calculation date (YYYY-MM-DD, default: today)'
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = PhysicsResultsExtractor(args.data_dir)
    
    # Run extraction
    if args.manual:
        success = extractor.manual_entry_mode(args.task_id)
    else:
        success = extractor.auto_extract_results(args.task_id, args.date)
    
    if not success:
        sys.exit(1)
    
    # Generate summary report
    extractor.logger.generate_physics_report()
    print("\nPhysics summary report generated!")

if __name__ == "__main__":
    main()
