#!/usr/bin/env python3
"""
Physics Results Extractor and Logger

Analyzes completed calculation results and extracts key physics quantities
for logging in the task summary CSV. This script can be run automatically
after completing a batch of calculations to automatically extract and log the results.

Usage:
    # Extract results from current directory
    python extract_physics_results.py --task-id batch_990_5
    
    # Extract results from specific directory
    python extract_physics_results.py --task-id batch_990_5 --data-dir /path/to/results
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_summary_logger import TaskSummaryLogger

class PhysicsResultsExtractor:
    """Extract physics results from calculation data files."""
    
    def __init__(self, data_dir=None):
        """
        Initialize extractor.
        
        Parameters:
        -----------
        data_dir : str or Path, optional
            Directory containing calculation results. Defaults to current directory.
        """
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
        self.logger = TaskSummaryLogger()
    
    def find_result_files(self):
        """Find all relevant result files in the data directory."""
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
        
        Handles patterns like:
        - axial_melting_data_mq15.0_lambda7.0.csv
        - phase_diagram_ml9.0_lambda15.000_20250716_163945.csv
        """
        params = {}
        
        filename_str = str(filename.name) if hasattr(filename, 'name') else str(filename)
        print(f"  Extracting from filename '{filename_str}': ", end='')
        
        # Pattern matches for different parameter formats
        patterns = {
            'mq': [r'mq([0-9]+\.?[0-9]*)', r'm_q([0-9]+\.?[0-9]*)'],
            'ml': [r'ml([0-9]+\.?[0-9]*)', r'm_l([0-9]+\.?[0-9]*)'], 
            'lambda1': [r'lambda([0-9]+\.?[0-9]*)', r'lambda1([0-9]+\.?[0-9]*)'],
            'gamma': [r'gamma([+-]?[0-9]+\.?[0-9]*)', r'g([+-]?[0-9]+\.?[0-9]*)'],
            'lambda4': [r'lambda4([0-9]+\.?[0-9]*)', r'l4([0-9]+\.?[0-9]*)']
        }
        
        for param_name, param_patterns in patterns.items():
            for pattern in param_patterns:
                match = re.search(pattern, filename_str, re.IGNORECASE)
                if match:
                    try:
                        param_value = float(match.group(1))
                        params[param_name] = param_value
                        print(f"{param_name}={param_value} ", end='')
                        break  # Use first match
                    except ValueError:
                        continue
        
        # Special case: if we found 'ml' but not 'mq', assume they're the same for some files
        if 'ml' in params and 'mq' not in params:
            params['mq'] = params['ml']
            print(f"mq={params['mq']}(from ml) ", end='')
        
        print()  # End the line
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
            # Read CSV with comment handling
            df = pd.read_csv(melting_file, comment='#')
            print(f"  Melting data shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Handle cases where columns might not have proper headers
            if df.shape[1] == 2 and (df.columns[0].startswith('0.') or df.columns[0].startswith('#')):
                # Data starts from first row, assign proper column names
                df.columns = ['mu', 'T_melting']
                print(f"  → Assigned column names: {list(df.columns)}")
            elif df.shape[1] >= 2:
                # Try to identify columns by pattern matching
                cols = list(df.columns)
                mu_col = None
                T_col = None
                
                for col in cols:
                    col_lower = str(col).lower()
                    if any(m in col_lower for m in ['mu', 'chemical', 'chem_pot']):
                        mu_col = col
                    elif any(t in col_lower for t in ['temp', 't_', 'melting', '_t']):
                        T_col = col
                
                if mu_col and T_col:
                    # Rename for consistency
                    df = df.rename(columns={mu_col: 'mu', T_col: 'T_melting'})
                    print(f"  → Mapped columns: μ='{mu_col}' → 'mu', T='{T_col}' → 'T_melting'")
                else:
                    print(f"  → Could not identify μ and T columns in: {cols}")
                    return results
            
            # Now work with standardized column names 'mu' and 'T_melting'
            if 'mu' in df.columns and 'T_melting' in df.columns:
                print(f"  Working with standardized columns: μ='mu', T='T_melting'")
                
                # Find melting temperature at smallest |μ|
                df_clean = df.dropna(subset=['mu', 'T_melting'])
                if len(df_clean) > 0:
                    # Convert to numeric if needed
                    df_clean['mu'] = pd.to_numeric(df_clean['mu'], errors='coerce')
                    df_clean['T_melting'] = pd.to_numeric(df_clean['T_melting'], errors='coerce')
                    df_clean = df_clean.dropna(subset=['mu', 'T_melting'])
                    
                    if len(df_clean) > 0:
                        min_mu_idx = df_clean['mu'].abs().idxmin()
                        closest_mu = df_clean.loc[min_mu_idx, 'mu']
                        axial_T_melting = df_clean.loc[min_mu_idx, 'T_melting']
                        
                        results['axial_melting_T_mu0'] = float(axial_T_melting)
                        results['axial_melting_mu_mu0'] = float(closest_mu)
                        
                        print(f"  → Axial melting: T={axial_T_melting:.2f} MeV at μ={closest_mu:.2f} MeV")
                        
                        # Additional statistics
                        results['axial_melting_T_range'] = [float(df_clean['T_melting'].min()), float(df_clean['T_melting'].max())]
                        results['axial_melting_mu_range'] = [float(df_clean['mu'].min()), float(df_clean['mu'].max())]
                        
                        print(f"  → T range: [{results['axial_melting_T_range'][0]:.1f}, {results['axial_melting_T_range'][1]:.1f}] MeV")
                        print(f"  → μ range: [{results['axial_melting_mu_range'][0]:.1f}, {results['axial_melting_mu_range'][1]:.1f}] MeV")
                        
                        # Check for unusual behavior (sharp transitions, etc.)
                        T_std = df_clean['T_melting'].std()
                        if T_std < 1.0:  # Very stable melting temperature
                            results['axial_melting_stability'] = 'high'
                        elif T_std > 10.0:  # Large variation
                            results['axial_melting_stability'] = 'low'
                        else:
                            results['axial_melting_stability'] = 'moderate'
                        
                        print(f"  → Melting stability: {results['axial_melting_stability']} (σ_T = {T_std:.2f})")
                        
                    else:
                        print("  → No valid numeric T,μ data found after cleaning")
                else:
                    print("  → No valid T,μ data found in melting file")
            else:
                print(f"  → Standardized columns not found. Available: {list(df.columns)}")
                
        except Exception as e:
            print(f"  → Error analyzing melting data: {e}")
        
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
            print(f"  Phase diagram shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Look for temperature and mu columns
            temp_cols = [col for col in df.columns if any(t in col.lower() for t in ['tc', 'temp', 't_crit', 'critical_t']) and 'num' not in col.lower()]
            mu_cols = [col for col in df.columns if any(m in col.lower() for m in ['mu', 'chemical', 'chem_pot']) and 'crit' not in col.lower()]
            
            T_col = None
            mu_col = None
            
            # Look for specific phase diagram column patterns
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == 'tc' or col_lower == 't_critical' or col_lower == 'critical_temp':
                    T_col = col
                elif col_lower == 'mu' or col_lower == 'chemical_potential':
                    mu_col = col
            
            # Fallback to first matches
            if not T_col and temp_cols:
                T_col = temp_cols[0]
            if not mu_col and mu_cols:
                mu_col = mu_cols[0]
            
            if T_col and mu_col:
                print(f"  Using columns: T='{T_col}', mu='{mu_col}'")
                
                # Find chiral transition at smallest |mu|
                df_clean = df.dropna(subset=[T_col, mu_col])
                if len(df_clean) > 0:
                    min_mu_idx = df_clean[mu_col].abs().idxmin()
                    closest_mu = df_clean.loc[min_mu_idx, mu_col]
                    chiral_T = df_clean.loc[min_mu_idx, T_col]
                    
                    results['chiral_critical_T_mu0'] = float(chiral_T)
                    results['chiral_critical_mu_mu0'] = float(closest_mu)
                    
                    print(f"  → Chiral transition: T={chiral_T:.2f} MeV at μ={closest_mu:.2f} MeV")
                    
                    # Look for phase transition order at the lowest μ value
                    phase_cols = [col for col in df.columns if any(p in col.lower() for p in ['phase', 'order', 'transition'])]
                    if phase_cols:
                        print(f"  → Found phase columns: {phase_cols}")
                        # Extract transition order at the lowest |μ| value
                        phase_col = phase_cols[0]
                        phase_val = df_clean.loc[min_mu_idx, phase_col]
                        
                        if pd.notna(phase_val):
                            # Convert to numeric if needed
                            try:
                                phase_val = float(phase_val)
                                # Interpretation: 1 = first-order, 2 = crossover/second-order
                                if phase_val == 1:
                                    results['chiral_transition_order'] = 1  # First-order
                                    order_text = "first-order"
                                elif phase_val >= 2:
                                    results['chiral_transition_order'] = 2  # Crossover
                                    order_text = "crossover"
                                else:
                                    results['chiral_transition_order'] = int(phase_val)
                                    order_text = f"order-{int(phase_val)}"
                                
                                print(f"  → Phase transition order at μ={closest_mu:.2f} MeV: {results['chiral_transition_order']} ({order_text})")
                                
                                # If transition at lowest μ is crossover, look for critical point
                                if results['chiral_transition_order'] == 2:  # Crossover at lowest μ
                                    print("  → Crossover at lowest μ - searching for critical point...")
                                    
                                    # Look for transition from crossover (2) to first-order (1) with increasing μ
                                    df_ordered = df_clean.sort_values(mu_col)
                                    prev_order = None
                                    critical_found = False
                                    
                                    for idx, row in df_ordered.iterrows():
                                        current_order = row[phase_col]
                                        current_mu = row[mu_col]
                                        current_T = row[T_col]
                                        
                                        if pd.notna(current_order):
                                            current_order = float(current_order)
                                            
                                            # Check for transition from crossover (2) to first-order (1)
                                            if prev_order is not None and prev_order >= 2 and current_order == 1:
                                                results['has_critical_point'] = True
                                                results['critical_point_T'] = float(current_T)
                                                results['critical_point_mu'] = float(current_mu)
                                                critical_found = True
                                                print(f"  → Critical point found: T={current_T:.2f} MeV, μ={current_mu:.2f} MeV")
                                                print(f"    (Transition from crossover to first-order)")
                                                break
                                            
                                            prev_order = current_order
                                    
                                    if not critical_found:
                                        print("  → No crossover→first-order transition found")
                                        results['has_critical_point'] = False
                                else:
                                    # First-order at lowest μ - check for explicit critical point data
                                    print("  → First-order at lowest μ - checking for explicit critical point data...")
                                    results['has_critical_point'] = False  # Default unless found explicitly
                                    
                            except (ValueError, TypeError):
                                print(f"  → Could not parse phase order value: {phase_val}")
                        else:
                            print(f"  → No phase order data at μ={closest_mu:.2f} MeV")
                    else:
                        print("  → No phase order columns found")
                    
                    # Look for explicit critical point data in columns
                    cp_cols = [col for col in df.columns if any(cp in col.lower() for cp in ['critical', 'cp_', 'crit']) and col.lower() != phase_col.lower()]
                    if cp_cols and 'has_critical_point' not in results:
                        print(f"  → Found explicit critical point columns: {cp_cols}")
                        
                        # Look for critical point T and mu values
                        cp_T_cols = [col for col in cp_cols if any(t in col.lower() for t in ['t', 'temp'])]
                        cp_mu_cols = [col for col in cp_cols if any(m in col.lower() for m in ['mu', 'chem'])]
                        
                        if cp_T_cols and cp_mu_cols:
                            cp_T_data = df_clean[cp_T_cols[0]].dropna()
                            cp_mu_data = df_clean[cp_mu_cols[0]].dropna()
                            
                            if len(cp_T_data) > 0 and len(cp_mu_data) > 0:
                                results['has_critical_point'] = True
                                results['critical_point_T'] = float(cp_T_data.iloc[0])
                                results['critical_point_mu'] = float(cp_mu_data.iloc[0])
                                print(f"  → Explicit critical point: T={results['critical_point_T']:.2f} MeV, μ={results['critical_point_mu']:.2f} MeV")
                    
                    # Default values if not found
                    if 'chiral_transition_order' not in results:
                        results['chiral_transition_order'] = 2  # Default to crossover
                    if 'has_critical_point' not in results:
                        results['has_critical_point'] = False
                        
                else:
                    print("  → No valid T,μ data found in phase diagram")
            else:
                print(f"  → Could not find T and μ columns. Available: {list(df.columns)}")
        
        except Exception as e:
            print(f"  → Error analyzing phase diagram: {e}")
        
        return results
    
    def analyze_critical_points_data(self, critical_files):
        """
        Analyze critical point data files for QCD critical endpoints.
        
        Returns:
        --------
        dict: Extracted critical point results
        """
        results = {}
        
        if not critical_files:
            return results
        
        for critical_file in critical_files:
            print(f"Analyzing critical points file: {critical_file.name}")
            
            try:
                # Handle different file formats
                if critical_file.suffix == '.csv':
                    df = pd.read_csv(critical_file)
                elif critical_file.suffix == '.npy':
                    data = np.load(critical_file)
                    # Convert numpy array to DataFrame if possible
                    if data.ndim == 2 and data.shape[1] >= 2:
                        df = pd.DataFrame(data, columns=['T', 'mu'][:data.shape[1]])
                    else:
                        print(f"  → Cannot parse numpy array shape: {data.shape}")
                        continue
                else:
                    print(f"  → Unsupported file format: {critical_file.suffix}")
                    continue
                
                print(f"  Critical points shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                
                # Look for temperature and mu columns
                temp_cols = [col for col in df.columns if any(t in col.lower() for t in ['temp', 't', '_crit']) and 'mu' not in col.lower()]
                mu_cols = [col for col in df.columns if any(m in col.lower() for m in ['mu', 'chemical', 'chem_pot']) and any(c in col.lower() for c in ['crit', '_c'])]
                
                T_col = None
                mu_col = None
                
                # Look for specific critical point column patterns
                for col in df.columns:
                    col_lower = col.lower()
                    if 'tc_crit' in col_lower or 'temp_crit' in col_lower or 't_crit' in col_lower:
                        T_col = col
                    elif 'mu_crit' in col_lower or 'chem_crit' in col_lower:
                        mu_col = col
                
                # Fallback to general patterns
                if not T_col and temp_cols:
                    T_col = temp_cols[0]
                if not mu_col and mu_cols:
                    mu_col = mu_cols[0]
                
                if T_col and mu_col:
                    print(f"  Using columns: T='{T_col}', mu='{mu_col}'")
                    
                    # Clean data
                    df_clean = df.dropna(subset=[T_col, mu_col])
                    if len(df_clean) > 0:
                        results['has_critical_points'] = True
                        results['num_critical_points'] = len(df_clean)
                        
                        # Extract critical endpoint (highest μ critical point typically)
                        max_mu_idx = df_clean[mu_col].idxmax()
                        cp_T = df_clean.loc[max_mu_idx, T_col]
                        cp_mu = df_clean.loc[max_mu_idx, mu_col]
                        
                        results['critical_endpoint_T'] = float(cp_T)
                        results['critical_endpoint_mu'] = float(cp_mu)
                        
                        print(f"  → Critical endpoint: T={cp_T:.2f} MeV, μ={cp_mu:.2f} MeV")
                        
                        # Statistics of critical points
                        results['cp_T_range'] = [float(df_clean[T_col].min()), float(df_clean[T_col].max())]
                        results['cp_mu_range'] = [float(df_clean[mu_col].min()), float(df_clean[mu_col].max())]
                        
                        print(f"  → T range: [{results['cp_T_range'][0]:.1f}, {results['cp_T_range'][1]:.1f}] MeV")
                        print(f"  → μ range: [{results['cp_mu_range'][0]:.1f}, {results['cp_mu_range'][1]:.1f}] MeV")
                        
                        # Look for additional critical point properties
                        prop_cols = [col for col in df.columns if col not in [T_col, mu_col]]
                        if prop_cols:
                            print(f"  → Additional properties: {prop_cols}")
                            for prop_col in prop_cols[:3]:  # Limit to first 3 additional properties
                                if pd.api.types.is_numeric_dtype(df_clean[prop_col]):
                                    prop_val = df_clean.loc[max_mu_idx, prop_col]
                                    if pd.notna(prop_val):
                                        results[f'cp_{prop_col}'] = float(prop_val)
                                        print(f"    {prop_col} = {prop_val:.3f}")
                        
                        break  # Use first successfully analyzed file
                    else:
                        print("  → No valid critical point data found")
                else:
                    print(f"  → Could not find T and μ columns")
            
            except Exception as e:
                print(f"  → Error analyzing critical points: {e}")
                continue
        
        # Default values if no critical points found
        if 'has_critical_points' not in results:
            results['has_critical_points'] = False
            results['num_critical_points'] = 0
        
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
        
        # Critical points analysis
        critical_results = self.analyze_critical_points_data(result_files['critical_points'])
        results.update(critical_results)
        
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract physics results from calculation data')
    parser.add_argument('--task-id', required=True, help='Task ID for this calculation')
    parser.add_argument('--data-dir', help='Directory containing result files (default: current directory)')
    parser.add_argument('--date', help='Calculation date (default: today)')
    
    args = parser.parse_args()
    
    extractor = PhysicsResultsExtractor(data_dir=args.data_dir)
    success = extractor.auto_extract_results(
        task_id=args.task_id,
        calculation_date=args.date
    )
    
    sys.exit(0 if success else 1)
