#!/usr/bin/env python3
"""
Task Summary Logger for Physics Calculation Results

Creates and maintains a CSV file tracking key physics results from each
calculation task, including input parameters and extracted physics quantities
like melting temperatures, critical points, and phase transition characteristics.

Usage:
    from task_summary_logger import TaskSummaryLogger
    
    logger = TaskSummaryLogger()
    
    # Log a new task
    logger.log_task_summary(
        # Input parameters
        mq=9.0, lambda1=7.438, gamma=15.2, lambda4=12.5,
        T_min=10.0, T_max=120.0, mu_min=0.0, mu_max=100.0, num_mu_values=11,
        
        # Physics results  
        axial_melting_T_mu0=85.3, chiral_critical_T_mu0=102.1,
        chiral_transition_order=2, has_critical_point=True,
        critical_point_T=95.2, critical_point_mu=45.8,
        
        # Additional analysis
        axial_vector_merge=True, merge_T=78.5, merge_mu=25.0,
        axial_chiral_cross=False, cross_T=None, cross_mu=None,
        
        # Metadata
        task_id="batch_990_5", calculation_date="2025-09-09"
    )
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import fcntl
import time

class TaskSummaryLogger:
    """
    Manages a CSV file containing summary results from physics calculation tasks.
    
    Each row represents one complete calculation task with its input parameters
    and extracted physics results.
    """
    
    def __init__(self, filename='task_summary.csv', data_dir='summary_data'):
        """
        Initialize the task summary logger.
        
        Parameters:
        -----------
        filename : str
            Name of the CSV file to store summaries
        data_dir : str
            Directory to store the summary file
        """
        self.data_dir = data_dir
        self.filename = filename
        self.filepath = os.path.join(data_dir, filename)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Define CSV column structure
        self.columns = [
            # Metadata
            'task_id', 'calculation_date', 'timestamp',
            
            # Input Parameters
            'mq', 'lambda1', 'gamma', 'lambda4',
            'T_min', 'T_max', 'mu_min', 'mu_max', 'num_mu_values',
            
            # Axial Results
            'axial_melting_T_mu0', 'axial_melting_mu_mu0',
            
            # Chiral Results  
            'chiral_critical_T_mu0', 'chiral_critical_mu_mu0',
            'chiral_transition_order',
            
            # Critical Point Analysis
            'has_critical_point', 'critical_point_T', 'critical_point_mu',
            
            # Melting Temperature Analysis
            'axial_vector_merge', 'merge_T', 'merge_mu',
            
            # Phase Line Intersection Analysis
            'axial_chiral_cross', 'cross_T', 'cross_mu',
            
            # Additional Metrics
            'total_calculations', 'convergence_issues', 'notes'
        ]
        
        # Create CSV file with headers if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.filepath):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.filepath, index=False)
            print(f"Created new task summary file: {self.filepath}")
    
    def _acquire_file_lock(self, file_handle, max_retries=10, retry_delay=0.1):
        """
        Acquire exclusive lock on file with retries.
        
        Parameters:
        -----------
        file_handle : file object
            Open file handle to lock
        max_retries : int
            Maximum number of retry attempts
        retry_delay : float
            Delay between retries in seconds
        """
        for attempt in range(max_retries):
            try:
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except IOError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return False
        return False
    
    def log_task_summary(self, task_id, calculation_date=None, **kwargs):
        """
        Log a completed task summary to the CSV file.
        
        Parameters:
        -----------
        task_id : str
            Unique identifier for this calculation task
        calculation_date : str, optional
            Date of calculation (default: today's date)
        **kwargs : dict
            Physics results and parameters to log
            
        Expected keyword arguments:
        - Input parameters: mq, lambda1, gamma, lambda4
        - Temperature/mu ranges: T_min, T_max, mu_min, mu_max, num_mu_values  
        - Axial results: axial_melting_T_mu0, axial_melting_mu_mu0
        - Chiral results: chiral_critical_T_mu0, chiral_critical_mu_mu0, chiral_transition_order
        - Critical point: has_critical_point, critical_point_T, critical_point_mu
        - Merging analysis: axial_vector_merge, merge_T, merge_mu
        - Crossing analysis: axial_chiral_cross, cross_T, cross_mu
        - Additional: total_calculations, convergence_issues, notes
        """
        if calculation_date is None:
            calculation_date = datetime.now().strftime("%Y-%m-%d")
        
        # Prepare the new row data
        new_row = {
            'task_id': task_id,
            'calculation_date': calculation_date,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add all provided parameters, using None for missing values
        for col in self.columns[3:]:  # Skip metadata columns we already set
            new_row[col] = kwargs.get(col, None)
        
        # Validate critical fields
        required_fields = ['mq', 'lambda1', 'gamma', 'lambda4']
        missing_fields = [field for field in required_fields if new_row.get(field) is None]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Check for duplicate task_id
        if self.task_exists(task_id):
            print(f"Warning: Task {task_id} already exists. Use update_task_summary() to modify.")
            return False
        
        # Thread-safe append to CSV file
        try:
            with open(self.filepath, 'a', newline='') as f:
                if self._acquire_file_lock(f):
                    try:
                        df = pd.DataFrame([new_row])
                        df.to_csv(f, header=False, index=False)
                        print(f"✓ Logged task summary: {task_id}")
                        return True
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                else:
                    print(f"Error: Could not acquire file lock for {self.filepath}")
                    return False
                    
        except Exception as e:
            print(f"Error logging task summary: {e}")
            return False
    
    def task_exists(self, task_id):
        """Check if a task_id already exists in the CSV."""
        try:
            df = pd.read_csv(self.filepath)
            return task_id in df['task_id'].values
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return False
    
    def update_task_summary(self, task_id, **kwargs):
        """
        Update an existing task summary with new information.
        
        Parameters:
        -----------
        task_id : str
            Task ID to update
        **kwargs : dict
            Fields to update
        """
        try:
            df = pd.read_csv(self.filepath)
            
            if task_id not in df['task_id'].values:
                print(f"Task {task_id} not found. Use log_task_summary() to create new entry.")
                return False
            
            # Update specified fields
            for field, value in kwargs.items():
                if field in df.columns:
                    df.loc[df['task_id'] == task_id, field] = value
                else:
                    print(f"Warning: Unknown field '{field}' ignored")
            
            # Update timestamp
            df.loc[df['task_id'] == task_id, 'timestamp'] = datetime.now().isoformat()
            
            # Save updated DataFrame
            with open(self.filepath, 'w', newline='') as f:
                if self._acquire_file_lock(f):
                    try:
                        df.to_csv(f, index=False)
                        print(f"✓ Updated task summary: {task_id}")
                        return True
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                else:
                    print(f"Error: Could not acquire file lock for {self.filepath}")
                    return False
                    
        except Exception as e:
            print(f"Error updating task summary: {e}")
            return False
    
    def get_summary_dataframe(self):
        """
        Load and return the complete task summary as a pandas DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            Complete task summary data
        """
        try:
            return pd.read_csv(self.filepath)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=self.columns)
    
    def generate_physics_report(self, output_file=None):
        """
        Generate a comprehensive physics summary report.
        
        Parameters:
        -----------
        output_file : str, optional
            Path to save the report (default: physics_summary_report.txt)
        """
        if output_file is None:
            output_file = os.path.join(self.data_dir, 'physics_summary_report.txt')
        
        df = self.get_summary_dataframe()
        
        if df.empty:
            print("No task summaries available for report generation")
            return
        
        with open(output_file, 'w') as f:
            f.write("Physics Calculation Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total tasks: {len(df)}\n\n")
            
            # Parameter ranges
            f.write("Input Parameter Ranges:\n")
            f.write("-" * 25 + "\n")
            for param in ['mq', 'lambda1', 'gamma', 'lambda4']:
                if param in df.columns and df[param].notna().any():
                    values = df[param].dropna()
                    f.write(f"{param}: {values.min():.3f} - {values.max():.3f}\n")
            
            # Physics results summary
            f.write("\nPhysics Results Summary:\n")
            f.write("-" * 25 + "\n")
            
            # Axial melting temperatures
            if 'axial_melting_T_mu0' in df.columns:
                temps = df['axial_melting_T_mu0'].dropna()
                if len(temps) > 0:
                    f.write(f"Axial melting temperatures (μ=0): {temps.min():.1f} - {temps.max():.1f} MeV\n")
            
            # Chiral critical temperatures
            if 'chiral_critical_T_mu0' in df.columns:
                temps = df['chiral_critical_T_mu0'].dropna()
                if len(temps) > 0:
                    f.write(f"Chiral critical temperatures (μ=0): {temps.min():.1f} - {temps.max():.1f} MeV\n")
            
            # Critical point statistics
            if 'has_critical_point' in df.columns:
                cp_count = df['has_critical_point'].sum()
                f.write(f"Tasks with critical points: {cp_count}/{len(df)} ({100*cp_count/len(df):.1f}%)\n")
            
            # Melting temperature merging
            if 'axial_vector_merge' in df.columns:
                merge_count = df['axial_vector_merge'].sum()
                f.write(f"Tasks with axial-vector merging: {merge_count}/{len(df)} ({100*merge_count/len(df):.1f}%)\n")
            
            # Phase line crossings
            if 'axial_chiral_cross' in df.columns:
                cross_count = df['axial_chiral_cross'].sum()
                f.write(f"Tasks with axial-chiral crossings: {cross_count}/{len(df)} ({100*cross_count/len(df):.1f}%)\n")
            
            f.write(f"\nDetailed data available in: {self.filepath}\n")
        
        print(f"Physics summary report saved to: {output_file}")

def create_example_task_log():
    """Create an example task log entry for demonstration."""
    logger = TaskSummaryLogger()
    
    # Example log entry
    logger.log_task_summary(
        task_id="example_batch_001",
        calculation_date="2025-09-09",
        
        # Input parameters
        mq=9.0,
        lambda1=7.438,
        gamma=15.2,
        lambda4=12.5,
        T_min=10.0,
        T_max=120.0,
        mu_min=0.0,
        mu_max=100.0,
        num_mu_values=11,
        
        # Physics results
        axial_melting_T_mu0=85.3,
        axial_melting_mu_mu0=0.0,
        chiral_critical_T_mu0=102.1,
        chiral_critical_mu_mu0=0.0,
        chiral_transition_order=2,  # crossover
        
        # Critical point
        has_critical_point=True,
        critical_point_T=95.2,
        critical_point_mu=45.8,
        
        # Melting analysis
        axial_vector_merge=True,
        merge_T=78.5,
        merge_mu=25.0,
        
        # Phase line crossings
        axial_chiral_cross=False,
        cross_T=None,
        cross_mu=None,
        
        # Additional info
        total_calculations=121,  # 11 mu values × 11 T values
        convergence_issues=0,
        notes="Successful full parameter scan"
    )
    
    print("Example task summary logged successfully!")

if __name__ == "__main__":
    # Create example entry for demonstration
    create_example_task_log()
    
    # Generate example report
    logger = TaskSummaryLogger()
    logger.generate_physics_report()
