#!/usr/bin/env python3
"""
Clean duplicate entries from sigma_calculations.csv for ML preparation.

Removes entries with identical physics parameters, keeping only the most recent
timestamp for each unique parameter combination. This prevents ML bias from
overrepresented parameter sets.

The script automatically creates a timestamped backup of the original file
before cleaning, unless disabled with --no-backup or in preview mode.

Usage:
    # On Obsidian cluster:
    python clean_sigma_duplicates.py /net/project/QUENCH/sigma_data/sigma_calculations.csv
    
    # Local usage:
    python clean_sigma_duplicates.py [input_file] [output_file] [--tolerance TOLERANCE]

Arguments:
    input_file: CSV file to clean (default: sigma_calculations.csv in current directory)
    output_file: Cleaned output file (default: input_file_cleaned.csv)
    --tolerance: Floating point tolerance for parameter comparison (default: 1e-10)
"""

import pandas as pd
import numpy as np
import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime

def clean_sigma_duplicates(input_file, output_file=None, tolerance=1e-10, create_backup=True):
    """
    Remove duplicate entries from sigma calculations CSV.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str, optional
        Path to output CSV file (default: input_file_cleaned.csv)
    tolerance : float
        Tolerance for floating point comparison (default: 1e-10)
    create_backup : bool
        Whether to create a backup of the original file (default: True)
    
    Returns:
    --------
    tuple: (original_count, cleaned_count, duplicates_removed)
    """
    
    print(f"Loading data from {input_file}...")
    
    try:
        # Try to read CSV with error handling for malformed lines
        try:
            # First try modern pandas approach
            df = pd.read_csv(input_file, on_bad_lines='skip')
        except TypeError:
            # Fallback for older pandas versions
            df = pd.read_csv(input_file, error_bad_lines=False, warn_bad_lines=True)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return None
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        print("This might be due to malformed CSV data.")
        print("Attempting manual line-by-line parsing...")
        
        # Last resort: manual parsing of CSV
        try:
            import csv
            good_lines = []
            expected_fields = None
            
            with open(input_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                expected_fields = len(header)
                good_lines.append(header)
                
                for line_num, row in enumerate(reader, start=2):
                    if len(row) == expected_fields:
                        good_lines.append(row)
                    else:
                        print(f"Skipping malformed line {line_num}: expected {expected_fields} fields, got {len(row)}")
            
            # Convert to DataFrame
            if len(good_lines) > 1:
                df = pd.DataFrame(good_lines[1:], columns=good_lines[0])
                print(f"Successfully parsed {len(df)} valid lines out of potentially more")
            else:
                print("No valid data lines found")
                return None
                
        except Exception as e2:
            print(f"Manual parsing also failed: {e2}")
            return None
    
    original_count = len(df)
    print(f"Original dataset: {original_count} entries")
    
    # Additional check: ensure all rows have the expected number of columns
    expected_cols = len(df.columns)
    print(f"Expected columns: {expected_cols}")
    
    # Filter out any rows with NaN values or wrong structure (safety check)
    initial_len = len(df)
    df_clean_structure = df.dropna(how='all')  # Remove completely empty rows
    
    # Check for rows with inconsistent data (e.g., string in numeric columns)
    malformed_rows = 0
    clean_indices = []
    
    for idx, row in df_clean_structure.iterrows():
        try:
            # Basic sanity check: ensure row has proper data structure
            is_valid = True
            
            # Check critical columns that must have values (not sigma2, sigma3, d0_2, d0_3)
            required_cols = ['T', 'mu', 'ui', 'uf', 'mq', 'mq_tolerance', 'lambda1', 'num_solutions']
            
            for col in required_cols:
                if col in df_clean_structure.columns:
                    val = row[col]
                    if pd.isna(val) or val == '' or str(val).strip() == '':
                        is_valid = False
                        break
                        
            # For sigma/d0 columns, allow empty values (when num_solutions < 3)
            # But if they exist, they should be numeric or empty
            optional_numeric_cols = ['sigma1', 'sigma2', 'sigma3', 'd0_1', 'd0_2', 'd0_3']
            for col in optional_numeric_cols:
                if col in df_clean_structure.columns:
                    val = row[col]
                    if val != '' and not pd.isna(val):
                        try:
                            float(val)  # Test if it's a valid number
                        except (ValueError, TypeError):
                            is_valid = False
                            break
                        
            if is_valid:
                clean_indices.append(idx)
            else:
                malformed_rows += 1
                
        except Exception as e:
            malformed_rows += 1
            continue
    
    if malformed_rows > 0:
        print(f"Filtering out {malformed_rows} malformed/incomplete rows")
        print("  (Rows missing required physics parameters or with invalid data types)")
        df = df_clean_structure.loc[clean_indices]
        print(f"After structure cleaning: {len(df)} entries")
    else:
        df = df_clean_structure
        if len(df) < initial_len:
            empty_removed = initial_len - len(df)
            print(f"Removed {empty_removed} completely empty rows")
            print(f"After structure cleaning: {len(df)} entries")
        
    print("Note: Empty sigma2/sigma3/d0_2/d0_3 fields are valid (when num_solutions < 3)")
    
    # Create backup of original file before cleaning
    if create_backup:
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = input_path.parent / f"{input_path.stem}_backup_{timestamp}{input_path.suffix}"
        
        print(f"Creating backup: {backup_file}")
        try:
            shutil.copy2(input_file, backup_file)
            print(f"✓ Backup created successfully")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
            print("Continuing without backup...")
    
    # Identify physics parameter columns (exclude timestamp and calculation metadata)
    exclude_cols = ['timestamp', 'calculation_time', 'task_id', 'job_id', 'node']
    physics_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Physics parameter columns: {physics_cols}")
    
    # Round physics parameters to avoid floating point precision issues
    df_rounded = df.copy()
    for col in physics_cols:
        if df[col].dtype in ['float64', 'float32']:
            # Use tolerance-based rounding
            df_rounded[col] = np.round(df[col] / tolerance) * tolerance
    
    # Sort by timestamp (most recent first) to keep latest entries
    if 'timestamp' in df.columns:
        df_rounded = df_rounded.sort_values('timestamp', ascending=False)
        print("Sorted by timestamp (keeping most recent duplicates)")
    else:
        print("Warning: No timestamp column found, keeping arbitrary duplicates")
    
    # Remove duplicates based on physics parameters only
    df_cleaned = df_rounded.drop_duplicates(subset=physics_cols, keep='first')
    
    cleaned_count = len(df_cleaned)
    structure_removed = original_count - len(df)  # Malformed/empty rows removed
    duplicates_removed = len(df) - cleaned_count  # Duplicates among valid rows
    total_removed = original_count - cleaned_count
    
    print(f"Cleaned dataset: {cleaned_count} entries")
    print(f"Structure issues removed: {structure_removed}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Total removed: {total_removed} ({100*total_removed/original_count:.1f}%)")
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    
    # Save cleaned data
    print(f"Saving cleaned data to {output_file}...")
    df_cleaned.to_csv(output_file, index=False)
    
    # Generate summary report
    report_file = Path(output_file).parent / f"{Path(output_file).stem}_report.txt"
    with open(report_file, 'w') as f:
        f.write("Sigma Data Cleaning Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Floating point tolerance: {tolerance}\n\n")
        f.write(f"Original entries: {original_count}\n")
        f.write(f"Structure issues removed: {structure_removed}\n")
        f.write(f"Duplicates removed: {duplicates_removed}\n")
        f.write(f"Cleaned entries: {cleaned_count}\n") 
        f.write(f"Total compression: {100*total_removed/original_count:.1f}%\n\n")
        f.write(f"Physics parameter columns used for deduplication:\n")
        for col in physics_cols:
            f.write(f"  - {col}\n")
    
    print(f"Summary report saved to {report_file}")
    
    return original_count, cleaned_count, total_removed

def main():
    parser = argparse.ArgumentParser(
        description="Clean duplicate entries from sigma_calculations.csv for ML preparation"
    )
    parser.add_argument(
        'input_file', 
        nargs='?', 
        default='sigma_calculations.csv',
        help='Input CSV file (default: sigma_calculations.csv)'
    )
    parser.add_argument(
        'output_file',
        nargs='?', 
        help='Output CSV file (default: input_file_cleaned.csv)'
    )
    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=1e-10,
        help='Floating point tolerance for parameter comparison (default: 1e-10)'
    )
    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Preview what would be cleaned without saving'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup of original file'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} does not exist")
        
        # Suggest common locations
        if 'sigma_calculations.csv' in args.input_file:
            print("\nCommon locations for sigma data:")
            print("  - On Obsidian cluster: /net/project/QUENCH/sigma_data/sigma_calculations.csv")
            print("  - Local directory: ./sigma_calculations.csv")
        
        print("\nAvailable CSV files in current directory:")
        csv_files = list(Path('.').glob('*.csv'))
        if csv_files:
            for f in csv_files:
                print(f"  - {f}")
        else:
            print("  (no CSV files found)")
        sys.exit(1)
    
    if args.preview:
        print("PREVIEW MODE - No files will be modified")
        # In preview mode, use a temporary output file and don't create backup
        result = clean_sigma_duplicates(args.input_file, '/tmp/preview_output.csv', 
                                      args.tolerance, create_backup=False)
        # Clean up temp file (compatible with older Python versions)
        try:
            Path('/tmp/preview_output.csv').unlink()
        except FileNotFoundError:
            pass
    else:
        result = clean_sigma_duplicates(args.input_file, args.output_file, 
                                      args.tolerance, create_backup=not args.no_backup)
    
    if result is None:
        sys.exit(1)
    
    original, cleaned, removed = result
    print(f"\nCleaning complete!")
    
    if removed > 0:
        print(f"Recommendation: Use the cleaned dataset for ML to avoid parameter bias")
        print(f"Space saved: ~{removed} rows")
    else:
        print("No duplicates found - dataset is already clean")

if __name__ == "__main__":
    main()
