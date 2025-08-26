#!/usr/bin/env python3
"""
Script to rename all data files in phase_data directory.

This script:
1. Adds underscores between parameter names and values
2. Appends gamma=-22.4 and lambda4=4.2 values to filenames
3. Handles both chiral_transition_ and phase_diagram_ file patterns
4. Preserves file extensions (.pkl, .csv, .png, etc.)

Example transformations:
- phase_diagram_improved_ml12.0_lambda16.0.csv 
  → phase_diagram_improved_ml_12.0_lambda1_6.0_gamma_-22.4_lambda4_4.2.csv
- chiral_transition_mq9_mu0_lambda15.000000_order1.pkl
  → chiral_transition_mq_9_mu_0_lambda1_5.000000_gamma_-22.4_lambda4_4.2_order_1.pkl
"""

import os
import re
import shutil
from pathlib import Path

def parse_and_rename_file(filename, directory):
    """
    Parse a filename and create the new renamed version.
    
    Args:
        filename (str): Original filename
        directory (str): Directory path
        
    Returns:
        tuple: (old_path, new_path, success)
    """
    # Split filename and extension
    name_parts = filename.rsplit('.', 1)
    if len(name_parts) == 2:
        base_name, extension = name_parts
    else:
        base_name = filename
        extension = ""
    
    # Define the gamma and lambda4 values to append
    gamma_val = -22.4
    lambda4_val = 4.2
    
    old_path = os.path.join(directory, filename)
    
    # Handle different file patterns
    if base_name.startswith('phase_diagram_improved_'):
        # Pattern: phase_diagram_improved_ml12.0_lambda16.0
        new_name = rename_phase_diagram_improved(base_name, gamma_val, lambda4_val)
    elif base_name.startswith('phase_diagram_'):
        # Pattern: phase_diagram_ml9.0_lambda14.5
        new_name = rename_phase_diagram(base_name, gamma_val, lambda4_val)
    elif base_name.startswith('chiral_transition_'):
        # Pattern: chiral_transition_mq9_mu0_lambda15.000000_order1
        new_name = rename_chiral_transition(base_name, gamma_val, lambda4_val)
    else:
        print(f"Unknown pattern for file: {filename}")
        return old_path, None, False
    
    if new_name is None:
        return old_path, None, False
    
    # Add extension back
    if extension:
        new_filename = f"{new_name}.{extension}"
    else:
        new_filename = new_name
    
    new_path = os.path.join(directory, new_filename)
    
    return old_path, new_path, True

def rename_phase_diagram_improved(base_name, gamma_val, lambda4_val):
    """
    Rename phase_diagram_improved_ files.
    Example: phase_diagram_improved_ml12.0_lambda16.0 
    → phase_diagram_improved_mq_12.0_lambda1_6.0_gamma_-22.4_lambda4_4.2
    """
    # Pattern: phase_diagram_improved_ml{value}_lambda1{value}
    pattern = r'phase_diagram_improved_ml([\d.]+)_lambda1([\d.]+)'
    match = re.match(pattern, base_name)
    
    if match:
        ml_val = match.group(1)
        lambda1_val = match.group(2)
        new_name = f"phase_diagram_improved_mq_{ml_val}_lambda1_{lambda1_val}_gamma_{gamma_val}_lambda4_{lambda4_val}"
        return new_name
    
    # Alternative pattern without '1' after lambda: ml{value}_lambda{value}
    pattern2 = r'phase_diagram_improved_ml([\d.]+)_lambda([\d.]+)'
    match2 = re.match(pattern2, base_name)
    
    if match2:
        ml_val = match2.group(1)
        lambda_val = match2.group(2)
        new_name = f"phase_diagram_improved_mq_{ml_val}_lambda1_{lambda_val}_gamma_{gamma_val}_lambda4_{lambda4_val}"
        return new_name
    
    print(f"Could not parse phase_diagram_improved pattern: {base_name}")
    return None

def rename_phase_diagram(base_name, gamma_val, lambda4_val):
    """
    Rename phase_diagram_ files.
    Example: phase_diagram_ml9.0_lambda14.5 
    → phase_diagram_ml_9.0_lambda1_14.5_gamma_-22.4_lambda4_4.2
    """
    # Pattern: phase_diagram_ml{value}_lambda{value}
    pattern = r'phase_diagram_ml([\d.]+)_lambda([\d.]+)'
    match = re.match(pattern, base_name)
    
    if match:
        ml_val = match.group(1)
        lambda_val = match.group(2)
        new_name = f"phase_diagram_ml_{ml_val}_lambda1_{lambda_val}_gamma_{gamma_val}_lambda4_{lambda4_val}"
        return new_name
    
    print(f"Could not parse phase_diagram pattern: {base_name}")
    return None

def rename_chiral_transition(base_name, gamma_val, lambda4_val):
    """
    Rename chiral_transition_ files.
    Example: chiral_transition_mq9_mu0_lambda15.000000_order1
    → chiral_transition_mq_9_mu_0_lambda1_5.000000_gamma_-22.4_lambda4_4.2_order_1
    """
    # Pattern: chiral_transition_mq{value}_mu{value}_lambda1{value}_order{value}
    pattern = r'chiral_transition_mq([\d.]+)_mu([\d.]+)_lambda1([\d.]+)_order([\d]+)'
    match = re.match(pattern, base_name)
    
    if match:
        mq_val = match.group(1)
        mu_val = match.group(2)
        lambda1_val = match.group(3)
        order_val = match.group(4)
        new_name = f"chiral_transition_mq_{mq_val}_mu_{mu_val}_lambda1_{lambda1_val}_gamma_{gamma_val}_lambda4_{lambda4_val}_order_{order_val}"
        return new_name
    
    # Alternative pattern without '1' after lambda
    pattern2 = r'chiral_transition_mq([\d.]+)_mu([\d.]+)_lambda([\d.]+)_order([\d]+)'
    match2 = re.match(pattern2, base_name)
    
    if match2:
        mq_val = match2.group(1)
        mu_val = match2.group(2)
        lambda_val = match2.group(3)
        order_val = match2.group(4)
        new_name = f"chiral_transition_mq_{mq_val}_mu_{mu_val}_lambda1_{lambda_val}_gamma_{gamma_val}_lambda4_{lambda4_val}_order_{order_val}"
        return new_name
    
    print(f"Could not parse chiral_transition pattern: {base_name}")
    return None

def main():
    """Main function to rename all files in phase_data directory."""
    
    # Define the directory
    cp_data_dir = "phase_data"
    
    # Check if directory exists
    if not os.path.exists(cp_data_dir):
        print(f"Directory {cp_data_dir} does not exist!")
        return
    
    # Get all files in the directory
    files = [f for f in os.listdir(cp_data_dir) if os.path.isfile(os.path.join(cp_data_dir, f))]
    
    if not files:
        print(f"No files found in {cp_data_dir} directory.")
        return
    
    print(f"Found {len(files)} files in {cp_data_dir} directory.")
    print("=" * 70)
    
    # Lists to track results
    successful_renames = []
    failed_renames = []
    skipped_files = []
    
    # Process each file
    for filename in files:
        print(f"Processing: {filename}")
        
        old_path, new_path, success = parse_and_rename_file(filename, cp_data_dir)
        
        if not success or new_path is None:
            failed_renames.append(filename)
            print(f"  ❌ Failed to parse filename pattern")
            continue
        
        # Check if new filename would be the same (already renamed)
        if old_path == new_path:
            skipped_files.append(filename)
            print(f"  ⏭️  Already has correct format, skipping")
            continue
        
        # Check if target file already exists
        if os.path.exists(new_path):
            print(f"  ⚠️  Target file already exists: {os.path.basename(new_path)}")
            failed_renames.append(filename)
            continue
        
        try:
            # Perform the rename
            shutil.move(old_path, new_path)
            successful_renames.append((filename, os.path.basename(new_path)))
            print(f"  ✅ Renamed to: {os.path.basename(new_path)}")
        except Exception as e:
            failed_renames.append(filename)
            print(f"  ❌ Error renaming file: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("RENAMING SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(files)}")
    print(f"Successfully renamed: {len(successful_renames)}")
    print(f"Skipped (already correct): {len(skipped_files)}")
    print(f"Failed: {len(failed_renames)}")
    
    if successful_renames:
        print(f"\n✅ SUCCESSFULLY RENAMED ({len(successful_renames)} files):")
        for old_name, new_name in successful_renames:
            print(f"  {old_name}")
            print(f"  → {new_name}")
            print()
    
    if skipped_files:
        print(f"\n⏭️  SKIPPED ({len(skipped_files)} files):")
        for filename in skipped_files:
            print(f"  {filename}")
    
    if failed_renames:
        print(f"\n❌ FAILED ({len(failed_renames)} files):")
        for filename in failed_renames:
            print(f"  {filename}")
    
    print("\nRenaming operation completed!")

if __name__ == "__main__":
    main()
