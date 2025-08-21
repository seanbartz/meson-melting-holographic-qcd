#!/usr/bin/env python3
"""
Script to rename and organize axial melting files with new naming convention.

This script:
1. Renames axial melting files to include gamma and lambda4 parameters with underscores
2. Creates subdirectories for data and plots
3. Moves files to appropriate subdirectories
4. Uses default values gamma=-22.4 and lambda4=4.2 for existing files

New naming convention:
- Data files: axial_melting_data_mq_{mq}_lambda1_{lambda1}_gamma_{gamma}_lambda4_{lambda4}.csv
- Plot files: axial_melting_curve_mq_{mq}_lambda1_{lambda1}_gamma_{gamma}_lambda4_{lambda4}.png

Created on August 4, 2025
"""

import os
import re
import shutil
from pathlib import Path

def parse_old_filename(filename):
    """
    Parse old filename to extract mq and lambda1 values.
    
    Expected formats:
    - axial_melting_data_mq{mq}_lambda{lambda1}.csv
    - axial_melting_curve_mq{mq}_lambda{lambda1}.png
    
    Returns:
        tuple: (file_type, mq, lambda1, extension) or None if parsing fails
    """
    # Pattern for data files
    data_pattern = r'axial_melting_data_mq([0-9]+\.?[0-9]*)_lambda([0-9]+\.?[0-9]*)\.csv'
    # Pattern for plot files  
    plot_pattern = r'axial_melting_curve_mq([0-9]+\.?[0-9]*)_lambda([0-9]+\.?[0-9]*)\.png'
    
    data_match = re.match(data_pattern, filename)
    if data_match:
        mq = float(data_match.group(1))
        lambda1 = float(data_match.group(2))
        return ('data', mq, lambda1, '.csv')
    
    plot_match = re.match(plot_pattern, filename)
    if plot_match:
        mq = float(plot_match.group(1))
        lambda1 = float(plot_match.group(2))
        return ('curve', mq, lambda1, '.png')
    
    return None

def generate_new_filename(file_type, mq, lambda1, gamma=-22.4, lambda4=4.2):
    """
    Generate new filename with updated naming convention.
    
    Args:
        file_type: 'data' or 'curve'
        mq: Quark mass value
        lambda1: Lambda1 parameter value
        gamma: Gamma parameter value (default: -22.4)
        lambda4: Lambda4 parameter value (default: 4.2)
    
    Returns:
        str: New filename
    """
    if file_type == 'data':
        return f'axial_melting_data_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv'
    elif file_type == 'curve':
        return f'axial_melting_curve_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.png'
    else:
        raise ValueError(f"Unknown file type: {file_type}")

def create_directories():
    """Create subdirectories for organizing files."""
    directories = [
        'axial_data',
        'axial_plots'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def rename_and_move_files(dry_run=False):
    """
    Rename and move axial melting files to new naming convention and directories.
    
    Args:
        dry_run: If True, only print what would be done without actually doing it
    """
    current_dir = Path('.')
    
    # Find all axial melting files
    axial_files = []
    for pattern in ['axial_melting_data_*.csv', 'axial_melting_curve_*.png']:
        axial_files.extend(current_dir.glob(pattern))
    
    print(f"Found {len(axial_files)} axial melting files")
    
    # Track statistics
    renamed_count = 0
    skipped_count = 0
    
    # Default parameter values for existing files
    default_gamma = -22.4
    default_lambda4 = 4.2
    
    for old_file in axial_files:
        filename = old_file.name
        
        # Parse the old filename
        parsed = parse_old_filename(filename)
        if parsed is None:
            print(f"SKIPPED: Could not parse filename: {filename}")
            skipped_count += 1
            continue
        
        file_type, mq, lambda1, extension = parsed
        
        # Generate new filename
        new_filename = generate_new_filename(file_type, mq, lambda1, default_gamma, default_lambda4)
        
        # Determine target directory
        if file_type == 'data':
            target_dir = Path('axial_data')
        elif file_type == 'curve':
            target_dir = Path('axial_plots')
        else:
            print(f"SKIPPED: Unknown file type for {filename}")
            skipped_count += 1
            continue
        
        target_path = target_dir / new_filename
        
        print(f"{'[DRY RUN] ' if dry_run else ''}RENAME: {filename}")
        print(f"{'[DRY RUN] ' if dry_run else ''}    -> {target_path}")
        print(f"{'[DRY RUN] ' if dry_run else ''}    mq={mq:.1f}, lambda1={lambda1:.1f}, gamma={default_gamma:.1f}, lambda4={default_lambda4:.1f}")
        
        if not dry_run:
            try:
                # Move and rename the file
                shutil.move(str(old_file), str(target_path))
                renamed_count += 1
                print(f"    SUCCESS")
            except Exception as e:
                print(f"    ERROR: {e}")
                skipped_count += 1
        else:
            renamed_count += 1
        
        print()
    
    print("=" * 60)
    print("SUMMARY:")
    print(f"Total files processed: {len(axial_files)}")
    print(f"Successfully renamed: {renamed_count}")
    print(f"Skipped: {skipped_count}")
    
    if dry_run:
        print("\nThis was a dry run. No files were actually moved.")
        print("Run with dry_run=False to perform the actual renaming.")

def main():
    """Main function"""
    print("Axial Melting File Renaming Script")
    print("=" * 60)
    print("This script will rename and organize axial melting files with the new naming convention.")
    print(f"Default values: gamma = -22.4, lambda4 = 4.2")
    print()
    
    # Create directories first
    print("Creating directories...")
    create_directories()
    print()
    
    # First do a dry run to show what will happen
    print("DRY RUN - Showing what will be done:")
    print("-" * 40)
    rename_and_move_files(dry_run=True)
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with the actual renaming? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        print("\nProceeding with actual renaming...")
        print("-" * 40)
        rename_and_move_files(dry_run=False)
    else:
        print("Operation cancelled.")

if __name__ == '__main__':
    main()
