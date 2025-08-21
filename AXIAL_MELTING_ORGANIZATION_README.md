# Axial Melting File Organization and Naming Convention Update

## Overview
This document describes the changes made to organize and standardize the naming convention for axial melting files, along with updates to the phase diagram mapping script.

## Changes Made

### 1. Created `rename_axial_melting_files.py`
A comprehensive script that:
- **Renames** all existing axial melting files to the new naming convention
- **Creates subdirectories** for better organization:
  - `axial_data/` - for CSV data files
  - `axial_plots/` - for PNG plot files
- **Moves files** to appropriate subdirectories
- **Applies default parameter values** (gamma=-22.4, lambda4=4.2) to existing files

#### New Naming Convention:
- **Data files**: `axial_melting_data_mq_{mq}_lambda1_{lambda1}_gamma_{gamma}_lambda4_{lambda4}.csv`
- **Plot files**: `axial_melting_curve_mq_{mq}_lambda1_{lambda1}_gamma_{gamma}_lambda4_{lambda4}.png`

#### Key Features:
- **Dry run mode** - shows what will be done before making changes
- **Interactive confirmation** - asks user before proceeding
- **Error handling** - gracefully handles parsing failures
- **Detailed logging** - shows progress and results
- **Statistics tracking** - reports success/failure counts

### 2. Updated `map_phase_diagram_improved.py`
Modified the script to work with the new file organization:

#### Changes Made:
1. **Updated `generate_axial_melting_data()` function**:
   - Creates `axial_data/` directory automatically
   - Uses new naming convention for output files
   - Passes gamma and lambda4 parameters to axial_melting_scan.py

2. **Updated `create_phase_diagram_plot()` function**:
   - Looks for axial data in `axial_data/` subdirectory
   - Uses new naming convention with underscores

3. **Enhanced directory management**:
   - Creates all necessary subdirectories: `CP_data/`, `CP_plots/`, `axial_data/`, `axial_plots/`
   - Ensures directories exist before file operations

4. **Improved parameter passing**:
   - Passes gamma and lambda4 to axial_melting_scan.py command
   - Maintains consistency across all file naming

## Directory Structure After Changes

```
MesonMelting/
├── CP_data/                    # Critical point data files
├── CP_plots/                   # Critical point plot files  
├── axial_data/                 # Axial melting data files
├── axial_plots/                # Axial melting plot files
├── map_phase_diagram_improved.py
├── rename_axial_melting_files.py
└── ... (other files)
```

## Usage Instructions

### Step 1: Run the Renaming Script
```bash
python rename_axial_melting_files.py
```
- The script will show a dry run first
- Review the proposed changes
- Confirm to proceed with actual renaming

### Step 2: Use Updated Phase Diagram Script
```bash
python map_phase_diagram_improved.py 6.0 0.5 --gamma -20.0 --lambda4 5.0
```
- The script will automatically look for axial data in the correct location
- Will generate new axial data with proper naming if needed
- All output files will use the new naming convention

## File Naming Examples

### Before (Old Convention):
```
axial_melting_data_mq0.5_lambda6.0.csv
axial_melting_curve_mq0.5_lambda6.0.png
```

### After (New Convention):
```
axial_data/axial_melting_data_mq_0.5_lambda1_6.0_gamma_-22.4_lambda4_4.2.csv
axial_plots/axial_melting_curve_mq_0.5_lambda1_6.0_gamma_-22.4_lambda4_4.2.png
```

## Benefits

1. **Consistent Naming**: All files now use the same underscore-separated convention
2. **Complete Parameter Information**: Filenames include all relevant parameters (mq, lambda1, gamma, lambda4)
3. **Organized Structure**: Files are organized in logical subdirectories
4. **Backward Compatibility**: Existing files are migrated with appropriate default values
5. **Future-Proof**: New files will automatically use the correct convention
6. **Parameter Studies**: Easy to identify files for specific parameter combinations

## Notes

- All existing files assume default values: gamma = -22.4, lambda4 = 4.2
- The renaming script includes safety features (dry run, confirmation)
- The updated phase diagram script is fully backward compatible
- Directory creation is automatic and safe (uses `exist_ok=True`)

## Files Modified/Created

1. **Created**: `rename_axial_melting_files.py` - File renaming and organization script
2. **Modified**: `map_phase_diagram_improved.py` - Updated to use new file organization
3. **Created**: This documentation file

This completes the standardization of axial melting file naming and organization to match the critical point data conventions.
