# Axial Melting Scan Updates

## Summary of Changes

Updated `axial_melting_scan.py` to accept gamma and lambda4 as optional CLI arguments, convert them to internal v3 and v4 parameters, use mug as a command line parameter, and implement the new file naming and directory conventions.

## Key Changes

### 1. Command Line Arguments
- Added `--gamma` (optional): Background metric parameter gamma
- Added `--lambda4` (optional): Fourth-order coupling parameter lambda4
- Added `--mug` (default: 440.0): mu_g parameter (previously imported from axial_spectra)

### 2. Parameter Conversion
- When both `--gamma` and `--lambda4` are provided:
  - `v3 = gamma / (6 * sqrt(2))`
  - `v4 = lambda4`
- If only one is provided, a warning is shown and defaults are used
- Backward compatibility maintained when neither are provided

### 3. Function Signatures Updated
- `initialize_chiral_field_and_derivative()`: Added `v3=None, v4=None` parameters
- `has_zeros_in_derivative()`: Added `v3=None, v4=None, mug=440.0` parameters
- `find_melting_temperature()`: Added `v3=None, v4=None, mug=440.0` parameters
- `scan_melting_temperatures()`: Added `v3=None, v4=None, mug=440.0` parameters
- `plot_melting_curve()`: Added `gamma=None, lambda4=None` parameters
- `save_data()`: Added `gamma=None, lambda4=None` parameters

### 4. File Naming Convention
- **Data files**: `axial_data/axial_melting_data_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv`
- **Plot files**: `axial_plots/axial_melting_curve_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.png`
- Falls back to old naming when gamma/lambda4 not provided: `axial_melting_data_mq_{mq:.1f}_lambda1_{lambda1:.1f}.csv`

### 5. Directory Structure
- Data files saved to: `axial_data/` (automatically created)
- Plot files saved to: `axial_plots/` (automatically created)

### 6. Backward Compatibility
- All existing functionality preserved when gamma/lambda4 not provided
- Default mug value of 440.0 maintains existing behavior
- Functions work with or without v3/v4 parameters

## Usage Examples

### New Usage (with gamma and lambda4):
```bash
python axial_melting_scan.py --mq 9.0 --lambda1 7.438 --gamma -22.4 --lambda4 4.2 --mug 440.0
```

### Backward Compatible Usage:
```bash
python axial_melting_scan.py --mq 9.0 --lambda1 7.438 --mug 450.0
```

### Called from map_phase_diagram_improved.py:
```bash
python axial_melting_scan.py --mq 9.0 --lambda1 7.438 --gamma -22.4 --lambda4 4.2 --no-display
```

## Integration with map_phase_diagram_improved.py

The script now accepts the same gamma and lambda4 parameters that `map_phase_diagram_improved.py` passes, ensuring consistency across the codebase. The `generate_axial_melting_data()` function in `map_phase_diagram_improved.py` calls this script with the appropriate parameters.

## Files Modified
- `/Users/seanbartz/Dropbox/ISU/Research/QGP/DilatonMixing/MesonMelting/axial_melting_scan.py`

## Files Created
- `/Users/seanbartz/Dropbox/ISU/Research/QGP/DilatonMixing/MesonMelting/AXIAL_MELTING_SCAN_UPDATES.md` (this file)
