# Axial Melting Scan Updates

## Summary of Changes

Updated `axial_melting_scan.py` to:
1. Accept gamma and lambda4 as optional CLI arguments
2. Convert them to internal v3 and v4 parameters
3. Use mug as a command line parameter
4. Implement the new file naming and directory conventions
5. **Align default μ range with phase diagram calculations (0-400 MeV)**

## Key Changes

### 1. Command Line Arguments
- Added `-gamma` (optional): Background metric parameter gamma
- Added `-lambda4` (optional): Fourth-order coupling parameter lambda4
- Added `-mug` (default: 440.0): mu_g parameter (previously imported from axial_spectra)
- **Updated `-mumax` default from 200.0 to 400.0 MeV** to align with phase diagram calculations

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

### Full Usage (with all parameters):
```bash
python axial_melting_scan.py -mq 9.0 -lambda1 7.438 -gamma -22.4 -lambda4 4.2 -mug 440.0 -mumin 0.0 -mumax 400.0 -mupoints 21
```

### Default Usage (uses aligned defaults):
```bash
# Default mumax is now 400.0 MeV (aligned with phase diagram)
python axial_melting_scan.py -mq 9.0 -lambda1 7.438 -gamma -22.4 -lambda4 4.2
```

### Custom μ Range (override defaults):
```bash
# Specify custom range if needed (e.g., for focused analysis)
python axial_melting_scan.py -mq 9.0 -lambda1 7.438 -gamma -22.4 -lambda4 4.2 -mumin 0.0 -mumax 200.0 -mupoints 21
```

### Called from map_phase_diagram_improved.py:
```bash
# Phase diagram script passes its μ range to axial melting scan
python axial_melting_scan.py -mq 9.0 -lambda1 7.438 -gamma -22.4 -lambda4 4.2 -mumin 0.0 -mumax 400.0 -mupoints 21 --no-display
```

## Integration with map_phase_diagram_improved.py

The script now accepts the same gamma and lambda4 parameters that `map_phase_diagram_improved.py` passes, ensuring consistency across the codebase. The `generate_axial_melting_data()` function in `map_phase_diagram_improved.py` calls this script with the appropriate parameters.

## Files Modified
- `/Users/seanbartz/Dropbox/ISU/Research/QGP/DilatonMixing/MesonMelting/axial_melting_scan.py`

## Files Created
- `/Users/seanbartz/Dropbox/ISU/Research/QGP/DilatonMixing/MesonMelting/AXIAL_MELTING_SCAN_UPDATES.md` (this file)
