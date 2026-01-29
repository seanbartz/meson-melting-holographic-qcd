# File Naming Standards

## Overview
This document defines the standardized file naming conventions used throughout the holographic QCD meson melting analysis codebase.

## General Principles

### 1. Parameter Separation
- Always use underscores to separate parameter names from their values
- Format: `parameter_value` not `parametervalue`
- Example: `mq_9.0` not `mq9.0`

### 2. Parameter Names
- Use `mq` for quark mass (not `ml`)
- Use `lambda1` for the λ₁ mixing parameter (not just `lambda`)
- Use consistent parameter abbreviations across all files

### 3. File Extensions
- Data files: `.csv`
- Plot files: `.png` (with optional `.pdf` versions)
- Pickle files: `.pkl`

## File Type Conventions

### Phase Diagram Files
**Location**: `phase_data/` directory
**Pattern**: `phase_diagram_improved_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv`
**Example**: `phase_diagram_improved_mq_9.0_lambda1_7.4_gamma_-22.4_lambda4_4.2.csv`

### Axial Melting Data Files  
**Location**: `axial_data/` directory
**Pattern**: `axial_melting_data_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv`
**Example**: `axial_melting_data_mq_15.0_lambda1_7.0_gamma_-22.6_lambda4_4.2.csv`

### Axial Melting Plot Files
**Location**: `axial_plots/` directory  
**Pattern**: `axial_melting_curve_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.png`
**Example**: `axial_melting_curve_mq_15.0_lambda1_7.0_gamma_-22.6_lambda4_4.2.png`

### Axial Spectral Data Files
**Location**: `mu_g_440/axial_data/` directory
**Pattern**: `axial_spectral_data_T_{T:.1f}_mu_{mu:.1f}_mq_{mq:.1f}_lambda1_{lambda1:.1f}.csv`
**Example**: `axial_spectral_data_T_98.0_mu_0.0_mq_9.0_lambda1_5.3.csv`

### Axial Peak Data Files
**Location**: `mu_g_440/axial_data/` directory
**Pattern**: `axial_peaks_data_T_{T:.1f}_mu_{mu:.1f}_mq_{mq:.1f}_lambda1_{lambda1:.1f}.csv`
**Example**: `axial_peaks_data_T_98.0_mu_0.0_mq_9.0_lambda1_5.3.csv`

### Combined Phase Diagram Plots
**Location**: `phase_plots/` directory
**Pattern**: `combined_phase_diagram_{parameter}_scan_mq_{mq:.1f}_lambda1_{lambda1:.1f}.png`
**Example**: `combined_phase_diagram_gamma_scan_mq_9.0_lambda1_5.0.png`

### Chiral Transition Files
**Location**: `phase_data/` directory
**Pattern**: `chiral_transition_mq_{mq}_mu_{mu}_lambda1_{lambda1}_gamma_{gamma}_lambda4_{lambda4}_order_{order}.pkl`
**Example**: `chiral_transition_mq_9_mu_0_lambda1_5.000000_gamma_-22.4_lambda4_4.2_order_1.pkl`

## Parameter Reference

### Standard Parameters
- `mq`: Quark mass in MeV (replaces old `ml`)
- `lambda1`: λ₁ mixing parameter (replaces old `lambda`)  
- `gamma`: Background metric parameter γ
- `lambda4`: Fourth-order coupling parameter λ₄
- `T`: Temperature in MeV
- `mu`: Chemical potential μ in MeV
- `mug`: μ_g parameter (usually 440.0 MeV)

### Formatting Guidelines
- Numerical values: Use 1 decimal place for most parameters (`.1f`)
- Exception: Very precise lambda1 values may use more decimals (`.3f` or `.6f`)
- Negative values: Include minus sign directly (e.g., `gamma_-22.4`)

## Migration from Old Format

### Old → New Mappings
- `ml` → `mq` (quark mass parameter name)
- `lambda` → `lambda1` (mixing parameter name)  
- `parametervalue` → `parameter_value` (add underscores)
- `phase_diagram_improved_ml` → `phase_diagram_improved_mq`

### Example Conversions
```
OLD: phase_diagram_improved_ml13.0_lambda16.0.png  
NEW: phase_diagram_improved_mq_13.0_lambda1_16.0_gamma_-22.4_lambda4_4.2.png

OLD: axial_melting_data_mq9.0_lambda7.4.csv
NEW: axial_melting_data_mq_9.0_lambda1_7.4_gamma_-22.6_lambda4_4.2.csv
```

## Implementation Notes

### Code Generation
All filename generation should use Python f-string formatting:
```python
filename = f"phase_diagram_improved_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv"
```

### Directory Structure
Files are organized into logical subdirectories:
- `phase_data/` - Critical point and phase diagram data
- `phase_plots/` - Critical point and phase diagram plots
- `axial_data/` - Axial meson data files
- `axial_plots/` - Axial meson plot files
- `mu_g_440/axial_data/` - Spectral function data

### Backward Compatibility
- Migration scripts are provided for converting old filename formats
- Legacy code should be updated to use new conventions
- Documentation examples should reflect current standards

## Benefits

1. **Consistency**: All files use the same underscore-separated convention
2. **Clarity**: Parameter names clearly distinguish quark mass (`mq`) from other masses
3. **Completeness**: Filenames include all relevant physics parameters
4. **Organization**: Logical directory structure separates different file types
5. **Maintainability**: Standardized patterns make automated processing easier

## Compliance

All new code should follow these naming conventions. Existing files can be migrated using the provided renaming scripts. Any deviations from these standards should be documented and justified.

Created: August 21, 2025
Updated: August 21, 2025
