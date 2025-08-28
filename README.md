# Meson Melting in Holographic QCD

This repository contains code for studying meson melting in holographic QCD using the gauge/gravity correspondence. The project focuses on computing spectral functions for axial and vector mesons at finite temperature and chemical potential.

## Overview

This research investigates the melting of mesons in a holographic model of QCD, particularly studying:
- Axial vector meson spectral functions
- Vector meson spectral functions  
- Phase diagrams and critical points
- Temperature and chemical potential dependence
- Chiral field dynamics

## Key Features

- **Spectral Function Calculations**: Compute spectral functions for axial and vector mesons
- **Phase Diagram Mapping**: Generate phase diagrams showing meson melting transitions
- **Critical Point Analysis**: Identify and analyze critical points in the phase diagram
- **Parallel Processing**: Efficient computation using joblib for large parameter scans
- **Data Management**: Automated data saving and organization with timestamp tracking

## Main Scripts

### Core Spectral Function Calculators
- `axial_spectra.py` - Calculate axial vector meson spectral functions
- `vector_spectra.py` - Calculate vector meson spectral functions  
- `chiral_solve_complete.py` - Solve for chiral field configurations

### Phase Diagram Tools
- `map_phase_diagram.py` - Generate phase diagrams
- `map_phase_diagram_improved.py` - Enhanced phase diagram generation
- `batch_phase_diagram_unified.py` ⭐ **NEW** - Unified batch scanner for any parameter combinations
- `batch_phase_diagram_scan.py` - Legacy batch processing for gamma/lambda4 scans
- `batch_phase_diagrams.py` - Legacy batch processing for lambda1 scans

### Analysis and Utilities
- `critical_zoom_improved.py` - High-resolution critical point analysis
- `critical_analysis_utils.py` - Utilities for critical point studies
- `calculate_peak_widths.py` - Analyze meson peak properties

### Scanning and Batch Processing
- `run_axial_temperature_scan.py` - Temperature scans for axial mesons
- `run_temperature_scan.py` - General temperature scanning
- `run_batch_scan.py` - Batch processing coordination

## Usage Examples

### Basic Spectral Function Calculation
```bash
# Calculate axial spectral function at T=17 MeV, μ=0 MeV
python axial_spectra.py -T 17.0 -mu 0.0 -mq 9.0 -lambda1 7.438

# Calculate with custom frequency range and resolution
python axial_spectra.py -T 20.0 -mu 50.0 -wi 500 -wf 3000 -wr 0.05
```

### Phase Diagram Generation
```bash
# Generate phase diagram for specific quark mass
python map_phase_diagram_improved.py -mq 9.0 -lambda1 7.438

# NEW: Unified batch scanner for any parameter combinations
python batch_phase_diagram_unified.py -gammarange -25.0 -20.0 -gammapoints 6 -mq 9.0 -lambda1 5.0

# NEW: Multi-parameter scan (Cartesian product) - minimal syntax
python batch_phase_diagram_unified.py -mqvalues 9.0 12.0 -gammavalues -25.0 -22.4 -lambda1 5.0

# NEW: Minimal usage (gamma=-22.4, lambda4=4.2 used as defaults)
python batch_phase_diagram_unified.py -mq 9.0 -lambda1 5.0

# Legacy batch processing
python run_batch_scan.py
```

### Critical Point Analysis
```bash
# High-resolution critical point study
python critical_zoom_improved.py -T 95.0 -mu 100.0
```

## Parameters

### Physical Parameters
- `T` - Temperature (MeV)
- `mu` - Chemical potential (MeV)  
- `mq` - Quark mass parameter
- `lambda1` - Coupling parameter (default: 7.438)

### Numerical Parameters
- `wi`, `wf` - Frequency range for spectral functions (MeV)
- `wcount` - Number of frequency points
- `wresolution` - Target frequency resolution
- `ui`, `uf` - Holographic coordinate range

## Output Structure

The code automatically organizes output into structured directories:

```
mu_g_440/
├── axial_data/          # Spectral function data files
├── axial_plots/         # Generated plots
├── phase_data/          # Critical point data
└── phase_plots/         # Critical point analysis plots
```

## Dependencies

- `numpy` - Numerical computations
- `scipy` - Integration and interpolation
- `matplotlib` - Plotting and visualization
- `pandas` - Data manipulation and CSV handling
- `joblib` - Parallel processing
- `argparse` - Command-line argument parsing

## Installation

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install numpy scipy matplotlib pandas joblib
   ```
3. Run calculations using the examples above

## Research Context

This code implements holographic QCD calculations based on the gauge/gravity correspondence. The model uses a dilaton-gravity background to study:

- Quark-gluon plasma properties
- Meson spectral functions at finite temperature/density  
- Chiral symmetry breaking and restoration
- Confinement/deconfinement transitions

## File Organization

### Configuration and Documentation
- `README_phase_diagram.md` - Phase diagram documentation
- `AXIAL_MELTING_ORGANIZATION_README.md` - Axial melting study documentation
- `BATCH_PHASE_DIAGRAM_README.md` - Batch processing documentation

### Jupyter Notebooks
- `Axial.ipynb` - Interactive axial meson analysis
- `Vector_Spectra.ipynb` - Interactive vector meson analysis
- `WorkbookJuly2025.ipynb` - Current research notebook

## Citation

If you use this code in your research, please cite the relevant publications:

[Add your publications here]

## Contact

Sean Bartz - sean.bartz@indstate.edu
Indiana State University

## License

[Add license information]
