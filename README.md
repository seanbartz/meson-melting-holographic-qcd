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
- **Phase Diagram Mapping**: Generate comprehensive phase diagrams with chiral, vector, and axial melting lines
- **Critical Point Analysis**: Identify and analyze critical points in the phase diagram
- **SLURM Cluster Support**: Automated job arrays for high-performance computing clusters
- **Parallel Processing**: Efficient computation using joblib and cluster parallelization
- **Data Management**: Automated data saving, organization, and duplicate cleaning for ML preparation
- **Machine Learning Ready**: Clean datasets with automatic duplicate removal

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
- `clean_sigma_duplicates.py` ⭐ **NEW** - Clean duplicate sigma data for ML preparation

### SLURM Cluster Support ⭐ **NEW**
- `slurm_batch_array.sh` - SLURM job array script for parameter combinations
- `submit_array_job.sh` - Helper script to calculate and submit job arrays
- `clean_cluster_sigma_data.sh` - Convenient cluster data cleaning wrapper

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

### SLURM Cluster Usage ⭐ **NEW**
```bash
# Automated parameter scan on clusters (e.g., Obsidian)
./submit_array_job.sh -mq 12.0 -lambda1values 5.8 5.9 6.0 -gamma -22.4 -lambda4 4.2

# Monitor running jobs
squeue -u $USER

# Check job outputs
tail -f slurm_logs/batch_JOBID_TASKID.out
```

### Data Cleaning for Machine Learning ⭐ **NEW**
```bash
# Clean duplicate sigma data entries (automatic backup)
python3 clean_sigma_duplicates.py /net/project/QUENCH/sigma_data/sigma_calculations.csv

# Preview cleaning without modifying files
python3 clean_sigma_duplicates.py sigma_calculations.csv --preview

# Cluster convenience script
./clean_cluster_sigma_data.sh preview  # Preview mode
./clean_cluster_sigma_data.sh          # Clean with backup
```

## Parameters

### Physical Parameters
- `T` - Temperature (MeV)
- `mu` - Chemical potential (MeV, default range: 0-400 MeV)  
- `mq` - Quark mass parameter
- `lambda1` - Coupling parameter (default: 7.438)
- `gamma` - Gamma parameter (default: -22.4)
- `lambda4` - Lambda4 parameter (default: 4.2)

### Numerical Parameters
- `wi`, `wf` - Frequency range for spectral functions (MeV)
- `wcount` - Number of frequency points
- `wresolution` - Target frequency resolution
- `ui`, `uf` - Holographic coordinate range
- `mupoints` - Number of chemical potential points (default: 21 for 20 MeV spacing)

## Output Structure

The code automatically organizes output into structured directories:

**Local/Development:**
```
mu_g_440/
├── axial_data/          # Spectral function data files
├── axial_plots/         # Generated plots
├── phase_data/          # Critical point data
└── phase_plots/         # Critical point analysis plots
```

**Cluster (Obsidian) Shared Storage:**
```
/net/project/QUENCH/
├── phase_data/          # Phase diagram CSV data
├── phase_plots/         # Phase diagram PNG plots  
└── sigma_data/          # Sigma calculation data for ML
    ├── sigma_calculations.csv        # All calculation results
    ├── sigma_calculations_cleaned.csv # Deduplicated for ML
    └── sigma_calculations_backup_*.csv # Timestamped backups
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

## Cluster Deployment (Obsidian)

### Setup
1. Clone repository on cluster: `git clone https://github.com/seanbartz/meson-melting-holographic-qcd.git`
2. Make scripts executable: `chmod +x *.sh *.py`
3. Ensure Python 3 is available: `python3 --version`

### Workflow
1. **Development**: Test parameters locally or with small jobs
2. **Scaling**: Use SLURM job arrays for large parameter scans
3. **Data Management**: Results automatically saved to shared project storage
4. **ML Preparation**: Clean duplicates from accumulated sigma data

### Best Practices
- Use `--preview` mode to test job arrays before submission
- Monitor disk usage in `/net/project/QUENCH/`  
- Clean sigma data periodically to remove duplicates
- Use `git pull` to sync latest improvements

## Research Context

This code implements holographic QCD calculations based on the gauge/gravity correspondence. The model uses a dilaton-gravity background to study:

- Quark-gluon plasma properties
- Meson spectral functions at finite temperature/density  
- Chiral symmetry breaking and restoration
- Confinement/deconfinement transitions

### Machine Learning Integration

The code generates comprehensive datasets suitable for machine learning applications:

- **Sigma calculations**: Complete phase space mapping for ML training
- **Duplicate removal**: Eliminates bias from overrepresented parameter combinations
- **Standardized format**: CSV files with consistent column structure
- **Metadata tracking**: Timestamps, computation details, and provenance

**Why clean duplicates?** During development and testing, identical parameter combinations are often calculated multiple times. For ML training, each unique physics configuration should appear only once to prevent:
- Training bias toward frequently-run parameters
- Inflated dataset sizes without additional information
- Statistical skewing of parameter space coverage

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
