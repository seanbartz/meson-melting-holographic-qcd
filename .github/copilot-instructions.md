# Meson Melting in Holographic QCD
Holographic QCD research project for computing meson spectral functions and phase diagrams using the gauge/gravity correspondence. The codebase studies meson melting, chiral symmetry breaking, and QCD phase transitions at finite temperature and chemical potential.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup (REQUIRED FIRST)
- Install Python dependencies: `pip3 install numpy scipy matplotlib pandas joblib`
- Make all scripts executable: `chmod +x *.sh *.py`
- Verify Python version: `python3 --version` (tested with Python 3.12.3)
- Verify installation: `python3 -c "import numpy, scipy, matplotlib, pandas, joblib; print('All dependencies installed successfully')"`

### Core Calculations and Timing
- **Basic spectral function**: `python3 axial_spectra.py -T 17.0 -mu 0.0 -mq 9.0 -lambda1 7.438 --no-plot`
  - Takes 10-15 minutes to complete. NEVER CANCEL. Set timeout to 30+ minutes.
- **Phase diagram (3 points)**: `python3 map_phase_diagram_improved.py -mq 9.0 -lambda1 5.0 -mupoints 3 --no-plot`
  - Takes ~1 minute for 3 points, ~30 minutes for full 21 points. NEVER CANCEL. Set timeout to 60+ minutes for full calculations.
- **Single parameter test**: `python3 test_phase_args.py` -- quick argument validation (< 1 second)

### Build and Test Workflow
- NO traditional build step required (pure Python)
- Test functionality: Run `python3 test_phase_args.py` and `python3 -c "import axial_spectra, map_phase_diagram_improved; print('Modules load successfully')"`
- Validate computations: Run a quick calculation with `python3 map_phase_diagram_improved.py -mq 9.0 -lambda1 5.0 -mupoints 3 --no-plot`
- Check outputs: Look for CSV files in `phase_data/` and plots in `mu_g_440/axial_plots/`

### SLURM Cluster Usage (Obsidian)
- **Setup shared output**: `./setup_shared_output.sh /net/project/QUENCH`
- **Submit job array**: `./submit_array_job.sh -mq 12.0 -lambda1values 5.8 5.9 6.0 -gamma -22.4`
  - Script automatically analyzes cluster and optimizes CPU allocation (1-20 CPUs/task)
  - Uses intelligent resource allocation: full nodes when available, smaller jobs when busy
  - Job arrays can take hours to complete. NEVER CANCEL. Monitor with `squeue -u $USER`
- **Conservative mode**: `export CONSERVATIVE_MODE=1` forces smaller CPU allocations for faster queue times
- **Check cluster**: `sinfo -N -o "%N %C"` shows node availability

## Validation

### Manual Validation Scenarios
ALWAYS run these scenarios after making changes to core calculation code:

**Scenario 1: Basic Spectral Function**
1. Run: `python3 axial_spectra.py -T 17.0 -mu 0.0 -mq 9.0 -lambda1 7.438 --no-plot`
2. Verify: Check for output files in `mu_g_440/axial_data/` containing spectral and peak data
3. Expected: Calculation completes with ~7 peaks found and saves CSV data files

**Scenario 2: Phase Diagram Generation**
1. Run: `python3 map_phase_diagram_improved.py -mq 9.0 -lambda1 5.0 -mupoints 3 --no-plot`
2. Verify: Check `phase_data/phase_diagram_improved_mq_9.0_lambda1_5.0_gamma_-22.4_lambda4_4.2.csv` exists
3. Expected: 3 rows of data with mu, Tc, order, max_sigma columns

**Scenario 3: Batch Parameter Scan**
1. Run: `python3 batch_phase_diagram_scan.py --parameter gamma --values -22.4 -22.6 -lambda1 5.0 -mq 9.0 -mupoints 3 --skip-existing`
2. Verify: Multiple CSV files created in `phase_data/` for different gamma values
3. Expected: Script processes multiple parameter values and creates combined analysis

### Data Integrity Validation
- Always check that CSV output files contain expected columns and data format
- Verify that spectral function peaks are physically reasonable (positive values, proper scaling)
- Ensure phase diagram critical temperatures are within expected ranges (50-100 MeV typically)

## CRITICAL Timing and Cancellation Warnings

### NEVER CANCEL Commands - Wait Times
- **Spectral function calculations**: 10-30 minutes. NEVER CANCEL. Set timeout to 45+ minutes.
- **Phase diagrams (full 21 μ points)**: 30-60 minutes. NEVER CANCEL. Set timeout to 90+ minutes.
- **Batch scans (multiple parameters)**: 1-6 hours. NEVER CANCEL. Set timeout to 8+ hours.
- **SLURM job arrays**: Can run for hours. NEVER CANCEL. Monitor with `squeue`, check logs in `slurm_logs/`

### Performance Expectations
- Single μ-point phase diagram: ~1 minute (3-10 iterations, ~36 temperature evaluations)
- Full 21-point phase diagram: 20-60 minutes depending on parameter complexity  
- Spectral function with 1700 frequency points: 10-30 minutes
- The code uses parallel processing (joblib) and automatically detects CPU cores

## Command Line Standards

### Parameter Naming (CRITICAL - follow exactly)
- Temperature: `-T` (MeV)
- Chemical potential: `-mu` (MeV)  
- Quark mass: `-mq` (MeV)
- Lambda parameters: `-lambda1`, `-lambda4`
- Gamma parameter: `-gamma`
- Frequency range: `-wi`, `-wf`, `-wc`, `-wr`
- Chemical potential range: `-mumin`, `-mumax`, `-mupoints`
- Temperature range: `-tmin`, `-tmax`
- Boolean flags: `--no-plot`, `--no-display`, `--skip-existing`

### Standard Usage Examples
```bash
# Basic spectral function
python3 axial_spectra.py -T 17.0 -mu 0.0 -mq 9.0 -lambda1 7.438

# Phase diagram with custom parameters  
python3 map_phase_diagram_improved.py -mq 9.0 -lambda1 5.0 -gamma -22.6 -lambda4 4.2

# Batch parameter scan
python3 batch_phase_diagram_scan.py --parameter gamma --values -25.0 -22.4 -20.0 -lambda1 5.0 -mq 9.0

# SLURM cluster submission
./submit_array_job.sh -mqvalues 9.0 12.0 15.0 -lambda1 5.0 -gamma -22.4
```

## Output Structure and Key Locations

### Local Development Output
```
mu_g_440/
├── axial_data/          # Spectral function CSV data
├── axial_plots/         # PNG plots of spectra
phase_data/              # Phase diagram CSV files  
phase_plots/             # Phase diagram PNG plots
sigma_data/              # Sigma calculation logs (CSV)
```

### Key Files to Check After Changes
- `phase_data/*.csv` - Phase diagram critical point data
- `mu_g_440/axial_data/*.csv` - Spectral function and peak data
- `sigma_data/sigma_calculations.csv` - Calculation log with all parameters

### Cluster Shared Storage (if PROJECT_DIR set)
```
/net/project/QUENCH/
├── phase_data/          # Shared phase diagram results
├── phase_plots/         # Shared phase diagram plots  
└── sigma_data/          # ML-ready sigma calculation data
```

## Common Tasks

### Data Cleaning and ML Preparation
```bash
# Clean duplicate sigma data entries (automatic backup)
python3 clean_sigma_duplicates.py sigma_data/sigma_calculations.csv

# Preview cleaning without modifying files  
python3 clean_sigma_duplicates.py sigma_calculations.csv --preview

# Cluster data cleaning
./clean_cluster_sigma_data.sh preview  # Preview mode
./clean_cluster_sigma_data.sh          # Clean with backup
```

### Development Workflow Checks
- ALWAYS test with minimal parameters first: `-mupoints 3` instead of default 21
- Use `--no-plot` or `--no-display` flags to avoid display issues in headless environments
- Use `--skip-existing` to avoid recalculating existing data during development
- Monitor `sigma_data/sigma_calculations.csv` to track all calculations performed

### Parameter Validation
- Quark mass (`mq`): Typically 9.0-24.0 MeV
- Lambda1: Typically 3.0-7.8  
- Gamma: Typically -25.0 to -20.0
- Lambda4: Typically 3.0-5.5
- Temperature ranges: 80-210 MeV for search, 17-50 MeV for spectral functions
- Chemical potential: 0-400 MeV typical range

## Troubleshooting

### Module Import Errors
If Python modules fail to import:
```bash
pip3 install --user numpy scipy matplotlib pandas joblib
python3 -c "import sys; print(sys.path)"  # Check Python path
```

### Long Running Calculations
If calculations seem to hang:
- DO NOT cancel - computations can take 30+ minutes
- Check for existing sigma calculations in `sigma_data/sigma_calculations.csv`
- Monitor progress via console output (iterations and temperature points)
- For cluster jobs: check `slurm_logs/batch_JOBID_TASKID.out`

### File Permission Issues
```bash
chmod +x *.sh *.py  # Make all scripts executable
ls -la *.py | head -5  # Verify executable permissions
```

### Memory Issues
- Reduce `mupoints` from 21 to 3-10 for testing
- Reduce frequency resolution with `-wr 1.0` instead of default 0.1
- Use `--no-plot` to avoid memory-intensive plotting

## Codebase Navigation

### Core Calculation Scripts
- `axial_spectra.py` - Main axial meson spectral function calculator
- `vector_spectra.py` - Vector meson spectral functions
- `map_phase_diagram_improved.py` - Phase diagram generation with critical point finding
- `chiral_solve_complete.py` - Solves chiral field equations

### Batch Processing and Analysis
- `batch_phase_diagram_scan.py` - Batch processing for parameter scans
- `run_batch_scan.py` - High-level batch coordination wrapper
- `critical_zoom_improved.py` - High-resolution critical point analysis

### SLURM Cluster Support  
- `submit_array_job.sh` - Intelligent cluster job submission with resource optimization
- `slurm_batch_array.sh` - SLURM job array script template
- `setup_shared_output.sh` - Shared storage configuration

### Test and Validation Scripts
- `test_T17.py` - Tests spectral function calculation at T=17 MeV
- `test_phase_args.py` - Quick argument parsing validation
- `test_*.py` - Various parameter validation tests

### Documentation Files (Reference These)
- `README.md` - Main documentation with usage examples
- `COMMAND_LINE_STANDARDS.md` - Parameter naming conventions
- `SLURM_DEPLOYMENT_GUIDE.md` - Cluster deployment procedures  
- `FILE_NAMING_STANDARDS.md` - Output file naming conventions
- `BATCH_PHASE_DIAGRAM_README.md` - Batch processing documentation

### Always Check These Locations for Changes
- After spectral function changes: `axial_spectra.py`, `vector_spectra.py`
- After phase diagram changes: `map_phase_diagram_improved.py`, `critical_zoom_improved.py`
- After parameter changes: Update all documentation files with new examples
- After file naming changes: Update filename generation in all scripts consistently