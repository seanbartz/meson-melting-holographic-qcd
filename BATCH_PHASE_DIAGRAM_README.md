# Batch Phase Diagram Scanner Documentation

This directory contains scripts for running batch scans of phase diagrams across ranges of parameters and creating combined plots.

## Scripts

### 1. `batch_phase_diagram_unified.py` ⭐ **RECOMMENDED**
**New unified script** that can scan over any combination of parameters (mq, lambda1, gamma, lambda4). This replaces the older separate scripts and provides much more flexibility.

### 2. `batch_phase_diagram_scan.py` (Legacy)
Original script that runs `map_phase_diagram_improved.py` for gamma or lambda4 parameter values only.

### 3. `batch_phase_diagrams.py` (Legacy) 
Original script that runs `map_phase_diagram.py` for lambda1 parameter values only.

### 4. `run_batch_scan.py`
Helper script with predefined common scan configurations (works with legacy scripts).

## Usage Examples - Unified Script

### Single Parameter Scans

```bash
# Scan gamma while keeping other parameters fixed (lambda4 uses default 4.2)
python batch_phase_diagram_unified.py -gammarange -25.0 -20.0 -gammapoints 6 -mq 9.0 -lambda1 5.0

# Scan mq with explicit values (gamma and lambda4 use defaults)
python batch_phase_diagram_unified.py -mqvalues 9.0 12.0 15.0 -lambda1 5.0

# Scan lambda1 using range (gamma and lambda4 use defaults)
python batch_phase_diagram_unified.py -lambda1range 3.0 7.0 -lambda1points 5 -mq 9.0

# Minimal usage - only required parameters
python batch_phase_diagram_unified.py -mq 9.0 -lambda1 5.0
```

### Multi-Parameter Scans (Cartesian Product)

```bash
# Scan both mq and gamma (lambda4 uses default 4.2)
python batch_phase_diagram_unified.py -mqvalues 9.0 12.0 -gammavalues -25.0 -22.4 -lambda1 5.0

# Range scan for lambda1 and explicit values for lambda4 (gamma uses default -22.4)
python batch_phase_diagram_unified.py -lambda1range 3.0 7.0 -lambda1points 4 -lambda4values 3.0 4.2 5.0 -mq 9.0
```

### Advanced Options

```bash
# Custom physical parameter ranges (gamma and lambda4 use defaults)
python batch_phase_diagram_unified.py -gammarange -25.0 -20.0 -gammapoints 8 \
    -mq 9.0 -lambda1 5.0 -mumax 300.0 -mupoints 30 -tmin 70 -tmax 220

# Custom output directories (minimal parameters)
python batch_phase_diagram_unified.py -mqvalues 9.0 12.0 15.0 -lambda1 5.0 \
    --output-dir my_phase_data --plot-dir my_plots
```

## Legacy Usage Examples

### Standard Scans (Legacy)

```bash
# Run a gamma scan from -25.0 to -20.0 with 6 points
python run_batch_scan.py gamma-scan -lambda1 5.0 -mq 9.0

# Run a lambda4 scan from 3.0 to 5.5 with 6 points
python run_batch_scan.py lambda4-scan -lambda1 5.0 -mq 9.0
```

## Output Files

### Data Files
- Individual CSV files saved in `phase_data/` directory (or custom `--output-dir`)
- **New naming convention**: `phase_diagram_improved_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv`
- **All parameters included in filename** for easy identification and organization

### Plot Files  
All plots saved in `phase_plots/` directory (or custom `--plot-dir`):

1. **Combined Phase Diagram**: Shows phase boundaries for all parameter combinations
   - Automatic filename based on varying parameters, e.g.:
     - `phase_diagram_combined_gamma_-25.0to-20.0.png` (gamma scan)
     - `phase_diagram_combined_mq_scan_lambda1_scan.png` (multi-parameter scan)
     - `phase_diagram_single.png` (single parameter combination)
   - Different colors/markers for each parameter combination
   - Includes legend with parameter values
   - **Files are overwritten** on subsequent runs with same parameters

## Migration from Legacy Scripts

The new unified script `batch_phase_diagram_unified.py` replaces both legacy scripts:

| Legacy Script | Legacy Usage | New Unified Usage |
|---------------|--------------|-------------------|
| `batch_phase_diagram_scan.py --parameter gamma --range -25.0 -20.0 --num-values 6` | | `-gammarange -25.0 -20.0 -gammapoints 6` |
| `batch_phase_diagrams.py` (lambda1 scan) | | `-lambda1range 3.0 7.0 -lambda1points 5` |

## Parameter Specification Syntax

For each parameter (mq, lambda1, gamma, lambda4), you can specify:

1. **Single value**: `-mq 9.0`
2. **Explicit values**: `-mqvalues 9.0 12.0 15.0` 
3. **Range**: `-mqrange 9.0 15.0 -mqpoints 4` (creates 4 evenly spaced points)

If a parameter is not specified, default values are used:
- mq: 9.0 MeV
- lambda1: 5.0  
- gamma: -22.4 (optional)
- lambda4: 4.2 (optional)

**Note**: `gamma` and `lambda4` are optional parameters. If not specified, they use standard holographic QCD values that work well for most calculations.
   - Distinguishes first-order transitions (solid lines) from crossovers (dashed lines)

2. **Summary Critical Lines**: `summary_critical_lines_{parameter}_scan_ml_{ml:.1f}_lambda1_{lambda1:.1f}.png`
   - Clean plot showing just the critical temperature lines
   - One line per parameter value

3. **Parameter Evolution**: `parameter_evolution_{parameter}_scan_ml_{ml:.1f}_lambda1_{lambda1:.1f}.png`
   - Shows how critical temperature changes with the parameter at fixed μ values
   - Multiple curves for different μ values (0, 50, 100, 150, 200 MeV)

## Command Line Options

### Required Arguments
- `--lambda1`: Lambda1 parameter value
- `--ml`: Quark mass in MeV
- `--parameter`: Parameter to scan (`gamma` or `lambda4`)
- Parameter values: Either `--values` (explicit list) or `--range` + `--num-values`

### Optional Arguments
- `--mu-min`, `--mu-max`, `--mu-points`: Chemical potential range and sampling
- `--tmin`, `--tmax`: Temperature search range  
- `--max-iterations`: Maximum iterations for critical point finding
- `--gamma-fixed`: Fixed gamma when scanning lambda4 (default: -22.4)
- `--lambda4-fixed`: Fixed lambda4 when scanning gamma (default: 4.2)
- `--skip-existing`: Skip calculations if output file already exists
- `--no-axial`: Don't show axial melting lines in plots
- `--no-vector`: Don't show vector melting line in plots

## Workflow

1. **Parameter Scan**: Script runs `map_phase_diagram_improved.py` for each parameter value
2. **Data Collection**: Loads all generated CSV files
3. **Plot Generation**: Creates three types of combined plots
4. **No Display**: All plots are saved but not displayed (suitable for batch processing)

## Performance Tips

- Use `--skip-existing` to avoid recalculating existing data
- Start with fewer `--mu-points` for testing, increase for final runs
- Monitor the `phase_data/` directory to track progress
- Each individual calculation can take 10-60 minutes depending on parameters

## Example Workflow

```bash
# 1. Quick test with few points
python run_batch_scan.py gamma-scan --lambda1 5.0 --ml 9.0 --mu-points 10

# 2. Full calculation if test looks good
python run_batch_scan.py gamma-scan --lambda1 5.0 --ml 9.0 --mu-points 25 --skip-existing

# 3. Check plots in phase_plots/ directory
ls phase_plots/

# 4. Run lambda4 scan with same parameters
python run_batch_scan.py lambda4-scan --lambda1 5.0 --ml 9.0 --mu-points 25 --skip-existing
```

## Notes

- All plots use LaTeX formatting for mathematical symbols
- Colors are chosen from matplotlib colormaps for good contrast  
- Scripts are designed to run without user interaction (suitable for cluster computing)
- Error handling includes timeouts for long-running calculations
- Progress is printed to stdout for monitoring
