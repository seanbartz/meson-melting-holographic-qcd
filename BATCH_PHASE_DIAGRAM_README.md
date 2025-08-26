# Batch Phase Diagram Scanner Documentation

This directory contains scripts for running batch scans of phase diagrams across ranges of parameters and creating combined plots.

## Scripts

### 1. `batch_phase_diagram_scan.py`
Main script that runs `map_phase_diagram_improved.py` for multiple parameter values and creates combined plots.

### 2. `run_batch_scan.py`
Helper script with predefined common scan configurations.

## Usage Examples

### Standard Scans

```bash
# Run a gamma scan from -25.0 to -20.0 with 6 points
python run_batch_scan.py gamma-scan -lambda1 5.0 -mq 9.0

# Run a lambda4 scan from 3.0 to 5.5 with 6 points
python run_batch_scan.py lambda4-scan -lambda1 5.0 -mq 9.0

# Add options to standard scans
python run_batch_scan.py gamma-scan -lambda1 5.0 -mq 9.0 --mu-points 30 --skip-existing
```

### Custom Scans

```bash
# Custom gamma values
python run_batch_scan.py custom --parameter gamma --values -25.0 -22.6 -20.0 -lambda1 5.0 -mq 9.0

# Custom lambda4 values
python run_batch_scan.py custom --parameter lambda4 --values 3.0 4.2 5.0 5.5 -lambda1 5.0 -mq 9.0
```

### Direct Usage of Main Script

```bash
# Scan gamma using range specification
python batch_phase_diagram_scan.py --parameter gamma --range -25.0 -20.0 --num-values 6 --lambda1 5.0 --ml 9.0

# Scan lambda4 with explicit values
python batch_phase_diagram_scan.py --parameter lambda4 --values 3.0 4.2 5.5 --lambda1 5.0 --ml 9.0

# Advanced options
python batch_phase_diagram_scan.py --parameter gamma --range -25.0 -20.0 --num-values 8 \
    --lambda1 5.0 --ml 9.0 --mu-points 25 --tmin 70 --tmax 220 --skip-existing
```

## Output Files

### Data Files
- Individual CSV files saved in `phase_data/` directory
- Naming convention: `phase_diagram_improved_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv`

### Plot Files  
All plots saved in `phase_plots/` directory:

1. **Combined Phase Diagram**: `combined_phase_diagram_{parameter}_scan_ml_{ml:.1f}_lambda1_{lambda1:.1f}.png`
   - Shows all critical lines on one plot with different colors for each parameter value
   - Includes axial and vector melting lines if available
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
