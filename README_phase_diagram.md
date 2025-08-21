# Phase Diagram Mapper

This script (`map_phase_diagram.py`) calls the `critical_zoom` function to map the QCD phase diagram over a range of chemical potential (μ) values for given λ₁ and quark mass parameters. It saves the critical points to CSV along with the transition order (1 for first order, 2 for crossover).

## Features

- Maps critical temperature as a function of chemical potential
- Automatically detects first-order vs crossover transitions
- Saves results to CSV with detailed metadata
- Creates phase diagram plots
- Robust error handling for failed calculations
- Parallel processing support (inherited from `criticalZoom.py`)

## Usage

### Basic Usage
```bash
python3 map_phase_diagram.py <lambda1> <ml>
```

### Examples

1. **Basic scan with default parameters:**
```bash
python3 map_phase_diagram.py 7.8 24.0
```

2. **Custom chemical potential range:**
```bash
python3 map_phase_diagram.py 7.5 30.0 --mu-min 50 --mu-max 200 --mu-points 25
```

3. **High-resolution scan with custom output:**
```bash
python3 map_phase_diagram.py 8.0 20.0 --mu-points 50 --numtemp 40 -o detailed_scan.csv
```

4. **Quick scan without plotting:**
```bash
python3 map_phase_diagram.py 7.2 25.0 --mu-points 10 --no-plot
```

## Arguments

### Required:
- `lambda1`: λ₁ parameter for mixing between dilaton and chiral field
- `ml`: Light quark mass in MeV

### Optional:
- `--mu-min`: Minimum chemical potential (default: 0.0 MeV)
- `--mu-max`: Maximum chemical potential (default: 200.0 MeV)  
- `--mu-points`: Number of μ points to sample (default: 20)
- `--tmin`: Minimum temperature for search (default: 80.0 MeV)
- `--tmax`: Maximum temperature for search (default: 210.0 MeV)
- `--numtemp`: Temperature points per iteration (default: 25)
- `--minsigma`: Minimum σ value for search (default: 0.0)
- `--maxsigma`: Maximum σ value for search (default: 400.0)
- `--a0`: Additional parameter a₀ (default: 0.0)
- `-o, --output`: Output CSV filename (auto-generated if not specified)
- `--no-plot`: Disable phase diagram plotting

## Output

### CSV File
The output CSV contains the following columns:
- `mu`: Chemical potential (MeV)
- `Tc`: Critical temperature (MeV)
- `order`: Transition order (1=first order, 2=crossover)
- `iterations`: Number of iterations required
- Search parameters: `tmin_search`, `tmax_search`, `numtemp`, `minsigma`, `maxsigma`

### Plot
Automatically generates a phase diagram plot showing:
- Red line with circles: First-order transitions
- Blue line with circles: Crossover transitions  
- Black square: Critical endpoint candidate (if order changes)

## Example Output Structure
```
mu,Tc,order,iterations,tmin_search,tmax_search,numtemp,minsigma,maxsigma
0.0,165.23,2,3,80.0,210.0,25,0.0,400.0
10.53,162.15,2,4,80.0,210.0,25,0.0,400.0
21.05,158.92,1,2,80.0,210.0,25,0.0,400.0
...
```

## Dependencies
- `criticalZoom.py` (contains the `critical_zoom` function)
- `numpy`
- `pandas` 
- `matplotlib`
- `argparse`

## Notes
- The script uses the same temperature and σ search parameters for all μ values
- Failed calculations are recorded with NaN values but don't stop the scan
- Progress is printed to console during execution
- Output files use format: `phase_diagram_ml{ml:.1f}_lambda1{lambda1:.1f}.csv`
- Phase diagram plots are saved as PNG files

## Typical Runtime
- 10 μ points: ~2-5 minutes
- 20 μ points: ~5-10 minutes  
- 50 μ points: ~15-30 minutes

Runtime depends on the complexity of the phase structure and convergence of the `critical_zoom` function.
