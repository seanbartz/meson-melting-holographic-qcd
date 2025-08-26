# Improved Phase Diagram Mapper

This script (`map_phase_diagram_improved.py`) calls the `critical_zoom_improved` function to map the QCD phase diagram over a range of chemical potential (μ) values for given λ₁ and quark mass parameters. It saves the critical points to CSV along with the transition order (1 for first order, 2 for crossover).

## Features

- Maps critical temperature as a function of chemical potential
- Automatically detects first-order vs crossover transitions
- Saves results to CSV with detailed metadata
- Creates phase diagram plots
- Robust error handling for failed calculations
- Efficient top-level parallel processing (one process per CPU)

## Usage

### Basic Usage (all parameters are labeled; no positional arguments)
```bash
python3 map_phase_diagram_improved.py -lambda1 <value> -mq <value>
```

### Examples

1. **Basic scan with default parameters:**
```bash
python3 map_phase_diagram_improved.py -lambda1 7.8 -mq 24.0
```

2. **Custom chemical potential range:**
```bash
python3 map_phase_diagram_improved.py -lambda1 7.5 -mq 30.0 -mumin 50 -mumax 200 -mupoints 25
```

3. **High-resolution scan with custom output:**
```bash
python3 map_phase_diagram_improved.py -lambda1 8.0 -mq 20.0 -mupoints 50 -maxiterations 12 -o phase_data/detailed_scan.csv
```

4. **Quick scan without plotting:**
```bash
python3 map_phase_diagram_improved.py -lambda1 7.2 -mq 25.0 -mupoints 10 --no-plot --no-display
```

5. **Override model parameters (γ, λ4):**
```bash
python3 map_phase_diagram_improved.py -lambda1 5.3 -mq 9.0 -gamma -22.6 -lambda4 4.2
```

## Arguments

### Required:
- `-lambda1`: λ₁ parameter for mixing between dilaton and chiral field
- `-mq`: Quark mass in MeV

### Optional:
- `-mumin`: Minimum chemical potential (default: 0.0 MeV)
- `-mumax`: Maximum chemical potential (default: 200.0 MeV)
- `-mupoints`: Number of μ points to sample (default: 20)
- `-tmin`: Minimum temperature for search (default: 80.0 MeV)
- `-tmax`: Maximum temperature for search (default: 210.0 MeV)
- `-ui`: Lower integration bound (default: 1e-2)
- `-uf`: Upper integration bound (default: 1-1e-4)
- `-d0lower`: Lower bound for d0 search (default: 0.0)
- `-d0upper`: Upper bound for d0 search (default: 10.0)
- `-mqtolerance`: Tolerance for quark mass matching (default: 0.01)
- `-maxiterations`: Maximum number of zoom iterations (default: 10)
- `-gamma`: Background metric parameter γ (default: -22.4)
- `-lambda4`: Fourth-order coupling parameter λ₄ (default: 4.2)
- `-o`: Output CSV filename (auto-generated if not specified)
- `--no-plot`: Do not create phase diagram plot
- `--no-display`: Do not display plot (still saves plot file)
- `--compare`: Compare with original method results if available

## Output

### CSV File
The output CSV contains at least the following columns:
- `mu`: Chemical potential (MeV)
- `Tc`: Critical temperature (MeV)
- `order`: Transition order (1=first order, 2=crossover)
- `iterations`: Number of iterations
- Search metadata: `tmin_search`, `tmax_search`, `numtemp_per_iter`, `total_temp_points`
- Integration/search parameters: `ui`, `uf`, `d0_lower`, `d0_upper`, `mq_tolerance`, `max_iterations`
- Model parameters and bookkeeping: `ml`, `lambda1`, `gamma`, `lambda4`, `adaptive_bounds_used`

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
- `critical_zoom_improved.py` (contains the `critical_zoom_improved` function)
- `numpy`
- `pandas` 
- `matplotlib`
- `argparse`

## Notes
- The script adaptively adjusts temperature bounds based on previous results and uses one process per CPU
- Failed calculations are recorded with NaN values but don't stop the scan
- Progress is printed to console during execution
- Output files use format: `phase_diagram_improved_mq_{mq:.1f}_lambda1_{lambda1:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv` and are saved in `phase_data/`
- Phase diagram plots are saved as PNG files in `phase_plots/`

## Typical Runtime
- 10 μ points: ~2-5 minutes
- 20 μ points: ~5-10 minutes  
- 50 μ points: ~15-30 minutes

Runtime depends on the complexity of the phase structure and convergence of the `critical_zoom_improved` function.
