# Command-Line Argument Standards

This document defines the standardized command-line argument conventions used across all programs in the meson melting analysis codebase.

## General Principles

1. **Single dash for numeric/scientific parameters**: Use single dash (`-T`, `-mu`, `-mq`) for all physical and numerical parameters
2. **No underscores**: Replace underscores with no separator (`-tmin` not `-t_min`)
3. **Double dash for boolean flags**: Use double dash (`--no-plot`, `--normalize`) for non-numeric boolean options
4. **No positional arguments**: All parameters should be labeled with flags, not positional
5. **Consistent naming**: Use the same parameter names across all programs

## Standard Parameter Names

### Physical Parameters
- `-T` - Temperature in MeV
- `-mu` - Chemical potential in MeV  
- `-mq` - Quark mass (replaces `-ml` or `-l1`)
- `-lambda1` - Lambda1 parameter
- `-lambda4` - Lambda4 parameter
- `-gamma` - Gamma parameter

### Frequency/Omega Parameters
- `-wi` - Initial frequency in MeV
- `-wf` - Final frequency in MeV
- `-wc` - Number of frequency points (omega count)
- `-wr` - Frequency resolution

### Coordinate Parameters
- `-ui` - Initial u coordinate
- `-uf` - Final u coordinate
- `-umin` - Minimum u value
- `-umax` - Maximum u value
- `-upoints` - Number of u points

### Range Parameters
- `-tmin` - Minimum temperature
- `-tmax` - Maximum temperature
- `-tstep` - Temperature step size
- `-mumin` - Minimum chemical potential
- `-mumax` - Maximum chemical potential
- `-mupoints` - Number of chemical potential points

### Search Parameters
- `-minsigma` - Minimum sigma value
- `-maxsigma` - Maximum sigma value
- `-maxiter` - Maximum iterations
- `-numtemp` - Number of temperature points

### Other Parameters
- `-ep` - Expected number of peaks
- `-a0` - Additional parameter a0

### Boolean Flags (Double Dash)
- `--no-plot` - Do not create plots
- `--no-display` - Do not display plots (but save them)
- `--normalize` - Normalize spectrum
- `--skip-existing` - Skip existing files
- `--dry-run` - Show what would be executed without running

### File Output Options
- `--sigma-out` - Path to write sigma value as CSV
- `--output-dir` - Base output directory

## Examples

### Axial Spectra Calculation
```bash
python axial_spectra.py -T 20.0 -mu 50.0 -mq 9.0 -lambda1 7.438 -wi 500 -wf 3000 --normalize
```

### Phase Diagram Generation
```bash
python map_phase_diagram_improved.py -lambda1 7.438 -mq 9.0 -mumin 0.0 -mumax 200.0 -mupoints 20
```

### Temperature Scan
```bash
python run_axial_temperatures.py -tmin 10.0 -tmax 50.0 -tstep 2.0 -mu 0.0 -mq 9.0 -lambda1 7.438
```

### Batch Processing
```bash
python batch_phase_diagrams.py -mq 9.0 -lambda1min 0.0 -lambda1max 10.0 -lambda1points 11

# NEW unified batch scanner
python batch_phase_diagram_unified.py -gammarange -25.0 -20.0 -gammapoints 6 -mq 9.0 -lambda1 5.0 -lambda4 4.2
```

## Migration from Old Syntax

### Common Changes Made
- `--temperature` → `-T`
- `--chemical-potential` → `-mu`
- `--quark-mass` → `-mq`
- `--omega-initial` → `-wi`
- `--omega-final` → `-wf`
- `--omega-count` → `-wc`
- `--omega-resolution` → `-wr`
- `-t_min` → `-tmin`
- `-t_max` → `-tmax`
- `-t_step` → `-tstep`
- `--ml` → `-mq`
- `-l1` → `-lambda1`
- `--mu-min` → `-mumin`
- `--mu-max` → `-mumax`
- `--mu-points` → `-mupoints`

## Files Updated

The following files have been updated to follow these conventions:

### Core Analysis Scripts
- `axial_spectra.py`
- `vector_spectra.py`
- `axial_potential_critical.py`

### Temperature Scanning
- `run_axial_temperatures.py`

### Phase Diagram Tools
- `map_phase_diagram.py`
- `map_phase_diagram_improved.py`
- `batch_phase_diagrams.py`
- `batch_phase_diagram_scan.py`
- `batch_phase_diagram_unified.py` ⭐ **NEW** - Follows all standards

### Batch Processing
- `run_batch_scan.py`

## Benefits

1. **Consistency**: All programs now use the same argument names and patterns
2. **Brevity**: Shorter argument names are faster to type
3. **Clarity**: Single dash for scientific parameters, double dash for flags
4. **Maintainability**: Easier to remember and maintain consistent interfaces
5. **Scriptability**: More predictable for use in batch scripts

## Future Development

When adding new command-line arguments to existing programs or creating new programs:

1. Follow the naming conventions above
2. Use single dashes for all numeric/scientific parameters
3. Use double dashes only for boolean flags and file paths
4. Avoid underscores in parameter names
5. Make all parameters optional with sensible defaults where possible
6. Update this document if new parameter categories are needed
