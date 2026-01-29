#!/usr/bin/env python3
"""
Plot mass squared vs. n for rho and a_1 mesons with uncertainties.

This script reads experimental meson mass data and plots m² vs. n,
properly propagating uncertainties from mass to mass squared.

For mass m with uncertainty δm, the uncertainty in m² is: δ(m²) = 2m × δm

Usage:
    python plot_meson_mass_squared.py [csv_file]
    
Default data file: simple_axial_scan_20250417_160050/meson experimental data.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
from pathlib import Path

def propagate_uncertainty_to_mass_squared(mass, uncertainty):
    """
    Propagate uncertainty from mass to mass squared.
    
    For m² where m has uncertainty δm:
    δ(m²) = |d(m²)/dm| × δm = |2m| × δm = 2m × δm
    
    Parameters:
    -----------
    mass : float or array
        Meson mass in MeV
    uncertainty : float or array  
        Uncertainty in mass in MeV
        
    Returns:
    --------
    tuple: (mass_squared, mass_squared_uncertainty)
    """
    mass_squared = mass**2
    mass_squared_uncertainty = 2 * mass * uncertainty
    
    return mass_squared, mass_squared_uncertainty

def plot_meson_mass_squared(csv_file, output_file=None, show_plot=True):
    """
    Plot mass squared vs. n for rho and a_1 mesons.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with meson data
    output_file : str, optional
        Output filename for saved plot
    show_plot : bool
        Whether to display the plot interactively
    """
    
    # Load the data
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded meson data from {csv_file}")
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found")
        return
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return
    
    # Extract data for plotting
    n_values = df['n'].values
    
    # Rho meson data
    rho_mass = df['rho mass'].values
    rho_uncertainty = df['rho uncertainty'].values
    
    # a_1 meson data (may have NaN values)
    a1_mass = df['a1 mass'].values
    a1_uncertainty = df['a1 uncertainty'].values
    
    # Remove NaN values for a_1 data
    a1_mask = ~np.isnan(a1_mass)
    n_a1 = n_values[a1_mask]
    a1_mass_clean = a1_mass[a1_mask]
    a1_uncertainty_clean = a1_uncertainty[a1_mask]
    
    # Calculate mass squared and propagate uncertainties
    rho_mass_sq, rho_mass_sq_err = propagate_uncertainty_to_mass_squared(rho_mass, rho_uncertainty)
    a1_mass_sq, a1_mass_sq_err = propagate_uncertainty_to_mass_squared(a1_mass_clean, a1_uncertainty_clean)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot rho meson data (no connecting lines)
    plt.errorbar(n_values, rho_mass_sq, yerr=rho_mass_sq_err, 
                fmt='o', color='red', linewidth=2, markersize=8,
                label=r'$\rho$ meson', capsize=5, capthick=2)
    
    # Plot a_1 meson data (no connecting lines)
    plt.errorbar(n_a1, a1_mass_sq, yerr=a1_mass_sq_err,
                fmt='s', color='blue', linewidth=2, markersize=8, 
                label=r'$a_1$ meson', capsize=5, capthick=2)
    
    # Formatting
    plt.xlabel('n', fontsize=14, fontweight='bold')
    plt.ylabel(r'$m^2$ (MeV$^2$)', fontsize=14, fontweight='bold')
    plt.title('Meson Mass Squared vs. Radial Quantum Number', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Format axes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add some styling
    plt.tight_layout()
    
    # Print summary statistics
    print("\n=== Mass Squared Analysis ===")
    print(f"Rho meson (n=1 to {len(n_values)}):")
    print(f"  Mass² range: {rho_mass_sq.min():.0f} - {rho_mass_sq.max():.0f} MeV²")
    print(f"  Average uncertainty: {rho_mass_sq_err.mean():.0f} MeV²")
    
    print(f"\na₁ meson (n=1 to {len(n_a1)}):")  
    print(f"  Mass² range: {a1_mass_sq.min():.0f} - {a1_mass_sq.max():.0f} MeV²")
    print(f"  Average uncertainty: {a1_mass_sq_err.mean():.0f} MeV²")
    
    # Calculate and display linear fits (using only n >= 3)
    print(f"\n=== Linear Fit Analysis (n ≥ 3) ===")
    
    # Filter rho data for n >= 3
    rho_fit_mask = n_values >= 3
    n_rho_fit = n_values[rho_fit_mask]
    rho_mass_sq_fit = rho_mass_sq[rho_fit_mask]
    rho_mass_sq_err_fit = rho_mass_sq_err[rho_fit_mask]
    
    # Filter a1 data for n >= 3
    a1_fit_mask = n_a1 >= 3
    n_a1_fit = n_a1[a1_fit_mask]
    a1_mass_sq_fit = a1_mass_sq[a1_fit_mask]
    a1_mass_sq_err_fit = a1_mass_sq_err[a1_fit_mask]
    
    # Fit rho data (n >= 3): m² = a*n + b
    rho_fit = np.polyfit(n_rho_fit, rho_mass_sq_fit, 1, w=1/rho_mass_sq_err_fit)
    rho_slope, rho_intercept = rho_fit
    print(f"Rho: m² = {rho_slope:.0f}n + {rho_intercept:.0f}")
    
    # Fit a1 data (n >= 3)
    a1_fit = np.polyfit(n_a1_fit, a1_mass_sq_fit, 1, w=1/a1_mass_sq_err_fit)
    a1_slope, a1_intercept = a1_fit
    print(f"a₁:  m² = {a1_slope:.0f}n + {a1_intercept:.0f}")
    
    # Add fit lines to plot (but don't show equations in legend)
    n_fit = np.linspace(3, max(n_values), 100)
    plt.plot(n_fit, rho_slope * n_fit + rho_intercept, '--', 
             color='red', alpha=0.7, linewidth=1.5)
    
    n_fit_a1 = np.linspace(3, max(n_a1), 100) 
    plt.plot(n_fit_a1, a1_slope * n_fit_a1 + a1_intercept, '--',
             color='blue', alpha=0.7, linewidth=1.5)
    
    # Save plot if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory
    
    return {
        'rho_mass_squared': rho_mass_sq,
        'rho_mass_squared_err': rho_mass_sq_err,
        'a1_mass_squared': a1_mass_sq,
        'a1_mass_squared_err': a1_mass_sq_err,
        'rho_fit': (rho_slope, rho_intercept),
        'a1_fit': (a1_slope, a1_intercept)
    }

def main():
    parser = argparse.ArgumentParser(
        description="Plot mass squared vs. n for rho and a_1 mesons with uncertainties"
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default='simple_axial_scan_20250417_160050/meson experimental data.csv',
        help='CSV file with meson data (default: simple_axial_scan_20250417_160050/meson experimental data.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output filename for saved plot (e.g., meson_mass_squared.png)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true', 
        help='Do not display plot interactively'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.csv_file).exists():
        print(f"Error: Input file {args.csv_file} does not exist")
        
        # Look for CSV files in current directory
        csv_files = list(Path('.').glob('**/*.csv'))
        if csv_files:
            print("\nAvailable CSV files:")
            for f in csv_files[:10]:  # Show first 10
                print(f"  - {f}")
            if len(csv_files) > 10:
                print(f"  ... and {len(csv_files)-10} more")
        sys.exit(1)
    
    # Generate default output filename if saving
    output_file = args.output
    if args.output is None and args.no_show:
        output_file = "meson_mass_squared_vs_n.png"
    
    # Create the plot
    results = plot_meson_mass_squared(
        args.csv_file, 
        output_file=output_file,
        show_plot=not args.no_show
    )
    
    if results:
        print(f"\n✓ Analysis complete!")
        if output_file:
            print(f"✓ Plot saved to {output_file}")

if __name__ == "__main__":
    main()
