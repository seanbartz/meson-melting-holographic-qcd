#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Diagram Mapper

This script calls the critical_zoom function to map the phase diagram over a range 
of chemical potential (mu) values for given input lambda1 and ml (quark mass).
Saves the critical points to CSV along with the order (1 for first order, 2 for crossover).

Created on July 15, 2025
@author: GitHub Copilot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Import the critical_zoom function from criticalZoom.py
from criticalZoom import critical_zoom

def map_phase_diagram(mu_min, mu_max, mu_points, lambda1, ml, 
                     tmin=80, tmax=210, numtemp=25, 
                     minsigma=0, maxsigma=400, a0=0.0,
                     gamma=-22.4, lambda4=4.2,
                     output_file=None, plot_results=True, display_plot=True):
    """
    Map the phase diagram over a range of chemical potential values.
    
    Args:
        mu_min: Minimum chemical potential (MeV)
        mu_max: Maximum chemical potential (MeV) 
        mu_points: Number of mu points to sample
        lambda1: Lambda1 parameter for mixing between dilaton and chiral field
        ml: Light quark mass (MeV)
        tmin: Minimum temperature for search (MeV)
        tmax: Maximum temperature for search (MeV)
        numtemp: Number of temperature points per iteration
        minsigma: Minimum sigma value for search
        maxsigma: Maximum sigma value for search
        a0: Additional parameter (default: 0.0)
        output_file: Output CSV filename (if None, auto-generated)
        plot_results: Whether to create a phase diagram plot
        display_plot: Whether to display the plot (default: True)
        
    Returns:
        DataFrame with critical points and phase transition information
    """
    
    # Create mu array
    mu_values = np.linspace(mu_min, mu_max, mu_points)
    
    # Lists to store results
    results = []
    
    print(f"Mapping phase diagram for ml={ml} MeV, lambda1={lambda1}")
    print(f"Chemical potential range: {mu_min} to {mu_max} MeV ({mu_points} points)")
    print("=" * 60)
    
    for i, mu in enumerate(mu_values):
        print(f"\nProgress: {i+1}/{mu_points} - Processing mu = {mu:.2f} MeV")
        
        try:
            # Call critical_zoom function
            order, iterationNumber, sigma_list, temps_list, Tc = critical_zoom(
                tmin, tmax, numtemp, minsigma, maxsigma, ml, mu, lambda1, a0
            )
            
            # Store results
            result_dict = {
                'mu': mu,
                'Tc': Tc,
                'order': order,
                'iterations': iterationNumber,
                'tmin_search': tmin,
                'tmax_search': tmax,
                'numtemp': numtemp,
                'minsigma': minsigma,
                'maxsigma': maxsigma
            }
            
            results.append(result_dict)
            
            # Print result for this mu
            order_str = "First order" if order == 1 else "Crossover/2nd order"
            print(f"  Result: Tc = {Tc:.2f} MeV, Order = {order} ({order_str})")
            print(f"  Iterations: {iterationNumber}")
            
        except Exception as e:
            print(f"  ERROR: Failed to process mu = {mu:.2f} MeV: {str(e)}")
            # Store error result
            result_dict = {
                'mu': mu,
                'Tc': np.nan,
                'order': np.nan,
                'iterations': np.nan,
                'tmin_search': tmin,
                'tmax_search': tmax,
                'numtemp': numtemp,
                'minsigma': minsigma,
                'maxsigma': maxsigma
            }
            results.append(result_dict)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure output directories exist
    data_dir = 'phase_data'
    plot_dir = 'phase_plots'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Generate output filename if not provided
    if output_file is None:
        output_file = os.path.join(data_dir, f"phase_diagram_mq_{ml:.1f}_lambda1_{lambda1:.1f}.csv")

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Total points processed: {len(df)}")
    successful = df['order'].notna().sum()
    print(f"Successful calculations: {successful}")
    if successful > 0:
        first_order = (df['order'] == 1).sum()
        crossover = (df['order'] == 2).sum()
        print(f"First order transitions: {first_order}")
        print(f"Crossover/2nd order: {crossover}")
        print(f"Critical temperature range: {df['Tc'].min():.2f} - {df['Tc'].max():.2f} MeV")
    
    # Create phase diagram plot if requested
    if plot_results and successful > 0:
        create_phase_diagram_plot(df, lambda1, ml, plot_dir, display_plot)
    
    return df

def create_phase_diagram_plot(df, lambda1, ml, plot_dir, display_plot=True):
    """
    Create a phase diagram plot showing the critical line.
    
    Args:
        df: DataFrame with phase diagram data
        lambda1: Lambda1 parameter value
        ml: Quark mass value
        plot_dir: Directory to save the plot
        display_plot: Whether to display the plot (default: True)
    """
    # Filter out failed calculations
    df_valid = df[df['order'].notna()].copy()
    
    if len(df_valid) == 0:
        print("No valid data points for plotting")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Load and plot vector melting data if it exists
    try:
        vector_data = np.loadtxt('vector_melting.csv', delimiter=',', skiprows=1)
        mu_vector = vector_data[:, 0]
        T_vector = vector_data[:, 1]
        plt.plot(mu_vector, T_vector, color='red', linestyle='--', label='Vector Melting Line', linewidth=2)
    except (FileNotFoundError, OSError):
        print("Vector melting data file not found, skipping vector melting line")
    
    # Load and plot axial melting data if it exists
    try:
        axial_filename = f'axial_melting_data_mq_{ml:.1f}_lambda1_{lambda1:.1f}.csv'
        axial_data = np.loadtxt(axial_filename, delimiter=',', skiprows=4)
        mu_axial = axial_data[:, 0]
        T_axial = axial_data[:, 1]
        plt.plot(mu_axial, T_axial, color='black', label='Axial Melting Line', linewidth=2)
    except (FileNotFoundError, OSError):
        print(f"Axial melting data file {axial_filename} not found, skipping axial melting line")
    
    # Separate first order and crossover points
    first_order = df_valid[df_valid['order'] == 1]
    crossover = df_valid[df_valid['order'] == 2]
    
    # Plot critical line
    if len(first_order) > 0:
        plt.plot(first_order['mu'], first_order['Tc'], 'ro-', 
                label='First order transition', linewidth=2, markersize=6)
    
    if len(crossover) > 0:
        plt.plot(crossover['mu'], crossover['Tc'], 'bo-', 
                label='Crossover/2nd order', linewidth=2, markersize=6)
    
    plt.xlabel('Chemical Potential μ (MeV)', fontsize=12)
    plt.ylabel('Critical Temperature Tc (MeV)', fontsize=12)
    plt.title(f'QCD Phase Diagram\n$m_q = {ml}$ MeV, $\\lambda_1 = {lambda1:.3f}$', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add text annotations for critical endpoint if transition order changes
    if len(first_order) > 0 and len(crossover) > 0:
        # Find potential critical endpoint (transition between orders)
        all_data = df_valid.sort_values('mu')
        order_changes = np.where(np.diff(all_data['order']) != 0)[0]
        if len(order_changes) > 0:
            cep_idx = order_changes[0]
            cep_mu = all_data.iloc[cep_idx]['mu']
            cep_tc = all_data.iloc[cep_idx]['Tc']
            plt.plot(cep_mu, cep_tc, 'ks', markersize=10, 
                    label='Critical endpoint candidate')
            plt.annotate(f'CEP?\n(μ={cep_mu:.1f}, T={cep_tc:.1f})', 
                        xy=(cep_mu, cep_tc), xytext=(10, 10),
                        textcoords='offset points', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(plot_dir, f"phase_diagram_mq_{ml:.1f}_lambda1_{lambda1:.1f}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Phase diagram plot saved to: {plot_filename}")
    
    # Only display plot if requested
    if display_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Map QCD phase diagram using critical_zoom function')
    
    # Required arguments
    parser.add_argument('-lambda1', type=float, required=True, help='Lambda1 parameter for mixing between dilaton and chiral field')
    parser.add_argument('-mq', type=float, required=True, help='Light quark mass in MeV')
    
    # Chemical potential range
    parser.add_argument('-mumin', type=float, default=0.0, help='Minimum chemical potential in MeV (default: 0.0)')
    parser.add_argument('-mumax', type=float, default=200.0, help='Maximum chemical potential in MeV (default: 200.0)')
    parser.add_argument('-mupoints', type=int, default=20, help='Number of mu points to sample (default: 20)')
    
    # Temperature search parameters
    parser.add_argument('-tmin', type=float, default=80.0, help='Minimum temperature for search in MeV (default: 80.0)')
    parser.add_argument('-tmax', type=float, default=210.0, help='Maximum temperature for search in MeV (default: 210.0)')
    parser.add_argument('-numtemp', type=int, default=25, help='Number of temperature points per iteration (default: 25)')
    
    # Sigma search parameters  
    parser.add_argument('-minsigma', type=float, default=0.0, help='Minimum sigma value for search (default: 0.0)')
    parser.add_argument('-maxsigma', type=float, default=400.0, help='Maximum sigma value for search (default: 400.0)')
    parser.add_argument('-a0', type=float, default=0.0, help='Additional parameter a0 (default: 0.0)')
    
    # Output options
    parser.add_argument('-o', type=str, help='Output CSV filename (if not specified, auto-generated)')
    parser.add_argument('--no-plot', action='store_true', help='Do not create phase diagram plot')
    parser.add_argument('--no-display', action='store_true', help='Do not display plot (still saves plot file)')
    
    args = parser.parse_args()
    
    # Run phase diagram mapping
    df = map_phase_diagram(
        mu_min=args.mumin,
        mu_max=args.mumax, 
        mu_points=args.mupoints,
        lambda1=args.lambda1,
        ml=args.mq,
        tmin=args.tmin,
        tmax=args.tmax,
        numtemp=args.numtemp,
        minsigma=args.minsigma,
        maxsigma=args.maxsigma,
        a0=args.a0,
        output_file=args.o,
        plot_results=not args.no_plot,
        display_plot=not args.no_display
    )
    
    return df

if __name__ == '__main__':
    df = main()
