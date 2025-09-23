#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Critical Point Zoom Script
Created on July 22, 2025

This script zooms in on the critical point using the calculate_sigma_values function
from chiral_solve_complete.py instead of the manual sigma search in the original criticalZoom.py.

This approach is more robust and accurate for finding sigma values and determining
phase transition characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import pandas as pd
import warnings
from chiral_solve_complete import calculate_sigma_values
import time
from contextlib import contextmanager

@contextmanager
def suppress_joblib_warnings():
    """Context manager to suppress joblib parallel warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", 
                              message=".*Loky-backed parallel loops cannot be called.*",
                              category=UserWarning)
        yield

def get_sigma_values_at_temperature(args):
    """
    Get sigma values at a specific temperature using the improved calculate_sigma_values function.
    
    Args:
        args: tuple containing (T, mu, mq, mq_tolerance, lambda1, ui, uf, d0_lower, d0_upper, v3, v4)
    
    Returns:
        Array of sigma values (up to 3 values, padded with zeros if fewer solutions)
    """
    T, mu, mq, mq_tolerance, lambda1, ui, uf, d0_lower, d0_upper, v3, v4 = args
    
    try:
        # Use the improved sigma calculation with n_jobs=1 to avoid nested parallelization
        # This forces sequential processing within each temperature calculation
        with suppress_joblib_warnings():
            # Temporarily override the internal parallelization to use only 1 job
            import os
            original_joblib_jobs = os.environ.get('JOBLIB_N_JOBS', None)
            os.environ['JOBLIB_N_JOBS'] = '1'
            
            result = calculate_sigma_values(mq, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower, d0_upper, v3, v4)
            sigma_values = result["sigma_values"]
            
            # Restore original setting
            if original_joblib_jobs is not None:
                os.environ['JOBLIB_N_JOBS'] = original_joblib_jobs
            else:
                os.environ.pop('JOBLIB_N_JOBS', None)
        
        # Convert to cube roots for comparison with original script
        sigma_cube_roots = []
        for sigma in sigma_values:
            if sigma >= 0:
                sigma_cube_roots.append(sigma**(1/3))
            else:
                sigma_cube_roots.append(-((-sigma)**(1/3)))
        
        # Pad with zeros to ensure we have exactly 3 values
        while len(sigma_cube_roots) < 3:
            sigma_cube_roots.append(0)
        
        # Take only the first 3 values if we have more
        sigma_cube_roots = sigma_cube_roots[:3]
        
        return np.array(sigma_cube_roots)
        
    except Exception as e:
        print(f"Error calculating sigma values at T={T}: {str(e)}")
        return np.array([0, 0, 0])

def get_all_sigmas_parallel(input_args, pool):
    """
    Calculate sigma values for all temperatures in parallel.
    """
    truesigma = pool.map(get_sigma_values_at_temperature, input_args)
    return np.array(truesigma)

def order_checker(tmin, tmax, numtemp, mq, mu, lambda1, ui, uf, d0_lower=0, d0_upper=10, mq_tolerance=0.1, v3=-22.6/(6*np.sqrt(2)), v4=4.2):
    """
    Check the order of phase transition by analyzing sigma values across temperature range.
    
    Args:
        tmin, tmax: Temperature range
        numtemp: Number of temperature points
        mq: Quark mass in MeV
        mu: Chemical potential in MeV
        lambda1: Lambda1 parameter
        ui, uf: Integration bounds
        d0_lower, d0_upper: Search bounds for d0
        mq_tolerance: Tolerance for quark mass matching
        v3: Chiral parameter v3 (default: -22.6/(6*sqrt(2)))
        v4: Chiral parameter v4 (default: 4.2)
    
    Returns:
        Updated bounds and transition information
    """
    temps = np.linspace(tmin, tmax, numtemp)
    
    # Prepare arguments for parallel computation
    input_args = [(T, mu, mq, mq_tolerance, lambda1, ui, uf, d0_lower, d0_upper, v3, v4) for T in temps]
    
    # Create a pool that uses all available cpus for top-level parallelization
    # This parallelizes across temperatures, while forcing sequential processing
    # within each temperature calculation to avoid nested parallelization overhead
    processes_count = os.cpu_count()
    processes_pool = Pool(processes_count)
    
    print(f"Calculating sigma values for {numtemp} temperatures from {tmin:.2f} to {tmax:.2f} MeV...")
    start_time = time.time()
    
    truesigma = get_all_sigmas_parallel(input_args, processes_pool)
    processes_pool.close()
    processes_pool.join()
    
    end_time = time.time()
    print(f"Calculation completed in {end_time - start_time:.2f} seconds")
    
    # Remove spurious values: if any values of truesigma[:,1] or truesigma[:,2] 
    # are greater than the maximum value of truesigma[:,0], set them to zero
    max_sigma0 = np.max(truesigma[:, 0])
    truesigma[truesigma[:, 1] > max_sigma0, 1] = 0
    truesigma[truesigma[:, 2] > max_sigma0, 2] = 0
    
    # Check if we have multiple solutions (indicating first-order transition)
    if np.max(truesigma[:, 1]) == 0:
        print("Crossover or 2nd order transition detected")
        
        # Find the temperature where the gradient of sigma^3 is most negative
        # This avoids accidentally identifying where sigma goes to zero as critical
        valid_indices = truesigma[:, 0] > 0
        if np.any(valid_indices):
            valid_sigma = truesigma[valid_indices, 0]
            valid_temps = temps[valid_indices]
            
            if len(valid_sigma) > 2:
                transition_index = np.argmin(np.gradient(valid_sigma**3))
                Tc = valid_temps[transition_index]
                
                # Set new bounds with buffer
                buffer = min(2, len(valid_temps) // 4)
                tmin_new = valid_temps[max(transition_index - buffer, 0)]
                tmax_new = valid_temps[min(transition_index + buffer, len(valid_temps) - 1)]
                
                print(f"Pseudo-critical temperature is between {tmin_new:.3f} and {tmax_new:.3f} MeV")
            else:
                # Not enough points for gradient calculation
                Tc = np.mean(valid_temps)
                tmin_new = tmin
                tmax_new = tmax
        else:
            # No valid sigma values found
            Tc = (tmin + tmax) / 2
            tmin_new = tmin
            tmax_new = tmax
        
        # Update sigma bounds
        maxsigma_new = np.max(truesigma) + 1
        minsigma_new = np.min(truesigma[truesigma > 0]) if np.any(truesigma > 0) else 0
        
        order = 2
    else:
        print("First order transition detected")
        # Critical temperature is the lowest temperature where we have three non-zero sigma values
        # This corresponds to the onset of the first-order transition
        
        # Find all temperatures where we have at least two non-zero sigma values (indicating multiple solutions)
        multiple_solutions_mask = truesigma[:, 1] > 0
        
        if np.any(multiple_solutions_mask):
            # Find temperatures with multiple solutions
            multiple_solution_temps = temps[multiple_solutions_mask]
            multiple_solution_sigmas = truesigma[multiple_solutions_mask]
            
            # Look for the lowest temperature where we have three non-zero sigma values
            three_sigma_mask = (multiple_solution_sigmas[:, 0] > 0) & \
                              (multiple_solution_sigmas[:, 1] > 0) & \
                              (multiple_solution_sigmas[:, 2] > 0)
            
            if np.any(three_sigma_mask):
                # Use the lowest temperature with three non-zero sigma values
                Tc = multiple_solution_temps[three_sigma_mask][0]  # First (lowest) temperature
                print(f"Critical temperature is {Tc:.3f} MeV (lowest T with 3 sigma values)")
            else:
                # Fallback: use the lowest temperature with multiple solutions
                Tc = multiple_solution_temps[0]  # First (lowest) temperature
                print(f"Critical temperature is {Tc:.3f} MeV (lowest T with multiple solutions)")
        else:
            # This shouldn't happen if we detected first order, but just in case
            first_order_index = np.argmax(truesigma[:, 1])
            Tc = temps[first_order_index]
            print(f"Critical temperature is {Tc:.3f} MeV (fallback method)")
        
        # Update sigma bounds
        maxsigma_new = np.max(truesigma) + 1
        minsigma_new = np.min(truesigma[truesigma > 0]) if np.any(truesigma > 0) else 0
        
        tmin_new = tmin
        tmax_new = tmax
        order = 1
    
    return tmin_new, tmax_new, minsigma_new, maxsigma_new, order, temps, truesigma, Tc

def critical_zoom_improved(tmin, tmax, numtemp=None, mq=9, mu=0, lambda1=5.3, ui=1e-4, uf=None, 
                          d0_lower=0, d0_upper=10, mq_tolerance=0.1, max_iterations=10, v3=-22.6/(6*np.sqrt(2)), v4=4.2):
    """
    Iteratively zoom in on the critical point until first-order transition is found.
    
    Args:
        tmin, tmax: Initial temperature range
        numtemp: Number of temperature points per iteration (if None, uses CPU count)
        mq: Quark mass in MeV
        mu: Chemical potential in MeV
        lambda1: Lambda1 parameter
        ui: Lower integration bound
        uf: Upper integration bound (if None, set to 1-ui)
        d0_lower, d0_upper: Search bounds for d0
        mq_tolerance: Tolerance for quark mass matching
        max_iterations: Maximum number of zoom iterations
        v3: Chiral parameter v3 (default: -22.6/(6*sqrt(2)))
        v4: Chiral parameter v4 (default: 4.2)
    
    Returns:
        order, iteration_number, sigma_list, temps_list, Tc
    """
    # Optimize numtemp based on CPU count if not specified
    if numtemp is None:
        numtemp = os.cpu_count()
        print(f"Auto-setting numtemp to {numtemp} (number of CPU cores) for optimal parallelization")
    
    if uf is None:
        uf = 1 - ui
    
    order = 2
    iteration_number = 0
    first_order_found = False
    
    # Lists to store results from each iteration
    sigma_list = []
    temps_list = []
    
    Tc = tmax  # Default value in case loop never runs
    
    # Initial sigma bounds (will be updated in first iteration)
    minsigma = 0
    maxsigma = 100
    
    print(f"Starting critical zoom for mq={mq} MeV, mu={mu} MeV, lambda1={lambda1}")
    print(f"Using v3={v3:.3f}, v4={v4:.3f}")
    print(f"Initial temperature range: {tmin:.2f} - {tmax:.2f} MeV")
    
    # Iteratively run until first-order transition found or convergence criteria met
    while (order == 2 and iteration_number < max_iterations and 
           tmin < tmax and abs(tmax - tmin) > 0.01):
        
        print(f"\n--- Iteration {iteration_number + 1} ---")
        
        tmin, tmax, minsigma, maxsigma, order, temps, truesigma, Tc = order_checker(
            tmin, tmax, numtemp, mq, mu, lambda1, ui, uf, d0_lower, d0_upper, mq_tolerance, v3, v4
        )
        
        # Force first iteration to be second order to ensure proper zooming
        if not first_order_found and order == 1:
            print("First order detected on first check - zooming in for more precision")
            
            # Find the steepest gradient region
            valid_indices = truesigma[:, 0] > 0
            if np.any(valid_indices):
                valid_sigma = truesigma[valid_indices, 0]
                valid_temps = temps[valid_indices]
                
                if len(valid_sigma) > 2:
                    transition_index = np.argmin(np.gradient(valid_sigma**3))
                    Tc = valid_temps[transition_index]
                    
                    buffer = min(2, len(valid_temps) // 4)
                    tmin = valid_temps[max(transition_index - buffer, 0)]
                    tmax = valid_temps[min(transition_index + buffer, len(valid_temps) - 1)]
                    
                    print(f"Zooming in on region: {tmin:.3f} - {tmax:.3f} MeV")
            
            order = 2  # Force another iteration
            first_order_found = True
        
        iteration_number += 1
        print(f"Transition order: {order}")
        
        if tmax < tmin:
            print("WARNING: Temperature bounds reversed!")
            break
        
        # Store results from this iteration
        sigma_list.append(truesigma)
        temps_list.append(temps)
    
    print(f"\nZoom completed after {iteration_number} iterations")
    print(f"Final critical temperature: {Tc:.3f} MeV")
    print(f"Final transition order: {order}")
    
    return order, iteration_number, sigma_list, temps_list, Tc

def plot_results(sigma_list, temps_list, mq, mu, lambda1, Tc, order):
    """
    Plot the sigma values for each iteration.
    """
    # Get standard matplotlib colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    plt.figure(figsize=(10, 6))
    
    # Plot all iterations
    for i, (temps, sigma) in enumerate(zip(temps_list, sigma_list)):
        # Convert cube roots back to original sigma values for plotting
        sigma_vals = sigma**3
        
        alpha = 0.7 if i < len(sigma_list) - 1 else 1.0  # Fade earlier iterations
        
        # Plot up to 3 sigma branches
        for j in range(3):
            non_zero_mask = sigma_vals[:, j] > 0
            if np.any(non_zero_mask):
                plt.scatter(temps[non_zero_mask], sigma_vals[non_zero_mask, j], 
                           color=colors[j % len(colors)], alpha=alpha, s=20,
                           label=f'σ{j+1}' if i == len(sigma_list) - 1 else "")
    
    # Mark critical temperature
    plt.axvline(Tc, color='red', linestyle='--', alpha=0.8, label=f'Tc = {Tc:.2f} MeV')
    
    plt.xlabel("Temperature (MeV)")
    plt.ylabel("σ (GeV³)")
    plt.title(f'Critical Point Analysis: mq={mq} MeV, μ={mu} MeV, λ₁={lambda1:.2f}\n'
              f'Transition Order: {order}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_results(sigma_list, temps_list, mq, mu, lambda1, order, Tc, filename_prefix="critical_zoom"):
    """
    Save results to a pickle file.
    """
    df_all_list = []
    
    for i, (temps, sigma) in enumerate(zip(temps_list, sigma_list)):
        df = pd.DataFrame()
        df['temps'] = temps
        df['sigma1'] = sigma[:, 0]**3  # Convert back to original sigma values
        df['sigma2'] = sigma[:, 1]**3
        df['sigma3'] = sigma[:, 2]**3
        df['iteration'] = i
        df['order'] = order
        df['mq'] = mq
        df['mu'] = mu
        df['lambda1'] = lambda1
        df['Tc'] = Tc
        df_all_list.append(df)
    
    # Combine all iterations
    df_all = pd.concat(df_all_list, ignore_index=True)
    
    # Create filename
    filename = f"{filename_prefix}_mq{mq}_mu{mu}_lambda1{lambda1:.2f}_order{order}.pkl"
    
    # Create directory if it doesn't exist
    os.makedirs('phase_data', exist_ok=True)
    filepath = os.path.join('phase_data', filename)
    
    # Save to pickle
    df_all.to_pickle(filepath)
    print(f"Results saved to {filepath}")
    
    return filepath

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    # Parameters
    tmin = 60      # Initial minimum temperature (MeV)
    tmax = 105     # Initial maximum temperature (MeV)
    # numtemp will be auto-set to CPU count for optimal parallelization
    
    mq = 9         # Quark mass (MeV)
    mu = 0         # Chemical potential (MeV)
    lambda1 = 5.3  # Lambda1 parameter
    
    # Integration parameters
    ui = 1e-4      # Lower integration bound
    uf = 1 - 1e-4  # Upper integration bound
    
    # Search parameters
    d0_lower = 0
    d0_upper = 10
    mq_tolerance = 0.1
    
    # Chiral parameters (optional - will use defaults if not specified)
    # v3 = -22.6/(6*np.sqrt(2))  # Default: ≈ -2.668
    # v4 = 4.2                   # Default: 4.2
    
    # Run the critical zoom analysis (numtemp auto-determined)
    order, iteration_number, sigma_list, temps_list, Tc = critical_zoom_improved(
        tmin, tmax, numtemp=None, mq=mq, mu=mu, lambda1=lambda1, ui=ui, uf=uf, 
        d0_lower=d0_lower, d0_upper=d0_upper, mq_tolerance=mq_tolerance
        # v3=v3, v4=v4  # Uncomment and define above to use custom values
    )
    
    # Plot results
    plot_results(sigma_list, temps_list, mq, mu, lambda1, Tc, order)
    
    # Save results
    save_results(sigma_list, temps_list, mq, mu, lambda1, order, Tc)
    
    print("\nAnalysis complete!")
