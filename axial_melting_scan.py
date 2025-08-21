#!/usr/bin/env python3
"""
Axial Melting Temperature Scanner

This script finds the axial melting temperature as a function of chemical potential.
The melting temperature is defined as the lowest temperature at which the axial
potential has no local minimum (i.e., the derivative has no zeros).

Usage:
    python axial_melting_scan.py --mu-min 0 --mu-max 200 --mu-points 21 --mq 9.0 --lambda1 7.438
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import argparse
import sys
import os

# Import necessary functions from our existing modules 
from chiral_solve_complete import calculate_sigma_values
# from axial_spectra import (
#     calculate_variables, mu_g, z_h, Q_func, lam_function,
#     ui, uf  # Integration bounds
# )

# Global variables for interpolation functions
chi_interp = None
chi_prime_interp = None
g5 = 2*np.pi  # Define g5 constant

def initialize_chiral_field_and_derivative(mq_value, mq_tolerance, T_value, mu_value, lambda1_value, ui_value, uf_value, v3=None, v4=None):
    """
    Initialize both chi and chi_prime interpolation functions
    """
    global chi_interp, chi_prime_interp
    
    try:
        # Pass v3 and v4 if provided, otherwise use defaults from chiral_solve_complete
        if v3 is not None and v4 is not None:
            result = calculate_sigma_values(mq_value, mq_tolerance, T_value, mu_value, lambda1_value, ui_value, uf_value, v3=v3, v4=v4)
        else:
            result = calculate_sigma_values(mq_value, mq_tolerance, T_value, mu_value, lambda1_value, ui_value, uf_value)
        sigma_values = result["sigma_values"]
        chiral_fields = result["chiral_fields"]
        chiral_derivatives = result["chiral_derivatives"]
        u_values = result["u_values"]
        
        # Print sigma solutions and their cube roots
        print(f"  Sigma solutions found: {len(sigma_values)}")
        for i, sigma in enumerate(sigma_values):
            cube_root = np.sign(sigma) * (np.abs(sigma))**(1/3)
            print(f"    σ_{i+1} = {sigma:.3e} MeV³ (∛|σ| = {cube_root:.2f} MeV)")
        
        # If multiple sigma values, use the smallest positive
        if len(sigma_values) > 1:
            positive_sigmas = [s for s in sigma_values if s > 0]
            if positive_sigmas:
                min_index = np.argmin(positive_sigmas)
                selected_sigma = positive_sigmas[min_index]
                orig_index = np.where(sigma_values == selected_sigma)[0][0]
                print(f"  Selected smallest positive: σ = {selected_sigma:.3e} MeV³ (∛|σ| = {(np.abs(selected_sigma))**(1/3):.2f} MeV)")
            else:
                min_index = np.argmin(np.abs(sigma_values))
                selected_sigma = sigma_values[min_index]
                orig_index = min_index
                print(f"  Selected smallest by absolute value: σ = {selected_sigma:.3e} MeV³ (∛|σ| = {np.sign(selected_sigma) * (np.abs(selected_sigma))**(1/3):.2f} MeV)")
        else:
            orig_index = 0
            selected_sigma = sigma_values[0]
            cube_root = np.sign(selected_sigma) * (np.abs(selected_sigma))**(1/3)
            print(f"  Using single solution: σ = {selected_sigma:.3e} MeV³ (∛|σ| = {cube_root:.2f} MeV)")
        
        # Create interpolation functions
        chi_interp = interp1d(u_values, chiral_fields[orig_index], bounds_error=False, fill_value="extrapolate")
        chi_prime_interp = interp1d(u_values, chiral_derivatives[orig_index], bounds_error=False, fill_value="extrapolate")
        
        return selected_sigma
    except Exception as e:
        print(f"Warning: Error initializing chiral field for T={T_value}, mu={mu_value}: {str(e)}")
        return None

def vector_potential(u, T, mu, mu_g, kappa=1):
    """
    Vector potential: f[u, Q] * (df[u, Q]*BT'[u] + f[u, Q]*(BT'[u]^2 + BT''[u]))
    """
    # Handle mu = 0 case
    if mu == 0:
        Q = 0
        zh = 1.0 / (np.pi * T)
    else:
        sqrt_term = np.sqrt(np.pi**2 * T**2 * kappa**2 + 2 * mu**2)
        Q = (-np.pi * T * kappa + sqrt_term) / mu
        zh = (kappa * (-np.pi * T * kappa + sqrt_term)) / mu**2
    
    # Blackness function: f[u, Q] = 1 - (1 + Q²)*u⁴ + Q²*u⁶
    f = 1 - (1 + Q**2) * u**4 + Q**2 * u**6
    
    # Derivative of blackness function: df[u, Q] = -4*(1 + Q²)*u³ + 6*Q²*u⁵
    df = -4 * (1 + Q**2) * u**3 + 6 * Q**2 * u**5
    
    # BT'[u] = -1/(2u) - u*(zh*μg)²
    BT_prime = -1.0/(2*u) - u * (zh * mu_g)**2
    
    # BT''[u] = 1/(2u²) - (zh*μg)²
    BT_double_prime = 1.0/(2*u**2) - (zh * mu_g)**2
    
    # Vector potential: f * (df*BT' + f*(BT'^2 + BT''))
    result = f * (df * BT_prime + f * (BT_prime**2 + BT_double_prime))
    
    return result

def dVT(u, T, mu, mu_g, kappa=1):
    """
    Derivative of the vector potential with respect to u
    """
    # Handle mu = 0 case
    if mu == 0:
        Q = 0
        zh = 1.0 / (np.pi * T)
    else:
        sqrt_term = np.sqrt(np.pi**2 * T**2 * kappa**2 + 2 * mu**2)
        Q = (-np.pi * T * kappa + sqrt_term) / mu
        zh = (kappa * (-np.pi * T * kappa + sqrt_term)) / mu**2
    
    # Blackness function and its derivatives
    f = 1 - (1 + Q**2) * u**4 + Q**2 * u**6
    df = -4 * (1 + Q**2) * u**3 + 6 * Q**2 * u**5
    d2f = -12 * (1 + Q**2) * u**2 + 30 * Q**2 * u**4
    
    # BT derivatives
    BT_prime = -1.0/(2*u) - u * (zh * mu_g)**2
    BT_double_prime = 1.0/(2*u**2) - (zh * mu_g)**2
    BT_triple_prime = -1.0/(u**3)
    
    # Apply product rule to differentiate f * (df*BT' + f*(BT'^2 + BT''))
    term1 = df * (df * BT_prime + f * (BT_prime**2 + BT_double_prime))
    term2 = f * (d2f * BT_prime + df * BT_double_prime + df * (BT_prime**2 + BT_double_prime) + f * (2 * BT_prime * BT_double_prime + BT_triple_prime))
    
    return term1 + term2

def axial_potential(u, T, mu, mu_g):
    """
    Axial potential: (g5^2/u^2) * chi(u)^2 * f(u)
    """
    # Handle mu = 0 case separately  
    if mu == 0:
        f = 1 - u**4
    else:
        root = np.sqrt(np.pi**2 * T**2 + 2 * mu**2)
        delta = -np.pi * T + root
        d2 = delta**2
        f = 1 + u**6 * d2/mu**2 - u**4 * (1 + d2/mu**2)
        
    return (g5**2 * f * chi_interp(u)**2 / u**2)

def d_axial_pot(u, T, mu, mu_g):
    """
    Derivative of the axial potential with respect to u
    """
    # Handle mu = 0 case separately  
    if mu == 0:
        f = 1 - u**4
        df = -4*u**3
    else:
        root = np.sqrt(np.pi**2 * T**2 + 2 * mu**2)
        delta = -np.pi * T + root
        d2 = delta**2
        f = 1 + u**6 * d2/mu**2 - u**4 * (1 + d2/mu**2)
        df = 6*u**5 * d2/mu**2 - 4*u**3 * (1 + d2/mu**2)

    return (
        -2 * g5**2 * f * chi_interp(u)**2 / u**3
        + 2 * g5**2 * f * chi_interp(u) * chi_prime_interp(u) / u**2
        + g5**2 * chi_interp(u)**2 * df / u**2
    )

def has_zeros_in_derivative(T_value, mu_value, mq_value, lambda1_value, mq_tolerance=0.01, u_min=1e-2, u_max=1-1e-4, u_points=1000, v3=None, v4=None, mug=440.0):
    """
    Check if the total axial potential derivative has any zeros.
    Returns True if zeros are found, False otherwise.
    """
    global chi_interp, chi_prime_interp
    
    # Initialize chiral field
    sigma = initialize_chiral_field_and_derivative(mq_value, mq_tolerance, T_value, mu_value, 
                                                 lambda1_value, u_min, u_max, v3, v4)
    if sigma is None:
        return False  # If chiral field initialization fails, assume no zeros
    
    # Get model parameters - use the passed mug value directly
    # mug = mu_g()  # Removed - now using parameter
    
    # Create u array
    u_values = np.linspace(u_min, u_max, u_points)
    
    try:
        # Calculate total derivative
        dVT_values = np.array([dVT(u, T_value, mu_value, mug) for u in u_values])
        d_axial_pot_values = np.array([d_axial_pot(u, T_value, mu_value, mug) for u in u_values])
        total_derivative = dVT_values + d_axial_pot_values
        
        # Check for sign changes (zeros)
        sign_changes = np.sum(np.diff(np.sign(total_derivative)) != 0)
        
        return sign_changes > 0
        
    except Exception as e:
        print(f"Error calculating derivatives for T={T_value}, mu={mu_value}: {str(e)}")
        return False

def find_melting_temperature(mu_value, mq_value, lambda1_value, T_start=200.0, T_min=1.0, T_tolerance=0.1, max_iterations=50, v3=None, v4=None, mug=440.0):
    """
    Find the melting temperature for a given chemical potential.
    Uses a more robust search strategy with smaller temperature decrements.
    
    Parameters:
    -----------
    mu_value : float
        Chemical potential in MeV
    mq_value : float
        Quark mass in MeV
    lambda1_value : float
        Lambda1 parameter
    T_start : float
        Starting temperature for the search
    T_min : float
        Minimum temperature to consider
    T_tolerance : float
        Temperature tolerance for convergence
    max_iterations : int
        Maximum number of iterations
    v3 : float, optional
        Internal parameter v3 (derived from gamma)
    v4 : float, optional
        Internal parameter v4 (same as lambda4)
    mug : float
        mu_g parameter (default: 440.0)
        
    Returns:
    --------
    float or None : Melting temperature in MeV, or None if not found
    """
    print(f"Finding melting temperature for μ = {mu_value} MeV...")
    
    # Start with high temperature and scan down with smaller steps
    T_current = T_start
    T_step = 5.0  # Start with 5 MeV steps
    
    # First, find a temperature without zeros (upper bound)
    while T_current < 500:
        try:
            if not has_zeros_in_derivative(T_current, mu_value, mq_value, lambda1_value, v3=v3, v4=v4, mug=mug):
                break
        except Exception as e:
            print(f"  Error at T={T_current}: {e}")
            T_current += T_step
            continue
        T_current += T_step
    
    if T_current >= 500:
        print(f"  Could not find high enough temperature without zeros for μ = {mu_value}")
        return None
    
    T_upper = T_current  # Temperature without zeros
    print(f"  Found upper bound without zeros: T = {T_upper} MeV")
    
    # Now scan down to find where zeros first appear
    T_current = T_upper - T_step
    T_lower = None
    
    while T_current >= T_min:
        try:
            if has_zeros_in_derivative(T_current, mu_value, mq_value, lambda1_value, v3=v3, v4=v4, mug=mug):
                T_lower = T_current
                break
        except Exception as e:
            print(f"  Error at T={T_current}: {e}")
            # If we get an error (e.g., chiral field can't be solved), 
            # this might be our effective T_min
            T_min = max(T_min, T_current + T_step)
            break
        T_current -= T_step
    
    if T_lower is None:
        print(f"  No zeros found down to T = {T_min} MeV for μ = {mu_value}")
        return T_min  # Melting temperature is below our minimum searchable temperature
    
    print(f"  Found lower bound with zeros: T = {T_lower} MeV")
    print(f"  Refining search between T = {T_lower} and T = {T_upper} MeV")
    
    # Binary search between T_lower (has zeros) and T_upper (no zeros)
    iteration = 0
    while (T_upper - T_lower) > T_tolerance and iteration < max_iterations:
        T_mid = (T_upper + T_lower) / 2.0
        
        try:
            if has_zeros_in_derivative(T_mid, mu_value, mq_value, lambda1_value, v3=v3, v4=v4, mug=mug):
                T_lower = T_mid  # Still has zeros, melting T is higher
            else:
                T_upper = T_mid  # No zeros, melting T is lower
        except Exception as e:
            print(f"  Error at T={T_mid}: {e}")
            # If we get an error, assume we can't go lower
            T_lower = T_mid
            
        print(f"  Iteration {iteration+1}: T_range = [{T_lower:.2f}, {T_upper:.2f}] MeV")
        iteration += 1
    
    melting_T = (T_upper + T_lower) / 2.0
    print(f"  Found melting temperature: {melting_T:.2f} MeV")
    
    return melting_T

def scan_melting_temperatures(mu_min, mu_max, mu_points, mq_value, lambda1_value, T_start=200.0, v3=None, v4=None, mug=440.0):
    """
    Scan melting temperatures across a range of chemical potentials.
    
    Parameters:
    -----------
    mu_min : float
        Minimum chemical potential
    mu_max : float
        Maximum chemical potential
    mu_points : int
        Number of chemical potential points
    mq_value : float
        Quark mass
    lambda1_value : float
        Lambda1 parameter
    T_start : float
        Starting temperature for the first scan
    v3 : float, optional
        Internal parameter v3 (derived from gamma)
    v4 : float, optional
        Internal parameter v4 (same as lambda4)
    mug : float
        mu_g parameter (default: 440.0)
        
    Returns:
    --------
    tuple : (mu_values, melting_temperatures)
    """
    mu_values = np.linspace(mu_min, mu_max, mu_points)
    melting_temperatures = []
    
    current_T_start = T_start
    
    for i, mu in enumerate(mu_values):
        melting_T = find_melting_temperature(mu, mq_value, lambda1_value, T_start=current_T_start, v3=v3, v4=v4, mug=mug)
        
        if melting_T is not None:
            melting_temperatures.append(melting_T)
            # Use the current melting temperature as starting point for next μ
            # (since melting T generally decreases with increasing μ)
            current_T_start = max(melting_T + 10, 10)  # Add some buffer
        else:
            print(f"Failed to find melting temperature for μ = {mu} MeV")
            melting_temperatures.append(np.nan)
    
    return mu_values, np.array(melting_temperatures)

def plot_melting_curve(mu_values, melting_temperatures, mq_value, lambda1_value, gamma=None, lambda4=None, show_plot=True):
    """
    Plot the melting temperature vs chemical potential.
    """
    # Filter out NaN values
    valid_mask = ~np.isnan(melting_temperatures)
    valid_mu = mu_values[valid_mask]
    valid_T = melting_temperatures[valid_mask]

    vector_melting = np.array([
    [0.01, 102.8], [1.01, 102.8], [2.01, 102.8], [3.01, 102.8], [4.01, 102.8], [5.01, 102.7], [6.01, 102.7], [7.01, 102.7], [8.01, 102.7], [9.01, 102.7],
    [10.01, 102.7], [11.01, 102.7], [12.01, 102.6], [13.01, 102.6], [14.01, 102.6], [15.01, 102.6], [16.01, 102.5], [17.01, 102.5], [18.01, 102.5], [19.01, 102.4],
    [20.01, 102.4], [21.01, 102.4], [22.01, 102.3], [23.01, 102.3], [24.01, 102.2], [25.01, 102.2], [26.01, 102.2], [27.01, 102.1], [28.01, 102.1], [29.01, 102.0],
    [30.01, 101.9], [31.01, 101.9], [32.01, 101.8], [33.01, 101.8], [34.01, 101.7], [35.01, 101.6], [36.01, 101.6], [37.01, 101.5], [38.01, 101.4], [39.01, 101.4],
    [40.01, 101.3], [41.01, 101.2], [42.01, 101.2], [43.01, 101.1], [44.01, 101.0], [45.01, 100.9], [46.01, 100.8], [47.01, 100.7], [48.01, 100.7], [49.01, 100.6],
    [50.01, 100.5], [51.01, 100.4], [52.01, 100.3], [53.01, 100.2], [54.01, 100.1], [55.01, 100.0], [56.01, 99.9], [57.01, 99.8], [58.01, 99.7], [59.01, 99.6],
    [60.01, 99.5], [61.01, 99.3], [62.01, 99.2], [63.01, 99.1], [64.01, 99.0], [65.01, 98.9], [66.01, 98.8], [67.01, 98.6], [68.01, 98.5], [69.01, 98.4],
    [70.01, 98.3], [71.01, 98.1], [72.01, 98.0], [73.01, 97.9], [74.01, 97.7], [75.01, 97.6], [76.01, 97.4], [77.01, 97.3], [78.01, 97.2], [79.01, 97.0],
    [80.01, 96.9], [81.01, 96.7], [82.01, 96.6], [83.01, 96.4], [84.01, 96.2], [85.01, 96.1], [86.01, 95.9], [87.01, 95.8], [88.01, 95.6], [89.01, 95.4],
    [90.01, 95.3], [91.01, 95.1], [92.01, 94.9], [93.01, 94.7], [94.01, 94.6], [95.01, 94.4], [96.01, 94.2], [97.01, 94.0], [98.01, 93.8], [99.01, 93.7],
    [100.01, 93.5], [101.01, 93.3], [102.01, 93.1], [103.01, 92.9], [104.01, 92.7], [105.01, 92.5], [106.01, 92.3], [107.01, 92.1], [108.01, 91.9], [109.01, 91.7],
    [110.01, 91.5], [111.01, 91.3], [112.01, 91.1], [113.01, 90.8], [114.01, 90.6], [115.01, 90.4], [116.01, 90.2], [117.01, 90.0], [118.01, 89.7], [119.01, 89.5],
    [120.01, 89.3], [121.01, 89.0], [122.01, 88.8], [123.01, 88.6], [124.01, 88.3], [125.01, 88.1], [126.01, 87.9], [127.01, 87.6], [128.01, 87.4], [129.01, 87.1],
    [130.01, 86.9], [131.01, 86.6], [132.01, 86.4], [133.01, 86.1], [134.01, 85.8], [135.01, 85.6], [136.01, 85.3], [137.01, 85.1], [138.01, 84.8], [139.01, 84.5],
    [140.01, 84.3], [141.01, 84.0], [142.01, 83.7], [143.01, 83.4], [144.01, 83.1], [145.01, 82.9], [146.01, 82.6], [147.01, 82.3], [148.01, 82.0], [149.01, 81.7],
    [150.01, 81.4], [151.01, 81.1], [152.01, 80.8], [153.01, 80.5], [154.01, 80.2], [155.01, 79.9], [156.01, 79.6], [157.01, 79.3], [158.01, 79.0], [159.01, 78.7],
    [160.01, 78.4], [161.01, 78.1], [162.01, 77.8], [163.01, 77.4], [164.01, 77.1], [165.01, 76.8], [166.01, 76.5], [167.01, 76.1], [168.01, 75.8], [169.01, 75.5],
    [170.01, 75.1], [171.01, 74.8], [172.01, 74.5], [173.01, 74.1], [174.01, 73.8], [175.01, 73.4], [176.01, 73.1], [177.01, 72.7], [178.01, 72.4], [179.01, 72.0],
    [180.01, 71.7], [181.01, 71.3], [182.01, 70.9], [183.01, 70.6], [184.01, 70.2], [185.01, 69.8], [186.01, 69.5], [187.01, 69.1], [188.01, 68.7], [189.01, 68.4],
    [190.01, 68.0], [191.01, 67.6], [192.01, 67.2], [193.01, 66.8], [194.01, 66.4], [195.01, 66.0], [196.01, 65.7], [197.01, 65.3], [198.01, 64.9], [199.01, 64.5],
    [200.01, 64.1], [201.01, 63.7], [202.01, 63.3], [203.01, 62.9], [204.01, 62.4], [205.01, 62.0], [206.01, 61.6], [207.01, 61.2], [208.01, 60.8], [209.01, 60.4],
    [210.01, 60.0], [211.01, 59.5], [212.01, 59.1], [213.01, 58.7], [214.01, 58.2], [215.01, 57.8], [216.01, 57.4], [217.01, 56.9], [218.01, 56.5], [219.01, 56.1],
    [220.01, 55.6], [221.01, 55.2], [222.01, 54.7], [223.01, 54.3], [224.01, 53.8], [225.01, 53.4], [226.01, 52.9], [227.01, 52.5], [228.01, 52.0], [229.01, 51.5],
    [230.01, 51.1], [231.01, 50.6], [232.01, 50.1], [233.01, 49.7], [234.01, 49.2], [235.01, 48.7], [236.01, 48.2], [237.01, 47.8], [238.01, 47.3], [239.01, 46.8],
    [240.01, 46.3], [241.01, 45.8], [242.01, 45.3], [243.01, 44.8], [244.01, 44.3], [245.01, 43.8], [246.01, 43.3], [247.01, 42.8], [248.01, 42.3], [249.01, 41.8],
    [250.01, 41.3], [251.01, 40.8], [252.01, 40.3], [253.01, 39.8], [254.01, 39.3], [255.01, 38.7], [256.01, 38.2], [257.01, 37.7], [258.01, 37.2], [259.01, 36.6],
    [260.01, 36.1], [261.01, 35.6], [262.01, 35.0], [263.01, 34.5], [264.01, 34.0], [265.01, 33.4], [266.01, 32.9], [267.01, 32.3], [268.01, 31.8], [269.01, 31.2],
    [270.01, 30.7], [271.01, 30.1], [272.01, 29.6], [273.01, 29.0], [274.01, 28.5], [275.01, 27.9], [276.01, 27.3], [277.01, 26.8], [278.01, 26.2], [279.01, 25.6],
    [280.01, 25.0], [281.01, 24.5], [282.01, 23.9], [283.01, 23.3], [284.01, 22.7], [285.01, 22.1], [286.01, 21.5], [287.01, 20.9], [288.01, 20.3], [289.01, 19.7],
    [290.01, 19.1], [291.01, 18.5], [292.01, 17.9], [293.01, 17.3], [294.01, 16.7], [295.01, 16.1], [296.01, 15.5], [297.01, 14.9], [298.01, 14.3], [299.01, 13.6],
    [300.01, 13.0], [301.01, 12.4], [302.01, 11.8], [303.01, 11.1], [304.01, 10.5], [305.01, 9.9], [306.01, 9.2], [307.01, 8.6], [308.01, 7.9], [309.01, 7.3],
    [310.01, 6.6], [311.01, 6.0], [312.01, 5.3], [313.01, 4.7], [314.01, 4.0], [315.01, 3.4], [316.01, 2.7], [317.01, 2.0], [318.01, 1.4], [319.01, 0.7]
])
    
    plt.figure(figsize=(10, 6))
    plt.plot(valid_mu, valid_T, 'bo-', linewidth=2, markersize=6, label='Axial Melting')
    plt.plot(vector_melting[:, 0], vector_melting[:, 1], 'r--', linewidth=2, label='Vector Melting')
    plt.xlabel('Chemical Potential μ (MeV)')
    plt.ylabel(' Melting Temperature (MeV)')
    plt.legend()
    
    # Update title to include gamma and lambda4 with proper LaTeX formatting
    plt.title(f'Axial Meson Melting Temperature vs Chemical Potential\n'
              f'$m_q = {mq_value}$ MeV, $\\lambda_1 = {lambda1_value}$, '
              f'$\\gamma = {gamma}$, $\\lambda_4 = {lambda4}$')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Ensure axial_plots directory exists
    os.makedirs('axial_plots', exist_ok=True)
    
    # Save the plot with new naming convention (always include gamma and lambda4)
    plot_filename = f'axial_plots/axial_melting_curve_mq_{mq_value:.1f}_lambda1_{lambda1_value:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.png'
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

def save_data(mu_values, melting_temperatures, mq_value, lambda1_value, gamma=None, lambda4=None):
    """
    Save the melting temperature data to a CSV file.
    """
    # Ensure axial_data directory exists
    os.makedirs('axial_data', exist_ok=True)
    
    # Generate filename with new naming convention (always include gamma and lambda4)
    filename = f'axial_data/axial_melting_data_mq_{mq_value:.1f}_lambda1_{lambda1_value:.1f}_gamma_{gamma:.1f}_lambda4_{lambda4:.1f}.csv'
    
    # Create header
    header = f"# Axial Meson Melting Temperature Data\n"
    header += f"# mq = {mq_value} MeV, lambda1 = {lambda1_value}, gamma = {gamma}, lambda4 = {lambda4}\n"
    header += f"# mu (MeV), T_melting (MeV)\n"
    
    # Combine data
    data = np.column_stack((mu_values, melting_temperatures))
    
    # Save to file
    np.savetxt(filename, data, fmt='%.6f', delimiter=',', header=header)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scan axial melting temperatures vs chemical potential')
    parser.add_argument('--mu-min', type=float, default=0.0, help='Minimum chemical potential (MeV)')
    parser.add_argument('--mu-max', type=float, default=200.0, help='Maximum chemical potential (MeV)')
    parser.add_argument('--mu-points', type=int, default=21, help='Number of chemical potential points')
    parser.add_argument('--mq', type=float, default=9.0, help='Quark mass (MeV)')
    parser.add_argument('--lambda1', type=float, default=7.438, help='Lambda1 parameter')
    parser.add_argument('--gamma', type=float, default=-22.6, help='Background metric parameter gamma (default: -22.6)')
    parser.add_argument('--lambda4', type=float, default=4.2, help='Fourth-order coupling parameter lambda4 (default: 4.2)')
    parser.add_argument('--mug', type=float, default=440.0, help='mu_g parameter (default: 440.0)')
    parser.add_argument('--T-start', type=float, default=200.0, help='Starting temperature for scan (MeV)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--no-display', action='store_true', help='Save plot but do not display it')
    parser.add_argument('--no-save', action='store_true', help='Skip saving data')
    
    args = parser.parse_args()
    
    # Use gamma and lambda4 values (either provided or defaults from argparse)
    import math
    
    gamma_used = args.gamma
    lambda4_used = args.lambda4
    
    print(f"Using gamma = {gamma_used:.3f}, lambda4 = {lambda4_used:.3f}")
    
    # Convert to internal parameters v3 and v4
    v3 = gamma_used / (6 * math.sqrt(2))
    v4 = lambda4_used
    print(f"Converted to internal parameters: v3 = {v3:.6f}, v4 = {v4:.3f}")
    
    print("=" * 60)
    print("AXIAL MESON MELTING TEMPERATURE SCANNER")
    print("=" * 60)
    print(f"Chemical potential range: {args.mu_min} - {args.mu_max} MeV ({args.mu_points} points)")
    print(f"Quark mass: {args.mq} MeV")
    print(f"Lambda1: {args.lambda1}")
    print(f"mu_g: {args.mug} MeV")
    print(f"Starting temperature: {args.T_start} MeV")
    print("=" * 60)
    
    # Perform the scan
    mu_values, melting_temperatures = scan_melting_temperatures(
        args.mu_min, args.mu_max, args.mu_points, args.mq, args.lambda1, args.T_start, v3, v4, args.mug
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for mu, T_melt in zip(mu_values, melting_temperatures):
        if not np.isnan(T_melt):
            print(f"μ = {mu:6.1f} MeV  →  T_melting = {T_melt:6.2f} MeV")
        else:
            print(f"μ = {mu:6.1f} MeV  →  T_melting = FAILED")
    
    # Save data
    if not args.no_save:
        save_data(mu_values, melting_temperatures, args.mq, args.lambda1, gamma_used, lambda4_used)
    
    # Plot results
    if not args.no_plot:
        show_plot = not args.no_display
        plot_melting_curve(mu_values, melting_temperatures, args.mq, args.lambda1, gamma_used, lambda4_used, show_plot=show_plot)
    
    print("\nScan complete!")
