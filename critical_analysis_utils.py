#!/usr/bin/env python3
"""
Critical Zoom Utility Functions
Additional utility functions for the improved critical zoom analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from chiral_solve_complete import calculate_sigma_values

def quick_sigma_scan(mq, mu, lambda1, T_range, ui=1e-4, uf=None, d0_lower=0, d0_upper=10, mq_tolerance=0.1):
    """
    Quick scan of sigma values across temperature range to get overview.
    
    Args:
        mq: Quark mass in MeV
        mu: Chemical potential in MeV  
        lambda1: Lambda1 parameter
        T_range: Array or tuple of temperatures to scan
        ui, uf: Integration bounds
        d0_lower, d0_upper: Search bounds for d0
        mq_tolerance: Tolerance for quark mass matching
    
    Returns:
        temperatures, sigma_values_array
    """
    if uf is None:
        uf = 1 - ui
    
    if isinstance(T_range, tuple) and len(T_range) == 3:
        # Assume (start, stop, num_points)
        temperatures = np.linspace(T_range[0], T_range[1], T_range[2])
    else:
        temperatures = np.array(T_range)
    
    sigma_results = []
    
    print(f"Scanning {len(temperatures)} temperature points...")
    
    for i, T in enumerate(temperatures):
        try:
            result = calculate_sigma_values(mq, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower, d0_upper)
            sigma_values = result["sigma_values"]
            
            # Convert to cube roots for consistency
            sigma_cube_roots = []
            for sigma in sigma_values:
                if sigma >= 0:
                    sigma_cube_roots.append(sigma**(1/3))
                else:
                    sigma_cube_roots.append(-((-sigma)**(1/3)))
            
            # Pad to 3 values
            while len(sigma_cube_roots) < 3:
                sigma_cube_roots.append(0)
            sigma_cube_roots = sigma_cube_roots[:3]
            
            sigma_results.append(sigma_cube_roots)
            
        except Exception as e:
            print(f"Error at T={T:.2f}: {str(e)}")
            sigma_results.append([0, 0, 0])
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{len(temperatures)} points")
    
    return temperatures, np.array(sigma_results)

def find_transition_candidates(temperatures, sigma_values):
    """
    Find potential transition regions by looking for multiple sigma solutions.
    
    Args:
        temperatures: Array of temperature values
        sigma_values: Array of sigma values (N_temp x 3)
    
    Returns:
        List of temperature indices where multiple solutions exist
    """
    transition_indices = []
    
    for i, sigma_row in enumerate(sigma_values):
        # Count non-zero sigma values
        non_zero_count = np.sum(sigma_row > 0)
        if non_zero_count > 1:
            transition_indices.append(i)
    
    return transition_indices

def estimate_critical_temperature(temperatures, sigma_values, method='gradient'):
    """
    Estimate critical temperature using different methods.
    
    Args:
        temperatures: Array of temperature values
        sigma_values: Array of sigma values (N_temp x 3)
        method: 'gradient' or 'multiple_solutions'
    
    Returns:
        Estimated critical temperature
    """
    if method == 'multiple_solutions':
        # Look for first temperature with multiple solutions
        transition_indices = find_transition_candidates(temperatures, sigma_values)
        if transition_indices:
            return temperatures[transition_indices[0]]
        else:
            return None
    
    elif method == 'gradient':
        # Find steepest negative gradient in sigma^3
        valid_mask = sigma_values[:, 0] > 0
        if np.sum(valid_mask) < 3:
            return None
        
        valid_temps = temperatures[valid_mask]
        valid_sigma = sigma_values[valid_mask, 0]
        
        # Calculate gradient of sigma^3
        gradient = np.gradient(valid_sigma**3)
        min_gradient_idx = np.argmin(gradient)
        
        return valid_temps[min_gradient_idx]
    
    else:
        raise ValueError("Method must be 'gradient' or 'multiple_solutions'")

def plot_sigma_overview(temperatures, sigma_values, mq, mu, lambda1, Tc_estimate=None):
    """
    Plot overview of sigma values across temperature range.
    """
    plt.figure(figsize=(12, 8))
    
    # Convert to original sigma values for plotting
    sigma_original = sigma_values**3
    
    colors = ['blue', 'red', 'green']
    labels = ['σ₁', 'σ₂', 'σ₃']
    
    for i in range(3):
        non_zero_mask = sigma_original[:, i] > 0
        if np.any(non_zero_mask):
            plt.plot(temperatures[non_zero_mask], sigma_original[non_zero_mask, i], 
                    'o-', color=colors[i], label=labels[i], markersize=4)
    
    if Tc_estimate is not None:
        plt.axvline(Tc_estimate, color='black', linestyle='--', alpha=0.7, 
                   label=f'Tc estimate = {Tc_estimate:.2f} MeV')
    
    plt.xlabel('Temperature (MeV)')
    plt.ylabel('σ (GeV³)')
    plt.title(f'Sigma Values Overview: mq={mq} MeV, μ={mu} MeV, λ₁={lambda1:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_phase_diagram_point(mq, mu, lambda1, T_range, num_points=50, plot=True):
    """
    Complete analysis of a single point in the phase diagram.
    
    Args:
        mq: Quark mass in MeV
        mu: Chemical potential in MeV
        lambda1: Lambda1 parameter
        T_range: Tuple (T_min, T_max) for temperature scan
        num_points: Number of temperature points to scan
        plot: Whether to create plots
    
    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing phase diagram point: mq={mq}, mu={mu}, lambda1={lambda1}")
    
    # Initial temperature scan
    T_min, T_max = T_range
    temperatures, sigma_values = quick_sigma_scan(
        mq, mu, lambda1, (T_min, T_max, num_points)
    )
    
    # Estimate critical temperature
    Tc_gradient = estimate_critical_temperature(temperatures, sigma_values, 'gradient')
    Tc_multiple = estimate_critical_temperature(temperatures, sigma_values, 'multiple_solutions')
    
    # Find transition candidates
    transition_indices = find_transition_candidates(temperatures, sigma_values)
    
    # Determine transition type
    if transition_indices:
        transition_type = "First Order"
        Tc_best = temperatures[transition_indices[0]]
    elif Tc_gradient is not None:
        transition_type = "Second Order/Crossover"
        Tc_best = Tc_gradient
    else:
        transition_type = "No Clear Transition"
        Tc_best = None
    
    results = {
        'mq': mq,
        'mu': mu,
        'lambda1': lambda1,
        'temperatures': temperatures,
        'sigma_values': sigma_values,
        'Tc_gradient': Tc_gradient,
        'Tc_multiple': Tc_multiple,
        'Tc_best': Tc_best,
        'transition_type': transition_type,
        'transition_indices': transition_indices,
        'T_range': T_range
    }
    
    if plot:
        plot_sigma_overview(temperatures, sigma_values, mq, mu, lambda1, Tc_best)
    
    print(f"Analysis complete:")
    print(f"  Transition type: {transition_type}")
    print(f"  Critical temperature: {Tc_best:.3f} MeV" if Tc_best else "  No critical temperature found")
    
    return results

# Example usage and test function
def test_analysis():
    """
    Test the analysis functions with example parameters.
    """
    # Test parameters
    mq = 9.0
    mu = 0.0
    lambda1 = 5.3
    T_range = (60, 105)
    
    # Run analysis
    results = analyze_phase_diagram_point(mq, mu, lambda1, T_range, num_points=30, plot=True)
    
    return results

if __name__ == '__main__':
    # Run test analysis
    test_results = test_analysis()
    print("\nTest analysis completed!")
