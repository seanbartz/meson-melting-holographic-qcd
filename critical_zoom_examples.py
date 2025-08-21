#!/usr/bin/env python3
"""
Example: Using the Improved Critical Zoom Analysis

This script demonstrates how to use the new critical_zoom_improved.py
to find critical points in the QCD phase diagram.
"""

from critical_zoom_improved import critical_zoom_improved, plot_results, save_results
from critical_analysis_utils import analyze_phase_diagram_point, quick_sigma_scan

def example_single_point():
    """
    Example: Analyze a single point in the phase diagram.
    """
    print("=== Single Point Analysis ===")
    
    # Parameters
    mq = 9.0      # Quark mass (MeV)
    mu = 0.0      # Chemical potential (MeV)
    lambda1 = 5.3 # Lambda1 parameter
    
    # Temperature range for analysis
    tmin = 60
    tmax = 105
    numtemp = 25
    
    # Run the improved critical zoom
    order, iterations, sigma_list, temps_list, Tc = critical_zoom_improved(
        tmin, tmax, numtemp, mq, mu, lambda1,
        ui=1e-4, uf=1-1e-4, max_iterations=8
    )
    
    # Plot and save results
    plot_results(sigma_list, temps_list, mq, mu, lambda1, Tc, order)
    save_results(sigma_list, temps_list, mq, mu, lambda1, order, Tc)
    
    return order, Tc

def example_parameter_scan():
    """
    Example: Scan over different lambda1 values to map phase diagram.
    """
    print("\n=== Parameter Scan ===")
    
    # Fixed parameters
    mq = 9.0
    mu = 0.0
    T_range = (60, 105)
    
    # Lambda1 values to scan
    lambda1_values = [4.5, 5.0, 5.3, 5.5, 6.0]
    
    results = {}
    
    for lambda1 in lambda1_values:
        print(f"\nAnalyzing lambda1 = {lambda1}")
        
        # Quick overview analysis
        result = analyze_phase_diagram_point(
            mq, mu, lambda1, T_range, num_points=30, plot=False
        )
        
        results[lambda1] = result
        
        print(f"  Transition type: {result['transition_type']}")
        if result['Tc_best']:
            print(f"  Critical temperature: {result['Tc_best']:.2f} MeV")
    
    return results

def example_mass_scan():
    """
    Example: Scan over different quark masses.
    """
    print("\n=== Quark Mass Scan ===")
    
    # Fixed parameters
    mu = 0.0
    lambda1 = 5.3
    T_range = (50, 120)
    
    # Quark mass values to scan
    mq_values = [3.0, 6.0, 9.0, 12.0, 15.0]
    
    critical_temperatures = []
    
    for mq in mq_values:
        print(f"\nAnalyzing mq = {mq} MeV")
        
        # Use quick analysis for overview
        result = analyze_phase_diagram_point(
            mq, mu, lambda1, T_range, num_points=25, plot=False
        )
        
        if result['Tc_best']:
            critical_temperatures.append((mq, result['Tc_best']))
            print(f"  Critical temperature: {result['Tc_best']:.2f} MeV")
        else:
            print("  No clear critical temperature found")
    
    # Plot critical line
    if critical_temperatures:
        import matplotlib.pyplot as plt
        
        masses, temps = zip(*critical_temperatures)
        plt.figure(figsize=(8, 6))
        plt.plot(masses, temps, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Quark Mass (MeV)')
        plt.ylabel('Critical Temperature (MeV)')
        plt.title(f'Critical Line: μ = {mu} MeV, λ₁ = {lambda1}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return critical_temperatures

def example_detailed_zoom():
    """
    Example: Detailed zoom analysis for precise critical temperature.
    """
    print("\n=== Detailed Zoom Analysis ===")
    
    # Parameters for detailed analysis
    mq = 9.0
    mu = 0.0
    lambda1 = 5.3
    
    # Start with wider range, more iterations for precision
    tmin = 70
    tmax = 90
    numtemp = 30  # More points for better resolution
    
    order, iterations, sigma_list, temps_list, Tc = critical_zoom_improved(
        tmin, tmax, numtemp, mq, mu, lambda1,
        ui=1e-4, uf=1-1e-4, 
        max_iterations=12,  # Allow more iterations
        mq_tolerance=0.05   # Tighter tolerance
    )
    
    print(f"\nDetailed Analysis Results:")
    print(f"  Final critical temperature: {Tc:.4f} MeV")
    print(f"  Transition order: {order}")
    print(f"  Number of zoom iterations: {iterations}")
    
    # Create detailed plot
    plot_results(sigma_list, temps_list, mq, mu, lambda1, Tc, order)
    
    return Tc, order

if __name__ == '__main__':
    print("Critical Zoom Analysis Examples")
    print("===============================")
    
    # Run different examples
    try:
        # Single point analysis with zoom
        order, Tc = example_single_point()
        print(f"\nSingle point result: Tc = {Tc:.3f} MeV, Order = {order}")
        
        # Parameter scan
        param_results = example_parameter_scan()
        
        # Mass scan
        mass_results = example_mass_scan()
        
        # Detailed zoom
        Tc_detailed, order_detailed = example_detailed_zoom()
        
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"Error in examples: {str(e)}")
        print("Make sure all required modules are available and working.")
