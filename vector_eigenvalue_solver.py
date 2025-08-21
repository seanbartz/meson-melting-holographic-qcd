#!/usr/bin/env python3
"""
Zero-Temperature Vector Equation of Motion Solver

Solves the eigenvalue problem:
-v(z)'' + (1/4 * omega'^2 - 1/2 * omega'') * v(z) = m^2 * v(z)

where omega = phi(z) + log(z)

Uses shooting method to find eigenvalues m subject to boundary condition v(zmax) = 0.
Initial conditions: v(zmin) = 0.1, v'(zmin) = 1, with zmin = 0.01, zmax = 6.

Created on January 24, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import brentq, minimize_scalar
import argparse
from joblib import Parallel, delayed
import warnings

# Suppress integration warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def phi_function(z, mu_g):
    """
    Define the dilaton field phi(z) function for zero temperature.
    phi = (mu_g * z)^2
    """
    return (mu_g * z)**2

def omega_function(z, mu_g):
    """
    Calculate omega(z) = phi(z) + log(z)
    """
    return phi_function(z, mu_g) + np.log(z)

def omega_prime(z, mu_g):
    """
    Calculate omega'(z) analytically.
    
    omega(z) = phi(z) + log(z) = (μ_g * z)² + log(z)
    omega'(z) = 2 * μ_g² * z + 1/z
    """
    return 2 * mu_g**2 * z + 1/z

def omega_double_prime(z, mu_g):
    """
    Calculate omega''(z) analytically.
    
    omega'(z) = 2 * μ_g² * z + 1/z
    omega''(z) = 2 * μ_g² - 1/z²
    """
    return 2 * mu_g**2 - 1/(z**2)

def vector_ode(z, y, m_squared, mu_g):
    """
    Vector equation of motion as a system of first-order ODEs.
    
    The equation: -v'' + (1/4 * omega'^2 - 1/2 * omega'') * v = m^2 * v
    
    Rearranged: v'' = (1/4 * omega'^2 - 1/2 * omega'' - m^2) * v
    
    System: y[0] = v(z), y[1] = v'(z)
    dy[0]/dz = y[1]
    dy[1]/dz = (1/4 * omega'^2 - 1/2 * omega'' - m^2) * y[0]
    
    Using your simplified form: 1/4 * omega'^2 - 1/2 * omega'' = 3/(4*z^2) + z^2*mu_g^2
    
    Args:
        z: Independent variable
        y: [v, v'] - solution vector
        m_squared: m^2 parameter
        mu_g: Confinement scale parameter
    
    Returns:
        [v', v''] - derivatives
    """
    v, v_prime = y
    
    # Use the simplified form you provided
    potential_term = 3.0/(4.0*z**2) + (z**2) * (mu_g**2)
    
    # The coefficient in the differential equation
    coefficient = potential_term - m_squared
    
    # Second derivative
    v_double_prime = coefficient * v
    
    return [v_prime, v_double_prime]

def solve_vector_equation(m_squared, mu_g, z_min=0.01, z_max=6.0, v0=0.1, v_prime0=1.0):
    """
    Solve the vector equation for a given m^2 value.
    
    Args:
        m_squared: The m^2 parameter
        mu_g: Confinement scale parameter
        z_min: Starting point
        z_max: End point
        v0: Initial value v(z_min)
        v_prime0: Initial derivative v'(z_min)
    
    Returns:
        Final value v(z_max)
    """
    # Initial conditions
    y0 = [v0, v_prime0]
    
    # Integration points
    z_span = [z_min, z_max]
    
    try:
        # Solve the ODE
        sol = solve_ivp(vector_ode, z_span, y0, args=(m_squared, mu_g), 
                       dense_output=True, rtol=1e-8, atol=1e-10)
        
        if sol.success:
            # Return final value v(z_max)
            return sol.y[0][-1]
        else:
            return np.inf  # Return large value if integration fails
            
    except Exception as e:
        return np.inf  # Return large value if integration fails

def find_eigenvalue_bracket(m_min, m_max, mu_g, n_points=100, **kwargs):
    """
    Find brackets [m1, m2] where the boundary condition changes sign.
    This helps locate eigenvalues.
    
    Args:
        m_min: Minimum m value to search
        m_max: Maximum m value to search
        mu_g: Confinement scale parameter
        n_points: Number of points to sample
        **kwargs: Additional arguments for solve_vector_equation
    
    Returns:
        List of (m1, m2) brackets where sign changes occur
    """
    m_values = np.linspace(m_min, m_max, n_points)
    
    # Evaluate boundary condition at all points
    print(f"Scanning for sign changes in range [{m_min:.2f}, {m_max:.2f}]...")
    boundary_values = []
    
    for m in m_values:
        m_squared = m**2
        v_final = solve_vector_equation(m_squared, mu_g, **kwargs)
        boundary_values.append(v_final)
        
    boundary_values = np.array(boundary_values)
    
    # Find sign changes
    brackets = []
    for i in range(len(boundary_values) - 1):
        if (np.isfinite(boundary_values[i]) and np.isfinite(boundary_values[i+1]) and
            boundary_values[i] * boundary_values[i+1] < 0):
            brackets.append((m_values[i], m_values[i+1]))
            
    print(f"Found {len(brackets)} sign changes")
    return brackets

def find_eigenvalue_in_bracket(bracket, mu_g, **kwargs):
    """
    Find eigenvalue within a bracket using Brent's method.
    
    Args:
        bracket: (m1, m2) where boundary condition changes sign
        mu_g: Confinement scale parameter
        **kwargs: Additional arguments for solve_vector_equation
    
    Returns:
        Eigenvalue m or None if not found
    """
    m1, m2 = bracket
    
    def boundary_condition(m):
        m_squared = m**2
        return solve_vector_equation(m_squared, mu_g, **kwargs)
    
    try:
        # Use Brent's method to find the root
        eigenvalue = brentq(boundary_condition, m1, m2, xtol=1e-8)
        return eigenvalue
    except Exception as e:
        print(f"Failed to converge in bracket [{m1:.4f}, {m2:.4f}]: {e}")
        return None

def find_multiple_eigenvalues(n_eigenvalues=4, m_min=0.1, m_max=20.0, 
                            mu_g=440.0, scan_points=1000, n_jobs=-1, **kwargs):
    """
    Find the first n eigenvalues using parallel computation.
    
    Args:
        n_eigenvalues: Number of eigenvalues to find
        m_min: Minimum m value to search
        m_max: Maximum m value to search
        mu_g: Confinement scale parameter
        scan_points: Number of points for initial scan
        n_jobs: Number of parallel jobs (-1 for all cores)
        **kwargs: Additional arguments for solve_vector_equation
    
    Returns:
        List of eigenvalues
    """
    print(f"Searching for {n_eigenvalues} eigenvalues...")
    
    # Find brackets with sign changes
    brackets = find_eigenvalue_bracket(m_min, m_max, mu_g, scan_points, **kwargs)
    
    if len(brackets) == 0:
        print("No sign changes found. Try adjusting the search range.")
        return []
    
    print(f"Refining eigenvalues in {len(brackets)} brackets...")
    
    # Find eigenvalues in parallel
    eigenvalues = Parallel(n_jobs=n_jobs)(
        delayed(find_eigenvalue_in_bracket)(bracket, mu_g, **kwargs) 
        for bracket in brackets
    )
    
    # Filter out None values and sort
    valid_eigenvalues = [ev for ev in eigenvalues if ev is not None]
    valid_eigenvalues.sort()
    
    # Return first n eigenvalues
    return valid_eigenvalues[:n_eigenvalues]

def plot_eigenfunction(eigenvalue, mu_g, z_min=0.01, z_max=6.0, v0=0.1, v_prime0=1.0):
    """
    Plot the eigenfunction for a given eigenvalue.
    
    Args:
        eigenvalue: The eigenvalue m
        mu_g: Confinement scale parameter
        z_min: Starting point
        z_max: End point
        v0: Initial value
        v_prime0: Initial derivative
    """
    m_squared = eigenvalue**2
    y0 = [v0, v_prime0]
    
    # Dense z array for plotting
    z = np.linspace(z_min, z_max, 1000)
    
    # Solve the ODE
    sol = solve_ivp(vector_ode, [z_min, z_max], y0, args=(m_squared, mu_g), 
                   t_eval=z, rtol=1e-8, atol=1e-10)
    
    if sol.success:
        plt.figure(figsize=(10, 6))
        plt.plot(z, sol.y[0], 'b-', linewidth=2, label=f'v(z), m = {eigenvalue:.6f}')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=z_max, color='r', linestyle='--', alpha=0.5, label=f'z_max = {z_max}')
        plt.xlabel('z')
        plt.ylabel('v(z)')
        plt.title(f'Vector Eigenfunction (m = {eigenvalue:.6f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Check boundary condition
        final_value = sol.y[0][-1]
        print(f"Final value v({z_max}) = {final_value:.2e} (should be ≈ 0)")
        
        return sol
    else:
        print("Failed to solve ODE for plotting")
        return None

def plot_boundary_condition_scan(m_min=0.1, m_max=20.0, mu_g=440.0, n_points=1000, **kwargs):
    """
    Plot the boundary condition v(z_max) as a function of m to visualize eigenvalues.
    
    Args:
        m_min: Minimum m value
        m_max: Maximum m value
        mu_g: Confinement scale parameter
        n_points: Number of points to evaluate
        **kwargs: Additional arguments for solve_vector_equation
    """
    m_values = np.linspace(m_min, m_max, n_points)
    boundary_values = []
    
    print(f"Evaluating boundary condition for {n_points} m values...")
    
    for m in m_values:
        m_squared = m**2
        v_final = solve_vector_equation(m_squared, mu_g, **kwargs)
        boundary_values.append(v_final)
    
    boundary_values = np.array(boundary_values)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Clip extreme values for better visualization
    boundary_clipped = np.clip(boundary_values, -100, 100)
    
    plt.plot(m_values, boundary_clipped, 'b-', linewidth=1)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='v(z_max) = 0')
    plt.xlabel('m')
    plt.ylabel('v(z_max)')
    plt.title('Boundary Condition vs m (Eigenvalues at Zero Crossings)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Mark zero crossings
    for i in range(len(boundary_values) - 1):
        if (np.isfinite(boundary_values[i]) and np.isfinite(boundary_values[i+1]) and
            boundary_values[i] * boundary_values[i+1] < 0):
            m_cross = (m_values[i] + m_values[i+1]) / 2
            plt.axvline(x=m_cross, color='g', linestyle=':', alpha=0.7)
            plt.text(m_cross, 0, f'm≈{m_cross:.2f}', rotation=90, 
                    verticalalignment='bottom', fontsize=8)

def verify_simplified_form(z, mu_g):
    """
    Verify that the simplified form 3/(4*z^2) + z^2*mu_g^2 matches
    the original 1/4 * omega'^2 - 1/2 * omega''
    """
    # Original calculation
    omega_p = omega_prime(z, mu_g)
    omega_pp = omega_double_prime(z, mu_g)
    original = 0.25 * omega_p**2 - 0.5 * omega_pp
    
    # Simplified form
    simplified = 3.0/(4.0*z**2) + (z**2) * (mu_g**2)
    
    return original, simplified

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description='Solve vector eigenvalue problem')
    parser.add_argument('--n-eigenvalues', type=int, default=4, 
                       help='Number of eigenvalues to find (default: 4)')
    parser.add_argument('--m-min', type=float, default=0.1,
                       help='Minimum m value for search (default: 0.1)')
    parser.add_argument('--m-max', type=float, default=4.5,
                       help='Maximum m value for search (default: 4.5)')
    parser.add_argument('--z-min', type=float, default=1e-4,
                       help='Starting z value (default: 1e-4)')
    parser.add_argument('--z-max', type=float, default=6.0,
                       help='Ending z value (default: 6.0)')
    parser.add_argument('--v0', type=float, default=0.0,
                       help='Initial value v(z_min) (default: 0.0)')
    parser.add_argument('--v-prime0', type=float, default=0.1,
                       help='Initial derivative v\'(z_min) (default: 0.1)')
    parser.add_argument('--scan-points', type=int, default=1000,
                       help='Number of points for initial scan (default: 1000)')
    parser.add_argument('--mu-g', type=float, default=440.0,
                       help='Confinement scale μ_g in MeV (default: 440.0)')
    parser.add_argument('--plot-scan', action='store_true',
                       help='Plot boundary condition scan')
    parser.add_argument('--plot-eigenfunctions', action='store_true',
                       help='Plot eigenfunctions')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel computation')
    
    args = parser.parse_args()
    
    
    print("=" * 60)
    print("VECTOR EIGENVALUE SOLVER")
    print("=" * 60)
    print(f"Equation: -v'' + (1/4*ω'^2 - 1/2*ω'')*v = m²*v")
    print(f"Using dilaton field: φ(z) = (μ_g*z)²")
    print(f"Parameters: μ_g = {args.mu_g} MeV (zero temperature)")
    print(f"Boundary conditions: v({args.z_min}) = {args.v0}, v'({args.z_min}) = {args.v_prime0}, v({args.z_max}) = 0")
    print(f"Search range: m ∈ [{args.m_min}, {args.m_max}]")
    print("=" * 60)
    
    # Set up parameters
    solve_params = {
        'z_min': args.z_min,
        'z_max': args.z_max,
        'v0': args.v0,
        'v_prime0': args.v_prime0
    }
    
    n_jobs = 1 if args.no_parallel else -1
    
    # Plot boundary condition scan if requested
    if args.plot_scan:
        print("\nPlotting boundary condition scan...")
        plot_boundary_condition_scan(args.m_min, args.m_max, args.mu_g, args.scan_points, **solve_params)
        plt.show()
    
    # Find eigenvalues
    eigenvalues = find_multiple_eigenvalues(
        n_eigenvalues=args.n_eigenvalues,
        m_min=args.m_min,
        m_max=args.m_max,
        mu_g=args.mu_g,
        scan_points=args.scan_points,
        n_jobs=n_jobs,
        **solve_params
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if eigenvalues:
        print(f"Found {len(eigenvalues)} eigenvalues:")
        for i, ev in enumerate(eigenvalues):
            print(f"  m_{i+1}^2 = {ev**2:.8f}")
            
        # Plot eigenfunctions if requested
        if args.plot_eigenfunctions:
            print("\nPlotting eigenfunctions...")
            for i, ev in enumerate(eigenvalues):
                print(f"Plotting eigenfunction {i+1}...")
                plot_eigenfunction(ev, args.mu_g, **solve_params)
            plt.show()
    else:
        print("No eigenvalues found in the specified range.")
        print("Try adjusting the search range or parameters.")
    
    print("\nSolver complete!")

if __name__ == '__main__':
    main()
