import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from chiral_solve_complete import calculate_sigma_values

# Import necessary functions from axial_spectra.py
from axial_spectra import (
    calculate_variables, mu_g, z_h, Q_func, lam_function,
    ui, uf  # Integration bounds
)

# Global variables for interpolation functions
chi_interp = None
chi_prime_interp = None
g5 = 2*np.pi  # Define g5 constant

def initialize_chiral_field_and_derivative(mq_value, mq_tolerance, T_value, mu_value, lambda1_value, ui_value, uf_value):
    """
    Initialize both chi and chi_prime interpolation functions
    """
    global chi_interp, chi_prime_interp
    
    try:
        result = calculate_sigma_values(mq_value, mq_tolerance, T_value, mu_value, lambda1_value, ui_value, uf_value)
        sigma_values = result["sigma_values"]
        chiral_fields = result["chiral_fields"]
        chiral_derivatives = result["chiral_derivatives"]  # This should be available from chiral_solve_complete
        u_values = result["u_values"]
        
        # If multiple sigma values, use the smallest positive
        if len(sigma_values) > 1:
            positive_sigmas = [s for s in sigma_values if s > 0]
            if positive_sigmas:
                min_index = np.argmin(positive_sigmas)
                selected_sigma = positive_sigmas[min_index]
                orig_index = np.where(sigma_values == selected_sigma)[0][0]
                print(f"Multiple sigma solutions found: {sigma_values}. Using the smallest positive: {selected_sigma}")
            else:
                min_index = np.argmin(np.abs(sigma_values))
                selected_sigma = sigma_values[min_index]
                print(f"Multiple sigma solutions found: {sigma_values}. No positive sigma found, using the smallest by absolute value: {selected_sigma}")
                orig_index = min_index
        else:
            orig_index = 0
            selected_sigma = sigma_values[0]
        
        # Create interpolation functions
        chi_interp = interp1d(u_values, chiral_fields[orig_index], bounds_error=False, fill_value="extrapolate")
        chi_prime_interp = interp1d(u_values, chiral_derivatives[orig_index], bounds_error=False, fill_value="extrapolate")
        
        return selected_sigma
    except Exception as e:
        print(f"Warning: Error initializing chiral field, using default: {str(e)}")
        # If sigma calculation fails, set chi=0 and chi_prime=0
        u_values = np.linspace(ui_value, uf_value, 1000)
        chi_values = np.zeros_like(u_values)  # Default to zero function
        chi_prime_values = np.zeros_like(u_values)  # Default derivative is also zero
        chi_interp = interp1d(u_values, chi_values, bounds_error=False, fill_value="extrapolate")
        chi_prime_interp = interp1d(u_values, chi_prime_values, bounds_error=False, fill_value="extrapolate")
        return mq_value  # Return mq as approximate sigma value

def vector_potential(u, T, mu, mu_g, kappa=1):
    """
    Vector potential: f[u, Q] * (df[u, Q]*BT'[u] + f[u, Q]*(BT'[u]^2 + BT''[u]))
    where:
    - f[u, Q] = 1 - (1 + Q^2)*u^4 + Q^2*u^6 is the blackness function
    - BT[u] = (As[u] - b[u, zh, μg])/2 with As[u] = -Log[u], b[u, zh, μg] = (u*zh*μg)^2
    - Q and zh depend on T, mu, kappa
    """
    
    # Handle mu = 0 case
    if mu == 0:
        Q = 0
        zh = 1.0 / (np.pi * T)
    else:
        # Q = (-π T κ + √(π² T² κ² + 2 μ²))/μ
        sqrt_term = np.sqrt(np.pi**2 * T**2 * kappa**2 + 2 * mu**2)
        Q = (-np.pi * T * kappa + sqrt_term) / mu
        
        # zh = (κ (-π T κ + √(π² T² κ² + 2 μ²)))/μ²
        zh = (kappa * (-np.pi * T * kappa + sqrt_term)) / mu**2
    
    # Blackness function: f[u, Q] = 1 - (1 + Q²)*u⁴ + Q²*u⁶
    f = 1 - (1 + Q**2) * u**4 + Q**2 * u**6
    
    # Derivative of blackness function: df[u, Q] = -4*(1 + Q²)*u³ + 6*Q²*u⁵
    df = -4 * (1 + Q**2) * u**3 + 6 * Q**2 * u**5
    
    # BT[u] = (As[u] - b[u, zh, μg])/2 = (-Log[u] - (u*zh*μg)²)/2
    # BT'[u] = -1/(2u) - u*(zh*μg)²
    BT_prime = -1.0/(2*u) - u * (zh * mu_g)**2
    
    # BT''[u] = 1/(2u²) - (zh*μg)²
    BT_double_prime = 1.0/(2*u**2) - (zh * mu_g)**2
    
    # Vector potential: f * (df*BT' + f*(BT'^2 + BT''))
    result = f * (df * BT_prime + f * (BT_prime**2 + BT_double_prime))
    
    return result

def dVT(u, T, mu, mu_g):
    # Derivative of the thermal vector potential.
    # Restoring to its previous state.
    
    # Handle mu = 0 case separately
    if mu == 0:
        return 0
    
    # common sub‑expressions
    root    = np.sqrt(np.pi**2 * T**2 + 2 * mu**2)
    delta   = -np.pi * T + root
    d2      = delta**2
    mu2     = mu**2
    mu4     = mu2**2

    # building blocks
    termA = 1 + u**6 * d2/mu2 - u**4 * (1 + d2/mu2)
    dA    = 6*u**5 * d2/mu2 - 4*u**3 * (1 + d2/mu2)
    E1    = 1/u**2       - 2*d2 * mu_g**2/mu4
    E2    = -1/u         - 2*u * d2 * mu_g**2/mu4
    C1    = -1/u**3      + 0.5 * E1 * E2
    D1    = 0.5 * E1     + 0.25 * E2**2

    # sub‑terms matching the Mathematica grouping
    S = (
        0.5 * dA * E1
      + 0.5 * ((30*u**4 * d2)/mu2 - 12*u**2 * (1 + d2/mu2)) * E2
      + termA * C1
      + dA * D1
    )
    T_term = (
        0.5 * dA * E2
      + termA * D1
    )

    # full expression
    return termA * S + dA * T_term

def axial_potential(u, T, mu, mu_g):
    # This is the axial potential, which includes the chiral field.
    # V_a = (g5^2/u^2) * chi(u)^2 * f(u)
    # This correctly goes to zero at u=1 because f(1)=0.
    
    # Handle mu = 0 case separately  
    if mu == 0:
        # For mu = 0, the blackness function f = 1 - u^4
        f = 1 - u**4
    else:
        # common sub‑expressions for f
        root  = np.sqrt(np.pi**2 * T**2 + 2 * mu**2)
        delta = -np.pi * T + root
        d2    = delta**2
        f  = 1 + u**6 * d2/mu**2 - u**4 * (1 + d2/mu**2)
        
    return (g5**2 * f * chi_interp(u)**2 / u**2)


def d_axial_pot(u, T, mu, mu_g):
    # This is the derivative of the axial potential.
    # d/du [ (g5^2/u^2) * chi(u)^2 * f(u) ]

    # Handle mu = 0 case separately  
    if mu == 0:
        # For mu = 0, the blackness function f = 1 - u^4, so df = -4*u^3
        f = 1 - u**4
        df = -4*u**3
    else:
        # common sub‑expressions for f and df
        root  = np.sqrt(np.pi**2 * T**2 + 2 * mu**2)
        delta = -np.pi * T + root
        d2    = delta**2

        f  = 1 + u**6 * d2/mu**2 - u**4 * (1 + d2/mu**2)
        df = 6*u**5 * d2/mu**2 - 4*u**3 * (1 + d2/mu**2)

    # assume g5, chi(u) and chi_prime(u) are defined elsewhere
    return (
        -2 * g5**2 * f * chi_interp(u)**2 / u**3
        + 2 * g5**2 * f * chi_interp(u) * chi_prime_interp(u) / u**2
        +     g5**2 * chi_interp(u)**2 * df / u**2
    )

def plot_axial_potential_derivative(T_value, mu_value, mq_value=9.0, lambda1_value=7.438, 
                                   mq_tolerance=0.01, u_min=0.1, u_max=1.0, u_points=1000):
    """
    Calculate and plot the derivative of the axial potential
    
    Args:
        T_value: Temperature in MeV
        mu_value: Chemical potential in MeV
        mq_value: Quark mass (default: 9.0)
        lambda1_value: Lambda1 parameter (default: 7.438)
        mq_tolerance: Tolerance for quark mass (default: 0.01)
        u_min: Minimum u value (default: 0.1)
        u_max: Maximum u value (default: 1.0)
        u_points: Number of u points (default: 1000)
    """
    global chi_interp, chi_prime_interp
    
    print(f"Calculating axial potential derivative for T = {T_value} MeV, μ = {mu_value} MeV")
    
    # Get model parameters
    mug = mu_g()
    
    # Initialize chiral field and its derivative
    sigma = initialize_chiral_field_and_derivative(mq_value, mq_tolerance, T_value, mu_value, 
                                                 lambda1_value, ui, uf)
    print(f"Chiral condensate σ = {sigma}")
    print(f"Cube root of chiral condensate σ^(1/3) = {sigma**(1/3):.1f}")
    
    # Check that interpolation functions were created
    if chi_interp is None or chi_prime_interp is None:
        print("Error: Failed to initialize chiral field interpolation functions")
        return
    
    # Create u array
    u_values = np.linspace(u_min, u_max, u_points)
    
    # Calculate derivative components
    dVT_values = np.array([dVT(u, T_value, mu_value, mug) for u in u_values])
    d_axial_pot_values = np.array([d_axial_pot(u, T_value, mu_value, mug) for u in u_values])
    
    # Calculate potential components (for confirmation)
    vector_pot_values = np.array([vector_potential(u, T_value, mu_value, mug) for u in u_values])
    axial_pot_values = np.array([axial_potential(u, T_value, mu_value, mug) for u in u_values])
    
    # The "total derivative" for finding critical points is the sum of the two derivatives
    total_derivative = dVT_values + d_axial_pot_values
    
    # Create the plot
    plt.figure(figsize=(15, 12))
    
    # Plot potential components (for confirmation)
    plt.subplot(3, 1, 1)
    plt.plot(u_values, vector_pot_values, label='Vector potential (V_T)', linewidth=2, color='blue')
    plt.plot(u_values, axial_pot_values, label='Axial potential (V_a)', linewidth=2, color='green')
    plt.plot(u_values, vector_pot_values + axial_pot_values, label='Total potential (V_T + V_a)', linewidth=2, color='red', linestyle='--')
    plt.xlabel('u')
    plt.ylabel('Potential')
    plt.title(f'Potential Components (T = {T_value} MeV, μ = {mu_value} MeV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot individual derivative components
    plt.subplot(3, 1, 2)
    plt.plot(u_values, dVT_values, label='dV_T/du', linewidth=2)
    plt.plot(u_values, d_axial_pot_values, label='dV_a/du', linewidth=2)
    plt.plot(u_values, total_derivative, label='Total derivative', linewidth=2, linestyle='--')
    #set the ylimits based on d_axial_pot_values, not total_derivative
    plt.ylim(np.min(d_axial_pot_values) * 1.2, np.max(d_axial_pot_values) * 1.2)
    plt.xlabel('u')
    plt.ylabel('Derivative')
    plt.title(f'Components of Potential Derivative (T = {T_value} MeV, μ = {mu_value} MeV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot just the total derivative
    plt.subplot(3, 1, 3)
    plt.plot(u_values, total_derivative, 'r-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('u')
    plt.ylabel('Total Derivative')
    plt.title('Total Axial Potential Derivative')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Find and mark zeros of total derivative
    total_zero_crossings = []
    for i in range(len(total_derivative)-1):
        if total_derivative[i] * total_derivative[i+1] < 0:
            # Linear interpolation to find more precise zero crossing
            u_zero = u_values[i] - total_derivative[i] * (u_values[i+1] - u_values[i]) / (total_derivative[i+1] - total_derivative[i])
            total_zero_crossings.append(u_zero)
    
    # Find and mark zeros of vector potential derivative
    vector_zero_crossings = []
    for i in range(len(dVT_values)-1):
        if dVT_values[i] * dVT_values[i+1] < 0:
            # Linear interpolation to find more precise zero crossing
            u_zero = u_values[i] - dVT_values[i] * (u_values[i+1] - u_values[i]) / (dVT_values[i+1] - dVT_values[i])
            vector_zero_crossings.append(u_zero)
    
    # Determine smart y-limits for the potential plot
    total_potential = vector_pot_values + axial_pot_values
    max_potential_at_zeros = []
    
    if total_zero_crossings:
        # Check each zero crossing to see if it's a local maximum
        for u_zero in total_zero_crossings:
            # Find the closest index in u_values
            idx = np.argmin(np.abs(u_values - u_zero))
            
            # Check if this is a local maximum by looking at neighboring derivatives
            # A local maximum occurs when derivative changes from positive to negative
            if idx > 0 and idx < len(total_derivative) - 1:
                if total_derivative[idx-1] > 0 and total_derivative[idx+1] < 0:
                    # This is a local maximum, record the potential value
                    potential_at_zero = total_potential[idx]
                    max_potential_at_zeros.append(potential_at_zero)
        
        if max_potential_at_zeros:
            # Set upper limit to 1.2 times the maximum potential at local maxima
            y_upper = 2 * max(max_potential_at_zeros)
        else:
            # No local maxima found, use default upper limit
            y_upper = 20
    else:
        # No zero crossings found, use default upper limit
        y_upper = 20
    
    # Apply the smart y-limits to the potential plot
    plt.subplot(3, 1, 1)
    plt.ylim(0, y_upper)
    
    # Mark zeros on the derivative plot
    plt.subplot(3, 1, 2)
    if vector_zero_crossings:
        plt.scatter(vector_zero_crossings, np.zeros(len(vector_zero_crossings)), 
                   color='blue', s=100, zorder=5, marker='s', label='Vector potential zeros')
    if total_zero_crossings:
        plt.scatter(total_zero_crossings, np.zeros(len(total_zero_crossings)), 
                   color='red', s=100, zorder=5, marker='o', label='Total axial potential zeros')
    
    # Mark zeros on the total derivative plot
    plt.subplot(3, 1, 3)
    if total_zero_crossings:
        plt.scatter(total_zero_crossings, np.zeros(len(total_zero_crossings)), 
                   color='red', s=100, zorder=5, marker='o')
    
    # Enhanced printout
    print("\n=== DERIVATIVE ZERO ANALYSIS ===")
    
    if vector_zero_crossings:
        print(f"Vector potential derivative zeros found at u = {[f'{u:.4f}' for u in vector_zero_crossings]}")
    else:
        print("No zeros found in vector potential derivative")
        
    if total_zero_crossings:
        print(f"Total axial potential derivative zeros found at u = {[f'{u:.4f}' for u in total_zero_crossings]}")
        if max_potential_at_zeros:
            print(f"Local maxima found with potential values: {[f'{v:.2f}' for v in max_potential_at_zeros]}")
            print(f"Setting y-axis upper limit to: {y_upper:.2f}")
        else:
            print("No local maxima found at zero crossings")
            print(f"Using default y-axis upper limit: {y_upper}")
    else:
        print("No zeros found in total axial potential derivative")
        print(f"Using default y-axis upper limit: {y_upper}")
    
    print("=================================\n")
    
    plt.tight_layout()
    
    # # Save the plot
    # plot_filename = f'axial_potential_derivative_T{T_value}_mu{mu_value}.png'
    # plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    # print(f"Plot saved to {plot_filename}")
    
    plt.show()
    
    return u_values, dVT_values, d_axial_pot_values, total_derivative, total_zero_crossings, vector_pot_values, axial_pot_values

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate and plot axial potential derivative')
    parser.add_argument('-T', type=float, default=17.0, help='Temperature in MeV (default: 17.0)')
    parser.add_argument('-mu', type=float, default=0.0, help='Chemical potential in MeV (default: 0.0)')
    parser.add_argument('-mq', type=float, default=9.0, help='Quark mass (default: 9.0)')
    parser.add_argument('-lambda1', type=float, default=7.438, help='Lambda1 parameter (default: 7.438)')
    parser.add_argument('-umin', type=float, default=0.01, help='Minimum u value (default: 0.01)')
    parser.add_argument('-umax', type=float, default=1-1e-5, help='Maximum u value (default: 1-1e-5)')
    parser.add_argument('-upoints', type=int, default=1000, help='Number of u points (default: 1000)')
    
    args = parser.parse_args()
    
    # Calculate and plot
    result = plot_axial_potential_derivative(
        T_value=args.T,
        mu_value=args.mu,
        mq_value=args.mq,
        lambda1_value=args.lambda1,
        u_min=args.umin,
        u_max=args.umax,
        u_points=args.upoints
    )