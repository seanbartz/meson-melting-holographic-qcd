import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from joblib import Parallel, delayed
import pandas as pd
import os
from datetime import datetime

# Constants from the paper
v4 = 4.2
v3 = -22.6/(6*np.sqrt(2))

def log_sigma_calculation(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower, d0_upper, 
                         v3, v4, sigma_values, d0_values, filename='sigma_calculations.csv'):
    """
    Log sigma calculation results to a CSV file for machine learning purposes.
    
    Parameters:
    -----------
    All input parameters plus results
    """
    # Convert v3, v4 back to gamma, lambda4 for logging
    gamma = v3 * 6 * np.sqrt(2)
    lambda4 = v4
    
    # Ensure we have exactly 3 sigma values (pad with NaN if needed)
    sigma_padded = np.full(3, np.nan)
    d0_padded = np.full(3, np.nan)
    
    if len(sigma_values) > 0:
        sigma_padded[:len(sigma_values)] = sigma_values
        d0_padded[:len(d0_values)] = d0_values
    
    # Create data row
    data_row = {
        'timestamp': datetime.now().isoformat(),
        'T': T,
        'mu': mu,
        'ui': ui,
        'uf': uf,
        'mq': mq_input,
        'mq_tolerance': mq_tolerance,
        'lambda1': lambda1,
        'gamma': gamma,
        'lambda4': lambda4,
        'd0_lower': d0_lower,
        'd0_upper': d0_upper,
        'num_solutions': len(sigma_values),
        'sigma1': sigma_padded[0],
        'sigma2': sigma_padded[1],
        'sigma3': sigma_padded[2],
        'd0_1': d0_padded[0],
        'd0_2': d0_padded[1],
        'd0_3': d0_padded[2]
    }
    
    # Create DataFrame
    df_new = pd.DataFrame([data_row])
    
    # Ensure data directory exists
    data_dir = 'sigma_data'
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    # Append to existing file or create new one
    if os.path.exists(filepath):
        df_new.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df_new.to_csv(filepath, mode='w', header=True, index=False)
        print(f"Created new sigma calculation log: {filepath}")
    
    print(f"Logged sigma calculation: {len(sigma_values)} solutions found for mq={mq_input}, T={T}, mu={mu}")

def load_sigma_data(filename='sigma_calculations.csv'):
    """
    Load sigma calculation data from CSV file for analysis or machine learning.
    
    Parameters:
    -----------
    filename : str
        Name of the CSV file to load
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing all logged sigma calculations
    """
    data_dir = 'sigma_data'
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"No data file found at {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def sigma_data_summary(filename='sigma_calculations.csv'):
    """
    Print a summary of the sigma calculation database.
    """
    df = load_sigma_data(filename)
    
    if len(df) == 0:
        print("No data found.")
        return
    
    print(f"Sigma Calculation Database Summary")
    print(f"=" * 40)
    print(f"Total calculations: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    print("Solution distribution:")
    solution_counts = df['num_solutions'].value_counts().sort_index()
    for num_sol, count in solution_counts.items():
        print(f"  {int(num_sol)} solutions: {count} calculations")
    print()
    
    print("Parameter ranges:")
    numeric_cols = ['T', 'mu', 'mq', 'lambda1', 'gamma', 'lambda4']
    for col in numeric_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].min():.3f} to {df[col].max():.3f}")
    
    print(f"\nRecent calculations:")
    recent = df.tail(3)[['timestamp', 'T', 'mu', 'mq', 'lambda1', 'num_solutions']]
    print(recent.to_string(index=False))

def blackness(T, mu):
    kappa = 1
    
    if mu == 0:
        zh = 1/(math.pi*T)
        q = 0
    else:
        zh = kappa/mu**2*(-kappa*math.pi*T+math.sqrt((kappa*math.pi*T)**2+2*mu**2))
        q = (mu**3+kappa*math.pi*T*mu*(kappa*math.pi*T+math.sqrt((kappa*math.pi*T)**2+2*mu**2)))/(2*kappa**3)
        
    return zh, q

def chiral(u, y, params):
    chi, chip = y
    v3, v4, lambda1, mu_g, a0, zh, q = params
    
    Q = q*zh**3
    
    # Ballon-Bayona version
    phi = (mu_g*zh*u)**2-a0*(mu_g*zh*u)**3/(1+(mu_g*zh*u)**4)
    phip = 2*u*(zh*mu_g)**2+a0*(4*u**6*(zh*mu_g)**7/(1+(u*zh*mu_g)**4)**2-3*u**2*(zh*mu_g)**3/(1+(u*zh*mu_g)**4))

    f = 1 - (1+Q**2)*u**4 + Q**2*u**6
    fp = -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    
    # EOM for chiral field
    derivs = [chip,
              (3/u-fp/f+phip)*chip - (3*chi+lambda1*phi*chi-3*v3*chi**2-4*v4*chi**3)/(u**2*f)]
            
    return derivs

def chiral_solve_IR(d0, lambda1, T, mu, ui, uf, v3=v3, v4=v4):
    numpoints = 10000
    u = np.linspace(ui, uf, numpoints)
    u_backward = np.linspace(uf, ui, numpoints)

    zeta = np.sqrt(3)/(2*np.pi)
    mu_g = 440
    a0 = 0
    lambda3 = v3
    lambda4 = v4

    zh, q = blackness(T, mu)
    Q = q*zh**3
    
    # Defining constants for Taylor expansion at horizon u=1
    d1 = (3 * d0 - 3 * d0**2 * lambda3 - 4 * d0**3 * lambda4 + d0 * zh**2 * lambda1 * mu_g**2) / (2 * (-2 + Q**2))

    d2 = (1 / (16 * (-2 + Q**2)**2)) * (6 * d1 * (-6 + Q**2 + Q**4) +
    4 * d0**3 * (14 - 13 * Q**2) * lambda4 + d0**2 * ((42 - 39 * Q**2) * lambda3 - 24 * d1 * (-2 + Q**2) * lambda4) -
    2 * d1 * (-2 + Q**2) * zh**2 * (-8 + 4 * Q**2 - lambda1) * mu_g**2 +
    3 * d0 * (-14 + 13 * Q**2 + 8 * d1 * lambda3 - 4 * d1 * Q**2 * lambda3 + (-2 + 3 * Q**2) * zh**2 * lambda1 * mu_g**2))
    
    # IR boundary condition
    chi0 = d0+d1*(1-uf)+d2*(1-uf)**2
    chip0 = -d1-2*d2*(1-uf)
    y0 = [chi0, chip0]

    params = v3, v4, lambda1, mu_g, a0, zh, q
    
    # Solve the EOM using solve_ivp
    sol = solve_ivp(chiral, [uf, ui], y0, t_eval=u_backward, args=(params,))
    chi = sol.y[0][::-1]
    chip = sol.y[1][::-1]

    # # Solve the EOM using odeint
    # sol = odeint(chiral, y0, u_backward, args=(params,))
    # # Reverse the solution to get it from ui to uf
    # chi = sol[:, 0][::-1]
    # chip = sol[:, 1][::-1]

 
    x = zeta*zh*ui
    # First-order approximation
    if v3 == 0:
        mq1 = chi[0]/(zeta*zh*ui)
    else:
        # Second-order approximation
        mq1 = (x-x*np.sqrt(1-12*v3*chi[0]))/(6*x**2*v3)

    return mq1, chi, chip, u

def chiral_solve_IR_parallel(d0_array, lambda1, T, mu, ui, uf, v3=v3, v4=v4, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(chiral_solve_IR)(d0, lambda1, T, mu, ui, uf, v3, v4) for d0 in d0_array
    )
    # Unpack the results
    mq_array, chi_array, chip_array, u_array = zip(*results)
    
    # Fix the deprecation warning by specifying dtype=object
    return np.array(mq_array), np.array(chi_array, dtype=object), np.array(chip_array, dtype=object)

def initial_d0_mq(T, mu, mq_target, lambda1, ui, uf, d0_array, v3=v3, v4=v4):
    mq_array = chiral_solve_IR_parallel(d0_array, lambda1, T, mu, ui, uf, v3, v4, n_jobs=-1)[0]

    max_d0 = d0_array[-1]
    # Find and remove indices where abs(mq_array) > 150
    indices = np.where(np.abs(mq_array) > 150)[0]
    if len(indices) > 0:
        max_d0 = d0_array[indices[0]]
        # No need to check for multiple indices and try to index max_d0
        # max_d0 is already a scalar

    d0_array = np.delete(d0_array, indices)
    mq_array = np.delete(mq_array, indices)
    
    iterations = 0
    step_size = d0_array[1] - d0_array[0] if len(d0_array) > 1 else 0.1
    while mq_array[-1] < mq_target and iterations < 20 and len(mq_array) > 0:
        # Create new d0 values with finer spacing
        step_size = step_size / 100
        d0_new = np.arange(max(d0_array), min(max(d0_array) + (d0_array[-1] - d0_array[0]) / 20, max_d0), step_size)
        
        # Check if d0_new is empty
        if len(d0_new) == 0:
            break

        d0_array = np.concatenate((d0_array, d0_new))
        # Calculate mq for new d0 values
        mq_new = chiral_solve_IR_parallel(d0_new, lambda1, T, mu, ui, uf, v3, v4, n_jobs=-1)[0]
        mq_array = np.concatenate((mq_array, mq_new))
        
        indices = np.where(np.abs(mq_array) > 150)[0]
        if len(indices) > 0:
            # Just take the first index, no need for further indexing
            max_d0 = d0_array[indices[0]]
        
        d0_array = np.delete(d0_array, indices)
        mq_array = np.delete(mq_array, indices)

        iterations += 1
        
    # Refine if there's a large jump and we have enough points
    if len(mq_array) >= 2 and mq_array[-1] - mq_array[-2] > 5:
        d0_new = np.linspace(d0_array[-2], d0_array[-1], 10)[1:-1]
        mq_new = chiral_solve_IR_parallel(d0_new, lambda1, T, mu, ui, uf, v3, v4, n_jobs=-1)[0]
        mq_array = np.concatenate((mq_array[:-1], mq_new, mq_array[-1:]))
        d0_array = np.concatenate((d0_array[:-1], d0_new, d0_array[-1:]))
        
    return d0_array, mq_array

def process_mq_target(mq_target, d0_array, mq_improved, sigma_improved, lambda1, T, mu, ui, uf, v3, v4):
    d0_list = d0_array.tolist()
    mq_list = mq_improved.tolist()
    sigma_list = sigma_improved.tolist()
    
    indices = np.where(np.diff(np.sign(mq_improved - mq_target)))[0]
    
    for index in indices:
        mq_approx = np.array([mq_improved[index], mq_improved[index + 1]])
        d0_approx = np.array([d0_array[index], d0_array[index + 1]])
        
        d0_interp = interp1d(mq_approx, d0_approx)
        d0 = d0_interp(mq_target)
        mq, chi, chip, u = chiral_solve_IR(d0, lambda1, T, mu, ui, uf, v3, v4)

        i = 10
        u = np.linspace(ui, uf, len(chip))
        u_int = u[i]
        chi0 = chi[i]
        chip0 = chip[i]

        # Solving for mq and sigma
        lambda3 = v3
        zeta = np.sqrt(3)/(2*np.pi)
        zh, q = blackness(T, mu)
        
        if lambda3 == 0:
            sigma_new = zeta*(chip0*u_int-chi0)/(2*u_int**3*zh**3)
            mq_new = mq
        else:
            mq_new = (1-np.sqrt(1-3*lambda3*(3*chi0-chip0*u_int)))/(3*lambda3*zeta*u_int*zh)
            sigma_new = zeta*(1-6*chi0*lambda3+3*chip0*u_int*lambda3-np.sqrt(1-3*lambda3*(3*chi0-chip0*u_int)))/(3*u_int**3*zh**3*lambda3)
        
        d0_list.append(d0)
        mq_list.append(mq_new)
        sigma_list.append(sigma_new)
        
    return np.array(d0_list), np.array(mq_list), np.array(sigma_list)

def new_function(lambda1, T, mu, mq_large, ui, uf, d0_lower, d0_upper, v3=v3, v4=v4, numpoints=100):
    # Initial d0 and mq arrays
    d0_array = np.linspace(d0_lower, d0_upper, numpoints)
    d0_array, mq_array = initial_d0_mq(T, mu, mq_large, lambda1, ui, uf, d0_array, v3, v4)

    # Further processing to get mq_improved and sigma_improved
    _, chis, chips = chiral_solve_IR_parallel(d0_array, lambda1, T, mu, ui, uf, v3, v4, n_jobs=-1)

    i = 10
    u = np.linspace(ui, uf, len(chips[0]))
    u_int = u[i]
    # print('u_int', u_int)  # Commented out the print statement
    
    chi0 = np.array(chis[:, i], dtype=float)
    chip0 = np.array(chips[:, i], dtype=float)

    lambda3 = v3  
    zeta = np.sqrt(3) / (2 * np.pi)
    zh, q = blackness(T, mu)  
    
    if lambda3 == 0:
        sigma_improved = zeta * (chip0 * u_int - chi0) / (2 * u_int ** 3 * zh ** 3)
        mq_improved = mq_array
    else:
        # Fix for the sqrt error - ensure we're working with numpy arrays consistently
        sqrt_term = 1 - 3 * lambda3 * (3 * chi0 - chip0 * u_int)
        # Handle potential negative values in the sqrt
        sqrt_term = np.maximum(sqrt_term, 0)
        
        # Convert to numpy array explicitly and then calculate square root
        sqrt_term_array = np.asarray(sqrt_term)
        sqrt_result = np.sqrt(sqrt_term_array)
        
        mq_improved = (1 - sqrt_result) / (3 * lambda3 * zeta * u_int * zh)
        
        # Similarly for sigma_improved
        sigma_improved = zeta * (1 - 6 * chi0 * lambda3 + 3 * chip0 * u_int * lambda3 - sqrt_result) / (3 * u_int ** 3 * zh ** 3 * lambda3)

    # Run process_mq_target in parallel
    numMass = 100
    mq_overlay = np.linspace(np.min(mq_improved), mq_large, numMass)
    results = Parallel(n_jobs=-1)(delayed(process_mq_target)(mq_target, d0_array, mq_improved, sigma_improved, lambda1, T, mu, ui, uf, v3, v4) for mq_target in mq_overlay)

    # Aggregate and sort results
    all_d0_arrays = [res[0] for res in results]
    all_mq_improved = [res[1] for res in results]
    all_sigma_improved = [res[2] for res in results]
    
    all_d0_array = np.concatenate(all_d0_arrays)
    all_mq_improved = np.concatenate(all_mq_improved)
    all_sigma_improved = np.concatenate(all_sigma_improved)

    sort_indices = np.argsort(all_d0_array)
    all_d0_array = all_d0_array[sort_indices]
    all_mq_improved = all_mq_improved[sort_indices]
    all_sigma_improved = all_sigma_improved[sort_indices]

    unique_indices = np.unique(all_d0_array, return_index=True)[1]
    final_d0_array = all_d0_array[unique_indices]
    mq_improved = all_mq_improved[unique_indices]
    sigma_improved = all_sigma_improved[unique_indices]

    return mq_improved, sigma_improved, final_d0_array

def sigma_of_T(mq_input, mq_tolerance, T, mu, lambda1, d0_lower, d0_upper, ui, uf, v3=v3, v4=v4):
    # uf = 1-ui
    mq_large = 2*mq_input
    mq_improved, sigma_improved, d0_array = new_function(lambda1, T, mu, mq_large, ui, uf, d0_lower, d0_upper, v3, v4, numpoints=100)
    
    # Find indices where mq_improved-mq_input changes sign
    indices = np.where(np.diff(np.sign(mq_improved-mq_input)))[0]
    
    sigma_list = []
    d0_approx_list = []
    for index in indices:
        # Check array bounds
        if index + 1 >= len(mq_improved):
            continue
            
        if np.abs(mq_improved[index]-mq_input) < mq_tolerance:
            sigma_input = sigma_improved[index]
            d0_approx = d0_array[index]
        elif np.abs(mq_improved[index+1]-mq_input) < mq_tolerance:
            sigma_input = sigma_improved[index+1]
            d0_approx = d0_array[index+1]
        else:
            try:
                sigma_interp = interp1d(mq_improved[index:index+2], sigma_improved[index:index+2])
                sigma_input = sigma_interp(mq_input)
                d0_interp = interp1d(mq_improved[index:index+2], d0_array[index:index+2])
                d0_approx = d0_interp(mq_input)
            except:
                # Skip if interpolation fails
                continue
                
        sigma_list.append(sigma_input)
        d0_approx_list.append(d0_approx)
        
    sigma_values = np.array(sigma_list)
    if len(sigma_values) < 3:
        sigma_values = np.pad(sigma_values, (0, 3-len(sigma_values)), 'constant')
        
    return sigma_values, np.max(d0_array) if len(d0_array) > 0 else 0, np.min(d0_array) if len(d0_array) > 0 else 0, d0_approx_list

def calculate_sigma_values(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower=0, d0_upper=10, v3=v3, v4=v4, log_results=True):
    """
    Calculate sigma values and corresponding chiral fields for a given quark mass.
    
    Parameters:
    -----------
    mq_input : float
        Target quark mass in MeV
    mq_tolerance : float
        Tolerance for quark mass matching
    T : float
        Temperature
    mu : float
        Chemical potential
    lambda1 : float
        Lambda1 parameter
    ui : float
        Lower bound for u
    uf : float
        Upper bound for u (should be close to 1, e.g. 1-ui)
    d0_lower : float
        Lower bound for d0 search
    d0_upper : float
        Upper bound for d0 search
    v3 : float
        Chiral parameter v3 (default: global value)
    v4 : float
        Chiral parameter v4 (default: global value)
    log_results : bool
        Whether to log the results to CSV file (default: True)
        
    Returns:
    --------
    result : dict
        Dictionary containing sigma values, chiral fields, and derivatives
    """
    # Find the sigma values and d0 approximations
    sigma_values, d0_max, d0_min, d0_approx_list = sigma_of_T(
        mq_input, mq_tolerance, T, mu, lambda1, d0_lower, d0_upper, ui, uf, v3, v4
    )
    
    # Filter out zero sigma values (padding values)
    non_zero_indices = []
    for i, sigma in enumerate(sigma_values):
        if i < len(d0_approx_list) and sigma != 0:
            non_zero_indices.append(i)
            
    valid_sigma_values = [sigma_values[i] for i in non_zero_indices]
    valid_d0_approx_list = [d0_approx_list[i] for i in non_zero_indices]
    
    # Log the results if requested
    if log_results:
        log_sigma_calculation(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower, d0_upper,
                            v3, v4, valid_sigma_values, valid_d0_approx_list)
    
    # For each d0, get the corresponding chiral field and derivative
    chiral_fields = []
    chiral_derivatives = []
    u_values = None
    
    for d0 in valid_d0_approx_list:
        _, chi, chip, u = chiral_solve_IR(d0, lambda1, T, mu, ui, uf, v3, v4)
        chiral_fields.append(chi)
        chiral_derivatives.append(chip)
        if u_values is None:
            u_values = u
    
    # Create and return the result dictionary
    result = {
        "sigma_values": np.array(valid_sigma_values),
        "d0_values": np.array(valid_d0_approx_list),
        "chiral_fields": chiral_fields,
        "chiral_derivatives": chiral_derivatives,
        "u_values": u_values
    }
    
    return result

def plot_chiral_fields(result, mq_input, T, mu, lambda1):
    """
    Plot the chiral fields and UV approximations.
    """
    if "sigma_values" not in result or len(result["sigma_values"]) == 0:
        print("No valid solutions found to plot")
        return
        
    zeta = np.sqrt(3) / (2 * np.pi)
    zh, q = blackness(T, mu)
    u = result["u_values"]
    
    plt.figure(figsize=(10, 6))
    
    # Plot chiral fields
    for i, chi in enumerate(result["chiral_fields"]):
        plt.loglog(u, chi, label=f"$\chi_{i+1}$ solution")
    
    # Plot UV approximations
    for i, sigma in enumerate(result["sigma_values"]):
        if sigma != 0:  # Only plot for non-zero values of sigma
            uv_approx = mq_input * zeta * zh * u + sigma / (zeta) * zh**3 * u**3
            plt.loglog(u, uv_approx, '--', label=f"UV limit (σ={abs(sigma)**(1/3):.3f} MeV³)")
    
    plt.xlabel('$u$')
    plt.ylabel('$\\chi$')
    plt.title(f'$\\lambda_1={lambda1}$, quark mass={mq_input} MeV')
    plt.legend()
    # plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Parameters
    mq_input = 10  # MeV
    mq_tolerance = 0.1
    T = 10
    mu = 0
    lambda1 = 7
    ui = 1e-2
    uf = 1 - 1e-4
    
    # Calculate sigma values and chiral fields
    result = calculate_sigma_values(mq_input, mq_tolerance, T, mu, lambda1, ui, uf)
    
    # Print results
    print("Sigma values (MeV³):", result["sigma_values"])
    print("d0 values:", result["d0_values"])
    
    # Plot the chiral fields
    plot_chiral_fields(result, mq_input, T, mu, lambda1)