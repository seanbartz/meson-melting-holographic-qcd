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
import fcntl  # For file locking

# Constants from the paper
v4 = 4.2
v3 = -22.6/(6*np.sqrt(2))

# Confinement scale (MeV). Used by the dilaton and by the cutoff scheme below.
MU_G = 440.0

# ---------------------------------------------------------------------------
# Integration tolerances
# ---------------------------------------------------------------------------
# solve_ivp's defaults (RK45, rtol=1e-3, atol=1e-6) are far too loose for this
# problem. The map d0 -> mq is exponentially ill conditioned at low T
# (d ln mq / d ln d0 grows from ~6e1 at T = 120 MeV to ~1e8 at T = 17 MeV), so
# the integration has to supply the digits the root find then consumes. At the
# physical d0 for mq = 9 MeV, T = 17 MeV, the old defaults return mq ~ 2e5
# instead of 9.
#
# 'extract_index' is the index into the (u_i -> u_f) grid at which mq and sigma
# are read off. It used to be hard-wired to 10, which on a 10000-point grid sits
# at u_i + 1.0e-3 regardless of u_i -- so any u_i below ~1e-3 was ignored and
# convergence tests in u_i measured nothing. It is now 0, i.e. the cutoff u_i
# itself.
SOLVER_DEFAULT = {'method': 'DOP853', 'rtol': 1e-10, 'atol': 1e-14,
                  'extract_index': 0}

# Exactly reproduces the pre-2026-10 behaviour. Kept for regression tests and
# for reproducing published numbers; not for new results.
SOLVER_LEGACY = {'method': 'RK45', 'rtol': 1e-3, 'atol': 1e-6,
                 'extract_index': 10}


def solver_settings(solver=None):
    """Resolve a solver spec (None, 'default', 'legacy', or a dict) to a dict."""
    if solver is None or solver == 'default':
        return dict(SOLVER_DEFAULT)
    if solver == 'legacy':
        return dict(SOLVER_LEGACY)
    if isinstance(solver, dict):
        settings = dict(SOLVER_DEFAULT)
        settings.update(solver)
        return settings
    raise ValueError(f"unrecognised solver spec: {solver!r}")


# ---------------------------------------------------------------------------
# Cutoff scheme
# ---------------------------------------------------------------------------
# u = z/z_h is a convenient coordinate but a bad place to fix cutoffs, because
# z_h = 1/(pi T) at mu = 0, so a fixed u_i means a physical UV cutoff
# z_i = u_i*z_h that drifts like 1/T. sigma extracted from the leading UV
# behaviour carries a residual ln(z_i) dependence (the lambda1*phi*chi term
# sources a z^3 ln z piece that the inversion below does not subtract), worth
# about 3 MeV in sigma^(1/3) per decade of z_i. Holding u_i fixed therefore
# makes the renormalization scheme itself a function of T: between T = 120 and
# T = 17 MeV that is ~1% in sigma^(1/3) (~3% in sigma), and it bends the shape
# of the low-T tail of sigma(T).
#
# The fix is to hold the cutoffs fixed in physical units instead. Note this
# needs no change to the equations of motion: within one solve z_h is a
# constant, so u = z/z_h and v = mu_g*z differ by a constant factor and give
# identical solutions. The only physical content of the coordinate choice is
# where the two cutoffs sit.
#
# UV: z_i fixed at 1/(40 GeV). Chosen to sit in the window where the leading-
# order inversion is accurate (truncation error in mq < 0.01%) but well above
# the round-off floor of the sigma extraction, which divides by z_i^3 and goes
# to noise below z_i ~ 1e-8 MeV^-1. At T = 120 MeV this reproduces the old
# u_i = 1e-2 almost exactly, so high-T results are continuous with the old ones.
Z_UV = 2.5e-5      # MeV^-1
UI_MAX = 0.05      # sanity cap on u_i; only bites above T ~ 640 MeV

# IR: the horizon Taylor expansion (d1, d2 below) carries coefficients that grow
# like (mu_g z_h)^2, so a fixed 1-u_f degrades rapidly as T falls. At T = 17 MeV
# the old 1-u_f = 1e-4 is 4.5% off in mq, and 1-u_f = 1e-3 diverges outright.
# Measured requirement for 0.1% accuracy scales as ((mu_g z_h)^2)^(-5/3); the
# coefficient below carries a safety factor of >2 against that measurement.
IR_GAP_COEFF = 5.0e-3
IR_GAP_POWER = -5.0/3.0
IR_GAP_MAX = 1e-4   # never coarser than the old default
IR_GAP_MIN = 1e-9   # the ODE has a regular singular point at u = 1


def physical_cutoffs(T, mu, z_uv=Z_UV, mu_g_value=MU_G):
    """Cutoffs (u_i, u_f) for a T-independent renormalization scheme.

    u_i is placed at a fixed physical z_i, and the horizon gap 1-u_f is chosen
    so the horizon Taylor expansion stays accurate as (mu_g z_h)^2 grows.
    """
    zh, _ = blackness(T, mu)
    ui = min(z_uv/zh, UI_MAX)
    dilaton = max((mu_g_value*zh)**2, 1e-12)
    gap = IR_GAP_COEFF * dilaton**IR_GAP_POWER
    gap = min(max(gap, IR_GAP_MIN), IR_GAP_MAX)
    return ui, 1.0 - gap


def resolve_cutoffs(T, mu, ui=None, uf=None):
    """Fill in whichever of (u_i, u_f) is None from the physical scheme."""
    if ui is None or uf is None:
        pui, puf = physical_cutoffs(T, mu)
        ui = pui if ui is None else ui
        uf = puf if uf is None else uf
    return ui, uf

def check_existing_calculation(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower, d0_upper, 
                             v3_param, v4_param, input_tolerance=1e-10, calc_tolerance=1e-8, filename='sigma_calculations.csv'):
    """
    Check if this exact calculation (parameters + results) already exists in the CSV.
    
    Parameters:
    -----------
    All calculation parameters (v3_param and v4_param are the actual values passed, not globals)
    input_tolerance : float
        Tolerance for input parameter comparison (default: 1e-10)
    calc_tolerance : float  
        Tolerance for calculated value comparison (default: 1e-8)
    filename : str
        CSV file to check
        
    Returns:
    --------
    bool
        True if exact calculation already exists, False otherwise
    """
    data_dir = 'sigma_data'
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        return False
    
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return False
        
        # Convert v3_param, v4_param back to gamma, lambda4 for comparison with CSV
        gamma_param = v3_param * 6 * np.sqrt(2)
        lambda4_param = v4_param
        
        # Create boolean mask for matching rows with different tolerances
        # Strict tolerance for input parameters (should be exact)
        input_mask = (
            (np.abs(df['T'] - T) < input_tolerance) &
            (np.abs(df['mu'] - mu) < input_tolerance) &
            (np.abs(df['ui'] - ui) < input_tolerance) &
            (np.abs(df['uf'] - uf) < input_tolerance) &
            (np.abs(df['mq'] - mq_input) < input_tolerance) &
            (np.abs(df['mq_tolerance'] - mq_tolerance) < input_tolerance) &
            (np.abs(df['lambda1'] - lambda1) < input_tolerance) &
            (np.abs(df['gamma'] - gamma_param) < input_tolerance) &
            (np.abs(df['lambda4'] - lambda4_param) < input_tolerance) &
            (np.abs(df['d0_lower'] - d0_lower) < input_tolerance) &
            (np.abs(df['d0_upper'] - d0_upper) < input_tolerance)
        )
        
        # For now, we only check input parameters for duplicates
        # (Calculated sigma/d0 values might vary slightly due to numerical precision)
        mask = input_mask
        
        # Check if any rows match all parameters
        matching_rows = df[mask]
        if len(matching_rows) > 0:
            print(f"Found {len(matching_rows)} existing calculation(s) with identical parameters")
            print(f"  Using v3={v3_param:.6f}, v4={v4_param:.1f} (gamma={gamma_param:.1f}, lambda4={lambda4_param:.1f})")
            print(f"  Most recent: {matching_rows['timestamp'].max()}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Warning: Could not check existing calculations: {e}")
        return False

def log_sigma_calculation(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower, d0_upper, 
                         v3_param, v4_param, sigma_values, d0_values, filename='sigma_calculations.csv'):
    """
    Log sigma calculation results to a CSV file with file locking and duplicate checking.
    
    Parameters:
    -----------
    All input parameters plus results (v3_param and v4_param are the actual values used)
    """
    # Convert v3_param, v4_param back to gamma, lambda4 for logging
    gamma_param = v3_param * 6 * np.sqrt(2)
    lambda4_param = v4_param
    
    # Check if this exact calculation already exists
    if check_existing_calculation(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, 
                                 d0_lower, d0_upper, v3_param, v4_param, filename=filename):
        print(f"Skipping duplicate calculation for mq={mq_input}, T={T}, mu={mu}, lambda1={lambda1}")
        print(f"  With v3={v3_param:.6f}, v4={v4_param:.1f}")
        return
    
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
        'gamma': gamma_param,
        'lambda4': lambda4_param,
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
    
    # Use file locking to prevent concurrent write issues
    try:
        with open(filepath, 'a', newline='') as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            # Check if file is empty (needs header)
            file_was_empty = f.tell() == 0
            
            # Write data with proper header handling
            if file_was_empty:
                df_new.to_csv(f, header=True, index=False)
                print(f"Created new sigma calculation log: {filepath}")
            else:
                df_new.to_csv(f, header=False, index=False)
            
            # Flush to ensure data is written
            f.flush()
            os.fsync(f.fileno())
            
            # Lock is automatically released when file closes
            
    except Exception as e:
        print(f"Warning: Could not log sigma calculation: {e}")
        # Fallback to old method without locking
        if os.path.exists(filepath):
            df_new.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df_new.to_csv(filepath, mode='w', header=True, index=False)
    
    print(f"Logged sigma calculation: {len(sigma_values)} solutions found")
    print(f"  Parameters: mq={mq_input}, T={T}, mu={mu}, lambda1={lambda1}, v3={v3_param:.6f}, v4={v4_param:.1f}")

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

def chiral_solve_IR(d0, lambda1, T, mu, ui=None, uf=None, v3=v3, v4=v4,
                    solver=None):
    ui, uf = resolve_cutoffs(T, mu, ui, uf)
    settings = solver_settings(solver)
    numpoints = 10000
    u = np.linspace(ui, uf, numpoints)
    u_backward = np.linspace(uf, ui, numpoints)

    zeta = np.sqrt(3)/(2*np.pi)
    mu_g = MU_G
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
    sol = solve_ivp(chiral, [uf, ui], y0, t_eval=u_backward, args=(params,),
                    method=settings['method'], rtol=settings['rtol'],
                    atol=settings['atol'])
    if sol.status != 0 or len(sol.t) != numpoints:
        # The integration stopped early. The old code silently re-gridded the
        # truncated output onto linspace(ui, uf, len(chi)), which quietly moved
        # the extraction point; return NaN so the caller rejects this d0.
        nan = np.full(numpoints, np.nan)
        return np.nan, nan, nan, u
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

def chiral_solve_IR_parallel(d0_array, lambda1, T, mu, ui=None, uf=None, v3=v3, v4=v4,
                             n_jobs=-1, solver=None):
    ui, uf = resolve_cutoffs(T, mu, ui, uf)
    results = Parallel(n_jobs=n_jobs)(
        delayed(chiral_solve_IR)(d0, lambda1, T, mu, ui, uf, v3, v4, solver) for d0 in d0_array
    )
    # Unpack the results
    mq_array, chi_array, chip_array, u_array = zip(*results)
    
    # Fix the deprecation warning by specifying dtype=object
    return np.array(mq_array), np.array(chi_array, dtype=object), np.array(chip_array, dtype=object)

def initial_d0_mq(T, mu, mq_target, lambda1, ui=None, uf=None, d0_array=None, v3=v3, v4=v4,
                  solver=None):
    ui, uf = resolve_cutoffs(T, mu, ui, uf)
    mq_array = chiral_solve_IR_parallel(d0_array, lambda1, T, mu, ui, uf, v3, v4,
                                        n_jobs=-1, solver=solver)[0]

    max_d0 = d0_array[-1]
    # Find and remove runaway solutions, and any d0 whose integration failed
    # (chiral_solve_IR now returns NaN rather than a silently re-gridded result)
    indices = np.where(~(np.abs(mq_array) <= 150))[0]
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
        mq_new = chiral_solve_IR_parallel(d0_new, lambda1, T, mu, ui, uf, v3, v4,
                                          n_jobs=-1, solver=solver)[0]
        mq_array = np.concatenate((mq_array, mq_new))
        
        indices = np.where(~(np.abs(mq_array) <= 150))[0]
        if len(indices) > 0:
            # Just take the first index, no need for further indexing
            max_d0 = d0_array[indices[0]]
        
        d0_array = np.delete(d0_array, indices)
        mq_array = np.delete(mq_array, indices)

        iterations += 1
        
    # Refine if there's a large jump and we have enough points
    if len(mq_array) >= 2 and mq_array[-1] - mq_array[-2] > 5:
        d0_new = np.linspace(d0_array[-2], d0_array[-1], 10)[1:-1]
        mq_new = chiral_solve_IR_parallel(d0_new, lambda1, T, mu, ui, uf, v3, v4,
                                          n_jobs=-1, solver=solver)[0]
        mq_array = np.concatenate((mq_array[:-1], mq_new, mq_array[-1:]))
        d0_array = np.concatenate((d0_array[:-1], d0_new, d0_array[-1:]))
        
    return d0_array, mq_array

def process_mq_target(mq_target, d0_array, mq_improved, sigma_improved, lambda1, T, mu, ui, uf, v3, v4,
                      solver=None):
    ui, uf = resolve_cutoffs(T, mu, ui, uf)
    settings = solver_settings(solver)
    d0_list = d0_array.tolist()
    mq_list = mq_improved.tolist()
    sigma_list = sigma_improved.tolist()
    
    indices = np.where(np.diff(np.sign(mq_improved - mq_target)))[0]
    
    for index in indices:
        mq_approx = np.array([mq_improved[index], mq_improved[index + 1]])
        d0_approx = np.array([d0_array[index], d0_array[index + 1]])
        
        d0_interp = interp1d(mq_approx, d0_approx)
        d0 = d0_interp(mq_target)
        mq, chi, chip, u = chiral_solve_IR(d0, lambda1, T, mu, ui, uf, v3, v4, solver)
        if not np.isfinite(mq):
            continue

        i = settings['extract_index']
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

def new_function(lambda1, T, mu, mq_large, ui=None, uf=None, d0_lower=0, d0_upper=10,
                 v3=v3, v4=v4, numpoints=100, solver=None):
    ui, uf = resolve_cutoffs(T, mu, ui, uf)
    settings = solver_settings(solver)
    # Initial d0 and mq arrays
    d0_array = np.linspace(d0_lower, d0_upper, numpoints)
    d0_array, mq_array = initial_d0_mq(T, mu, mq_large, lambda1, ui, uf, d0_array, v3, v4,
                                       solver=solver)

    # Further processing to get mq_improved and sigma_improved
    _, chis, chips = chiral_solve_IR_parallel(d0_array, lambda1, T, mu, ui, uf, v3, v4,
                                              n_jobs=-1, solver=solver)

    i = settings['extract_index']
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
    results = Parallel(n_jobs=-1)(delayed(process_mq_target)(mq_target, d0_array, mq_improved, sigma_improved, lambda1, T, mu, ui, uf, v3, v4, solver) for mq_target in mq_overlay)

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

def sigma_of_T(mq_input, mq_tolerance, T, mu, lambda1, d0_lower, d0_upper, ui=None, uf=None,
               v3=v3, v4=v4, solver=None):
    ui, uf = resolve_cutoffs(T, mu, ui, uf)
    mq_large = 2*mq_input
    mq_improved, sigma_improved, d0_array = new_function(lambda1, T, mu, mq_large, ui, uf,
                                                         d0_lower, d0_upper, v3, v4,
                                                         numpoints=100, solver=solver)
    
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

def calculate_sigma_values(mq_input, mq_tolerance, T, mu, lambda1, ui=None, uf=None, d0_lower=0, d0_upper=10, v3=v3, v4=v4, log_results=True, solver=None):
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
    ui : float or None
        Lower (UV) bound for u. None uses the T-independent physical scheme,
        u_i = Z_UV/z_h -- see physical_cutoffs.
    uf : float or None
        Upper (IR) bound for u. None uses the physical scheme, which shrinks
        the horizon gap 1-u_f as (mu_g z_h)^2 grows.
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
    solver : None, 'legacy' or dict
        Integration settings; see solver_settings. None uses the accurate
        defaults, 'legacy' reproduces the pre-2026-10 behaviour.
        
    Returns:
    --------
    result : dict
        Dictionary containing sigma values, chiral fields, and derivatives
    """
    
    ui, uf = resolve_cutoffs(T, mu, ui, uf)

    # Check if we already have this calculation (avoid duplicate logging)
    if log_results:
        existing_calculation = check_existing_calculation(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower, d0_upper, v3, v4)
        if existing_calculation:
            print(f"Found existing calculation for mq={mq_input}, T={T}, mu={mu}, lambda1={lambda1}")
            print("Recalculating chiral fields (not saved in CSV) but skipping duplicate logging...")
            log_results = False  # Skip logging since parameters already exist
    
    # Find the sigma values and d0 approximations
    sigma_values, d0_max, d0_min, d0_approx_list = sigma_of_T(
        mq_input, mq_tolerance, T, mu, lambda1, d0_lower, d0_upper, ui, uf, v3, v4,
        solver=solver
    )
    
    # Filter out zero sigma values (padding values)
    non_zero_indices = []
    for i, sigma in enumerate(sigma_values):
        if i < len(d0_approx_list) and sigma != 0:
            non_zero_indices.append(i)
            
    valid_sigma_values = [sigma_values[i] for i in non_zero_indices]
    valid_d0_approx_list = [d0_approx_list[i] for i in non_zero_indices]
    
    # Log the results if requested (and not already existing)
    if log_results:
        log_sigma_calculation(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower, d0_upper,
                            v3, v4, valid_sigma_values, valid_d0_approx_list)
    
    # For each d0, get the corresponding chiral field and derivative
    # (These are always calculated fresh since they're not stored in CSV)
    chiral_fields = []
    chiral_derivatives = []
    u_values = None
    
    for d0 in valid_d0_approx_list:
        _, chi, chip, u = chiral_solve_IR(d0, lambda1, T, mu, ui, uf, v3, v4, solver)
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
    # None -> the T-independent physical cutoff scheme (see physical_cutoffs).
    ui = None
    uf = None
    
    # Calculate sigma values and chiral fields
    result = calculate_sigma_values(mq_input, mq_tolerance, T, mu, lambda1, ui, uf)
    print("Cutoffs used:", physical_cutoffs(T, mu))
    
    # Print results
    print("Sigma values (MeV³):", result["sigma_values"])
    print("d0 values:", result["d0_values"])
    
    # Plot the chiral fields
    plot_chiral_fields(result, mq_input, T, mu, lambda1)