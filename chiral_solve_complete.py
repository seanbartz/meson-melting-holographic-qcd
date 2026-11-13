"""
chiral_solve_complete.py -- chiral background chi(u) of the soft-wall model in the
AdS-Reissner-Nordstrom black hole, and the quark mass m_q and condensate sigma it
carries.

Method: shoot from the horizon (value d0 of chi at u = 1) to the boundary, read off
(m_q, sigma) from the UV series, and root-find d0 so that m_q equals the target.

Numerical scheme (PROJECT_ASSESSMENT.md sections 3.3, 3.3b; same choices as
seanbartz/2p1-critical-point, physics/model.py and physics/series.py)
----------------------------------------------------------------------------------
* Dimensionless, energy unit mu_g.  zhat = mu_g z, b = (mu_g z_h)^2,
  a = m_q zeta / mu_g (source), c = sigma / (zeta mu_g^3) (condensate).  In u = z/z_h
  the equation is

      chi'' = (3/u - f'/f + 2 b u) chi' + F / (u^2 f),
      F = (-3 - lam1 b u^2) chi + 3 v3 chi^2 + 4 v4 chi^3,

  so mu_g and zeta enter only through b and the unit conversion.
* Cutoffs scale with mu_g, not with z_h.  The UV matching point sits at a fixed
  zhat_UV = mu_g z_UV (``SOLVER_SETTINGS['z_uv']``), i.e. at u = zhat_UV / sqrt(b), not
  at a fixed u.  The old fixed-u cutoff put the renormalization point at
  z_UV = u_i z_h, which drifts like 1/T.  The horizon start 1 - u is chosen from the
  convergence of a 10th-order horizon series (coefficients grow like b^n), so it moves
  toward the horizon by itself at low T instead of sitting at a fixed 1 - u_f.
* UV extraction with the logarithms.  chi = a zhat + B zhat^2 + (c + D ln zhat) zhat^3
  + (E + F ln zhat) zhat^4 + (G + H ln zhat) zhat^5 (chiral_uv_series_generated.py,
  derived by derivations/derive_chiral_uv_series.py) is inverted for (a, c) by Newton
  iteration.  The old two-term inversion dropped D ln zhat, so sigma absorbed
  zeta mu_g^3 D ln(zhat_i).  sigma is now independent of the matching point.  Its
  scheme is "mu_g inside the logarithm"; ``condensate_in_scheme`` converts to another
  scale (the shift is proportional to m_q and does not depend on T or mu).
* Tolerances.  DOP853 at rtol = 1e-11 instead of solve_ivp's RK45 defaults
  (rtol = 1e-3), which moved sigma^(1/3) by up to 1% and diverged at T = 17 MeV.
* Root finding.  Each bracket of m_q(d0) - m_q_target from the d0 scan is solved with
  Brent's method instead of a linear interpolation across the bracket.
* A failed or divergent shoot returns NaN; it is never re-gridded or saved as data.

The arguments ``ui`` and ``uf`` of the public functions only set the u grid on which
chi(u) is returned to the spectral codes; they no longer affect m_q or sigma.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from joblib import Parallel, delayed
import pandas as pd
import os
from datetime import datetime
import fcntl  # For file locking

from chiral_uv_series_generated import uv_series, uv_log_coefficient
import meson_core as mc

# Constants from the paper
v4 = 4.2
v3 = -22.6/(6*np.sqrt(2))

# Units: the only place where MeV enters the solver
MU_G = 440.0                       # dilaton scale mu_g [MeV]
ZETA = np.sqrt(3) / (2 * np.pi)    # normalization of the UV coefficients, sqrt(N_c)/(2 pi)

# Numerical settings of one shoot.  The tests vary each of them.
SOLVER_SETTINGS = {
    'z_uv': 0.005,              # zhat = mu_g z of the UV matching point (fixed in physical units)
    'horizon_order': 10,        # order of the Taylor series at the horizon
    'horizon_series_tol': 1e-13,  # allowed relative truncation error of that series
    'horizon_eps_max': 0.05,    # largest allowed 1 - u at the start of the integration
    'method': 'numba',          # compiled DOP853 (meson_core.dop853); 'DOP853' etc. = scipy solve_ivp (oracle)
    'rtol': 1e-11,
    'atol': 1e-16,
    'chi_blowup': 1e3,          # |chi| beyond this: the shoot has run away (no physical solution)
}

# Version tag written to the sigma log, so results from the old scheme are never mixed in
SOLVER_SCHEME = 'zhat_uv_log_v1'
SIGMA_LOG_FILE = 'sigma_calculations_v2.csv'
def check_existing_calculation(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower, d0_upper, 
                             v3_param, v4_param, input_tolerance=1e-10, calc_tolerance=1e-8, filename=SIGMA_LOG_FILE):
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
            (np.abs(df['d0_upper'] - d0_upper) < input_tolerance) &
            (np.abs(df['z_uv'] - SOLVER_SETTINGS['z_uv']) < input_tolerance) &
            (df['scheme'] == SOLVER_SCHEME)
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
                         v3_param, v4_param, sigma_values, d0_values, filename=SIGMA_LOG_FILE):
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
        'd0_3': d0_padded[2],
        'z_uv': SOLVER_SETTINGS['z_uv'],
        'scheme': SOLVER_SCHEME,
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

def load_sigma_data(filename=SIGMA_LOG_FILE):
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

def sigma_data_summary(filename=SIGMA_LOG_FILE):
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

# ---------------------------------------------------------------------------
# Background: (T, mu) -> horizon
# ---------------------------------------------------------------------------
def blackness(T, mu):
    """z_h [1/MeV] and q [MeV^3] of the AdS-RN black hole (kappa = 1), Q = q z_h^3 = mu z_h."""
    kappa = 1

    if mu == 0:
        zh = 1/(math.pi*T)
        q = 0
    else:
        zh = kappa/mu**2*(-kappa*math.pi*T+math.sqrt((kappa*math.pi*T)**2+2*mu**2))
        q = (mu**3+kappa*math.pi*T*mu*(kappa*math.pi*T+math.sqrt((kappa*math.pi*T)**2+2*mu**2)))/(2*kappa**3)

    return zh, q


def horizon_hat(T, mu, mu_g=MU_G):
    """
    Dimensionless horizon data: b = (mu_g z_h)^2, Q^2 and zhat_h = mu_g z_h.

    Inverts T = (1 - Q^2/2)/(pi z_h), mu = Q/z_h in the form that stays accurate as
    mu -> 0 (equal to ``blackness``).
    """
    T_hat = T / mu_g
    mu_hat = mu / mu_g
    zh_hat = 2.0 / (np.pi * T_hat + np.sqrt(np.pi**2 * T_hat**2 + 2.0 * mu_hat**2))
    Q2 = (mu_hat * zh_hat)**2
    return zh_hat**2, Q2, zh_hat


def chiral(u, y, b, Q2, lambda1, lam3, lam4):
    """Right-hand side of the chiral EOM in u = z/z_h, written with b = (mu_g z_h)^2."""
    chi, chip = y
    f = 1 - (1 + Q2)*u**4 + Q2*u**6
    fp = -4*(1 + Q2)*u**3 + 6*Q2*u**5
    phip = 2*b*u
    force = (-3 - lambda1*b*u**2)*chi + 3*lam3*chi**2 + 4*lam4*chi**3
    return [chip, (3/u - fp/f + phip)*chip + force/(u**2*f)]


# ---------------------------------------------------------------------------
# Horizon series: chi = sum_n d_n x^n, x = 1 - u   (compiled, meson_core.py)
# ---------------------------------------------------------------------------
def horizon_coefficients(d0, b, Q2, lambda1, lam3, lam4, order):
    """
    Taylor coefficients d_0..d_order of the regular solution at the horizon, by
    recursion (any order; no typed-in formulas).  d_1 and d_2 equal the closed forms
    used before (checked in tests/test_numerical_scheme.py).
    """
    return mc.horizon_coefficients(float(d0), float(b), float(Q2), float(lambda1), float(lam3), float(lam4), int(order))


def horizon_evaluate(d, eps):
    """[chi, dchi/du] at u = 1 - eps (eps may be an array)."""
    eps = np.asarray(eps, dtype=float)
    n = np.arange(d.size)
    chi = np.polynomial.polynomial.polyval(eps, d)
    dchi = -np.polynomial.polynomial.polyval(eps, (n[1:] * d[1:]))
    return chi, dchi


def choose_horizon_eps(d, tol, eps_max):
    """Largest eps <= eps_max with |d_N| eps^N <= tol |d_0|."""
    return mc.choose_horizon_eps(d, tol, eps_max)


# ---------------------------------------------------------------------------
# UV extraction
# ---------------------------------------------------------------------------
def extract_uv(z, chi, dchi_dz, lambda1, lam3, lam4, k4):
    """
    Invert the UV series at zhat = z for (a, c) by Newton iteration.
    Returns (a, c, converged).
    """
    target = np.array([chi, dchi_dz])
    # leading order: chi = a z + c z^3
    p = np.array([(3.0 * chi - z * dchi_dz) / (2.0 * z), (z * dchi_dz - chi) / (2.0 * z**3)])
    size = abs(chi) + abs(z * dchi_dz)
    scale = np.array([size / z, size / z**3]) + 1e-300
    for _ in range(40):
        S, JS = uv_series(z, p[0], p[1], lambda1, lam3, lam4, k4)
        try:
            step = np.linalg.solve(JS, target - S)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, False
        p = p + step
        rel = np.max(np.abs(step) / scale)
        if not np.isfinite(rel):
            return np.nan, np.nan, False
        if rel < 1e-13:
            return p[0], p[1], True
    return p[0], p[1], False


def condensate_in_scheme(sigma, mq, lambda1, v3_param=v3, v4_param=v4, scale_MeV=1.0, mu_g=MU_G):
    """
    Convert sigma from the code's scheme (mu_g inside the UV logarithm) to the scheme
    with scale M = ``scale_MeV`` inside the logarithm: c_M = c - D ln(M / mu_g).
    The default M = 1 MeV is the ln(z [MeV^-1]) of the legacy criticalZoom.py.
    The shift depends on m_q only, not on T or mu.
    """
    a = mq * ZETA / mu_g
    D = uv_log_coefficient(a, lambda1, v3_param, v4_param)
    return sigma - ZETA * mu_g**3 * D * np.log(scale_MeV / mu_g)


# ---------------------------------------------------------------------------
# One shoot
# ---------------------------------------------------------------------------
def _core_args(s):
    return (float(s['z_uv']), int(s['horizon_order']), float(s['horizon_series_tol']),
            float(s['horizon_eps_max']), float(s['rtol']), float(s['atol']), float(s['chi_blowup']))


def chiral_shoot(d0, lambda1, T, mu, v3=v3, v4=v4, u_grid=None, settings=None, mu_g=MU_G):
    """
    Integrate from the horizon value d0 to the UV and extract (m_q, sigma).

    Returns a dict with mq [MeV], sigma [MeV^3], a, c, ok, and, if ``u_grid`` is given
    (ascending), chi and chip = dchi/du on that grid.  A failed or divergent shoot has
    ok = False and mq = sigma = NaN (and NaN fields).

    settings['method'] = 'numba' (default) uses the compiled integrator of meson_core;
    any scipy method name ('DOP853', ...) uses solve_ivp, kept as the oracle.
    """
    s = dict(SOLVER_SETTINGS)
    if settings:
        s.update(settings)
    if s['method'] != 'numba':
        return _chiral_shoot_scipy(d0, lambda1, T, mu, v3, v4, u_grid, s, mu_g)
    b, Q2, zh_hat = horizon_hat(T, mu, mu_g)
    a, c, status, eps = mc.chiral_shoot_core(float(d0), b, Q2, float(lambda1), float(v3), float(v4), *_core_args(s))
    out = {'mq': np.nan, 'sigma': np.nan, 'a': np.nan, 'c': np.nan, 'ok': False,
           'd0': float(d0), 'u_match': s['z_uv'] / zh_hat, 'horizon_eps': eps}
    if u_grid is not None:
        out['chi'] = np.full(len(u_grid), np.nan)
        out['chip'] = np.full(len(u_grid), np.nan)
    if status != 0:
        return out
    out.update(a=a, c=c, mq=a * mu_g / ZETA, sigma=ZETA * mu_g**3 * c, ok=True)
    if u_grid is not None:
        u_grid = np.ascontiguousarray(u_grid, dtype=float)
        chi_g, chip_g, st = mc.chiral_on_grid(float(d0), b, Q2, float(lambda1), float(v3), float(v4),
                                              u_grid, *_core_args(s)[1:])
        if st == 0:
            out['chi'], out['chip'] = chi_g, chip_g
        else:
            out['ok'] = False
    return out


def _chiral_shoot_scipy(d0, lambda1, T, mu, v3, v4, u_grid, s, mu_g):
    """The same shoot with scipy's solve_ivp (reference implementation)."""
    b, Q2, zh_hat = horizon_hat(T, mu, mu_g)
    k4 = (1.0 + Q2) / zh_hat**4

    d = horizon_coefficients(float(d0), b, Q2, lambda1, v3, v4, s['horizon_order'])
    eps = choose_horizon_eps(d, s['horizon_series_tol'], s['horizon_eps_max'])
    chi0, chip0 = horizon_evaluate(d, eps)
    u_start = 1.0 - eps

    u_x = s['z_uv'] / zh_hat                    # UV matching point in u
    u_end = u_x if u_grid is None else min(u_x, float(np.min(u_grid)))
    out = {'mq': np.nan, 'sigma': np.nan, 'a': np.nan, 'c': np.nan, 'ok': False,
           'd0': float(d0), 'u_match': u_x, 'horizon_eps': eps}
    if u_grid is not None:
        out['chi'] = np.full(len(u_grid), np.nan)
        out['chip'] = np.full(len(u_grid), np.nan)
    if not (0.0 < u_x < u_start):
        return out

    def blowup(u, y, *args):
        return abs(y[0]) - s['chi_blowup']
    blowup.terminal = True

    try:
        sol = solve_ivp(chiral, [u_start, u_end], [chi0, chip0], method=s['method'],
                        rtol=s['rtol'], atol=s['atol'], dense_output=True,
                        events=blowup, args=(b, Q2, lambda1, v3, v4))
    except Exception:
        return out
    if sol.status != 0 or not np.all(np.isfinite(sol.y[:, -1])):
        return out

    chi_x, chip_x = sol.sol(u_x)
    a, c, converged = extract_uv(s['z_uv'], chi_x, chip_x / zh_hat, lambda1, v3, v4, k4)
    if not converged:
        return out
    out.update(a=a, c=c, mq=a * mu_g / ZETA, sigma=ZETA * mu_g**3 * c, ok=True)

    if u_grid is not None:
        u_grid = np.asarray(u_grid, dtype=float)
        inside = u_grid <= u_start
        chi_g = np.empty_like(u_grid)
        chip_g = np.empty_like(u_grid)
        if np.any(inside):
            y = sol.sol(u_grid[inside])
            chi_g[inside], chip_g[inside] = y[0], y[1]
        if np.any(~inside):  # between the integration start and the horizon: the series
            chi_g[~inside], chip_g[~inside] = horizon_evaluate(d, 1.0 - u_grid[~inside])
        out['chi'], out['chip'] = chi_g, chip_g
    return out


def chiral_solve_IR(d0, lambda1, T, mu, ui, uf, v3=v3, v4=v4):
    """
    Backward-compatible wrapper: (mq, chi, chip, u) with chi, chip = dchi/du on
    u = linspace(ui, uf, 10000).  ui and uf only set that grid.
    """
    u = np.linspace(ui, uf, 10000)
    res = chiral_shoot(d0, lambda1, T, mu, v3, v4, u_grid=u)
    return res['mq'], res['chi'], res['chip'], u


def _shoot_mq_sigma(d0, lambda1, T, mu, v3, v4):
    res = chiral_shoot(d0, lambda1, T, mu, v3, v4)
    return res['mq'], res['sigma']


def chiral_scan(d0_array, lambda1, T, mu, v3=v3, v4=v4, n_jobs=-1, mu_g=MU_G):
    """
    (mq, sigma) arrays along d0_array (NaN where the shoot fails).  The loop is compiled
    (about 30 microseconds per shoot); n_jobs is kept for the old signature.
    """
    d0_array = np.ascontiguousarray(d0_array, dtype=float)
    if len(d0_array) == 0:
        return np.array([]), np.array([])
    if SOLVER_SETTINGS['method'] != 'numba':
        results = [_shoot_mq_sigma(d0, lambda1, T, mu, v3, v4) for d0 in d0_array]
        mq, sigma = zip(*results)
        return np.array(mq, dtype=float), np.array(sigma, dtype=float)
    b, Q2, _ = horizon_hat(T, mu, mu_g)
    a, c = mc.chiral_scan_core(d0_array, b, Q2, float(lambda1), float(v3), float(v4), *_core_args(SOLVER_SETTINGS))
    return a * mu_g / ZETA, ZETA * mu_g**3 * c


def chiral_solve_IR_parallel(d0_array, lambda1, T, mu, ui, uf, v3=v3, v4=v4, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(chiral_solve_IR)(d0, lambda1, T, mu, ui, uf, v3, v4) for d0 in d0_array
    )
    mq_array, chi_array, chip_array, u_array = zip(*results)
    return np.array(mq_array, dtype=float), np.array(chi_array, dtype=object), np.array(chip_array, dtype=object)


# ---------------------------------------------------------------------------
# Scan in d0 and root finding
# ---------------------------------------------------------------------------
MQ_RUNAWAY = 150.0   # MeV; |m_q| beyond this (or NaN) marks the end of the useful d0 range


def initial_d0_mq(T, mu, mq_target, lambda1, ui, uf, d0_array, v3=v3, v4=v4):
    """
    Scan m_q(d0) on d0_array, cut at the first runaway (|m_q| > 150 MeV or a failed
    shoot), and refine toward that edge until m_q exceeds mq_target.  At low T the
    whole physical range of d0 crowds against that edge (assessment §3.3b item 4).
    ui, uf are unused (kept for the old signature).
    """
    d0_array = np.asarray(d0_array, dtype=float)
    mq_array, _ = chiral_scan(d0_array, lambda1, T, mu, v3, v4)

    def cut(d0s, mqs, max_d0):
        bad = np.where(~np.isfinite(mqs) | (np.abs(mqs) > MQ_RUNAWAY))[0]
        if len(bad) > 0:
            max_d0 = min(max_d0, d0s[bad[0]])
            d0s, mqs = d0s[:bad[0]], mqs[:bad[0]]
        return d0s, mqs, max_d0

    d0_array, mq_array, max_d0 = cut(d0_array, mq_array, d0_array[-1])
    if len(d0_array) < 2:
        return d0_array, mq_array

    step_size = d0_array[1] - d0_array[0]
    iterations = 0
    while mq_array[-1] < mq_target and iterations < 20:
        step_size = step_size / 100
        # at most 200 new points (built without allocating the whole range up to max_d0,
        # which has ~100^iterations points)
        d0_new = d0_array[-1] + step_size * np.arange(1, 201)
        d0_new = d0_new[d0_new < max_d0]
        if len(d0_new) == 0:
            break
        mq_new, _ = chiral_scan(d0_new, lambda1, T, mu, v3, v4)
        d0_new, mq_new, max_d0 = cut(d0_new, mq_new, max_d0)
        d0_array = np.concatenate((d0_array, d0_new))
        mq_array = np.concatenate((mq_array, mq_new))
        iterations += 1

    return d0_array, mq_array


def solve_d0_for_mq(mq_target, d0_left, d0_right, lambda1, T, mu, v3=v3, v4=v4):
    """Brent root of m_q(d0) = mq_target in [d0_left, d0_right]; returns (d0, sigma) or None."""
    def g(d0):
        mq = chiral_shoot(d0, lambda1, T, mu, v3, v4)['mq']
        if not np.isfinite(mq):
            raise ValueError("shoot failed inside the bracket")
        return mq - mq_target
    try:
        d0 = brentq(g, d0_left, d0_right, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)
    except (ValueError, RuntimeError):
        return None
    res = chiral_shoot(d0, lambda1, T, mu, v3, v4)
    if not res['ok']:
        return None
    return d0, res['sigma']


def new_function(lambda1, T, mu, mq_large, ui, uf, d0_lower, d0_upper, v3=v3, v4=v4, numpoints=100):
    """(mq, sigma, d0) along the d0 scan (kept for the old interface)."""
    d0_array = np.linspace(d0_lower, d0_upper, numpoints)
    d0_array, _ = initial_d0_mq(T, mu, mq_large, lambda1, ui, uf, d0_array, v3, v4)
    mq_array, sigma_array = chiral_scan(d0_array, lambda1, T, mu, v3, v4)
    ok = np.isfinite(mq_array) & np.isfinite(sigma_array)
    return mq_array[ok], sigma_array[ok], d0_array[ok]


def sigma_of_T(mq_input, mq_tolerance, T, mu, lambda1, d0_lower, d0_upper, ui, uf, v3=v3, v4=v4):
    """
    All sigma with m_q(d0) = mq_input.  Every sign change of m_q - mq_input along the
    d0 scan is solved with Brent's method (to machine precision; mq_tolerance is kept
    for the old signature).  Returns (sigma_values padded with zeros to length 3,
    max d0 scanned, min d0 scanned, list of d0 roots).
    """
    d0_array = np.linspace(d0_lower, d0_upper, 100)
    d0_array, mq_array = initial_d0_mq(T, mu, 2 * mq_input, lambda1, ui, uf, d0_array, v3, v4)

    sigma_list = []
    d0_list = []
    diff = mq_array - mq_input
    for i in range(len(d0_array) - 1):
        if diff[i] == 0.0:
            root = (d0_array[i], chiral_shoot(d0_array[i], lambda1, T, mu, v3, v4)['sigma'])
        elif diff[i] * diff[i + 1] < 0:
            root = solve_d0_for_mq(mq_input, d0_array[i], d0_array[i + 1], lambda1, T, mu, v3, v4)
        else:
            continue
        if root is not None and np.isfinite(root[1]):
            d0_list.append(root[0])
            sigma_list.append(root[1])

    sigma_values = np.array(sigma_list, dtype=float)
    if len(sigma_values) < 3:
        sigma_values = np.pad(sigma_values, (0, 3 - len(sigma_values)), 'constant')
    d0_max = np.max(d0_array) if len(d0_array) > 0 else 0
    d0_min = np.min(d0_array) if len(d0_array) > 0 else 0
    return sigma_values, d0_max, d0_min, d0_list

def calculate_sigma_values(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower=0, d0_upper=10, v3=v3, v4=v4, log_results=True):
    """
    Calculate sigma values and corresponding chiral fields for a given quark mass.

    Parameters:
    -----------
    mq_input : float
        Target quark mass in MeV
    mq_tolerance : float
        Kept for the old interface and the log; roots are now solved to machine precision
    T, mu : float
        Temperature and quark chemical potential in MeV
    lambda1 : float
        Lambda1 parameter
    ui, uf : float
        Range of the u grid on which chi(u) is returned.  They do NOT affect sigma:
        the UV matching point is at fixed mu_g z (SOLVER_SETTINGS['z_uv']) and the
        horizon start is chosen from the horizon series.
    d0_lower, d0_upper : float
        Range of the d0 scan
    v3, v4 : float
        Chiral couplings (v3 = gamma/(6 sqrt 2), v4 = lambda4)
    log_results : bool
        Whether to log the results to sigma_data/SIGMA_LOG_FILE

    Returns:
    --------
    result : dict
        sigma_values [MeV^3] (scheme: mu_g inside the UV logarithm), d0_values,
        chiral_fields, chiral_derivatives (d/du) and u_values.  Only solutions whose
        fields are finite on the whole grid are returned.
    """

    if log_results:
        existing_calculation = check_existing_calculation(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower, d0_upper, v3, v4)
        if existing_calculation:
            print(f"Found existing calculation for mq={mq_input}, T={T}, mu={mu}, lambda1={lambda1}")
            print("Recalculating chiral fields (not saved in CSV) but skipping duplicate logging...")
            log_results = False

    sigma_values, d0_max, d0_min, d0_approx_list = sigma_of_T(
        mq_input, mq_tolerance, T, mu, lambda1, d0_lower, d0_upper, ui, uf, v3, v4
    )

    u_values = np.linspace(ui, uf, 10000)
    valid_sigma_values = []
    valid_d0_approx_list = []
    chiral_fields = []
    chiral_derivatives = []
    for i, d0 in enumerate(d0_approx_list):
        res = chiral_shoot(d0, lambda1, T, mu, v3, v4, u_grid=u_values)
        if not res['ok'] or not np.all(np.isfinite(res['chi'])):
            print(f"Warning: dropping solution d0={d0}: chiral field not finite on the grid")
            continue
        valid_sigma_values.append(sigma_values[i])
        valid_d0_approx_list.append(d0)
        chiral_fields.append(res['chi'])
        chiral_derivatives.append(res['chip'])

    if log_results and valid_sigma_values:
        log_sigma_calculation(mq_input, mq_tolerance, T, mu, lambda1, ui, uf, d0_lower, d0_upper,
                              v3, v4, valid_sigma_values, valid_d0_approx_list)
    elif log_results:
        print(f"No valid chiral solution for mq={mq_input}, T={T}, mu={mu}, lambda1={lambda1}; nothing logged")

    return {
        "sigma_values": np.array(valid_sigma_values),
        "d0_values": np.array(valid_d0_approx_list),
        "chiral_fields": chiral_fields,
        "chiral_derivatives": chiral_derivatives,
        "u_values": u_values if chiral_fields else None,
    }


def plot_chiral_fields(result, mq_input, T, mu, lambda1, v3_param=v3, v4_param=v4):
    """
    Plot the chiral fields and their UV series (with the log terms).
    """
    if "sigma_values" not in result or len(result["sigma_values"]) == 0:
        print("No valid solutions found to plot")
        return

    b, Q2, zh_hat = horizon_hat(T, mu)
    k4 = (1.0 + Q2) / zh_hat**4
    u = result["u_values"]
    zhat = zh_hat * u

    plt.figure(figsize=(10, 6))
    for i, chi in enumerate(result["chiral_fields"]):
        plt.loglog(u, np.abs(chi), label=f"$\\chi_{i+1}$ solution")

    a = mq_input * ZETA / MU_G
    for i, sigma in enumerate(result["sigma_values"]):
        c = sigma / (ZETA * MU_G**3)
        uv = np.array([uv_series(zz, a, c, lambda1, v3_param, v4_param, k4)[0][0] for zz in zhat])
        mask = zhat < 0.5
        plt.loglog(u[mask], np.abs(uv[mask]), '--',
                   label=f"UV series ($\\sigma^{{1/3}}$={np.cbrt(sigma):.3f} MeV)")

    plt.xlabel('$u$')
    plt.ylabel('$|\\chi|$')
    plt.title(f'$\\lambda_1={lambda1}$, quark mass={mq_input} MeV')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    mq_input = 10  # MeV
    mq_tolerance = 0.1
    T = 10
    mu = 0
    lambda1 = 7
    ui = 1e-2
    uf = 1 - 1e-4

    result = calculate_sigma_values(mq_input, mq_tolerance, T, mu, lambda1, ui, uf)

    print("Sigma values (MeV³):", result["sigma_values"])
    print("d0 values:", result["d0_values"])

    plot_chiral_fields(result, mq_input, T, mu, lambda1)
