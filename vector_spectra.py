import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
import os
import datetime
import argparse
import meson_core as mc

# Global parameters
mu = 0  # MeV, Chemical Potential
T = 50  # MeV, Temperature
kappa = 1  # Constant
q = 0

# Integration parameters
ui = 1e-4
uf = 1-(1e-5)
ucount = 1000

# Frequency range (now passed as parameters)
# wi = 700
# wf = 1800
# wcount = 1100
# wresolution = .1  # MeV (no longer used as global default)

#Blackness function. Returns f(u) and f'(u) as a list
def f(u, Q):
    f = 1-((1+(Q**2))*(u**4))+((Q**2)*(u**6))
    f_p = -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    return [f, f_p]
    
#Dilaton field.
def phi(u, zh, mug):
    phi = (mug**2)*(zh**2)*(u**2)
    phi_p = 2*u*(mug**2)*(zh**2)
    return [phi, phi_p]
    
#440 MeV. The confinement scale
def mu_g():
    return 440 #440
    
#Related to black hole charge. Must return a value from 0 < Q^2 < 2
#Manually checked for mu = 0
def Q_func(zh):
    global q #Prevents function from interpretting q as a local variable
    
    if mu == 0:
        q = 0
    else:
        if q == 0:
            q = (mu**3 + kappa * np.pi * T * mu * (kappa * np.pi * T + np.sqrt((kappa * np.pi * T)**2 + 2 * mu**2))) / (2 * kappa**3)
    return mu * zh / kappa

#Black Hole Horizon. calculated from T, mu, and k
#Manually checked for mu = 0
def z_h():
    if mu == 0:
        return 1 / (np.pi * T)
    
    return ((-T * np.pi * (kappa**2)) + ((kappa**2) * np.sqrt((T**2 * np.pi**2) + (2 * mu**2 / kappa**2)))) / mu**2   

#Values of lambda 1-4
def lam_function():
    return [7.438, 0.0, -22.6/(6*(2**(1/2))), 4.2]

def calculate_variables():
    mug = mu_g()
    zh = z_h()
    Q = Q_func(zh)
    lam = lam_function()
    
    return (mug, zh, Q, lam)

#Coefficients
def c2():
    return 1

#Manually checked w/ calculator, correct output
def c4(w, zh, mug, c_2):
    c_4 = (1/8)*(
        (4*c_2*(zh**2)*(mug**2))
        -(c_2*(zh**2)*(w**2))
    )
    return c_4

#Manually checked w/ calculator, correct output
def c6(w, zh, mug, Q, c_2, c_4):
    c_6 = (1/24)*(
        (8*c_2)
        + (8*c_2*(Q**2))
        + (8*c_4*(zh**2)*(mug**2))
        - (c_4*(zh**2)*(w**2))
    )
    return c_6

def d0():
    return 1

def d2():
    return 0

#Manually checked w/ calculator, correct output
def d(w, zh, d_0):
    return -(d_0*(zh**2)*(w**2))/(2*c2())

#Manually checked w/ calculator, correct output
def d4(w, zh, mug, d_0, d_2):
    return (1/64)*(
        (32*d_2*(zh**2)*(mug**2))
        - (8*d_2*(zh**2)*(w**2))
        + (4*d_0*(zh**4)*(mug**2)*(w**2))
        - (3*d_0*(zh**4)*(w**4))
    )

#Manually checked w/ calculator, correct output
def d6(w, zh, mug, Q, d_0, d_2, d_4):
    return (1/4608)*(
        (1536*d_2)
        + (1536*d_2*(Q**2))
        + (1536*d_4*(zh**2)*(mug**2))
        - (448*d_0*(zh**2)*(w**2))
        - (192*d_4*(zh**2)*(w**2))
        - (448*d_0*(Q**2)*(zh**2)*(w**2))
        + (64*d_0*(zh**6)*(mug**4)*(w**2))
        - (36*d_0*(zh**6)*(mug**2)*(w**4))
        + (5*d_0*(zh**6)*(w**6))
    )

def coefficients(w, zh, mug, Q):
    c_2 = c2()
    c_4 = c4(w, zh, mug, c_2)
    c_6 = c6(w, zh, mug, Q, c_2, c_4)
    
    d_0 = d0()
    d_2 = d2()
    d_4 = d4(w, zh, mug, d_0, d_2)
    d_6 = d6(w, zh, mug, Q, d_0, d_2, d_4)
    d_d = d(w, zh, d_0)
    
    return (c_2, c_4, c_6, d_0, d_2, d_4, d_6, d_d)

#Scalar Field Solutions
def v1(u, cs):
    c_2, c_4, c_6 = cs
    
    v_1 = (c_2*(u**2))+(c_4*(u**4))+(c_6*(u**6))
    v_1p = (2*c_2*u)+(4*c_4*(u**3))+(6*c_6*(u**5))

    return [v_1, v_1p]

def v2(u, v, ds):
    d_0, d_2, d_4, d_6, d_d = ds
    v_2 = (d_0
        +(d_2*(u**2))
        +(d_4*(u**4))
        +(d_6*(u**6))
        +(d_d*v[0]*np.log(u)))
    
    v_2p = ((2*d_2*(u))
        +(4*d_4*(u**3))
        +(6*d_6*(u**5))
        +(d_d*((v[1]*np.log(u))+(v[0]*(1/u))))
           )
    return [v_2, v_2p]

def vs(u, w, zh, mug, Q):
    c_2, c_4, c_6, d_0, d_2, d_4, d_6, d_d = coefficients(w, zh, mug, Q)
    
    cs = (c_2, c_4, c_6)
    ds = (d_0, d_2, d_4, d_6, d_d)
    
    v_1 = v1(u, cs)
    v_2 = v2(u, v_1, ds)
    return (v_1, v_2)

#Vector Equation of Motion; Must be solved using solve_ivp
def vector_eom(v, u, w, Q, zh, mug):
    
    fu = f(u, Q)
    phiu = phi(u, zh, mug)
    
    v1, v2 = v
    
    v1p = v2
    v2p = ((v2*(
            (1/u)-(fu[1]/fu[0])+(phiu[1])
        ))
        -(
            v1*(
                (w**2)*(zh**2)/(fu[0]**2)
            )
        ))
    
    return [v1p, v2p]

#Asymptotic Solution at the Horizon
def psi_m(u, w, zh, Q):
    c = -complex(0, 1)*w*zh/(2*(2-(Q**2)))
    
    psim = (1-u)**(c)
    psim_p = -c*((1-u)**(c-1))
    
    return [psim, psim_p]

# Tolerances of the spectral integration (was LSODA at rtol 1e-8 with two separate solves)
SPECTRAL_RTOL = 1e-10
SPECTRAL_ATOL = 1e-30


def wronskian_growth(ui, uf, zh, mug, Q):
    """
    W(uf)/W(ui) for psi'' = (1/u - f'/f + phi') psi' + (...) psi (Abel's identity):
    W(uf) = W(ui) (uf/ui) (f(ui)/f(uf)) exp[phi(uf) - phi(ui)].
    """
    return (uf/ui)*(f(ui, Q)[0]/f(uf, Q)[0])*np.exp(phi(uf, zh, mug)[0] - phi(ui, zh, mug)[0])


def BA_from_horizon_data(psi1, psi1p, psi2, psi2p, W_uf, w, zh, Q, uf):
    """
    B/A = (psi2' - i k psi2)/(psi1' - i k psi1), k = w zh/(2(2-Q^2)(1-uf)), with the
    imaginary part k W(uf)/(psi1'^2 + k^2 psi1^2) evaluated with the exact Wronskian
    (PROJECT_ASSESSMENT.md §3.4b; same as axial_spectra.BA_from_horizon_data).
    """
    k = w*zh/(2*(2 - Q**2)*(1 - uf))
    denom = psi1p**2 + (k*psi1)**2
    return complex((psi1p*psi2p + k**2*psi1*psi2)/denom, k*W_uf/denom)


_NO_CHI = np.zeros(2)


def BA_array(ws, Q, zh, mug):
    """
    B/A for an array of frequencies with the compiled integrator (meson_core), threaded
    over frequencies.  Same initial data (UV series vs) and same formula as ``BA_scipy``.
    """
    ws = np.atleast_1d(np.asarray(ws, dtype=float))
    y0s = np.empty((len(ws), 4))
    for i, w in enumerate(ws):
        v_1, v_2 = vs(ui, w, zh, mug, Q)
        y0s[i] = (v_1[0], v_1[1], v_2[0], v_2[1])
    ys, status = mc.legacy_two_solutions_array(y0s, float(ui), float(uf), (ws*zh)**2, (mug*zh)**2, float(Q)**2, 0.0,
                                               0.0, 1.0, _NO_CHI, _NO_CHI, SPECTRAL_RTOL, SPECTRAL_ATOL)
    growth = wronskian_growth(ui, uf, zh, mug, Q)
    out = np.empty(len(ws), dtype=complex)
    for i, w in enumerate(ws):
        if status[i] != 0 or not np.all(np.isfinite(ys[i])):
            # Raise rather than return NaN: a NaN point would reach peak finding and the saved CSV
            raise RuntimeError(f"vector integration failed at w = {w} MeV (status {status[i]})")
        W_ui = y0s[i, 0]*y0s[i, 3] - y0s[i, 2]*y0s[i, 1]
        out[i] = BA_from_horizon_data(ys[i, 0], ys[i, 1], ys[i, 2], ys[i, 3], W_ui*growth, w, zh, Q, uf)
        if not np.isfinite(out[i].real) or not np.isfinite(out[i].imag):
            raise RuntimeError(f"vector spectral function not finite at w = {w} MeV: {out[i]}")
    return out


#Ratio of B and A
def BA(w, Q, zh, mug):
    """B/A at one frequency (compiled integrator)."""
    return complex(BA_array([w], Q, zh, mug)[0])


def BA_scipy(w, Q, zh, mug):
    """The same with scipy's solve_ivp (reference path)."""
    v_1, v_2 = vs(ui, w, zh, mug, Q)

    def eom2(u, y):
        d1 = vector_eom(y[0:2], u, w, Q, zh, mug)
        d2 = vector_eom(y[2:4], u, w, Q, zh, mug)
        return [d1[0], d1[1], d2[0], d2[1]]

    sol = solve_ivp(eom2, [ui, uf], [v_1[0], v_1[1], v_2[0], v_2[1]],
                    method='DOP853', rtol=SPECTRAL_RTOL, atol=SPECTRAL_ATOL)
    psi1, psi1p, psi2, psi2p = sol.y[:, -1]
    if sol.status != 0 or not np.all(np.isfinite(sol.y[:, -1])):
        raise RuntimeError(f"vector integration failed at w = {w} MeV: {sol.message}")

    W_ui = v_1[0]*v_2[1] - v_2[0]*v_1[1]
    W_uf = W_ui*wronskian_growth(ui, uf, zh, mug, Q)
    result = BA_from_horizon_data(psi1, psi1p, psi2, psi2p, W_uf, w, zh, Q, uf)
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise RuntimeError(f"vector spectral function not finite at w = {w} MeV: {result}")
    return result

#calculate the data set
def calculate_data(wi, wf, wcount, wresolution, Q, zh, mug, expected_peaks=None, rel_prominence=0.02):
    """
    Spectral function on a grid of wcount points in [wi, wf], and its peaks.  Same peak
    criterion and refinement as axial_spectra.calculate_data: relative prominence of
    |Im(B/A)|/omega^2, each peak located to wresolution by a bounded maximization, no
    global refinement of the grid.  ``expected_peaks`` is ignored (old interface).
    """
    from meson_spectra import refine_peaks
    ws = np.linspace(wi, wf, wcount)
    print(f'Frequency grid: {wcount} points, spacing {(wf - wi)/max(wcount - 1, 1):.4g} MeV')
    BAs = BA_array(ws, Q, zh, mug)

    extra_w, extra_BA = [], []

    def normalized(w):
        ba = BA(w, Q, zh, mug)
        extra_w.append(w)
        extra_BA.append(ba)
        return abs(ba.imag)/w**2

    peaks = refine_peaks(ws, np.abs(np.imag(BAs))/ws**2, normalized, rel_prominence, tol=wresolution)
    peakws = np.array([p['omega'] for p in peaks])
    peakBAs = np.array([BA(w, Q, zh, mug) for w in peakws], dtype=complex)
    print(f"Peaks found: {len(peakws)} at {[round(w, 2) for w in peakws]} MeV")
    for p in peaks:
        print(f"  omega = {p['omega']:.3f} MeV, (omega/mu_g)^2 = {(p['omega']/mug)**2:.3f}, FWHM = {p['fwhm']:.3f} MeV")

    ws = np.concatenate([ws, extra_w, peakws])
    BAs = np.concatenate([BAs, extra_BA, peakBAs])
    ws, unique = np.unique(ws, return_index=True)
    return (ws, BAs[unique], peakws, peakBAs)

def plot_results(ws, BAs, peakws, peakBAs, mug, T, mu, plot_file):
    # Prefix 'normalized_' to plot filename if normalization is requested
    if normalize_spectrum:
        plot_file = 'normalized_' + os.path.basename(plot_file)
    
    # Ensure inputs are numpy arrays for indexing
    ws = np.asarray(ws)
    BAs = np.asarray(BAs)
    
    # Sort ws and BAs by ws in ascending order
    sort_indices = np.argsort(ws)
    ws_sorted = ws[sort_indices]
    BAs_sorted = BAs[sort_indices]
    
    plt.figure(figsize=(10, 6))
    # Compute x and y values, normalize if requested
    x_vals = (ws_sorted/mug)**2
    spec = np.abs(np.imag(BAsorted:=BAs_sorted))
    if normalize_spectrum:
        y_vals = spec / x_vals
    else:
        y_vals = spec
    plt.plot(x_vals, y_vals)
    max_val = np.max(ws_sorted/mug)**2
    y_limit = min(3000, max_val * 1.1)  # Add 10% padding    
    plt.ylabel("$| \\mathrm{imag}(B/A)|$")
    plt.title(f"T = {T} MeV, μ = {mu} MeV")
    
    # Also sort the peak data
    peak_sort_indices = np.argsort(peakws)
    peakws_sorted = peakws[peak_sort_indices]
    peakBAs_sorted = peakBAs[peak_sort_indices]
    
    # Clean up the peaks data
    peakws_sorted = np.unique(peakws_sorted)
    
    # Scatter peaks
    peak_x = (peakws_sorted/mug)**2
    peak_spec = np.abs(np.imag(peakBAs_sorted))
    if normalize_spectrum:
        peak_y = peak_spec / peak_x
    else:
        peak_y = peak_spec
    plt.scatter(peak_x, peak_y, color='orange')
    
    print('The peaks are located at omega values of:')
    print(peakws_sorted)
    print('The peaks are located at dimensionless values of:')
    print((peakws_sorted/mug)**2)
    print('The spectral function values at the peaks are:')
    print(np.abs(np.imag(peakBAs_sorted)))
    
    plt.tight_layout()
    
    # Save the figure to the specified plot file, placing it under mu_g_440/vector_plots
    # Ensure plot output directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, 'mu_g_440', 'vector_plots')
    os.makedirs(plot_dir, exist_ok=True)
    # If plot_file is a basename, prefix with plot_dir
    if not os.path.dirname(plot_file):
        plot_file = os.path.join(plot_dir, plot_file)
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved to {plot_file}")

def save_data(ws, BAs, peakws, peakBAs, mug, T, mu, Q, zh, output_file=None):
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mu_g_440', 'vector_data')
    os.makedirs(data_dir, exist_ok=True)
    ws = np.asarray(ws)
    BAs = np.asarray(BAs)
    # Sort ws and BAs
    sort_indices = np.argsort(ws)
    ws_sorted = ws[sort_indices]
    BAs_sorted = BAs[sort_indices]
    
    # Create main dataframe with ws and spectral function
    df_main = pd.DataFrame({
        'omega': ws_sorted,
        'omega_squared_dimensionless': (ws_sorted/mug)**2,
        'spectral_function': np.abs(np.imag(BAs_sorted))
    })
    
    # Also sort the peak data
    peak_sort_indices = np.argsort(peakws)
    peakws_sorted = peakws[peak_sort_indices]
    peakBAs_sorted = peakBAs[peak_sort_indices]
    peakws_sorted = np.unique(peakws_sorted)
    
    # Create peaks dataframe
    df_peaks = pd.DataFrame({
        'peak_omega': peakws_sorted,
        'peak_omega_squared_dimensionless': (peakws_sorted/mug)**2,
        'peak_spectral_function': np.abs(np.imag(peakBAs_sorted[:len(peakws_sorted)]))
    })
    
    # Create parameters dataframe
    df_params = pd.DataFrame({
        'parameter': ['T', 'mu', 'mug', 'Q', 'zh', 'kappa'],
        'value': [T, mu, mug, Q, zh, kappa]
    })
    # If an explicit output file is provided, save only the main dataframe there
    if output_file:
        df_main.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        return
    
    # Save dataframes
    df_main.to_csv(os.path.join(data_dir, f'spectral_data_T{T}_mu{mu}.csv'), index=False)
    df_peaks.to_csv(os.path.join(data_dir, f'peaks_data_T{T}_mu{mu}.csv'), index=False)
    df_params.to_csv(os.path.join(data_dir, f'params_T{T}_mu{mu}.csv'), index=False)
    
    print(f"Data saved to {data_dir} directory with prefix T{T}_mu{mu}")

def main():
    # Calculate initial variables
    mug, zh, Q, lam = calculate_variables()
    
    # Calculate the data
    ws, BAs, peakws, peakBAs = calculate_data(wi, wf, wcount, wresolution, Q, zh, mug)
    
    # Plot the results and save to file
    plot_results(ws, BAs, peakws, peakBAs, mug, T, mu, f'vector_spectra_T{T}_mu{mu}.png')
    
    # Save data to files
    save_data(ws, BAs, peakws, peakBAs, mug, T, mu, Q, zh)

# Add global normalization flag
normalize_spectrum = False

# Modify entry point to parse command-line parameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run vector spectra')
    parser.add_argument('-wi', type=float, default=700, help='Initial frequency in MeV')
    parser.add_argument('-wf', type=float, default=1800, help='Final frequency in MeV')
    parser.add_argument('-wc', type=int, default=1100, help='Number of frequency points')
    parser.add_argument('-wr', type=float, default=0.1, help='Target frequency resolution')
    parser.add_argument('-T', type=float, default=T, help='Temperature in MeV')
    parser.add_argument('-mu', type=float, default=mu, help='Chemical potential in MeV')
    parser.add_argument('--normalize', action='store_true', help='Normalize spectrum by dividing by (ω/μ_g)^2')
    args = parser.parse_args()
    # Override global parameters
    wi = args.wi
    wf = args.wf
    wcount = args.wc
    wresolution = args.wr
    T = args.T
    mu = args.mu
    # Set normalization flag
    normalize_spectrum = args.normalize
    main()
