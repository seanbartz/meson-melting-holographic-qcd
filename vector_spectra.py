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

#Ratio of B and A; 
def BA(w, Q, zh, mug):
    v_1, v_2 = vs(ui, w, zh, mug, Q)
    
    psi1s = solve_ivp(
        lambda u, y: vector_eom(y, u, w, Q, zh, mug),
        [ui, uf],
        v_1,
        method='LSODA',
        rtol=1e-8,
        atol=1e-10
    )
    
    psi2s = solve_ivp(
        lambda u, y: vector_eom(y, u, w, Q, zh, mug),
        [ui, uf],
        v_2,
        method='LSODA',
        rtol=1e-8,
        atol=1e-10
    )
    
    psi1 = psi1s.y[0][-1]
    if(psi1 == 0 or psi1 == float('NaN')): 
        print("psi1 " + str(psi1), flush=True)
        print("w " + str(w), flush=True)
    
    psi1p = psi1s.y[1][-1]
    if(psi1p == 0 or psi1p == float('NaN')): 
        print("psi1 " + str(psi1p), flush=True)
        print("w " + str(w), flush=True)
    
    psi2 = psi2s.y[0][-1]
    if(psi2 == 0 or psi2 == float('NaN')): 
        print("psi2 " + str(psi2), flush=True)
        print("w " + str(w), flush=True)
    
    psi2p = psi2s.y[1][-1]
    if(psi2p == 0 or psi2p == float('NaN')): 
        print("psi2p " + str(psi2p), flush=True)
        print("w " + str(w), flush=True)
        
    psim = psi_m(uf, w, zh, Q)
    if(psim == 0 or psim == float('NaN')): 
        print("psi2p " + str(psim), flush=True)
        print("w " + str(w), flush=True)
    
    numerator = ((psim[0]*psi2p)-(psim[1]*psi2))
    denominator = ((psim[0]*psi1p)-(psim[1]*psi1))

    try:
        B_A = (numerator/denominator)
        
    except:
        print("Error at w value: " + str(w), flush=True)
        print("psi-minus at w: " + str(psim[0]), flush=True)
        print("psi-minus-prime at w: " + str(psim[1]), flush=True)
        print("psi1 at w: " + str(psi1), flush=True)
        print("psi1-prime at w: " + str(psi1p), flush=True)
        print("psi2 at w: " + str(psi2), flush=True)
        print("psi2-prime at w: " + str(psi2p), flush=True)
        
        B_A = 0
    
    return B_A

#calculate the data set
def calculate_data(wi, wf, wcount, wresolution, Q, zh, mug, expected_peaks=None):
    
    ws = np.linspace(wi, wf, wcount)

    currentresolution = np.abs((wf-wi))/wcount
    print('Starting resolution ' + str(currentresolution))

    BAs = Parallel(n_jobs=-1)(delayed(BA)(w, Q, zh, mug) for w in ws)

    # If expected_peaks is not provided, calculate it based on zero temperature estimate
    if expected_peaks is None:
        expected_peaks = np.rint((wf/mug)**2/4)
        print(f'Expected number of peaks at zero temperature: {expected_peaks}')
    else:
        print(f'Using provided expected peaks: {expected_peaks}')
    
    while True:
        
        # prepare spectrum for peak finding, normalized if requested
        x_vals = (ws/mug)**2
        if normalize_spectrum:
            spec_for_peaks = np.abs(np.imag(BAs)) / x_vals
        else:
            spec_for_peaks = np.abs(np.imag(BAs))
        peaks, _ = find_peaks(spec_for_peaks, prominence=0.01, distance=10)
        if len(peaks) < expected_peaks and currentresolution > wresolution: 
            midpoints = (ws[:-1] + ws[1:]) / 2.0
            currentresolution = currentresolution / 2.0
            print(f'Found {len(peaks)} peaks')
            print(f'Resolution updated to {currentresolution}')

            ws = np.insert(ws, np.arange(1, len(ws)), midpoints)
            
            newBAs = Parallel(n_jobs=-1)(delayed(BA)(w, Q, zh, mug) for w in midpoints)
            
            BAs = np.insert(BAs, np.arange(1, len(BAs)), newBAs)

            if len(ws) % 2 == 1:
                ws = np.append(ws, ws[-1] + currentresolution)
                BAs = np.append(BAs, BA(ws[-1], Q, zh, mug))
        else:
            if currentresolution > wresolution:
                for i in peaks:
                    print(f'checking peak at w value {ws[i]}')
                    print(f'Which is {(ws[i]/mug)**2} in dimensionless units')
                
                    wmax = ws[i] + (currentresolution)
                    wmin = ws[i] - (currentresolution)
                    count = int(np.ceil((wmax - wmin)/wresolution))
                    wset = np.linspace(wmax, wmin, count)
                    BAset = Parallel(n_jobs=-1)(delayed(BA)(w, Q, zh, mug) for w in wset)
                    ws = np.append(ws, wset)
                    BAs = np.append(BAs, BAset)
            print("peaks found: " + str(len(peaks)))
            
            peakws = np.zeros(len(peaks))
            peakBAs = np.zeros(len(peaks), dtype=complex)
            for i in range(len(peaks)):
                peakws[i] = ws[peaks[i]]
                peakBAs[i] = BAs[peaks[i]]
            break
    
    return (ws, BAs, peakws, peakBAs)

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
