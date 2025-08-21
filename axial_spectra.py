#!/usr/bin/env python3
# axial_spectra.py
# Compute axial vector meson spectral functions in holographic QCD

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
import glob
import datetime
from chiral_solve_complete import calculate_sigma_values

# Global parameters (defaults that can be overridden)
# mu = 0  # MeV, Chemical Potential
# T = 15  # MeV, Temperature
# mq = 9  # Quark mass
# mq_tolerance = 0.01
kappa = 1  # Constant

# Integration parameters
ui = 1e-2
uf = 1-(1e-4)
ucount = 1000

# # Frequency range
# wi = 700
# wf = 2400
# wcount = 170
# wresolution = 0.01

# Global variable for chi interpolation
chi_interp = None

# Default chiral field for when interpolation not available
def default_chi(u, mq_value=None):
    #If chi can't be solved for, use zero function. This just keeps the code from crashing, but results will be meaningless.
    return 0

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
    return  440
    
#Related to black hole charge. Must return a value from 0 < Q^2 < 2
#Manually checked for mu = 0
def Q_func(zh, mu_value=None):
    global q #Prevents function from interpreting q as a local variable
    kappa = 1
    
    # Use provided mu value if given, otherwise use global mu
    mu_val = mu_value if mu_value is not None else mu
    
    # Initialize q variable to ensure it exists before use
    if 'q' not in globals():
        global q
        q = 0
    
    if mu_val == 0:
        q = 0
    else:
        if q == 0:
            q = (mu_val**3 + kappa * np.pi * T * mu_val * (kappa * np.pi * T + np.sqrt((kappa * np.pi * T)**2 + 2 * mu_val**2))) / (2 * kappa**3)
    return mu_val * zh / kappa

#Black Hole Horizon. calculated from T, mu, and k
#Manually checked for mu = 0
def z_h(T_value=None, mu_value=None):
    # Use provided values if given, otherwise use globals
    T_val = T_value if T_value is not None else T
    mu_val = mu_value if mu_value is not None else mu
    kappa = 1
    
    if mu_val == 0:
        return 1 / (np.pi * T_val)
    
    return ((-T_val * np.pi * (kappa**2)) + ((kappa**2) * np.sqrt((T_val**2 * np.pi**2) + (2 * mu_val**2 / kappa**2)))) / mu_val**2   

#Values of lambda 1-4
def lam_function(lambda1_value=7.438):
    return (lambda1_value, -22.6/(6*(2**(1/2))), 4.2)

def calculate_variables(lambda1_value=7.438, T_value=None, mu_value=None):
    mug = mu_g()
    zh = z_h(T_value, mu_value)
    Q = Q_func(zh, mu_value)
    lambda1, lambda3, lambda4 = lam_function(lambda1_value)
    g5 = 2*np.pi
    zeta = np.sqrt(3)/(2*np.pi)
    return (mug, zh, Q, lambda1, lambda3, lambda4, g5, zeta)

#Coefficients
def d2():
    return 0

def d4(w, zh, mug, d_0, d_2):
    return (1/64)*(
        (32*d_2*(zh**2)*(mug**2))
        - (8*d_2*(zh**2)*(w**2))
        + (4*d_0*(zh**4)*(mug**2)*(w**2))
        - (3*d_0*(zh**4)*(w**4))
    )

def b2():
    return 1

def b4(w, g5, zh, mq, zeta, mug, b_2):
    b_4 = (b_2/8)*(
        (g5**2*mq**2*zh**2*zeta**2)
        +(4*zh**2*mug**2)
        -(zh**2*w**2)
    )
    return b_4

def b5(g5, mq, zh, zeta, b_2, d_2):
    b_5 = (2/15)*(
        b_2 * d_2 * (g5**2) * mq * zh * zeta
    )
    return b_5

def b6(w, g5, zh, zeta, mq, mug, Q, sigma, b_2, b_4, d_2):
    b_6 = (1/24)*(
        8*b_2
        + b_2*(d_2**2)*(g5**2)
        + 8*b_2*(Q**2)
        + b_4*(g5**2)*(mq**2)*(zh**2)*(zeta**2)
        + 8 * b_4*zh**2*mug**2
        + 2*b_2*(g5**2)*mq*(zh**4)*sigma
        - b_4*(zh**2)*(w**2)
    )
    return b_6

def b7(w, g5, zh, zeta, mq, mug, Q, sigma, b_2, b_4, b_5, d_2, d_4):
    b_7 = (1/(35*zeta))*(
        2*b_4*d_2*(g5**2)*mq*zh*(zeta**2)
        + 2*b_2*d_4*(g5**2)*mq*(zh**5)*(zeta**2)
        + b_5*(g5**2)*(mq**2)*(zh**2)*(zeta**3)
        + 10*b_5*(zh**2)*zeta*(mug**2)
        + 2*b_2*d_2*(g5**2)*(zh**3)*sigma
        - b_5*(zh**2)*zeta*(w**2)
    )
    return b_7

def h(w, g5, zh, zeta, mq, b_2, h_0):
    return (h_0*zh**2/(2*b_2))*((-w**2)+(g5**2*mq**2*zeta**2))

def h0():
    return 1

def h1():
    return 0

def h2():
    return 0

def h3(w, g5, zh, mug, zeta, mq, lambda3, h_0, h_1):
    return (
        (-2*h_0*g5**2*mq**3*zh**3*zeta**3*lambda3)
        +(h_1/3)*(
            (-w**2*zh**2)
            +(g5**2*mq**2*zh**2*zeta**2)
            +(2*zh**2*mug**2)
        )
    )

def h4(w, g5, mq, zh, mug, zeta, sigma, lambda3, b_2, b_4, h_h, h_0, h_1, h_2):
    return (
        (-3/4)*h_1*g5**2*mq**3*zh**3*zeta**3*lambda3
        + (h_2/8) * (
            -w**2*zh**2
            +g5**2*mq**2*zh**2*zeta**2
            +4*zh**2*mug**2
        )
        + (h_h/8) * (
            -6*b_4
            +2*b_2*zh**2*mug**2
        )
        + (h_0/8) * (
            9*g5**2*mq**4*zh**4*zeta**4*lambda3**2
            +2*g5**2*mq*zh**4*sigma
        )
    )

def h5(w, g5, zh, zeta, mq, mug, sigma, Q, lambda1, lambda3, lambda4, h_h, h_0, h_1, h_2, h_3, b_5):
    return (
        -(8*b_5*h_h/15)
        -(2/5)*h_2*g5**2*mq**3*zh**3*zeta**3*lambda3
        + (h_3/15)*(
            -w**2*zh**2
            +g5**2*mq**2*zh**2*zeta**2
            +6*zh**2*mug**2
        )
        +(h_1/15)*(
            4
            +4*Q**2
            +9*g5**2*mq**4*zh**4*zeta**4*lambda3**2
            +2*g5**2*mq*zh**4*sigma
        )
        +(h_0/15)*(
            18*g5**2*mq**5*zh**5*zeta**5*lambda3**3
            -24*g5**2*mq**5*zh**5*zeta**5*lambda3*lambda4
            -8*g5**2*mq**3*zh**5*zeta**3*lambda3*mug**2
            +2*g5**2*mq**3*zh**5*zeta**3*lambda1*lambda3*mug**2
            -2*g5**2*mq**2*zh**5*zeta*lambda3*sigma
        )
    )

def h6(w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4, b_2, b_4, b_5, b_6, h_h, h_0, h_1, h_2, h_3, h_4):
    return (
        (-1/4)*h_3*g5**2*mq**3*zh**3*zeta**3*lambda3
        +(h_4/(24*zeta**2))*(
            -w**2*zh**2*zeta**2
            +g5**2*mq**2*zh**2*zeta**4
            +8*zh**2*zeta**2*mug**2
        )
        +(h_h/(24*zeta**2))*(
            4*b_2*zeta**2
            -10*b_5*zeta**2
            -10*b_6*zeta**2
            +4*b_2*Q**2*zeta**2
            +2*b_4*zh**2*zeta**2*mug**2
        )
        +(h_2/(24*zeta**2))*(
            8*zeta**2
            +8*Q**2*zeta**2
            +9*g5**2*mq**4*zh**4*zeta**6*lambda3**2
            +2*g5**2*mq*zh**4*zeta**2*sigma
        )
        +(h_1/(24*zeta**2))*(
            18*g5**2*mq**5*zh**5*zeta**7*lambda3**3
            -24*g5**2*mq**5*zh**5*zeta**7*lambda3*lambda4
            -8*g5**2*mq**3*zh**5*zeta**5*lambda3*mug**2
            +2*g5**2*mq**3*zh**5*zeta**5*lambda1*lambda3*mug**2
            -2*g5**2*mq**2*zh**5*zeta**3*lambda3*sigma
        )
        +(h_0/(24*zeta**2))*(
            -2*w**2*zh**2*zeta**2
            -2*Q**2*w**2*zh**2*zeta**2
            +g5**2*mq**2*zh**2*zeta**4
            +g5**2*mq**2*Q**2*zh**2*zeta**4
            -54*g5**2*mq**6*zh**6*zeta**8*lambda3**4
            +72*g5**2*mq**6*zh**6*zeta**8*lambda3**2*lambda4
            +24*g5**2*mq**4*zh**6*zeta**6*lambda3**2*mug**2
            -6*g5**2*mq**4*zh**6*zeta**6*lambda1*lambda3**2*mug**2
            -12*g5**2*mq**3*zh**6*zeta**4*lambda3**2*sigma
            +g5**2*zh**6*sigma**2
        )
    )

def h7(w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4, b_7, h_h, h_0, h_1, h_2, h_3, h_4, h_5, h_6):
    return (
        -12*b_7*h_h/35
        -6*h_4*g5**2*mq**3*zh**3*zeta**3*lambda3
        + (h_5/(35*zeta**2)) * (
            -w**2*zh**2*zeta**2
            +g5**2*mq**2*zh**2*zeta**4
            +10*zh**2*zeta**2*mug**2
        )
        + (h_3/(35*zeta**2)) * (
            12*zeta**2
            +12*Q**2*zeta**2
            +9*g5**2*mq**4*zh**4*zeta**6*lambda3**2
            +2*g5**2*mq*zh**4*zeta**2*sigma
        )
        + (h_2/(35*zeta**2)) * (
            18*g5**2*mq**5*zh**5*zeta**7*lambda3**3
            -24*g5**2*mq**5*zh**5*zeta**7*lambda3*lambda4
            -8*g5**2*mq**3*zh**5*zeta**5*lambda3*mug**2
            +2*g5**2*mq**3*zh**5*zeta**5*lambda1*lambda3*mug**2
            -2*g5**2*mq**2*zh**5*zeta**3*lambda3*sigma
        )
        + (h_1/(35*zeta**2)) * (
            -6*Q**2*zeta**2
            -2*w**2*zh**2*zeta**2
            -2*Q**2*w**2*zh**2*zeta**2
            +g5**2*mq**2*zh**2*zeta**4
            +g5**2*mq**2*Q**2*zh**2*zeta**4
            -54*g5**2*mq**6*zh**6*zeta**8*lambda3**4
            +72*g5**2*mq**6*zh**6*zeta**8*lambda3**2*lambda4
            +24*g5**2*mq**4*zh**6*zeta**6*lambda3**2*mug**2
            -6*g5**2*mq**4*zh**6*zeta**6*lambda1*lambda3**2*mug**2
            -12*g5**2*mq**3*zh**6*zeta**4*lambda3**2*sigma
            +g5**2*zh**6*sigma**2
        )
        + (h_0/(35*zeta**2)) * (
            -6*g5**2*mq**3*zh**3*zeta**5*lambda3
            -6*g5**2*mq**3*Q**2*zh**3*zeta**5*lambda3
            +18*g5**2*mq**4*zh**7*zeta**5*lambda3**3*sigma
            -25*g5**2*mq**4*zh**7*lambda3*lambda4*sigma
            -8*g5**2*mq**2*zh**7*zeta**3*lambda3*mug**2*sigma
            +2*g5**2*mq**2*zh**7*zeta**3*lambda1*lambda3*mug**2*sigma
            +4*g5**2*mq*zh**7*zeta*lambda3*sigma**2
        )
    )

def coefficients(w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4):
    d_0 = 1
    d_2 = d2()
    d_4 = d4(w, zh, mug, d_0, d_2)
    
    b_2 = b2()
    b_4 = b4(w, g5, zh, mq, zeta, mug, b_2)
    b_5 = b5(g5, mq, zh, zeta, b_2, d_2)
    b_6 = b6(w, g5, zh, zeta, mq, mug, Q, sigma, b_2, b_4, d_2)
    b_7 = b7(w, g5, zh, zeta, mq, mug, Q, sigma, b_2, b_4, b_5, d_2, d_4)
    
    h_0 = h0()
    h_1 = h1()
    h_2 = h2()
    h_h = h(w, g5, zh, zeta, mq, b_2, h_0)
    h_3 = h3(w, g5, zh, mug, zeta, mq, lambda3, h_0, h_1)
    h_4 = h4(w, g5, mq, zh, mug, zeta, sigma, lambda3, b_2, b_4, h_h, h_0, h_1, h_2)
    h_5 = h5(w, g5, zh, zeta, mq, mug, sigma, Q, lambda1, lambda3, lambda4, h_h, h_0, h_1, h_2, h_3, b_5)
    h_6 = h6(w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4, b_2, b_4, b_5, b_6, h_h, h_0, h_1, h_2, h_3, h_4)
    h_7 = h7(w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4, b_7, h_h, h_0, h_1, h_2, h_3, h_4, h_5, h_6)
    
    return (b_2, b_4, b_5, b_6, b_7, h_0, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_h)

def a1(u, bs):
    b_2, b_4, b_5, b_6, b_7 = bs
    
    a_1 = ((b_2*(u**2))
           +(b_4*(u**4))
           +(b_5*(u**5))
           +(b_6*(u**6))
           +(b_7*(u**7))
           )
           
    
    a_1p = ((2*b_2*(u))
            +(4*b_4*(u**3))
            +(5*b_5*(u**4))
            +(6*b_6*(u**5))
            +(7*b_7*(u**6))
            )

    return [a_1, a_1p]

def a2(u, a, hs):
    h_0, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_h = hs
    a_2 = (h_0
        +(h_1*(u))
        +(h_2*(u**2))
        +(h_3*(u**3))
        +(h_4*(u**4))
        #+(h_6*(u**6))
        #+(h_7*(u**7))
        +(h_h*a[0]*np.log(u)))
    
    a_2p = (h_1
        +(2*h_2*(u))
        +(3*h_3*(u**2))
        +(4*h_4*(u**3))
        #+(6*h_6*(u**5))
        #+(7*h_7*(u**6))
        +(h_h*((a[1]*np.log(u))+(a[0]*(1/u))))
           )
    return [a_2, a_2p]

def a_s(u, w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4):
    b_2, b_4, b_5, b_6, b_7, h_0, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_h = coefficients(w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4)
    
    bs = (b_2, b_4, b_5, b_6, b_7)
    hs = (h_0, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_h)
    
    a_1 = a1(u, bs)
    a_2 = a2(u, a_1, hs)
    return (a_1, a_2)

#Axial Equation of Motion; Must be solved using solve_ivp
def Axial_eom(a, u, w, g5, Q, zh, mug, mq_value=None):
    fu = f(u, Q)
    phiu = phi(u, zh, mug)
    
    # Use provided mq_value if given, otherwise use global mq
    mq_val = mq_value if mq_value is not None else mq
    
    # Check if chi_interp is available, otherwise use default
    global chi_interp
    if chi_interp is None:
        chiu = default_chi(u, mq_val)
    else:
        chiu = chi_interp(u)
    
    a1, a2 = a
    
    a1p = a2
    a2p = ((a2*(
            (1/u)-(fu[1]/fu[0])+(phiu[1])
        ))
        -(
            a1*(
                (w**2)*(zh**2)/(fu[0]**2)
                -g5**2*chiu**2/(u**2*fu[0])
            )
        ))
    
    return [a1p, a2p]

#Asymptotic Solution at the Horizon
def psi_m(u, w, zh, Q):
    c = -complex(0, 1)*w*zh/(2*(2-(Q**2)))
    
    psim = (1-u)**(c)
    psim_p = -c*((1-u)**(c-1)) #psim_p = c*((1-u)**(-c))
    
    return [psim, psim_p]

#Ratio of B and A
def BA(ui, uf, w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4):
    a_1, a_2 = a_s(ui, w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4)
    
    psi1s = solve_ivp(
        lambda u, y: Axial_eom(y, u, w, g5, Q, zh, mug, mq),
        [ui, uf],
        a_1,
        method='LSODA',
        rtol=1e-8,
        atol=1e-10
    )
    
    psi2s = solve_ivp(
        lambda u, y: Axial_eom(y, u, w, g5, Q, zh, mug, mq),
        [ui, uf],
        a_2,
        method='LSODA',
        rtol=1e-8,
        atol=1e-10
    )
    
    psi1 = psi1s.y[0][-1]
    if(psi1 == 0 or np.isnan(psi1)): 
        print(f"psi1 {psi1}", flush=True)
        print(f"w {w}", flush=True)
    
    psi1p = psi1s.y[1][-1]
    if(psi1p == 0 or np.isnan(psi1p)): 
        print(f"psi1p {psi1p}", flush=True)
        print(f"w {w}", flush=True)
    
    psi2 = psi2s.y[0][-1]
    if(psi2 == 0 or np.isnan(psi2)): 
        print(f"psi2 {psi2}", flush=True)
        print(f"w {w}", flush=True)
    
    psi2p = psi2s.y[1][-1]
    if(psi2p == 0 or np.isnan(psi2p)): 
        print(f"psi2p {psi2p}", flush=True)
        print(f"w {w}", flush=True)
    
    psim = psi_m(uf, w, zh, Q)
    if(psim[0] == 0 or np.isnan(psim[0])): 
        print(f"psim {psim}", flush=True)
        print(f"w {w}", flush=True)
    
    numerator = ((psim[0]*psi2p)-(psim[1]*psi2))
    denominator = ((psim[0]*psi1p)-(psim[1]*psi1))

    try:
        B_A = (numerator/denominator)
    except:
        print(f"Error at w value: {w}", flush=True)
        print(f"psi-minus at w: {psim[0]}", flush=True)
        print(f"psi-minus-prime at w: {psim[1]}", flush=True)
        print(f"psi1 at w: {psi1}", flush=True)
        print(f"psi1-prime at w: {psi1p}", flush=True)
        print(f"psi2 at w: {psi2}", flush=True)
        print(f"psi2-prime at w: {psi2p}", flush=True)
        
        B_A = 0
    
    return B_A

def initialize_chiral_field(mq_value, mq_tolerance_value, T_value, mu_value, lambda1_value, ui_value, uf_value):
    global chi_interp
    
    try:
        result = calculate_sigma_values(mq_value, mq_tolerance_value, T_value, mu_value, lambda1_value, ui_value, uf_value)
        sigma_values = result["sigma_values"]
        chiral_fields = result["chiral_fields"]
        u_values = result["u_values"]
        
        # Always print all sigma solutions and their cube roots
        print(f"Sigma solutions found: {sigma_values}")
        cube_roots = [s**(1/3) if s >= 0 else -((-s)**(1/3)) for s in sigma_values]
        print(f"Cube roots of sigma: {[f'{cr:.1f}' for cr in cube_roots]}")
        
        # If multiple sigma values, use the smallest positive and print a message
        if len(sigma_values) > 1:
            positive_sigmas = [s for s in sigma_values if s > 0]
            if positive_sigmas:
                min_index = np.argmin(positive_sigmas)
                selected_sigma = positive_sigmas[min_index]
                # Find the index in the original sigma_values array
                orig_index = np.where(sigma_values == selected_sigma)[0][0]
                print(f"Using the smallest positive: {selected_sigma}")
            else:
                # Fallback: use the smallest by absolute value
                min_index = np.argmin(np.abs(sigma_values))
                selected_sigma = sigma_values[min_index]
                print(f"No positive sigma found, using the smallest by absolute value: {selected_sigma}")
                orig_index = min_index
        else:
            orig_index = 0
            selected_sigma = sigma_values[0]
            print(f"Using sigma: {selected_sigma}")
        chi_interp = interp1d(u_values, chiral_fields[orig_index], bounds_error=False, fill_value="extrapolate")
        return selected_sigma  # Return the selected sigma value
    except Exception as e:
        print(f"Warning: Error initializing chiral field, using default: {str(e)}")
        # If sigma calculation fails, set chi=0
        u_values = np.linspace(ui_value, uf_value, 1000)
        chi_values = 0  # Default to zero function
        chi_interp = interp1d(u_values, chi_values, bounds_error=False, fill_value="extrapolate")
        return mq_value  # Return mq as approximate sigma value

def clean_peaks(peakws, peakBAs, min_separation=10.0):
    """
    Clean peaks that are too close to each other, keeping only the one with largest spectral value.
    
    Args:
        peakws: Array of peak frequencies
        peakBAs: Array of spectral function values at peaks
        min_separation: Minimum separation between peaks in MeV (default: 10.0)
        
    Returns:
        Cleaned arrays of peak frequencies and spectral function values
    """
    if len(peakws) <= 1:
        return peakws, peakBAs
    
    # Sort peaks by frequency
    indices = np.argsort(peakws)
    sorted_peakws = peakws[indices]
    sorted_peakBAs = peakBAs[indices]
    
    # Identify peaks that are too close
    keep_mask = np.ones(len(sorted_peakws), dtype=bool)
    
    i = 0
    while i < len(sorted_peakws):
        # Find all peaks that are within min_separation of the current peak
        close_peak_indices = []
        j = i
        while j < len(sorted_peakws) and sorted_peakws[j] - sorted_peakws[i] <= min_separation:
            close_peak_indices.append(j)
            j += 1
        
        # If there are multiple peaks within min_separation
        if len(close_peak_indices) > 1:
            # Find the peak with the largest spectral value
            spectral_values = [np.abs(np.imag(sorted_peakBAs[k])) for k in close_peak_indices]
            max_value_idx = np.argmax(spectral_values)
            max_value_peak_idx = close_peak_indices[max_value_idx]
            
            # Mark all others for removal
            for k in close_peak_indices:
                if k != max_value_peak_idx:
                    keep_mask[k] = False
            
            i = j  # Skip to the next peak after the group
        else:
            i += 1  # Move to next peak
    
    # Apply mask to keep only selected peaks
    cleaned_peakws = sorted_peakws[keep_mask]
    cleaned_peakBAs = sorted_peakBAs[keep_mask]
    
    if len(cleaned_peakws) < len(peakws):
        print(f"Peak cleaning: removed {len(peakws) - len(cleaned_peakws)} peaks that were within {min_separation} MeV of stronger peaks")
    
    return cleaned_peakws, cleaned_peakBAs

def calculate_data(wi, wf, wcount, wresolution, ui, uf, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4, expected_peaks=None):
    ws = np.linspace(wi, wf, wcount)
    # Determine normalization for peak finding
    normalize = globals().get('normalize_spectrum', False)
    # Precompute scaling for normalization
    x_vals = (ws/mug)**2
    currentresolution = np.abs((wf-wi))/wcount
    print(f'Starting resolution {currentresolution}')

    # Ensure chi_interp is available for parallel workers
    global chi_interp
    if chi_interp is None:
        print("Warning: chi_interp not initialized, creating default linear chi field")
        u_values = np.linspace(ui, uf, 1000)
        chi_values = mq * u_values
        chi_interp = interp1d(u_values, chi_values, bounds_error=False, fill_value="extrapolate")
    
    # Make a local copy of chi_interp for parallel processing
    local_chi_interp = chi_interp
    
    # Modified BA function to use a local copy of chi_interp
    def BA_with_chi(ui, uf, w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4, chi_interp_func):
        # Store the global chi_interp
        global chi_interp
        old_chi_interp = chi_interp
        
        # Set the chi_interp to the provided function 
        chi_interp = chi_interp_func
        
        try:
            # Call the original BA function
            result = BA(ui, uf, w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4)
            return result
        finally:
            # Restore the original chi_interp
            chi_interp = old_chi_interp

    # Pass the chi_interp function to each parallel task
    BAs = Parallel(n_jobs=-1)(
        delayed(BA_with_chi)(
            ui, uf, w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4, local_chi_interp
        ) for w in ws
    )

    # Determine expected peak count
    if expected_peaks is not None:
        # Use provided expected peak count
        expectedPeaks = expected_peaks
        print(f"Using provided expected peak count: {expectedPeaks}")
    else:
        # Try to find matching vector peaks file for expected peak count
        vector_peaks_file_pattern = f'data/peaks_data_T{T:.1f}_mu{mu:.1f}*.csv'
        vector_peaks_files = glob.glob(vector_peaks_file_pattern)
        if vector_peaks_files:
            most_recent_peaks_file = max(vector_peaks_files, key=os.path.getmtime)
            print(f"Loading vector peaks data from: {most_recent_peaks_file}")
            
            # Load the peaks data
            vector_peaks_data = pd.read_csv(most_recent_peaks_file)
            vector_peakws = vector_peaks_data.iloc[:, 0].values
            expectedPeaks = len(vector_peakws)
            print(f"Expected peaks count from vector data: {expectedPeaks}")
        else:
            expectedPeaks = int((wf/mug)**2/4)  # Approximate expected peaks
            print(f"No vector peaks data found. Using default expected peaks count: {expectedPeaks}")

    while True:
        # Prepare data array for peak finding, normalized if requested
        if normalize:
            spec_for_peaks = np.abs(np.imag(BAs)) / x_vals
        else:
            spec_for_peaks = np.abs(np.imag(BAs))
        # Lower prominence threshold to detect more peaks
        # Allow peaks to be closer together for improved detection
        peaks, _ = find_peaks(spec_for_peaks, prominence=0.1, distance=10)
        
        if len(peaks) < expectedPeaks and currentresolution > wresolution: 
            midpoints = (ws[:-1] + ws[1:]) / 2.0
            currentresolution = currentresolution / 2.0
            # Progress message: print new resolution and, if not first, count and locations of peaks
            print(f"[Progress] Omega resolution updated to {currentresolution:.6f}")
            if len(peaks) > 0:
                print(f"[Progress] {len(peaks)} peaks found at omega values: {[round(ws[p], 2) for p in peaks]}")
            ws = np.insert(ws, np.arange(1, len(ws)), midpoints)
            newBAs = Parallel(n_jobs=-1)(
                delayed(BA_with_chi)(
                    ui, uf, w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4, local_chi_interp
                ) for w in midpoints
            )
            BAs = np.insert(BAs, np.arange(1, len(BAs)), newBAs)
            if len(ws) % 2 == 1:
                ws = np.append(ws, ws[-1] + currentresolution)
                BAs = np.append(BAs, BA_with_chi(ui, uf, ws[-1], g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4, local_chi_interp))
        else:
            if currentresolution > wresolution:
                for i in peaks:
                    print(f'Checking peak at w value {ws[i]:.2f}')
                    print(f'Which is {(ws[i]/mug)**2:.2f} in dimensionless units')
                    
                    wmax = ws[i] + currentresolution
                    wmin = ws[i] - currentresolution
                    count = int(np.ceil((wmax - wmin)/wresolution))
                    wset = np.linspace(wmax, wmin, count)
                    BAset = Parallel(n_jobs=-1)(
                        delayed(BA_with_chi)(
                            ui, uf, w, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4, local_chi_interp
                        ) for w in wset
                    )
                    ws = np.append(ws, wset)
                    BAs = np.append(BAs, BAset)
                    
            print(f"Peaks found: {len(peaks)}")
            
            peakws = np.zeros(len(peaks))
            peakBAs = np.zeros(len(peaks), dtype=complex)
            for i in range(len(peaks)):
                peakws[i] = ws[peaks[i]]
                peakBAs[i] = BAs[peaks[i]]
            
            # Clean the peaks to remove closely spaced duplicates
            peakws, peakBAs = clean_peaks(peakws, peakBAs, min_separation=10.0)
            print(f"Peaks after cleaning: {len(peakws)}")
            
            break
    
    # After refining, return peaks based on normalized or raw spectral data
    return (ws, BAs, peakws, peakBAs)

def plot_results(ws, BAs, peakws, peakBAs, mug, T, mu, mq, show_plot=True, lambda1=None):
    """
    Plot and save the spectral function results in a lambda1 subfolder (truncated to 1 decimal place)
    
    Args:
        ws: Frequency values
        BAs: Spectral function values
        peakws: Peak frequencies
        peakBAs: Spectral function values at peaks
        mug: Confinement scale
        T: Temperature in MeV
        mu: Chemical potential in MeV
        mq: Quark mass
        show_plot: Whether to display the plot (default: True)
        lambda1: Lambda1 value for subfolder (default: None)
    """
    plt.figure(figsize=(12, 8))
    # Determine normalization
    normalize = globals().get('normalize_spectrum', False)

    # Compute spectral values
    spec_vals = np.abs(np.imag(BAs))
    x_vals = (ws/mug)**2
    if normalize:
        y_vals = spec_vals/(x_vals)
        ylabel = r"|Im$(B/A)$|/$(\omega/\mu_g)^2$"
    else:
        y_vals = spec_vals
        ylabel = r"|Im$(B/A)$|"
    # Plot spectral function
    plt.plot(x_vals, y_vals)
    plt.xlabel(r"$(\omega/\mu_g)^2$", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f"Axial Spectral Function (T={T} MeV, μ={mu} MeV, m_q={mq})", fontsize=16)
    
    # Mark the peaks
    # Scatter peaks with same normalization
    peak_x = (peakws/mug)**2
    if normalize:
        peak_y = np.abs(np.imag(peakBAs))/(peak_x)
    else:
        peak_y = np.abs(np.imag(peakBAs))
    plt.scatter(peak_x, peak_y, color='red', s=50)
    
    # Set y-axis limits to either 0-2000 or max value (whichever is smaller)
    max_val = np.max(y_vals)
    y_limit = min(3000, max_val * 1.1)  # Add 10% padding
    plt.ylim(0, y_limit)
    
    # Print peak information
    print("Axial Peaks (ω):")
    print([round(w, 1) for w in peakws])
    print("Axial Peaks (ω/μg)²:")
    print([round((w/mug)**2, 2) for w in peakws])
    
    # Add grid and improve appearance
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Always save the plot, regardless of show_plot value
    # Create subdirectory named by lambda1, mq, and mu
    subdir = f"l1_{lambda1:.1f}_mq_{mq:.1f}_mu_{mu:.1f}" if lambda1 is not None else "default"
    plot_dir = os.path.join('mu_g_440', 'axial_plots', subdir)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the plot to a standardized filename without timestamp
    plot_filename = os.path.join(plot_dir, f'axial_spectrum_T{T:.1f}_mu{mu:.1f}_mq{mq:.1f}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    
    # Also save as PDF for publication quality
    pdf_filename = os.path.join(plot_dir, f'axial_spectrum_T{T:.1f}_mu{mu:.1f}_mq{mq:.1f}.pdf')
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    
    # Show plot only if requested
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

def save_data(ws, BAs, peakws, peakBAs, mug, T, mu, mq, lambda1):
    """Save the spectral data and peak data to CSV files in a lambda1 subfolder (truncated to 1 decimal place)"""
    # Ensure data directory exists in mu_g_440 directory (no further subfolders)
    data_dir = os.path.join('mu_g_440', 'axial_data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save spectral data
    spectral_data = pd.DataFrame({
        'omega': ws,
        'spectral_function_real': np.real(BAs),
        'spectral_function_imag': np.imag(BAs),
        'spectral_function_abs': np.abs(np.imag(BAs))
    })
    spectral_filename = os.path.join(data_dir, f"axial_spectral_data_T_{T:.1f}_mu_{mu:.1f}_mq_{mq:.1f}_lambda1_{lambda1:.1f}.csv")
    spectral_data.to_csv(spectral_filename, index=False)
    print(f"Spectral data saved to {spectral_filename}")
    
    # Save peaks data
    peaks_data = pd.DataFrame({
        'peak_omega': peakws,
        'peak_omega_squared_dimensionless': (peakws/mug)**2,
        'peak_spectral_function_real': np.real(peakBAs),
        'peak_spectral_function_imag': np.imag(peakBAs),
        'peak_spectral_function_abs': np.abs(np.imag(peakBAs))
    })
    peaks_filename = os.path.join(data_dir, f"axial_peaks_data_T_{T:.1f}_mu_{mu:.1f}_mq_{mq:.1f}_lambda1_{lambda1:.1f}.csv")
    peaks_data.to_csv(peaks_filename, index=False)
    print(f"Peaks data saved to {peaks_filename}")
    
    # Calculate zh for parameter saving
    zh = z_h(T, mu)  # Call the z_h function with explicit parameters
    Q = Q_func(zh, mu)  # Get Q value with explicit parameter
    
    # Save parameters
    params_data = pd.DataFrame({
        'temperature': [T],
        'chemical_potential': [mu],
        'quark_mass': [mq],
        'lambda1': [lambda1],
        'mu_g': [mug],
        'z_h': [zh],
        'Q': [Q]
    })
    params_filename = os.path.join(data_dir, f"axial_params_T_{T:.1f}_mu_{mu:.1f}_mq_{mq:.1f}_lambda1_{lambda1:.1f}.csv")
    params_data.to_csv(params_filename, index=False)
    print(f"Parameters saved to {params_filename}")
    
    return spectral_filename, peaks_filename, params_filename

def main(T_value=17, mu_value=0, mq_value=9, lambda1_value=7.438, 
         mq_tolerance_value=0.01, wi_value=700, wf_value=2400, 
         wcount_value=1700, wresolution_value=0.1, ui_value=1e-2, uf_value=1-1e-4, 
         expected_peaks=None, show_plot=True):
    """
    Main function to compute axial vector meson spectral functions
    
    Args:
        T_value: Temperature in MeV (default: 17)
        mu_value: Chemical potential in MeV (default: 0)
        mq_value: Quark mass (default: 9)
        lambda1_value: Lambda1 parameter (default: 7.438)
        mq_tolerance_value: Tolerance for quark mass (default: 0.01)
        wi_value: Initial frequency in MeV (default: 700)
        wf_value: Final frequency in MeV (default: 1800)
        wcount_value: Number of frequency points (default: 110)
        wresolution_value: Frequency resolution (default: 0.1)
        ui_value: Initial u coordinate (default: 1e-2)
        uf_value: Final u coordinate (default: 1-1e-4)
        expected_peaks: Expected number of peaks (default: None)
        show_plot: Whether to display the plot (default: True)
        
    Returns:
        Tuple of (ws, BAs, peakws, peakBAs, mug, sigma)
    """
    # Set global parameters
    global T, mu, mq, mq_tolerance, wi, wf, wcount, wresolution, chi_interp
    
    T = T_value
    mu = mu_value
    mq = mq_value
    mq_tolerance = mq_tolerance_value
    wi = wi_value
    wf = wf_value
    wcount = wcount_value
    wresolution = wresolution_value
    
    print(f"Computing axial vector spectral function with parameters:")
    print(f"T = {T} MeV, μ = {mu} MeV, m_q = {mq}, λ₁ = {lambda1_value}")
    if expected_peaks is not None:
        print(f"Expected peak count provided: {expected_peaks}")
    
    # Calculate model variables
    mug, zh, Q, lambda1, lambda3, lambda4, g5, zeta = calculate_variables(lambda1_value, T_value, mu_value)
    
    # Initialize chiral field and get sigma value
    sigma = initialize_chiral_field(mq, mq_tolerance, T, mu_value, lambda1, ui_value, uf_value)
    
    # Verify chi_interp was set properly
    if chi_interp is None:
        print("Warning: chi_interp is still None after initialization, creating default")
        u_values = np.linspace(ui_value, uf_value, 1000)
        chi_values = mq * u_values
        chi_interp = interp1d(u_values, chi_values, bounds_error=False, fill_value="extrapolate")
    
    # Calculate spectral function data with expected peak count
    ws, BAs, peakws, peakBAs = calculate_data(wi, wf, wcount, wresolution, ui_value, uf_value, g5, zh, mq, mug, zeta, sigma, Q, lambda1, lambda3, lambda4, expected_peaks)
    
    # Sort the data by frequency
    sort_indices = np.argsort(ws)
    ws = ws[sort_indices]
    BAs = np.array(BAs)[sort_indices]

    # Sort peak data
    peak_sort_indices = np.argsort(peakws)
    peakws = peakws[peak_sort_indices]
    peakBAs = np.array(peakBAs)[peak_sort_indices]
    
    # Save the data
    save_data(ws, BAs, peakws, peakBAs, mug, T, mu, mq, lambda1)
    # Plot and save the results (will always save, but only display if show_plot is True)
    plot_results(ws, BAs, peakws, peakBAs, mug, T, mu, mq, show_plot, lambda1=lambda1_value)
    # Return sigma alongside spectral data
    return (ws, BAs, peakws, peakBAs, mug, sigma)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate axial vector meson spectral functions')
    parser.add_argument('-T', type=float, default=17.0, help='Temperature in MeV (default: 17.0)')
    parser.add_argument('-mu', type=float, default=0.0, help='Chemical potential in MeV (default: 0.0)')
    parser.add_argument('-mq', type=float, default=9.0, help='Quark mass (default: 9.0)')
    parser.add_argument('-lambda1', type=float, default=7.438, help='Lambda1 parameter (default: 7.438)')
    parser.add_argument('-wi', type=float, default=700.0, help='Initial frequency in MeV (default: 700.0)')
    parser.add_argument('-wf', type=float, default=2400.0, help='Final frequency in MeV (default: 2400.0)')
    parser.add_argument('-wc', type=int, default=1700, help='Number of frequency points (default: 1700)')
    parser.add_argument('-wr', type=float, default=0.01, help='Frequency resolution (default: 0.01)')
    parser.add_argument('-ui', type=float, default=1e-2, help='Initial u coordinate (default: 1e-2)')
    parser.add_argument('-uf', type=float, default=1-1e-4, help='Final u coordinate (default: 1-1e-4)')
    parser.add_argument('-ep', type=int, help='Expected number of peaks (optional)')
    parser.add_argument('--no-plot', action='store_true', help='Do not display the plot')
    parser.add_argument('--sigma-out', type=str, help='Path to write sigma value as CSV')
    parser.add_argument('--normalize', action='store_true', help='Normalize spectrum by dividing by (ω/μ_g)^2')

    args = parser.parse_args()
    # Set normalization flag for plotting
    normalize_spectrum = args.normalize

    # Run main to get results and sigma
    result = main(
        T_value=args.T,
        mu_value=args.mu,
        mq_value=args.mq,
        lambda1_value=args.lambda1,
        wi_value=args.wi,
        wf_value=args.wf,
        wcount_value=args.wc,
        wresolution_value=args.wr,
        ui_value=args.ui,
        uf_value=args.uf,
        expected_peaks=args.ep,
        show_plot=not args.no_plot
    )
    # main now returns (ws, BAs, peakws, peakBAs, mug, sigma)
    *_, sigma = result
    # Write sigma to CSV if requested
    if args.sigma_out:
        import pandas as pd
        df_sigma = pd.DataFrame({'sigma': [sigma]})
        df_sigma.to_csv(args.sigma_out, index=False)
        print(f"Sigma value saved to {args.sigma_out}")