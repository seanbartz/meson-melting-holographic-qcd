#!/usr/bin/env python3
"""
Test script to investigate why T=17 MeV isn't finding peaks in the temperature scan
but works when running axial_spectra.py directly.
"""

import os
import numpy as np
import pandas as pd
import axial_spectra
import matplotlib.pyplot as plt
from run_axial_temperature_scan import clean_peaks_data

# Set parameters exactly as in the temperature scan
T = 17.0
mu = 0.0
mq = 9.0
lambda1 = 7.438
wi = 700
wf = 2400
wcount = 1700
wresolution = 0.1

print(f"Running axial_spectra calculation for T={T} MeV with:")
print(f"- Quark mass: {mq}")
print(f"- Lambda1: {lambda1}")
print(f"- Frequency range: {wi} to {wf} MeV")
print(f"- Frequency points: {wcount}")

# Run the calculation
try:
    print("\nRunning direct calculation...")
    ws, BAs, peakws, peakBAs, mug, _ = axial_spectra.main(
        T_value=T,
        mu_value=mu,
        mq_value=mq,
        lambda1_value=lambda1,
        wi_value=wi,
        wf_value=wf,
        wcount_value=wcount,
        show_plot=False
    )
    
    # Clean the peaks data to remove duplicates within 5 MeV of each other
    cleaned_peakws, cleaned_peakBAs = clean_peaks_data(peakws, peakBAs, omega_threshold=5.0)
    
    # Display peak information
    print(f"\nFound {len(peakws)} peaks before cleaning")
    print(f"Found {len(cleaned_peakws)} peaks after cleaning")
    print(f"\nPeak omega values: {[round(w, 1) for w in cleaned_peakws]}")
    print(f"Peak (ω/μg)² values: {[round((w/mug)**2, 2) for w in cleaned_peakws]}")
    
    # Save the spectral function plot
    plt.figure(figsize=(12, 8))
    plt.plot((ws/mug)**2, np.abs(np.imag(BAs)))
    plt.scatter((cleaned_peakws/mug)**2, np.abs(np.imag(cleaned_peakBAs)), color='red', s=50)
    plt.xlabel(r"$(\omega/\mu_g)^2$", fontsize=14)
    plt.ylabel(r"|Im$(B/A)$|", fontsize=14)
    plt.title(f"Axial Spectral Function (T={T} MeV, μ={mu} MeV, m_q={mq})", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"test_T{T}_spectral_function.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to test_T{T}_spectral_function.png")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nTest complete.")
