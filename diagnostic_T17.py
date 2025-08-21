#!/usr/bin/env python3
"""
Diagnostic script to test why run_axial_temperature_scan.py isn't finding peaks at T=17 MeV
when axial_spectra.py run directly does find them.
"""

import os
import sys
import numpy as np
from run_axial_temperature_scan import run_axial_temperature_scan
import axial_spectra

def test_direct_call():
    """Test direct call to axial_spectra.main"""
    print("\n===== DIRECT CALL TO axial_spectra.main =====")
    T = 17.0
    mu = 0.0
    mq = 9.0
    lambda1 = 7.438
    wi = 700
    wf = 2400
    wcount = 1700
    
    print(f"Running axial_spectra.main with T={T}, mu={mu}, mq={mq}, lambda1={lambda1}")
    print(f"Frequency range: {wi} to {wf} with {wcount} points")
    
    # Run the calculation directly
    ws, BAs, peakws, peakBAs, mug = axial_spectra.main(
        T_value=T,
        mu_value=mu,
        mq_value=mq,
        lambda1_value=lambda1,
        wi_value=wi,
        wf_value=wf,
        wcount_value=wcount,
        show_plot=False
    )
    
    # Print peak information
    print(f"Direct call found {len(peakws)} peaks:")
    print(f"Peak ω values: {[round(w, 1) for w in peakws]}")
    print(f"Peak (ω/μg)² values: {[round((w/mug)**2, 2) for w in peakws]}")
    
    return peakws, mug

def test_scan_call():
    """Test call through run_axial_temperature_scan"""
    print("\n===== CALL THROUGH run_axial_temperature_scan =====")
    t_min = 17.0
    t_max = 17.0  # Only one temperature
    t_step = 1.0  # Doesn't matter since min=max
    mu_value = 0.0
    mq_value = 9.0
    lambda1_value = 7.438
    wi_value = 700
    wf_value = 2400
    wcount_value = 1700
    
    print(f"Running temperature scan with T={t_min}, mu={mu_value}, mq={mq_value}, lambda1={lambda1_value}")
    print(f"Frequency range: {wi_value} to {wf_value} with {wcount_value} points")
    
    # Run the temperature scan for just one temperature
    scan_dir, summary_file = run_axial_temperature_scan(
        t_min, 
        t_max, 
        t_step, 
        mu_value,
        mq_value,
        lambda1_value,
        wi_value,
        wf_value,
        wcount_value
    )
    
    # Print results
    print(f"Temperature scan completed. Results in {scan_dir}")
    print(f"Summary file: {summary_file}")
    
    return scan_dir, summary_file

def main():
    """Main diagnostic function"""
    print("STARTING DIAGNOSTIC FOR T=17 MeV")
    print("================================")
    
    # First run the direct call
    direct_peakws, mug = test_direct_call()
    
    # Then run through the scan function
    scan_dir, summary_file = test_scan_call()
    
    # Compare results
    print("\n===== COMPARISON =====")
    print(f"Direct call found {len(direct_peakws)} peaks")
    
    # Try to read the summary file from the scan
    try:
        import pandas as pd
        if os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
            scan_peaks = summary_df[summary_df['temperature'] == 17.0]['peak_omega'].values
            print(f"Scan call found {len(scan_peaks)} peaks")
            if len(scan_peaks) > 0:
                print(f"Scan peak ω values: {[round(w, 1) for w in scan_peaks]}")
                print(f"Scan peak (ω/μg)² values: {[round((w/mug)**2, 2) for w in scan_peaks]}")
            else:
                print("Scan found NO peaks!")
        else:
            print(f"No summary file found at {summary_file}")
    except Exception as e:
        print(f"Error reading summary file: {e}")
    
    print("\nDIAGNOSIS COMPLETE")

if __name__ == "__main__":
    main()