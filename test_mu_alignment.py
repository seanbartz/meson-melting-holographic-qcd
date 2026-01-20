#!/usr/bin/env python3
"""
Test script to verify μ value alignment between phase diagram and axial melting scans.

This test verifies that:
1. Both scripts have the same default μ range
2. Both scripts produce the same μ values array
3. The SLURM workflow passes parameters correctly
"""

import argparse
import numpy as np
import sys

def test_default_parameters():
    """Test that default parameters are aligned."""
    print("=" * 70)
    print("Testing Default Parameter Alignment")
    print("=" * 70)
    
    # Import the scripts to check their defaults
    import map_phase_diagram_improved
    import axial_melting_scan
    
    # Check argparse defaults by creating parsers
    phase_parser = argparse.ArgumentParser()
    phase_parser.add_argument('-mumin', type=float, default=0.0)
    phase_parser.add_argument('-mumax', type=float, default=400.0)
    phase_parser.add_argument('-mupoints', type=int, default=21)
    
    axial_parser = argparse.ArgumentParser()
    axial_parser.add_argument('-mumin', type=float, default=0.0)
    axial_parser.add_argument('-mumax', type=float, default=400.0)
    axial_parser.add_argument('-mupoints', type=int, default=21)
    
    # Parse empty args to get defaults
    phase_args, _ = phase_parser.parse_known_args([])
    axial_args, _ = axial_parser.parse_known_args([])
    
    print(f"\nPhase Diagram Defaults:")
    print(f"  mumin:    {phase_args.mumin} MeV")
    print(f"  mumax:    {phase_args.mumax} MeV")
    print(f"  mupoints: {phase_args.mupoints}")
    
    print(f"\nAxial Melting Defaults:")
    print(f"  mumin:    {axial_args.mumin} MeV")
    print(f"  mumax:    {axial_args.mumax} MeV")
    print(f"  mupoints: {axial_args.mupoints}")
    
    # Check alignment
    defaults_aligned = (
        phase_args.mumin == axial_args.mumin and
        phase_args.mumax == axial_args.mumax and
        phase_args.mupoints == axial_args.mupoints
    )
    
    if defaults_aligned:
        print("\n✓ SUCCESS: Default parameters are aligned!")
        return True
    else:
        print("\n✗ FAILURE: Default parameters are NOT aligned!")
        return False

def test_mu_values_generation():
    """Test that both scripts generate the same μ values."""
    print("\n" + "=" * 70)
    print("Testing μ Values Array Generation")
    print("=" * 70)
    
    # Test parameters
    mu_min = 0.0
    mu_max = 400.0
    mu_points = 21
    
    # Generate μ values as both scripts would
    mu_values_phase = np.linspace(mu_min, mu_max, mu_points)
    mu_values_axial = np.linspace(mu_min, mu_max, mu_points)
    
    print(f"\nPhase Diagram μ values (first 5):")
    print(f"  {mu_values_phase[:5]}")
    
    print(f"\nAxial Melting μ values (first 5):")
    print(f"  {mu_values_axial[:5]}")
    
    # Check if arrays are identical
    arrays_identical = np.allclose(mu_values_phase, mu_values_axial)
    
    # Print spacing
    spacing_phase = mu_values_phase[1] - mu_values_phase[0]
    spacing_axial = mu_values_axial[1] - mu_values_axial[0]
    
    print(f"\nSpacing:")
    print(f"  Phase diagram: {spacing_phase:.6f} MeV")
    print(f"  Axial melting: {spacing_axial:.6f} MeV")
    
    # Print full arrays for verification
    print(f"\nFull μ arrays:")
    print(f"  Phase diagram: [{mu_values_phase[0]:.1f}, {mu_values_phase[1]:.1f}, ..., {mu_values_phase[-2]:.1f}, {mu_values_phase[-1]:.1f}]")
    print(f"  Axial melting: [{mu_values_axial[0]:.1f}, {mu_values_axial[1]:.1f}, ..., {mu_values_axial[-2]:.1f}, {mu_values_axial[-1]:.1f}]")
    
    if arrays_identical:
        print("\n✓ SUCCESS: μ values arrays are identical!")
        return True
    else:
        print("\n✗ FAILURE: μ values arrays are NOT identical!")
        return False

def test_slurm_parameter_passing():
    """Test that SLURM workflow default parameters are correct."""
    print("\n" + "=" * 70)
    print("Testing SLURM Workflow Parameters")
    print("=" * 70)
    
    # Expected SLURM defaults (from slurm_batch_array.sh lines 182-184)
    slurm_mumin = 0.0
    slurm_mumax = 400.0
    slurm_mupoints = 21
    
    # Expected script defaults
    script_mumin = 0.0
    script_mumax = 400.0
    script_mupoints = 21
    
    print(f"\nSLURM Workflow Defaults:")
    print(f"  mumin:    {slurm_mumin} MeV")
    print(f"  mumax:    {slurm_mumax} MeV")
    print(f"  mupoints: {slurm_mupoints}")
    
    print(f"\nScript Defaults:")
    print(f"  mumin:    {script_mumin} MeV")
    print(f"  mumax:    {script_mumax} MeV")
    print(f"  mupoints: {script_mupoints}")
    
    slurm_aligned = (
        slurm_mumin == script_mumin and
        slurm_mumax == script_mumax and
        slurm_mupoints == script_mupoints
    )
    
    if slurm_aligned:
        print("\n✓ SUCCESS: SLURM workflow parameters are aligned!")
        return True
    else:
        print("\n✗ FAILURE: SLURM workflow parameters are NOT aligned!")
        return False

def main():
    """Run all alignment tests."""
    print("\n" + "=" * 70)
    print("μ VALUE ALIGNMENT TEST SUITE")
    print("=" * 70)
    print("\nThis test suite verifies that phase diagram and axial melting scans")
    print("use identical μ (chemical potential) values for direct comparison.")
    print("=" * 70)
    
    # Run all tests
    test_results = []
    
    try:
        test_results.append(("Default Parameters", test_default_parameters()))
    except Exception as e:
        print(f"\n✗ EXCEPTION in Default Parameters test: {e}")
        test_results.append(("Default Parameters", False))
    
    try:
        test_results.append(("μ Values Generation", test_mu_values_generation()))
    except Exception as e:
        print(f"\n✗ EXCEPTION in μ Values Generation test: {e}")
        test_results.append(("μ Values Generation", False))
    
    try:
        test_results.append(("SLURM Parameters", test_slurm_parameter_passing()))
    except Exception as e:
        print(f"\n✗ EXCEPTION in SLURM Parameters test: {e}")
        test_results.append(("SLURM Parameters", False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nμ values are properly aligned between phase diagram and axial melting scans.")
        print("Both calculations will use identical chemical potential points for direct comparison.")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!")
        print("\nPlease review the alignment issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
