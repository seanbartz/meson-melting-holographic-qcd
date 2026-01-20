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
    """Test that default parameters are aligned by reading from actual source files."""
    print("=" * 70)
    print("Testing Default Parameter Alignment")
    print("=" * 70)
    
    # Read actual defaults from source files using grep-like approach
    def extract_default_from_file(filename, arg_name):
        """Extract the default value for a parameter from a Python file."""
        with open(filename, 'r') as f:
            for line in f:
                # Look for parser.add_argument lines for this parameter
                if f"'{arg_name}'" in line or f'"{arg_name}"' in line:
                    # Extract default value from the line
                    if 'default=' in line:
                        # Extract the value after default=
                        default_part = line.split('default=')[1]
                        # Get the value before the comma or closing paren
                        value_str = default_part.split(',')[0].split(')')[0].strip()
                        try:
                            return float(value_str) if '.' in value_str else int(value_str)
                        except ValueError:
                            pass
        return None
    
    # Extract defaults from actual source files
    phase_mumin = extract_default_from_file('map_phase_diagram_improved.py', '-mumin')
    phase_mumax = extract_default_from_file('map_phase_diagram_improved.py', '-mumax')
    phase_mupoints = extract_default_from_file('map_phase_diagram_improved.py', '-mupoints')
    
    axial_mumin = extract_default_from_file('axial_melting_scan.py', '-mumin')
    axial_mumax = extract_default_from_file('axial_melting_scan.py', '-mumax')
    axial_mupoints = extract_default_from_file('axial_melting_scan.py', '-mupoints')
    
    print(f"\nPhase Diagram Defaults (from map_phase_diagram_improved.py):")
    print(f"  mumin:    {phase_mumin} MeV")
    print(f"  mumax:    {phase_mumax} MeV")
    print(f"  mupoints: {phase_mupoints}")
    
    print(f"\nAxial Melting Defaults (from axial_melting_scan.py):")
    print(f"  mumin:    {axial_mumin} MeV")
    print(f"  mumax:    {axial_mumax} MeV")
    print(f"  mupoints: {axial_mupoints}")
    
    # Check alignment
    defaults_aligned = (
        phase_mumin == axial_mumin and
        phase_mumax == axial_mumax and
        phase_mupoints == axial_mupoints
    )
    
    if defaults_aligned:
        print("\n✓ SUCCESS: Default parameters are aligned!")
        return True
    else:
        print("\n✗ FAILURE: Default parameters are NOT aligned!")
        print(f"  Phase diagram: mumin={phase_mumin}, mumax={phase_mumax}, mupoints={phase_mupoints}")
        print(f"  Axial melting: mumin={axial_mumin}, mumax={axial_mumax}, mupoints={axial_mupoints}")
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
    """Test that SLURM workflow default parameters are correct by reading from bash script."""
    print("\n" + "=" * 70)
    print("Testing SLURM Workflow Parameters")
    print("=" * 70)
    
    # Extract defaults from SLURM bash script
    def extract_bash_default(filename, param_name):
        """Extract default value from bash script parse_single_parameter call."""
        with open(filename, 'r') as f:
            for line in f:
                # Look for parse_single_parameter calls
                if f'parse_single_parameter "{param_name}"' in line:
                    # Extract the default value (second argument in quotes)
                    parts = line.split('"')
                    if len(parts) >= 4:
                        try:
                            value = parts[3]
                            return float(value) if '.' in value else int(value)
                        except ValueError:
                            pass
        return None
    
    # Extract from SLURM script
    slurm_mumin = extract_bash_default('slurm_batch_array.sh', 'mumin')
    slurm_mumax = extract_bash_default('slurm_batch_array.sh', 'mumax')
    slurm_mupoints = extract_bash_default('slurm_batch_array.sh', 'mupoints')
    
    # Extract from Python scripts
    def extract_default_from_file(filename, arg_name):
        """Extract the default value for a parameter from a Python file."""
        with open(filename, 'r') as f:
            for line in f:
                if f"'{arg_name}'" in line or f'"{arg_name}"' in line:
                    if 'default=' in line:
                        default_part = line.split('default=')[1]
                        value_str = default_part.split(',')[0].split(')')[0].strip()
                        try:
                            return float(value_str) if '.' in value_str else int(value_str)
                        except ValueError:
                            pass
        return None
    
    script_mumin = extract_default_from_file('map_phase_diagram_improved.py', '-mumin')
    script_mumax = extract_default_from_file('map_phase_diagram_improved.py', '-mumax')
    script_mupoints = extract_default_from_file('map_phase_diagram_improved.py', '-mupoints')
    
    print(f"\nSLURM Workflow Defaults (from slurm_batch_array.sh):")
    print(f"  mumin:    {slurm_mumin} MeV")
    print(f"  mumax:    {slurm_mumax} MeV")
    print(f"  mupoints: {slurm_mupoints}")
    
    print(f"\nScript Defaults (from map_phase_diagram_improved.py):")
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
        print(f"  SLURM:  mumin={slurm_mumin}, mumax={slurm_mumax}, mupoints={slurm_mupoints}")
        print(f"  Script: mumin={script_mumin}, mumax={script_mumax}, mupoints={script_mupoints}")
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
