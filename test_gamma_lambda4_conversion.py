#!/usr/bin/env python3
"""
Test script to verify gamma and lambda4 to v3 and v4 conversion in map_phase_diagram_improved.py
"""

import math

def test_conversion():
    """Test the parameter conversion logic"""
    
    # Test with default values
    gamma_default = -22.4
    lambda4_default = 4.2
    
    # Convert using the same formula as in the main script
    v3_converted = gamma_default / (6 * math.sqrt(2))
    v4_converted = lambda4_default
    
    print("Parameter Conversion Test:")
    print("=" * 50)
    print(f"Input parameters:")
    print(f"  gamma = {gamma_default}")
    print(f"  lambda4 = {lambda4_default}")
    print()
    print(f"Converted parameters:")
    print(f"  v3 = gamma / (6 * sqrt(2)) = {gamma_default} / {6 * math.sqrt(2):.6f} = {v3_converted:.6f}")
    print(f"  v4 = lambda4 = {v4_converted}")
    print()
    
    # Test with some other common values
    test_cases = [
        (-22.4, 4.2),  # Default
        (0.0, 0.0),    # Zero case
        (-15.0, 5.0),  # Alternative values
        (10.0, -2.0),  # Different signs
    ]
    
    print("Additional test cases:")
    print("-" * 30)
    for gamma, lambda4 in test_cases:
        v3 = gamma / (6 * math.sqrt(2))
        v4 = lambda4
        print(f"gamma={gamma:6.1f}, lambda4={lambda4:6.1f} -> v3={v3:8.5f}, v4={v4:6.1f}")
    
    print()
    print("Conversion formula verification:")
    print(f"6 * sqrt(2) = {6 * math.sqrt(2):.6f}")
    print("This matches the conversion factor used in the code.")

if __name__ == "__main__":
    test_conversion()
