#!/usr/bin/env python3
"""
Example usage of the updated chiral_solve_complete.py with custom v3 and v4 parameters.

This file demonstrates how to use the updated functions with custom v3 and v4 values
while maintaining backward compatibility.
"""

# Example usage patterns:

# ===== BACKWARD COMPATIBLE USAGE =====
# All existing code will continue to work without changes:

"""
# Original calls (still work exactly the same):
result = calculate_sigma_values(mq_input, mq_tolerance, T, mu, lambda1, ui, uf)
mq, chi, chip, u = chiral_solve_IR(d0, lambda1, T, mu, ui, uf)
sigma_values, d0_max, d0_min, d0_list = sigma_of_T(mq_input, mq_tolerance, T, mu, lambda1, d0_lower, d0_upper, ui, uf)
"""

# ===== NEW USAGE WITH CUSTOM PARAMETERS =====
# Now you can also specify custom v3 and v4 values:

"""
# Custom v3 and v4 values
custom_v3 = -2.0
custom_v4 = 5.5

# Use custom values in the main calculation function:
result = calculate_sigma_values(
    mq_input, mq_tolerance, T, mu, lambda1, ui, uf,
    d0_lower=0, d0_upper=10,
    v3=custom_v3, v4=custom_v4
)

# Use custom values in individual functions:
mq, chi, chip, u = chiral_solve_IR(
    d0, lambda1, T, mu, ui, uf,
    v3=custom_v3, v4=custom_v4
)

sigma_values, d0_max, d0_min, d0_list = sigma_of_T(
    mq_input, mq_tolerance, T, mu, lambda1, d0_lower, d0_upper, ui, uf,
    v3=custom_v3, v4=custom_v4
)

# Parallel processing also supports custom values:
mq_array, chi_array, chip_array = chiral_solve_IR_parallel(
    d0_array, lambda1, T, mu, ui, uf,
    v3=custom_v3, v4=custom_v4, n_jobs=-1
)
"""

# ===== DEFAULT VALUES =====
# The default values are defined at the top of chiral_solve_complete.py:
"""
v4 = 4.2
v3 = -22.6/(6*np.sqrt(2))  # ≈ -2.668
"""

print("Updated chiral_solve_complete.py now supports custom v3 and v4 parameters!")
print("\nKey features:")
print("1. All functions accept v3 and v4 as optional parameters")
print("2. Default values are the original constants from the paper")
print("3. Full backward compatibility - existing code continues to work")
print("4. New functionality - can now vary v3 and v4 for parameter studies")
print("5. All parallel processing functions support the new parameters")
print("\nDefault values:")
print(f"  v4 = 4.2")
print(f"  v3 = -22.6/(6*sqrt(2)) ≈ -2.668")
