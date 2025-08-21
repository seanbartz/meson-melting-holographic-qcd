#!/usr/bin/env python3
"""
Example usage of map_phase_diagram.py

This script demonstrates how to use the phase diagram mapper
with specific parameter values.
"""

from map_phase_diagram import map_phase_diagram

# Example 1: Basic usage with default parameters
print("Example 1: Basic phase diagram mapping")
df1 = map_phase_diagram(
    mu_min=0.0,
    mu_max=150.0, 
    mu_points=15,
    lambda1=7.8,
    ml=24.0
)

print("\nFirst few rows of results:")
print(df1.head())

# Example 2: More detailed scan with custom temperature range
print("\n" + "="*60)
print("Example 2: Detailed scan with custom parameters")
df2 = map_phase_diagram(
    mu_min=50.0,
    mu_max=200.0,
    mu_points=25,
    lambda1=7.5,
    ml=30.0,
    tmin=100.0,
    tmax=250.0,
    numtemp=30,
    output_file="custom_phase_diagram.csv",
    plot_results=True
)

# Example 3: Quick scan without plotting
print("\n" + "="*60)
print("Example 3: Quick scan without plotting")
df3 = map_phase_diagram(
    mu_min=0.0,
    mu_max=100.0,
    mu_points=10,
    lambda1=8.0,
    ml=20.0,
    plot_results=False
)

print(f"\nSummary of results:")
print(f"Dataset 1: {len(df1)} points, {(df1['order']==1).sum()} first order, {(df1['order']==2).sum()} crossover")
print(f"Dataset 2: {len(df2)} points, {(df2['order']==1).sum()} first order, {(df2['order']==2).sum()} crossover") 
print(f"Dataset 3: {len(df3)} points, {(df3['order']==1).sum()} first order, {(df3['order']==2).sum()} crossover")
