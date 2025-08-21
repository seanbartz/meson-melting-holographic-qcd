#!/usr/bin/env python3
# Test script to verify argument parsing for map_phase_diagram.py

import argparse

def test_args():
    """Test just the argument parsing part"""
    parser = argparse.ArgumentParser(description='Map QCD phase diagram using critical_zoom function')
    
    # Required arguments
    parser.add_argument('lambda1', type=float, help='Lambda1 parameter for mixing between dilaton and chiral field')
    parser.add_argument('ml', type=float, help='Light quark mass in MeV')
    
    # Chemical potential range
    parser.add_argument('--mu-min', type=float, default=0.0, help='Minimum chemical potential in MeV (default: 0.0)')
    parser.add_argument('--mu-max', type=float, default=200.0, help='Maximum chemical potential in MeV (default: 200.0)')
    parser.add_argument('--mu-points', type=int, default=20, help='Number of mu points to sample (default: 20)')
    
    # Temperature search parameters
    parser.add_argument('--tmin', type=float, default=80.0, help='Minimum temperature for search in MeV (default: 80.0)')
    parser.add_argument('--tmax', type=float, default=210.0, help='Maximum temperature for search in MeV (default: 210.0)')
    parser.add_argument('--numtemp', type=int, default=25, help='Number of temperature points per iteration (default: 25)')
    
    # Sigma search parameters  
    parser.add_argument('--minsigma', type=float, default=0.0, help='Minimum sigma value for search (default: 0.0)')
    parser.add_argument('--maxsigma', type=float, default=400.0, help='Maximum sigma value for search (default: 400.0)')
    parser.add_argument('--a0', type=float, default=0.0, help='Additional parameter a0 (default: 0.0)')
    
    # Output options
    parser.add_argument('-o', '--output', type=str, help='Output CSV filename (if not specified, auto-generated)')
    parser.add_argument('--no-plot', action='store_true', help='Do not create phase diagram plot')

    # Test with example arguments
    test_args = ['7.8', '24.0', '--mu-min', '0', '--mu-max', '150', '--mu-points', '10']
    args = parser.parse_args(test_args)
    
    print("Arguments parsed successfully:")
    print(f"lambda1: {args.lambda1}")
    print(f"ml: {args.ml}")
    print(f"mu range: {args.mu_min} to {args.mu_max} MeV")
    print(f"mu points: {args.mu_points}")
    print(f"Temperature range: {args.tmin} to {args.tmax} MeV")
    print(f"Output file: {args.output}")
    print(f"No plot: {args.no_plot}")

if __name__ == "__main__":
    test_args()
