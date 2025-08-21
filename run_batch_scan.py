#!/usr/bin/env python3
"""
Helper script for running batch phase diagram scans

Examples:
    # Scan gamma from -25 to -20 with 6 points
    python run_batch_scan.py gamma-scan --lambda1 5.0 --ml 9.0
    
    # Scan lambda4 from 3.0 to 5.5 with 6 points  
    python run_batch_scan.py lambda4-scan --lambda1 5.0 --ml 9.0
    
    # Custom gamma scan
    python run_batch_scan.py custom --parameter gamma --values -25.0 -22.6 -20.0 --lambda1 5.0 --ml 9.0
"""

import subprocess
import sys
import argparse

def run_gamma_scan(lambda1, ml, **kwargs):
    """Run a standard gamma scan from -25 to -20"""
    cmd = [
        sys.executable, 'batch_phase_diagram_scan.py',
        '--parameter', 'gamma',
        '--range', '-25.0', '-20.0',
        '--num-values', '6',
        '--lambda1', str(lambda1),
        '--ml', str(ml)
    ]
    
    # Add optional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    return subprocess.run(cmd)

def run_lambda4_scan(lambda1, ml, **kwargs):
    """Run a standard lambda4 scan from 3.0 to 5.5"""
    cmd = [
        sys.executable, 'batch_phase_diagram_scan.py',
        '--parameter', 'lambda4',
        '--range', '3.0', '5.5',
        '--num-values', '6',
        '--lambda1', str(lambda1),
        '--ml', str(ml)
    ]
    
    # Add optional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    return subprocess.run(cmd)

def run_custom_scan(parameter, values, lambda1, ml, **kwargs):
    """Run a custom scan with specified values"""
    cmd = [
        sys.executable, 'batch_phase_diagram_scan.py',
        '--parameter', parameter,
        '--values'
    ]
    cmd.extend([str(v) for v in values])
    cmd.extend(['--lambda1', str(lambda1), '--ml', str(ml)])
    
    # Add optional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    return subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='Helper for batch phase diagram scans')
    subparsers = parser.add_subparsers(dest='command', help='Scan type')
    
    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('-lambda1', type=float, required=True, help='Lambda1 parameter')
    common.add_argument('-mq', type=float, required=True, help='Quark mass')
    common.add_argument('-mupoints', type=int, help='Number of mu points')
    common.add_argument('--skip-existing', action='store_true', help='Skip existing files')
    
    # Gamma scan
    gamma_parser = subparsers.add_parser('gamma-scan', parents=[common], 
                                        help='Standard gamma scan (-25 to -20)')
    
    # Lambda4 scan  
    lambda4_parser = subparsers.add_parser('lambda4-scan', parents=[common],
                                          help='Standard lambda4 scan (3.0 to 5.5)')
    
    # Custom scan
    custom_parser = subparsers.add_parser('custom', parents=[common],
                                         help='Custom parameter scan')
    custom_parser.add_argument('--parameter', choices=['gamma', 'lambda4'], required=True)
    custom_parser.add_argument('--values', type=float, nargs='+', required=True)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Prepare kwargs
    kwargs = {}
    if hasattr(args, 'mupoints') and args.mupoints:
        kwargs['mu_points'] = args.mupoints
    if hasattr(args, 'skip_existing') and args.skip_existing:
        kwargs['skip_existing'] = True
    
    # Run appropriate scan
    if args.command == 'gamma-scan':
        result = run_gamma_scan(args.lambda1, args.mq, **kwargs)
    elif args.command == 'lambda4-scan':
        result = run_lambda4_scan(args.lambda1, args.mq, **kwargs)
    elif args.command == 'custom':
        result = run_custom_scan(args.parameter, args.values, args.lambda1, args.mq, **kwargs)
    else:
        print(f"Unknown command: {args.command}")
        return 1
    
    return result.returncode

if __name__ == '__main__':
    exit(main())
