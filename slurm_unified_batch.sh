#!/bin/bash
#SBATCH --job-name=unified_phase_scan
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/unified_%j.out
#SBATCH --error=slurm_logs/unified_%j.err

# SLURM Job for Unified Batch Phase Diagram Scanner
# Usage: sbatch slurm_unified_batch.sh
#
# This script runs the unified batch scanner as a single job.
# Good for smaller parameter scans or when you want everything in one job.

# Create log directory
mkdir -p slurm_logs

echo "=== SLURM Unified Batch Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "============================"

# Load required modules (adjust for obsidian)
module load python/3.9

# Set up environment
export PYTHONPATH=$PYTHONPATH:$PWD
cd $SLURM_SUBMIT_DIR

echo "Starting unified batch phase diagram scan at $(date)"

# Run the unified batch scanner with your desired parameters
# MODIFY THESE PARAMETERS AS NEEDED:

python batch_phase_diagram_unified.py \
    -mqvalues 9.0 12.0 15.0 \
    -lambda1range 3.0 7.0 -lambda1points 5 \
    -gamma -22.4 \
    -lambda4 4.2 \
    -mumin 0.0 \
    -mumax 200.0 \
    -mupoints 20 \
    -tmin 80.0 \
    -tmax 210.0 \
    -maxiter 10

# Alternative examples (comment out the one above and use one of these):

# Example 1: Scan over gamma values
# python batch_phase_diagram_unified.py \
#     -mq 9.0 \
#     -lambda1 5.0 \
#     -gammarange -25.0 -20.0 -gammapoints 6 \
#     -lambda4 4.2

# Example 2: Multi-parameter scan
# python batch_phase_diagram_unified.py \
#     -mqvalues 9.0 12.0 15.0 \
#     -lambda1values 3.0 5.0 7.0 \
#     -gammavalues -25.0 -22.4 -20.0 \
#     -lambda4 4.2

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Unified batch scan completed at $(date)"
    echo "Check phase_data/ for CSV files and phase_plots/ for combined plots"
else
    echo "ERROR: Unified batch scan failed with exit code $EXIT_CODE at $(date)"
fi

echo "Job finished at $(date)"
exit $EXIT_CODE
