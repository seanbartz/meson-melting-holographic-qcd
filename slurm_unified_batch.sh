#!/bin/bash
#SBATCH --job-name=unified_phase_scan
#SBATCH --partition=all
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
# Note: Python3 is available at /usr/bin/python3 - no module load needed
# module load python/3.9  # Not available on obsidian
# module load numpy scipy matplotlib pandas

# Set up environment
export PYTHONPATH=$PYTHONPATH:$PWD
cd $SLURM_SUBMIT_DIR

# Create pre-job snapshot of shared sigma file if it exists
if [[ -n "$SHARED_SIGMA_DIR" && -d "$SHARED_SIGMA_DIR" && -f "$SHARED_SIGMA_DIR/sigma_calculations.csv" ]]; then
    SNAPSHOT_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    SNAPSHOT_FILE="$SHARED_SIGMA_DIR/sigma_snapshot_before_unified_${SNAPSHOT_TIMESTAMP}.csv"
    cp "$SHARED_SIGMA_DIR/sigma_calculations.csv" "$SNAPSHOT_FILE"
    echo "Created pre-job snapshot: $SNAPSHOT_FILE"
    
    # Log the snapshot creation
    SNAPSHOT_LINES=$(wc -l < "$SNAPSHOT_FILE")
    echo "Snapshot contains $SNAPSHOT_LINES lines of existing sigma data"
fi

echo "Starting unified batch phase diagram scan at $(date)"

# Set up output directories
# Local output (in user's directory)
LOCAL_DATA_DIR="phase_data"
LOCAL_PLOT_DIR="phase_plots"
LOCAL_SIGMA_DIR="sigma_data"

# Shared project output (optional - set PROJECT_DIR to enable)
PROJECT_DIR="${PROJECT_DIR:-/net/project/QUENCH}"
if [[ -n "$PROJECT_DIR" && -d "$PROJECT_DIR" ]]; then
    SHARED_DATA_DIR="$PROJECT_DIR/phase_data"
    SHARED_PLOT_DIR="$PROJECT_DIR/phase_plots"
    SHARED_SIGMA_DIR="$PROJECT_DIR/sigma_data"
    
    # Create shared directories if they don't exist
    mkdir -p "$SHARED_DATA_DIR" "$SHARED_PLOT_DIR" "$SHARED_SIGMA_DIR"
    
    echo "Shared output will be saved to:"
    echo "  Data: $SHARED_DATA_DIR"
    echo "  Plots: $SHARED_PLOT_DIR"
    echo "  Sigma: $SHARED_SIGMA_DIR"
else
    echo "PROJECT_DIR not set or directory doesn't exist - using local output only"
    SHARED_DATA_DIR=""
    SHARED_PLOT_DIR=""
    SHARED_SIGMA_DIR=""
fi

# Run the unified batch scanner with your desired parameters
# MODIFY THESE PARAMETERS AS NEEDED:

python3 batch_phase_diagram_unified.py \
    -mqvalues 9.0 12.0 15.0 \
    -lambda1range 3.0 7.0 -lambda1points 5 \
    -gamma -22.4 \
    -lambda4 4.2 \
    -mumin 0.0 \
    -mumax 200.0 \
    -mupoints 20 \
    -tmin 80.0 \
    -tmax 210.0 \
    -maxiter 10 \
    --output-dir "$LOCAL_DATA_DIR" \
    --plot-dir "$LOCAL_PLOT_DIR"

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

# Copy results to shared directory if configured
if [[ -n "$SHARED_DATA_DIR" && -d "$SHARED_DATA_DIR" ]]; then
    echo "Copying results to shared project directory..."
    
    # Copy data files
    if [[ -d "$LOCAL_DATA_DIR" ]]; then
        echo "Copying data files: $LOCAL_DATA_DIR/* -> $SHARED_DATA_DIR/"
        cp -v "$LOCAL_DATA_DIR"/*.csv "$SHARED_DATA_DIR/" 2>/dev/null || echo "No CSV files to copy"
    fi
    
    # Copy plot files  
    if [[ -d "$LOCAL_PLOT_DIR" ]]; then
        echo "Copying plot files: $LOCAL_PLOT_DIR/* -> $SHARED_PLOT_DIR/"
        cp -v "$LOCAL_PLOT_DIR"/*.png "$SHARED_PLOT_DIR/" 2>/dev/null || echo "No PNG files to copy"
    fi
    
    # Handle sigma data files (append to shared file to avoid overwriting)
    if [[ -d "$LOCAL_SIGMA_DIR" && -f "$LOCAL_SIGMA_DIR/sigma_calculations.csv" ]]; then
        echo "Processing sigma calculation data..."
        SHARED_SIGMA_FILE="$SHARED_SIGMA_DIR/sigma_calculations.csv"
        LOCAL_SIGMA_FILE="$LOCAL_SIGMA_DIR/sigma_calculations.csv"
        
        if [[ -f "$SHARED_SIGMA_FILE" ]]; then
            # Shared file exists - append data without header
            echo "Appending to existing shared sigma file: $SHARED_SIGMA_FILE"
            # Skip the header line (first line) when appending
            tail -n +2 "$LOCAL_SIGMA_FILE" >> "$SHARED_SIGMA_FILE"
        else
            # No shared file exists - copy the entire file including header
            echo "Creating new shared sigma file: $SHARED_SIGMA_FILE"
            cp "$LOCAL_SIGMA_FILE" "$SHARED_SIGMA_FILE"
        fi
        
        # Check if this job generated substantial sigma data before creating backup
        LOCAL_LINES=$(wc -l < "$LOCAL_SIGMA_FILE" 2>/dev/null || echo "0")
        if [[ $LOCAL_LINES -gt 50 ]]; then
            # Only create backup for substantial data generation
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            BACKUP_FILE="$SHARED_SIGMA_DIR/sigma_backup_unified_${TIMESTAMP}.csv"
            cp "$LOCAL_SIGMA_FILE" "$BACKUP_FILE"
            echo "Backup copy saved: $BACKUP_FILE (${LOCAL_LINES} lines)"
        else
            echo "Skipping backup (only ${LOCAL_LINES} lines generated)"
        fi
    fi
    
    echo "Results copied to shared directory: $PROJECT_DIR"
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Unified batch scan completed at $(date)"
    echo "Check phase_data/ for CSV files and phase_plots/ for combined plots"
    if [[ -n "$SHARED_DATA_DIR" ]]; then
        echo "Shared copies available at: $PROJECT_DIR"
    fi
else
    echo "ERROR: Unified batch scan failed with exit code $EXIT_CODE at $(date)"
fi

echo "Job finished at $(date)"
exit $EXIT_CODE
