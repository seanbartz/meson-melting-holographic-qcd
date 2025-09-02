#!/bin/bash
#SBATCH --job-name=phase_diagram_batch
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=1-20%5
#SBATCH --output=slurm_logs/phase_%A_%a.out
#SBATCH --error=slurm_logs/phase_%A_%a.err

# SLURM Job Array for Phase Diagram Calculations
# Usage: sbatch slurm_phase_diagram_array.sh
# 
# This script runs multiple parameter combinations in parallel using SLURM job arrays.
# Adjust the parameter arrays below and the --array range above to match your needs.

# Create log directory
mkdir -p slurm_logs

# Load required modules (adjust for obsidian's specific environment)
# Check available modules with: module avail
module load python/3.9
# module load numpy scipy matplotlib pandas

# Set up environment
export PYTHONPATH=$PYTHONPATH:$PWD
cd $SLURM_SUBMIT_DIR

echo "=== SLURM Job Array Task ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"  
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "========================="

# Parameter arrays - MODIFY THESE AS NEEDED
MQ_VALUES=(9.0 12.0 15.0 18.0)
LAMBDA1_VALUES=(3.0 4.0 5.0 6.0 7.0)
GAMMA_VALUES=(-22.4)  # Add more values if scanning gamma: (-25.0 -22.4 -20.0)
LAMBDA4_VALUES=(4.2)  # Add more values if scanning lambda4: (3.0 4.2 5.5)

# Calculate total combinations
TOTAL_MQ=${#MQ_VALUES[@]}
TOTAL_LAMBDA1=${#LAMBDA1_VALUES[@]}
TOTAL_GAMMA=${#GAMMA_VALUES[@]}
TOTAL_LAMBDA4=${#LAMBDA4_VALUES[@]}
TOTAL_COMBINATIONS=$((TOTAL_MQ * TOTAL_LAMBDA1 * TOTAL_GAMMA * TOTAL_LAMBDA4))

echo "Parameter space:"
echo "  mq values: ${MQ_VALUES[*]}"
echo "  lambda1 values: ${LAMBDA1_VALUES[*]}"
echo "  gamma values: ${GAMMA_VALUES[*]}"
echo "  lambda4 values: ${LAMBDA4_VALUES[*]}"
echo "  Total combinations: $TOTAL_COMBINATIONS"

# Calculate indices for this array job
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))

if [ $TASK_ID -ge $TOTAL_COMBINATIONS ]; then
    echo "ERROR: Task ID $TASK_ID exceeds total combinations $TOTAL_COMBINATIONS"
    echo "Update the --array parameter in this script to match the total combinations"
    exit 1
fi

# Calculate parameter indices using modular arithmetic
MQ_IDX=$((TASK_ID % TOTAL_MQ))
LAMBDA1_IDX=$(((TASK_ID / TOTAL_MQ) % TOTAL_LAMBDA1))
GAMMA_IDX=$(((TASK_ID / (TOTAL_MQ * TOTAL_LAMBDA1)) % TOTAL_GAMMA))
LAMBDA4_IDX=$(((TASK_ID / (TOTAL_MQ * TOTAL_LAMBDA1 * TOTAL_GAMMA)) % TOTAL_LAMBDA4))

# Get parameter values
MQ=${MQ_VALUES[$MQ_IDX]}
LAMBDA1=${LAMBDA1_VALUES[$LAMBDA1_IDX]}
GAMMA=${GAMMA_VALUES[$GAMMA_IDX]}
LAMBDA4=${LAMBDA4_VALUES[$LAMBDA4_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID processing:"
echo "  mq=$MQ (index $MQ_IDX)"
echo "  lambda1=$LAMBDA1 (index $LAMBDA1_IDX)"
echo "  gamma=$GAMMA (index $GAMMA_IDX)"
echo "  lambda4=$LAMBDA4 (index $LAMBDA4_IDX)"

# Check if output file already exists
OUTPUT_FILE="phase_data/phase_diagram_improved_mq_${MQ}_lambda1_${LAMBDA1}_gamma_${GAMMA}_lambda4_${LAMBDA4}.csv"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file already exists: $OUTPUT_FILE"
    echo "Skipping calculation for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

# Run single parameter combination
echo "Starting calculation at $(date)"
echo "Command: python map_phase_diagram_improved.py -mq $MQ -lambda1 $LAMBDA1 -gamma $GAMMA -lambda4 $LAMBDA4"

python map_phase_diagram_improved.py \
    -mq $MQ \
    -lambda1 $LAMBDA1 \
    -gamma $GAMMA \
    -lambda4 $LAMBDA4 \
    -mumin 0.0 \
    -mumax 200.0 \
    -mupoints 20 \
    -tmin 80.0 \
    -tmax 210.0 \
    -maxiterations 10 \
    --no-display

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Task $SLURM_ARRAY_TASK_ID completed at $(date)"
    echo "Output file: $OUTPUT_FILE"
else
    echo "ERROR: Task $SLURM_ARRAY_TASK_ID failed with exit code $EXIT_CODE at $(date)"
fi

echo "Task $SLURM_ARRAY_TASK_ID finished"
exit $EXIT_CODE
