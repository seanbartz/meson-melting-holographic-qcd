#!/bin/bash
#SBATCH --job-name=batch_phase_array
#SBATCH --partition=general
#SBATCH --array=1-N%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/batch_%A_%a.out
#SBATCH --error=slurm_logs/batch_%A_%a.err

# SLURM Job Array for Individual Parameter Combinations
# Usage: 
#   sbatch --array=1-N%3 slurm_batch_array.sh -mqvalues 9.0 12.0 15.0 -lambda1values 3.0 5.0 7.0 -gamma -22.4 -lambda4 4.2
#   
#   Where N = number of parameter combinations (calculated automatically)
#   The %3 limits concurrent jobs to 3 (adjust as needed)
#   
# Examples:
#   sbatch --array=1-9%3 slurm_batch_array.sh -mqvalues 9.0 12.0 15.0 -lambda1values 3.0 5.0 7.0 -gamma -22.4 -lambda4 4.2
#   sbatch --array=1-6%2 slurm_batch_array.sh -mq 9.0 -lambda1 5.0 -gammavalues -25.0 -22.4 -20.0 -lambda4values 4.0 4.2

# Create log directory
mkdir -p slurm_logs

echo "=== SLURM Job Array Task ==="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Command line args: $@"
echo "=========================="

# Parse command line arguments
parse_parameter_list() {
    local param_name=$1
    shift
    local values=()
    
    # Look for parameter specification in command line args
    local found=false
    local i=1
    while [ $i -le $# ]; do
        arg="${!i}"
        if [[ "$arg" == "-${param_name}values" ]]; then
            found=true
            ((i++))
            # Collect values until next parameter or end
            while [ $i -le $# ] && [[ "${!i}" != -* ]]; do
                values+=("${!i}")
                ((i++))
            done
            break
        elif [[ "$arg" == "-${param_name}" ]]; then
            found=true
            ((i++))
            if [ $i -le $# ]; then
                values+=("${!i}")
            fi
            break
        else
            ((i++))
        fi
    done
    
    # Return the values as a space-separated string
    printf '%s\n' "${values[@]}"
}

# Parse all parameters from command line
MQ_VALUES_RAW=$(parse_parameter_list "mq" "$@")
LAMBDA1_VALUES_RAW=$(parse_parameter_list "lambda1" "$@")  
GAMMA_VALUES_RAW=$(parse_parameter_list "gamma" "$@")
LAMBDA4_VALUES_RAW=$(parse_parameter_list "lambda4" "$@")

# Convert to arrays
if [[ -n "$MQ_VALUES_RAW" ]]; then
    read -ra MQ_VALUES <<< "$MQ_VALUES_RAW"
else
    MQ_VALUES=(9.0)  # Default
fi

if [[ -n "$LAMBDA1_VALUES_RAW" ]]; then
    read -ra LAMBDA1_VALUES <<< "$LAMBDA1_VALUES_RAW"
else
    LAMBDA1_VALUES=(5.0)  # Default  
fi

if [[ -n "$GAMMA_VALUES_RAW" ]]; then
    read -ra GAMMA_VALUES <<< "$GAMMA_VALUES_RAW"
else
    GAMMA_VALUES=(-22.4)  # Default
fi

if [[ -n "$LAMBDA4_VALUES_RAW" ]]; then
    read -ra LAMBDA4_VALUES <<< "$LAMBDA4_VALUES_RAW"
else
    LAMBDA4_VALUES=(4.2)  # Default
fi

echo "Parsed parameter arrays:"
echo "  MQ_VALUES: ${MQ_VALUES[@]}"
echo "  LAMBDA1_VALUES: ${LAMBDA1_VALUES[@]}"
echo "  GAMMA_VALUES: ${GAMMA_VALUES[@]}"
echo "  LAMBDA4_VALUES: ${LAMBDA4_VALUES[@]}"

# Calculate which parameters to use for this task
TASK_ID=$SLURM_ARRAY_TASK_ID

# Calculate indices (1-based to 0-based conversion)
let TASK_ID_0=$TASK_ID-1

N_MQ=${#MQ_VALUES[@]}
N_LAMBDA1=${#LAMBDA1_VALUES[@]}
N_GAMMA=${#GAMMA_VALUES[@]}
N_LAMBDA4=${#LAMBDA4_VALUES[@]}

# Calculate which combination this task represents
MQ_IDX=$((TASK_ID_0 % N_MQ))
LAMBDA1_IDX=$(((TASK_ID_0 / N_MQ) % N_LAMBDA1))
GAMMA_IDX=$(((TASK_ID_0 / (N_MQ * N_LAMBDA1)) % N_GAMMA))
LAMBDA4_IDX=$((TASK_ID_0 / (N_MQ * N_LAMBDA1 * N_GAMMA)))

# Get actual parameter values
MQ=${MQ_VALUES[$MQ_IDX]}
LAMBDA1=${LAMBDA1_VALUES[$LAMBDA1_IDX]}
GAMMA=${GAMMA_VALUES[$GAMMA_IDX]}
LAMBDA4=${LAMBDA4_VALUES[$LAMBDA4_IDX]}

echo "Task $TASK_ID parameters:"
echo "  mq = $MQ"
echo "  lambda1 = $LAMBDA1"
echo "  gamma = $GAMMA"
echo "  lambda4 = $LAMBDA4"

# Parse additional command line parameters for physical parameters
parse_single_parameter() {
    local param_name=$1
    local default_value=$2
    shift 2
    
    local i=1
    while [ $i -le $# ]; do
        if [[ "${!i}" == "-${param_name}" ]]; then
            ((i++))
            if [ $i -le $# ]; then
                echo "${!i}"
                return
            fi
        fi
        ((i++))
    done
    
    echo "$default_value"
}

# Parse physical parameters with defaults
MUMIN=$(parse_single_parameter "mumin" "0.0" "$@")
MUMAX=$(parse_single_parameter "mumax" "200.0" "$@")
MUPOINTS=$(parse_single_parameter "mupoints" "20" "$@")
TMIN=$(parse_single_parameter "tmin" "80.0" "$@")
TMAX=$(parse_single_parameter "tmax" "210.0" "$@")
MAXITER=$(parse_single_parameter "maxiter" "10" "$@")

echo "Physical parameters:"
echo "  mumin = $MUMIN"
echo "  mumax = $MUMAX"
echo "  mupoints = $MUPOINTS"
echo "  tmin = $TMIN"
echo "  tmax = $TMAX"
echo "  maxiter = $MAXITER"

# Load modules
module load python/3.9

# Set environment and change to submit directory
export PYTHONPATH=$PYTHONPATH:$PWD
cd $SLURM_SUBMIT_DIR

echo "Starting batch_phase_diagram_scan.py at $(date)"

# Run the batch phase diagram scan for this parameter combination
python batch_phase_diagram_scan.py \
    -mq $MQ \
    -lambda1 $LAMBDA1 \
    -gamma $GAMMA \
    -lambda4 $LAMBDA4 \
    -mumin $MUMIN \
    -mumax $MUMAX \
    -mupoints $MUPOINTS \
    -tmin $TMIN \
    -tmax $TMAX \
    -maxiter $MAXITER \
    --make-combined-plot

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Task $TASK_ID completed at $(date)"
    echo "Parameters: mq=$MQ, lambda1=$LAMBDA1, gamma=$GAMMA"
else
    echo "ERROR: Task $TASK_ID failed with exit code $EXIT_CODE at $(date)"
fi

echo "Task finished at $(date)"
exit $EXIT_CODE
