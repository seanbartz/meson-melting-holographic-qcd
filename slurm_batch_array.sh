#!/bin/bash
#SBATCH --job-name=meson_melting_aggressive
#SBATCH --array=1-N%20
#SBATCH --time=24:00:00
#SBATCH --partition=general

# AGGRESSIVE RESOURCE ALLOCATION FOR MAXIMUM PERFORMANCE
# Requests 8-20 CPUs per task depending on availability and job count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${AGGRESSIVE_CPUS_PER_TASK:-16}
#SBATCH --mem-per-cpu=${AGGRESSIVE_MEMORY_PER_CPU:-3G}
#SBATCH --share

# OUTPUT/ERROR FILES  
#SBATCH --output=slurm_logs/batch_%A_%a.out
#SBATCH --error=slurm_logs/batch_%A_%a.err

# PERFORMANCE OPTIMIZATIONS - Use all allocated CPUs aggressively
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# SLURM Job Array with Aggressive Resource Allocation
# Usage: 
#   export AGGRESSIVE_CPUS_PER_TASK=16; export AGGRESSIVE_MEMORY_PER_CPU=3G
#   sbatch --array=1-N%20 slurm_batch_array.sh -mq 9.0 12.0 15.0 -lambda1 3.0 5.0 7.0 -gamma -22.4 -lambda4 4.2
#   
# Key improvements:
#   - Requests 8-20 CPUs per task (scales with job count)
#   - Uses --share for flexible node usage (no --exclusive)
#   - Can grab any available CPUs across the cluster
#   - Up to 20 concurrent tasks for typical workloads
#   - Aggressive threading for maximum performance

# Create log directory
mkdir -p slurm_logs

echo "=== AGGRESSIVE SLURM Job Array Task ==="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK / $(nproc) on node"
echo "Memory allocated: ${SLURM_MEM_PER_CPU}MB per CPU"
echo "Total task memory: $((SLURM_CPUS_PER_TASK * SLURM_MEM_PER_CPU))MB"
echo "Threading: $OMP_NUM_THREADS threads"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Command line args: $@"
echo "========================================"

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
            while [ $i -le $# ]; do
                next_arg="${!i}"
                # Stop if we hit a parameter flag (but allow negative numbers)
                if [[ "$next_arg" =~ ^-[a-zA-Z] ]]; then
                    break
                fi
                values+=("$next_arg")
                ((i++))
            done
            break
        elif [[ "$arg" == "-${param_name}" ]]; then
            found=true
            ((i++))
            # For single parameter, only take one value and stop
            if [ $i -le $# ]; then
                next_arg="${!i}"
                # Allow negative numbers (check if it's a number, not a parameter flag)
                if [[ "$next_arg" =~ ^-?[0-9]+\.?[0-9]*$ ]] || [[ "$next_arg" != -* ]]; then
                    values+=("$next_arg")
                fi
            fi
            break
        else
            ((i++))
        fi
    done
    
    # Return the values as a space-separated string
    printf '%s ' "${values[@]}"
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
echo "  Array sizes: MQ=${#MQ_VALUES[@]}, LAMBDA1=${#LAMBDA1_VALUES[@]}, GAMMA=${#GAMMA_VALUES[@]}, LAMBDA4=${#LAMBDA4_VALUES[@]}"
echo "  Indices: MQ_IDX=$MQ_IDX, LAMBDA1_IDX=$LAMBDA1_IDX, GAMMA_IDX=$GAMMA_IDX, LAMBDA4_IDX=$LAMBDA4_IDX"
echo "  Values: mq=$MQ, lambda1=$LAMBDA1, gamma=$GAMMA, lambda4=$LAMBDA4"

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
MUMAX=$(parse_single_parameter "mumax" "400.0" "$@")
MUPOINTS=$(parse_single_parameter "mupoints" "21" "$@")
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

# Load required modules (adjust for obsidian's specific environment)  
# Note: Python3 is available at /usr/bin/python3 - no module load needed
# module load python/3.9  # Not available on obsidian
# module load numpy scipy matplotlib pandas

# Set environment and change to submit directory
export PYTHONPATH=$PYTHONPATH:$PWD
cd $SLURM_SUBMIT_DIR

# Create pre-job snapshot of shared sigma file (only for first task to avoid duplicates)
if [[ $TASK_ID -eq 1 && -n "$SHARED_SIGMA_DIR" && -d "$SHARED_SIGMA_DIR" && -f "$SHARED_SIGMA_DIR/sigma_calculations.csv" ]]; then
    SNAPSHOT_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    SNAPSHOT_FILE="$SHARED_SIGMA_DIR/sigma_snapshot_before_array_${SLURM_ARRAY_JOB_ID}_${SNAPSHOT_TIMESTAMP}.csv"
    
    # Use simple file lock for snapshot creation
    SNAPSHOT_LOCK="$SHARED_SIGMA_DIR/.snapshot_lock"
    if [[ ! -f "$SNAPSHOT_LOCK" ]]; then
        touch "$SNAPSHOT_LOCK"
        cp "$SHARED_SIGMA_DIR/sigma_calculations.csv" "$SNAPSHOT_FILE"
        echo "Created pre-array snapshot: $SNAPSHOT_FILE"
        
        # Log the snapshot creation
        SNAPSHOT_LINES=$(wc -l < "$SNAPSHOT_FILE")
        echo "Snapshot contains $SNAPSHOT_LINES lines of existing sigma data"
        rm -f "$SNAPSHOT_LOCK"
    fi
elif [[ $TASK_ID -eq 1 ]]; then
    echo "No existing shared sigma file found - no snapshot needed"
fi

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
    
    echo "Shared output configured:"
    echo "  Data: $SHARED_DATA_DIR"
    echo "  Plots: $SHARED_PLOT_DIR"
    echo "  Sigma: $SHARED_SIGMA_DIR"
else
    echo "PROJECT_DIR not set or directory doesn't exist - using local output only"
    SHARED_DATA_DIR=""
    SHARED_PLOT_DIR=""
    SHARED_SIGMA_DIR=""
fi

echo "Starting map_phase_diagram_improved.py at $(date)"

# Run map_phase_diagram_improved.py directly for this parameter combination
python3 map_phase_diagram_improved.py \
    -mq $MQ \
    -lambda1 $LAMBDA1 \
    -gamma $GAMMA \
    -lambda4 $LAMBDA4 \
    -mumin $MUMIN \
    -mumax $MUMAX \
    -mupoints $MUPOINTS \
    -tmin $TMIN \
    -tmax $TMAX \
    -maxiterations $MAXITER \
    --no-display

EXIT_CODE=$?

# Copy results to shared directory if configured
if [[ -n "$SHARED_DATA_DIR" && -d "$SHARED_DATA_DIR" ]]; then
    echo "Copying results to shared project directory..."
    
    # Copy data files (CSV files from phase_data/)
    if [[ -d "$LOCAL_DATA_DIR" ]]; then
        for csv_file in "$LOCAL_DATA_DIR"/*.csv; do
            if [[ -f "$csv_file" ]]; then
                echo "Copying: $(basename "$csv_file") -> $SHARED_DATA_DIR/"
                cp -v "$csv_file" "$SHARED_DATA_DIR/"
            fi
        done
    fi
    
    # Copy plot files (PNG files from phase_plots/)
    if [[ -d "$LOCAL_PLOT_DIR" ]]; then
        for png_file in "$LOCAL_PLOT_DIR"/*.png; do
            if [[ -f "$png_file" ]]; then
                echo "Copying: $(basename "$png_file") -> $SHARED_PLOT_DIR/"  
                cp -v "$png_file" "$SHARED_PLOT_DIR/"
            fi
        done
    fi
    
    # Handle sigma data files (append to avoid overwriting)
    if [[ -d "$LOCAL_SIGMA_DIR" && -f "$LOCAL_SIGMA_DIR/sigma_calculations.csv" ]]; then
        echo "Processing sigma calculation data for task $TASK_ID..."
        SHARED_SIGMA_FILE="$SHARED_SIGMA_DIR/sigma_calculations.csv"
        LOCAL_SIGMA_FILE="$LOCAL_SIGMA_DIR/sigma_calculations.csv"
        
        # Use file locking to prevent race conditions between parallel tasks
        LOCK_FILE="$SHARED_SIGMA_DIR/.sigma_lock"
        
        # Wait for lock (up to 30 seconds)
        lock_attempts=0
        while [[ -f "$LOCK_FILE" ]] && [[ $lock_attempts -lt 30 ]]; do
            echo "Waiting for sigma file lock... (attempt $((lock_attempts+1))/30)"
            sleep 1
            ((lock_attempts++))
        done
        
        # Create lock
        touch "$LOCK_FILE"
        
        if [[ -f "$SHARED_SIGMA_FILE" ]]; then
            # Shared file exists - append data without header
            echo "Appending to existing shared sigma file: $SHARED_SIGMA_FILE"
            tail -n +2 "$LOCAL_SIGMA_FILE" >> "$SHARED_SIGMA_FILE"
        else
            # No shared file exists - copy the entire file including header
            echo "Creating new shared sigma file: $SHARED_SIGMA_FILE"
            cp "$LOCAL_SIGMA_FILE" "$SHARED_SIGMA_FILE"
        fi
        
        # Only create backup if this task generated significant new data
        LOCAL_LINES=$(wc -l < "$LOCAL_SIGMA_FILE" 2>/dev/null || echo "0")
        if [[ $LOCAL_LINES -gt 10 ]]; then
            # Create timestamped backup only for substantial contributions
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            BACKUP_FILE="$SHARED_SIGMA_DIR/sigma_backup_task${TASK_ID}_${TIMESTAMP}.csv"
            cp "$LOCAL_SIGMA_FILE" "$BACKUP_FILE"
            echo "Task backup saved: $BACKUP_FILE (${LOCAL_LINES} lines)"
        else
            echo "Skipping backup for task $TASK_ID (only ${LOCAL_LINES} lines)"
        fi
        
        # Remove lock
        rm -f "$LOCK_FILE"
    fi
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Task $TASK_ID completed at $(date)"
    echo "Parameters: mq=$MQ, lambda1=$LAMBDA1, gamma=$GAMMA"
    if [[ -n "$SHARED_DATA_DIR" ]]; then
        echo "Results copied to shared directory: $PROJECT_DIR"
    fi
    
    # Automatically log task summary results
    echo "Logging task summary results..."
    TASK_NAME="batch_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
    
    # Prepare task summary command
    TASK_LOG_CMD="python3 log_task_results.py $TASK_NAME"
    TASK_LOG_CMD="$TASK_LOG_CMD --mq $MQ --lambda1 $LAMBDA1 --gamma $GAMMA --lambda4 $LAMBDA4"
    TASK_LOG_CMD="$TASK_LOG_CMD --T-min $TMIN --T-max $TMAX --mu-min $MUMIN --mu-max $MUMAX --num-mu-values $MUPOINTS"
    TASK_LOG_CMD="$TASK_LOG_CMD --notes 'SLURM batch array job - parameters: mq=$MQ lambda1=$LAMBDA1 gamma=$GAMMA lambda4=$LAMBDA4'"
    TASK_LOG_CMD="$TASK_LOG_CMD --auto-detect"
    
    # Run task logging (don't fail the job if logging fails)
    eval $TASK_LOG_CMD || echo "Warning: Task summary logging failed, but calculation completed successfully"
    
else
    echo "ERROR: Task $TASK_ID failed with exit code $EXIT_CODE at $(date)"
fi

echo "Task finished at $(date)"
exit $EXIT_CODE
