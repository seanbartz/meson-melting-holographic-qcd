#!/bin/bash
# Helper script to calculate job array size and submit SLURM job array
# Usage: ./submit_array_job.sh -mqvalues 9.0 12.0 15.0 -lambda1values 3.0 5.0 7.0 -gamma -22.4 -lambda4 4.2

# Parse command line arguments to count parameter combinations
count_values() {
    local param_name=$1
    shift
    local count=0
    
    # Look for parameter specification
    local i=1
    while [ $i -le $# ]; do
        arg="${!i}"
        if [[ "$arg" == "-${param_name}values" ]]; then
            ((i++))
            # Count values until next parameter or end
            while [ $i -le $# ] && [[ "${!i}" != -* ]]; do
                ((count++))
                ((i++))
            done
            break
        elif [[ "$arg" == "-${param_name}" ]]; then
            count=1
            break
        else
            ((i++))
        fi
    done
    
    # Return 1 if no values found (default will be used)
    if [ $count -eq 0 ]; then
        count=1
    fi
    
    echo $count
}

# Count parameter values
MQ_COUNT=$(count_values "mq" "$@")
LAMBDA1_COUNT=$(count_values "lambda1" "$@")
GAMMA_COUNT=$(count_values "gamma" "$@")
LAMBDA4_COUNT=$(count_values "lambda4" "$@")

# Calculate total combinations
TOTAL_JOBS=$((MQ_COUNT * LAMBDA1_COUNT * GAMMA_COUNT * LAMBDA4_COUNT))

echo "=== Aggressive Resource Allocation (1-20 CPUs/task) ==="
echo "Parameter counts:"
echo "  mq: $MQ_COUNT"
echo "  lambda1: $LAMBDA1_COUNT"  
echo "  gamma: $GAMMA_COUNT"
echo "  lambda4: $LAMBDA4_COUNT"
echo "  Total job array size: $TOTAL_JOBS"
echo ""

# Flexible resource allocation - set reasonable CPU count based on job size
# Since SLURM doesn't support CPU ranges, use intelligent defaults
if [ $TOTAL_JOBS -le 5 ]; then
    CPUS_PER_TASK=16  # High CPU for small jobs
elif [ $TOTAL_JOBS -le 20 ]; then
    CPUS_PER_TASK=8   # Balanced for medium jobs
elif [ $TOTAL_JOBS -le 50 ]; then
    CPUS_PER_TASK=4   # Conservative for larger jobs
else
    CPUS_PER_TASK=2   # Minimal for very large jobs
fi

MEMORY_PER_CPU="3G"

# Set reasonable concurrency based on total jobs
if [ $TOTAL_JOBS -le 10 ]; then
    CONCURRENT_JOBS=$TOTAL_JOBS
elif [ $TOTAL_JOBS -le 50 ]; then
    CONCURRENT_JOBS=20
else
    CONCURRENT_JOBS=40
fi

TOTAL_MEMORY_PER_TASK=$((CPUS_PER_TASK * ${MEMORY_PER_CPU%G}))

echo "Flexible resource allocation:"
echo "  CPUs per task: $CPUS_PER_TASK (optimized for $TOTAL_JOBS total tasks)"
echo "  Memory per CPU: $MEMORY_PER_CPU"
echo "  Memory per task: ${TOTAL_MEMORY_PER_TASK}G"
echo "  Max concurrent jobs: $CONCURRENT_JOBS"
echo "  Uses --share for flexible node utilization"
echo ""

# Ask for confirmation
read -p "Submit aggressive job array with $TOTAL_JOBS tasks? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Ensure task logging script is available
    if [ ! -f "log_task_results.py" ]; then
        echo "Warning: log_task_results.py not found in current directory"
        echo "Task summary logging will be skipped"
    else
        echo "Task summary logging enabled"
        echo "Results will be logged to: /net/project/QUENCH/summary_data/task_summary.csv"
    fi
    
    # Submit with flexible resource allocation
    echo "Submitting: sbatch --array=1-$TOTAL_JOBS%$CONCURRENT_JOBS slurm_batch_array.sh $@"
    export FLEXIBLE_CPUS_PER_TASK=$CPUS_PER_TASK
    export FLEXIBLE_MEMORY_PER_CPU=$MEMORY_PER_CPU
    sbatch --array=1-$TOTAL_JOBS%$CONCURRENT_JOBS slurm_batch_array.sh "$@"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Aggressive resource job submitted successfully!"
        echo ""
        echo "Resource allocation:"
        echo "  → $CPUS_PER_TASK CPUs per task (optimized for job array size)"
        echo "  → ${TOTAL_MEMORY_PER_TASK}G memory per task"
        echo "  → Can utilize partial nodes with --share"
        echo "  → Up to $CONCURRENT_JOBS tasks running concurrently"
        echo "  → Intelligent scaling: more CPUs for fewer tasks"
        echo ""
        echo "Monitoring commands:"
        echo "  Monitor jobs: squeue -u \$USER"
        echo "  Job efficiency: seff JOBID"
        echo "  Resource usage: sstat -j JOBID.TASKID"
        echo "  Cancel all: scancel -u \$USER"
        echo ""
        echo "Results:"
        echo "  Task summaries: summary_data/task_summary.csv"
        echo "  Physics plots: phase_plots/"
    fi
else
    echo "Job submission cancelled."
fi
