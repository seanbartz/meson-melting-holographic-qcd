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

echo "=== Aggressive Resource Allocation (Up to 20 CPUs/task) ==="
echo "Parameter counts:"
echo "  mq: $MQ_COUNT"
echo "  lambda1: $LAMBDA1_COUNT"  
echo "  gamma: $GAMMA_COUNT"
echo "  lambda4: $LAMBDA4_COUNT"
echo "  Total job array size: $TOTAL_JOBS"
echo ""

# Determine aggressive resource allocation based on job count
if [ $TOTAL_JOBS -le 5 ]; then
    CPUS_PER_TASK=20
    CONCURRENT_JOBS=$TOTAL_JOBS
    MEMORY_PER_CPU="3G"
elif [ $TOTAL_JOBS -le 10 ]; then
    CPUS_PER_TASK=16
    CONCURRENT_JOBS=$TOTAL_JOBS
    MEMORY_PER_CPU="3G"
elif [ $TOTAL_JOBS -le 25 ]; then
    CPUS_PER_TASK=10
    CONCURRENT_JOBS=15
    MEMORY_PER_CPU="3G"
else
    CPUS_PER_TASK=8
    CONCURRENT_JOBS=20
    MEMORY_PER_CPU="3G"
fi

TOTAL_MEMORY_PER_TASK=$((CPUS_PER_TASK * ${MEMORY_PER_CPU%G}))

echo "Aggressive resource allocation:"
echo "  CPUs per task: $CPUS_PER_TASK"
echo "  Memory per CPU: $MEMORY_PER_CPU"
echo "  Total memory per task: ${TOTAL_MEMORY_PER_TASK}G"
echo "  Max concurrent jobs: $CONCURRENT_JOBS"
echo "  Node sharing: YES (can use partial nodes)"
echo "  Can scale up to full 20-CPU nodes when available"
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
    
    # Submit with aggressive resource allocation
    echo "Submitting: sbatch --array=1-$TOTAL_JOBS%$CONCURRENT_JOBS slurm_batch_array.sh $@"
    export AGGRESSIVE_CPUS_PER_TASK=$CPUS_PER_TASK
    export AGGRESSIVE_MEMORY_PER_CPU=$MEMORY_PER_CPU
    sbatch --array=1-$TOTAL_JOBS%$CONCURRENT_JOBS slurm_batch_array.sh "$@"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Aggressive resource job submitted successfully!"
        echo ""
        echo "Resource allocation:"
        echo "  → ${CPUS_PER_TASK} CPUs per task (up to full 20-CPU nodes)"
        echo "  → ${TOTAL_MEMORY_PER_TASK}GB memory per task"
        echo "  → Can utilize any available CPUs on any node"
        echo "  → Up to $CONCURRENT_JOBS tasks running concurrently"
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
