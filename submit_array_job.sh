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

# Flexible resource allocation - analyze cluster availability and optimize
# Check current cluster state to determine optimal CPU allocation
echo "Analyzing cluster availability..."

# Get idle CPUs per node
IDLE_CPUS_INFO=$(sinfo -h -N -o "%N %C" | awk -F/ '{
    node=$1; idle=$3; total=$4
    if (idle > 0) print node, idle
}' | sort -k2 -nr)

echo "Nodes with idle CPUs:"
echo "$IDLE_CPUS_INFO"

# Calculate total idle CPUs and distribution
TOTAL_IDLE=$(echo "$IDLE_CPUS_INFO" | awk '{sum+=$2} END {print sum+0}')
NODE_COUNT=$(echo "$IDLE_CPUS_INFO" | wc -l)
NODE_COUNT=${NODE_COUNT:-0}

echo "Total idle CPUs across cluster: $TOTAL_IDLE"
echo "Nodes with available CPUs: $NODE_COUNT"

if [ $NODE_COUNT -eq 0 ] || [ $TOTAL_IDLE -eq 0 ]; then
    echo "WARNING: No idle CPUs found. Jobs will queue until resources become available."
    CPUS_PER_TASK=1
    IMMEDIATE_CAPACITY=0
else
    # Calculate optimal CPU allocation
    # Priority: Use full nodes (20 CPUs) when available for faster individual job completion
    
    # Check for completely empty nodes (20 idle CPUs)
    EMPTY_NODES=$(echo "$IDLE_CPUS_INFO" | awk '$2 >= 20 {count++} END {print count+0}')
    LARGE_NODES=$(echo "$IDLE_CPUS_INFO" | awk '$2 >= 10 {count++} END {print count+0}')
    
    echo "Empty nodes (≥20 CPUs): $EMPTY_NODES"
    echo "Large availability nodes (≥10 CPUs): $LARGE_NODES"
    
    if [ $EMPTY_NODES -gt 0 ]; then
        # Use full 20-CPU nodes for faster completion
        CPUS_PER_TASK=20
        echo "Strategy: Using full 20-CPU nodes for maximum performance per job"
    elif [ $LARGE_NODES -gt 0 ] && [ $TOTAL_JOBS -le 10 ]; then
        # Use large allocations for small job counts
        CPUS_PER_TASK=10
        echo "Strategy: Using 10-CPU allocation for fast completion of small job set"
    elif [ $TOTAL_JOBS -le 5 ] && [ $TOTAL_IDLE -ge 20 ]; then
        # For very small focused runs, use substantial resources
        CPUS_PER_TASK=8
        echo "Strategy: High-resource allocation for focused small run"
    elif [ $TOTAL_JOBS -le $NODE_COUNT ]; then
        # We have enough nodes for each task to get a dedicated node
        AVG_IDLE_PER_NODE=$((TOTAL_IDLE / NODE_COUNT))
        CPUS_PER_TASK=$AVG_IDLE_PER_NODE
        [ $CPUS_PER_TASK -gt 8 ] && CPUS_PER_TASK=8  # Cap at 8 CPUs for this case
        [ $CPUS_PER_TASK -lt 1 ] && CPUS_PER_TASK=1  # Minimum 1 CPU
        echo "Strategy: Dedicated node allocation ($CPUS_PER_TASK CPUs per task)"
    else
        # More tasks than nodes with idle CPUs - be conservative but still reasonable
        if [ $TOTAL_IDLE -ge $((TOTAL_JOBS * 2)) ]; then
            CPUS_PER_TASK=2  # Can afford 2 CPUs per task
            echo "Strategy: Conservative 2-CPU allocation for large job array"
        else
            CPUS_PER_TASK=1
            echo "Strategy: Minimal 1-CPU allocation for maximum parallel execution"
        fi
    fi
    
    # Estimate how many jobs can start immediately
    IMMEDIATE_CAPACITY=$((TOTAL_IDLE / CPUS_PER_TASK))
fi

# Set reasonable concurrency based on immediate capacity and total jobs
if [ $IMMEDIATE_CAPACITY -ge $TOTAL_JOBS ]; then
    CONCURRENT_JOBS=$TOTAL_JOBS  # All jobs can start immediately
elif [ $IMMEDIATE_CAPACITY -gt 0 ]; then
    CONCURRENT_JOBS=$((IMMEDIATE_CAPACITY + 5))  # Start available + a few queued
else
    # No immediate capacity, use standard limits
    if [ $TOTAL_JOBS -le 10 ]; then
        CONCURRENT_JOBS=$TOTAL_JOBS
    elif [ $TOTAL_JOBS -le 50 ]; then
        CONCURRENT_JOBS=20
    else
        CONCURRENT_JOBS=40
    fi
fi

TOTAL_MEMORY_ESTIMATE="SLURM default per CPU"

echo "Cluster-aware resource allocation:"
echo "  CPUs per task: $CPUS_PER_TASK (optimized for current cluster state)"
echo "  Memory per task: $TOTAL_MEMORY_ESTIMATE (SLURM manages automatically)"
echo "  Max concurrent jobs: $CONCURRENT_JOBS"
echo "  Immediate capacity: $IMMEDIATE_CAPACITY tasks can start now"
echo "  Strategy: Analyzed $NODE_COUNT nodes with $TOTAL_IDLE total idle CPUs"
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
    
    # Submit with flexible resource allocation - let SLURM handle memory
    PARTITION_ARG="--partition=${SUBMIT_PARTITION:-all}"
    echo "Submitting: sbatch $PARTITION_ARG --cpus-per-task=$CPUS_PER_TASK --array=1-$TOTAL_JOBS%$CONCURRENT_JOBS slurm_batch_array.sh $@"
    export FLEXIBLE_CPUS_PER_TASK=$CPUS_PER_TASK
    sbatch $PARTITION_ARG --cpus-per-task=$CPUS_PER_TASK --array=1-$TOTAL_JOBS%$CONCURRENT_JOBS slurm_batch_array.sh "$@"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Aggressive resource job submitted successfully!"
        echo ""
        echo "Resource allocation:"
        echo "  → $CPUS_PER_TASK CPUs per task (cluster-optimized)"
        echo "  → Memory allocated automatically by SLURM (typically 2-4GB per CPU)"
        echo "  → $IMMEDIATE_CAPACITY/$TOTAL_JOBS tasks can start immediately"
        echo "  → Up to $CONCURRENT_JOBS tasks running concurrently"
        echo "  → Analyzed real-time cluster availability for optimal scheduling"
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
