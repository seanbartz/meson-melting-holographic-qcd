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

echo "Parameter counts:"
echo "  mq: $MQ_COUNT"
echo "  lambda1: $LAMBDA1_COUNT"  
echo "  gamma: $GAMMA_COUNT"
echo "  lambda4: $LAMBDA4_COUNT"
echo "  Total job array size: $TOTAL_JOBS"
echo ""

# Ask for confirmation
read -p "Submit job array with $TOTAL_JOBS tasks? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Submit with calculated array size (limit to 10 concurrent jobs - one per available node)
    echo "Submitting: sbatch --array=1-$TOTAL_JOBS%10 slurm_batch_array.sh $@"
    sbatch --array=1-$TOTAL_JOBS%10 slurm_batch_array.sh "$@"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Job submitted successfully!"
        echo "Monitor with: squeue -u \$USER"
        echo "Cancel with: scancel JOBID  (where JOBID is shown above)"
        echo "View logs in: slurm_logs/batch_JOBID_TASKID.out"
    fi
else
    echo "Job submission cancelled."
fi
