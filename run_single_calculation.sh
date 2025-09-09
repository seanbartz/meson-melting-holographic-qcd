#!/bin/bash
# Single Parameter Run with Task Logging
# Wrapper script for running single parameter calculations with automatic task summary logging
#
# Usage examples:
#   ./run_single_calculation.sh --mq 9.0 --lambda1 7.438 --gamma 15.2 --lambda4 12.5
#   ./run_single_calculation.sh --mq 15.0 --lambda1 7.0 --task-name "test_run_001"

# Parse command line arguments
TASK_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-name)
            TASK_NAME="$2"
            shift 2
            ;;
        *)
            # Pass all other arguments to the main calculation script
            CALC_ARGS="$CALC_ARGS $1"
            shift
            ;;
    esac
done

# Generate task name if not provided
if [[ -z "$TASK_NAME" ]]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    TASK_NAME="single_calc_${TIMESTAMP}"
fi

echo "=== Single Parameter Calculation ==="
echo "Task name: $TASK_NAME"
echo "Calculation args: $CALC_ARGS"
echo "Start time: $(date)"
echo "================================="

# Extract parameters for logging
extract_param_value() {
    local param_name=$1
    echo "$CALC_ARGS" | grep -o -- "--${param_name} [^ ]*" | cut -d' ' -f2
}

MQ=$(extract_param_value "mq")
LAMBDA1=$(extract_param_value "lambda1")
GAMMA=$(extract_param_value "gamma")
LAMBDA4=$(extract_param_value "lambda4")
MUMIN=$(extract_param_value "mumin")
MUMAX=$(extract_param_value "mumax")
MUPOINTS=$(extract_param_value "mupoints")
TMIN=$(extract_param_value "tmin")
TMAX=$(extract_param_value "tmax")

echo "Extracted parameters:"
echo "  mq: ${MQ:-'default'}"
echo "  lambda1: ${LAMBDA1:-'default'}"
echo "  gamma: ${GAMMA:-'default'}"
echo "  lambda4: ${LAMBDA4:-'default'}"
echo "  Temperature range: ${TMIN:-'default'} - ${TMAX:-'default'}"
echo "  Chemical potential range: ${MUMIN:-'default'} - ${MUMAX:-'default'} (${MUPOINTS:-'default'} points)"

# Run the main calculation
echo ""
echo "Starting calculation..."
python3 map_phase_diagram_improved.py $CALC_ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Calculation completed at $(date)"
    
    # Automatically log task summary
    echo "Logging task summary results..."
    
    # Build task logging command
    TASK_LOG_CMD="python3 log_task_results.py $TASK_NAME"
    
    # Add parameters if they were provided
    [[ -n "$MQ" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --mq $MQ"
    [[ -n "$LAMBDA1" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --lambda1 $LAMBDA1" 
    [[ -n "$GAMMA" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --gamma $GAMMA"
    [[ -n "$LAMBDA4" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --lambda4 $LAMBDA4"
    [[ -n "$TMIN" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --T-min $TMIN"
    [[ -n "$TMAX" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --T-max $TMAX"
    [[ -n "$MUMIN" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --mu-min $MUMIN"
    [[ -n "$MUMAX" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --mu-max $MUMAX"
    [[ -n "$MUPOINTS" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --num-mu-values $MUPOINTS"
    
    # Add notes and auto-detection
    TASK_LOG_CMD="$TASK_LOG_CMD --notes 'Single parameter calculation - $CALC_ARGS'"
    TASK_LOG_CMD="$TASK_LOG_CMD --auto-detect"
    
    # Run task logging (don't fail if logging fails)
    eval $TASK_LOG_CMD || echo "Warning: Task summary logging failed, but calculation completed successfully"
    
    echo ""
    echo "Task summary logged to: summary_data/task_summary.csv"
    echo "Use 'python3 extract_physics_results.py --task-id $TASK_NAME --manual' to add detailed physics results"
    
else
    echo ""
    echo "ERROR: Calculation failed with exit code $EXIT_CODE at $(date)"
    
    # Log failed calculation
    if [[ -f "log_task_results.py" ]]; then
        echo "Logging failed calculation..."
        TASK_LOG_CMD="python3 log_task_results.py ${TASK_NAME}_FAILED"
        [[ -n "$MQ" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --mq $MQ"
        [[ -n "$LAMBDA1" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --lambda1 $LAMBDA1"
        [[ -n "$GAMMA" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --gamma $GAMMA"
        [[ -n "$LAMBDA4" ]] && TASK_LOG_CMD="$TASK_LOG_CMD --lambda4 $LAMBDA4"
        TASK_LOG_CMD="$TASK_LOG_CMD --convergence-issues 1 --notes 'FAILED - exit code $EXIT_CODE - $CALC_ARGS'"
        
        eval $TASK_LOG_CMD || echo "Task logging also failed"
    fi
fi

echo "Calculation finished at $(date)"
exit $EXIT_CODE
