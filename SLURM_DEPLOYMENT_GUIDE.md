# SLURM Deployment Guide for Phase Diagram Scanning

## Overview

This directory contains SLURM batch job scripts for running phase diagram calculations on obsidian. There are two approaches:

1. **Single Job**: Use `slurm_unified_batch.sh` to run all parameter combinations in one job
2. **Job Array**: Use `slurm_batch_array.sh` to run each parameter combination as a separate array task

## Prerequisites

### 1. SSH Key Setup on Obsidian
First, ensure your SSH key on obsidian is added to your GitHub account:

```bash
# On obsidian, display your public key:
cat ~/.ssh/id_rsa.pub

# Copy this key and add it to GitHub:
# https://github.com/settings/keys
```

### 2. Clone Repository
```bash
# On obsidian:
cd ~/QGP/
git clone git@github.com:seanbartz/meson-melting-holographic-qcd.git
cd meson-melting-holographic-qcd
```

### 3. Configure Shared Output (Optional)
To save results to the shared project directory `/net/project/QUENCH`:

**Option A: Use the setup script (recommended):**
```bash
./setup_shared_output.sh
# This will check permissions, create directories, and configure environment
```

**Option B: Manual setup:**
```bash
# Set the PROJECT_DIR environment variable before running jobs
export PROJECT_DIR="/net/project/QUENCH"

# Or set it in your ~/.bashrc for persistence:
echo 'export PROJECT_DIR="/net/project/QUENCH"' >> ~/.bashrc
source ~/.bashrc

# Create directories if needed:
mkdir -p "$PROJECT_DIR/phase_data" "$PROJECT_DIR/phase_plots"
```

When `PROJECT_DIR` is set, the scripts will:
- Create `phase_data/` and `phase_plots/` directories in the shared location
- Copy all results there after local processing
- Maintain local copies in your individual directory

**For your student:** They can run the same setup in their own directory clone, and all results will be automatically shared.

## Option 1: Single Unified Job

Use `slurm_unified_batch.sh` when you want all calculations in one job:

### 1. Edit Parameters
```bash
nano slurm_unified_batch.sh
```

Modify the python command section:
```bash
python batch_phase_diagram_unified.py \
    -mqvalues 9.0 12.0 15.0 \
    -lambda1range 3.0 7.0 -lambda1points 5 \
    -gamma -22.4 \
    -lambda4 4.2
```

### 2. Submit Job
```bash
sbatch slurm_unified_batch.sh
```

### 5. Monitor Array Jobs
```bash
squeue -u $USER                           # Check all your jobs
squeue -u $USER -t RUNNING                # Only running jobs
sacct -j JOBID --format=JobID,State,ExitCode  # Check job completion status

# Check individual task logs:
tail -f slurm_logs/batch_JOBID_TASKID.out
ls slurm_logs/batch_*                     # List all log files
```

## Option 2: Job Array with Command-Line Parameters

Use `slurm_batch_array.sh` for parallel parameter combinations with command-line flexibility:

### 1. Simple Submission with Helper Script
The easiest way is to use the provided helper script:

```bash
# Example: Scan over multiple mq values with fixed other parameters
./submit_array_job.sh -mqvalues 9.0 12.0 15.0 -lambda1 5.0 -gamma -22.4 -lambda4 4.2

# Example: Scan over gamma and lambda4 combinations
./submit_array_job.sh -mq 9.0 -lambda1 5.0 -gammavalues -25.0 -22.4 -20.0 -lambda4values 4.0 4.2

# The helper script automatically calculates array size and asks for confirmation
```

### 2. Manual Submission
If you prefer manual control:

```bash
# Calculate total combinations manually:
# mq: 3 values, lambda1: 1 value, gamma: 1 value, lambda4: 1 value = 3 total jobs

sbatch --array=1-3%2 slurm_batch_array.sh -mqvalues 9.0 12.0 15.0 -lambda1 5.0 -gamma -22.4 -lambda4 4.2
```

### 3. Command-Line Parameter Options

The job array script accepts the same parameters as `batch_phase_diagram_unified.py`:

**Parameter Specification:**
- `-mq VALUE` or `-mqvalues VALUE1 VALUE2 VALUE3`
- `-lambda1 VALUE` or `-lambda1values VALUE1 VALUE2 VALUE3`
- `-gamma VALUE` or `-gammavalues VALUE1 VALUE2 VALUE3`
- `-lambda4 VALUE` or `-lambda4values VALUE1 VALUE2 VALUE3`

**Physical Parameters:**
- `-mumin VALUE` (default: 0.0)
- `-mumax VALUE` (default: 200.0)
- `-mupoints VALUE` (default: 20)
- `-tmin VALUE` (default: 80.0)
- `-tmax VALUE` (default: 210.0)
- `-maxiter VALUE` (default: 10)

### 4. Examples

```bash
# Large parameter scan
./submit_array_job.sh -mqvalues 9.0 12.0 15.0 -lambda1values 3.0 5.0 7.0 -gammavalues -25.0 -22.4 -20.0

# Focused gamma scan with custom mu range
./submit_array_job.sh -mq 9.0 -lambda1 5.0 -gammavalues -26.0 -24.0 -22.0 -lambda4 4.2 -mumax 150.0 -mupoints 15

# Single parameter with multiple values
./submit_array_job.sh -mqvalues 8.0 9.0 10.0 11.0 12.0 -lambda1 5.0
```

## Resource Guidelines

### Single Job (`slurm_unified_batch.sh`)
- **CPUs**: 8 (for multiple parameter sets)
- **Memory**: 16G 
- **Time**: 48:00:00 (adjust based on parameter count)

### Job Array (`slurm_batch_array.sh`)
- **CPUs**: 4 (per parameter combination)
- **Memory**: 8G
- **Time**: 12:00:00 (adjust based on mu/T grid size)

## Output Organization

Both scripts will create results in two locations:

### Local Output (always created):
- **Data**: `phase_data/phase_diagram_ml{mq}_lambda{lambda1}.csv`
- **Plots**: `phase_plots/phase_diagram_ml{mq}_lambda{lambda1}.png`
- **Sigma**: `sigma_data/sigma_calculations.csv` (chiral condensate calculations)
- **Logs**: `slurm_logs/` directory with job output/error files

### Shared Project Output (if PROJECT_DIR is set):
- **Data**: `$PROJECT_DIR/phase_data/phase_diagram_ml{mq}_lambda{lambda1}.csv`
- **Plots**: `$PROJECT_DIR/phase_plots/phase_diagram_ml{mq}_lambda{lambda1}.png`
- **Sigma**: `$PROJECT_DIR/sigma_data/sigma_calculations.csv` (aggregated from all jobs)
- **Backups**: `$PROJECT_DIR/sigma_data/sigma_backup_*.csv` (only for substantial data contributions)
- **Snapshots**: `$PROJECT_DIR/sigma_data/sigma_snapshot_*.csv` (pre-job copies of master file)

**Backup Strategy:**
- **Snapshots**: Created before each batch job starts (protects against overwrite)
- **Backups**: Created per SLURM job/task, not per individual sigma calculation
- Only jobs that generate >10 lines (array tasks) or >50 lines (unified jobs) create backups
- Use `./clean_sigma_backups.sh` to remove backups and snapshots older than 7 days

### Setting Up Shared Output:
```bash
# For QUENCH project (recommended):
export PROJECT_DIR="/net/project/QUENCH"

# Check if directory exists and is writable:
ls -la $PROJECT_DIR
touch $PROJECT_DIR/test_write && rm $PROJECT_DIR/test_write

# Submit jobs (will now copy to shared directory):
./submit_array_job.sh -mqvalues 9.0 12.0 15.0 -lambda1 5.0
```

### Benefits of Shared Output:
- **Collaboration**: All team members can access results
- **Centralized**: No need to search individual directories
- **Backup**: Results exist in both locations
- **Permissions**: Shared directory may have better backup policies
- **Sigma Aggregation**: All chiral condensate calculations combined in one master file
- **Data Safety**: Timestamped backups prevent accidental data loss
- **Pre-Job Snapshots**: Master file copied before each batch job starts
- **Parallel Safe**: File locking prevents conflicts between simultaneous jobs
- **Smart Cleanup**: Remove old backups/snapshots while preserving recent ones

## Common Commands

### Check Job Status
```bash
squeue -u $USER                    # Current jobs
sacct -j JOBID                     # Job accounting info
scancel JOBID                      # Cancel job
```

### Check Resources
```bash
sinfo -p general                   # Partition info
scontrol show partition general    # Detailed partition info
```

### Array Job Management
```bash
scancel JOBID_[1-10]              # Cancel specific array tasks
scancel JOBID                     # Cancel entire array
```

### Sigma Data Maintenance
```bash
# Check sigma data status
ls -la $PROJECT_DIR/sigma_data/

# Clean old backup and snapshot files (older than 7 days)
./clean_sigma_backups.sh

# Clean files older than specific days
./clean_sigma_backups.sh /net/project/QUENCH 3

# Check file sizes and line counts
wc -l $PROJECT_DIR/sigma_data/*.csv
du -h $PROJECT_DIR/sigma_data/
```

## Troubleshooting

### Module Loading
If Python modules fail, check available modules:
```bash
module avail python
module load python/3.9  # or whatever version is available
```

### Memory Issues
If jobs fail with memory errors:
- Increase `--mem` in SLURM directives
- Reduce `mupoints` or temperature ranges
- Check actual memory usage with `sacct -j JOBID --format=JobID,MaxRSS`

### GitHub Access
If git clone fails:
```bash
# Test SSH connection:
ssh -T git@github.com

# If that fails, check your SSH key:
cat ~/.ssh/id_rsa.pub
# Add this key to GitHub settings
```

## Script Selection Guide

Choose **Single Job** (`slurm_unified_batch.sh`) when:
- Small parameter scans (< 10 combinations)
- You want everything in one place
- Parameter combinations are interdependent
- You prefer the full Python argument parsing

Choose **Job Array** (`slurm_batch_array.sh`) when:
- Large parameter scans (> 10 combinations)
- You want parallel execution
- Individual combinations can fail independently
- You need fine-grained resource control per combination
- You want to use the convenient `submit_array_job.sh` helper

**Recommended:** Use the job array approach with `./submit_array_job.sh` for most cases - it's more flexible and handles the array size calculation automatically.
