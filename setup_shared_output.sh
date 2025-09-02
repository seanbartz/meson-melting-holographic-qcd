#!/bin/bash
# Setup script for shared QUENCH project directory
# Usage: ./setup_shared_output.sh [project_directory]

PROJECT_DIR="${1:-/net/project/QUENCH}"

echo "=== QUENCH Project Shared Output Setup ==="
echo "Project directory: $PROJECT_DIR"
echo ""

# Check if directory exists
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "❌ ERROR: Project directory does not exist: $PROJECT_DIR"
    echo "Please check the path or contact your system administrator."
    exit 1
fi

echo "✓ Project directory exists: $PROJECT_DIR"

# Check write permissions
if ! touch "$PROJECT_DIR/test_write_$$" 2>/dev/null; then
    echo "❌ ERROR: Cannot write to project directory: $PROJECT_DIR"
    echo "Check permissions or contact your system administrator."
    exit 1
else
    rm -f "$PROJECT_DIR/test_write_$$"
    echo "✓ Write permissions confirmed"
fi

# Create subdirectories
echo ""
echo "Creating output subdirectories..."
mkdir -p "$PROJECT_DIR/phase_data" && echo "✓ Created: $PROJECT_DIR/phase_data"
mkdir -p "$PROJECT_DIR/phase_plots" && echo "✓ Created: $PROJECT_DIR/phase_plots"
mkdir -p "$PROJECT_DIR/sigma_data" && echo "✓ Created: $PROJECT_DIR/sigma_data"

# Set up environment variable
echo ""
echo "Setting up environment variable..."
echo "export PROJECT_DIR=\"$PROJECT_DIR\"" >> ~/.bashrc
echo "✓ Added PROJECT_DIR to ~/.bashrc"

# Source for current session
export PROJECT_DIR="$PROJECT_DIR"
echo "✓ Set PROJECT_DIR for current session"

# Show final status
echo ""
echo "=== Setup Complete ==="
echo "Shared output is now configured:"
echo "  PROJECT_DIR = $PROJECT_DIR"
echo "  Data will be saved to: $PROJECT_DIR/phase_data/"
echo "  Plots will be saved to: $PROJECT_DIR/phase_plots/"
echo "  Sigma calculations will be saved to: $PROJECT_DIR/sigma_data/"
echo ""
echo "Usage:"
echo "  ./submit_array_job.sh -mqvalues 9.0 12.0 15.0 -lambda1 5.0"
echo "  (Results will be copied to shared directory automatically)"
echo ""
echo "Sigma data handling:"
echo "  - Local sigma_calculations.csv files are appended to shared file"
echo "  - Timestamped backups are created to prevent data loss"
echo "  - File locking prevents conflicts between parallel jobs"
echo ""
echo "To verify setup:"
echo "  echo \$PROJECT_DIR"
echo "  ls -la \$PROJECT_DIR"
echo ""

# Show current directory contents
if [[ -d "$PROJECT_DIR/phase_data" ]] || [[ -d "$PROJECT_DIR/phase_plots" ]] || [[ -d "$PROJECT_DIR/sigma_data" ]]; then
    echo "Current shared directory contents:"
    ls -la "$PROJECT_DIR/phase_data/" 2>/dev/null | head -5
    ls -la "$PROJECT_DIR/phase_plots/" 2>/dev/null | head -5
    ls -la "$PROJECT_DIR/sigma_data/" 2>/dev/null | head -5
fi
