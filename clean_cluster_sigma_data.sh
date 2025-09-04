#!/bin/bash
# Helper script to clean sigma data on Obsidian cluster
# Usage: ./clean_cluster_sigma_data.sh [preview]

SIGMA_DIR="/net/project/QUENCH/sigma_data"
SIGMA_FILE="$SIGMA_DIR/sigma_calculations.csv"
CLEANED_FILE="$SIGMA_DIR/sigma_calculations_cleaned.csv"
BACKUP_FILE="$SIGMA_DIR/sigma_calculations_backup_$(date +%Y%m%d_%H%M%S).csv"

echo "=== Sigma Data Cleaning on Obsidian ==="
echo "Data directory: $SIGMA_DIR"
echo "Original file: $SIGMA_FILE"

# Check if sigma data exists
if [ ! -f "$SIGMA_FILE" ]; then
    echo "Error: Sigma data file not found at $SIGMA_FILE"
    echo "Available files in $SIGMA_DIR:"
    ls -la "$SIGMA_DIR" 2>/dev/null || echo "Directory not accessible"
    exit 1
fi

# Show file info
echo "Current sigma file size:"
ls -lh "$SIGMA_FILE"
echo "Number of lines: $(wc -l < "$SIGMA_FILE")"

# Preview mode
if [ "$1" == "preview" ]; then
    echo ""
    echo "=== PREVIEW MODE ==="
    python3 clean_sigma_duplicates.py "$SIGMA_FILE" --preview
    exit 0
fi

# Ask for confirmation
echo ""
read -p "Create backup and clean duplicates? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled"
    exit 0
fi

# Create backup
echo "Creating backup: $BACKUP_FILE"
cp "$SIGMA_FILE" "$BACKUP_FILE"

# Clean duplicates
echo "Cleaning duplicates..."
python3 clean_sigma_duplicates.py "$SIGMA_FILE" "$CLEANED_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Cleaning Results ==="
    echo "Original file: $(wc -l < "$SIGMA_FILE") lines"
    echo "Cleaned file: $(wc -l < "$CLEANED_FILE") lines" 
    echo "Backup saved: $BACKUP_FILE"
    echo ""
    echo "To use cleaned data for ML:"
    echo "  mv $CLEANED_FILE $SIGMA_FILE"
    echo ""
    echo "To restore original data:"
    echo "  mv $BACKUP_FILE $SIGMA_FILE"
else
    echo "Error: Cleaning failed"
    rm -f "$BACKUP_FILE"  # Remove backup if cleaning failed
    exit 1
fi
