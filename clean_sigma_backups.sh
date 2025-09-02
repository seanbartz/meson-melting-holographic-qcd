#!/bin/bash
# Cleanup script for sigma backup files in QUENCH directory
# Usage: ./clean_sigma_backups.sh [PROJECT_DIR] [days_old]

PROJECT_DIR="${1:-/net/project/QUENCH}"
DAYS_OLD="${2:-7}"

echo "=== Sigma Backup Cleanup ==="
echo "Project directory: $PROJECT_DIR"
echo "Removing backups older than: $DAYS_OLD days"
echo ""

SIGMA_DIR="$PROJECT_DIR/sigma_data"

if [[ ! -d "$SIGMA_DIR" ]]; then
    echo "❌ Sigma directory not found: $SIGMA_DIR"
    exit 1
fi

echo "Current sigma backup and snapshot files:"
ls -la "$SIGMA_DIR"/sigma_backup_*.csv 2>/dev/null || echo "No backup files found"
ls -la "$SIGMA_DIR"/sigma_snapshot_*.csv 2>/dev/null || echo "No snapshot files found"
echo ""

# Count files before cleanup
BACKUP_BEFORE=$(ls "$SIGMA_DIR"/sigma_backup_*.csv 2>/dev/null | wc -l)
SNAPSHOT_BEFORE=$(ls "$SIGMA_DIR"/sigma_snapshot_*.csv 2>/dev/null | wc -l)
TOTAL_BEFORE=$((BACKUP_BEFORE + SNAPSHOT_BEFORE))

if [[ $TOTAL_BEFORE -eq 0 ]]; then
    echo "No backup or snapshot files to clean up."
    exit 0
fi

echo "Found $BACKUP_BEFORE backup files and $SNAPSHOT_BEFORE snapshot files"

# Remove files older than specified days
echo "Removing backup and snapshot files older than $DAYS_OLD days..."
find "$SIGMA_DIR" -name "sigma_backup_*.csv" -type f -mtime +$DAYS_OLD -delete
find "$SIGMA_DIR" -name "sigma_snapshot_*.csv" -type f -mtime +$DAYS_OLD -delete

# Count files after cleanup
BACKUP_AFTER=$(ls "$SIGMA_DIR"/sigma_backup_*.csv 2>/dev/null | wc -l)
SNAPSHOT_AFTER=$(ls "$SIGMA_DIR"/sigma_snapshot_*.csv 2>/dev/null | wc -l)
TOTAL_AFTER=$((BACKUP_AFTER + SNAPSHOT_AFTER))
REMOVED_COUNT=$((TOTAL_BEFORE - TOTAL_AFTER))

echo ""
echo "Cleanup complete:"
echo "  Files removed: $REMOVED_COUNT"
echo "  Backup files remaining: $BACKUP_AFTER"
echo "  Snapshot files remaining: $SNAPSHOT_AFTER"

if [[ $TOTAL_AFTER -gt 0 ]]; then
    echo ""
    echo "Remaining files:"
    ls -la "$SIGMA_DIR"/sigma_backup_*.csv 2>/dev/null
    ls -la "$SIGMA_DIR"/sigma_snapshot_*.csv 2>/dev/null
fi

echo ""
echo "Master sigma file status:"
if [[ -f "$SIGMA_DIR/sigma_calculations.csv" ]]; then
    MASTER_LINES=$(wc -l < "$SIGMA_DIR/sigma_calculations.csv")
    MASTER_SIZE=$(du -h "$SIGMA_DIR/sigma_calculations.csv" | cut -f1)
    echo "  Lines: $MASTER_LINES"
    echo "  Size: $MASTER_SIZE"
else
    echo "  Master file not found"
fi
