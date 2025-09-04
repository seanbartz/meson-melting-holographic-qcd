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

# Analyze malformed lines first
MALFORMED_FILE="$SIGMA_DIR/sigma_malformed_lines.csv"
echo ""
echo "=== ANALYZING MALFORMED LINES ==="
python3 -c "
import csv
import sys

input_file = '$SIGMA_FILE'
malformed_file = '$MALFORMED_FILE'
expected_fields = None
malformed_lines = []
total_lines = 0

print('Scanning for malformed lines...')
with open(input_file, 'r') as f:
    reader = csv.reader(f)
    try:
        header = next(reader)
        expected_fields = len(header)
        print(f'Expected fields per line: {expected_fields}')
        
        # Write malformed lines to separate file
        with open(malformed_file, 'w', newline='') as mf:
            writer = csv.writer(mf)
            # Write header for malformed file
            malformed_header = header + ['actual_fields', 'line_number', 'extra_data']
            writer.writerow(malformed_header)
            
            for line_num, row in enumerate(reader, start=2):
                total_lines += 1
                if len(row) != expected_fields:
                    # Store malformed line info
                    malformed_info = row[:expected_fields] + [len(row), line_num] + row[expected_fields:]
                    writer.writerow(malformed_info)
                    malformed_lines.append((line_num, len(row), row))
                    
                    if len(malformed_lines) <= 5:  # Show first 5 examples
                        print(f'Line {line_num}: {len(row)} fields (expected {expected_fields})')
                        if len(row) > expected_fields:
                            extra = row[expected_fields:]
                            print(f'  Extra data: {extra[:3]}...' if len(extra) > 3 else f'  Extra data: {extra}')
        
        print(f'')
        print(f'Total data lines scanned: {total_lines}')
        print(f'Malformed lines found: {len(malformed_lines)}')
        print(f'Malformed lines saved to: {malformed_file}')
        
        if len(malformed_lines) > 0:
            print(f'Corruption rate: {100*len(malformed_lines)/total_lines:.2f}%')
            
            # Analyze patterns in malformed lines
            field_counts = {}
            for line_num, field_count, row in malformed_lines:
                field_counts[field_count] = field_counts.get(field_count, 0) + 1
            
            print(f'Field count distribution in malformed lines:')
            for count in sorted(field_counts.keys()):
                print(f'  {count} fields: {field_counts[count]} lines')
        else:
            print('No malformed lines detected - CSV appears clean!')
            
    except Exception as e:
        print(f'Error analyzing file: {e}')
        sys.exit(1)
"

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
    
    # Analyze what might be causing malformed lines
    echo "=== MALFORMED LINE ANALYSIS ==="
    if [ -f "$MALFORMED_FILE" ]; then
        malformed_count=$(tail -n +2 "$MALFORMED_FILE" | wc -l)
        if [ "$malformed_count" -gt 0 ]; then
            echo "Malformed lines file: $MALFORMED_FILE ($malformed_count lines)"
            echo ""
            echo "Potential causes of CSV corruption:"
            echo "1. Concurrent writes to CSV file from multiple processes"
            echo "2. Process interruption during CSV write operations"
            echo "3. Data containing unescaped commas, quotes, or newlines"
            echo "4. Buffer flushing issues during high-throughput calculations"
            echo ""
            echo "Sample analysis of extra data (first 3 malformed lines):"
            python3 -c "
import csv
malformed_file = '$MALFORMED_FILE'
try:
    with open(malformed_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for i, row in enumerate(reader):
            if i >= 3:  # Only show first 3
                break
            line_num = row[-len(row)+len(header)-3] if len(row) > len(header)-3 else 'N/A'
            actual_fields = row[-len(row)+len(header)-2] if len(row) > len(header)-2 else 'N/A'
            extra_data = row[len(header)-3:] if len(row) > len(header)-3 else []
            print(f'Line {line_num}: {actual_fields} fields')
            if extra_data:
                print(f'  Extra: {extra_data[:5]}' + ('...' if len(extra_data) > 5 else ''))
except Exception as e:
    print(f'Could not analyze malformed data: {e}')
"
        else
            echo "No malformed lines found - CSV is clean!"
        fi
    fi
    
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
