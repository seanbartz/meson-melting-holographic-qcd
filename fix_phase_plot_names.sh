#!/bin/bash
"""
Fix inconsistent phase plot filenames

Issues to fix:
1. "ml" -> "mq" 
2. Missing underscores between variable names and values
3. Missing gamma and lambda4 values (add defaults: gamma=-22.4, lambda4=4.2)
4. Standardize format: phase_diagram_improved_mq_X_lambda1_Y_gamma_Z_lambda4_W.png
"""

cd /Users/seanbartz/Dropbox/ISU/Research/QGP/DilatonMixing/MesonMelting/phase_plots

echo "=== Phase Plot Filename Standardization ==="
echo "Current directory: $(pwd)"
echo "Total files before: $(ls *.png | wc -l)"
echo ""

# Track renamed files
RENAMED_COUNT=0

# Function to rename with confirmation
rename_file() {
    local old_name="$1"
    local new_name="$2"
    
    if [[ "$old_name" != "$new_name" && -f "$old_name" ]]; then
        echo "RENAME: $old_name"
        echo "     -> $new_name"
        
        # Check if target exists
        if [[ -f "$new_name" ]]; then
            echo "     WARNING: Target file already exists, skipping"
            echo ""
            return 1
        fi
        
        mv "$old_name" "$new_name"
        ((RENAMED_COUNT++))
        echo ""
        return 0
    else
        return 1
    fi
}

echo "1. Fixing 'ml' -> 'mq' in phase_diagram_improved files..."

# Fix ml -> mq in files without gamma/lambda4
for file in phase_diagram_improved_ml*.png; do
    if [[ -f "$file" ]]; then
        new_name=$(echo "$file" | sed 's/phase_diagram_improved_ml/phase_diagram_improved_mq_/')
        
        # Add missing underscores and default gamma/lambda4 if not present
        if [[ ! "$new_name" =~ gamma ]]; then
            # Extract mq and lambda1 values
            if [[ "$new_name" =~ phase_diagram_improved_mq_([0-9]+\.?[0-9]*)_lambda1([0-9]+\.?[0-9]*) ]]; then
                mq_val="${BASH_REMATCH[1]}"
                lambda1_val="${BASH_REMATCH[2]}"
                new_name="phase_diagram_improved_mq_${mq_val}_lambda1_${lambda1_val}_gamma_-22.4_lambda4_4.2.png"
            fi
        fi
        
        rename_file "$file" "$new_name"
    fi
done

echo "2. Fixing missing underscores in phase_diagram files..."

# Fix phase_diagram files with ml and missing underscores
for file in phase_diagram_ml*.png; do
    if [[ -f "$file" && ! "$file" =~ _Old ]]; then
        # Extract values and standardize
        if [[ "$file" =~ phase_diagram_ml([0-9]+\.?[0-9]*)_lambda([0-9]+\.?[0-9]*) ]]; then
            mq_val="${BASH_REMATCH[1]}"
            lambda1_val="${BASH_REMATCH[2]}"
            new_name="phase_diagram_improved_mq_${mq_val}_lambda1_${lambda1_val}_gamma_-22.4_lambda4_4.2.png"
            rename_file "$file" "$new_name"
        fi
    fi
done

echo "3. Fixing 'ml' in other files with gamma/lambda4 already present..."

# Fix files that have ml but already have gamma/lambda4
for file in phase_diagram_improved_ml_*.png; do
    if [[ -f "$file" ]]; then
        new_name=$(echo "$file" | sed 's/phase_diagram_improved_ml_/phase_diagram_improved_mq_/')
        rename_file "$file" "$new_name"
    fi
done

echo "4. Fixing combined and parameter files..."

# Fix combined files with ml
for file in *_ml_*.png; do
    if [[ -f "$file" ]]; then
        new_name=$(echo "$file" | sed 's/_ml_/_mq_/')
        rename_file "$file" "$new_name"
    fi
done

echo "5. Summary of changes..."
echo "Files renamed: $RENAMED_COUNT"
echo "Total files after: $(ls *.png | wc -l)"
echo ""

echo "=== Final filename check ==="
echo "Files with 'ml' remaining:"
ls *ml*.png 2>/dev/null | wc -l

echo ""
echo "Files with standardized naming:"
ls phase_diagram_improved_mq_*_lambda1_*_gamma_*_lambda4_*.png 2>/dev/null | wc -l

echo ""
echo "All .png files:"
ls *.png | head -10
if [[ $(ls *.png | wc -l) -gt 10 ]]; then
    echo "... and $(($(ls *.png | wc -l) - 10)) more files"
fi

echo ""
echo "=== Rename completed ==="
