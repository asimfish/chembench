#!/bin/bash
# Batch convert all Zarr datasets in a directory to HDF5 format

# Default values
ZARR_DIR="data/zarr/data/grasp/part1"
OUTPUT_BASE="dataset"
CAMERA_NAMES="head_camera chest_camera third_camera"
DUPLICATE_ARMS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --zarr_dir)
            ZARR_DIR="$2"
            shift 2
            ;;
        --output_base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --camera_names)
            CAMERA_NAMES="$2"
            shift 2
            ;;
        --no_duplicate_arms)
            DUPLICATE_ARMS=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --zarr_dir DIR          Directory containing .zarr files (default: data/zarr/data/grasp/part1)"
            echo "  --output_base DIR       Base directory for output (default: dataset)"
            echo "  --camera_names NAMES    Space-separated camera names (default: head_camera chest_camera third_camera)"
            echo "  --no_duplicate_arms     Don't duplicate single arm data to dual-arm format"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if zarr directory exists
if [ ! -d "$ZARR_DIR" ]; then
    echo "Error: Zarr directory not found: $ZARR_DIR"
    exit 1
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE"

# Find all .zarr directories
zarr_files=($(find "$ZARR_DIR" -maxdepth 1 -name "*.zarr" -type d))

if [ ${#zarr_files[@]} -eq 0 ]; then
    echo "No .zarr files found in $ZARR_DIR"
    exit 1
fi

echo "Found ${#zarr_files[@]} Zarr datasets"
echo "=========================================="
echo ""

# Process each zarr file
success_count=0
fail_count=0

for zarr_file in "${zarr_files[@]}"; do
    # Get basename without .zarr extension
    basename=$(basename "$zarr_file" .zarr)
    output_dir="$OUTPUT_BASE/grasp_$basename"
    
    echo "[$((success_count + fail_count + 1))/${#zarr_files[@]}] Processing: $basename"
    echo "  Input:  $zarr_file"
    echo "  Output: $output_dir"
    
    # Build convert command
    convert_cmd="python convert_zarr_to_hdf5.py --zarr_path \"$zarr_file\" --output_dir \"$output_dir\" --camera_names $CAMERA_NAMES"
    
    if [ "$DUPLICATE_ARMS" = true ]; then
        convert_cmd="$convert_cmd --duplicate_arms"
    fi
    
    # Run conversion
    if eval "$convert_cmd"; then
        echo "  ✓ Conversion successful"
        
        # Verify the converted data
        if python verify_converted_data.py --dataset_dir "$output_dir" > /dev/null 2>&1; then
            echo "  ✓ Verification passed"
            ((success_count++))
        else
            echo "  ✗ Verification failed"
            ((fail_count++))
        fi
    else
        echo "  ✗ Conversion failed"
        ((fail_count++))
    fi
    
    echo ""
done

# Summary
echo "=========================================="
echo "CONVERSION SUMMARY"
echo "=========================================="
echo "Total datasets: ${#zarr_files[@]}"
echo "Successful: $success_count"
echo "Failed: $fail_count"
echo ""

if [ $fail_count -eq 0 ]; then
    echo "✓ All conversions completed successfully!"
    exit 0
else
    echo "⚠ Some conversions failed. Please check the output above."
    exit 1
fi

