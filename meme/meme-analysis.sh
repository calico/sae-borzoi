#!/bin/bash

# Check if input directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_directory>"
    echo "Input directory should contain .fa files"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${INPUT_DIR}/meme_analysis_results"
DATE_STAMP=$(date +%Y%m%d_%H%M%S)
MEME_INSTALL_DIR="/home/anya/code/meme-5.4.1"

# Create output directory structure
mkdir -p "${OUTPUT_DIR}/${DATE_STAMP}/meme_test_out"
mkdir -p "${OUTPUT_DIR}/${DATE_STAMP}/tomtom_test_archetype_out"

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed or not in PATH"
        exit 1
    fi
}

# Check if required commands are available
check_command "${MEME_INSTALL_DIR}/src/meme"
check_command "${MEME_INSTALL_DIR}/src/tomtom"

# Process each .fa file in the input directory
for fa_file in "${INPUT_DIR}"/node_seqs_test_1000/*.fa; do
    if [ -f "$fa_file" ]; then
        filename=$(basename "$fa_file" .fa)
        echo "Processing $filename..."

        # if "${OUTPUT_DIR}/${DATE_STAMP}/meme_test_out/${filename}" does not exist
        if [ -d "${OUTPUT_DIR}/${DATE_STAMP}/meme_test_out/${filename}" ]; then
            echo "Output directory for $filename already exists. Skipping..."
            continue
        fi

        # Run MEME
        echo "Running MEME on $filename..."
        ${MEME_INSTALL_DIR}/src/meme "$fa_file" \
            -o "${OUTPUT_DIR}/${DATE_STAMP}/meme_test_out/${filename}" \
            -dna \
            -nmotifs 2 \
            -oc "${OUTPUT_DIR}/${DATE_STAMP}/meme_test_out/${filename}"

        # Check if MEME was successful
        if [ $? -eq 0 ]; then
            echo "MEME analysis completed for $filename"

            python ./meme-parser.py "${OUTPUT_DIR}/${DATE_STAMP}/meme_test_out/${filename}/meme.txt" -o "${OUTPUT_DIR}/${DATE_STAMP}/meme_test_out/${filename}/motifs.tsv"
            
            # Run TomTom on MEME output
            echo "Running TomTom on MEME results for $filename..."
            ${MEME_INSTALL_DIR}/src/tomtom \
                "${OUTPUT_DIR}/${DATE_STAMP}/meme_test_out/${filename}/meme.txt" \
                "${MEME_INSTALL_DIR}/motif_databases/archetype_motifs.meme" \
                -o "${OUTPUT_DIR}/${DATE_STAMP}/tomtom_test_archetype_out/${filename}"

            if [ $? -eq 0 ]; then
                echo "TomTom analysis completed for $filename"
            else
                echo "Error: TomTom analysis failed for $filename"
            fi
        else
            echo "Error: MEME analysis failed for $filename"
        fi
    fi
done

# Create summary report
echo "Creating summary report..."
cat > "${OUTPUT_DIR}/${DATE_STAMP}/summary.txt" << EOF
MEME Suite Analysis Summary
Date: $(date)
Input Directory: ${INPUT_DIR}

Processed Files:
$(ls -1 "${INPUT_DIR}"/node_seqs_sorted_1000/*.fa)

Results Location:
MEME results: ${OUTPUT_DIR}/${DATE_STAMP}/meme_out/
TomTom results: ${OUTPUT_DIR}/${DATE_STAMP}/tomtom_out/
EOF

echo "Analysis complete! Results are in ${OUTPUT_DIR}/${DATE_STAMP}/"
