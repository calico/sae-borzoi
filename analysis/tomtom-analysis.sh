#!/bin/bash

# Check if input directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_directory>"
    echo "Input directory should contain .fa files"
    exit 1
fi

INPUT_DIR="$1"
MEME_INSTALL_DIR="/home/anya/code/meme-5.4.1"

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
for filename in "${INPUT_DIR}"/meme_test_out/*; do
    # Run TomTom on MEME output
    echo "Running TomTom on MEME results for $filename..."
    basename=$(basename "$filename")

    # if ${INPUT_DIR}/tomtom_test_archetype_out/${basename}/tomtom.tsv exists then skip
    if [ -f "${INPUT_DIR}/tomtom_test_archetype_out/${basename}/tomtom.tsv" ]; then
        echo "TomTom results for $basename already exist. Skipping..."
        continue
    fi
    
    ${MEME_INSTALL_DIR}/src/tomtom \
        "${filename}/meme.txt" \
        "${MEME_INSTALL_DIR}/motif_databases/archetype_motifs.meme" \
        -o "${INPUT_DIR}/tomtom_test_archetype_out/${basename}"

    if [ $? -eq 0 ]; then
        echo "TomTom analysis completed for $basename"
    else
        echo "Error: TomTom analysis failed for $basename"
    fi

done
