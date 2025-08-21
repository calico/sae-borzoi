#!/bin/bash

# Path to your motif-visualization directory
PROJ_DIR="/home/anya/code/sae-vis"

# Base paths to your data
MEME_BASE="/home/anya/code/sae_borzoi_1/models/conv1d_1_noabs_4_topk0.05_lr1e-05/meme_analysis_results/20250620_105828/meme_test_out"
TOMTOM_BASE="/home/anya/code/sae_borzoi_1/models/conv1d_1_noabs_4_topk0.05_lr1e-05/meme_analysis_results/20250620_105828/tomtom_test_archetype_rna_out"

# Get list of node IDs (assuming they are directory names in the MEME_BASE path)
NODE_IDS=$(ls "$MEME_BASE")

# Create symlinks for each node
for NODE_ID in $NODE_IDS; do
    echo "Creating symlinks for node $NODE_ID"
    
    # Create target directory
    mkdir -p "$PROJ_DIR/data_l1/$NODE_ID"
    
    # Create symlinks to MEME files
    ln -sf "$MEME_BASE/$NODE_ID/meme.xml" "$PROJ_DIR/data_l1/$NODE_ID/meme.xml"
    ln -sf "$MEME_BASE/$NODE_ID/meme.txt" "$PROJ_DIR/data_l1/$NODE_ID/meme.txt"
    ln -sf "$MEME_BASE/$NODE_ID/meme.html" "$PROJ_DIR/data_l1/$NODE_ID/meme.html"
    ln -sf "$MEME_BASE/$NODE_ID/motifs.tsv" "$PROJ_DIR/data_l1/$NODE_ID/motifs.tsv"
    ln -sf "$MEME_BASE/$NODE_ID/logo1.png" "$PROJ_DIR/data_l1/$NODE_ID/logo1.png"
    
    # Create symlinks to TomTom files (if they exist)
    if [ -d "$TOMTOM_BASE/$NODE_ID" ]; then
        ln -sf "$TOMTOM_BASE/$NODE_ID/tomtom.xml" "$PROJ_DIR/data_l1/$NODE_ID/tomtom.xml"
        ln -sf "$TOMTOM_BASE/$NODE_ID/tomtom.tsv" "$PROJ_DIR/data_l1/$NODE_ID/tomtom.tsv"
    fi
done

echo "Finished creating symlinks"