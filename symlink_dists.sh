#!/bin/bash

# Path to your motif-visualization directory
PROJ_DIR="/home/anya/code/sae-vis"

# Base paths to your data
DIST_BASE="/home/anya/code/sae_borzoi_4/models/conv1d_4_noabs_4_topk0.05_lr1e-05/node_seqs_test_1000"

# Get list of node IDs (assuming they are directory names in the MEME_BASE path)
NODE_IDS=$(ls "$DIST_BASE")

# Create symlinks for each node
for NODE_ID in $NODE_IDS; do
    echo "Creating .fa symlinks for node $NODE_ID"
    # take NODE_ID without the last 3 characters
    NODE_ID=${NODE_ID:0:-3}

    # Create symlinks to MEME files
    ln -sf "$DIST_BASE/$NODE_ID.fa" "$PROJ_DIR/data_l4/$NODE_ID/loci.fa"

done

DIST_BASE1="/home/anya/code/sae_borzoi_4/models/conv1d_4_noabs_4_topk0.05_lr1e-05/node_act_dists_test"

# Create symlinks for each node
for NODE_ID in $NODE_IDS; do
    echo "Creating dist symlinks for node $NODE_ID"
    # take NODE_ID without the last 3 characters
    NODE_ID=${NODE_ID:0:-3}

    # Create symlinks to MEME files
    ln -sf "$DIST_BASE1/$NODE_ID.npy" "$PROJ_DIR/data_l4/$NODE_ID/dist_acts.npy"

done

echo "Finished creating symlinks"
