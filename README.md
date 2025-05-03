# Sparse autoencoders for mechanistic interpretability of the DNA sequence-based model [Borzoi](https://www.nature.com/articles/s41588-024-02053-6) @ f(DNA) Calico

We aim to use sparse autoencoders to decompose the activations of the pre-trained Borzoi model into monosemantic concepts that map to known and unknown regulatory motifs.

We use the top K approach for sparsity described by [L. Gao et al.](https://cdn.openai.com/papers/sparse-autoencoders.pdf) to reconstruct the activations of the first few convoluational layers.

Example: SAE is learning from the raw conv1d_2 layer output. At this point, each position has seen 18 nucleotides of the input. Training params: expansion factor = 4, LR = 1e-5, global max is on (for each feature dimension in the output, the global max among activations of training sequences was found, and activations were divided by the value), top K activations to keep = 5% (topK along sequence, i.e. top 10% for each seqlet). No L1 loss was used, only MSE and topK. Training was done on input sequences split into 4 parts each (input_len = 5kb/4) to fit into memory more efficiently.

To analyze activations, I first found the top K (=8) chunks of sequence activating each SAE node, i.e. (8, hidden_dim) per input sequence for 400 sequences, where the input sequence is ¼ of the original length. I extracted seqlets corresponding to these activations, and found nodes with 1) at least 1000 seqlets with nonzero activation, 2) at least 200 seqlets with activations > 1/2 mean node activation. The nodes are sorted by mean activation for analysis, i.e. node 1_nXXXX is the top node by mean activation value.

All seqlets from these nodes were saved into separate .fa files and analyzed with MEME with max 2 motifs discovered per node, and subsequently TomTom for each node-discovered PWMs using J. Vierstra's archetypical [motif database](https://www.vierstra.org/resources/motif_clustering). Significance was filtered both for MEME results (E-value<0.05) and TomTom (p-val, E-val, q-val all <0.05). Resulting nodes are visualized with the SAE-vis server, local network address: http://10.11.12.147:7080/.
