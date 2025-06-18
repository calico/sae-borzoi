#!bin/sh

cp layer* ~/code/sae-vis/
cp /home/anya/code/sae_borzoi_4/models/conv1d_4_noabs_4_topk0.05_lr1e-05/umap_data.csv ~/code/sae-vis/layer4_umap_data.csv
cp /home/anya/code/sae_borzoi_4/models/conv1d_4_noabs_4_topk0.05_lr1e-05/model_cCREs_intersect.csv ~/code/sae-vis/layer4_ccres.csv
cp /home/anya/code/sae_borzoi_4/models/conv1d_4_noabs_4_topk0.05_lr1e-05/model_rmsk_intersect_top_nodes.csv ~/code/sae-vis/layer4_interesting_nodes_rmsk.csv
cp /home/anya/code/sae_borzoi_4/models/conv1d_4_noabs_4_topk0.05_lr1e-05/model_rmsk_intersect.csv ~/code/sae-vis/layer4_rmsk.csv
cp /home/anya/code/sae_borzoi_4/models/conv1d_4_noabs_4_topk0.05_lr1e-05/pct_seqlets_has_acts_test.csv ~/code/sae-vis/layer4_freq.csv