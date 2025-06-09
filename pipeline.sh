#!bin/sh

python find_global_max.py
python batch_train.py
python batch_infer.py
python save_seqlets.py
python meme/run_meme_multi.py
python meme/run_meme_post.py
python meme/run_tomtom_multi.py
# cCRE and TE overlaps
# jaccard
# umap