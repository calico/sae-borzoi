#!bin/sh

source ~/.bashrc
conda activate basenji-torch2

stage=7
stop_stage=9

. ./parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python find_global_max.py
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python batch_train.py
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python batch_infer.py
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python save_seqlets.py
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    python meme/run_meme_multi.py
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    python meme/run_meme_post.py
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    python meme/run_tomtom_multi.py
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    python meme/umap_analysis.py
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    python meme/seqlet_overlaps_analysis.py
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    python meme/jaccard.py
fi
