#!bin/sh

source ~/.bashrc
conda activate basenji-torch2

stage=9
stop_stage=9

. ./parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python scripts/find_global_max.py
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python scripts/batch_train.py
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python scripts/batch_infer.py
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python scripts/save_seqlets.py
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    python analysis/run_meme_multi.py
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # ensure meme run is done
    while python analysis/check_readiness.py --stage meme | grep -q False; do
        echo "Waiting for MEME analysis to complete... Trying again in 2 hours."
        sleep 7200  # 2 hours in seconds
    done
    echo "MEME analysis is complete. Proceeding with post-analysis."
    python analysis/run_meme_post.py
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    python analysis/run_tomtom_multi.py
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    # ensure tomtom run is done
    while python analysis/check_readiness.py --stage tomtom | grep -q False; do
        echo "Waiting for TomTom analysis to complete... Trying again in 30 minutes."
        sleep 1800  # 30m
    done
    echo "TomTom analysis is complete. Proceeding with post-analysis."
    python analysis/umap_analysis.py
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    python analysis/seqlet_overlaps_analysis.py
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    python analysis/jaccard.py
fi
