#!bin/sh

source ~/.bashrc
conda activate basenji-torch2

stage=5
stop_stage=5

. ./parse_options.sh || exit 1;

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python scripts/batch_infer_.py
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python scripts/save_seqlets_batch.py
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python analysis/run_meme_batch.py
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    python analysis/run_meme_post_batch.py
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    python analysis/run_tomtom_batch.py
fi
