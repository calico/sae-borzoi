import os
import json

import torch.nn as nn
import numpy as np
import subprocess

from sae import *
from dataset import *


params_file = "params_grid.json"
with open(params_file) as params_open:
    params = json.load(params_open)

config_file = "config.json"
with open(config_file) as configs_open:
    configs = json.load(configs_open)

if not os.path.exists('temp'):
    os.makedirs('temp')

for exp_factor in params['expansion_factors']:
    for topk in params['topk_pct']:
        for lr in params['learning_rate']:
            model_save_path = f"models/{configs['layer_name']}_noabs_{exp_factor}_topk{topk}_lr{lr}"

            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            slurm_string = '#!/bin/bash \n \n'
            slurm_string += '#SBATCH -p minigpu \n \n'
            slurm_string += '#SBATCH -n 1 \n#SBATCH -c 2 \n#SBATCH -J sae-borzoi \n'
            slurm_string += f"#SBATCH -o /home/anya/code/sae_borzoi/{model_save_path}/job0.out \n"
            slurm_string += f"#SBATCH -e /home/anya/code/sae_borzoi/{model_save_path}/job0.err \n"
            slurm_string += '#SBATCH --mem 28000 \n#SBATCH --time 2-0:0:0 \n'
            slurm_string += '#SBATCH --gres=gpu:1 \n'  # Request 1 GPU
            slurm_string += f'. /home/anya/anaconda3/etc/profile.d/conda.sh; conda activate basenji-torch2; echo $HOSTNAME; python train_one_instance.py --topk {topk} --exp_factor {exp_factor} --lr {lr}'

            # save to temp/job_{}_opk{}_lr{}.sb

            with open(f'temp/job_{exp_factor}_topk{topk}_lr{lr}.sb', 'w') as f:
                f.write(slurm_string)

            # run with subprocess
            subprocess.run(['sbatch', f'temp/job_{exp_factor}_topk{topk}_lr{lr}.sb'])
            