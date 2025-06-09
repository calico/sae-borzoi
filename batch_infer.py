import os
import json

import numpy as np
import subprocess

from sae import *
from dataset import *


config_file = "config.json"
with open(config_file) as configs_open:
    configs = json.load(configs_open)

if not os.path.exists('temp'):
    os.makedirs('temp')

exp_factor = configs['expansion_factor']
topk = configs['topk_pct']
lr = configs['learning_rate']
global_model_path = configs['model_save_path']

if not os.path.exists(global_model_path):
    os.makedirs(global_model_path)

model_save_path = os.path.join(global_model_path, f"{configs['layer_name']}_exp{exp_factor}_topk{topk}_lr{lr}")

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

slurm_string = '#!/bin/bash \n \n'
slurm_string += '#SBATCH -p gpu \n \n'
slurm_string += '#SBATCH -n 1 \n#SBATCH -c 2 \n#SBATCH -J sae-infer \n'
slurm_string += f"#SBATCH -o {model_save_path}/infer.out \n"
slurm_string += f"#SBATCH -e {model_save_path}/infer.err \n"
slurm_string += '#SBATCH --mem 28000 \n#SBATCH --time 1-0:0:0 \n'
slurm_string += '#SBATCH --gres=gpu:1 \n'  # Request 1 GPU
slurm_string += f'. /home/anya/anaconda3/etc/profile.d/conda.sh; conda activate basenji-torch2; echo $HOSTNAME; python infer_one_instance.py --topk {topk} --exp_factor {exp_factor} --lr {lr} --top_acts 8'

with open(f'temp/job_infer_{exp_factor}_topk{topk}_lr{lr}.sb', 'w') as f:
    f.write(slurm_string)

# run with subprocess
subprocess.run(['sbatch', f'temp/job_infer_{exp_factor}_topk{topk}_lr{lr}.sb'])
