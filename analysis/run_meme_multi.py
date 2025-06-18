import os
import subprocess
import json


cwd = os.getcwd()

if not os.path.exists('meme/temp'):
    os.makedirs('meme/temp')

config_file = "config/config.json"
with open(config_file) as config_open:
    configs = json.load(config_open)

exp_factor = configs['expansion_factor']
topk = configs['topk_pct']
lr = configs['learning_rate']
global_model_path = configs['model_save_path']

if configs['prespecified_model'] is not None:
    model_path = os.path.join(global_model_path, configs['prespecified_model'])
else:
    model_path = os.path.join(global_model_path, f"{configs['layer_name']}_exp{exp_factor}_topk{topk}_lr{lr}")

slurm_string = '#!/bin/bash \n \n'
slurm_string += '#SBATCH -p standard \n \n'
slurm_string += '#SBATCH -n 1 \n#SBATCH -c 2 \n#SBATCH -J meme \n'
slurm_string += f"#SBATCH -o meme/temp/job_meme.out \n"
slurm_string += f"#SBATCH -e meme/temp/job_meme.err \n"
slurm_string += '#SBATCH --mem 22000 \n#SBATCH --time 7-0:0:0 \n'
bash_script = os.path.join(cwd, 'meme/meme-analysis.sh')
slurm_string += f"source /home/anya/.bashrc; echo $HOSTNAME; bash {bash_script} {model_path}"

with open(os.path.join(cwd, f'meme/temp/job_meme.sb'), 'w') as f:
    f.write(slurm_string)

# run with subprocess
subprocess.run(['sbatch', f'meme/temp/job_meme.sb'])
