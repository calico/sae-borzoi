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

layer_number = configs['layer_name'][-1]
database_path = os.path.join(model_path, f'discovered_motifs_database_l{layer_number}.meme')

try:
    result_path = os.path.join(model_path, "meme_analysis_results")

    timestamp_results = os.listdir(result_path)
    # pick the latest timestamp
    timestamp_results = sorted(timestamp_results, reverse=True)
    timestamp_results = timestamp_results[0]
    tomtom_path = os.path.join(result_path, timestamp_results)

    if not os.path.exists(os.path.join(tomtom_path, 'tomtom_self_out')):
        os.makedirs(os.path.join(tomtom_path, 'tomtom_self_out'))

    slurm_string = '#!/bin/bash \n \n'
    slurm_string += '#SBATCH -p standard \n \n'
    slurm_string += f'#SBATCH -n 1 \n#SBATCH -c 2 \n#SBATCH -J tmtm \n'
    slurm_string += f"#SBATCH -o meme/temp/job_tomtom_self.out \n"
    slurm_string += f"#SBATCH -e meme/temp/job_tomtom_self.err \n"
    slurm_string += '#SBATCH --mem 22000 \n#SBATCH --time 2-0:0:0 \n'
    bash_script = os.path.join(cwd, 'analysis/tomtom-analysis-self.sh')
    slurm_string += f"source /home/anya/.bashrc; echo $HOSTNAME; bash {bash_script} {tomtom_path} {database_path}"

    with open(os.path.join(cwd, f'meme/temp/job_tomtom_self.sb'), 'w') as f:
        f.write(slurm_string)

    # run with subprocess
    subprocess.run(['sbatch', f'meme/temp/job_tomtom_self.sb'])
    
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check if the model path and result path are correct.")
    exit(1)
    