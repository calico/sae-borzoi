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

try:
    result_path = os.path.join(model_path, "meme_analysis_results")
    timestamp_results = os.listdir(result_path)
    # pick the latest timestamp
    timestamp_results = sorted(timestamp_results, reverse=True)
    timestamp_results = timestamp_results[0]
    meme_path = os.path.join(result_path, timestamp_results)

    node_ids = os.path.join(meme_path, 'meme_test_out')
    if len(os.listdir(node_ids)) == 0:
        print(f"No node IDs found in {node_ids} for {meme_path}.")

    for node_id in os.listdir(node_ids):
        # if file doesn't already exist
        output_file = os.path.join(meme_path, 'meme_test_out', node_id, 'motifs.tsv')
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping.")
            continue

        bash_script = os.path.join(cwd, 'meme/meme-parser.py')
        input_file = os.path.join(meme_path, 'meme_test_out', node_id, 'meme.txt')
        output_file = os.path.join(meme_path, 'meme_test_out', node_id, 'motifs.tsv')

        # run via subprocess
        subprocess.run(['python', bash_script, input_file, '-o', output_file])
        
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check if the model path and result path are correct.")
    exit(1)
