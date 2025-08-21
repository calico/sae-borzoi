import os
import sys
import json
import argparse
import os


# parse arguments
parser = argparse.ArgumentParser(description="Check if a path exists.")
parser.add_argument('--stage', type=str, help='Stage to check: either "meme" or "tomtom"', required=True)
args = parser.parse_args()

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

expected_folders = len(os.listdir(os.path.join(model_path, "node_seqs_test_1000")))

if args.stage == 'meme':
    result_path = os.path.join(model_path, "meme_analysis_results")
    timestamp_results = os.listdir(result_path)
    # pick the latest timestamp
    timestamp_results = sorted(timestamp_results, reverse=True)
    timestamp_results = timestamp_results[0]
    meme_path = os.path.join(result_path, timestamp_results)
    if not os.path.exists(os.path.join(meme_path, 'meme_test_out')):
        print(False)
        exit(1)
    else:
        node_ids = os.path.join(meme_path, 'meme_test_out')
        if len(os.listdir(node_ids)) != expected_folders:
            print(False)
            exit(1)
        else:
            print(True)
            exit(0)

elif args.stage == 'tomtom':
    result_path = os.path.join(model_path, "meme_analysis_results")
    timestamp_results = os.listdir(result_path)
    # pick the latest timestamp
    timestamp_results = sorted(timestamp_results, reverse=True)
    timestamp_results = timestamp_results[0]
    meme_path = os.path.join(result_path, timestamp_results)
    if not os.path.exists(os.path.join(meme_path, 'tomtom_test_archetype_rna_out')):
        print(False)
        exit(1)
    else:
        node_ids = os.path.join(meme_path, 'tomtom_test_archetype_rna_out')
        if len(os.listdir(node_ids)) != expected_folders:
            print(len(os.listdir(node_ids)), expected_folders)
            # print the missing node IDs
            node_names = os.listdir(os.path.join(model_path, "node_seqs_test_1000"))
            node_names = [x.split('.')[0] for x in node_names]
            missing_node_ids =  set(node_names) - set(os.listdir(node_ids))
            print("Missing node IDs:", missing_node_ids)
            if len(missing_node_ids)>10:
                print(False)
                exit(1)
            else:
                print(True)
                exit(0)
        else:
            print(True)
            exit(0)
