import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json


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

cwd = os.getcwd()

result_path = os.path.join(model_path, "meme_analysis_results")
timestamp_results = os.listdir(result_path)
# pick the latest timestamp
timestamp_results = sorted(timestamp_results, reverse=True)
timestamp_results = timestamp_results[0]
tomtom_path = os.path.join(result_path, timestamp_results, 'tomtom_self_out')

# load json
with open(f"layer{configs['layer_name'][-1]}_motif_nodes.json", "r") as f:
    dict_motifs = json.load(f)

for motif in tqdm(dict_motifs.keys()):
    if len(dict_motifs[motif])>1:
        print(f"Processing motif: {motif}")
        jaccard_df = pd.read_csv(os.path.join(model_path, f"jaccard/{motif}_jaccard.csv"))
        # set diagonal values to 0
        for i in range(len(jaccard_df)):
            jaccard_df.iloc[i, i] = 0.0
        max_jaccards = []
        max_evals = []
        eval_nodes = []
        strands = []
        for n1 in dict_motifs[motif]:
            try:
                df_tomtom = pd.read_csv(os.path.join(tomtom_path, f"{n1}/tomtom.tsv"), sep="\t", header=0)
                df_tomtom.dropna(inplace=True)
                df_tomtom['target_node'] = ['_'.join(x.split('_')[1:3]) for x in list(df_tomtom['Target_ID'])]
                df_tomtom = df_tomtom[df_tomtom['target_node']!=n1]

                df_tomtom.sort_values(by='E-value', inplace=True, ascending=True)
                top_node = df_tomtom.iloc[0]['target_node']
                top_eval = min(-np.log10(df_tomtom.iloc[0]['E-value']), 100)
                # get max jaccard df value for the n1 column
                max_jaccard = jaccard_df[n1].max()
                max_jaccards.append(max_jaccard)
                max_evals.append(top_eval)
                eval_nodes.append(top_node)
                strands.append(df_tomtom.iloc[0]['Orientation'])
            except Exception as e:
                print(f"Error processing motif {motif} for node {n1}: {e}")
                max_jaccards.append(0.0)
                max_evals.append(0.0)
                eval_nodes.append(None)
                strands.append(None)

        # make a new df
        df = pd.DataFrame({
            'node': dict_motifs[motif],
            'max_jaccard': max_jaccards,
            'max_eval': max_evals,
            'similar_node': eval_nodes,
            'strand': strands
        })
        df.to_csv(os.path.join(model_path, f"jaccard/{motif}_self_similarity.csv"), index=False)
