import shutil
import subprocess
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import json


proc = subprocess.Popen(["which", "intersectBed"], stdout=subprocess.PIPE)
out = proc.stdout.read().decode("utf-8")
bedtools_exec = "/".join(out.strip("\n").split("/")[:-1])
print("bedtools executable path to be used:", bedtools_exec)

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

def compute_jaccard_index(intersection, union):
    jaccard_index = intersection / union if union != 0 else 0

    return jaccard_index

if not os.path.exists(os.path.join(model_path, 'jaccard')):
    os.makedirs(os.path.join(model_path, 'jaccard'))
    
cwd = os.getcwd()
temp_folder = os.path.join(cwd, "temp")

# load json
with open(f"layer{configs['layer_name'][-1]}_motif_nodes.json", "r") as f:
    dict_motifs = json.load(f)

for motif in tqdm(dict_motifs.keys()):
    if len(dict_motifs[motif])>1:
        dict_int = {}
        for n1 in dict_motifs[motif]:
            dict_int[f"{n1}"] = []
            for n2 in dict_motifs[motif]:
                # Copy the files to the temporary directory
                node1_bed = os.path.join(temp_folder, f"{n1}.bed")
                node2_bed = os.path.join(temp_folder, f"{n2}.bed")

                if n1 == n2:
                    dict_int[f"{n1}"].append(1.0)
                    continue

                # Create output file for the intersectn2ion
                intersect_bed = os.path.join(temp_folder, f"{n1}_{n2}_intersect.bed")
                    
                # Run intersectBed to find overlaps
                with open(intersect_bed, "w") as f:
                    subprocess.call(
                        [
                            f"{bedtools_exec}/intersectBed",
                            "-a",
                            node1_bed,
                            "-b",
                            node2_bed,
                            "-wa",
                        ],
                        stdout=f,
                    )

                # Count unique sequences from node1 that overlap with sequences from node2
                try:
                    df_int = pd.read_csv(intersect_bed, sep='\t', index_col=None, header=None)
                    intersection = len(df_int[3].unique())
                except pd.errors.EmptyDataError:
                    print(f"Empty intersection file for {n1} and {n2}")
                    intersection = 0
                
                try:
                    df_1 = pd.read_csv(node1_bed, sep='\t', index_col=None, header=None)
                    tot1 = len(df_1[3].unique())
                except pd.errors.EmptyDataError:
                    tot1 = 0

                try:
                    df_2 = pd.read_csv(node2_bed, sep='\t', index_col=None, header=None)
                    tot2 = len(df_2[3].unique())
                except pd.errors.EmptyDataError:
                    tot2 = 0
                    
                union = tot1 + tot2 - intersection
                jaccard_index = compute_jaccard_index(intersection, union)
                dict_int[f"{n1}"].append(jaccard_index)

                # Clean up the temporary intersection file
                #os.remove(intersect_bed)

        # Save the Jaccard index results to a CSV file
        jaccard_df = pd.DataFrame(dict_int)
        jaccard_df.to_csv(os.path.join(model_path, f"jaccard/{motif}_jaccard.csv"), index=False)

