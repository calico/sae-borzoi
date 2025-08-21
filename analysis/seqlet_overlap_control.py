import shutil
import argparse
import subprocess
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import json


def create_bed_from_fasta(fasta_file, output_bed):
    """
    Convert FASTA file to BED format
    
    Parameters:
    fasta_file (str): Path to FASTA file
    output_bed (str): Path to output BED file
    
    Returns:
    int: Number of sequences in the FASTA file
    """
    seq_count = 0
    with open(output_bed, 'w') as bed_out:
        with open(fasta_file, mode="r") as handle:
            # process each record in .fa file
            for record in SeqIO.parse(handle, "fasta"):
                identifier = record.id
                sequence = str(record.seq)
                
                # Parse the chromosome and coordinates from the identifier
                if ':' in identifier and '-' in identifier:
                    try:
                        parts = identifier.split(':')
                        chrom = parts[0]
                        start_end = parts[1].split('-')
                        start = int(start_end[0])
                        end = int(start_end[1])
                        
                        # Write to BED file (BED format is 0-based for start position)
                        bed_out.write(f"{chrom}\t{start}\t{end}\t{identifier}\t0\t+\n")
                        seq_count += 1
                    except ValueError:
                        print(f"Skipping record {identifier} in {fasta_file} due to parsing error: chr, start-end: {chrom} {start_end}")

        
    return seq_count


def parse_args():
    parser = argparse.ArgumentParser(description="Process sequence overlaps with BED files.")
    parser.add_argument('--no_bedtools_step', default=False, action='store_true', help='Do not perform the bedtools step.')
    parser.add_argument('--no_zscore', default=False, action='store_true', help='Do not perform the z-score normalization step.')
    return parser.parse_args()

args = parse_args()

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

umap = pd.read_csv(os.path.join(model_path, 'umap_data.csv'), header=0, index_col=0)

cwd = os.getcwd()
temp_folder = os.path.join(cwd, "temp")

if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

if not os.path.exists(os.path.join(model_path, 'node_overlaps')):
    os.makedirs(os.path.join(model_path, 'node_overlaps'))
    
out_dir = os.path.join(model_path, 'node_overlaps')

node_dir = list(umap.index)
node_dir = [f"{x}.fa" for x in node_dir]

seq_counts = {}

# Process each node and create BED files
if not args.no_bedtools_step:
    for node in tqdm(node_dir, desc="Creating BED files"):
        node_fasta = f"{model_path}/node_seqs_test_1000/{node}"
        node_name = node.split(".")[0]
        node_bed = os.path.join(temp_folder, f"{node_name}.bed")

        node_sh_bed = os.path.join(temp_folder, f"{node_name}_sh.bed")

        with open(node_sh_bed, "w") as f:
            subprocess.call(
                [
                    f"{bedtools_exec}/shuffleBed",
                    "-i",
                    node_bed,
                    "-g",
                    "/home/anya/workspace/genome/hg38.genome",
                ],
                stdout=f,
            )

        # Create BED file and get sequence count
        seq_count = create_bed_from_fasta(node_fasta, node_bed)
        seq_counts[node] = seq_count

    # Calculate pairwise overlaps
    total_pairs = len(node_dir)
    with tqdm(total=total_pairs, desc="Calculating cCRE overlaps") as pbar:
        for i, node1 in enumerate(node_dir):
            node1_name = node1.split(".")[0]
            node1_bed = os.path.join(temp_folder, f"{node1_name}_sh.bed")


            ccre = configs['ccre_bed']
                
            # Create output file for the intersection
            intersect_bed = os.path.join(temp_folder, f"{node1_name}_intersect_sh.bed")
                
            # Run intersectBed to find overlaps
            with open(intersect_bed, "w") as f:
                subprocess.call(
                    [
                        f"{bedtools_exec}/intersectBed",
                        "-a",
                        node1_bed,
                        "-b",
                        ccre,
                        "-wa",
                        "-wb",
                    ],
                    stdout=f,
                )

            # Count unique sequences from node1 that overlap with sequences from node2
            try:
                df_int = pd.read_csv(intersect_bed, sep='\t', index_col=None, header=None)
                df_int.to_csv(os.path.join(out_dir, f"{node1_name}_{seq_counts[node1]}_sh.csv"), index=None)

                # Clean up the temporary intersection file
                os.remove(intersect_bed)

            except Exception as e:
                print(f"Error processing intersection of {node1_name}: {e}")

            pbar.update(1)


    # Calculate pairwise overlaps
    total_pairs = len(node_dir)
    with tqdm(total=total_pairs, desc="Calculating RMSK overlaps") as pbar:
        for i, node1 in enumerate(node_dir):
            node1_name = node1.split(".")[0]
            node1_bed = os.path.join(temp_folder, f"{node1_name}_sh.bed")

            ccre = configs['rmsk_bed']
                
            # Create output file for the intersection
            intersect_bed = os.path.join(temp_folder, f"{node1_name}_intersect_sh.bed")
                
            # Run intersectBed to find overlaps
            with open(intersect_bed, "w") as f:
                subprocess.call(
                    [
                        f"{bedtools_exec}/intersectBed",
                        "-a",
                        node1_bed,
                        "-b",
                        ccre,
                        "-wa",
                        "-wb",
                    ],
                    stdout=f,
                )

            # Count unique sequences from node1 that overlap with sequences from node2
            try:
                df_int = pd.read_csv(intersect_bed, sep='\t', index_col=None, header=None)
                df_int.to_csv(os.path.join(out_dir, f"{node1_name}_TE_{seq_counts[node1]}_sh.csv"), index=None)

                # Clean up the temporary intersection file
                os.remove(intersect_bed)

            except Exception as e:
                print(f"Error processing intersection of {node1_name}: {e}")

            pbar.update(1)


ccres = pd.read_csv(configs['ccre_bed'], sep='\t', header=None)
#ccres = pd.read_csv('rmsk_hg38.bed', sep='\t', header=None)
elems = list(ccres[5].unique())

node_dir = list(umap.index)
node_dir = [f"{x}.fa" for x in node_dir]

pct_states = {}

for i, node1 in tqdm(enumerate(node_dir)):
    node1_name = node1.split(".")[0]
    pct_states[node1_name] = []

    try:
        df_int = pd.read_csv(os.path.join(out_dir, f"{node1_name}_1000_sh.csv"), index_col=None, header=0)
        for el in elems:
            df_int_ = df_int[df_int['11']==el]
            pct_states[node1_name].append(len(df_int_['3'].unique()))
    except FileNotFoundError:
        print(node1_name, "not found")
        for el in elems:
            pct_states[node1_name].append(0)

df_states = pd.DataFrame(pct_states)
df_states.index = elems
df_states = df_states.T

if not args.no_zscore:
    df_states = (df_states - df_states.mean(axis=0))/df_states.std(axis=0)
    df_states.fillna(0, inplace=True)
    df_states.to_csv(os.path.join(model_path, 'model_cCREs_intersect_sh.csv'))
else:
    df_states.fillna(0, inplace=True)
    df_states.to_csv(os.path.join(model_path, 'model_cCREs_intersect_raw_sh.csv'))
# TE overlap

pct_states = {}

rmsk = pd.read_csv(configs['rmsk_bed'], sep='\t', index_col=None, header=None)
elems = list(rmsk[5].unique())

for i, node1 in tqdm(enumerate(node_dir)):
    node1_name = node1.split(".")[0]
    pct_states[node1_name] = []

    try:
        df_int = pd.read_csv(os.path.join(out_dir, f"{node1_name}_TE_1000_sh.csv"), index_col=None, header=0)
        for el in elems:
            df_int_ = df_int[df_int['11']==el]
            pct_states[node1_name].append(len(df_int_['3'].unique()))
    except FileNotFoundError:
        print(node1_name, "not found")
        for el in elems:
            pct_states[node1_name].append(0)

df_rmsk = pd.DataFrame(pct_states)
df_rmsk.index = elems
df_rmsk = df_rmsk.T

if not args.no_zscore:
    df_rmsk.fillna(0, inplace=True)
    df_rmsk = (df_rmsk - df_rmsk.mean(axis=0))/df_rmsk.std(axis=0)
    df_rmsk.to_csv(os.path.join(model_path, 'model_rmsk_intersect_sh.csv'))
else:
    df_rmsk.fillna(0, inplace=True)
    df_rmsk.to_csv(os.path.join(model_path, 'model_rmsk_intersect_raw_sh.csv'))

int_nodes = []
for c in df_rmsk.columns:
    int_nodes.append([df_rmsk.sort_values(by=c, ascending=False).index[0].split('_')[0]])
    
df_int_nodes = pd.DataFrame(int_nodes, columns=['node'])
df_int_nodes = df_int_nodes.T
df_int_nodes.columns = df_rmsk.columns

df_int_nodes.to_csv(os.path.join(model_path, 'model_rmsk_intersect_top_nodes.csv'), index=None)