import os
import json

import numpy as np
import subprocess
import gc

from tqdm import tqdm
import pandas as pd
import json

from Bio import SeqIO


genome_fasta = '/home/anya/workspace/genome/hg38.fa'
# dictionary to store hg19 sequences
seq_dict = {}

with open('/home/anya/workspace/genome/hg38.fa', mode="r") as handle:
    # process each record in .fa file if there's more than one
    for record in SeqIO.parse(handle, "fasta"):
        identifier = record.id
        sequence = record.seq
        seq_dict[identifier] = str(sequence)


def make_acts(model_path, list_files, receptive_field):
    for file in list_files:
        if file.startswith('strongest_activations_test'):
            acts_i = pd.read_csv(os.path.join(model_path, file), index_col=0)
            acts_i = acts_i[acts_i['activation']!=0]
            if 'acts' not in locals():
                acts = acts_i
            else:
                acts = pd.concat([acts, acts_i])
            print(len(acts))

            # pad to receptive_field
            acts['center'] = (acts['end'] + acts['start'])//2
            acts['start_pad'] = acts['center'] - receptive_field//2
            acts['end_pad'] = acts['center'] + receptive_field//2
    gc.collect()
    return acts


def make_node_stats(acts, model_path):
    num_nonzero = []
    mean_act = []
    std_act = []
    median_act = []
    max_act = []
    acts_above_global_mean = []
    acts_above_node_mean = []
    acts_above_node_half = []
    
    mean_activation = np.mean(acts['activation'])
    median_activation = np.median(acts['activation'])
            
    for node in tqdm(acts['node_id'].unique()):
        acts_node = acts[acts['node_id']==node]
        num_nonzero.append(len(acts_node))
        mean_act.append(acts_node['activation'].mean())
        std_act.append(acts_node['activation'].std())
        median_act.append(acts_node['activation'].median())
        max_act.append(acts_node['activation'].max())
        # find number of seqs with activation above global mean
        seqs_above_global_mean = len(np.where(acts_node['activation']>mean_activation)[0])
        # find number of seqs with activation above mean of this node
        seqs_above_node_mean = len(np.where(acts_node['activation']>acts_node['activation'].mean())[0])
        seqs_above_node_half = len(np.where(acts_node['activation']>0.5*acts_node['activation'].max())[0])
        
        acts_above_node_half.append(seqs_above_node_half)
        acts_above_global_mean.append(seqs_above_global_mean)
        acts_above_node_mean.append(seqs_above_node_mean)


    df_act_stats = pd.DataFrame({'node_id': acts['node_id'].unique(), 
                                 'num_nonzero': num_nonzero, 'mean_act': mean_act, 
                                 'max_act': max_act,
                                 'std_act': std_act, 'median_act': median_act,
                                 'acts_above_global_mean': acts_above_global_mean,
                                 'acts_above_node_mean': acts_above_node_mean,
                                 'acts_above_node_half': acts_above_node_half})

    df_act_stats.sort_values('mean_act', ascending=False, inplace=True)
    df_act_stats.to_csv(os.path.join(model_path, 'node_stats.csv'))
    return df_act_stats

config_file = "config/config.json"

with open(config_file) as configs_open:
    configs = json.load(configs_open)

layer_name = configs['layer_name']

if layer_name=='conv1d_1':
    receptive_field = 14
if layer_name=='conv1d_2':
    receptive_field = 18
if layer_name=='conv1d_3':
    receptive_field = 26
if layer_name=='conv1d_4':
    receptive_field = 34

exp_factors = [8]
topks = [0.05, 0.1]
lrs = [0.0001, 1e-05]

for exp_factor in exp_factors:
    for topk in topks:
        for lr in lrs:

            global_model_path = configs['model_save_path']
            # conv1d_3_noabs_4_topk0.05_lr1e-05
            model_save_path = os.path.join(global_model_path, f"{configs['layer_name']}_noabs_{exp_factor}_topk{topk}_lr{lr}")

            list_files = os.listdir(model_save_path)
            acts = make_acts(model_save_path, list_files, receptive_field)

            df_act_stats = make_node_stats(acts, model_save_path)

            nodes_to_analyze = df_act_stats[df_act_stats['num_nonzero']>1000]['node_id'].values #(df_act_stats['acts_above_node_half']>200) & (

            N_sample = 1000

            node_seq_path = os.path.join(model_save_path, f'node_seqs_test_{N_sample}')
            if not os.path.exists(node_seq_path):
                os.makedirs(node_seq_path)


            # for each node, write the sequence to a file
            for ind,node in tqdm(enumerate(nodes_to_analyze)):
                acts_node = acts[acts['node_id']==node]
                acts_node = acts_node[acts_node['activation']!=0]
                acts_node.sort_values(by='activation', inplace=True, ascending=False)
                # sample top N_sample seqs
                acts_node_ = acts_node[:N_sample]
                with open(os.path.join(node_seq_path, f'{ind+1}_n{node}.fa'), 'w') as f:
                    for i, row in acts_node_.iterrows():
                        chrom = row['chrom']
                        start = row['start_pad']
                        end = row['end_pad']
                        seq = seq_dict[chrom][start:end].upper()
                        if len(seq)!=receptive_field:
                            continue
                        f.write(f'>{chrom}:{start}-{end}\n')
                        f.write(f"{seq}\n")


