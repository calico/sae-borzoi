import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap as mp
#import umap.umap_ as umap
import os
import h5py

from tqdm import tqdm
import subprocess
import json


def make_prot_dict(prot_file):
    prot = pd.read_csv(prot_file, sep='\t', index_col=0)

    names = []
    fams = []

    for i,row in prot.iterrows():
        try:
            genes = row['Gene Names'].split(' ')
            if len(genes)>1:
                names.extend(genes)
                fams.extend([prot['Protein families']]*len(genes))
            else:
                names.extend(genes)
                fams.extend(prot['Protein families'])
        except AttributeError:
            print(i)

    dict_prot = dict(zip(names, fams))

    return dict_prot

def flatten_values(d):
    f = []
    for k in d.keys():
        f.extend(d[k])
    return f

def process_meme_results(model_folder, configs):

    cwd = os.getcwd()

    dict_motifs = json.load(open(configs['cisbp_1'], 'r'))
    dict_motifs1 = json.load(open(configs['cisbp_2'], 'r'))

    dict_motifs.update(dict_motifs1)

    list_motifs = list(set(dict_motifs.keys()))
    num_motifs_all = len(list_motifs)

    models = []
    pct_motifs_discovered = []
    pct_motifs_discovered_1node = []
    pct_motifs_discovered_uniq = []

    dict_model = {}
    dict_widths = {}
    all_motifs = []

    dict_meme = {}
    dict_tomtom = {}

    node_folders_clean = []
    top_genes = []
    motif_consensus = []

    result_path = os.path.join(model_folder, "meme_analysis_results")

    timestamp_results = os.listdir(result_path)
    # pick the latest timestamp
    timestamp_results = sorted(timestamp_results, reverse=True)

    timestamp_results = timestamp_results[0]
    model_path = os.path.join(result_path, timestamp_results)

    models.append(model_folder)
    node_ids = os.path.join(model_path, 'meme_test_out')
    node_folders = os.listdir(os.path.join(model_folder, 'node_seqs_test_1000'))
    node_folders = [x for x in node_folders if 'tomtom_all' not in x]
    node_folders = [x for x in node_folders if '.DS_' not in x]
    node_folders = [x.split('.')[0] for x in node_folders]
    
    for ind,node_id in tqdm(enumerate(node_folders)):
        
        if 'tomtom_all' in node_id:
            continue
        if '.DS_' in node_id:
            continue
            
        node_folders_clean.append(node_id)

        meme_file = os.path.join(model_path, 'meme_test_out', node_id, 'motifs.tsv')
        try:
            df_meme = pd.read_csv(meme_file, sep='\t')
            df_meme.sort_values(by='sites', ascending=False, inplace=True)
            if len(df_meme) == 0:
                motif_consensus.append(None)
            else:
                motif_consensus.append(str(df_meme.iloc[0]['sequence']))
        except Exception as e:
            print(f"Error reading meme file for node {node_id}: {e}")
            motif_consensus.append(None)
            
        if os.path.exists(os.path.join(model_path, 'tomtom_test_archetype_rna_out', node_id, 'tomtom.tsv')) == False:
            print(f"Motif file for node {node_id} does not exist.")

        output_file = os.path.join(model_path, 'tomtom_test_archetype_rna_out', node_id, 'tomtom.tsv')
        try:
            df = pd.read_csv(output_file, sep='\t')
            df = df[df['Query_ID']==motif_consensus[-1]]
            if len(df) == 0:
                top_genes.append(None)
            else:
                df.sort_values(by='E-value', ascending=True, inplace=True)
                top_match = str(df.iloc[0]['Target_ID'])
                top_genes.append(top_match)
        except Exception as e:
            print(f"Error reading tomtom file for node {node_id}: {e}")
            top_genes.append(None)

    df_genes = pd.DataFrame({'node': node_folders_clean, 'gene': top_genes, 'PWM': motif_consensus})
    dict_genes = {k:v for k,v in zip(node_folders_clean, top_genes)}
    dict_seqs = {k:v for k,v in zip(node_folders_clean, motif_consensus)}

    return df_genes, dict_genes, dict_seqs
    

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

X_new = []
with h5py.File(os.path.join(model_path, 'dict_top_acts_test.h5'), "r") as f:
    print(f.keys())
    keys = list(f.keys())
    # get first object name/key; may or may NOT be a group
    print(np.shape(f[keys[0]]))

    # only keep rows with non-zero values
    dist_sum = np.sum(f[keys[0]][()], axis=1)
    
    for key in keys:
        X_new.append(f[key][()])
    
X_new = X_new[0]
X_new = np.concatenate(X_new, axis=0)

fasta = os.listdir(os.path.join(model_path, 'node_seqs_test_1000'))
sorted_node_ind = [int(x.split('_')[0]) for x in fasta]
init_node_ind = [int(x.split('_')[1][1:-3]) for x in fasta]
node_names = [x.split('.')[0] for x in fasta]

# make a df with the fasta file names and the node names
fasta_df = pd.DataFrame({'new_id': sorted_node_ind, 'init_id': init_node_ind, 'fasta': fasta, 'node_name': node_names})
fasta_df.sort_values(by='new_id', inplace=True)

# REINDEX
X_new_orig_ind = X_new[:, list(fasta_df['init_id'])]

threshold = np.mean(X_new_orig_ind)

pct_has_it = []
mean_values = []

num_seqs = X_new_orig_ind.shape[0]

if not os.path.exists(os.path.join(model_path, "node_act_dists_test")):
    os.makedirs(os.path.join(model_path, "node_act_dists_test"))

for inode in tqdm(range(len(fasta_df['node_name'].values))):
    has_it = len(np.where(X_new_orig_ind[:,inode]>threshold)[0])
    row = X_new_orig_ind[:,inode]
    nonzero = row[np.where(row>0)[0]]
    
    np.save((os.path.join(model_path, "node_act_dists_test", fasta_df['node_name'].values[inode]+'.npy')), nonzero)
    
    mean_values.append(np.mean(nonzero))
    pct_has_it.append(has_it*100/num_seqs)
    

df_pct = pd.DataFrame({'node': fasta_df['node_name'].values, 'pct': pct_has_it, 'mean': mean_values})
print(np.shape(df_pct))
df_pct.to_csv((os.path.join(model_path, "pct_seqlets_has_acts_test.csv")))

# Standardize the data
scaler = StandardScaler()

# K-means clustering
n_clusters = 25  

# n_neighbors=10, min_dist=0.05
reducer = mp.UMAP(random_state=42, min_dist=0.02, n_components=2)
embedding = reducer.fit_transform(X_new_orig_ind.T)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embedding)

dict_prot = make_prot_dict(configs['uniprot'])
df_genes, dict_genes, dict_seqs = process_meme_results(model_path, configs)

df_embed = pd.DataFrame({'UMAP1': embedding[:, 0], 'UMAP2': embedding[:, 1],
                        'Cluster': cluster_labels}, index=fasta_df['node_name'].values)

df_embed['Gene'] = [dict_genes[n] if n in dict_genes.keys() else 'nan' for n in df_embed.index]
df_embed['Sequence'] = [dict_seqs[n] if n in dict_seqs.keys() else 'None' for n in df_embed.index]

family = []

for i,row in df_embed.iterrows():
    gene = row['Gene']
    if gene is not None:
        if '_' in gene:
            gene = gene.split('_')[0]
        if '+' in gene:
            gene = gene.split('+')[0]
        
    if gene in dict_prot.keys() and gene is not None:
        family.append(dict_prot[gene])
    else:
        family.append(np.nan)

df_embed['Uniprot'] = family

cluster_descr_dict = {}

for cl in df_embed['Cluster'].unique():
    df1_ = df_embed[df_embed['Cluster']==cl]
    cluster_descr_dict[cl] = list(df1_['Uniprot'].dropna().value_counts()[:3].index)

for cl in cluster_descr_dict.keys():
    cluster_descr_dict[cl] = '; '.join(cluster_descr_dict[cl])

df_embed['Cluster Description'] = [cluster_descr_dict[c] for c in df_embed['Cluster']]
df_embed.to_csv(os.path.join(model_path, "umap_data.csv"))

genes_dict = {}
for g in df_embed['Gene'].unique():
    if g!='NaN' or g!=np.nan:
        genes_dict[g] = df_embed[df_embed['Gene'] == g].index.tolist()

with open(f"layer{configs['layer_name'][-1]}_motif_nodes.json", 'w') as f:
    json.dump(genes_dict, f)

genes_df = pd.DataFrame.from_dict(genes_dict, orient='index').reset_index()
genes = []
num_nodes = []
for g in genes_dict.keys():
    genes.append(g)
    num_nodes.append(len(genes_dict[g]))

genes_df = pd.DataFrame({'Gene': genes, 'Num_nodes': num_nodes})
genes_df = genes_df.sort_values(by='Num_nodes', ascending=True)

genes_df.to_csv(f"layer{configs['layer_name'][-1]}_motif_nodes_num.csv", index=False)