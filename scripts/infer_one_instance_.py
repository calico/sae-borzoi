import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.sae import *
from core.dataset import *
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--topk', help='topk', type=float, default=0.1)
parser.add_argument('--exp_factor', help='expansion_factor', type=int, default=16)
parser.add_argument('--lr', help='learning_rate', type=float, default=1e-5)
parser.add_argument('--top_acts', help='top acts to save', type=int, default=16)

args = parser.parse_args()


config_file = "config/config.json"
with open(config_file) as config_open:
    configs = json.load(config_open)

activations_path = configs['activations_path']
train_seqs_path = f"{activations_path}/train_seqs.bed"

pad = (524288-196608)//2

#load global max if it exists
if os.path.exists(configs["global_max_save_path"]):
    with open(configs["global_max_save_path"]) as file_open:
        global_max = json.load(file_open)
    print("Using global max from file")
else:
    global_max = None

exp_factor = args.exp_factor
topk = args.topk
lr = args.lr

global_model_path = configs['model_save_path']
model_save_path = os.path.join(global_model_path, f"{configs['layer_name']}_noabs_{exp_factor}_topk{topk}_lr{lr}")

infer_sparse_autoencoder(
    model_save_path,
    activations_path,
    configs["input_channels"],
    configs["expansion_factor"]*configs["input_channels"],
    int(configs["topk_pct"]*configs["input_channels"]),
    transform=NormalizeActivations(global_max=global_max), 
    resolution=524288//configs["seq_len"], 
    pad=pad, 
    top_chunk_num=args.top_acts)