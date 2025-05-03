import os
import torch
import torch.nn as nn
from sae import *
from dataset import *
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--topk', help='topk', type=float, default=0.1)
parser.add_argument('--exp_factor', help='expansion_factor', type=int, default=16)
parser.add_argument('--lr', help='learning_rate', type=float, default=1e-5)

args = parser.parse_args()


config_file = "config.json"
with open(config_file) as config_open:
    configs = json.load(config_open)

# Example usage
input_dim = configs["input_channels"]  # Your activation dimension
hidden_dim = args.exp_factor * configs["input_channels"]  # Desired hidden layer dimension
print("args:", args.topk, args.exp_factor, args.lr)
k = int(args.topk * configs["input_channels"]) # Number of top activations to keep
learning_rate = args.lr
sparsity_target = configs["sparsity_target"]

model_save_path = f"models/{configs['layer_name']}_noabs_{args.exp_factor}_topk{args.topk}_lr{args.lr}"

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

#load global max if it exists
if os.path.exists(configs["global_max_save_path"]):
    with open(configs["global_max_save_path"]) as file_open:
        global_max = json.load(file_open)
    print("Using global max from file:", global_max)
else:
    global_max = None

model = train_sparse_autoencoder(
    train_dir=configs["activations_path"],
    val_dir=configs["activations_path_val"],
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    k=k,
    batch_size=2,
    num_epochs=100,
    learning_rate=learning_rate,
    sparsity_factor=10.0,
    sparsity_target=sparsity_target,
    patience=2,
    checkpoint_dir=model_save_path,
    global_max=global_max
)