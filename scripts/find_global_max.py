import os
from core.dataset import *
import json
import numpy as np


config_file = "config/config.json"
with open(config_file) as config_open:
    configs = json.load(config_open)

# Find global maximum across all files
activation_pattern = configs["activations_path"]+"/*.h5"
global_max = find_global_max(activation_pattern, num_channels=configs["input_channels"])
print(f"Global maximum values: {global_max}")

# convert global_max dict to string
global_max = {str(k): float(v) for k, v in global_max.items()}

# save to json file
with open(configs["global_max_save_path"], 'w') as file_open:
    json.dump(global_max, file_open)