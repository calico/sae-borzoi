import torch
from torch.utils.data import Dataset
import h5py
import glob
import numpy as np
from tqdm import tqdm

def find_global_max(activation_path_pattern, num_channels=608):
    """
    Find the global maximum value across all HDF5 files matching the pattern.
    Processes files one at a time to minimize memory usage.
    
    Args:
        activation_path_pattern (str): Glob pattern for activation files
            e.g., "v2_activations/conv1d_1/activations_*.h5"
            
    Returns:
        float: Global maximum value across all files
    """
    file_paths = sorted(glob.glob(activation_path_pattern))
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {activation_path_pattern}")
    
    # initialize as dict for all channels
    global_max = dict(zip(range(num_channels), [float('-inf')]*num_channels))
    
    # Process each file individually
    for file_path in tqdm(file_paths, desc="Processing files"):
        with h5py.File(file_path, 'r') as f:
            first_key = list(f.keys())[0]
            # Load the dataset but don't read it into memory yet
            dataset = f[first_key][()]
            maxes = np.max(dataset, axis=(0,1))

            for i in range(num_channels):
                channel_max = maxes[i]
                global_max[i] = max(global_max[i], channel_max)
    
    return global_max

class ActivationDataset(Dataset):
    def __init__(self, activation_path_pattern, transform=None):
        """
        Dataset for loading activation data from HDF5 files.
        
        Args:
            activation_path_pattern (str): Glob pattern for activation files
                e.g., "v2_activations/conv1d_1/activations_*.h5"
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.transform = transform
        
        # Get list of all activation files
        self.file_paths = sorted(glob.glob(activation_path_pattern))
        if not self.file_paths:
            raise ValueError(f"No files found matching pattern: {activation_path_pattern}")
            
        # Get total number of samples and shape
        with h5py.File(self.file_paths[0], 'r') as f:
            first_key = list(f.keys())[0]  # Get the first dataset key
            chunk_shape = f[first_key].shape
            self.activation_shape = chunk_shape[1:]  # Shape of single activation
            
        self.chunk_size = chunk_shape[0]  # Number of samples per file
        self.seq_divisor = 4 # divide the sequence length by this number to reduce memory usage
        self.total_samples = len(self.file_paths) * self.chunk_size * self.seq_divisor
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Get a single activation sample.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            torch.Tensor: Activation tensor
        """
        # Calculate which file and which sample within the file
        file_idx = idx // (self.chunk_size * self.seq_divisor)
        sample_idx = (idx % (self.chunk_size * self.seq_divisor)) // self.seq_divisor
        seq_idx = (idx % (self.chunk_size * self.seq_divisor)) % self.seq_divisor
        
        # Load the appropriate chunk
        with h5py.File(self.file_paths[file_idx], 'r') as f:
            first_key = list(f.keys())[0]  # Get the first dataset key
            seq_len = f[first_key].shape[1]
            activation = f[first_key][sample_idx][seq_idx*seq_len//self.seq_divisor:(seq_idx+1)*seq_len//self.seq_divisor]
            
        # Convert to tensor
        activation = torch.tensor(activation, dtype=torch.float32)
        
        if self.transform:
            activation = self.transform(activation)
            
        return activation


class ActivationDatasetInfer(Dataset):
    def __init__(self, activation_path_pattern, transform=None, resolution=8, pad=163840, top_chunk_pct=0.25):
        """
        Dataset for loading activation data from HDF5 files.
        
        Args: 
            activation_path_pattern (str): Glob pattern for activation files
                e.g., "v2_activations/conv1d_1/activations_*.h5"
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.transform = transform
        
        # Get list of all activation files
        self.file_paths = sorted(glob.glob(activation_path_pattern))
        if not self.file_paths:
            raise ValueError(f"No files found matching pattern: {activation_path_pattern}")
            
        # Get total number of samples and shape
        with h5py.File(self.file_paths[0], 'r') as f:
            first_key = list(f.keys())[0]  # Get the first dataset key
            chunk_shape = f[first_key].shape
            self.activation_shape = chunk_shape[1:]  # Shape of single activation
            
        self.chunk_size = chunk_shape[0]  # Number of samples per file
        self.seq_divisor = 4 # divide the sequence length by this number to reduce memory usage
        self.total_samples = len(self.file_paths) * self.chunk_size * self.seq_divisor
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Get a single activation sample.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            torch.Tensor: Activation tensor
        """
        # Calculate which file and which sample within the file
        file_idx = idx // (self.chunk_size * self.seq_divisor)
        sample_idx = (idx % (self.chunk_size * self.seq_divisor)) // self.seq_divisor
        seq_idx = (idx % (self.chunk_size * self.seq_divisor)) % self.seq_divisor
        
        # Load the appropriate chunk
        with h5py.File(self.file_paths[file_idx], 'r') as f:
            first_key = list(f.keys())[0]  # Get the first dataset key
            seq_len = f[first_key].shape[1]
            activation = f[first_key][sample_idx][seq_idx*seq_len//self.seq_divisor:(seq_idx+1)*seq_len//self.seq_divisor]
            
        # Convert to tensor
        activation = torch.tensor(activation, dtype=torch.float32)
        
        if self.transform:
            activation = self.transform(activation)
            
        return activation
    

# Example transform for preprocessing activations
class NormalizeActivations:
    def __init__(self, global_max=None):
        self.global_max = global_max

    def __call__(self, x):
        """
        Normalize activation values.
        """
        if self.global_max is not None:
            global_max_flat = list(self.global_max.values())
            # divide along last dimension
            return x / torch.tensor(global_max_flat, dtype=torch.float32)
        return (x - x.mean()) / (x.std() + 1e-5)