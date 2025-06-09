from borzoi_pytorch import Borzoi
import torch
from sae import *
import zarr
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import torch.nn.functional as F
import copy


def hook_sparse_autoencoder(
    borzoi_model,
    checkpoint_path: str,
    layer_name,
    zarr_seqs,
    zarr_file,
    input_dim: int,
    hidden_dim: int,
    k: int,
    sparsity_method: str = "topk_o",
    transform=None,
    resolution=8,
    pad=163840,
    top_chunk_num=16,
):
    seq_depth = 4
    dict_layers = {"conv1d_2": 2, "conv1d_3": 4, "conv1d_4": 6}

    # Initialize model, optimizer, and loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_model = SparseAutoencoder(
        input_dim, hidden_dim, k, sparsity_method=sparsity_method
    ).to(device)
    sae_model.load_pretrained(str(Path(checkpoint_path) / "best_model.pt"))
    sae_model.eval()

    borzoi_model = borzoi_model.to(device)
    
    # Create hook handlers for both original and reconstructed forward passes
    class ActivationHook:
        def __init__(self):
            self.activation = None
            
        def __call__(self, module, input, output):
            self.activation = output.detach()
    
    # Create instances of the hook handlers
    activation_hook = ActivationHook()
    
    # Register the hook for the original model
    target_layer = borzoi_model.res_tower[dict_layers[layer_name]].conv_layer
    hook_handle = target_layer.register_forward_hook(activation_hook)

    # Store results for correlation metrics
    pearson_r_values = []
    r_squared_values = []
    
    pearson_r_values_o = []
    r_squared_values_o = []
    
    pearson_r_values_s = []
    r_squared_values_s = []
    
    # Helper function to process activations through SAE in chunks
    def process_activation_through_sae(activation, chunk_size=32768):
        # Original shape: [1, 736, 131072]
        # First, we need to transpose to [1, 131072, 736]
        activation_t = activation.transpose(1, 2)
        
        # Get the sequence length
        seq_len = activation_t.size(1)
        
        # Create a tensor to store the reconstructed activations
        reconstructed_t = torch.zeros_like(activation_t)
        
        # Process in chunks
        for start_idx in range(0, seq_len, chunk_size):
            end_idx = min(start_idx + chunk_size, seq_len)
            
            # Extract chunk and reshape to [chunk_size, 736]
            chunk = activation_t[:, start_idx:end_idx, :] #.squeeze(0)
            
            # Process through SAE
            with torch.no_grad():
                # Encode then decode (reconstruct)
                _, x_recon = sae_model.infer(chunk.to(device))
            
            # Store back in the reconstructed tensor
            reconstructed_t[:, start_idx:end_idx, :] = x_recon #.unsqueeze(0)
        
        # Transpose back to original shape [1, 736, 131072]
        reconstructed = reconstructed_t.transpose(1, 2)
        
        return reconstructed
    
    # Function to compute Pearson R and R² between original and reconstructed outputs
    def compute_correlation_metrics(original_output, reconstructed_output, labels=False):
        # Shape: [1, 7611, 6144]
        if not labels:
            original = original_output.squeeze(0).cpu().numpy()  # [7611, 6144]
        else:
            original = original_output
        reconstructed = reconstructed_output.squeeze(0).cpu().numpy()  # [7611, 6144]
        
        pearson_r = []
        r_squared = []

        # Compute metrics for each of the 7611 rows
        for i in range(original.shape[0]):
            r, p_value = pearsonr(original[i], reconstructed[i])
            pearson_r.append(r)
            r_squared.append(r**2)

        return np.array(pearson_r), np.array(r_squared)

    try:
        with zarr.open(zarr_file, "r") as zarr_data:
            # Process each sample
            max_samples = len(zarr_data["sequence"])
            for i in range(max_samples):
                seq_index = zarr_data["sequence"][i]
                target = zarr_data["target"][i]

                seq_1hot = np.zeros((seq_depth, len(seq_index)), dtype="bool")
                seq_1hot[seq_index, np.arange(len(seq_index))] = 1

                # Pass seq_1hot through the model for original output
                seq_1hot_tensor = torch.tensor(seq_1hot, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get original output
                with torch.no_grad():
                    original_output = borzoi_model(seq_1hot_tensor)
                
                # Get the activations from the hook
                if activation_hook.activation is not None:
                    # Get original activations
                    original_act = activation_hook.activation
                    
                    #print(f"Original activation shape: {original_act.shape}")
                    
                    # Process through SAE in chunks
                    reconstructed_act = process_activation_through_sae(original_act)
                    
                    #print(f"Reconstructed activation shape: {reconstructed_act.shape}")
                    #
                    # Now we need to perform a forward pass with the reconstructed activations
                    # Create a copy of the model for reconstructed forward pass
                    borzoi_copy = copy.deepcopy(borzoi_model)
                    
                    # Define a hook to replace the activations
                    class ReplacementHook:
                        def __init__(self, replacement_tensor):
                            self.replacement_tensor = replacement_tensor
                            
                        def __call__(self, module, input, output):
                            return self.replacement_tensor
                    
                    # Register the replacement hook
                    replace_hook = ReplacementHook(reconstructed_act)
                    replacement_handle = borzoi_copy.res_tower[dict_layers[layer_name]].conv_layer.register_forward_hook(replace_hook)
                    
                    # Get reconstructed output
                    with torch.no_grad():
                        reconstructed_output = borzoi_copy(seq_1hot_tensor)
                    
                    # Remove the replacement hook
                    replacement_handle.remove()
                    
                    #print(f"Original output shape: {original_output.shape}")
                    #print(f"Reconstructed output shape: {reconstructed_output.shape}")
                    #print(f"Target shape: {target.shape}")
                    
                    # Compute correlation metrics
                    pearson_r, r_squared = compute_correlation_metrics(original_output, reconstructed_output)
                    pearson_r_ori, r_squared_ori = compute_correlation_metrics(target, original_output, labels=True)
                    pearson_r_sae, r_squared_sae = compute_correlation_metrics(target, reconstructed_output, labels=True)

                    # Store the metrics
                    pearson_r_values.append(pearson_r)
                    r_squared_values.append(r_squared)

                    pearson_r_values_o.append(pearson_r_ori)
                    r_squared_values_o.append(r_squared_ori)
                    
                    pearson_r_values_s.append(pearson_r_sae)
                    r_squared_values_s.append(r_squared_sae)
                    # Print summary statistics
                    print(f"Sample {i} - Mean Pearson R: {np.nanmean(pearson_r):.4f}, Mean R²: {np.nanmean(r_squared):.4f}")
                    print(f"Sample {i} - Mean Pearson R, original vs targets: {np.nanmean(pearson_r_ori):.4f}, Mean R²: {np.nanmean(r_squared_ori):.4f}")
                    print(f"Sample {i} - Mean Pearson R, SAE vs targets: {np.nanmean(pearson_r_sae):.4f}, Mean R²: {np.nanmean(r_squared_sae):.4f}")
                    
                else:
                    print(f"No activations captured for sample {i}")
                
                # Optionally break after processing a few samples for testing
                if i >= 75:  
                    break
    finally:
        # Always remove the hook when done
        hook_handle.remove()
    
    return pearson_r_values, r_squared_values, pearson_r_values_o, r_squared_values_o, pearson_r_values_s, r_squared_values_s

# Example usage
zarr_file = "/scratch4/drk/seqnn/data/v9/hg38/examples/fold3.zarr"

zarr_seqs = pd.read_csv('/scratch4/drk/seqnn/data/v9/hg38/sequences.bed', sep='\t', header=None)
zarr_seqs.columns = ['chrom', 'start', 'end', 'fold']
zarr_seqs = zarr_seqs[zarr_seqs['fold']=='fold3']

borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0') # 'johahi/borzoi-replicate-[0-3][-mouse]'

pearson_r_results, r_squared_results, pearson_r_results_o, r_squared_results_o, pearson_r_results_s, r_squared_results_s = hook_sparse_autoencoder(
    borzoi,  # Assuming borzoi is defined elsewhere
    "/home/anya/code/sae_borzoi_3/models/conv1d_3_noabs_4_topk0.05_lr1e-05",
    "conv1d_3",
    zarr_seqs,  # Assuming zarr_seqs is defined elsewhere
    zarr_file,
    896,
    896 * 4,
    int(0.05 * 896),
)

# Analyze the results
if pearson_r_results:
    # Calculate overall statistics
    all_pearson_r = np.concatenate(pearson_r_results)
    all_r_squared = np.concatenate(r_squared_results)
    
    all_pearson_r_o = np.concatenate(pearson_r_results_o)
    all_r_squared_o = np.concatenate(r_squared_results_o)

    all_pearson_r_s = np.concatenate(pearson_r_results_s)
    all_r_squared_s = np.concatenate(r_squared_results_s)
    
    print("\nOverall Statistics:")
    print(f"Mean Pearson R: {np.nanmean(all_pearson_r):.4f}")
    print(f"Mean R²: {np.nanmean(all_r_squared):.4f}")

    print(f"Mean Pearson R, original vs targets: {np.nanmean(all_pearson_r_o):.4f}")
    print(f"Mean R², original vs targets: {np.nanmean(all_r_squared_o):.4f}")
    
    print(f"Mean Pearson R, SAE vs targets: {np.nanmean(all_pearson_r_s):.4f}")
    print(f"Mean R², SAE vs targets: {np.nanmean(all_r_squared_s):.4f}")
    # Maybe plot a histogram of the correlation values
    
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(all_pearson_r, bins=50)
        plt.title('Distribution of Pearson R - Borzoi vs. SAE')
        plt.xlabel('Pearson R')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(all_r_squared, bins=50)
        plt.title('Distribution of R²')
        plt.xlabel('R²')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('temp/ori_to_sae_hist.png')
        plt.gcf().clf()
        
        # original
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(all_pearson_r_o, bins=50)
        plt.title('Distribution of Pearson R - Borzoi vs. labels')
        plt.xlabel('Pearson R')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(all_r_squared_o, bins=50)
        plt.title('Distribution of R²')
        plt.xlabel('R²')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('temp/ori_to_targets_hist.png')
        plt.gcf().clf()
        
        # SAE
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(all_pearson_r_s, bins=50)
        plt.title('Distribution of Pearson R - SAE vs. labels')
        plt.xlabel('Pearson R')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(all_r_squared_s, bins=50)
        plt.title('Distribution of R²')
        plt.xlabel('R²')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('temp/sae_to_targets_hist.png')
        
    except ImportError:
        print("Matplotlib not available for plotting histograms")