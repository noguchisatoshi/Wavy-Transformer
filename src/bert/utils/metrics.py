import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

def compute_cosine_similarity(hidden_states: torch.Tensor) -> (float, float):
    B, S, D = hidden_states.shape
    normed = F.normalize(hidden_states, p=2, dim=-1)  # (B, S, D)
    sim_matrices = torch.bmm(normed, normed.transpose(1, 2))
    
    idx = torch.arange(S, device=sim_matrices.device)
    sim_matrices[:, idx, idx] = 0.0
    flattened = sim_matrices.flatten()

    denom = B * S * (S - 1)

    mean_similarity = flattened.sum().item() / denom
    variance_similarity = sim_matrices.view(B, -1).mean(dim=-1).var().item()

    return mean_similarity, variance_similarity

def measure_over_smoothing(all_hidden_states):
    """
    all_hidden_states: List[Tensor], hidden states at each layer (B, S, H)
    
    Returns:
        layerwise_means: List[float]
        layerwise_variances: List[float]
    """
    layerwise_means = []
    layerwise_variances = []
    for hs in all_hidden_states:
        mean_sim, var_sim = compute_cosine_similarity(hs)
        layerwise_means.append(mean_sim)
        layerwise_variances.append(var_sim)
    return layerwise_means, layerwise_variances

def save_over_smoothing_image(layerwise_means, layerwise_variances, exp_dir, image_name=None):
    """
    layerwise_means: List[float]
    layerwise_variances: List[float]
    """
    layers = range(1, len(layerwise_means) + 1)
    means = torch.tensor(layerwise_means)
    variances = torch.tensor(layerwise_variances)
    stds = torch.sqrt(variances) 
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(layers, layerwise_means, marker='o', color='blue', label='Mean Cosine Similarity')
    
    plt.fill_between(layers, 
                     layerwise_means - stds.numpy(), 
                     layerwise_means + stds.numpy(), 
                     color='blue', alpha=0.2, label='±1 Std Dev')
    
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title('Layer-wise Cosine Similarity with Variance')
    plt.grid(True)
    plt.ylim(0, 1) 
    plt.legend()
    
    if image_name is None:
        save_path = os.path.join(exp_dir, "over_smoothing_with_variance.png")
    else:
        save_path = os.path.join(exp_dir, f"{image_name}_over_smoothing_with_variance.png")
    plt.savefig(save_path)
    plt.close()

def compute_dirichlet_energy(hidden_states_layers):
    """
    Compute the Dirichlet energy (intra-layer energy) per layer.

    Args:
        hidden_states_layers: (B, L, S, D) tensor of hidden states.

    For each layer i, the Dirichlet energy is computed as:
        E(i) = sum_{j,k} ||h_j - h_k||^2
    where h_j and h_k are the token embeddings in that layer.

    Returns:
        Tensor of shape (L,) with the average Dirichlet energy per layer (averaged over the batch).
    """
    B, L, S, D = hidden_states_layers.shape
    intra_energy_list = []
    for layer in range(L):
        if layer > 0:
            hs = hidden_states_layers[:, layer]         # (B, S, D)     # (B, S, S)
            diff = hs.unsqueeze(2) - hs.unsqueeze(1)       # (B, S, S, D)
            diff_sq = (diff ** 2).sum(dim=-1)              # (B, S, S)
            energy = (diff_sq).sum(dim=-1)               # (B, S)
            energy_sum = energy.sum(dim=-1) / (S * S)          # (B,)
            intra_energy_list.append(energy_sum)
    intra_energy = torch.stack(intra_energy_list, dim=1)  # (B, L)

    avg_energy_per_layer = intra_energy.mean(dim=0)

    return avg_energy_per_layer

def compute_avg_energy(hidden_states_layers, intra_attn_weights, tau=1.0):
    """
    hidden_states_layers: (B, L, S, D) tensor with hidden states.
    intra_attn_weights: (B, L, S, S) tensor with softmax-normalized attention weights.
    
    For each sample and for each layer i (0 <= i < L-1):
      - Compute intra-layer energy:
          E_intra(i) = sum_{j,k (j != k)} a^{(i)}_{jk} * ||h_j - h_k||^2
      - Compute inter-layer energy:
          E_inter(i) = sum_{j} ||h^{(i)}_j - h^{(i+1)}_j||^2
      - Total energy for layer i: E_total(i) = E_intra(i) + E_inter(i)
    
    Returns:
      A tensor of shape (L-1,) with the average energy per layer (averaged over the batch).
    """
    if intra_attn_weights is None:
        raise ValueError("intra_attn_weights must be provided.")
    
    B, L, S, D = hidden_states_layers.shape

    # Compute intra-layer energy for each layer per sample
    intra_energy_list = []
    for layer in range(L - 1):
        hs = hidden_states_layers[:, layer]         # (B, S, D)
        attn = intra_attn_weights[:, layer]          # (B, S, S)
        diff = hs.unsqueeze(2) - hs.unsqueeze(1)       # (B, S, S, D)
        diff_sq = (diff ** 2).sum(dim=-1)              # (B, S, S)
        energy = (attn * diff_sq).sum(dim=-1)           # (B, S)
        energy_sum = energy.sum(dim=-1)                # (B,)
        intra_energy_list.append(energy_sum)
    intra_energy = torch.stack(intra_energy_list, dim=1)  # (B, L)

    # Compute inter-layer energy for adjacent layers per sample
    inter_energy_list = []
    for layer in range(L - 1):
        hs_current = hidden_states_layers[:, layer]   # (B, S, D)
        hs_next = hidden_states_layers[:, layer + 1]    # (B, S, D)
        diff = ( hs_current - hs_next ) / tau                    # (B, S, D)
        diff_sq = (diff ** 2).sum(dim=-1)               # (B, S)
        energy_sum = diff_sq.sum(dim=-1)                # (B,)
        inter_energy_list.append(energy_sum)
    inter_energy = torch.stack(inter_energy_list, dim=1) if L > 1 else None  # (B, L-1)

    # Sum intra and inter energies for layers 0 to L-2 per sample
    total_energy_list = []
    for layer in range(L - 1):
        energy_sample = intra_energy[:, layer] + inter_energy[:, layer]  # (B,)
        total_energy_list.append(energy_sample)
    total_energy = torch.stack(total_energy_list, dim=1)  # (B, L-1)

    # Average energy per layer over the batch
    avg_energy_per_layer = total_energy.mean(dim=0)  # (L-1,)

    return avg_energy_per_layer

def save_energy_transition_image(avg_energy_per_layer, exp_dir, image_name=None):
    """
    avg_energy_per_layer: (L-1,) tensor with average energy per layer.
    exp_dir: Directory to save the image.
    image_name: Optional base name for the image file.
    """
    # Layers are numbered from 1 to L-1
    layers = range(1, avg_energy_per_layer.size(0) + 1)
    energy_values = avg_energy_per_layer.cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    # Plot average energy per layer
    plt.plot(layers, energy_values, marker='o', color='blue', label='Average Energy')
    
    plt.xlabel('Layer')
    plt.ylabel('Energy')
    plt.title('Layer-wise Average Energy Transition')
    plt.grid(True)
    plt.legend()
    
    if image_name is None:
        save_path = os.path.join(exp_dir, "energy_transition.png")
    else:
        save_path = os.path.join(exp_dir, f"{image_name}_energy_transition.png")
    
    plt.savefig(save_path)
    plt.close()