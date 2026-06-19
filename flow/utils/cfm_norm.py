import os
import torch
import numpy as np

def compute_norm_stats(dataset_dir, max_samples=100):

    """
    Validate cached 5D CFM dataset files and return fixed normalization metadata.
    """
    print(f"Scanning {dataset_dir} to compute normalization statistics...")
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No cached dataset files found in {dataset_dir}")

    files = sorted(files)[:min(max_samples, len(files))]
    has_seed_conditioned_labels = False
    for f in files:
        data = torch.load(os.path.join(dataset_dir, f), map_location='cpu')
        has_seed_conditioned_labels = has_seed_conditioned_labels or (
            'seed_grasp_rot_lie_list' in data and 'seed_valid_mask' in data
        )

    if not has_seed_conditioned_labels:
        raise ValueError("No seed-conditioned CFM labels found in cached dataset files.")

    return {'target_dim': 5}

def normalize_x(x, stats=None):
    """
    Normalize 5D grasp target x = [omega(3), width(1), depth(1)].

    Args:
        x (torch.Tensor): Grasp targets with shape [..., 5].
        stats (dict | None): Optional normalization metadata.

    Returns:
        torch.Tensor: Normalized grasp targets with shape [..., 5].
    """
    assert x.shape[-1] == 5, f"normalize_x: expected last dimension 5, got {x.shape[-1]}"
    
    omega = x[..., :3]
    w = x[..., 3:4]
    d = x[..., 4:5]

    # Normalize rotation Lie algebra by pi.
    omega_norm = omega / np.pi
    
    # Normalize width by 0.1m.
    w_norm = w / 0.1
    
    # Normalize depth meters from [0.01, 0.04] to [0, 1].
    d_norm = (d - 0.01) / 0.03
    
    x_norm = torch.cat([omega_norm, w_norm, d_norm], dim=-1)
    return x_norm

def denormalize_x(x_norm, stats=None):
    """
    Denormalize 5D grasp target x = [omega(3), width(1), depth(1)].

    Args:
        x_norm (torch.Tensor): Normalized grasp targets with shape [..., 5].
        stats (dict | None): Optional normalization metadata.

    Returns:
        torch.Tensor: Denormalized grasp targets with shape [..., 5].
    """
    assert x_norm.shape[-1] == 5, f"denormalize_x: expected last dimension 5, got {x_norm.shape[-1]}"
    
    omega_norm = x_norm[..., :3]
    w_norm = x_norm[..., 3:4]
    d_norm = x_norm[..., 4:5]
    
    omega = omega_norm * np.pi
    w = w_norm * 0.1
    d = d_norm * 0.03 + 0.01
    
    x = torch.cat([omega, w, d], dim=-1)
    return x
