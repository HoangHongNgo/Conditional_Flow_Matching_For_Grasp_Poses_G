import os
import torch
import numpy as np

def compute_norm_stats(dataset_dir, max_samples=100):

    """
    Scan cached .pt files in dataset_dir and compute empirical std
    for translation p relative to the seed points median in each scene.
    """
    print(f"Scanning {dataset_dir} to compute normalization statistics...")
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No cached dataset files found in {dataset_dir}")
    
    # Limit number of scanned files to keep it fast
    files = sorted(files)[:min(max_samples, len(files))]
    
    all_points_centered = []
    for f in files:
        data = torch.load(os.path.join(dataset_dir, f), map_location='cpu')
        if 'batch_grasp_point' in data and 'xyz_graspable' in data:
            points = data['batch_grasp_point'][0].float() # [num_grasps, 3]
            seed_xyz = data['xyz_graspable'].squeeze(0).float() # [1024, 3]
            
            # Compute median of seed points in this scene
            seed_median = torch.median(seed_xyz, dim=0)[0] # [3]
            
            # Center the grasp translation by subtracting seed median
            points_centered = points - seed_median # [num_grasps, 3]
            all_points_centered.append(points_centered)
            
    if not all_points_centered:
        raise ValueError("No valid grasp and seed point data found in files!")
        
    all_points_centered = torch.cat(all_points_centered, dim=0) # [total_grasps, 3]
    std_p = all_points_centered.std(dim=0)
    
    # Avoid division by zero
    std_p[std_p < 1e-5] = 1.0
    
    stats = {
        'std_p': std_p
    }
    return stats

def normalize_x(x, stats, seed_median):
    """
    Normalize 8D grasp pose x = [p(3), omega(3), w(1), d(1)].
    x: [..., 8] Tensor
    stats: dict containing 'std_p'
    seed_median: [3] Tensor (median coordinates of seed points in this scene)
    """
    device = x.device
    std_p = stats['std_p'].to(device)
    
    p = x[..., :3]
    omega = x[..., 3:6]
    w = x[..., 6:7]
    d = x[..., 7:8]
    
    # 1. Normalize translation relative to seed median
    p_centered = p - seed_median.to(device)
    p_norm = p_centered / std_p
    
    # 2. Normalize rotation Lie algebra by Pi
    omega_norm = omega / np.pi
    
    # 3. Normalize width by 0.1 m
    w_norm = w / 0.1
    
    # 4. Normalize depth (index 0-3 converted to meters, mapped [0.01, 0.04] -> [0, 1])
    d_norm = (d - 0.01) / 0.03
    
    x_norm = torch.cat([p_norm, omega_norm, w_norm, d_norm], dim=-1)
    return x_norm

def denormalize_x(x_norm, stats, seed_median):
    """
    Denormalize 8D grasp pose back to physical units using seed median.
    x_norm: [..., 8] Tensor
    stats: dict containing 'std_p'
    seed_median: [3] Tensor (median coordinates of seed points in this scene)
    """
    device = x_norm.device
    std_p = stats['std_p'].to(device)
    
    p_norm = x_norm[..., :3]
    omega_norm = x_norm[..., 3:6]
    w_norm = x_norm[..., 6:7]
    d_norm = x_norm[..., 7:8]
    
    # 1. Denormalize translation back to camera coordinates
    p_centered = p_norm * std_p
    p = p_centered + seed_median.to(device)
    
    # 2. Denormalize rotation, width, and depth
    omega = omega_norm * np.pi
    w = w_norm * 0.1
    d = d_norm * 0.03 + 0.01
    
    x = torch.cat([p, omega, w, d], dim=-1)
    return x
