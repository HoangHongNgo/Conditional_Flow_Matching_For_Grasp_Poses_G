import os
import torch
from torch.utils.data import Dataset
from flow.utils.cfm_norm import normalize_x

class CFMDataset(Dataset):
    """
    Loads cached .pt files containing pre-extracted seed features, seed point xyz,
    and ground truth grasps. Normalizes grasps using computed stats and pads
    the grasps array to exactly 50 poses per scene.
    """
    def __init__(self, dataset_dir, stats_path=None, limit=None):
        self.dataset_dir = dataset_dir
        self.files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.pt')])
        if limit is not None:
            self.files = self.files[:limit]
            
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {dataset_dir}")
            
        # Load or compute normalization stats
        if stats_path is not None and os.path.exists(stats_path):
            self.stats = torch.load(stats_path, map_location='cpu')
            print(f"Loaded normalization stats from: {stats_path}")
        else:
            print("Warning: stats_path not found. Please calculate statistics first.")
            self.stats = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.dataset_dir, self.files[idx])
        data = torch.load(file_path, map_location='cpu')
        
        # 1. Seed XYZ and Features
        # Original shapes:
        #   xyz_graspable: [1, 1024, 3] -> squeeze to [1024, 3]
        #   seed_features_graspable: [1, 512, 1024] -> squeeze and transpose to [1024, 512]
        seed_xyz = data['xyz_graspable'].squeeze(0).float()
        seed_feats = data['seed_features_graspable'].squeeze(0).permute(1, 0).float()
        
        # 2. Extract Ground Truth components from data (lists of length 1 containing Tensor)
        gt_points = data['batch_grasp_point'][0].float()      # [num_grasps, 3]
        gt_rot_lie = data['batch_grasp_views_rot_lie'][0].float() # [num_grasps, 3]
        gt_width = data['batch_grasp_width'][0].float()        # [num_grasps]
        gt_depth = data['batch_grasp_depth'][0].float()        # [num_grasps]
        
        num_grasps = gt_points.shape[0]
        
        # Depth index mapping:
        # In original labels, batch_grasp_depth is integer indices 0 to 3.
        # We need to map these to actual depth values in meters [0.01, 0.02, 0.03, 0.04]
        # Depth mapping: depth_meters = 0.01 + depth_index * 0.01
        gt_depth_meters = 0.01 + gt_depth * 0.01
        
        # Compute median of seed points
        seed_median = torch.median(seed_xyz, dim=0)[0] # [3]
        
        # 3. Concatenate to 8D: [p, omega, w, d]
        # Shape: [num_grasps, 8]
        x1_raw = torch.cat([
            gt_points,
            gt_rot_lie,
            gt_width.unsqueeze(-1),
            gt_depth_meters.unsqueeze(-1)
        ], dim=-1)
        
        # 4. Normalize
        if self.stats is not None:
            x1_norm = normalize_x(x1_raw, self.stats, seed_median)
        else:
            x1_norm = x1_raw

            
        # 5. Take exactly 50 grasps
        # Since every scene has a rich set of top grasps, we slice to 50.
        # If any scene has fewer than 50, we replicate existing grasps to reach 50,
        # ensuring no zero padding or grasp masking is ever needed.
        max_grasps = 50
        if num_grasps >= max_grasps:
            x1 = x1_norm[:max_grasps]
        else:
            repeat_factor = (max_grasps + num_grasps - 1) // num_grasps
            x1 = x1_norm.repeat(repeat_factor, 1)[:max_grasps]
        
        return {
            'x1': x1,
            'seed_feats': seed_feats,
            'seed_xyz': seed_xyz,
            'scene_name': data['scene_name'],
            'frame_id': data['frame_id']
        }

