import os
import torch
from torch.utils.data import Dataset
from flow.utils.cfm_norm import normalize_x

class CFMDataset(Dataset):
    """
    Load cached seed-conditioned CFM samples and draw 5D grasp targets.

    Each scene stores ragged grasp pools for all 1024 seeds. This dataset keeps
    all seeds, samples one grasp configuration for each valid seed, and marks
    invalid seeds so the training loss can ignore them.
    """
    def __init__(self, dataset_dir, stats_path=None, limit=None):
        """Initialize the cached CFM dataset.

        Args:
            dataset_dir (str): Directory containing cached .pt scene files.
            stats_path (str | None): Optional path to normalization stats.
            limit (int | None): Optional maximum number of files to load.
        """
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
        """Return the number of cached scene files."""
        return len(self.files)

    def __getitem__(self, idx):
        """Sample one score-weighted 5D grasp target for every seed in a scene."""
        file_path = os.path.join(self.dataset_dir, self.files[idx])
        data = torch.load(file_path, map_location='cpu')
        
        # 1. Seed XYZ and Features
        # Original shapes:
        #   xyz_graspable: [1, 1024, 3] -> squeeze to [1024, 3]
        #   seed_features_graspable: [1, 512, 1024] -> squeeze to [512, 1024]
        seed_xyz_all = data['xyz_graspable'].squeeze(0).float()  # [N, 3]
        seed_feats_all = data['seed_features_graspable'].squeeze(0).float()  # [512, N]

        seed_valid_mask = data['seed_valid_mask'].squeeze(0).bool()  # [N]
        num_seed = seed_xyz_all.shape[0]

        rot_pools = data['seed_grasp_rot_lie_list'][0]
        width_pools = data['seed_grasp_width_list'][0]
        depth_pools = data['seed_grasp_depth_list'][0]
        score_pools = data['seed_grasp_score_list'][0]

        x1_raw = torch.zeros((num_seed, 5), dtype=torch.float32)  # [N, 5]
        sampled_scores = torch.zeros((num_seed,), dtype=torch.float32)  # [N]
        for seed_idx in range(num_seed):
            if not seed_valid_mask[seed_idx]:
                continue

            scores = score_pools[seed_idx].float()  # [Ki]
            probs = scores / scores.sum()
            grasp_idx = torch.multinomial(probs, num_samples=1).item()

            x1_raw[seed_idx, :3] = rot_pools[seed_idx][grasp_idx].float()  # [3]
            x1_raw[seed_idx, 3] = width_pools[seed_idx][grasp_idx].float()
            x1_raw[seed_idx, 4] = depth_pools[seed_idx][grasp_idx].float()
            sampled_scores[seed_idx] = scores[grasp_idx].float()

        if self.stats is not None:
            x1 = normalize_x(x1_raw, self.stats)
        else:
            x1 = x1_raw
        
        return {
            'x1': x1,
            'target_valid_mask': seed_valid_mask,
            'seed_xyz': seed_xyz_all,
            'seed_feats': seed_feats_all,
            'sampled_scores': sampled_scores,
            'scene_name': data['scene_name'],
            'frame_id': data['frame_id']
        }
