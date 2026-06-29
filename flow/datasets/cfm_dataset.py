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
    def __init__(self, dataset_dir, stats_path=None, limit=None, target_sampling='top8_rot_weighted'):
        """Initialize the cached CFM dataset.

        Args:
            dataset_dir (str): Directory containing cached .pt scene files.
            stats_path (str | None): Optional path to fixed normalization metadata.
            limit (int | None): Optional maximum number of files to load.
            target_sampling (str): Strategy for selecting one target from each
                seed's grasp pool. Use 'top8_rot_weighted' to sample from the
                best-score grasp and its 7 nearest rotation neighbors, 'best_score'
                for a stable unimodal target, 'score_weighted' for stochastic
                score-weighted sampling, or 'uniform' for unweighted stochastic
                sampling.
        """
        self.dataset_dir = dataset_dir
        self.target_sampling = target_sampling
        self.files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.pt')])
        if limit is not None:
            self.files = self.files[:limit]
            
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {dataset_dir}")

        valid_sampling = {'top8_rot_weighted', 'best_score', 'score_weighted', 'uniform'}
        if self.target_sampling not in valid_sampling:
            raise ValueError(
                f"target_sampling must be one of {sorted(valid_sampling)}, got {target_sampling!r}"
            )
            
        # Load fixed normalization metadata.
        if stats_path is not None and os.path.exists(stats_path):
            self.stats = torch.load(stats_path, map_location='cpu', weights_only=True)
            print(f"Loaded normalization metadata from: {stats_path}")
        else:
            print("Warning: stats_path not found. Targets will use raw scale.")
            self.stats = None

    def __len__(self):
        """Return the number of cached scene files."""
        return len(self.files)

    def __getitem__(self, idx):
        """Sample one 5D grasp target for every valid seed in a scene."""
        file_path = os.path.join(self.dataset_dir, self.files[idx])
        data = torch.load(file_path, map_location='cpu', weights_only=True)
        
        # 1. Seed XYZ and Features
        # Original shapes:
        #   xyz_graspable: [1, 1024, 3] -> squeeze to [1024, 3]
        #   seed_features_graspable: [1, 512, 1024] -> squeeze to [512, 1024]
        seed_xyz_all = data['xyz_graspable'].squeeze(0).float()  # [N, 3]
        seed_feats_all = data['seed_features_graspable'].squeeze(0).float()  # [512, N]

        seed_valid_mask = data['seed_valid_mask'].squeeze(0).bool()  # [N]
        num_seed = seed_xyz_all.shape[0]

        rot_pools = data['seed_grasp_rot_lie'].squeeze(0).float()  # [N, K, 3]
        width_pools = data['seed_grasp_width'].squeeze(0).float()  # [N, K]
        depth_pools = data['seed_grasp_depth'].squeeze(0).float()  # [N, K]
        score_pools = data['seed_grasp_score'].squeeze(0).float()  # [N, K]
        grasp_count = data['seed_grasp_count'].squeeze(0).long()  # [N]

        x1_raw = torch.zeros((num_seed, 5), dtype=torch.float32)  # [N, 5]
        sampled_scores = torch.zeros((num_seed,), dtype=torch.float32)  # [N]
        for seed_idx in range(num_seed):
            if not seed_valid_mask[seed_idx]:
                continue

            valid_count = int(grasp_count[seed_idx].item())
            scores = score_pools[seed_idx, :valid_count]  # [Ki]
            if self.target_sampling == 'best_score':
                grasp_idx = torch.argmax(scores).item()
            elif self.target_sampling == 'score_weighted':
                probs = scores / scores.sum().clamp_min(1e-8)
                grasp_idx = torch.multinomial(probs, num_samples=1).item()
            elif self.target_sampling == 'top8_rot_weighted':
                best_idx = torch.argmax(scores)  # []
                rot_candidates = rot_pools[seed_idx, :valid_count]  # [Ki, 3]
                best_rot = rot_candidates[best_idx].unsqueeze(0)  # [1, 3]
                rot_dist = torch.linalg.norm(rot_candidates - best_rot, dim=-1)  # [Ki]

                if valid_count <= 8:
                    selected_idx = torch.arange(valid_count, dtype=torch.long)  # [Ki]
                else:
                    candidate_idx = torch.arange(valid_count, dtype=torch.long)
                    neighbor_idx = candidate_idx[candidate_idx != best_idx]  # [Ki - 1]
                    neighbor_dist = rot_dist[neighbor_idx]  # [Ki - 1]
                    nearest_local = torch.argsort(
                        neighbor_dist,
                        stable=True,
                    )[:7]  # [7]
                    selected_idx = torch.cat(
                        [best_idx.reshape(1), neighbor_idx[nearest_local]],
                        dim=0,
                    )  # [8]
                selected_scores = scores[selected_idx]  # [<=8]
                probs = selected_scores / selected_scores.sum().clamp_min(1e-8)
                local_idx = torch.multinomial(probs, num_samples=1).item()
                grasp_idx = selected_idx[local_idx].item()
            else:
                grasp_idx = torch.randint(valid_count, (1,)).item()

            x1_raw[seed_idx, :3] = rot_pools[seed_idx, grasp_idx]  # [3]
            x1_raw[seed_idx, 3] = width_pools[seed_idx, grasp_idx]
            x1_raw[seed_idx, 4] = depth_pools[seed_idx, grasp_idx]
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
