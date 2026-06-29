import os

import torch
from torch.utils.data import Dataset


class ScoringDataset(Dataset):
    """Load compact scoring samples and draw fixed-size grasp config batches."""

    def __init__(
        self,
        dataset_dir,
        configs_per_frame=8192,
        positive_fraction=0.2,
        sampling='balanced',
        limit=None,
    ):
        """Initialize the scoring dataset from cached .pt frame samples."""
        self.dataset_dir = dataset_dir
        self.configs_per_frame = int(configs_per_frame)
        self.positive_fraction = float(positive_fraction)
        self.sampling = sampling

        if self.configs_per_frame <= 0:
            raise ValueError("configs_per_frame must be positive.")
        if not 0.0 <= self.positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be in [0, 1].")
        if self.sampling not in {'balanced', 'random'}:
            raise ValueError(f"sampling must be 'balanced' or 'random', got {sampling!r}.")
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Scoring dataset directory not found: {self.dataset_dir}")

        self.files = sorted(
            entry for entry in os.listdir(self.dataset_dir)
            if entry.endswith('.pt')
        )
        if limit is not None:
            self.files = self.files[:int(limit)]
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {self.dataset_dir}")

    def __len__(self):
        """Return the number of cached frame samples."""
        return len(self.files)

    def __getitem__(self, idx):
        """Return sampled grasp config inputs and 11-class score labels."""
        sample_path = os.path.join(self.dataset_dir, self.files[idx])
        sample = torch.load(sample_path, map_location='cpu', weights_only=True)
        return self.build_training_sample(sample)

    def build_training_sample(self, sample):
        """Sample fixed-size config tensors from one compact scoring frame."""
        seed_valid_mask = sample['seed_valid_mask'].squeeze(0).bool()  # [N]
        valid_seed_idx = torch.where(seed_valid_mask)[0]  # [Nv]
        if valid_seed_idx.numel() == 0:
            raise ValueError("Scoring sample has no valid seeds.")

        rot_lie = sample['seed_grasp_rot_lie'].squeeze(0).float()  # [N, 300, 3]
        group_features = sample['seed_group_features'].squeeze(0).float()  # [N, 128]
        score_labels = unpack_score_label(sample).squeeze(0).long()  # [N, 300]
        width = unpack_width(sample).squeeze(0)  # [N, 300]
        depth = unpack_depth(sample).squeeze(0)  # [N, 300]

        valid_scores = score_labels[valid_seed_idx]  # [Nv, 300]
        selected_flat_idx = self.sample_flat_indices(valid_scores)  # [K]
        local_seed_idx = torch.div(selected_flat_idx, valid_scores.shape[1], rounding_mode='floor')  # [K]
        view_idx = selected_flat_idx % valid_scores.shape[1]  # [K]
        seed_idx = valid_seed_idx[local_seed_idx]  # [K]

        selected_rot = rot_lie[seed_idx, view_idx]  # [K, 3]
        selected_width = width[seed_idx, view_idx].unsqueeze(-1)  # [K, 1]
        selected_depth = depth[seed_idx, view_idx].unsqueeze(-1)  # [K, 1]
        selected_features = group_features[seed_idx]  # [K, 128]
        selected_labels = score_labels[seed_idx, view_idx]  # [K]

        grasp_config = torch.cat(
            [selected_rot, selected_width, selected_depth, selected_features],
            dim=-1,
        ).contiguous()  # [K, 133]

        return {
            'grasp_config': grasp_config,
            'score_label': selected_labels.contiguous(),
            'scene_name': sample['scene_name'],
            'frame_id': int(sample['frame_id']),
        }

    def sample_flat_indices(self, valid_scores):
        """Sample flattened config indices from valid-seed score labels."""
        flat_scores = valid_scores.reshape(-1)  # [Nv * 300]
        all_idx = torch.arange(flat_scores.numel(), dtype=torch.long)
        if self.sampling == 'random':
            return sample_candidates(all_idx, self.configs_per_frame)

        positive_idx = torch.where(flat_scores > 0)[0]
        zero_idx = torch.where(flat_scores == 0)[0]
        num_positive = int(round(self.configs_per_frame * self.positive_fraction))
        num_zero = self.configs_per_frame - num_positive

        selected_parts = []
        if num_positive > 0:
            source = positive_idx if positive_idx.numel() > 0 else zero_idx
            selected_parts.append(sample_candidates(source, num_positive))
        if num_zero > 0:
            source = zero_idx if zero_idx.numel() > 0 else positive_idx
            selected_parts.append(sample_candidates(source, num_zero))

        if not selected_parts:
            return sample_candidates(all_idx, self.configs_per_frame)
        selected = torch.cat(selected_parts, dim=0)  # [K]
        shuffle_idx = torch.randperm(selected.numel())
        return selected[shuffle_idx]


def sample_candidates(candidates, sample_count):
    """Sample candidate indices with replacement only when needed."""
    if candidates.numel() == 0:
        raise ValueError("Cannot sample from an empty candidate set.")
    if candidates.numel() >= sample_count:
        perm = torch.randperm(candidates.numel())[:sample_count]
        return candidates[perm]
    draw_idx = torch.randint(candidates.numel(), (sample_count,), dtype=torch.long)
    return candidates[draw_idx]


def unpack_score_label(sample):
    """Return uint8 score labels in the 0..10 class range."""
    if 'seed_grasp_score_uint8' in sample:
        return sample['seed_grasp_score_uint8'].long()
    return torch.clamp(torch.round(sample['seed_grasp_score'].float() * 10.0), 0, 10).long()


def unpack_width(sample):
    """Return grasp widths in meters as float32 tensors."""
    if 'seed_grasp_width_uint8' in sample:
        return sample['seed_grasp_width_uint8'].float() / 1000.0
    return sample['seed_grasp_width'].float()


def unpack_depth(sample):
    """Return grasp depths in meters as float32 tensors."""
    if 'seed_grasp_depth_uint8' in sample:
        return 0.01 + sample['seed_grasp_depth_uint8'].float() * 0.01
    return sample['seed_grasp_depth'].float()
