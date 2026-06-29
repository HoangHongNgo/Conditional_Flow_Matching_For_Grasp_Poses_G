import torch
import torch.nn as nn
import torch.nn.functional as F

import libs.pointnet2.pytorch_utils as pt_utils
from libs.pointnet2.pointnet2_utils import QueryAndGroup
from models.modules_economicgrasp import AttentionModule

class Sphere_Grouping_Global_Interaction(nn.Module):
    """Group seed features in a 5mm sphere and model local global interaction."""

    def __init__(self, nsample, seed_feature_dim, sphere_radius=0.005, out_dim=128):
        """Initialize spherical grouping around each graspable seed point.

        Args:
            nsample (int): Maximum number of neighbor points sampled per seed.
            seed_feature_dim (int): Channel dimension of seed features.
            sphere_radius (float): Ball-query radius in meters. Defaults to 5mm.
            out_dim (int): Output channel dimension of the per-seed conditioner.
        """
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.sphere_radius = sphere_radius
        self.out_dim = out_dim
        mlps = [3 + self.in_dim, out_dim, out_dim]
        mlps2 = [3 + out_dim, out_dim, out_dim]

        self.grouper = QueryAndGroup(radius=sphere_radius, nsample=nsample, use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)
        # Local interaction module, mirroring Cylinder_Grouping_Global_Interaction.
        self.local_interaction_module = AttentionModule(dim=3 + out_dim, n_head=1, msa_dropout=0.05)
        self.mlps2 = pt_utils.SharedMLP(mlps2, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable):
        """Apply spherical grouping and local self-attention to seed features.

        Args:
            seed_xyz_graspable (torch.Tensor): Graspable seed coordinates with shape [B, N, 3].
            seed_features_graspable (torch.Tensor): Seed features with shape [B, C, N].

        Returns:
            torch.Tensor: Grouped and interacted seed features with shape [B, out_dim, N].
        """
        # [B, 3, N, nsample]
        coords = seed_xyz_graspable.transpose(-1, -2).unsqueeze(-1).expand(-1, -1, -1, self.nsample)

        # Ball query gathers neighbors within sphere_radius around each seed point.
        # [B, 3 + C, N, nsample]
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, seed_features_graspable)
        new_features = self.mlps(grouped_feature)

        # [B * N, nsample, out_dim + 3]
        new_features = torch.cat([new_features, coords], dim=1).permute(0, 2, 3, 1).contiguous().view(-1, self.nsample, self.out_dim + 3)
        new_features = self.local_interaction_module(new_features, new_features, new_features, mask=None)

        # [B, out_dim + 3, N, nsample]
        new_features = new_features.view(
            seed_xyz_graspable.shape[0],
            seed_xyz_graspable.shape[1],
            self.nsample,
            3 + self.out_dim,
        ).permute(0, 3, 1, 2).contiguous()
        new_features = self.mlps2(new_features)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        new_features = new_features.squeeze(-1)

        return new_features
