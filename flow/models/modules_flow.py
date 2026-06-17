import torch
import torch.nn as nn
import torch.nn.functional as F

import libs.pointnet2.pytorch_utils as pt_utils
from libs.pointnet2.pointnet2_utils import QueryAndGroup
from models.modules_economicgrasp import AttentionModule

class Sphere_Grouping_Global_Interaction(nn.Module):
    """
    Groups features in a local spherical neighborhood using QueryAndGroup
    and models global interaction using Self-Attention.
    """
    def __init__(self, nsample, seed_feature_dim, sphere_radius=0.05):
        """
        Initialize the Sphere_Grouping_Global_Interaction module.

        Args:
            nsample (int): Maximum number of features to gather in the sphere.
            seed_feature_dim (int): Dimension of the input seed features.
            sphere_radius (float): Radius of the sphere for grouping.
            
        Returns:
            None
        """
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.sphere_radius = sphere_radius
        mlps = [3 + self.in_dim, 256, 256]
        mlps2 = [3 + 256, 256, 256]

        self.grouper = QueryAndGroup(radius=sphere_radius, nsample=nsample, use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)
        
        # Local interaction module
        self.local_interaction_module = AttentionModule(dim=3 + 256, n_head=1, msa_dropout=0.05)
        self.mlps2 = pt_utils.SharedMLP(mlps2, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable):
        """
        Forward pass for the Sphere Grouping.

        Args:
            seed_xyz_graspable (torch.Tensor): Coordinates of the graspable seed points. Shape: [B, 1024, 3]
            seed_features_graspable (torch.Tensor): Features of the graspable seed points. Shape: [B, C, 1024] where C is seed_feature_dim.

        Returns:
            torch.Tensor: Grouped and interacted features. Shape: [B, 256, 1024]
        """
        # [B, 3, 1024, nsample]
        coords = seed_xyz_graspable.transpose(-1, -2).unsqueeze(-1).expand(-1, -1, -1, self.nsample)
        
        # Group features using QueryAndGroup (spherical neighborhood)
        # [B, 3 + C, 1024, nsample]
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, seed_features_graspable)
        
        # [B, 256, 1024, nsample]
        new_features = self.mlps(grouped_feature)
        
        # Concatenate coordinates and prepare for AttentionModule
        # [B * 1024, nsample, 256 + 3]
        new_features = torch.cat([new_features, coords], dim=1).permute(0, 2, 3, 1).contiguous().view(-1, self.nsample, 256 + 3)
        
        # Apply Self-Attention for local interaction
        # [B * 1024, nsample, 256 + 3]
        new_features = self.local_interaction_module(new_features, new_features, new_features, mask=None)
        
        # Reshape back to feature maps
        # [B, 256 + 3, 1024, nsample]
        new_features = new_features.view(seed_xyz_graspable.shape[0], seed_xyz_graspable.shape[1], self.nsample, 3 + 256).permute(0, 3, 1, 2).contiguous()
        
        # Process through the second MLP
        # [B, 256, 1024, nsample]
        new_features = self.mlps2(new_features)
        
        # Extract features with Max-Pooling across the nsample dimension
        # [B, 256, 1024, 1]
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        
        # Squeeze the last dimension
        # [B, 256, 1024]
        new_features = new_features.squeeze(-1)
        
        return new_features
