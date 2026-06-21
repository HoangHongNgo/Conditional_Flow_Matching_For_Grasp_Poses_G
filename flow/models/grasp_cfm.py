import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np

from models.backbone import TDUnet
from models.modules_economicgrasp import GraspableNet
from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation

class SinusoidalPosEmb(nn.Module):
    """Encode scalar timesteps with sinusoidal positional embeddings."""

    def __init__(self, dim):
        """Initialize the sinusoidal embedding dimension."""
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Map timesteps with shape [B] or [B, 1] to embeddings [B, dim]."""
        x = x.reshape(-1)
        half_dim = self.dim // 2
        if half_dim == 0:
            return x.unsqueeze(-1)

        device = x.device
        dtype = x.dtype
        emb_scale = np.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=dtype) * -emb_scale
        )
        emb = x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.dim % 2 == 1:
            emb = torch.cat((emb, torch.zeros_like(emb[:, :1])), dim=-1)
        return emb


class GraspVelocityMLP(nn.Module):
    """
    MLP that models the conditional vector field v_theta.
    Maps [x_t, t, seed_cond] to the target flow velocity vector.
    """
    def __init__(self, grasp_dim=5, cond_dim=128, hidden_dim=512, state_dim=128):
        """Initialize the velocity MLP for seed-conditioned CFM.

        Args:
            grasp_dim (int): Target grasp dimension. Defaults to 5 for
                [omega(3), width(1), depth(1)].
            cond_dim (int): Per-seed condition feature dimension.
            hidden_dim (int): Hidden layer width.
            state_dim (int): Encoded dimension for the x_t and t branches.
        """
        super().__init__()
        self.grasp_dim = grasp_dim
        self.cond_dim = cond_dim
        self.state_dim = state_dim

        self.step_encoder = nn.Sequential(
            SinusoidalPosEmb(state_dim),
            nn.Linear(state_dim, state_dim * 2),
            nn.Mish(),
            nn.Linear(state_dim * 2, state_dim),
        )

        self.sample_encoder = nn.Sequential(
            nn.Linear(grasp_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )

        in_dim = (2 * state_dim) + cond_dim

        self.prediction_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, grasp_dim),
        )

    def forward(self, x_t, t, seed_cond):
        """
        Predict the CFM velocity for each sampled seed target.

        Args:
            x_t (torch.Tensor): Noisy grasp targets with shape [B, M, 5].
            t (torch.Tensor): Flow time with shape [B], [B, 1], [B, M],
                or [B, M, 1].
            seed_cond (torch.Tensor): Per-target seed features with shape [B, M, 128].

        Returns:
            torch.Tensor: Predicted velocity with shape [B, M, 5].
        """
        B, M, D = x_t.shape
        assert seed_cond.shape[:2] == (B, M), (
            f"seed_cond must have shape [B, M, C], got {tuple(seed_cond.shape)}"
        )
        
        if t.ndim == 1:
            t_flat = t[:, None, None].expand(B, M, 1).reshape(B * M, 1)
        elif t.ndim == 2 and t.shape[1] == 1:
            t_flat = t[:, None, :].expand(B, M, 1).reshape(B * M, 1)
        elif t.ndim == 2 and t.shape == (B, M):
            t_flat = t.reshape(B * M, 1)
        elif t.ndim == 3 and t.shape[:2] == (B, M) and t.shape[2] == 1:
            t_flat = t.reshape(B * M, 1)
        else:
            raise ValueError(
                f"t must have shape [B], [B, 1], [B, M], or [B, M, 1], got {tuple(t.shape)}"
            )

        # Flatten per-seed condition for all sampled seeds.
        cond_flat = seed_cond.reshape(B * M, -1)  # [B * M, 128]
        x_t_flat = x_t.reshape(B * M, D)  # [B * M, 5]

        # Shape: [B * M, state_dim]
        step_feat = self.step_encoder(t_flat)
        # Shape: [B * M, state_dim]
        sample_feat = self.sample_encoder(x_t_flat)

        fused_feat = torch.cat([sample_feat, step_feat, cond_flat], dim=-1)
        v_flat = self.prediction_head(fused_feat)  # [B * M, 5]
        
        v = v_flat.reshape(B, M, D)
        return v


class economic_graspable(nn.Module):
    """
    Extracts the base features from the input point cloud and predicts graspable seed points.
    Uses MinkowskiEngine for sparse convolution and Furthest Point Sampling to select seeds.
    """
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True, voxel_size=0.005, attn_layer=False, is_refine=False):
        """
        Initialize the economic_graspable module.

        Args:
            cylinder_radius (float): Radius of the cylinder grouping.
            seed_feat_dim (int): Dimension of the seed features.
            is_training (bool): Whether the model is in training mode.
            voxel_size (float): Voxel size for MinkowskiEngine quantization.
            attn_layer (bool): Flag for using attention layer.
            is_refine (bool): Flag for refine stage.

        Returns:
            None
        """
        super().__init__()
        from utils.arguments import cfgs

        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.M_points = cfgs.m_point
        self.graspness_threshold = cfgs.graspness_threshold
        self.voxel_size = voxel_size

        # Backbone
        self.backbone = TDUnet(
            in_channels=3, out_channels=self.seed_feature_dim, D=3)

        # Objectness and graspness
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)


    def forward(self, end_points):
        """
        Forward pass for the base feature extraction and seed sampling.

        Args:
            end_points (dict): A dictionary containing input data.
                - 'point_clouds': Raw point clouds. Shape: [B, N, 3]
                - 'coordinates_for_voxel': Coordinates list for MinkowskiEngine.

        Returns:
            dict: The updated end_points dictionary containing:
                - 'xyz_graspable': Selected graspable coordinates. Shape: [B, 1024, 3]
                - 'seed_features_graspable': Features of selected coordinates. Shape: [B, 512, 1024]
        """
        # use all sampled point cloud
        # [B, N, 3] where N is point_num (usually 15000 or 20000)
        seed_xyz = end_points['point_clouds']
        B, point_num, _ = seed_xyz.shape

        # Generate input to meet the Minkowski Engine
        coordinates_batch, features_batch = ME.utils.sparse_collate(
            [coord for coord in end_points['coordinates_for_voxel']],
            [feat for feat in np.ones_like(seed_xyz.cpu()).astype(np.float32)])
        coordinates_batch, features_batch, _, end_points['quantize2original'] = \
            ME.utils.sparse_quantize(
                coordinates_batch, features_batch, return_index=True, return_inverse=True)

        # [points of the whole scenes after quantize, 4] where 4 is (batch_idx, x, y, z)
        coordinates_batch = coordinates_batch.to(seed_xyz.device)
        # [points of the whole scenes after quantize, 3]
        features_batch = features_batch.to(seed_xyz.device)

        end_points['coors'] = coordinates_batch
        end_points['feats'] = features_batch
        
        mink_input = ME.SparseTensor(
            features_batch, coordinates=coordinates_batch)

        # Minkowski Backbone
        # [points of the whole scenes after quantize, 512]
        seed_features = self.backbone(mink_input).F
        
        # [B, 512, N]
        seed_features = seed_features[end_points['quantize2original']].view(
            B, point_num, -1).transpose(1, 2)

        # Generate the masks of the objectness and the graspness
        end_points = self.graspable(seed_features, end_points)
        
        # [B, N, 512]
        seed_features_flipped = seed_features.transpose(1, 2)
        
        # [B, 2, N]
        objectness_score = end_points['objectness_score']
        
        # [B, N]
        graspness_score = end_points['graspness_score'].squeeze(1)
        
        # [B, N]
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > self.graspness_threshold
        graspable_mask = objectness_mask & graspness_mask

        # Generate the downsample point (1024 per scene) using the furthest point sampling
        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.
        
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]
            cur_seed_xyz = seed_xyz[i][cur_mask]

            # [1, M, 3]
            cur_seed_xyz = cur_seed_xyz.unsqueeze(0)
            
            # fps_idxs: [1, 1024]
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            
            # cur_seed_xyz_flipped: [1, 3, M]
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
            
            # [1024, 3]
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(
                1, 2).squeeze(0).contiguous()
                
            # [1, 512, M]
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
            
            # [512, 1024]
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
            
        # [B, 1024, 3]
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
        
        # [B, 512, 1024]
        seed_features_graspable = torch.stack(seed_features_graspable)
        
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['seed_features_graspable'] = seed_features_graspable
        end_points['D: Graspable Points'] = graspable_num_batch / B

        return end_points
