import os
import sys
import torch
import numpy as np

from libs.knn.knn_modules import knn
from utils.loss_utils import (batch_viewpoint_params_to_matrix, transform_point_cloud,
                              generate_grasp_views, compute_pointwise_dists)
from utils.arguments import cfgs
from flow.utils.lie import rotation_matrix_to_lie_vector

def process_grasp_labels(end_points):
    """
    Build seed-level ragged grasp pools for seed-conditioned 5D CFM.

    For each seed point, this function keeps every grasp configuration whose
    grasp point is within 5mm and whose normalized score is greater than 0.7.
    The target translation is implicit: each selected grasp uses the seed point
    itself as its application position. The learned CFM target is therefore
    [omega(3), width(1), depth(1)].

    Args:
        end_points: dict containing model predictions and ground truth labels. 
            Expected keys include:
            - 'xyz_graspable': Tensor of shape [B, 1024, 3] representing seed points.
            - 'object_poses_list': List of length B, containing lists of object poses [3, 4].
            - 'grasp_points_list': List of length B, containing lists of object grasp points [P, 3].
            - 'grasp_rotations_list': List of length B, containing lists of object grasp rotations [P, 300].
            - 'grasp_depth_list': List of length B, containing lists of object grasp depths [P, 300].
            - 'grasp_scores_list': List of length B, containing lists of object grasp scores [P, 300].
            - 'grasp_widths_list': List of length B, containing lists of object grasp widths [P, 300].
            - 'top_view_index_list': List of length B, containing lists of object top view indices [P, 300].
        
    Returns:
        dict: The input dictionary updated with seed-level ragged lists:
            - 'seed_grasp_rot_lie_list': List[B][N] of tensors with shape [Ki, 3].
            - 'seed_grasp_width_list': List[B][N] of tensors with shape [Ki].
            - 'seed_grasp_depth_list': List[B][N] of tensors with shape [Ki], in meters.
            - 'seed_grasp_score_list': List[B][N] of tensors with shape [Ki].
            - 'seed_valid_mask': Boolean tensor with shape [B, N].
    """
    seed_xyzs = end_points['xyz_graspable']  # [B, 1024, 3]
    B, num_seed, _ = seed_xyzs.shape
    device = seed_xyzs.device
    
    DIST_THRESH = 0.005  # 5mm
    SCORE_THRESH = 0.7

    seed_grasp_rot_lie_list = []
    seed_grasp_width_list = []
    seed_grasp_depth_list = []
    seed_grasp_score_list = []
    seed_valid_mask = torch.zeros((B, num_seed), dtype=torch.bool, device=device)  # [B, N]

    for b in range(B):
        seed_xyz = seed_xyzs[b]  # [1024, 3]
        poses = end_points['object_poses_list'][b]  # list of poses [3, 4] for each object in scene
        
        grasp_points_merged = []
        grasp_rotations_merged = []
        grasp_depth_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        grasp_views_rot_merged = []
        top_view_index_merged = []

        scene_rot_lie_list = []
        scene_width_list = []
        scene_depth_list = []
        scene_score_list = []
        
        # 1. Merge all object GT labels into scene coordinate
        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points['grasp_points_list'][b][obj_idx]  # [P, 3]
            grasp_rotations = end_points['grasp_rotations_list'][b][obj_idx]  # [P, 300]
            grasp_depth = end_points['grasp_depth_list'][b][obj_idx]  # [P, 300]
            grasp_scores = end_points['grasp_scores_list'][b][obj_idx]  # [P, 300]
            grasp_widths = end_points['grasp_widths_list'][b][obj_idx]  # [P, 300]
            top_view_index = end_points['top_view_index_list'][b][obj_idx]  # [P, 300]
            num_grasp_points = grasp_points.size(0)

            # Transform template views
            grasp_views = generate_grasp_views(cfgs.num_view).to(device)  # [300, 3]
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')  # [P, 3]
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')  # [300, 3]

            # Rotation matrix of template views
            angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=device)  # [300]
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)  # [300, 3, 3]
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)  # [300, 3, 3]

            # Match top views transformed to scene coordinate
            grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)  # [1, 3, 300]
            grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)  # [1, 3, 300]

            view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1  # [300]

            # Match top views transformed to scene coordinate
            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)  # [300, 3, 3]
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1)
            # [P, 300, 3, 3]

            # Initialize as -1 for unmatched views (invalid)
            top_view_index_trans = -1 * torch.ones((num_grasp_points, grasp_rotations.shape[1]), dtype=torch.long, device=device)  # [P, 300]
            tpid, tvip, tids = torch.where(view_inds == top_view_index.unsqueeze(-1))
            top_view_index_trans[tpid, tvip] = tids  # [P, 300]

            grasp_points_merged.append(grasp_points_trans)
            top_view_index_merged.append(top_view_index_trans)
            grasp_rotations_merged.append(grasp_rotations)
            grasp_depth_merged.append(grasp_depth)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)
            grasp_views_rot_merged.append(grasp_views_rot_trans)

        if len(grasp_points_merged) == 0:
            empty_vec = seed_xyz.new_zeros((0, 3))
            empty_attr = seed_xyz.new_zeros((0,))
            seed_grasp_rot_lie_list.append([empty_vec for _ in range(num_seed)])
            seed_grasp_width_list.append([empty_attr for _ in range(num_seed)])
            seed_grasp_depth_list.append([empty_attr for _ in range(num_seed)])
            seed_grasp_score_list.append([empty_attr for _ in range(num_seed)])
            continue
            
        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # [M, 3]
        top_view_index_merged = torch.cat(top_view_index_merged, dim=0)  # [M, 300]
        grasp_rotations_merged = torch.cat(grasp_rotations_merged, dim=0)  # [M, 300]
        grasp_depth_merged = torch.cat(grasp_depth_merged, dim=0)  # [M, 300]
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)  # [M, 300]
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # [M, 300]
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # [M, 300, 3, 3]

        # 2. Calculate pairwise distances between seed points and all GT points
        dists = torch.cdist(seed_xyz.unsqueeze(0), grasp_points_merged.unsqueeze(0)).squeeze(0)  # [1024, M]
        
        # 3. Filter all valid grasp configurations for each seed point
        for s_idx in range(num_seed):
            # Mask of valid neighbor points
            in_radius = dists[s_idx] <= DIST_THRESH  # [M]
            
            if in_radius.sum() == 0:
                scene_rot_lie_list.append(seed_xyz.new_zeros((0, 3)))
                scene_width_list.append(seed_xyz.new_zeros((0,)))
                scene_depth_list.append(seed_xyz.new_zeros((0,)))
                scene_score_list.append(seed_xyz.new_zeros((0,)))
                continue
            
            # Extract indices of neighbor GT points
            nb_indices = torch.where(in_radius)[0]  # [num_nb]
            
            # Gather 300 view slots of neighbor points
            m_scores = grasp_scores_merged[nb_indices]  # [num_nb, 300]
            m_top_views = top_view_index_merged[nb_indices]  # [num_nb, 300]
            
            # Valid slots: Score > 0.7 AND top_view_index is not -1
            valid_slots = (m_scores > SCORE_THRESH) & (m_top_views != -1)  # [num_nb, 300]
            
            num_valid = valid_slots.sum().item()
            if num_valid == 0:
                scene_rot_lie_list.append(seed_xyz.new_zeros((0, 3)))
                scene_width_list.append(seed_xyz.new_zeros((0,)))
                scene_depth_list.append(seed_xyz.new_zeros((0,)))
                scene_score_list.append(seed_xyz.new_zeros((0,)))
                continue
                
            # Get exact (m, v) indices of valid slots
            m_rel_idx, v_idx = torch.where(valid_slots)  # [num_valid], [num_valid]
            m_abs_idx = nb_indices[m_rel_idx]  # Convert back to absolute M indices: [num_valid]

            top_view_idx = top_view_index_merged[m_abs_idx, v_idx]  # [Ki]
            views_rot = grasp_views_rot_merged[m_abs_idx, top_view_idx]  # [Ki, 3, 3]
            angle_idx = grasp_rotations_merged[m_abs_idx, v_idx].to(views_rot.dtype)  # [Ki]
            grasp_angle = angle_idx * np.pi / 12.0  # [Ki]

            ones = torch.ones(num_valid, dtype=views_rot.dtype, device=device)
            zeros = torch.zeros(num_valid, dtype=views_rot.dtype, device=device)
            sin = torch.sin(grasp_angle)
            cos = torch.cos(grasp_angle)
            angle_rot = torch.stack(
                [ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos],
                dim=-1
            ).reshape(num_valid, 3, 3)  # [Ki, 3, 3]

            full_rot = torch.matmul(views_rot, angle_rot)  # [Ki, 3, 3]
            rot_lie = rotation_matrix_to_lie_vector(full_rot)  # [Ki, 3]
            depth_meters = 0.01 + grasp_depth_merged[m_abs_idx, v_idx].float() * 0.01  # [Ki]

            scene_rot_lie_list.append(rot_lie)
            scene_width_list.append(grasp_widths_merged[m_abs_idx, v_idx])  # [Ki]
            scene_depth_list.append(depth_meters)  # [Ki]
            scene_score_list.append(grasp_scores_merged[m_abs_idx, v_idx])  # [Ki]
            seed_valid_mask[b, s_idx] = True

        seed_grasp_rot_lie_list.append(scene_rot_lie_list)
        seed_grasp_width_list.append(scene_width_list)
        seed_grasp_depth_list.append(scene_depth_list)
        seed_grasp_score_list.append(scene_score_list)

    # 4. Pack seed-conditioned ragged targets into end_points.
    end_points['seed_grasp_rot_lie_list'] = seed_grasp_rot_lie_list  # List[B][N] of [Ki, 3]
    end_points['seed_grasp_width_list'] = seed_grasp_width_list  # List[B][N] of [Ki]
    end_points['seed_grasp_depth_list'] = seed_grasp_depth_list  # List[B][N] of [Ki]
    end_points['seed_grasp_score_list'] = seed_grasp_score_list  # List[B][N] of [Ki]
    end_points['seed_valid_mask'] = seed_valid_mask  # [B, N]

    return end_points
