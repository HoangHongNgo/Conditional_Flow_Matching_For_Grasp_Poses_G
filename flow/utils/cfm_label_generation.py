import os
import sys
import torch
import numpy as np

from libs.knn.knn_modules import knn
from utils.loss_utils import (batch_viewpoint_params_to_matrix, transform_point_cloud,
                              generate_grasp_views, compute_pointwise_dists)
from utils.arguments import cfgs
from flow.utils.lie import rotation_matrix_to_lie_vector

def process_grasp_labels(end_points, max_k=192):
    """
    Build padded seed-level grasp pools for seed-conditioned 5D CFM.

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
            max_k (int): Maximum number of grasp configs kept per seed after padding.
        
    Returns:
        dict: The input dictionary updated with padded seed-level tensors:
            - 'seed_grasp_rot_lie': Tensor with shape [B, N, K, 3].
            - 'seed_grasp_width': Tensor with shape [B, N, K].
            - 'seed_grasp_depth': Tensor with shape [B, N, K], in meters.
            - 'seed_grasp_score': Tensor with shape [B, N, K].
            - 'seed_grasp_slot_mask': Boolean tensor with shape [B, N, K].
            - 'seed_grasp_count': Long tensor with shape [B, N].
            - 'seed_valid_mask': Boolean tensor with shape [B, N].
    """
    seed_xyzs = end_points['xyz_graspable']  # [B, 1024, 3]
    B, num_seed, _ = seed_xyzs.shape
    device = seed_xyzs.device
    
    DIST_THRESH = 0.005  # 5mm
    SCORE_THRESH = 0.7
    seed_grasp_rot_lie = torch.zeros((B, num_seed, max_k, 3), dtype=torch.float32, device=device)  # [B, N, K, 3]
    seed_grasp_width = torch.zeros((B, num_seed, max_k), dtype=torch.float32, device=device)  # [B, N, K]
    seed_grasp_depth = torch.zeros((B, num_seed, max_k), dtype=torch.float32, device=device)  # [B, N, K]
    seed_grasp_score = torch.zeros((B, num_seed, max_k), dtype=torch.float32, device=device)  # [B, N, K]
    seed_grasp_slot_mask = torch.zeros((B, num_seed, max_k), dtype=torch.bool, device=device)  # [B, N, K]
    seed_grasp_count = torch.zeros((B, num_seed), dtype=torch.long, device=device)  # [B, N]
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
            keep_count = min(num_valid, max_k)

            # Sort by score so truncation keeps stronger grasp configs first.
            scores = grasp_scores_merged[m_abs_idx, v_idx]  # [Ki]
            if num_valid > max_k:
                top_idx = torch.topk(scores, k=max_k, largest=True, sorted=True).indices  # [K]
                rot_lie = rot_lie[top_idx]  # [K, 3]
                width = grasp_widths_merged[m_abs_idx, v_idx][top_idx]  # [K]
                depth_meters = depth_meters[top_idx]  # [K]
                scores = scores[top_idx]  # [K]
            else:
                width = grasp_widths_merged[m_abs_idx, v_idx]  # [Ki]

            seed_grasp_rot_lie[b, s_idx, :keep_count] = rot_lie[:keep_count]  # [keep_count, 3]
            seed_grasp_width[b, s_idx, :keep_count] = width[:keep_count]  # [keep_count]
            seed_grasp_depth[b, s_idx, :keep_count] = depth_meters[:keep_count]  # [keep_count]
            seed_grasp_score[b, s_idx, :keep_count] = scores[:keep_count]  # [keep_count]
            seed_grasp_slot_mask[b, s_idx, :keep_count] = True  # [keep_count]
            seed_grasp_count[b, s_idx] = keep_count
            seed_valid_mask[b, s_idx] = True

    # 4. Pack seed-conditioned ragged targets into end_points.
    end_points['seed_grasp_rot_lie'] = seed_grasp_rot_lie  # [B, N, K, 3]
    end_points['seed_grasp_width'] = seed_grasp_width  # [B, N, K]
    end_points['seed_grasp_depth'] = seed_grasp_depth  # [B, N, K]
    end_points['seed_grasp_score'] = seed_grasp_score  # [B, N, K]
    end_points['seed_grasp_slot_mask'] = seed_grasp_slot_mask  # [B, N, K]
    end_points['seed_grasp_count'] = seed_grasp_count  # [B, N]
    end_points['seed_valid_mask'] = seed_valid_mask  # [B, N]
    end_points['seed_grasp_max_k'] = max_k

    return end_points
