import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.arguments import cfgs
from flow.models.grasp_cfm import economic_graspable
from libs.knn.knn_modules import knn
from utils.loss_utils import batch_viewpoint_params_to_matrix, transform_point_cloud, generate_grasp_views

def survey_configs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading a test batch...")
    dataset = GraspNetDataset(
        cfgs.dataset_root, split='test_seen', camera=cfgs.camera, 
        num_points=cfgs.num_point, remove_outlier=True, load_label=True, augment=False
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)
    batch_data = next(iter(dataloader))
    
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        elif 'graph' in key:
            for i in range(len(batch_data[key])):
                batch_data[key][i] = batch_data[key][i].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
            
    print("Initializing base net...")
    base_net = economic_graspable(seed_feat_dim=512, is_training=False).to(device)
    checkpoint_path = "checkpoints/economicgrasp_realsense.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    base_net.eval()
    
    print("Extracting seed points...")
    with torch.no_grad():
        end_points = base_net(batch_data)
        
    seed_xyzs = end_points['xyz_graspable']
    B, num_seed, _ = seed_xyzs.shape
    
    DIST_THRESH = 0.005
    SCORE_THRESH = 0.8
    
    all_valid_counts = []
    
    print("Calculating distances and matching labels...")
    for b in range(B):
        seed_xyz = seed_xyzs[b]
        poses = batch_data['object_poses_list'][b]
        
        grasp_points_merged = []
        top_view_index_merged = []
        grasp_scores_merged = []
        
        for obj_idx, pose in enumerate(poses):
            grasp_points = batch_data['grasp_points_list'][b][obj_idx]
            grasp_scores = batch_data['grasp_scores_list'][b][obj_idx]
            top_view_index = batch_data['top_view_index_list'][b][obj_idx]
            num_grasp_points = grasp_points.size(0)

            grasp_views = generate_grasp_views(cfgs.num_view).to(device)
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')

            angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=device)
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)

            grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
            grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
            view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1

            top_view_index_trans = -1 * torch.ones((num_grasp_points, grasp_scores.shape[1]), dtype=torch.long, device=device)
            tpid, tvip, tids = torch.where(view_inds == top_view_index.unsqueeze(-1))
            top_view_index_trans[tpid, tvip] = tids

            grasp_points_merged.append(grasp_points_trans)
            top_view_index_merged.append(top_view_index_trans)
            grasp_scores_merged.append(grasp_scores)

        if len(grasp_points_merged) == 0:
            continue
            
        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)
        top_view_index_merged = torch.cat(top_view_index_merged, dim=0)
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)

        dists = torch.cdist(seed_xyz.unsqueeze(0), grasp_points_merged.unsqueeze(0)).squeeze(0)
        
        for s_idx in range(num_seed):
            in_radius = dists[s_idx] <= DIST_THRESH
            if in_radius.sum() == 0:
                all_valid_counts.append(0)
                continue
                
            nb_indices = torch.where(in_radius)[0]
            m_scores = grasp_scores_merged[nb_indices]
            m_top_views = top_view_index_merged[nb_indices]
            
            valid_slots = (m_scores > SCORE_THRESH) & (m_top_views != -1)
            num_valid = valid_slots.sum().item()
            all_valid_counts.append(num_valid)

    all_valid_counts = np.array(all_valid_counts)
    print(f"\n--- Survey Results (Score > {SCORE_THRESH}, Dist <= {DIST_THRESH}m) ---")
    print(f"Total Seed Points evaluated: {len(all_valid_counts)}")
    print(f"Mean valid configs per seed: {np.mean(all_valid_counts):.2f}")
    print(f"Median valid configs per seed: {np.median(all_valid_counts)}")
    print(f"Max valid configs per seed: {np.max(all_valid_counts)}")
    print(f"Min valid configs per seed: {np.min(all_valid_counts)}")
    print(f"Seed points with 0 configs: {np.sum(all_valid_counts == 0)} ({(np.sum(all_valid_counts == 0)/len(all_valid_counts))*100:.2f}%)")
    print(f"Seed points with > 128 configs: {np.sum(all_valid_counts > 128)} ({(np.sum(all_valid_counts > 128)/len(all_valid_counts))*100:.2f}%)")

if __name__ == '__main__':
    survey_configs()
