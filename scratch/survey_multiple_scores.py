import os
import sys
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.arguments import cfgs
from flow.models.grasp_cfm import economic_graspable
from libs.knn.knn_modules import knn
from utils.loss_utils import batch_viewpoint_params_to_matrix, transform_point_cloud, generate_grasp_views

def survey_multiple_scores():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading test dataset...")
    dataset = GraspNetDataset(
        cfgs.dataset_root, split='test_seen', camera=cfgs.camera, 
        num_points=cfgs.num_point, remove_outlier=True, load_label=True, augment=False
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    print("Initializing base net...")
    base_net = economic_graspable(seed_feat_dim=512, is_training=False).to(device)
    checkpoint_path = "checkpoints/economicgrasp_realsense.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    base_net.eval()
    
    DIST_THRESH = 0.005
    SCORE_THRESHOLDS = [0.7]
    
    # Store geodesic distances
    results = {thresh: {'dist_mean': [], 'dist_median': [], 'dist_90th': [], 'all_dists': []} for thresh in SCORE_THRESHOLDS}
    
    total_scenes = 0
    max_scenes = 12
    
    print(f"Calculating distances and matching labels across {max_scenes} random scenes...")
    
    for batch_idx, batch_data in enumerate(dataloader):
        if total_scenes >= max_scenes:
            break
            
        B = len(batch_data['point_clouds'])
        total_scenes += B
        print(f"Processing batch {batch_idx+1} (Total scenes processed: {total_scenes}/{max_scenes})")
        
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
                
        with torch.no_grad():
            end_points = base_net(batch_data)
            
        seed_xyzs = end_points['xyz_graspable']
        
        for b in range(B):
            seed_xyz = seed_xyzs[b]
            poses = batch_data['object_poses_list'][b]
            
            grasp_points_merged = []
            top_view_index_merged = []
            grasp_scores_merged = []
            grasp_widths_merged = []
            grasp_depth_merged = []
            grasp_rotations_merged = []
            grasp_views_rot_merged = []
            
            for obj_idx, pose in enumerate(poses):
                grasp_points = batch_data['grasp_points_list'][b][obj_idx]
                grasp_scores = batch_data['grasp_scores_list'][b][obj_idx]
                top_view_index = batch_data['top_view_index_list'][b][obj_idx]
                grasp_widths = batch_data['grasp_widths_list'][b][obj_idx]
                grasp_depth = batch_data['grasp_depth_list'][b][obj_idx]
                grasp_rotations = batch_data['grasp_rotations_list'][b][obj_idx]
                
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

                grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)
                grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1)

                top_view_index_trans = -1 * torch.ones((num_grasp_points, grasp_scores.shape[1]), dtype=torch.long, device=device)
                tpid, tvip, tids = torch.where(view_inds == top_view_index.unsqueeze(-1))
                top_view_index_trans[tpid, tvip] = tids

                grasp_points_merged.append(grasp_points_trans)
                top_view_index_merged.append(top_view_index_trans)
                grasp_scores_merged.append(grasp_scores)
                grasp_widths_merged.append(grasp_widths)
                grasp_depth_merged.append(grasp_depth)
                grasp_rotations_merged.append(grasp_rotations)
                grasp_views_rot_merged.append(grasp_views_rot_trans)

            if len(grasp_points_merged) == 0:
                continue
                
            grasp_points_merged = torch.cat(grasp_points_merged, dim=0)
            top_view_index_merged = torch.cat(top_view_index_merged, dim=0)
            grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)
            grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)
            grasp_depth_merged = torch.cat(grasp_depth_merged, dim=0)
            grasp_rotations_merged = torch.cat(grasp_rotations_merged, dim=0)
            grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)

            dists = torch.cdist(seed_xyz.unsqueeze(0), grasp_points_merged.unsqueeze(0)).squeeze(0)
            
            for s_idx in range(100):
                in_radius = dists[s_idx] <= DIST_THRESH
                if in_radius.sum() == 0:
                    continue
                    
                nb_indices = torch.where(in_radius)[0]
                m_scores = grasp_scores_merged[nb_indices]
                m_top_views = top_view_index_merged[nb_indices]
                
                valid_view_mask = (m_top_views != -1)
                
                for thresh in SCORE_THRESHOLDS:
                    valid_slots = (m_scores > thresh) & valid_view_mask
                    num_valid = valid_slots.sum().item()
                    
                    if num_valid > 1:
                        m_rel_idx, v_idx = torch.where(valid_slots)
                        m_abs_idx = nb_indices[m_rel_idx]
                        
                        # Reconstruct full 3x3 rotation matrix
                        R0 = grasp_views_rot_merged[m_abs_idx, v_idx] # [N, 3, 3]
                        rot_idx = grasp_rotations_merged[m_abs_idx, v_idx].float() # [N]
                        theta = rot_idx * (math.pi / 6.0)
                        
                        cos_t = torch.cos(theta)
                        sin_t = torch.sin(theta)
                        zeros = torch.zeros_like(theta)
                        ones = torch.ones_like(theta)
                        
                        Rx = torch.stack([ones, zeros, zeros, zeros, cos_t, -sin_t, zeros, sin_t, cos_t], dim=-1).reshape(-1, 3, 3)
                        R = torch.bmm(R0, Rx) # [N, 3, 3]
                        
                        # Pairwise geodesic distance
                        traces = torch.einsum('nab,mab->nm', R, R)
                        cos_dist = (traces - 1.0) / 2.0
                        cos_dist = torch.clamp(cos_dist, -1.0 + 1e-6, 1.0 - 1e-6)
                        dists_mat = torch.acos(cos_dist) * (180.0 / math.pi) # [N, N] in degrees
                        
                        # Get upper triangular part
                        idx_upper = torch.triu_indices(num_valid, num_valid, offset=1)
                        pw_dists = dists_mat[idx_upper[0], idx_upper[1]].cpu().numpy()
                        
                        results[thresh]['dist_mean'].append(np.mean(pw_dists))
                        results[thresh]['dist_median'].append(np.median(pw_dists))
                        results[thresh]['dist_90th'].append(np.percentile(pw_dists, 90))
                        results[thresh]['all_dists'].extend(pw_dists.tolist())

    print("\n" + "="*60)
    print("SURVEY RESULTS: ROTATION ENTROPY (GEODESIC DISTANCE)")
    print("="*60)
    
    for thresh in SCORE_THRESHOLDS:
        all_dists = np.array(results[thresh]['all_dists'])
        mean_arr = np.array(results[thresh]['dist_mean'])
        median_arr = np.array(results[thresh]['dist_median'])
        p90_arr = np.array(results[thresh]['dist_90th'])
        
        if len(all_dists) > 0:
            print(f"\n[ Score > {thresh:.1f} ]")
            print(f"  - Overall Mean Pairwise Dist  : {np.mean(all_dists):.2f} degrees")
            print(f"  - Overall Median Pairwise Dist: {np.median(all_dists):.2f} degrees")
            print(f"  - Overall 90th Percentile     : {np.percentile(all_dists, 90):.2f} degrees")
            
            print(f"  - Avg of Seed Point Means     : {np.mean(mean_arr):.2f} degrees")
            print(f"  - Avg of Seed Point Medians   : {np.mean(median_arr):.2f} degrees")
            print(f"  - Avg of Seed Point 90th Pct  : {np.mean(p90_arr):.2f} degrees")
            
            # Plot histogram
            plt.figure(figsize=(10, 6))
            plt.hist(all_dists, bins=60, range=(0, 180), color='skyblue', edgecolor='black')
            plt.title(f"Pairwise Geodesic Distance Distribution (Score > {thresh:.1f})")
            plt.xlabel("Geodesic Distance (degrees)")
            plt.ylabel("Frequency")
            plt.grid(axis='y', alpha=0.75)
            plt.savefig(f"scratch/rotation_entropy_score_{thresh:.1f}.png")
            plt.close()
            
        else:
            print(f"\n[ Score > {thresh:.1f} ] No seed points with >1 valid grasps.")
            
    print("\nHistograms saved as scratch/rotation_entropy_score_X.png")

if __name__ == '__main__':
    survey_multiple_scores()
