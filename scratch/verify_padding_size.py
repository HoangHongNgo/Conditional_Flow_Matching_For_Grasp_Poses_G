import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.arguments import cfgs
from models.economicgrasp import economicgrasp
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from libs.knn.knn_modules import knn
from utils.loss_utils import (batch_viewpoint_params_to_matrix, transform_point_cloud,
                              generate_grasp_views)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = GraspNetDataset(
        '/media/dsp520/Grasp_2T/graspnet', camera='realsense', split='train',
        voxel_size=cfgs.voxel_size, num_points=cfgs.num_point,
        remove_outlier=True, augment=False
    )
    
    # 2 frames is enough to prove the point
    subset = torch.utils.data.Subset(dataset, [0, 1])
    dataloader = DataLoader(subset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    net = economicgrasp(seed_feat_dim=512, is_training=False)
    net.to(device)
    ckpt = torch.load('checkpoints/economicgrasp_realsense.tar', map_location=device, weights_only=False)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()

    with torch.no_grad():
        for batch_data in dataloader:
            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)

            end_points = net(batch_data)
            seed_xyzs = end_points['xyz_graspable']
            
            for b in range(2):
                seed_xyz = seed_xyzs[b]
                poses = end_points['object_poses_list'][b]
                
                grasp_points_merged = []
                grasp_scores_merged = []
                top_view_index_merged = []
                
                for obj_idx, pose in enumerate(poses):
                    grasp_points = end_points['grasp_points_list'][b][obj_idx]
                    grasp_scores = end_points['grasp_scores_list'][b][obj_idx]
                    top_view_index = end_points['top_view_index_list'][b][obj_idx]
                    num_grasp_points = grasp_points.size(0)

                    grasp_views = generate_grasp_views(cfgs.num_view).to(device)
                    grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
                    grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')

                    grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
                    grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
                    view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1

                    top_view_index_trans = -1 * torch.ones((num_grasp_points, grasp_scores.shape[1]), dtype=torch.long, device=device)
                    tpid, tvip, tids = torch.where(view_inds == top_view_index.unsqueeze(-1))
                    top_view_index_trans[tpid, tvip] = tids

                    grasp_points_merged.append(grasp_points_trans)
                    top_view_index_merged.append(top_view_index_trans)
                    grasp_scores_merged.append(grasp_scores)

                grasp_points_merged = torch.cat(grasp_points_merged, dim=0)
                top_view_index_merged = torch.cat(top_view_index_merged, dim=0)
                grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)

                dists = torch.cdist(seed_xyz.unsqueeze(0), grasp_points_merged.unsqueeze(0)).squeeze(0)

                counts_without_view_filter = []
                counts_with_view_filter = []

                for s_idx in range(seed_xyz.shape[0]):
                    in_radius = dists[s_idx] <= 0.005
                    if in_radius.sum() == 0:
                        counts_without_view_filter.append(0)
                        counts_with_view_filter.append(0)
                        continue
                    
                    nb_indices = torch.where(in_radius)[0]
                    m_scores = grasp_scores_merged[nb_indices]
                    m_top_views = top_view_index_merged[nb_indices]
                    
                    # Like the survey script
                    valid_slots_1 = (m_scores > 0.7)
                    counts_without_view_filter.append(valid_slots_1.sum().item())
                    
                    # Like cfm_label_generation
                    valid_slots_2 = (m_scores > 0.7) & (m_top_views != -1)
                    counts_with_view_filter.append(valid_slots_2.sum().item())

                import numpy as np
                print(f"Batch {b}:")
                print(f"  Mean valid configs (survey method - NO view filter)  : {np.mean(counts_without_view_filter):.2f}")
                print(f"  Mean valid configs (cfm method - WITH view filter) : {np.mean(counts_with_view_filter):.2f}")
                print(f"  Max valid configs (survey method) : {np.max(counts_without_view_filter)}")
                print(f"  Max valid configs (cfm method)    : {np.max(counts_with_view_filter)}")
                print("-" * 50)
            break

if __name__ == '__main__':
    main()
