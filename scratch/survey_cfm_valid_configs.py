import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
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
    
    NUM_SCENES = 5
    FRAMES_PER_SCENE = 10
    indices = []
    for i in range(NUM_SCENES):
        for j in range(FRAMES_PER_SCENE):
            indices.append(i * 256 + j)
            
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    net = economicgrasp(seed_feat_dim=512, is_training=False)
    net.to(device)
    ckpt = torch.load('checkpoints/economicgrasp_realsense.tar', map_location=device, weights_only=False)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()

    print(f"Khảo sát CFM Label Filter: {NUM_SCENES} scenes, {FRAMES_PER_SCENE} frames/scene ({len(indices)} frames total).")
    
    all_counts = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing Frames")):
            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)

            end_points = net(batch_data)
            seed_xyzs = end_points['xyz_graspable']
            
            b = 0 # batch size is 1
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

            if len(grasp_points_merged) == 0:
                for s_idx in range(seed_xyz.shape[0]):
                    all_counts.append(0)
                continue

            grasp_points_merged = torch.cat(grasp_points_merged, dim=0)
            top_view_index_merged = torch.cat(top_view_index_merged, dim=0)
            grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)

            dists = torch.cdist(seed_xyz.unsqueeze(0), grasp_points_merged.unsqueeze(0)).squeeze(0)

            for s_idx in range(seed_xyz.shape[0]):
                in_radius = dists[s_idx] <= 0.005
                if in_radius.sum() == 0:
                    all_counts.append(0)
                    continue
                
                nb_indices = torch.where(in_radius)[0]
                m_scores = grasp_scores_merged[nb_indices]
                m_top_views = top_view_index_merged[nb_indices]
                
                # STRICT FILTER AS IN CFM_LABEL_GENERATION
                valid_slots = (m_scores > 0.7) & (m_top_views != -1)
                all_counts.append(valid_slots.sum().item())

    c = np.array(all_counts)

    print(f"\n{'='*60}")
    print(f"  TỔNG HỢP KẾT QUẢ KHẢO SÁT (VỚI CFM VIEW FILTER)")
    print(f"{'='*60}")
    print(f"  Tổng số seed points đã khảo sát: {len(c)}")
    print(f"  Mean   : {c.mean():.1f}")
    print(f"  Median : {np.median(c):.0f}")
    print(f"  Max    : {c.max()}")
    print(f"  Phân vị (Percentiles):")
    print(f"    50% (Median)     : <= {np.percentile(c, 50):.0f} configs")
    print(f"    75%              : <= {np.percentile(c, 75):.0f} configs")
    print(f"    80%              : <= {np.percentile(c, 80):.0f} configs")
    print(f"    85%              : <= {np.percentile(c, 85):.0f} configs")
    print(f"    90%              : <= {np.percentile(c, 90):.0f} configs")
    print(f"    95%              : <= {np.percentile(c, 95):.0f} configs")
    print(f"    99%              : <= {np.percentile(c, 99):.0f} configs")
    
    print("\n  Khuyến nghị chọn MAX_K (để padding):")
    for pct in [75, 80, 85, 90, 95, 99]:
        val = int(np.percentile(c, pct))
        loss_pct = (c > val).mean() * 100
        print(f"    Nếu MAX_K = {val:<3} (phủ {pct}% seeds) -> Cắt bỏ bớt config của {loss_pct:.2f}% seed points.")

if __name__ == '__main__':
    main()
