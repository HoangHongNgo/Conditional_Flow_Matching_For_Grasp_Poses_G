import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.arguments import cfgs
from models.economicgrasp import economicgrasp
from dataset.graspnet_dataset import GraspNetDataset, collate_fn

NUM_SCENES = 5
FRAMES_PER_SCENE = 10
RADIUS_MM = 5.0
SCORE_THRESH = 0.7

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = GraspNetDataset(
        '/media/dsp520/Grasp_2T/graspnet', camera='realsense', split='train',
        voxel_size=cfgs.voxel_size, num_points=cfgs.num_point,
        remove_outlier=True, augment=False
    )
    
    # GraspNetDataset adds 256 frames per scene linearly.
    # To get 10 frames from 5 scenes, we select specific indices:
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

    print(f"Khảo sát {NUM_SCENES} scenes, mỗi scene {FRAMES_PER_SCENE} frames (tổng cộng {len(indices)} frames).")
    print(f"Điều kiện: Bán kính <= {RADIUS_MM}mm, Score >= {SCORE_THRESH}")

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
            seed_xyz = end_points['xyz_graspable'][0]  # [1024, 3]

            poses = batch_data['object_poses_list'][0]
            all_pts, all_scores = [], []
            for obj_idx, pose in enumerate(poses):
                pts = batch_data['grasp_points_list'][0][obj_idx]   # [P, 3]
                scs = batch_data['grasp_scores_list'][0][obj_idx]   # [P, 60]
                pts_trans = torch.matmul(pts, pose[:3, :3].t()) + pose[:3, 3]
                all_pts.append(pts_trans)
                all_scores.append(scs)

            if len(all_pts) == 0:
                continue
                
            gt_pts = torch.cat(all_pts, dim=0)       # [M, 3]
            gt_scores = torch.cat(all_scores, dim=0)  # [M, 60]

            dists_mm = torch.cdist(seed_xyz.unsqueeze(0), gt_pts.unsqueeze(0)).squeeze(0) * 1000.0

            for s_idx in range(seed_xyz.shape[0]):
                in_radius = dists_mm[s_idx] <= RADIUS_MM  # [M]
                if in_radius.sum() == 0:
                    all_counts.append(0)
                    continue
                nb_scores = gt_scores[in_radius]  # [num_nb, 60]
                num_configs = (nb_scores >= SCORE_THRESH).sum().item()
                all_counts.append(num_configs)

    c = np.array(all_counts)

    print(f"\n{'='*60}")
    print(f"  TỔNG HỢP KẾT QUẢ KHẢO SÁT")
    print(f"{'='*60}")
    print(f"  Tổng số seed points đã khảo sát: {len(c)}")
    print(f"  Mean   : {c.mean():.1f}")
    print(f"  Median : {np.median(c):.0f}")
    print(f"  Max    : {c.max()}")
    print(f"  Phân vị (Percentiles):")
    print(f"    50% (Median)     : <= {np.percentile(c, 50):.0f} configs")
    print(f"    75%              : <= {np.percentile(c, 75):.0f} configs")
    print(f"    90%              : <= {np.percentile(c, 90):.0f} configs")
    print(f"    95%              : <= {np.percentile(c, 95):.0f} configs")
    print(f"    99%              : <= {np.percentile(c, 99):.0f} configs")
    print(f"    99.9%            : <= {np.percentile(c, 99.9):.0f} configs")
    
    print("\n  Khuyến nghị chọn MAX_K (để padding):")
    for pct in [75, 90, 95, 99]:
        val = int(np.percentile(c, pct))
        loss_pct = (c > val).mean() * 100
        print(f"    Nếu MAX_K = {val:<3} (phủ {pct}% seeds) -> Cắt bỏ bớt config của {loss_pct:.2f}% seed points.")

if __name__ == '__main__':
    main()
